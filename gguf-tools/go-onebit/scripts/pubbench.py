#!/usr/bin/env python3
# pubbench.py — 公共标尺 harness: HumanEval(Python) / HumanEval-X(Go) 子集, greedy, 原始输出全落盘。
# 量化口径: 同一套题对 基线 与 量化 各跑一遍(--tag 区分), --compare 出 per-task delta 表。
# 判决器: Python=官方同构(拼 prompt+completion+test 子进程执行, 超时即败);
#         Go=拼包后 go test(拼装规则首火可能需拧一扣, 结果标 experimental 直到人工抽查确认)。
# 注意: 会执行模型生成的代码(与官方 human-eval harness 同风险面), 只在本机跑。
import argparse, gzip, io, json, os, re, subprocess, sys, tempfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

# 默认仍是 /tmp(旧行为不变)。PUBBENCH_CACHE 指向 gguf-tools/go-onebit/pubbench_data/
# 可跑在联不上外网的机器上(Spark 直连 GitHub/HF 超时) —— 那两份 164 题 jsonl 已入库。
CACHE = os.environ.get("PUBBENCH_CACHE", "/tmp/pubbench_cache")

HUMANEVAL_URLS = [
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
]
# THUDM/humaneval-x 的 go 分片路径有过变动, 依次尝试
HUMANEVALX_GO_URLS = [
    "https://huggingface.co/datasets/THUDM/humaneval-x/resolve/main/data/go/data/humaneval.jsonl",
    "https://huggingface.co/datasets/THUDM/humaneval-x/resolve/main/go/data/humaneval.jsonl",
]

PY_STOP = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n```"]
GO_STOP = ["\nfunc main(", "\n// Test", "\npackage ", "\n```"]


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def fetch(urls, name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    last = None
    for u in urls:
        try:
            log(f"[fetch] {u}")
            req = urllib.request.Request(u, headers={"User-Agent": "pubbench/1.0"})
            data = urllib.request.urlopen(req, timeout=120).read()
            if u.endswith(".gz"):
                data = gzip.GzipFile(fileobj=io.BytesIO(data)).read()
            with open(path, "wb") as f:
                f.write(data)
            return path
        except Exception as e:  # noqa: BLE001
            last = e
            log(f"[fetch] fail: {e}")
    raise RuntimeError(f"dataset {name} fetch failed: {last}")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# 本机直连, 绕全局代理(07-14 502 教训: 裸 urlopen 会吃 http_proxy env)
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

BOS = "<｜begin▁of▁sentence｜>"


def call_server(url, prompt, max_tokens, timeout, mode="code", api="chat"):
    # api=completions: BASE 模型口径 = /v1/completions raw 裸续写(BOS+题面), 返回纯续写文本。
    #   与官方 human-eval 协议同形; CLI 双机裸续写已验证的语义(2026-07-27)。
    # api=chat: 原 /v1/chat/completions mode:code(instruct/chat 部署用)。
    if api == "completions":
        body = {
            "model": "ds4",
            "prompt": BOS + prompt,
            "temperature": 0,
            "max_tokens": max_tokens,
            "raw": True,
        }
        endpoint = "/v1/completions"
    else:
        body = {
            "model": "ds4",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
            "mode": mode,
        }
        endpoint = "/v1/chat/completions"
    req = urllib.request.Request(
        url.rstrip("/") + endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    raw = _OPENER.open(req, timeout=timeout).read().decode("utf-8", "replace")
    dt = time.time() - t0
    txt = ""
    try:
        j = json.loads(raw)
        c = j["choices"][0]
        txt = c["text"] if api == "completions" else c["message"]["content"]
    except Exception:  # noqa: BLE001
        txt = raw
    return txt, raw, dt


def extract_completion(text, prompt, stops, strip_prompt=True, close_brace=False):
    # 服务端 mode:code 常给围栏; 取第一个围栏体, 否则用原文
    fences = re.findall(r"```[a-zA-Z0-9_+-]*\n(.*?)(?:```|\Z)", text, re.S)
    body = fences[0] if fences else text
    # chat 完整重写形态(2026-08-18): 围栏体自带入口函数完整定义(def entry(... / func Entry(...)
    # + 函数体) —— 这是 instruct 模型的合法答案形态, 续写式剥重叠对它必然错位(sig 锚 \"\"\"
    # 会把答案从 docstring 处劈开)。此形态直接原样返回, eval 侧当独立完整程序拼 test。
    if strip_prompt and fences:
        # 响应常有多个 fence(复述题面 + 解答): 取含入口函数定义的最长者(解答几乎恒长于复述);
        # 复述 fence 的函数体常是空/pass, 当解答交卷=全场 AssertionError(08-18 实证)。
        dm = re.findall(r"(?:def|func)\s+(\w+)\s*\(", prompt)
        if dm:
            ent = re.escape(dm[-1])
            cands = [f for f in fences
                     if re.search(r"(?:def|func)\s+" + ent + r"\s*\(", f)
                     and re.search(r"(?:def|func)\s+" + ent + r"\s*\([^\n]*\)[^\n]*(?:\{|:)", f)]
            if cands:
                best = max(cands, key=len)
                return "\x00FULL\x00" + best.rstrip() + "\n"
    # 若模型把题面(签名)复述了, 剥掉与 prompt 重叠的头部。
    # completions 裸续写(strip_prompt=False)禁用: response 本就是纯续写, 而"最后一行锚"
    # 对 HumanEval 恒为 \"\"\" — 会命中续写里任意 docstring 把正确答案整段扔掉(t4 实证)。
    p_tail = prompt.rstrip()
    if not strip_prompt:
        pass
    elif p_tail and p_tail in body:
        body = body.split(p_tail, 1)[1]
    else:
        sig = p_tail.splitlines()[-1].strip() if p_tail.splitlines() else ""
        if strip_prompt and sig and sig in body:
            body = body.split(sig, 1)[1]
            if body.startswith(":"):
                body = body[1:]
    # 客户端截断(不依赖服务端 stop 参数)
    cut = len(body)
    for s in stops:
        i = body.find(s)
        if i != -1:
            cut = min(cut, i)
    body = body[:cut]
    # Go 漂移截断(2026-07-27 v2): 首版"第0列}含括截断"误砍合法辅助函数(Go/10 Reverse)。
    # 精确策略: ①入口函数被重复声明处=确定的漂移起点, 截掉(Go/0 实证形态);
    # ②花括号配平截尾: 续写从函数体内(深度1)起算, 截到最后一次回到深度0 —— 保留完整
    # 辅助函数, 丢掉被 token 上限截断的残缺漂移函数(否则残缺函数破坏编译)。
    if close_brace:
        ms = list(re.finditer(r"func\s+(\w+)\s*\(", prompt))
        if ms:
            j = body.find(f"func {ms[-1].group(1)}(")
            if j != -1:
                body = body[:j]
        # 尾部残缺漂移丢弃: 截到最后一个第0列 "}"(gofmt 顶层函数闭合必在列0)。
        # 不用花括号深度配平 — Go 类型语法的内联括号(interface{}/struct{})会骗计数器
        # (Go/12 实证: 截进 `func X(...) interface{}` 声明中间 → missing function body)。
        j2 = body.rfind("\n}")
        if j2 != -1:
            body = body[: j2 + 2]
    return body.rstrip() + "\n"


def eval_python(row, completion, timeout_s):
    if completion.startswith("\x00FULL\x00"):
        prog = completion[6:] + "\n" + row["test"] + f"\ncheck({row['entry_point']})\n"
    else:
        prog = row["prompt"] + completion + "\n" + row["test"] + f"\ncheck({row['entry_point']})\n"
    try:
        r = subprocess.run(
            [sys.executable, "-c", prog],
            capture_output=True, text=True, timeout=timeout_s,
        )
        return (r.returncode == 0), (r.stderr[-800:] if r.returncode != 0 else "")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001
        return False, f"harness: {e}"


def _go_import_paths(s):
    out = []
    for blk in re.findall(r"(?ms)^import\s*\((.*?)\)", s):
        out += re.findall(r'"([^"]+)"', blk)
    out += re.findall(r'(?m)^import\s+"([^"]+)"', s)
    return out


def _go_strip_headers(s):
    s = re.sub(r"(?ms)^import\s*\(.*?\)\s*", "", s)
    s = re.sub(r'(?m)^import\s+"[^"]+"\s*$', "", s)
    s = re.sub(r"(?m)^package\s+\w+\s*$", "", s)
    return s


def eval_go(row, completion, timeout_s):
    # 单文件拼装(2026-07-27 修): humaneval-x go 的 test 与解答共享同文件 import(实证 Go/2:
    # 两文件拼装下 sol_test.go undefined: math)。合并全部 import 路径去重 → 一个 _test.go;
    # goimports -w 删未用 import(合并块是超集, 只删不加, 无需解析远程包)。
    d = tempfile.mkdtemp(prefix="pubbench_go_")
    setup = row.get("test_setup", "") or ""
    if completion.startswith("\x00FULL\x00"):
        sol = completion[6:]   # chat 完整重写: 自含入口函数, 不拼题面(strip_headers 剥其 package/import)
    else:
        sol = row["prompt"] + completion
    paths = []
    for s in (row.get("import", "") or "", sol, setup, row["test"]):
        for p in _go_import_paths(s):
            if p not in paths:
                paths.append(p)
    imports = ("import (\n" + "".join(f'    "{p}"\n' for p in paths) + ")\n") if paths else ""
    src_all = ("package main\n\n" + imports + "\n"
               + _go_strip_headers(sol) + "\n" + _go_strip_headers(setup + "\n" + row["test"]))
    try:
        p = os.path.join(d, "sol_test.go")
        open(p, "w").write(src_all)
        gi = os.path.expanduser("~/go/bin/goimports")
        if os.path.exists(gi):
            subprocess.run([gi, "-w", p], capture_output=True, timeout=60)
        subprocess.run(["go", "mod", "init", "pubbench"], cwd=d, capture_output=True, timeout=60)
        subprocess.run(["go", "mod", "tidy"], cwd=d, capture_output=True, timeout=120)
        r = subprocess.run(["go", "test", "./..."], cwd=d, capture_output=True, text=True, timeout=timeout_s)
        return (r.returncode == 0), (r.stdout[-400:] + r.stderr[-400:] if r.returncode != 0 else "")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001
        return False, f"harness: {e}"


def run_suite(a):
    if a.suite == "humaneval":
        rows = load_jsonl(fetch(HUMANEVAL_URLS, "HumanEval.jsonl"))
        stops, ev, lang = PY_STOP, eval_python, "python"
    else:
        rows = load_jsonl(fetch(HUMANEVALX_GO_URLS, "humaneval_x_go.jsonl"))
        stops, ev, lang = GO_STOP, eval_go, "go"
    rows = rows[a.offset : a.offset + a.limit]
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, f"pubbench_{a.suite}_{a.tag}.jsonl")
    n_pass = 0
    done_n = [0]
    def one_task(k_row):
        k, row = k_row
        tid = row.get("task_id", f"{a.suite}/{a.offset + k}")
        # ★Go 生成端 prompt 补全(2026-08-05 用户判"引擎有bug"实锤)★: 判定端(eval_go)
        # 一直补 package+import 编译, 生成端却发裸函数 — BASE 模型的 Go 语料函数永远
        # 在文件头之后, 裸函数=分布外 ⇒ TODO 弃权/风格漂移(两代同病, 冠军 Go 10/20 同压)。
        # 组装完整文件头; 判定端 _go_strip_headers 对 sol 先剥后统一重组, 天然兼容。
        gen_prompt = row["prompt"]
        if lang == "go":
            imp = (row.get("import", "") or "").strip()
            gen_prompt = "package main\n\n" + (imp + "\n\n" if imp else "") + row["prompt"].lstrip("\n")
        try:
            text, raw, dt = call_server(a.url, gen_prompt, a.max_tokens, a.http_timeout, a.mode, a.api)
            comp = extract_completion(text, gen_prompt, stops, strip_prompt=(a.api != "completions"), close_brace=(a.suite != "humaneval"))
            ok, err = ev(row, comp, a.eval_timeout)
        except Exception as e:  # noqa: BLE001
            text, raw, dt, comp, ok, err = "", "", 0.0, "", False, f"request: {e}"
        rec = {
            "task_id": tid, "lang": lang, "tag": a.tag, "pass": ok, "err": err,
            "gen_seconds": round(dt, 1), "prompt": row["prompt"],
            "completion": comp, "response_text": text, "response_raw": raw,
        }
        done_n[0] += 1
        log(f"[{done_n[0]}/{len(rows)}] {tid} gen={dt:.0f}s {'PASS' if ok else 'FAIL'}"
            + (f" ({err.splitlines()[-1][:80]})" if err else ""))
        return k, rec
    # ★并发(2026-08-18 用户令"测试并发不要串行")★: server 推理单 worker 串行, 但并发请求
    # 消掉 client 间隙+判题(go test 1-5s/题)与生成流水; 结果按题序落盘。
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        results = dict(ex.map(one_task, enumerate(rows)))
    with open(out_path, "w", encoding="utf-8") as out:
        for k in range(len(rows)):
            rec = results[k]
            n_pass += int(rec["pass"])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    note = " (go judge=experimental, 人工抽查后作数)" if a.suite != "humaneval" else ""
    log(f"[done] {a.suite} tag={a.tag} pass@1 = {n_pass}/{len(rows)}{note}")
    log(f"[raw] {out_path}")
    print(json.dumps({"suite": a.suite, "tag": a.tag, "pass": n_pass, "total": len(rows), "raw": out_path}))


def rejudge(path, api="completions"):
    # 离线重判: 用已存 response_text 重新抽取+评测(修抽取器后免重生成), 原地重写 pass/completion。
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n_pass = 0
    def _rejudge_one(r):
        stops, ev = (PY_STOP, eval_python) if r["lang"] == "python" else (GO_STOP, eval_go)
        row = {"prompt": r["prompt"], "test": r.get("test", ""), "entry_point": r.get("entry_point", "")}
        # 数据集字段(test/entry_point)不在 jsonl 里 → 从数据集按 task_id 回填
        if not row["test"]:
            suite = "humaneval" if r["lang"] == "python" else "humaneval-x-go"
            ds = load_jsonl(fetch(HUMANEVAL_URLS, "HumanEval.jsonl")) if suite == "humaneval"                  else load_jsonl(fetch(HUMANEVALX_GO_URLS, "humaneval_x_go.jsonl"))
            full = {d["task_id"]: d for d in ds}[r["task_id"]]
            row.update(test=full["test"], entry_point=full.get("entry_point", ""))
            for k in ("test_setup", "import"):
                if k in full:
                    row[k] = full[k]
        comp = extract_completion(r["response_text"], r["prompt"], stops, strip_prompt=(api != "completions"), close_brace=(r["lang"] != "python"))
        ok, err = ev(row, comp, 15)
        changed = "" if ok == r["pass"] else f"  [改判 {r['pass']}→{ok}]"
        log(f"{r['task_id']}: {'PASS' if ok else 'FAIL'}{changed}" + (f" ({(err.splitlines() or [''])[-1][:70]})" if err else ""))
        r["pass"], r["err"], r["completion"] = ok, err, comp
        return int(ok)
    with ThreadPoolExecutor(max_workers=8) as ex:   # 判题并发(用户令 08-18)
        n_pass = sum(ex.map(_rejudge_one, rows))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"[rejudge] {path}: pass@1 = {n_pass}/{len(rows)}")


def verdict(B, Q, N, base_tag, quant_tag):
    # 两个互相独立的决策闸。阈值是"决策带"不是测量值: 粗到经得起 n=20 的 ±2 题噪声,
    # 编码的是 2026-07-25 对话已定的逻辑(fable5.md): 参赛的标的=可运行作品+诚实方法论;
    # 买机的标的=产品阶段(驻留/MTP/KV/日用速度)是否真实存在。不是大赛评审标准。
    print()
    print(f"== 判决闸门: 基线={base_tag} B={B}/{N} | 量化={quant_tag} Q={Q}/{N} | n={N} 噪声底≈±2题 ==")
    if B < 0.6 * N:
        print(f"[INVALID] 基线 {B}/{N} < 60% — q2 基线编程域不该这么低; 先查 harness/服务层"
              "(历史教训: 服务层bug曾把好引擎打成零代码), 本轮分数不作任何决策依据")
        return
    ret = Q / B
    # 故事口径(用户裁决 07-25): 消费级单机+量化模型+直接可用编程。双机组网/慢速运行不构成故事。
    # 阈值=Claude 的专业判断(07-25 晚定稿, 责任署名, 错误可检验形态在 fable5.md 当日条目):
    # 锚=竞争替代 — 64G Mac 上人人五分钟可跑 Qwen3-coder 30B 级免费模型, 故事必须明显打赢它;
    # 50% 保留(≈CodeLlama-13B 档)打不赢, 保留七成的 frontier 级 MoE + 1M ctx + 自研引擎才成立。
    if Q >= 0.6 * N and ret >= 0.7:
        g1 = ("GO — 故事成立: 单机可用速度实录为 Demo; 现役双机只作开发素材, "
              "购机为截稿(08-16)前关键路径")
    elif Q >= 0.5 * N and ret >= 0.5:
        g1 = ("BORDER — 打不赢'懒人替代'(64G 免费 30B coder), quant 故事不达标; "
              "fallback=引擎+官方q2单机(96G)故事存在但评级偏弱, 默认弃本届")
    else:
        g1 = "NO — 质量主张撑不住, 什么机器都救不了故事; 回算法迭代"
    if Q >= 0.6 * N and ret >= 0.7:
        g2 = ("GO — 买 96G M4 Max(非 64G: 多的几千块买'量化 vs 官方q2 同机对照演示'位+日常 parity 参照; "
              "按 9.5t/s@120GB/s 线性外推 546GB/s≈30-40t/s, 到手 24h ds4-bench 实测替换)")
    else:
        g2 = "不买本周期 — 竞争锚下故事不成立, 机器无标的; 下个质量里程碑再判"
    print(f"[GOAI 参赛闸] {g1}")
    print(f"[买机闸]      {g2}")
    print("[口径] 质量闸先行, 两闸都由质量分决定; GO 态下购机档位服务'消费级单机直接编程'故事; "
          "报名前读 goaihz.com 章程 (三赛道截止 2026-08-16, 具身 08-20)")
    if g1.startswith("GO") or g2.startswith("GO"):
        print("[下单前加固] GO≠下单, 动钱前三点全过: "
              "① LIMIT=164 跑满全套收窄置信区间(n=20 delta CI≈±31% → n=164 ≈±11%) "
              "② 第二条腿 gen_coding_probe 真实任务可用性过(HumanEval 是短函数补全, ≠真实编辑工作流) "
              "③ 新机到手 24h 内 ds4-bench 实测替换所有速度外推, 不达可用速度按残值退/售")


def compare(fa, fb):
    A = {r["task_id"]: r for r in load_jsonl(fa)}
    B = {r["task_id"]: r for r in load_jsonl(fb)}
    ka, kb = next(iter(A.values()))["tag"], next(iter(B.values()))["tag"]
    both = sorted(set(A) & set(B))
    pa = sum(A[t]["pass"] for t in both)
    pb = sum(B[t]["pass"] for t in both)
    print(f"(顺序约定: A=基线={ka}, B=量化={kb})")
    print(f"{'task_id':<18} {ka:>8} {kb:>8}")
    for t in both:
        print(f"{t:<18} {'PASS' if A[t]['pass'] else '.':>8} {'PASS' if B[t]['pass'] else '.':>8}")
    print(f"{'TOTAL':<18} {pa:>5}/{len(both)} {pb:>5}/{len(both)}   delta({kb}-{ka}) = {pb - pa:+d}")
    verdict(pa, pb, len(both), ka, kb)


def selftest(suite, limit, offset, eval_timeout):
    # harness 自检: 拿数据集自带的 canonical_solution 冒充模型输出, 走与真跑完全同一条
    # 抽取+判定链路。**期望 164/164** —— 标准答案判不过 = harness 坏了(缺 go 工具链/
    # 抽取器误剥/拼装规则跑偏), 与被测模型的质量无关。换机器/换 Go 版本后先跑这个,
    # 免得把环境问题算到模型头上(2026-07-27 那次抽取器"最后一行锚"误剥就是这么漏掉的)。
    if suite == "humaneval":
        rows = load_jsonl(fetch(HUMANEVAL_URLS, "HumanEval.jsonl"))
        ev, lang = eval_python, "python"
    else:
        rows = load_jsonl(fetch(HUMANEVALX_GO_URLS, "humaneval_x_go.jsonl"))
        ev, lang = eval_go, "go"
    rows = rows[offset : offset + limit]
    n_pass, failed = 0, []
    t0 = time.time()
    for k, row in enumerate(rows):
        tid = row.get("task_id", f"{suite}/{offset + k}")
        # canonical_solution 就是"续写部分", 与 completions 裸续写口径同形, 直接当 completion
        ok, err = ev(row, row["canonical_solution"], eval_timeout)
        n_pass += int(ok)
        if not ok:
            failed.append(tid)
            log(f"{tid}: FAIL  ({(err.splitlines() or [''])[-1][:100]})")
        elif (k + 1) % 20 == 0:
            log(f"[selftest] {k + 1}/{len(rows)} ... {n_pass} pass")
    log(f"[selftest] {suite}: {n_pass}/{len(rows)} 标准答案通过 ({time.time() - t0:.0f}s)")
    if failed:
        log(f"[selftest] 未通过: {' '.join(failed)}")
        log("[selftest] ★harness 有问题★ — 标准答案本应全过, 先修环境/判定器再跑模型")
    return n_pass == len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["humaneval", "humaneval-x-go"], default="humaneval")
    ap.add_argument("--url", default=os.environ.get("DS4_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--tag", required=False, default="run")   # 例: base_q2 / vq14
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--mode", default="code")                 # ds4 server mode:code
    ap.add_argument("--api", choices=["chat", "completions"],
                    default=os.environ.get("PUBBENCH_API", "chat"))  # completions=BASE 裸续写口径
    ap.add_argument("--http-timeout", type=int, default=900)  # 慢机 2-5 t/s 留足
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("PUBBENCH_JOBS", "4")))  # 并发(生成+判题流水)
    ap.add_argument("--eval-timeout", type=int, default=15)
    ap.add_argument("--out-dir", default=os.environ.get("PUBBENCH_OUT",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "pubbench")))
    ap.add_argument("--compare", nargs=2, metavar=("A.jsonl", "B.jsonl"))
    ap.add_argument("--rejudge", metavar="FILE.jsonl")
    ap.add_argument("--selftest", action="store_true",
                    help="用 canonical_solution 自检判定链路(期望满分), 不碰 server")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.suite, a.limit, a.offset, a.eval_timeout) else 1)
    if a.rejudge:
        rejudge(a.rejudge, a.api)
        return
    if a.compare:
        compare(*a.compare)
    else:
        run_suite(a)


if __name__ == "__main__":
    main()
