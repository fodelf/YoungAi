#!/usr/bin/env python3
"""bench_suite.py — 按 unsloth 的口径对着我们的服务跑任务基准, 按时间预算出 pass@1 与截断率。

只做编排(发请求/收答案/判对错/记账), 不碰任何数值路。口径照抄
https://unsloth.ai/docs/models/deepseek-v4.1-flash 的 Test Methodology: 上限 32,768 token, 温 1.0 / top_p 1.0,
"截断" = 撞上 max_tokens(finish_reason=length)。题面与判分方式按任务:
  math (AIME25 / HMMT): 题面 + "Please reason step by step, and put your final answer within \\boxed{}."
        判 = 最后一个 \\boxed{} 里的答案(整数直接比; 非整数做字符串归一后比), 没有 boxed 退回正文最后一个整数。
  mc   (GPQA-D / MMLU-Pro): OpenAI simple-evals 多选模板(题面在 prep 里拼好), 判 = 最后一个 "Answer: X",
        退回 \\boxed{X} / "answer is (X)" / 单独一个字母。
  raw  (IFEval / LCB): 只生成不判, 判分走 ifeval_grade.py / lcb_grade.py。

数据 = gguf-tools/scripts/bench_prep.py 出的 jsonl(每行 id / prompt / gold / grade / system?)。题序在这里按
固定种子打乱, 所以"一小时预算"跑到哪算哪都是随机子集, 报 n 与 ±SE。

用法(spark 上, 对着 ds4-server 直连口 8001):
  python3 gguf-tools/scripts/bench_suite.py --data gguf-tools/bench/data/aime25.jsonl \\
      --out gguf/v41/bench/<日期>/aime25 --budget-min 60 --jobs 3 [--samples 16]
产物: <out>/answers.jsonl 每答案一行(断点续跑, 已有的 (题,样本) 跳过); <out>/raw/<id>_<样本>.json 原始返回
      (思考 + 正文全留, 供人工复核); <out>/summary.json。进度逐答案打印一行。
"""
import argparse, json, os, re, sys, threading, time, random, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

SEED_BASE = 20261029   # unsloth 表里的 seed base; 只为口径可对, 不同引擎同种子不会同结果


# ── 判分 ──────────────────────────────────────────────────────────────────────
def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def norm_math(s):
    """字符串归一: 去空格/\\left\\right/\\dfrac→\\frac/\\text{}/逗号/末尾句点, 够比 HMMT 这类整式答案。"""
    t = s.strip()
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    t = t.replace("\\left", "").replace("\\right", "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    t = t.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("$", "").replace(" ", "")
    t = t.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("%", "")
    t = re.sub(r"^[a-zA-Z]\s*=", "", t)              # "x=42"
    t = t.rstrip(".")
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)          # 1,234
    return t


def math_equiv(pred, gold):
    """字符串归一比不上时的符号等价(math-verify, 装了才用; spark 上没装, 所以 HMMT 这类整式题要在 Mac 上 --regrade 复判)。
    实撞: 10-31 HMMT 11 题里 3 题被字符串比误判——3.5 vs 7/2、(4-π)/2 vs 2-π/2、2√435/3 vs √1740/3。"""
    try:
        from math_verify import parse, verify
    except ImportError:
        return False
    try:
        return bool(verify(parse("$" + str(gold) + "$"), parse("$" + str(pred) + "$")))
    except Exception:   # 解析器对怪异 LaTeX 会抛各种异常, 一律按不等价
        return False


def grade_math(content, gold):
    boxed = last_boxed(content or "")
    g = norm_math(str(gold))
    if boxed is not None:
        b = norm_math(boxed)
        if b == g:
            return boxed, "boxed", True
        try:                                         # 整数金标, 答案写成 042 / 42.0 这类
            if re.fullmatch(r"-?\d+", g) and float(b) == float(g):
                return boxed, "boxed", True
        except ValueError:
            pass
        return boxed, "boxed", math_equiv(boxed, gold)
    ints = re.findall(r"(?<![\d.])-?\d+(?![\d.])", content or "")   # unsloth: fallback exact integer
    if ints:
        return ints[-1], "fallback", norm_math(ints[-1]) == g
    return None, "none", False


def grade_mc(content, gold):
    c = content or ""
    m = re.findall(r"Answer\s*:\s*\**\(?([A-J])\)?", c)
    if m:
        return m[-1], "answer", m[-1] == gold
    b = last_boxed(c)
    if b and re.fullmatch(r"\(?([A-J])\)?", b.strip()):
        x = re.sub(r"[()]", "", b.strip()); return x, "boxed", x == gold
    m = re.findall(r"answer is \(?([A-J])\)?", c)
    if m:
        return m[-1], "answer_is", m[-1] == gold
    m = re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", c.strip()[-40:])
    if m:
        return m[-1], "lone", m[-1] == gold
    return None, "none", False


def grade(row, content):
    if row["grade"] == "math":
        return grade_math(content, row["gold"])
    if row["grade"] == "mc":
        return grade_mc(content, row["gold"])
    return None, "raw", None


# ── 请求 ──────────────────────────────────────────────────────────────────────
def ask(server, row, args, seed):
    msgs = ([{"role": "system", "content": row["system"]}] if row.get("system") else []) + \
           [{"role": "user", "content": row["prompt"]}]
    body = {"model": "ds4", "messages": msgs, "max_tokens": args.max_tokens, "temperature": args.temp,
            "top_p": args.top_p, "seed": seed, "reasoning_effort": args.effort}
    req = urllib.request.Request(server + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        # errors="replace": 11-01 04:41 IFEval 跑到 36 分钟服务端回了一段不是合法 UTF-8 的正文(温 1 采出的字节级 token 拼不成字),
        # 严格解码炸掉整项; 坏字换成 U+FFFD 继续, 判分器看到的是 �(服务端该不该转义是引擎侧的事, 记在 fable5)
        return json.loads(r.read().decode(errors="replace")), time.time() - t0


def summarize(recs, args, n_items):
    n = len(recs)
    if not n:
        return {"answers": 0}
    graded = [a for a in recs if a["correct"] is not None]
    acc = sum(a["correct"] for a in graded) / len(graded) if graded else None
    se = (acc * (1 - acc) / len(graded)) ** 0.5 if graded else None
    by = {}
    for a in graded:
        by.setdefault(a["id"], []).append(a["correct"])
    return {"answers": n, "items_seen": len({a["id"] for a in recs}), "items_total": n_items,
            "graded": len(graded), "pass_at_1": None if acc is None else round(100 * acc, 2),
            "se_pct": None if se is None else round(100 * se, 2),
            "truncated_pct": round(100 * sum(a["truncated"] for a in recs) / n, 2),
            "no_answer": sum(a["how"] == "none" for a in recs),
            "mean_completion_tokens": round(sum(a["completion_tokens"] or 0 for a in recs) / n),
            "wall_min": round(sum(a["secs"] for a in recs) / 60 / max(1, args.jobs), 1),
            "per_item": {str(k): f"{sum(v)}/{len(v)}" for k, v in sorted(by.items(), key=lambda kv: str(kv[0]))},
            "config": vars(args)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8001")
    ap.add_argument("--samples", type=int, default=16, help="每题最多样本数(样本优先级: 先把所有题各跑 1 个, 再第 2 个…)")
    ap.add_argument("--budget-min", type=float, default=0, help="时间预算(分钟), 到点不再发新请求; 0 = 不限")
    ap.add_argument("--jobs", type=int, default=1, help="并发请求数")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--effort", default="high", help="ds4-server: high|max 开思考; none 关")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--regrade", action="store_true", help="不发请求, 只用 raw/ 里的原始返回重判已有答案并重写 answers/summary")
    ap.add_argument("--exclude", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "data", "trained_exclude.jsonl"),
                    help="进过训练料的题(bench_fail_to_task.py exclude 写的清单), 跑分跳过; 给 '' 不排除")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    rnd = random.Random(SEED_BASE); rnd.shuffle(rows)   # 先打乱再排除: 剩下的题顺序与排除前一致, 前后趟可比
    task = os.path.basename(args.data).rsplit(".jsonl", 1)[0]
    if args.exclude and os.path.exists(args.exclude):
        ex = {r["id"] for r in map(json.loads, open(args.exclude)) if r["task"] == task}
        n0 = len(rows); rows = [r for r in rows if r["id"] not in ex]
        if n0 - len(rows): print(f"# 排除进过训练料的题 {n0 - len(rows)} 道({args.exclude})", flush=True)
    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    ans_path = os.path.join(args.out, "answers.jsonl")
    done = {}
    if os.path.exists(ans_path):
        for l in open(ans_path):
            if l.strip():
                a = json.loads(l); done[(a["id"], a["sample"])] = a
    if args.regrade:
        by_id = {r["id"]: r for r in rows}; changed = 0
        for key, a in done.items():
            raw = json.load(open(os.path.join(args.out, "raw", f"{a['id']}_{a['sample']}.json")))
            content = raw["response"]["choices"][0]["message"].get("content") or ""
            pred, how, ok = grade(by_id[a["id"]], content)
            if (pred, how, ok) != (a["pred"], a["how"], a["correct"]):
                print(f"题 {a['id']} s{a['sample']}: {a['correct']} → {ok} (答 {str(pred)[:30]} 金 {str(a['gold'])[:30]})", flush=True)
                changed += 1
            a.update(pred=pred, how=how, correct=ok)
        with open(ans_path, "w") as f:
            for a in done.values():
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        summary = summarize(list(done.values()), args, len(rows)); summary["regraded"] = changed
        json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), ensure_ascii=False, indent=1)
        print(json.dumps({k: v for k, v in summary.items() if k not in ("per_item", "config")}, ensure_ascii=False), flush=True)
        return
    todo = [(r, s) for s in range(args.samples) for r in rows if (r["id"], s) not in done]
    print(f"# {os.path.basename(args.data)}: {len(rows)} 题, 最多 {args.samples} 样本, 待跑 {len(todo)}, 已有 {len(done)}; "
          f"预算 {args.budget_min} min, 并发 {args.jobs}, max_tokens={args.max_tokens} temp={args.temp} top_p={args.top_p} "
          f"effort={args.effort}", flush=True)
    t_start = time.time(); lock = threading.Lock(); fa = open(ans_path, "a"); stop = {"flag": False}

    def one(item):
        r, s = item
        if stop["flag"] or (args.budget_min and time.time() - t_start > args.budget_min * 60):
            stop["flag"] = True; return
        try:
            resp, secs = ask(args.server, r, args, SEED_BASE + s)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            print(f"题 {r['id']} 样本 {s}: 请求失败 {e!r}; 后面的不发了, 查服务再续跑", flush=True)
            stop["flag"] = True; return
        ch = resp["choices"][0]; msg = ch["message"]
        content = msg.get("content") or ""; reasoning = msg.get("reasoning_content") or ""
        pred, how, ok = grade(r, content)
        usage = resp.get("usage", {})
        rec = {"id": r["id"], "sample": s, "gold": r.get("gold"), "pred": pred, "how": how, "correct": ok,
               "finish_reason": ch.get("finish_reason"), "truncated": ch.get("finish_reason") == "length",
               "completion_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
               "reasoning_chars": len(reasoning), "content_chars": len(content), "secs": round(secs, 1)}
        with open(os.path.join(args.out, "raw", f"{r['id']}_{s}.json"), "w") as fr:
            json.dump({"request": {"prompt": r["prompt"], "system": r.get("system")}, "response": resp}, fr, ensure_ascii=False, indent=1)
        with lock:
            done[(r["id"], s)] = rec
            fa.write(json.dumps(rec, ensure_ascii=False) + "\n"); fa.flush()
            n = len(done); g = [a for a in done.values() if a["correct"] is not None]
            acc = f"{100 * sum(a['correct'] for a in g) / len(g):.2f}%" if g else "—"
            tr = 100 * sum(a["truncated"] for a in done.values()) / n
            mark = {True: "对", False: "错", None: "·"}[ok]
            print(f"[{n}] {(time.time() - t_start) / 60:5.1f} min 题 {str(r['id'])[:12]:>12} s{s}: {mark} 答 {str(pred)[:20]} 金 {str(r.get('gold'))[:20]} ({how}) "
                  f"{rec['completion_tokens']} tok {secs:.0f}s {rec['finish_reason']} | 累计 pass@1 {acc} 截断 {tr:.1f}%", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(one, todo))
    fa.close()
    summary = summarize(list(done.values()), args, len(rows))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_item", "config")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
