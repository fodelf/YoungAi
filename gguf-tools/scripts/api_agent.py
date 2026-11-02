#!/usr/bin/env python3
"""api_agent.py — agent 多轮料(2026-10-26 晚): dsh 抓到的官方会话每一轮, 让官方 API 在同一前缀上重采一次拿 top-20, 再渲染成学生看见的字节。

为什么: 学生在 dsh 里"看一次全 Good 一笔不改"(v3)或"改 125 轮收不了手"(v3g) —— "看了结果之后交还是接着改"是多轮决策, 单轮料里没有。
料的每一轮 = 前缀(system + tools + 到这一轮为止的历史, dsh 真发出的请求体)+ 靶(官方从这个前缀重采的 思考/正文/工具调用 + top-20)。
前缀用 ds4-server --render-anthropic 按部署同一套渲染(10-24 教训: 训练格式 ≠ 判决格式, 收益不兑现); 工具调用的 DSML 由同一渲染器写, API 不给它的分布, one-hot。

三步(spark 仓库根, ~/v41env 的 python; 抓包目录由 Mac 侧 tests/asst/dsh_agent_collect.sh --ship 送来):
  resample --key K --captures gguf/go-onebit/restore/agent_capture/<批> --out agent_raw.jsonl --hf HF [--jobs 4]
      每个 req_NNN.json(Anthropic 请求体)→ 转 OpenAI chat 体(system / user / assistant{reasoning_content, content, tool_calls} / tool)→
      chat/completions 流式 + logprobs top-20 → 记 reasoning / content / tool_calls / lp(与 api_teacher collect 同格式)/ 原请求体路径。
  rows     --raw agent_raw.jsonl --out rows.jsonl --gguf <底座 GGUF> [--server ./ds4-server]
      每条: 请求体 + (请求体 ∪ 重采出的 assistant 消息) 两份都渲染, 后者以前者为字节前缀, 差 = 靶文本(思考</think>正文 DSML <｜end▁of▁sentence｜>);
      训练行 {"prefix": 前缀文本(去掉 BOS 字面), "text": 靶文本, "eos": false(EOS 字面已在 text 里)}。ptf 按内容在序列里找思考块/正文块, 其余 one-hot。
出错会怎样: 带图片块的轮(read_image)直接跳过并计数(训练器不吃图); 渲染两份不成前缀关系 → 整条作废并打出来(渲染器与模板对不上要查, 不猜)。"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import http.client
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_teacher import hf_byte_table, post, text_from_lp  # noqa: E402
from think_trace_collect import API_URL  # noqa: E402

BOS = "<｜begin▁of▁sentence｜>"


def blocks_text(c):
    """Anthropic content(字符串或块数组)里的文字; 有图片块返回 None(这轮不要)"""
    if isinstance(c, str): return c
    out = []
    for b in c or []:
        t = b.get("type")
        if t == "text": out.append(b.get("text", ""))
        elif t == "image": return None
        elif t == "tool_result":
            inner = blocks_text(b.get("content", ""))
            if inner is None: return None
            out.append(inner)
    return "\n\n".join(out)


def to_openai(body):
    """Anthropic Messages 请求体 → OpenAI chat 体(messages + tools); 返回 None = 含图片"""
    msgs = []
    sys_t = blocks_text(body.get("system", ""))
    if sys_t: msgs.append({"role": "system", "content": sys_t})
    for m in body.get("messages", []):
        c = m.get("content")
        if m["role"] == "user":
            if isinstance(c, str): msgs.append({"role": "user", "content": c}); continue
            texts = []
            for b in c or []:
                if b.get("type") == "tool_result":
                    inner = blocks_text(b.get("content", ""))
                    if inner is None: return None
                    if texts: msgs.append({"role": "user", "content": "\n\n".join(texts)}); texts = []
                    msgs.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": inner})
                elif b.get("type") == "text": texts.append(b.get("text", ""))
                elif b.get("type") == "image": return None
            if texts: msgs.append({"role": "user", "content": "\n\n".join(texts)})
        elif m["role"] == "assistant":
            a = {"role": "assistant", "content": ""}; calls = []; think = []
            for b in (c if isinstance(c, list) else [{"type": "text", "text": c or ""}]):
                t = b.get("type")
                if t == "text": a["content"] += b.get("text", "")
                elif t == "thinking": think.append(b.get("thinking", ""))
                elif t == "tool_use": calls.append({"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}), ensure_ascii=False)}})
            if think: a["reasoning_content"] = "".join(think)
            if calls: a["tool_calls"] = calls
            msgs.append(a)
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {"type": "object"})}} for t in body.get("tools", [])]
    out = {"messages": msgs}
    if tools: out["tools"] = tools
    for k in ("reasoning_effort", "thinking"):   # dsh 经 Anthropic 接口带的思考档/扩展字段原样带过去, 两边同一家 API 认同一套
        if k in body: out[k] = body[k]
    return out


def stream_chat(key, body, timeout):
    ent = {"reasoning_content": [], "content": []}; txt = {"reasoning_content": [], "content": []}; calls = {}; usage = finish = None
    with post(API_URL, key, body, timeout) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data:") or ln[5:].strip() == "[DONE]": continue
            ev = json.loads(ln[5:]); usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}; lp = ch.get("logprobs") or {}
                for k in ent:
                    if d.get(k): txt[k].append(d[k])
                    ent[k].extend(lp.get(k) or [])
                for tc in d.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    slot["id"] = tc.get("id") or slot["id"]; fn = tc.get("function") or {}
                    slot["name"] = fn.get("name") or slot["name"]; slot["arguments"] += fn.get("arguments") or ""
                finish = ch.get("finish_reason") or finish
    return txt, ent, [calls[i] for i in sorted(calls)], finish, usage


def resample_one(key, model, item, timeout, b2id, id2b):
    body = dict(item["oai"], model=model, max_tokens=item["max_tokens"], logprobs=True, top_logprobs=20, stream=True, stream_options={"include_usage": True})
    delay = 5
    for attempt in range(5):
        try:
            t0 = time.time(); txt, ent, calls, finish, usage = stream_chat(key, body, timeout)
            lp = {}
            for k in ("reasoning_content", "content"):
                rows = []
                for e in ent[k]:
                    cid = [b2id.get(bytes(t["bytes"] or [])) for t in e.get("top_logprobs") or []]
                    rows.append([b2id.get(bytes(e["bytes"] or [])), e["logprob"], [c for c in cid if c is not None], [t["logprob"] for c, t in zip(cid, e["top_logprobs"]) if c is not None]])
                lp[k] = rows
            rs, cs = text_from_lp(lp["reasoning_content"], id2b), text_from_lp(lp["content"], id2b)
            if rs is None: rs = "".join(txt["reasoning_content"])
            if cs is None: cs = "".join(txt["content"])
            return dict(item, oai=None, ok=finish in ("stop", "tool_calls"), finish=finish, reasoning=rs, content=cs, tool_calls=calls, lp=lp, usage=usage, model=model,
                        secs=round(time.time() - t0, 1))
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError, http.client.HTTPException) as e:   # IncompleteRead 不是 OSError(见 api_teacher.collect_one)
            msg = f"HTTP {e.code} {e.read().decode('utf-8', 'replace')[:300]}" if isinstance(e, urllib.error.HTTPError) else type(e).__name__
            if attempt == 4: return dict(item, oai=None, ok=False, finish=f"error:{msg}", reasoning="", content="", tool_calls=[], lp={})
        print(f"  [{item['id']}] {msg}, {delay}s 后重试", flush=True); time.sleep(delay); delay *= 3


def cmd_resample(a):
    key = open(a.key, encoding="utf-8").read().strip(); _, b2id = hf_byte_table(a.hf); id2b = {i: b for b, i in b2id.items()}
    items, nimg, ntitle = [], 0, 0
    for run in sorted(os.listdir(a.captures)):
        d = os.path.join(a.captures, run)
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if not (f.startswith("req_") and f.endswith(".json")): continue
            body = json.load(open(os.path.join(d, f), encoding="utf-8"))
            if not body.get("tools"): ntitle += 1; continue   # dsh 顺手发的会话起标题之类的小请求(没有工具表), 不是 agent 轮
            oai = to_openai(body)
            if oai is None: nimg += 1; continue
            items.append({"id": f"{run}/{f[4:-5]}", "req": os.path.join(d, f), "oai": oai, "max_tokens": int(body.get("max_tokens") or 65536), "src": "agent"})
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out, encoding="utf-8"):
            r = json.loads(ln)
            if r.get("ok"): done.add(r["id"])
    todo = [it for it in items if it["id"] not in done]
    print(f"resample: 抓包轮 {len(items) + nimg + ntitle}, 带图跳过 {nimg}, 无工具表(起标题等)跳过 {ntitle}, 已有 ok {len(done)}, 本次 {len(todo)}, {a.jobs} 路", flush=True)
    lock = threading.Lock(); t0 = time.time(); nok = 0
    with open(a.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.jobs) as ex:
        for i, fu in enumerate(as_completed([ex.submit(resample_one, key, a.model, it, a.timeout, b2id, id2b) for it in todo]), 1):
            r = fu.result(); nok += bool(r["ok"])
            with lock: f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(f"  {i}/{len(todo)} [{r['id']}] {'ok' if r['ok'] else '★' + str(r['finish'])} 思考 {len(r.get('reasoning', ''))} 字 正文 {len(r.get('content', ''))} 字 工具调用 {len(r.get('tool_calls') or [])} {r.get('secs', '?')} s, 累计 {time.time() - t0:.0f} s", flush=True)
    print(f"resample 完成: ok {nok}/{len(todo)} → {a.out}", flush=True)


def assistant_blocks(r):
    blocks = []
    if r.get("reasoning"): blocks.append({"type": "thinking", "thinking": r["reasoning"]})
    if r.get("content"): blocks.append({"type": "text", "text": r["content"]})
    for c in r.get("tool_calls") or []:
        try: inp = json.loads(c["arguments"]) if c.get("arguments") else {}
        except json.JSONDecodeError: inp = {"_raw": c.get("arguments", "")}
        blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": inp})
    return blocks


def head_cut(recs, n, hf, out):
    """--head N(11-01, 风格档位轮): 每轮只留官方思考的前 n 个 token 当靶(文字按 API token 字节拼回, lp 同步截, 正文/工具调用清空, 标 head), 截过的原料另存 out 给 ptf 配表。
    为什么: 鹈鹕质感差定在 agent 上下文里规划开头"抬不抬标准"一位(fable5 11-01), 整轮 1~2 万位里只有开头几千位是靶, 其余只是算力; 与 api_branch.py head 同一做法。
    rows 见 head 只渲染到思考截断处: 不接 </think>、不补 EOS(接了等于教它在这儿停止思考)。"""
    from api_branch import prefix_text
    _, b2id = hf_byte_table(hf); id2b = {i: b for b, i in b2id.items()}; o = []
    for r in recs:
        rows = (r.get("lp") or {}).get("reasoning_content") or []
        if len(rows) < 16 or any(x[0] not in id2b for x in rows[:n + 8]): continue
        txt, cut = prefix_text([x[0] for x in rows], min(n, len(rows)), id2b)
        o.append(dict(r, id=f"{r['id']}#h{n}", reasoning=txt, content="", tool_calls=[], head=True, finish="length", lp={"reasoning_content": rows[:cut], "content": []}))
    with open(out, "w", encoding="utf-8") as f:
        for r in o: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"head: {len(o)}/{len(recs)} 轮 × 思考前 {n} 位 → {out}")
    return o


def cmd_rows(a):
    recs = [json.loads(ln) for ln in open(a.raw, encoding="utf-8") if ln.strip()]; recs = [r for r in recs if r.get("ok") and r.get("lp")]
    if a.head: recs = head_cut(recs, a.head, a.hf, a.head_raw)
    with tempfile.TemporaryDirectory(prefix="agent_render_") as td:
        lst = os.path.join(td, "list.tsv")
        with open(lst, "w", encoding="utf-8") as lf:
            for i, r in enumerate(recs):
                body = json.load(open(r["req"], encoding="utf-8")); body.pop("stream", None)
                full = dict(body, messages=body["messages"] + [{"role": "assistant", "content": assistant_blocks(r)}])
                for tag, b in (("p", body), ("f", full)):
                    p = os.path.join(td, f"{i:04d}{tag}.json"); json.dump(b, open(p, "w", encoding="utf-8"), ensure_ascii=False)
                    lf.write(f"{p}\t{os.path.join(td, f'{i:04d}{tag}')}\n")
        cmd = [a.server, "--cpu", "-m", a.gguf, "--render-anthropic", lst]
        rc = subprocess.run(cmd, capture_output=True, text=True)
        if rc.returncode != 0: print(rc.stderr[-2000:]); sys.exit(f"渲染失败 rc={rc.returncode}")
        n = bad = 0
        with open(a.out, "w", encoding="utf-8") as f:
            for i, r in enumerate(recs):
                pt = open(os.path.join(td, f"{i:04d}p.txt"), encoding="utf-8").read(); ft = open(os.path.join(td, f"{i:04d}f.txt"), encoding="utf-8").read()
                if not ft.startswith(pt) or not pt.startswith(BOS):
                    bad += 1; print(f"  ★[{r['id']}] 整轮渲染不以前缀渲染开头(前缀 {len(pt)} 字 / 整轮 {len(ft)} 字), 作废★"); continue
                text = ft[len(pt):]
                if r.get("reasoning") and not text.startswith(r["reasoning"]): bad += 1; print(f"  ★[{r['id']}] 靶文本不以重采的思考开头, 作废★"); continue
                if r.get("head"): text = r["reasoning"]   # 只到思考截断处(渲染器给整轮补的 </think> + EOS 不要)
                f.write(json.dumps({"prefix": pt[len(BOS):], "text": text, "eos": False}, ensure_ascii=False) + "\n"); n += 1
    print(f"rows: {n} 行 → {a.out}(作废 {bad}); 靶 = 思考</think>正文 + DSML 工具调用 + <｜end▁of▁sentence｜>, 前缀零梯度")


def cmd_step2(a):
    """第一步重采结果 → 第二步请求体(11-01): 官方在 dsh 里第一步只看目录(思考 200 字 + bash `pwd && ls -la` / glob), 规划(抬标准 / 配色 / 坐标)全在第二步;
    把第一步的 assistant 块(思考 + 工具调用)和【本机真跑出来的】工具结果接在首请求体后面 → <out>/<任务>/req_003.json, 再拿去 resample 就是规划段。
    工具结果照 dsh 的样子: bash = 在空的 git init 临时目录里真跑, stdout 原文(路径换成请求体里声称的工作目录; 非零退出补 [exit code: N]); glob = 空目录 "No files found"。
    别的工具(todo_write 等)这里不会模拟, 那条任务跳过并打出来。在 Mac 跑(dsh 官方臂就是在这台 Mac 上跑的, node / Chrome 探测结果才是同一个环境)。"""
    import re, shutil, subprocess as sp, uuid
    os.makedirs(a.out, exist_ok=True); n = skip = 0
    for r in (json.loads(l) for l in open(a.raw, encoding="utf-8") if l.strip()):
        if not r.get("ok"): skip += 1; print(f"  跳过 {r['id']}: 第一步 {r.get('finish')}"); continue
        task = r["id"].split("/")[0]; body = json.load(open(os.path.join(a.captures, task, "req_002.json"), encoding="utf-8"))
        cwd = re.search(r'Current working directory: "([^"]+)"', blocks_text(body["messages"][0]["content"]) or "").group(1)
        tmp = tempfile.mkdtemp(prefix="project."); sp.run(["git", "init", "-q"], cwd=tmp, check=True); real = os.path.realpath(tmp)
        blocks, results, bad = [{"type": "thinking", "thinking": r["reasoning"], "signature": str(uuid.uuid4())}], [], None
        if r.get("content"): blocks.append({"type": "text", "text": r["content"]})
        for c in r.get("tool_calls") or []:
            inp = json.loads(c["arguments"]) if c.get("arguments") else {}
            blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": inp}); err = False
            if c["name"] == "bash":
                p = sp.run(inp.get("command", ""), shell=True, cwd=tmp, capture_output=True, text=True, timeout=60)
                txt = (p.stdout + p.stderr).replace(real, cwd).replace(tmp, cwd); err = p.returncode != 0
                if err: txt += f"\n[exit code: {p.returncode}]"
            elif c["name"] == "glob": txt = "No files found"
            else: bad = c["name"]; break
            results.append({"type": "tool_result", "tool_use_id": c["id"], "content": [{"type": "text", "text": txt}], "is_error": err})
        shutil.rmtree(tmp, ignore_errors=True)
        if bad: skip += 1; print(f"  跳过 {task}: 第一步调了 {bad}, 这里不模拟"); continue
        body["messages"] = body["messages"][:1] + [{"role": "assistant", "content": blocks}, {"role": "user", "content": results}]
        os.makedirs(os.path.join(a.out, task), exist_ok=True)
        json.dump(body, open(os.path.join(a.out, task, "req_003.json"), "w", encoding="utf-8"), ensure_ascii=False); n += 1
    print(f"step2: {n} 个第二步请求体 → {a.out}(跳过 {skip})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("resample"); r.add_argument("--key", required=True); r.add_argument("--captures", required=True); r.add_argument("--out", required=True); r.add_argument("--hf", required=True)
    r.add_argument("--model", default="deepseek-flash"); r.add_argument("--jobs", type=int, default=4); r.add_argument("--timeout", type=int, default=3600)
    w = sp.add_parser("rows"); w.add_argument("--raw", required=True); w.add_argument("--out", required=True); w.add_argument("--gguf", required=True); w.add_argument("--server", default="./ds4-server")
    w.add_argument("--head", type=int, default=0); w.add_argument("--hf"); w.add_argument("--head-raw")   # --head N 要 --hf(拼字节)与 --head-raw(截过的原料给 ptf)
    s2 = sp.add_parser("step2"); s2.add_argument("--raw", required=True); s2.add_argument("--captures", required=True); s2.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "rows" and a.head and not (a.hf and a.head_raw): ap.error("--head 要同时给 --hf 和 --head-raw")
    {"resample": cmd_resample, "rows": cmd_rows, "step2": cmd_step2}[a.cmd](a)


if __name__ == "__main__":
    main()
