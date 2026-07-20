#!/usr/bin/env python3
"""dsml_verify.py — P2 判决探针: 长上下文参数命中率 A/B (步骤 6/6)。

判决指标 (Gate v2): 工具参数值逐字==用户指令里的真值。
  短上下文 (~0 pad) N 条 + 长上下文 (~8k tok pad) N 条, 各算命中率。
  探针是 held-out: 与训练语料 (dsml_corpus_gen.py) 不同 seed + 不相交的
  目录/文件/函数池 —— 侧车若只是背了训练值, 这里不会得分。

用法: python3 dsml_verify.py --label baseline [--n 20] [--port 8013]
  → /tmp/dsml_verify_{label}.json (逐条原始输出全存 —— 铁律: 原始输出给用户,
    我方判读只是参考); 若 baseline 与 sidecar 两份都在, 自动打印 A/B 对比。
前置: tools/svc.sh 服务在跑 (baseline=不带 CORR, sidecar=mount 之后)。
"""
import argparse
import json
import os
import random
import sys
import urllib.request

# held-out 池: 与 dsml_corpus_gen.py 的 DIRS/FILES/FUNCS 完全不相交
DIRS = ['/var/svc/app', './internal/store', '/Users/qa/repo42', './cmd/cli',
        '/data/pipeline', './pkg/render']
FILES = ['server.go', 'store_test.go', 'client.go', 'middleware.go',
         'worker.go', 'codec.go']
FUNCS = ['validateToken', 'flattenTree', 'rateLimit', 'decodeFrame']
TOOLS = {
    'Read':  ('file_path', lambda r, d, f: f'{d}/{f}'),
    'Bash':  ('command',  lambda r, d, f: r.choice([
                  f'go test {d}/...', f'cat {d}/{f}', f'go vet {d}/...'])),
    'Write': ('file_path', lambda r, d, f: f'{d}/{f}'),
    'Glob':  ('pattern',   lambda r, d, f: f'{d}/**/*.go'),
}
ASKS = {
    'Read':  ['{v} 里的 {fn} 有问题, 先读这个文件。', 'Open {v} and inspect {fn}.'],
    'Bash':  ['先跑 `{v}` 看看输出。', 'Run {v} and report.'],
    'Write': ['把修复写回 {v}。', 'Save the fix to {v}.'],
    'Glob':  ['列出匹配 {v} 的文件。', 'List files matching {v}.'],
}
SCHEMAS = [
    {"name": "Read", "description": "Reads a file from the local filesystem.",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}},
                      "required": ["file_path"]}},
    {"name": "Bash", "description": "Executes a bash command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}},
                      "required": ["command"]}},
    {"name": "Write", "description": "Writes a file to the local filesystem.",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"},
                                                        "content": {"type": "string"}},
                      "required": ["file_path", "content"]}},
    {"name": "Glob", "description": "Fast file pattern matching.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"]}},
]
PAD_SNIPPET = ('func helper%d(xs []int) int {\n\tacc := 0\n\tfor i, x := range xs {\n'
               '\t\tif x%%2 == 0 { acc += x } else { acc -= i }\n\t}\n\treturn acc\n}\n')


def make_probes(n, pad_tok, seed):
    r = random.Random(seed)
    probes = []
    for _ in range(n):
        tool = r.choice(list(TOOLS))
        pname, vgen = TOOLS[tool]
        d, f = r.choice(DIRS), r.choice(FILES)
        v = vgen(r, d, f)
        ask = r.choice(ASKS[tool]).format(v=v, fn=r.choice(FUNCS))
        pad = ''
        if pad_tok:
            chunks, approx, k = [], 0, 0
            while approx < pad_tok * 3:
                chunks.append(PAD_SNIPPET % k)
                approx += len(PAD_SNIPPET)
                k += 1
            pad = ('这是当前项目的部分代码(与任务无关的背景):\n```go\n'
                   + ''.join(chunks) + '```\n\n')
        probes.append({"tool": tool, "param": pname, "gold": v, "ask": pad + ask})
    return probes


def call(port, probe, max_tokens):
    body = json.dumps({
        "model": "ds4", "max_tokens": max_tokens, "tools": SCHEMAS,
        "temperature": 0,   # server 只在工具语法段强制 temp0, 参数值区间用这里
        "messages": [{"role": "user", "content": probe["ask"]}],
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages", data=body,
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8013)))
    ap.add_argument("--pad", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    os.environ.setdefault("NO_PROXY", "*")   # 本地代理会劫持 localhost → 502

    out = {"label": a.label, "cases": []}
    for variant, pad in (("short", 0), ("long", a.pad)):
        hits = 0
        probes = make_probes(a.n, pad, seed=20260708)
        for i, p in enumerate(probes):
            try:
                resp = call(a.port, p, a.max_tokens)
            except Exception as e:  # noqa: BLE001 — 判决脚本: 任何失败都记 miss
                resp = {"error": str(e)}
            tus = [b for b in resp.get("content", []) if b.get("type") == "tool_use"]
            got = tus[0]["input"].get(p["param"]) if tus and tus[0].get("name") == p["tool"] else None
            hit = got == p["gold"]
            hits += hit
            out["cases"].append({"variant": variant, "tool": p["tool"],
                                 "gold": p["gold"], "got": got, "hit": hit,
                                 "raw": resp})
            print(f"[{variant} {i+1}/{a.n}] {'HIT ' if hit else 'MISS'} "
                  f"{p['tool']} gold={p['gold']!r} got={got!r}",
                  file=sys.stderr, flush=True)
        out[variant] = {"hits": hits, "total": a.n}
        print(f"== {a.label} {variant}: {hits}/{a.n} ==", file=sys.stderr, flush=True)

    path = f"/tmp/dsml_verify_{a.label}.json"
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"saved {path}", file=sys.stderr)

    # 两份都在 → 打 A/B 判决表
    base_p, side_p = "/tmp/dsml_verify_baseline.json", "/tmp/dsml_verify_sidecar.json"
    if os.path.isfile(base_p) and os.path.isfile(side_p):
        b, s = json.load(open(base_p)), json.load(open(side_p))
        print("\n==== A/B 判决 (参数命中率) ====")
        for var in ("short", "long"):
            print(f"  {var:5s}: baseline {b[var]['hits']}/{b[var]['total']}"
                  f"  →  sidecar {s[var]['hits']}/{s[var]['total']}")


if __name__ == "__main__":
    main()
