#!/bin/bash
# agent_loop_probe.sh — 最小两轮 agent 回路真实场景探针(2026-07-22 固化)。
# 判决: 轮1=任务→真 tool_use(伪造检测); 轮2=消化真 tool_result→是否推进(复读吸引子检测)。
# 铁则(实证): max_tokens 必须 > 栈 FREE_BUDGET(96), 否则 primer 注帧无空间(轮0 教训)。
# 用法: [PORT=8013] [MT=256] ./agent_loop_probe.sh   # 输出全量原样+机器可判行
set -uo pipefail
PORT="${PORT:-8013}" MT="${MT:-256}" python3 - <<'PEOF'
import json, urllib.request, time, os
PORT=os.environ["PORT"]; MT=int(os.environ["MT"])
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def call(messages, tag):
    body = {"model":"ds4","max_tokens":MT,"temperature":0,
        "tools":[{"name":"Bash","description":"Run a shell command",
                  "input_schema":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}],
        "messages":messages}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/messages",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json","anthropic-version":"2023-06-01"})
    t0=time.time()
    with OP.open(req, timeout=3600) as r: d=json.loads(r.read())
    print(f"───轮{tag} 用时{time.time()-t0:.0f}s stop_reason={d.get('stop_reason')}")
    for b in d.get("content",[]):
        if b["type"]=="text": print("TEXT:", repr(b["text"])[:600])
        elif b["type"]=="tool_use": print("TOOL_USE:", b["name"], json.dumps(b["input"],ensure_ascii=False)[:300])
    return d
m=[{"role":"user","content":"go test failed: panic at store.go:42 (index out of range). Find the bad line and fix it."}]
d1=call(m,1)
tu=[b for b in d1["content"] if b["type"]=="tool_use"]
if not tu:
    print("VERDICT round1=NO_TOOL_USE (检查 max_tokens>FREE_BUDGET / primer)"); raise SystemExit(1)
print("VERDICT round1=REAL_TOOL_USE")
m.append({"role":"assistant","content":d1["content"]})
m.append({"role":"user","content":[{"type":"tool_result","tool_use_id":tu[0]["id"],
    "content":"store.go:40:func sum(arr []int) int {\nstore.go:41:\ttotal := 0\nstore.go:42:\tfor i := 0; i <= len(arr); i++ {\nstore.go:43:\t\ttotal += arr[i]"}]})
d2=call(m,2)
tu2=[b for b in d2["content"] if b["type"]=="tool_use"]
same = bool(tu2) and tu2[0]["input"].get("command")==tu[0]["input"].get("command")
if same: print("VERDICT round2=REPEATED_CALL (复读吸引子, 未推进)")
elif tu2: print("VERDICT round2=ADVANCED (新动作:", tu2[0]["input"].get("command","")[:80], ")")
else: print("VERDICT round2=TEXT_ONLY (看 TEXT 是否含正确修复语义)")
PEOF
