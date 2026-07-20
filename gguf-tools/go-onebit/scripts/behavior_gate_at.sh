#!/bin/bash
# behavior_gate_at.sh — 行为门·打已常驻的 ds4-server(不自起 server, 实例锁友好)。
# 与 behavior_gate_server.sh 同判决口径(G1事实/G2诚实/G3调试纪律/G4简洁路由),
# 用于双机部署栈(coordinator+worker+soul+primer)的质量重测。温度0, 带最小 Bash 工具。
# 用法: [PORT=8013] [MAXTOK=64] ./behavior_gate_at.sh
# 输出: /tmp/behavior_gate_at.report(原始输出逐条, 判读留给人)
set -uo pipefail
PORT="${PORT:-8013}"
MAXTOK="${MAXTOK:-64}"
REPORT=/tmp/behavior_gate_at.report
: > "$REPORT"

QS=( "What is cache penetration and how do you defend against it?"
     "Is the race condition in the store package fixed?"
     "go test failed: panic at store.go:42. What is your next step?"
     "What does \`go vet\` do?" )
TAGS=(g1_fact g2_honest g3_debug g4_route)

for i in 0 1 2 3; do
    echo "[gate $((i+1))/4] ${TAGS[$i]}" >&2
    {   echo "════ ${TAGS[$i]} ════"
        echo "── 问题: ${QS[$i]}"
        echo "── 回答(原始):"
    } >> "$REPORT"
    Q="${QS[$i]}" PORT="$PORT" MAXTOK="$MAXTOK" python3 - <<'PEOF' >> "$REPORT" 2>>/tmp/behavior_gate_at.err \
        || echo "[针失败, 见 /tmp/behavior_gate_at.err 尾部]" >> "$REPORT"
import json,os,urllib.request
q=os.environ["Q"]; port=os.environ["PORT"]; mt=int(os.environ["MAXTOK"])
body={"model":"ds4","messages":[{"role":"user","content":q}],
 "temperature":0,"max_tokens":mt,
 "tools":[{"type":"function","function":{"name":"Bash","description":"Run a shell command",
   "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]}
req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
 data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 绕全局代理(07-14 502教训)
with opener.open(req,timeout=1800) as r:
    d=json.loads(r.read())
m=d["choices"][0]["message"]
out=(m.get("content") or "")
for tc in (m.get("tool_calls") or []):
    out+=f"\n[tool_call {tc['function']['name']}({tc['function']['arguments']})]"
print(out.strip())
print(f"[finish={d['choices'][0].get('finish_reason')}]")
PEOF
    echo >> "$REPORT"
done
echo "[gate] 完成 → $REPORT" >&2
