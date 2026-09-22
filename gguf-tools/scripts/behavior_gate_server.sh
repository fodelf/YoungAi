#!/bin/bash
# behavior_gate_server.sh — 行为门 v2(真部署路径, 跑在 M1 本机):
# ds4-server base-native+默认stops+--tool-primer, A/B = 无 --soul vs --soul(纯示例 v3)。
# 4 场景 curl 探针带最小工具 schema(soul 注入在 tools header, 无 tools 不注入)。
# CLI 裸续写门(v1)的两个缺陷在此修正: ①规则清单→纯示例载体 ②无 stops→server 默认 stops。
# 输出: /tmp/behavior_gate_srv.report ; footprint 看门狗 11776MB 红线全程在。
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL="gguf/go-onebit/ds4-code1b-v2.gguf"
SOUL="gguf-tools/data/corpus/soul/soul_server_v3.txt"
PORT=8022
REPORT=/tmp/behavior_gate_srv.report
: > "$REPORT"

QS=( "What is cache penetration and how do you defend against it?"
     "Is the race condition in the store package fixed?"
     "go test failed: panic at store.go:42. What is your next step?"
     "What does \`go vet\` do?" )
TAGS=(g1_fact g2_honest g3_debug g4_route)

probe() { # $1=question → stdout: content + tool_calls + finish_reason
python3 - "$1" <<'PEOF'
import json,sys,urllib.request
q=sys.argv[1]
body={"model":"ds4","messages":[{"role":"user","content":q}],
 "temperature":0,"max_tokens":64,
 "tools":[{"type":"function","function":{"name":"Bash","description":"Run a shell command",
   "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]}
req=urllib.request.Request("http://127.0.0.1:8022/v1/chat/completions",
 data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 绕全局代理(07-14 502教训)
try:
  with opener.open(req,timeout=900) as r:
    d=json.loads(r.read())
  m=d["choices"][0]["message"]
  out=(m.get("content") or "")
  for tc in (m.get("tool_calls") or []):
    out+=f"\n[tool_call {tc['function']['name']}({tc['function']['arguments']})]"
  print(out.strip())
  print(f"[finish={d['choices'][0].get('finish_reason')}]")
except Exception as e:
  print(f"[probe error: {e}]")
PEOF
}

leg() { # $1=腿名 $2=soul文件(空=bare)
  pkill -9 -x ds4-server 2>/dev/null; pkill -9 -x ds4 2>/dev/null; sleep 2
  local args=(-m "$MODEL" --port "$PORT" --ctx 8192 --nothink --tool-primer
              --base-native --residual gguf/sidecars/code-hot-res-v2.gguf
              --mem-budget-mb 12000 --prefill-chunk 512)
  [ -n "$2" ] && args+=(--soul "$2")
  echo "[srv] 起 $1 server..." >&2
  ./ds4-server "${args[@]}" > "/tmp/bg_srv_$1.log" 2>&1 &
  local SP=$!
  ( while kill -0 "$SP" 2>/dev/null; do   # 全程看门狗
      MB=$(footprint -p "$SP" 2>/dev/null | grep -Eo 'Footprint: *[0-9.]+ *[KMG]B' | head -1 \
           | awk '{v=$2;u=$3;if(u=="GB")v*=1024;else if(u=="KB")v/=1024;printf "%d",v}')
      if [ -n "${MB:-}" ] && [ "$MB" -gt 11776 ]; then
        echo "[watchdog] $1 ${MB}MB 超红线 → kill" >&2; kill -9 "$SP" 2>/dev/null; break
      fi; sleep 3
    done ) & local WD=$!
  local up=0
  for _ in $(seq 1 150); do
    curl --noproxy '*' -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" && { up=1; break; }
    kill -0 "$SP" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" != 1 ]; then
    echo "════ $1: server 未就绪(日志尾: $(tail -2 /tmp/bg_srv_$1.log))" >> "$REPORT"
    kill -9 "$SP" "$WD" 2>/dev/null; return 1
  fi
  echo "[srv] $1 就绪, 探针开始" >&2
  for k in 0 1 2 3; do
    echo "[gate] $1/${TAGS[$k]}" >&2
    { echo "════ $1 ${TAGS[$k]} ════"; echo "Q: ${QS[$k]}"; probe "${QS[$k]}"; echo; } >> "$REPORT"
  done
  kill -9 "$SP" "$WD" 2>/dev/null
}

# LEGS=bare|souled|both(默认): 单腿重跑用(如只修了影响单腿的 bug)
case "${LEGS:-both}" in
    bare)   leg bare "" ;;
    souled) leg souled "$SOUL" ;;
    *)      leg bare ""; leg souled "$SOUL" ;;
esac
pkill -9 -x ds4-server 2>/dev/null
echo "[gate] 完成 → $REPORT" >&2
