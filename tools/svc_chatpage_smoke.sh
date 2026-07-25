#!/usr/bin/env bash
# svc_chatpage_smoke.sh — 网页 chat 路由上线冒烟: 等服务空闲 → 只重启 coordinator
# (worker/看门狗保留, fast-probe 铁律: worker 常驻复用) → 三连冒烟:
#   ① GET /            (web/chat.html, 新路由)
#   ② GET /v1/models   (页面的连接探测)
#   ③ 8-token 流式 /v1/chat/completions (与页面发送的请求字节同构)
#
# 用法: MODEL=gguf/go-onebit/ds4-code2b.gguf RESID=gguf/sidecars/go2b-hot-overlay.gguf \
#       [WAIT_PID=<先等这个进程退出>] tools/svc_chatpage_smoke.sh
# 环境经 svc.sh 透传; MODEL/RESID 必须与在跑服务一致(禁换模型的隐式漂移)。
set -uo pipefail
DIR=${DIR:-/Users/fodelf/git/ds4-main}
PORT=${PORT:-8013}
WAIT_PID=${WAIT_PID:-}
log(){ echo "[chatpage-smoke $(date +%H:%M:%S)] $*"; }

if [ -n "$WAIT_PID" ]; then
  log "等待占用方 pid=$WAIT_PID 退出…"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  log "占用方已退出"
fi

# 空闲闸: 不打断任何在跑请求 (最多再等 10 分钟)
for _ in $(seq 1 60); do
  est=$(lsof -nP -iTCP:"$PORT" -sTCP:ESTABLISHED 2>/dev/null | grep -c ESTABLISHED || true)
  [ "${est:-0}" -eq 0 ] && break
  sleep 10
done
est=$(lsof -nP -iTCP:"$PORT" -sTCP:ESTABLISHED 2>/dev/null | grep -c ESTABLISHED || true)
[ "${est:-0}" -eq 0 ] || { log "★端口仍有活跃连接, 拒绝重启★"; exit 3; }

log "杀 coordinator(只进程; worker/看门狗不动)"
pkill -f "^\./ds4-server .*--port $PORT" 2>/dev/null
sleep 2
( cd "$DIR" && MODEL="${MODEL:?必须显式给 MODEL(与在跑服务一致)}" RESID="${RESID-}" tools/svc.sh up )

log "── 冒烟① GET / (chat 页面) ──"
code=$(curl -s --noproxy '*' -o /tmp/chatpage_smoke.html -w "%{http_code} %{size_download}" "http://127.0.0.1:$PORT/")
echo "HTTP $code"
grep -o "<title>[^<]*</title>" /tmp/chatpage_smoke.html || true
log "── 冒烟② GET /v1/models ──"
curl -s --noproxy '*' "http://127.0.0.1:$PORT/v1/models" | head -c 300; echo
log "── 冒烟③ 8-token 流式 chat (与页面请求同构) ──"
curl -sN --noproxy '*' "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"1+1=?"}],"stream":true,"stream_options":{"include_usage":true},"temperature":0,"max_tokens":8}' \
  | head -30
log "冒烟结束 (页面地址: http://127.0.0.1:$PORT/)"
