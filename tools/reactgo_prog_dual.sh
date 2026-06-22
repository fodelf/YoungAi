#!/usr/bin/env bash
# reactgo-prog 双机(非对称)运行驱动 —— M4 全模型 0:13 + M1 分片 14:output。
# 复用 mtp_pipe 的安全结构: nohup+PID worker / ready 等待 / 双机 RSS 看门狗 / 超时 / 只杀进程不删文件(铁律)。
# 这是诊断/运行驱动, 不改模型。TEMP 是本轮 quality 诊断的关键变量(贪心塌缩 vs 真 bug)。
#
# 安全闸(铁律, 硬约束):
#   - DS4_MEM_BUDGET_MB 各自 GPU 上限内(M4 11500<11840 / M1 10400<10670)+ 引擎 L1 resident gate 拒超预算启动。
#   - 后台 RSS 看门狗: 两机各 12 GiB 红线, 超即杀两边 + abort。
#   - 超时 RUN_TIMEOUT_SEC 兜底。cleanup 只杀 ds4 进程, 绝不删任何文件(尤其不碰 M1 上的分片)。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

M1=${M1:-192.168.1.2}
REMOTE_DIR=${REMOTE_DIR:-/Users/fodelf/ds4-main}
PORT=${PORT:-5599}
M4_MODEL=${M4_MODEL:-gguf/reactgo-prog.gguf}
M1_MODEL=${M1_MODEL:-gguf/reactgo-prog-worker.gguf}
COORD_LAYERS=${COORD_LAYERS:-0:13}
WORKER_LAYERS=${WORKER_LAYERS:-14:output}
M4_BUDGET=${M4_BUDGET:-11500}
M1_BUDGET=${M1_BUDGET:-10400}
MAX_GB=${MAX_GB:-12}
CTX=${CTX:-2048}
NPRED=${NPRED:-80}
TEMP=${TEMP:-0.7}
SEED=${SEED:-1}
RUN_TIMEOUT_SEC=${RUN_TIMEOUT_SEC:-220}
PROMPT=${PROMPT:-"Complete this React todo app component:

import { useState } from 'react';

export default function TodoApp() {
  const [todos, setTodos] = useState([]);
  const [input, setInput] = useState('');
"}
# PREFILL_CHUNK: 降 prefill 分块 → 缩小 MoE prefill scratch(模型贴满 GPU 上限时让 prefill 装得下)。空=引擎默认。
PREFILL_CHUNK=${PREFILL_CHUNK:-}
PC_ENV=""; [ -n "$PREFILL_CHUNK" ] && PC_ENV="DS4_METAL_PREFILL_CHUNK=$PREFILL_CHUNK"
COMMON_ENV="DS4_DIST_REVERSE_CONNECT=1 DS4_METAL_EXPERT_OFFLOAD=0 DS4_METAL_NO_MODEL_WARMUP=1 $PC_ENV"
# DUMP: 设成文件路径 → coordinator 加 --dump-logprobs(贪心续写 + 每步 top-k 分布 JSON, 诊断重复可救性)。
DUMP=${DUMP:-}
DUMP_ARG=""; [ -n "$DUMP" ] && DUMP_ARG="--dump-logprobs $DUMP"
# EXTRA_ENV: 透传任意 env 给 coordinator(如 DS4_METAL_GRAPH_DUMP_* 逐层 dump 中间张量定位 bug)
EXTRA_ENV=${EXTRA_ENV:-}
# FREQ: 文件路径 → 两机各加 DS4_ROUTER_FREQ_FILE(收集 base router raw top-k 偏好, 退出各写各的)
FREQ=${FREQ:-}
FREQ_ENV=""; [ -n "$FREQ" ] && FREQ_ENV="DS4_ROUTER_FREQ_FILE=$FREQ"
COMMON_ENV="$COMMON_ENV $FREQ_ENV"
# SPEC: 任意投机解码 env 透传两机(如 DS4_DIST_COPY_SPEC=1 DS4_DIST_COPY_SPEC_DRAFT=12 DS4_DIST_SPEC_PIPE=1)
SPEC=${SPEC:-}
COMMON_ENV="$COMMON_ENV $SPEC"
WORKER_LOG=/tmp/reactgo_prog_worker.log
WORKER_PID_FILE=/tmp/reactgo_prog_worker.pid
COORD_LOG=/tmp/reactgo_prog_coord.log
COORD_OUT=/tmp/reactgo_prog_coord.out
MAX_KB=$((MAX_GB*1024*1024))
COORD_PID=""
log(){ echo "[reactgo-dual] $*"; }
rss_kb_local(){ ps -o rss= -p "$1" 2>/dev/null | tr -d ' '; }
rss_kb_remote(){ ssh "$M1" "pid=\$(cat '$WORKER_PID_FILE' 2>/dev/null); [ -n \"\$pid\" ] && ps -o rss= -p \$pid 2>/dev/null | tr -d ' '" 2>/dev/null; }

cleanup(){
  trap - INT TERM EXIT
  # 铁律: 只杀进程, 绝不删文件。
  [ -n "$COORD_PID" ] && kill "$COORD_PID" 2>/dev/null || true
  pkill -f 'ds4 -m .*reactgo-prog' 2>/dev/null || true
  ssh "$M1" "pid=\$(cat '$WORKER_PID_FILE' 2>/dev/null); [ -n \"\$pid\" ] && kill \"\$pid\" 2>/dev/null; pkill -f 'ds4 -m .*reactgo-prog' 2>/dev/null; true" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# ---- 预检(只读, 内存安全): 两机无残留 + 分片层范围对得上 ----
log "预检: 清残留 ds4 + 释放端口 $PORT"
pkill -f 'ds4 -m .*reactgo-prog' 2>/dev/null || true
ssh "$M1" "pkill -f 'ds4 -m .*reactgo-prog' 2>/dev/null; lsof -nP -iTCP:$PORT -t 2>/dev/null | xargs -r kill -9 2>/dev/null; true" 2>/dev/null || true
sleep 1

# ---- 起 M1 worker (nohup, control listen 等 coordinator) ----
log "启动 M1 worker: -m $M1_MODEL --layers $WORKER_LAYERS (budget ${M1_BUDGET}MB, reverse-accept)"
ssh "$M1" "cd '$REMOTE_DIR' && rm -f '$WORKER_LOG'; \
  $COMMON_ENV DS4_MEM_BUDGET_MB=$M1_BUDGET \
  nohup ./ds4 -m '$M1_MODEL' --role worker --listen '$M1' '$PORT' \
  --layers '$WORKER_LAYERS' -c '$CTX' --temp 0 --nothink > '$WORKER_LOG' 2>&1 & echo \$! > '$WORKER_PID_FILE'; echo launched" 2>/dev/null

log "等 M1 worker 就绪 (control listen)…"
wok=0
for _ in $(seq 1 90); do
  if ssh "$M1" "grep -qE 'waiting for coordinator|control listen' '$WORKER_LOG'" 2>/dev/null; then wok=1; break; fi
  if ssh "$M1" "grep -qiE 'refusing to load|fatal|Address already|invalid|Insufficient Memory|out of range|not found' '$WORKER_LOG'" 2>/dev/null; then
    log "✗ M1 worker 启动失败, 日志尾:"; ssh "$M1" "tail -12 '$WORKER_LOG'"; exit 1
  fi
  rk=$(rss_kb_remote); [ -n "$rk" ] && [ "$rk" -gt "$MAX_KB" ] && { log "✗ M1 加载阶段 RSS $((rk/1024/1024))G 超 ${MAX_GB}G → 杀"; exit 2; }
  sleep 1
done
[ "$wok" = 1 ] || { log "✗ M1 worker 90s 未就绪:"; ssh "$M1" "tail -12 '$WORKER_LOG'"; exit 1; }
sleep 1

# ---- 起本机 coordinator (主动拨 M1) ----
log "启动 M4 coordinator: -m $M4_MODEL --layers $COORD_LAYERS temp=$TEMP n=$NPRED (budget ${M4_BUDGET}MB)"
rm -f "$COORD_LOG" "$COORD_OUT"
# env(非 eval): 直接子进程, $! 就是 ds4 PID, 看门狗 RSS 才读得到。
env DS4_DIST_REVERSE_CONNECT=1 DS4_METAL_EXPERT_OFFLOAD=0 DS4_METAL_NO_MODEL_WARMUP=1 $PC_ENV $EXTRA_ENV $FREQ_ENV $SPEC DS4_MEM_BUDGET_MB=$M4_BUDGET \
  ./ds4 -m "$M4_MODEL" --role coordinator --coordinator "$M1" "$PORT" \
  --layers "$COORD_LAYERS" -c "$CTX" -n "$NPRED" --temp "$TEMP" --seed "$SEED" --nothink \
  $DUMP_ARG -p "$PROMPT" > "$COORD_OUT" 2> "$COORD_LOG" &
COORD_PID=$!

# ---- 看门狗: 等 coordinator 一次性生成结束; 双机 RSS 红线 + 超时 ----
log "运行中… (超 ${RUN_TIMEOUT_SEC}s 或任一机 >${MAX_GB}G 即两边同杀)"
done_flag=0
for _ in $(seq 1 "$RUN_TIMEOUT_SEC"); do
  if ! kill -0 "$COORD_PID" 2>/dev/null; then done_flag=1; break; fi
  lk=$(rss_kb_local "$COORD_PID"); rk=$(rss_kb_remote)
  lg=$(awk "BEGIN{printf \"%.1f\",${lk:-0}/1048576}"); rg=$(awk "BEGIN{printf \"%.1f\",${rk:-0}/1048576}")
  printf "\r[reactgo-dual] M4=%sG/%dG  M1=%sG/%dG    " "$lg" "$MAX_GB" "$rg" "$MAX_GB"
  [ -n "$lk" ] && [ "$lk" -gt "$MAX_KB" ] && { echo; log "✗ M4 RSS ${lg}G 超限 → 两边同杀"; exit 2; }
  [ -n "$rk" ] && [ "$rk" -gt "$MAX_KB" ] && { echo; log "✗ M1 RSS ${rg}G 超限 → 两边同杀"; exit 2; }
  sleep 1
done
echo
[ "$done_flag" = 1 ] || { log "✗ 超时 ${RUN_TIMEOUT_SEC}s 未完成 → 杀"; exit 3; }
wait "$COORD_PID" 2>/dev/null || true

# ---- 结果 ----
echo "============================ 生成输出 (temp=$TEMP) ============================"
cat "$COORD_OUT"
echo
echo "============================ 速度 / 关键日志 ============================"
grep -iE "prefill|generation|t/s|tokens|reduced-expert|keep-map|clamp" "$COORD_LOG" | tail -8
