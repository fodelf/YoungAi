#!/usr/bin/env bash
# 双机张量并行(TP) q2 81GiB — Phase 1 (shared-FFN width-split) 验证脚本。
#
# 与 tp_k4_speed.sh 的区别: 跑 81GiB 全模型(不是能整进 RAM 的 k4)。两机各载*全栈*
# (TP 要求 BOTH 载全 layer stack), 故 backbone resident 每机 ~8.4GiB, 远大于 k4 半栈。
# 因此必须走 A3 expert-offload 流式(experts 不常驻, 按需从 SSD 拉), 内存配置直接复刻
# mtp_pipe_q2_speed.sh 已验证安全的那套。
#
# === 内存安全 (硬约束, 见 memory feedback_memory_safety_gate) ===
#   1. L1 resident gate: DS4_MEM_BUDGET_MB=12000 — planned-resident 超预算则*拒绝启动*
#      (= 跑前的内存安全证明: 启动通过 ⇒ 规划常驻 ≤12G)。
#   2. 看门狗: 本机/ M1 footprint(或 RSS) 超 12G ⇒ 两边同杀(只杀进程, 不删文件)。
#   3. expert-offload + NO_MODEL_WARMUP: 不预读/不 warmup 全模型, 避免 81GiB 缺页洪流。
#   单台 16G 全载 81GiB + 流式 = A3 单机 baseline 已证安全; TP 每机内存画像 = 单机全载。
#
# 网络: 本机=coordinator connect, M1=worker listen (DS4_TP_REVERSE_CONNECT=1,
#   绕 M1 出站 connect EHOSTUNREACH; TP all-reduce 对称求和, 方向不影响结果)。
#
# Phase 1 验证目标:
#   (a) 跨机 all-reduce 接线双机端到端能跑(从没验过);
#   (b) DS4_TP_SHARED_SPLIT=1 输出 sane, 与 =0 (skeleton 全量重组 baseline) 文本一致
#       (我的 column/row split vs bit-exact 重组, 应仅 ~1e-6 漂移);
#   (c) 两机均 ≤12G。
#   A/B: SHARED_SPLIT=1(默认, Phase 1 split) vs SHARED_SPLIT=0(skeleton)。
set -uo pipefail

# ---------------- 配置 (均可 env 覆盖) ----------------
REMOTE=${REMOTE:-192.168.1.2}
REMOTE_DIR=${REMOTE_DIR:-/Users/fodelf/ds4-main}
LOCAL_DIR=${LOCAL_DIR:-/Users/fodelf/git/ds4-main}
RENDEZVOUS_IP=${RENDEZVOUS_IP:-192.168.1.2}    # M1 雷电 IP: worker listen, coordinator connect
PORT=${PORT:-5599}
MODEL=${MODEL:-ds4flash.gguf}                  # 81GiB q2 symlink (两机都要有)
CTX=${CTX:-1024}
NPRED=${NPRED:-200}
# 短问答默认 (copy-spec 不触发); 回显/编辑型负载想触发 copy-spec verify 批, 自定义 PROMPT。
PROMPT=${PROMPT:-Explain what a hash table is in one paragraph.}
TP_LAYERS=${TP_LAYERS:-2}                       # Phase 1: 仅前 2 层做 split + all-reduce
SHARED_SPLIT=${SHARED_SPLIT:-1}                 # 1=Phase 1 split; 0=skeleton 全量重组 baseline

# 内存硬预算 (与 mtp_pipe 一致, 物理红线 12/12)
LOCAL_MAX_GB=${LOCAL_MAX_GB:-12}
REMOTE_MAX_GB=${REMOTE_MAX_GB:-12}
LOCAL_BUDGET_MB=${LOCAL_BUDGET_MB:-12000}       # L1 gate planned-resident 硬预算
REMOTE_BUDGET_MB=${REMOTE_BUDGET_MB:-12000}
GATHER_THREADS=${GATHER_THREADS:-4}
# 默认 RSS 监控(开销小); 诊断真实 mmap footprint 设 MEM_WATCH_MODE=footprint(慢)。
MEM_WATCH_MODE=${MEM_WATCH_MODE:-rss}

# TP 全模型运行 env: A3 expert 流式(81GiB 不能常驻) + 不 warmup + reverse-connect + Phase1 开关。
# 不含 DS4_DIST_*(那是 layer-pipeline, TP 不用)。backbone mlock 故意不开(全栈 backbone
# ~8.4GiB, pin 上去会顶预算)。
AR_LOG=${AR_LOG:-1}                             # wave-72 debug: 逐步诊断/计时 decode TP all-reduce
EXPERT_SPLIT=${EXPERT_SPLIT:-0}                  # Phase 3: 每机只 gather 半数 routed 专家 (halve IO); 默认 OFF=skeleton
EXPERT_IO_PROFILE=${EXPERT_IO_PROFILE:-0}        # 开 = 每 routed-MoE 调用打 ds4-io 分解 (看 gather 是否halve)
# IO_OPT=1: 叠加 mtp_pipe 已验证的 metal 层 IO 优化 (与 TP 正交)。默认 0 保 A/B 干净。
# copy-spec 已是引擎天然默认 (TP leader verify 批; miss 零成本自动退化 bare round), 无开关。
IO_OPT=${IO_OPT:-0}
IO_OPT_ENV=""
if [ "$IO_OPT" = 1 ]; then
  IO_OPT_ENV="DS4_METAL_MOE_OVERLAP=1 DS4_METAL_MOE_OVERLAP_PASSES=2 DS4_METAL_EXPERT_SORT_IDS=1 DS4_METAL_EXPERT_FULL_LAYER_STREAM=1 DS4_METAL_EXPERT_PREFETCH_AHEAD=1 DS4_METAL_EXPERT_PREFETCH_TOP=8 DS4_METAL_EXPERT_PREFETCH_DEPTH=1 DS4_METAL_EXPERT_EVENT_DRAIN=1"
fi
TP_RUN_ENV=${TP_RUN_ENV:-"DS4_TP_REVERSE_CONNECT=1 DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_EXPERT_OFFLOAD_DIRECT=0 DS4_METAL_EXPERT_PREAD=1 DS4_METAL_EXPERT_GATHER_THREADS=$GATHER_THREADS DS4_METAL_NO_MODEL_WARMUP=1 DS4_TP_SHARED_SPLIT=$SHARED_SPLIT DS4_TP_EXPERT_SPLIT=$EXPERT_SPLIT DS4_METAL_EXPERT_IO_PROFILE=$EXPERT_IO_PROFILE DS4_TP_AR_LOG=$AR_LOG $IO_OPT_ENV"}

LEADER_LOG=/tmp/tp_q2_leader.log
FOLLOWER_LOG=/tmp/tp_q2_follower.log
WORKER_PID_FILE=/tmp/tp_q2_worker.pid
LEADER_PID=""

log(){ echo "[tp-q2] $*"; }

# ---------------- 清理: 两边同杀 (幂等, 只杀进程不删文件) ----------------
cleanup(){
  trap - INT TERM EXIT
  echo
  log "cleanup: 杀两边 ds4 进程 (只杀进程, 不删任何文件)"
  [ -n "$LEADER_PID" ] && kill "$LEADER_PID" 2>/dev/null || true
  pkill -f 'ds4 -m' 2>/dev/null || true
  ssh "$REMOTE" "pid=\$(cat '$WORKER_PID_FILE' 2>/dev/null); [ -n \"\$pid\" ] && kill \"\$pid\" 2>/dev/null || true; pkill -f 'ds4 -m' 2>/dev/null || true" 2>/dev/null || true
  log "done."
}
trap cleanup INT TERM EXIT

# ---------------- 内存监控 (GiB), footprint 或 RSS ----------------
rss_gb_local(){ local kb; kb=$(ps -o rss= -p "$1" 2>/dev/null | tr -d ' '); [ -n "$kb" ] && awk "BEGIN{printf \"%.2f\",$kb/1048576}" || echo 0; }
rss_gb_remote(){
  local kb; kb=$(ssh "$REMOTE" "pid=\$(cat '$WORKER_PID_FILE' 2>/dev/null); [ -z \"\$pid\" ] && pid=\$(pgrep -f 'ds4 -m' | head -1); [ -n \"\$pid\" ] && ps -o rss= -p \$pid 2>/dev/null | tr -d ' '" 2>/dev/null)
  [ -n "$kb" ] && awk "BEGIN{printf \"%.2f\",$kb/1048576}" || echo 0
}
footprint_gb_local(){
  local bytes; bytes=$(footprint -pid "$1" -f bytes --noCategories 2>/dev/null | awk '/phys_footprint:/ {print $2; exit}')
  [ -n "$bytes" ] && awk "BEGIN{printf \"%.2f\",$bytes/1073741824}" || rss_gb_local "$1"
}
footprint_gb_remote(){
  local bytes; bytes=$(ssh "$REMOTE" "pid=\$(cat '$WORKER_PID_FILE' 2>/dev/null); [ -z \"\$pid\" ] && pid=\$(pgrep -f 'ds4 -m' | head -1); [ -n \"\$pid\" ] && footprint -pid \$pid -f bytes --noCategories 2>/dev/null | awk '/phys_footprint:/ {print \$2; exit}'" 2>/dev/null)
  [ -n "$bytes" ] && awk "BEGIN{printf \"%.2f\",$bytes/1073741824}" || rss_gb_remote
}
mem_gb_local(){ [ "$MEM_WATCH_MODE" = footprint ] && footprint_gb_local "$1" || rss_gb_local "$1"; }
mem_gb_remote(){ [ "$MEM_WATCH_MODE" = footprint ] && footprint_gb_remote || rss_gb_remote; }
over(){ awk "BEGIN{a=$1+0;b=$2+0;exit !(a>b)}"; }

# ---------------- 0. 前置检查 (两机都要有 81GiB 模型) ----------------
[ -f "$LOCAL_DIR/$MODEL" ] || { log "本机缺模型 $MODEL"; exit 1; }
ssh "$REMOTE" "[ -f '$REMOTE_DIR/$MODEL' ]" 2>/dev/null || { log "M1 缺模型 $MODEL"; exit 1; }

# ---------------- 1. 同步代码 → M1 ----------------
log "同步源码 → $REMOTE:$REMOTE_DIR (共享 CORE_OBJS, 两机必须都重编)"
rsync -a --exclude '.git' --exclude '*.o' --exclude '*.gguf' --exclude 'gguf/' \
  --exclude '*.bin' --exclude 'ds4' --exclude 'ds4-server' --exclude 'ds4-bench' \
  --exclude 'ds4-eval' --exclude 'ds4-agent' --exclude 'e0-pingpong' --exclude 'ds4_test' \
  "$LOCAL_DIR"/ "$REMOTE:$REMOTE_DIR"/ || { log "rsync 失败"; exit 1; }

# ---------------- 2. 两边 build (默认增量, 只重编改动的文件 → 快; CLEAN=1 强制全量) ----------------
CLEAN=${CLEAN:-0}
CLEAN_CMD=""; [ "$CLEAN" = 1 ] && CLEAN_CMD="make clean >/dev/null 2>&1 &&"
log "本机: ${CLEAN:+clean+}make ds4 (增量)"
( cd "$LOCAL_DIR" && eval "$CLEAN_CMD make ds4 >/tmp/tp_q2_build_local.log 2>&1" ) \
  || { log "本机编译失败:"; tail -8 /tmp/tp_q2_build_local.log; exit 1; }
log "M1:   ${CLEAN:+clean+}make ds4 (增量)"
ssh "$REMOTE" "cd '$REMOTE_DIR' && $CLEAN_CMD make ds4 >/tmp/tp_q2_build_remote.log 2>&1" \
  || { log "M1 编译失败:"; ssh "$REMOTE" "tail -8 /tmp/tp_q2_build_remote.log"; exit 1; }

# ---------------- 3. 清旧进程 + 端口 ----------------
log "清理两边旧 ds4 进程与 $PORT 端口占用"
pkill -f 'ds4 -m' 2>/dev/null || true
ssh "$REMOTE" "pkill -f 'ds4 -m' 2>/dev/null; lsof -nP -iTCP:$PORT -t 2>/dev/null | xargs -r kill -9 2>/dev/null; true" 2>/dev/null || true
sleep 1

# ---------------- 4. 先起 M1 worker (listen) — 带 L1 gate + 加载期看门狗 ----------------
log "启动 M1 worker: --role worker --listen $RENDEZVOUS_IP:$PORT --tp --tp-layers $TP_LAYERS (SHARED_SPLIT=$SHARED_SPLIT)"
ssh "$REMOTE" "cd '$REMOTE_DIR' && rm -f '$FOLLOWER_LOG'; \
  $TP_RUN_ENV DS4_MEM_BUDGET_MB=$REMOTE_BUDGET_MB \
  nohup ./ds4 -m '$MODEL' --role worker --listen '$RENDEZVOUS_IP' '$PORT' \
  --tp --tp-layers '$TP_LAYERS' -c '$CTX' -n '$NPRED' --temp 0 --nothink \
  -p '$PROMPT' > '$FOLLOWER_LOG' 2>&1 & echo \$! > '$WORKER_PID_FILE'; echo launched" 2>/dev/null

log "等 M1 worker 加载完并 listen (加载期看门狗 ${REMOTE_MAX_GB}G)…"
wok=0
for _ in $(seq 1 120); do
  if ssh "$REMOTE" "grep -qE 'backend initialized|waiting for coordinator|control listen|tp .*listen' '$FOLLOWER_LOG'" 2>/dev/null; then wok=1; break; fi
  # 失败模式必须精确: L1 gate 成功行含 "within budget", 拒绝行才是 "refusing to load"
  # (裸 'budget' 会误杀成功行 — wave-72 首跑教训)。
  if ssh "$REMOTE" "grep -qiE 'refusing to load|fatal error|Address already|Insufficient Memory|TP peer connection failed|exceeds .*budget' '$FOLLOWER_LOG'" 2>/dev/null; then
    log "M1 worker 启动失败/被 L1 gate 拒 (日志尾):"; ssh "$REMOTE" "tail -10 '$FOLLOWER_LOG'"; cleanup; exit 1
  fi
  rg=$(mem_gb_remote); over "$rg" "$REMOTE_MAX_GB" && { log "M1 加载期 ${rg}G 超 ${REMOTE_MAX_GB}G → 两边同杀"; cleanup; exit 2; }
  sleep 1
done
[ "$wok" = 1 ] || { log "M1 worker 120s 未就绪 → 放弃"; ssh "$REMOTE" "tail -10 '$FOLLOWER_LOG'"; cleanup; exit 1; }
sleep 1

# ---------------- 5. 起本机 coordinator (connect M1) ----------------
cd "$LOCAL_DIR"
log "启动本机 coordinator: --role coordinator --coordinator $RENDEZVOUS_IP:$PORT --tp --tp-layers $TP_LAYERS"
rm -f "$LEADER_LOG"
env $TP_RUN_ENV DS4_MEM_BUDGET_MB=$LOCAL_BUDGET_MB \
  ./ds4 -m "$MODEL" --role coordinator --coordinator "$RENDEZVOUS_IP" "$PORT" \
  --tp --tp-layers "$TP_LAYERS" -c "$CTX" -n "$NPRED" --temp 0 --nothink \
  -p "$PROMPT" > "$LEADER_LOG" 2>&1 &
LEADER_PID=$!

# ---------------- 6. 看门狗主循环 (内存 + 超时) ----------------
RUN_TIMEOUT_SEC=${RUN_TIMEOUT_SEC:-600}   # 超时两边同杀, 防 decode 挂死无限跑
WD_T0=$(date +%s)
log "运行中… (Ctrl+C / 本机>${LOCAL_MAX_GB}G / M1>${REMOTE_MAX_GB}G / >${RUN_TIMEOUT_SEC}s 均两边同杀)"
while kill -0 "$LEADER_PID" 2>/dev/null; do
  lg=$(mem_gb_local "$LEADER_PID"); rg=$(mem_gb_remote)
  el=$(( $(date +%s) - WD_T0 ))
  printf "\r[tp-q2] mem 本机=%sG/%dG  M1=%sG/%dG  %ds/%ds    " "$lg" "$LOCAL_MAX_GB" "$rg" "$REMOTE_MAX_GB" "$el" "$RUN_TIMEOUT_SEC"
  over "$lg" "$LOCAL_MAX_GB" && { echo; log "本机 ${lg}G 超限 → 两边同杀"; cleanup; exit 2; }
  over "$rg" "$REMOTE_MAX_GB" && { echo; log "M1 ${rg}G 超限 → 两边同杀"; cleanup; exit 2; }
  [ "$el" -gt "$RUN_TIMEOUT_SEC" ] && { echo; log "超时 ${el}s > ${RUN_TIMEOUT_SEC}s → 两边同杀"; cleanup; exit 3; }
  sleep 1
done
echo

# ---------------- 7. 结果 ----------------
log "coordinator 结束。生成文本:"
grep -v -E '^ds4:|^ds4_profile' "$LEADER_LOG" | tail -6
echo "----------------------------------------"
if grep -q "tok/s" "$LEADER_LOG"; then
  log "速度:"; grep "tok/s" "$LEADER_LOG"
else
  log "未拿到计时行, leader 日志尾:"; tail -12 "$LEADER_LOG"
fi
cleanup
