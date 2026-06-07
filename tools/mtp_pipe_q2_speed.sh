#!/usr/bin/env bash
# 双机 *层切分(layer-pipeline) + MTP 跨机投机* q2 smoke/测速脚本 (mtp.md Phase 1 方案A)。
#
# 拓扑 (本机 M4 扛大部分层, M1 扛 MTP + 少部分末段层):
#   本机 M4 = coordinator: 持前段 *大部分* 层 (0:N) + token_embd, 做 tokenize/sample/编排,
#             跑 -p 一次性生成 (生成文本+计时在本机, 直接打印)。
#   M1      = worker: 持末段 *少部分* 层 (N:output) + output head + MTP drafter (--mtp 草稿模型)。
#             MTP 必须跟 output head 同机 (mtp_for_worker_draft 硬约束), 故 MTP 落 M1=worker。
#
# *** reverse-connect (DS4_DIST_REVERSE_CONNECT=1) ***
#   默认 layer-pipeline 是 worker 主动拨 coordinator。本拓扑反过来:
#   M1 worker 只 listen/accept (零出站), M4 coordinator 主动拨 M1 + 主动拨 M1 的数据通道。
#   原因: macOS 本地网络隐私拒绝 M1 上 ds4 走雷电桥 bridge0 出站 (connect EHOSTUNREACH, nc 却通);
#   ssh 无头启动弹不出授权框。reverse 后 M1 零出站, 不需要该权限; M4 (可交互弹框授权) 全部出站。
#   ⟹ 一劳永逸绕开权限问题, 且正好是用户要的 M4 大层 / M1 MTP 布局。
#   注: 首次在 M4 上跑会弹"本地网络"授权框, 点允许即可 (M4 是你正在用的机器)。
#
# 数据流: M4(embed+层0:N) → hidden → M1(层N:output + output head + MTP draft) → logits → 回 M4 采样。
#
# 启动顺序 (reverse): 先起 M1 worker (listen 等待) → 再起本机 coordinator (主动拨 M1)。
#
# 安全闸 (任一触发"两边同杀, 只杀进程不删文件"): Ctrl+C / 本机 coordinator RSS 超 LOCAL_MAX_GB /
#   M1 worker RSS 超 REMOTE_MAX_GB。DS4_MEM_BUDGET_MB 和 L1 resident-budget gate 负责模型常驻硬预算；
#   可设 MEM_WATCH_MODE=footprint 做慢速诊断，但 footprint 每秒扫描 VM 会明显拖慢测速。
# !! 本机 M4 现扛大部分层 (~8G), 若 M4 被其他程序占满会 GPU OOM —— 先腾内存再跑。!!
set -uo pipefail

# ---------------- 配置 (均可 env 覆盖) ----------------
REMOTE=${REMOTE:-192.168.1.2}                  # M1 (worker), ssh 目标
REMOTE_DIR=${REMOTE_DIR:-/Users/fodelf/ds4-main}
LOCAL_DIR=${LOCAL_DIR:-/Users/fodelf/git/ds4-main}
WORKER_IP=${WORKER_IP:-192.168.1.2}            # M1 雷电 IP: worker 在此 listen 控制端口, M4 coordinator 拨此
PORT=${PORT:-5599}
# 档位: 默认 q2-imatrix 完整模型；脚本只做双机层切分 + A3 按需专家加载 smoke。
# 分层等待诊断默认：本机 0:22，M1 23:output。这样故意把更多后段层放到 M1，
# 用 --debug telemetry 观察是否 coordinator 在等 worker；若 M1 8GB 预算拒绝或变慢，再回 0:33/34:output。
MODEL=${MODEL:-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}
MTP_GGUF=${MTP_GGUF:-gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf} # 草稿模型, 仅 M1 worker 加载
# 层切分 (block_count=43, layers 0..42)。本机 M4 扛大部分前段, M1 扛少部分末段 + output + MTP。
SPLIT_COORD=${SPLIT_COORD:-0:22}               # 本机 M4 coordinator 层切片 (诊断: 前 23 层)
SPLIT_WORKER=${SPLIT_WORKER:-23:output}        # M1 worker 层切片 (诊断: 后 20 层 + output)
CTX=${CTX:-200000}
NPRED=${NPRED:-128}
DRAFT=${DRAFT:-2}                              # --mtp-draft N: 分布式 MTP 默认 2；4 在当前 M1 worker 上实测负收益
NO_MTP=${NO_MTP:-1}                            # 默认关 MTP：当前 M1 drafter 尽管接受率高但端到端负收益；设 NO_MTP=0 做实验
if [ "$NO_MTP" = 1 ]; then
  COORD_MTP_ARGS=""
  WORKER_MTP_ARGS=""
else
  COORD_MTP_ARGS="--mtp-role worker --mtp-draft $DRAFT"
  WORKER_MTP_ARGS="--mtp $MTP_GGUF --mtp-role worker --mtp-draft $DRAFT"
fi
PROMPT=${PROMPT:-"写一个 Python 函数判断字符串是否回文，并解释它的原理。"}
SEED=${SEED:-1}
LOCAL_MAX_GB=${LOCAL_MAX_GB:-12}
REMOTE_MAX_GB=${REMOTE_MAX_GB:-8}
LOCAL_BUDGET_MB=${LOCAL_BUDGET_MB:-12000}
REMOTE_BUDGET_MB=${REMOTE_BUDGET_MB:-8000}
# Metal prefill scratch chunk。默认 4096 会让 M1 worker 单个命令缓冲的 wired working set
# (模型常驻 ~6.88G + scratch 池) 越过 M1 Pro GPU 的 ~10.67G 工作集天花板 → kIOGPU OOM
# (= 曾经"本机爆了"的真因; 实测 currentAllocated 11.23G > recommendedMax 10.67G)。prompt 短时
# 4096 的 scratch 几乎全浪费; 512 缩小 scratch 池, worker 不再 OOM (实测 coordinator prefill 21.5 t/s)。
# 要更大上下文/吞吐再上调并实测。
PREFILL_CHUNK=${PREFILL_CHUNK:-512}
# 源专家 LRU（最终有效方案）：不再让 GPU 从大 expert pool 随机读。
# LRU 只保活/复制专家源字节；每 token 仍走 compact A3 scratch，保持连续 GPU 读。
# HARD_COPY=1 时 cache 真正占用匿名内存保存专家副本，命中后从热 DRAM 副本拷到 scratch，避免 mmap 冷页/页缓存抖动。
LOCAL_EXPERT_SOURCE_CACHE_MB=${LOCAL_EXPERT_SOURCE_CACHE_MB:-3072}
REMOTE_EXPERT_SOURCE_CACHE_MB=${REMOTE_EXPERT_SOURCE_CACHE_MB:-512}
LOCAL_EXPERT_SOURCE_CACHE_LAYER_START=${LOCAL_EXPERT_SOURCE_CACHE_LAYER_START:-3}
LOCAL_EXPERT_SOURCE_CACHE_LAYER_END=${LOCAL_EXPERT_SOURCE_CACHE_LAYER_END:-22}
REMOTE_EXPERT_SOURCE_CACHE_LAYER_START=${REMOTE_EXPERT_SOURCE_CACHE_LAYER_START:-23}
REMOTE_EXPERT_SOURCE_CACHE_LAYER_END=${REMOTE_EXPERT_SOURCE_CACHE_LAYER_END:-42}
EXPERT_SOURCE_CACHE_ADMIT_AFTER=${EXPERT_SOURCE_CACHE_ADMIT_AFTER:-2}
EXPERT_SOURCE_CACHE_MLOCK=${EXPERT_SOURCE_CACHE_MLOCK:-0}
EXPERT_SOURCE_CACHE_HARD_COPY=${EXPERT_SOURCE_CACHE_HARD_COPY:-0}
# 旧 GPU expert pool 保留为实验开关，但默认关闭：日志已证明 Shared pool 随机读 + decode 同步填池比 compact scratch 慢。
LOCAL_EXPERT_POOL_MB=${LOCAL_EXPERT_POOL_MB:-0}
REMOTE_EXPERT_POOL_MB=${REMOTE_EXPERT_POOL_MB:-0}
LOCAL_EXPERT_POOL_LAYER_START=${LOCAL_EXPERT_POOL_LAYER_START:-16}
LOCAL_EXPERT_POOL_LAYER_END=${LOCAL_EXPERT_POOL_LAYER_END:-22}
REMOTE_EXPERT_POOL_LAYER_START=${REMOTE_EXPERT_POOL_LAYER_START:-23}
REMOTE_EXPERT_POOL_LAYER_END=${REMOTE_EXPERT_POOL_LAYER_END:-42}
EXPERT_POOL_PREFETCH_LOOKAHEAD=${EXPERT_POOL_PREFETCH_LOOKAHEAD:-0}
EXPERT_POOL_PREFETCH_TOP=${EXPERT_POOL_PREFETCH_TOP:-0}
EXPERT_POOL_PREFETCH_SELF=${EXPERT_POOL_PREFETCH_SELF:-0}
EXPERT_POOL_PREFETCH_ADJACENT=${EXPERT_POOL_PREFETCH_ADJACENT:-0}
EXPERT_POOL_WAIT_INFLIGHT=${EXPERT_POOL_WAIT_INFLIGHT:-0}
EXPERT_POOL_FOREGROUND_FILL=${EXPERT_POOL_FOREGROUND_FILL:-1}
EXPERT_POOL_WARM_BATCH=${EXPERT_POOL_WARM_BATCH:-1}
EXPERT_POOL_PREFETCH_EVICT=${EXPERT_POOL_PREFETCH_EVICT:-0}
EXPERT_POOL_ADMIT_AFTER=${EXPERT_POOL_ADMIT_AFTER:-1}
EXPERT_POOL_HOTLOCK_TOP=${EXPERT_POOL_HOTLOCK_TOP:-0}
EXPERT_POOL_PREFETCH_QUEUE=${EXPERT_POOL_PREFETCH_QUEUE:-0}
EXPERT_POOL_MIN_LAYER_SLOTS=${EXPERT_POOL_MIN_LAYER_SLOTS:-12}
# Expert offload 命中率模拟日志：不改变推理，只在 A3 offload 每次 CPU-gather 后统计
# 「如果有一个 LRU 专家池」的命中率/省下的拷贝量，并打印每层单专家 slot 内存。
# coordinator 跑的层多，1GiB 连 34 层单 token 活跃集都放不下；默认给本机模拟池 4GiB。
# 这是 profiler 的模拟 cache，不实际分配这些 expert 常驻内存；真实 expert pool 后续再按预算实现。
# EXPERT_PROFILE_ALL=1 会在进程正常退出时打印所有出现过的专家；默认只打印每层 top-N。
EXPERT_PROFILE=${EXPERT_PROFILE:-1}
LOCAL_EXPERT_PROFILE_CACHE_MB=${LOCAL_EXPERT_PROFILE_CACHE_MB:-4096}
REMOTE_EXPERT_PROFILE_CACHE_MB=${REMOTE_EXPERT_PROFILE_CACHE_MB:-1024}
# 兼容旧变量：如果只设置 EXPERT_PROFILE_CACHE_MB，则两边都用这个值。
if [ -n "${EXPERT_PROFILE_CACHE_MB:-}" ]; then
  LOCAL_EXPERT_PROFILE_CACHE_MB=$EXPERT_PROFILE_CACHE_MB
  REMOTE_EXPERT_PROFILE_CACHE_MB=$EXPERT_PROFILE_CACHE_MB
fi
EXPERT_PROFILE_TOP=${EXPERT_PROFILE_TOP:-8}
EXPERT_PROFILE_INTERVAL=${EXPERT_PROFILE_INTERVAL:-64}
EXPERT_PROFILE_ALL=${EXPERT_PROFILE_ALL:-0}
LOCAL_PROFILE_ENV=""
REMOTE_PROFILE_ENV=""
if [ "$EXPERT_PROFILE" = 1 ]; then
  LOCAL_PROFILE_ENV="DS4_METAL_EXPERT_OFFLOAD_PROFILE=1 DS4_METAL_EXPERT_PROFILE_CACHE_MB=$LOCAL_EXPERT_PROFILE_CACHE_MB DS4_METAL_EXPERT_PROFILE_TOP=$EXPERT_PROFILE_TOP DS4_METAL_EXPERT_PROFILE_INTERVAL=$EXPERT_PROFILE_INTERVAL"
  REMOTE_PROFILE_ENV="DS4_METAL_EXPERT_OFFLOAD_PROFILE=1 DS4_METAL_EXPERT_PROFILE_CACHE_MB=$REMOTE_EXPERT_PROFILE_CACHE_MB DS4_METAL_EXPERT_PROFILE_TOP=$EXPERT_PROFILE_TOP DS4_METAL_EXPERT_PROFILE_INTERVAL=$EXPERT_PROFILE_INTERVAL"
  [ "$EXPERT_PROFILE_ALL" = 1 ] && LOCAL_PROFILE_ENV="$LOCAL_PROFILE_ENV DS4_METAL_EXPERT_PROFILE_ALL=1"
  [ "$EXPERT_PROFILE_ALL" = 1 ] && REMOTE_PROFILE_ENV="$REMOTE_PROFILE_ENV DS4_METAL_EXPERT_PROFILE_ALL=1"
fi
[ "$LOCAL_EXPERT_POOL_MB" != 0 ] && LOCAL_PROFILE_ENV="$LOCAL_PROFILE_ENV DS4_METAL_EXPERT_POOL_MB=$LOCAL_EXPERT_POOL_MB DS4_METAL_EXPERT_POOL_INTERVAL=$EXPERT_PROFILE_INTERVAL DS4_METAL_EXPERT_POOL_LAYER_START=$LOCAL_EXPERT_POOL_LAYER_START DS4_METAL_EXPERT_POOL_LAYER_END=$LOCAL_EXPERT_POOL_LAYER_END DS4_METAL_EXPERT_POOL_MIN_LAYER_SLOTS=$EXPERT_POOL_MIN_LAYER_SLOTS DS4_METAL_EXPERT_POOL_PREFETCH_LOOKAHEAD=$EXPERT_POOL_PREFETCH_LOOKAHEAD DS4_METAL_EXPERT_POOL_PREFETCH_TOP=$EXPERT_POOL_PREFETCH_TOP DS4_METAL_EXPERT_POOL_PREFETCH_SELF=$EXPERT_POOL_PREFETCH_SELF DS4_METAL_EXPERT_POOL_PREFETCH_ADJACENT=$EXPERT_POOL_PREFETCH_ADJACENT DS4_METAL_EXPERT_POOL_WAIT_INFLIGHT=$EXPERT_POOL_WAIT_INFLIGHT DS4_METAL_EXPERT_POOL_FOREGROUND_FILL=$EXPERT_POOL_FOREGROUND_FILL DS4_METAL_EXPERT_POOL_WARM_BATCH=$EXPERT_POOL_WARM_BATCH DS4_METAL_EXPERT_POOL_PREFETCH_EVICT=$EXPERT_POOL_PREFETCH_EVICT DS4_METAL_EXPERT_POOL_ADMIT_AFTER=$EXPERT_POOL_ADMIT_AFTER DS4_METAL_EXPERT_POOL_HOTLOCK_TOP=$EXPERT_POOL_HOTLOCK_TOP DS4_METAL_EXPERT_POOL_PREFETCH_QUEUE=$EXPERT_POOL_PREFETCH_QUEUE"
[ "$REMOTE_EXPERT_POOL_MB" != 0 ] && REMOTE_PROFILE_ENV="$REMOTE_PROFILE_ENV DS4_METAL_EXPERT_POOL_MB=$REMOTE_EXPERT_POOL_MB DS4_METAL_EXPERT_POOL_INTERVAL=$EXPERT_PROFILE_INTERVAL DS4_METAL_EXPERT_POOL_LAYER_START=$REMOTE_EXPERT_POOL_LAYER_START DS4_METAL_EXPERT_POOL_LAYER_END=$REMOTE_EXPERT_POOL_LAYER_END DS4_METAL_EXPERT_POOL_MIN_LAYER_SLOTS=$EXPERT_POOL_MIN_LAYER_SLOTS DS4_METAL_EXPERT_POOL_PREFETCH_LOOKAHEAD=$EXPERT_POOL_PREFETCH_LOOKAHEAD DS4_METAL_EXPERT_POOL_PREFETCH_TOP=$EXPERT_POOL_PREFETCH_TOP DS4_METAL_EXPERT_POOL_PREFETCH_SELF=$EXPERT_POOL_PREFETCH_SELF DS4_METAL_EXPERT_POOL_PREFETCH_ADJACENT=$EXPERT_POOL_PREFETCH_ADJACENT DS4_METAL_EXPERT_POOL_WAIT_INFLIGHT=$EXPERT_POOL_WAIT_INFLIGHT DS4_METAL_EXPERT_POOL_FOREGROUND_FILL=$EXPERT_POOL_FOREGROUND_FILL DS4_METAL_EXPERT_POOL_WARM_BATCH=$EXPERT_POOL_WARM_BATCH DS4_METAL_EXPERT_POOL_PREFETCH_EVICT=$EXPERT_POOL_PREFETCH_EVICT DS4_METAL_EXPERT_POOL_ADMIT_AFTER=$EXPERT_POOL_ADMIT_AFTER DS4_METAL_EXPERT_POOL_HOTLOCK_TOP=$EXPERT_POOL_HOTLOCK_TOP DS4_METAL_EXPERT_POOL_PREFETCH_QUEUE=$EXPERT_POOL_PREFETCH_QUEUE"
[ "$LOCAL_EXPERT_SOURCE_CACHE_MB" != 0 ] && LOCAL_PROFILE_ENV="$LOCAL_PROFILE_ENV DS4_METAL_EXPERT_SOURCE_CACHE_MB=$LOCAL_EXPERT_SOURCE_CACHE_MB DS4_METAL_EXPERT_SOURCE_CACHE_LAYER_START=$LOCAL_EXPERT_SOURCE_CACHE_LAYER_START DS4_METAL_EXPERT_SOURCE_CACHE_LAYER_END=$LOCAL_EXPERT_SOURCE_CACHE_LAYER_END DS4_METAL_EXPERT_SOURCE_CACHE_ADMIT_AFTER=$EXPERT_SOURCE_CACHE_ADMIT_AFTER DS4_METAL_EXPERT_SOURCE_CACHE_INTERVAL=$EXPERT_PROFILE_INTERVAL DS4_METAL_EXPERT_SOURCE_CACHE_MLOCK=$EXPERT_SOURCE_CACHE_MLOCK DS4_METAL_EXPERT_SOURCE_CACHE_HARD_COPY=$EXPERT_SOURCE_CACHE_HARD_COPY"
[ "$REMOTE_EXPERT_SOURCE_CACHE_MB" != 0 ] && REMOTE_PROFILE_ENV="$REMOTE_PROFILE_ENV DS4_METAL_EXPERT_SOURCE_CACHE_MB=$REMOTE_EXPERT_SOURCE_CACHE_MB DS4_METAL_EXPERT_SOURCE_CACHE_LAYER_START=$REMOTE_EXPERT_SOURCE_CACHE_LAYER_START DS4_METAL_EXPERT_SOURCE_CACHE_LAYER_END=$REMOTE_EXPERT_SOURCE_CACHE_LAYER_END DS4_METAL_EXPERT_SOURCE_CACHE_ADMIT_AFTER=$EXPERT_SOURCE_CACHE_ADMIT_AFTER DS4_METAL_EXPERT_SOURCE_CACHE_INTERVAL=$EXPERT_PROFILE_INTERVAL DS4_METAL_EXPERT_SOURCE_CACHE_MLOCK=$EXPERT_SOURCE_CACHE_MLOCK DS4_METAL_EXPERT_SOURCE_CACHE_HARD_COPY=$EXPERT_SOURCE_CACHE_HARD_COPY"
DIST_DEBUG=${DIST_DEBUG:-1}
DEBUG_ARGS=""
[ "$DIST_DEBUG" = 1 ] && DEBUG_ARGS="--debug"
# reverse-connect 是本拓扑的核心 (见顶部注释)。两机都要带。
# DS4_METAL_PREFILL_CHUNK 限制 prefill scratch, 防 M1 worker GPU 命令缓冲 OOM (见上)。
# DS4_METAL_EXPERT_OFFLOAD 让 q2 routed experts 走 A3 按需 scratch; NO_MODEL_WARMUP 避免启动时扫冷 expert views。
BASE_RUN_ENV=${BASE_RUN_ENV:-"DS4_DIST_REVERSE_CONNECT=1 DS4_METAL_PREFILL_CHUNK=$PREFILL_CHUNK DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_NO_MODEL_WARMUP=1"}
# RUN_ENV 仍可一把覆盖两边；LOCAL_RUN_ENV/REMOTE_RUN_ENV 可分别覆盖。
LOCAL_RUN_ENV=${LOCAL_RUN_ENV:-${RUN_ENV:-"$BASE_RUN_ENV $LOCAL_PROFILE_ENV"}}
REMOTE_RUN_ENV=${REMOTE_RUN_ENV:-${RUN_ENV:-"$BASE_RUN_ENV $REMOTE_PROFILE_ENV"}}

COORD_LOG=/tmp/mtp_pipe_coord.log              # 本机 M4 路径 (coordinator stderr: ds4 日志/速度/profile)
COORD_OUT=/tmp/mtp_pipe_coord.out              # 本机 M4 路径 (coordinator stdout: 纯生成文本)
WORKER_LOG=/tmp/mtp_pipe_worker.log            # M1 上的路径 (worker)
WORKER_PID_FILE=/tmp/mtp_pipe_worker.pid        # M1 上本脚本启动的 worker pid
COORD_PID=""

log(){ echo "[mtp-pipe] $*"; }

# ---------------- 清理: 两边同杀 (幂等, 只杀进程) ----------------
cleanup(){
  trap - INT TERM EXIT
  echo
  log "cleanup: 杀两边 ds4 进程 (只杀进程, 不删任何文件)"
  [ -n "$COORD_PID" ] && kill "$COORD_PID" 2>/dev/null || true
  ssh "$REMOTE" "pid=\$(cat '$WORKER_PID_FILE' 2>/dev/null); [ -n \"\$pid\" ] && kill \"\$pid\" 2>/dev/null || true; pkill -f 'ds4 -m' 2>/dev/null || true" 2>/dev/null || true
  pkill -f 'ds4 -m' 2>/dev/null || true
  log "done."
}
trap cleanup INT TERM EXIT

# ---------------- 内存看门狗 (GiB) ----------------
# 默认用 ps RSS，开销小、适合测速；DS4_MEM_BUDGET_MB/L1 gate 仍是模型常驻硬预算。
# 诊断 mmap/Metal 真实 footprint 时设置 MEM_WATCH_MODE=footprint，但 footprint 很慢，会拖低 t/s。
MEM_WATCH_MODE=${MEM_WATCH_MODE:-rss}
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
mem_label(){ [ "$MEM_WATCH_MODE" = footprint ] && echo footprint || echo RSS; }
over(){ awk "BEGIN{a=$1+0;b=$2+0;exit !(a>b)}"; }

# ---------------- 0. 前置检查 ----------------
[ -f "$LOCAL_DIR/$MODEL" ] || { log "本机缺模型 $MODEL"; exit 1; }
[ "$NO_MTP" = 1 ] || ssh "$REMOTE" "[ -f '$REMOTE_DIR/$MTP_GGUF' ]" 2>/dev/null \
  || { log "M1 缺草稿模型 $MTP_GGUF。设 MTP_GGUF=... 覆盖, 或 NO_MTP=1 跳过"; exit 1; }
ssh "$REMOTE" "[ -f '$REMOTE_DIR/$MODEL' ]" 2>/dev/null || { log "M1 缺模型 $MODEL"; exit 1; }

# ---------------- 1. 同步代码 → M1 ----------------
log "同步源码 → $REMOTE:$REMOTE_DIR"
rsync -a --exclude '.git' --exclude '*.o' --exclude '*.gguf' --exclude 'gguf/' \
  --exclude '*.bin' --exclude 'ds4' --exclude 'ds4-server' --exclude 'ds4-bench' \
  --exclude 'ds4-eval' --exclude 'ds4-agent' --exclude 'e0-pingpong' --exclude 'ds4_test' \
  "$LOCAL_DIR"/ "$REMOTE:$REMOTE_DIR"/ || { log "rsync 失败"; exit 1; }

# ---------------- 2. 两边 clean + build (共享 CORE_OBJS, 必须都重编) ----------------
log "本机: make clean && make ds4"
( cd "$LOCAL_DIR" && make clean >/dev/null 2>&1 && make ds4 >/tmp/mtp_pipe_build_local.log 2>&1 ) \
  || { log "本机编译失败:"; tail -8 /tmp/mtp_pipe_build_local.log; exit 1; }
log "M1:   make clean && make ds4"
ssh "$REMOTE" "cd '$REMOTE_DIR' && make clean >/dev/null 2>&1 && make ds4 >/tmp/mtp_pipe_build_remote.log 2>&1" \
  || { log "M1 编译失败:"; ssh "$REMOTE" "tail -8 /tmp/mtp_pipe_build_remote.log"; exit 1; }

# ---------------- 3. 清旧进程 + 清端口 ----------------
log "清理两边旧 ds4 进程与 $PORT 端口占用"
pkill -f 'ds4 -m' 2>/dev/null || true
ssh "$REMOTE" "pkill -f 'ds4 -m' 2>/dev/null; lsof -nP -iTCP:$PORT -t 2>/dev/null | xargs -r kill -9 2>/dev/null; true" 2>/dev/null || true
sleep 1

# ---------------- 4. 先起 M1 worker (control listen, 等 coordinator 来拨) ----------------
log "启动 M1 worker: --listen $WORKER_IP:$PORT --layers $SPLIT_WORKER ${WORKER_MTP_ARGS:-(无 MTP)} (reverse, 只 accept)"
ssh "$REMOTE" "cd '$REMOTE_DIR' && rm -f '$WORKER_LOG'; \
  $REMOTE_RUN_ENV DS4_MEM_BUDGET_MB=$REMOTE_BUDGET_MB \
  nohup ./ds4 -m '$MODEL' --role worker --listen '$WORKER_IP' '$PORT' \
  --layers '$SPLIT_WORKER' $DEBUG_ARGS $WORKER_MTP_ARGS \
  -c '$CTX' --temp 0 --nothink > '$WORKER_LOG' 2>&1 & echo \$! > '$WORKER_PID_FILE'; echo launched" 2>/dev/null

log "等 M1 worker backend 就绪并开始 control listen…"
wok=0
for _ in $(seq 1 90); do
  if ssh "$REMOTE" "grep -q 'waiting for coordinator\|control listen' '$WORKER_LOG'" 2>/dev/null; then wok=1; break; fi
  if ssh "$REMOTE" "grep -qiE 'refusing to load|fatal|Address already|invalid|Insufficient Memory' '$WORKER_LOG'" 2>/dev/null; then
    log "M1 worker 启动失败, 日志尾:"; ssh "$REMOTE" "tail -10 '$WORKER_LOG'"; cleanup; exit 1
  fi
  rg=$(mem_gb_remote); over "$rg" "$REMOTE_MAX_GB" && { log "M1 加载阶段 footprint ${rg}G 超 ${REMOTE_MAX_GB}G → 杀"; cleanup; exit 2; }
  sleep 1
done
[ "$wok" = 1 ] || { log "M1 worker 90s 未就绪 → 放弃"; ssh "$REMOTE" "tail -10 '$WORKER_LOG'"; cleanup; exit 1; }
sleep 1

# ---------------- 5. 起本机 coordinator (主动拨 M1 worker, 跑一次性生成) ----------------
log "启动本机 coordinator: --coordinator $WORKER_IP:$PORT --layers $SPLIT_COORD ${COORD_MTP_ARGS:-(无 MTP)} (reverse, 主动拨)"
[ "$EXPERT_PROFILE" = 1 ] && log "专家命中率模拟 cache: coordinator=${LOCAL_EXPERT_PROFILE_CACHE_MB}MiB worker=${REMOTE_EXPERT_PROFILE_CACHE_MB}MiB (仅模拟)"
log "真实 expert pool: coordinator=${LOCAL_EXPERT_POOL_MB}MiB layers=${LOCAL_EXPERT_POOL_LAYER_START}:${LOCAL_EXPERT_POOL_LAYER_END} worker=${REMOTE_EXPERT_POOL_MB}MiB layers=${REMOTE_EXPERT_POOL_LAYER_START}:${REMOTE_EXPERT_POOL_LAYER_END} per-layer-LRU(min_slots=$EXPERT_POOL_MIN_LAYER_SLOTS,warm_batch=$EXPERT_POOL_WARM_BATCH) + 预测异步预取(lookahead=$EXPERT_POOL_PREFETCH_LOOKAHEAD top=$EXPERT_POOL_PREFETCH_TOP self=$EXPERT_POOL_PREFETCH_SELF adjacent=$EXPERT_POOL_PREFETCH_ADJACENT wait=$EXPERT_POOL_WAIT_INFLIGHT fg_fill=$EXPERT_POOL_FOREGROUND_FILL pf_evict=$EXPERT_POOL_PREFETCH_EVICT admit_after=$EXPERT_POOL_ADMIT_AFTER hotlock=$EXPERT_POOL_HOTLOCK_TOP)"
log "源专家 LRU cache: coordinator=${LOCAL_EXPERT_SOURCE_CACHE_MB}MiB layers=${LOCAL_EXPERT_SOURCE_CACHE_LAYER_START}:${LOCAL_EXPERT_SOURCE_CACHE_LAYER_END} worker=${REMOTE_EXPERT_SOURCE_CACHE_MB}MiB layers=${REMOTE_EXPERT_SOURCE_CACHE_LAYER_START}:${REMOTE_EXPERT_SOURCE_CACHE_LAYER_END} admit_after=$EXPERT_SOURCE_CACHE_ADMIT_AFTER hard_copy=$EXPERT_SOURCE_CACHE_HARD_COPY mlock=$EXPERT_SOURCE_CACHE_MLOCK"
log "  (首次可能弹 macOS 本地网络授权框 → 点允许)"
cd "$LOCAL_DIR"
rm -f "$COORD_LOG" "$COORD_OUT"
env $LOCAL_RUN_ENV DS4_MEM_BUDGET_MB=$LOCAL_BUDGET_MB \
  ./ds4 -m "$MODEL" --role coordinator --coordinator "$WORKER_IP" "$PORT" \
  --layers "$SPLIT_COORD" $DEBUG_ARGS $COORD_MTP_ARGS \
  -c "$CTX" -n "$NPRED" --temp 0 --seed "$SEED" --nothink \
  -p "$PROMPT" > "$COORD_OUT" 2> "$COORD_LOG" &
COORD_PID=$!

# ---------------- 6. 看门狗: 等本机 coordinator 一次性生成结束 ----------------
log "运行中… coordinator(本机) 生成完即退出。(Ctrl+C 两边同杀; 本机>${LOCAL_MAX_GB}G 或 M1>${REMOTE_MAX_GB}G 也同杀)"
done_flag=0
for _ in $(seq 1 1800); do
  if ! kill -0 "$COORD_PID" 2>/dev/null; then done_flag=1; break; fi
  lg=$(mem_gb_local "$COORD_PID"); rg=$(mem_gb_remote)
  mlabel=$(mem_label)
  printf "\r[mtp-pipe] %s 本机coord=%sG/%dG  M1worker=%sG/%dG    " "$mlabel" "$lg" "$LOCAL_MAX_GB" "$rg" "$REMOTE_MAX_GB"
  over "$lg" "$LOCAL_MAX_GB" && { echo; log "本机 coordinator ${mlabel} ${lg}G 超限 → 两边同杀"; cleanup; exit 2; }
  over "$rg" "$REMOTE_MAX_GB" && { echo; log "M1 worker ${mlabel} ${rg}G 超限 → 两边同杀"; cleanup; exit 2; }
  sleep 1
done
echo

# ---------------- 7. 结果 (在本机 coordinator 日志) ----------------
# 注: 生成文本可能含大量换行 token (尤其退化的 reduced-expert 模型), 末尾常是成片空行。
# 必须先滤掉纯空白行再 tail, 否则 tail 全抓到尾部空行 → 看着像"没输出文本"(其实内容在前面)。
log "本机 coordinator 生成文本 (stdout, 已滤纯空白行):"
gen_lines=$(grep -v -E '^[[:space:]]*$' "$COORD_OUT" 2>/dev/null || true)
if [ -n "$gen_lines" ]; then
  printf '%s\n' "$gen_lines" | tail -40
else
  log "(stdout 无非空文本行: 本次生成可能全是换行/空白 token；或 coordinator 未完成输出 flush)"
fi
echo "----------------------------------------"
if grep -qiE 'prefill:|generation:|t/s' "$COORD_LOG" 2>/dev/null; then
  log "速度 (本机 coordinator):"; grep -iE 'prefill:|generation:|t/s' "$COORD_LOG" | tail -2
else
  log "未拿到计时行, 本机 coordinator 日志尾:"; tail -12 "$COORD_LOG"
fi
if grep -qiE 'dist-mtp|adaptive-disable' "$COORD_LOG" 2>/dev/null; then
  echo "----------------------------------------"
  log "MTP 摘要 (本机 coordinator):"; grep -iE 'dist-mtp|adaptive-disable' "$COORD_LOG" | tail -20
fi
if grep -qiE 'expert-(profile|pool|source-cache)' "$COORD_LOG" 2>/dev/null; then
  echo "----------------------------------------"
  log "专家 LRU/pool/source 摘要 (本机 coordinator):"
  grep -iE 'expert-(profile|pool|source-cache) (summary|live|layer|  L|memory|enabled|enabled via)' "$COORD_LOG" | tail -100 || true
fi
if ssh "$REMOTE" "grep -qiE 'expert-(profile|pool|source-cache)' '$WORKER_LOG'" 2>/dev/null; then
  echo "----------------------------------------"
  log "专家 LRU/pool/source 摘要 (M1 worker):"
  ssh "$REMOTE" "grep -iE 'expert-(profile|pool|source-cache) (summary|live|layer|  L|memory|enabled|enabled via)' '$WORKER_LOG' | tail -100" 2>/dev/null || true
fi
[ "$done_flag" = 1 ] || log "(注: coordinator 未正常结束, 上面是当前日志快照)"
cleanup
