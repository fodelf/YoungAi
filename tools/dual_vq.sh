#!/bin/bash
# dual_vq.sh — 双机层切分跑 VQ overlay 模型 (ds4-code1b.gguf + DS4_VQ_DIR 侧车)。
# 目的: 单机 43 层单-CB 撞 16GB OOM (kIOGPUCommandBufferCallbackErrorOutOfMemory);
#       层切分后每机 ~20 层 → CB 工作集减半 → 绕过 OOM + prefill 流水线。
# 拓扑(reverse-connect): M1 worker 只 listen/accept; M4 coordinator 主动拨 M1。
#   M4 coord: 层 0:SPLIT + token_embd + output head, 做采样。
#   M1 worker: 层 SPLIT:output backbone。
# 裸续写(BASE 模型): --nothink + BOS+代码前缀 prompt。
# 内存安全(铁律): 两机各带 footprint 看门狗, 超 MAXMB 红线杀两边(只杀进程不删文件)。
set -uo pipefail

REMOTE=${REMOTE:-192.168.1.2}
WORKER_IP=${WORKER_IP:-192.168.1.2}
LOCAL_DIR=${LOCAL_DIR:-/Users/fodelf/git/ds4-main}
REMOTE_DIR=${REMOTE_DIR:-/Users/fodelf/ds4-main}
MODEL=${MODEL:-gguf/go-onebit/ds4-code1b.gguf}
# COORD_MODEL: 协调机侧模型覆盖(合一 VQ GGUF 行为门用: M4 merged vs M1 旧模式混部)
COORD_MODEL=${COORD_MODEL:-$MODEL}
# CORR: corr 快校准侧车(相对路径, 两机各自解析; 空=不挂)
CORR=${CORR:-}
LOCAL_VQ=${LOCAL_VQ:-/Users/fodelf/git/ds4-main/gguf/go-onebit/layers}
REMOTE_VQ=${REMOTE_VQ:-/Users/fodelf/ds4-main/gguf/go-onebit/layers}
PORT=${PORT:-5599}
SPLIT_COORD=${SPLIT_COORD:-0:19}
SPLIT_WORKER=${SPLIT_WORKER:-20:output}
CTX=${CTX:-512}
NPRED=${NPRED:-16}
SEED=${SEED:-1}
MAXMB=${MAXMB:-11800}
BUDGET_MB=${BUDGET_MB:-11264}
RUN_TIMEOUT=${RUN_TIMEOUT:-600}
# VQ 路: 默认走 CPU MoE(稳,无OOM); DS4_VQ_GPU=1 走 GPU F16W(快,但每机层数需不撞OOM)
VQ_GPU_ENV=""
[ "${VQ_GPU:-0}" = 1 ] && VQ_GPU_ENV="DS4_VQ_GPU=1"
# REPEAT_FREQ 默认 0(2026-07-27 消融判决): =1 是 mono/go2b 治退化配方遗留, 对 go1b/VQ
# 模型把代码高频 token(`}`,value,items,Cache)压出"假质量洞"(漏字段/漏初始化/丢返回值);
# 引擎策略 ds4.c:1790 本就禁对 go1b 武装(07-13 实测)。A/B: REPEAT_FREQ=0 vs =1 两版
# LRU 对照在 reports/cli_coding_vqfree{_clean,}*; 归因二分证 LOOP_BREAK(引擎默认ON)无害。
NUM_ENV="DS4_METAL_MATH_SAFE=1 DS4_METAL_KV_RAW_F32=1 DS4_METAL_ROPE_EXP2_LOG2=1 DS4_REPEAT_FREQ=${REPEAT_FREQ:-0} DS4_METAL_PREFILL_CHUNK=8"
[ "${VQ_GT:-0}" = 1 ] && NUM_ENV="$NUM_ENV DS4_VQ_GT=1"
[ -n "${GATHER_THREADS:-}" ] && NUM_ENV="$NUM_ENV DS4_METAL_EXPERT_GATHER_THREADS=$GATHER_THREADS"
# CORR_SCALE: corr 阻尼 α(DS4_CORR_SCALE, 23层复利爆炸历史雷的解药; 空=1.0)
[ -n "${CORR_SCALE:-}" ] && NUM_ENV="$NUM_ENV DS4_CORR_SCALE=$CORR_SCALE"
# CAP_DIR: 双侧引擎轨迹采集(DS4_CAP_DIR, 行为校准 Step A 用; coordinator 层落本机, worker 层落 M1 同路径)
[ -n "${CAP_DIR:-}" ] && NUM_ENV="$NUM_ENV DS4_CAP_DIR=$CAP_DIR"
# 分布式性能 env(之前漏了 → hop 同步串行, 慢 5×):
#  SPEC_PIPE=hop 投机流水线重叠(深度4), NO_MODEL_WARMUP=不扫冷专家, MM_ID_MIN=8, PREFILL_CAP, GATHER_THREADS=8
# DS4_DIST_PREFILL_CAP 注意: 它会覆盖 session prefill_cap(仅当 ≤ctx 时生效, ds4.c:21165)。
# VQ lane 的 span 受 VQ scratch 墙约束(span~20tok 即 3GB), 必须与 PREFILL_CHUNK 同档;
# q2 lane 的 2048 大 span 在 ctx≥2048 时会悄然生效→整 prompt 一 span→scratch 9.9GB 崩(v5 实证)。
DIST_ENV="DS4_DIST_REVERSE_CONNECT=1 DS4_DIST_SPEC_PIPE=${SPEC_PIPE:-1} DS4_DIST_SPEC_PIPE_DEPTH=${SPEC_PIPE_DEPTH:-4} DS4_METAL_NO_MODEL_WARMUP=1 DS4_METAL_MOE_MM_ID_MIN=8 DS4_DIST_PREFILL_CAP=${DIST_PREFILL_CAP:-8} DS4_METAL_EXPERT_GATHER_THREADS=${GATHER_THREADS:-8}"

# 默认裸续写 prompt (BASE 模型, twoSum 历史可比)
if [ -z "${PROMPT+x}" ]; then
  PROMPT=$(cat <<'PEOF'
<｜begin▁of▁sentence｜>// twoSum returns the indices of the two numbers in nums that add up to target.
func twoSum(nums []int, target int) []int {
PEOF
)
fi

WLOG=/tmp/dual_vq_worker.log; CLOG=/tmp/dual_vq_coord.log; COUT=/tmp/dual_vq_coord.out
log(){ echo "[dual_vq $(date +%H:%M:%S)] $*" >&2; }

cleanup(){
  log "cleanup: 杀两边进程(不删文件)"
  ssh "$REMOTE" "pkill -9 -f 'ds4 .*--role worker' 2>/dev/null" 2>/dev/null || true
  [ -n "${CPID:-}" ] && kill -9 "$CPID" 2>/dev/null || true
  kill -9 ${TAILPID:-0} 2>/dev/null || true
}
trap 'cleanup; exit 130' INT TERM

mem_remote(){ ssh "$REMOTE" "footprint -p \$(pgrep -f 'ds4 --role worker'|head -1) 2>/dev/null | grep -Eo 'Footprint: *[0-9.]+ *[KMG]B'|head -1|awk '{v=\$2;u=\$3;if(u==\"GB\")v*=1024;else if(u==\"KB\")v/=1024;printf \"%d\",v}'" 2>/dev/null; }
mem_local(){ footprint -p "${CPID:-0}" 2>/dev/null | grep -Eo 'Footprint: *[0-9.]+ *[KMG]B'|head -1|awk '{v=$2;u=$3;if(u=="GB")v*=1024;else if(u=="KB")v/=1024;printf "%d",v}'; }

# 1. 起 M1 worker (listen, 等 coordinator)
log "起 M1 worker: --listen $WORKER_IP:$PORT --layers $SPLIT_WORKER (VQ_DIR=$REMOTE_VQ ${VQ_GPU_ENV:-CPU-MoE})"
ssh "$REMOTE" "cd '$REMOTE_DIR' && pkill -9 -f 'ds4 .*--role worker' 2>/dev/null; sleep 1; rm -f '$WLOG'; \
  $DIST_ENV $NUM_ENV $VQ_GPU_ENV DS4_VQ_DIR='$REMOTE_VQ' DS4_MEM_BUDGET_MB=$BUDGET_MB \
  nohup ./ds4 -m '$MODEL' ${CORR:+--corr '$CORR'} --role worker --listen '$WORKER_IP' '$PORT' --layers '$SPLIT_WORKER' \
  -c '$CTX' --temp 0 --nothink > '$WLOG' 2>&1 & echo launched" 2>/dev/null

log "等 worker 就绪 (control listen)…"
wok=0
for _ in $(seq 1 120); do
  if ssh "$REMOTE" "grep -qiE 'waiting for coordinator|control listen|backend initialized' '$WLOG'" 2>/dev/null; then wok=1; break; fi
  if ssh "$REMOTE" "grep -qiE 'refusing|already running|fatal|Insufficient Memory|invalid|error' '$WLOG'" 2>/dev/null; then
    log "worker 启动失败:"; ssh "$REMOTE" "tail -12 '$WLOG'"; cleanup; exit 1; fi
  rg=$(mem_remote); [ -n "${rg:-}" ] && [ "${rg:-0}" -gt "$MAXMB" ] && { log "worker footprint ${rg}MB>红线 → 杀"; cleanup; exit 2; }
  sleep 1
done
[ "$wok" = 1 ] || { log "worker 120s 未就绪"; ssh "$REMOTE" "tail -12 '$WLOG'"; cleanup; exit 1; }
sleep 1

# 2. 起 M4 coordinator (拨 M1, 一次性生成)
log "起 M4 coordinator: --coordinator $WORKER_IP:$PORT --layers $SPLIT_COORD (VQ_DIR=$LOCAL_VQ)"
cd "$LOCAL_DIR"; rm -f "$CLOG" "$COUT"
env $DIST_ENV $NUM_ENV $VQ_GPU_ENV DS4_VQ_DIR="$LOCAL_VQ" DS4_MEM_BUDGET_MB=$BUDGET_MB \
  ./ds4 -m "$COORD_MODEL" ${CORR:+--corr "$CORR"} --role coordinator --coordinator "$WORKER_IP" "$PORT" --layers "$SPLIT_COORD" \
  -c "$CTX" -n "$NPRED" --temp 0 --seed "$SEED" --nothink ${DUMP_LP:+--dump-logprobs "$DUMP_LP"} -p "$PROMPT" > "$COUT" 2> "$CLOG" &
CPID=$!
tail -n +1 -f "$COUT" 2>/dev/null & TAILPID=$!

# 3. 看门狗
log "运行中 (超 ${RUN_TIMEOUT}s 或任一机 >${MAXMB}MB 杀两边)…"
T0=$(date +%s)
while kill -0 "$CPID" 2>/dev/null; do
  EL=$(( $(date +%s) - T0 ))
  [ "$EL" -ge "$RUN_TIMEOUT" ] && { log "超时 ${EL}s → 杀"; cleanup; break; }
  ml=$(mem_local); [ -n "${ml:-}" ] && [ "${ml:-0}" -gt "$MAXMB" ] && { log "coord ${ml}MB>红线 → 杀"; cleanup; exit 9; }
  rg=$(mem_remote); [ -n "${rg:-}" ] && [ "${rg:-0}" -gt "$MAXMB" ] && { log "worker ${rg}MB>红线 → 杀"; cleanup; exit 9; }
  sleep 3
done
RC=0; wait "$CPID" 2>/dev/null || RC=$?
kill -9 "${TAILPID:-0}" 2>/dev/null || true
ssh "$REMOTE" "pkill -9 -f 'ds4 .*--role worker' 2>/dev/null" 2>/dev/null || true

echo "" >&2
echo "===== 原始输出(逐字) =====" >&2
cat "$COUT" 2>/dev/null >&2
echo "" >&2
echo "===== 速度/内存行 =====" >&2
grep -aE "t/s|prefill|decode|budget|resident|hidden|worker" "$CLOG" 2>/dev/null | tail -8 >&2
log "rc=$RC 用时=$(( $(date +%s) - T0 ))s (coord日志 $CLOG, worker日志 M1:$WLOG)"
