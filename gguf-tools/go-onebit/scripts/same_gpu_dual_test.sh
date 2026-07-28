#!/bin/sh
# same_gpu_dual_test.sh — 决定性诊断: 双机层切分协议, 但 worker 也跑在本机 M4 (同一块 GPU, localhost)。
# 目的: 隔离"双机漂移"是 (A) 层切分/协议 BUG 还是 (B) 跨 GPU 硬件 fp。
#   同-GPU 双机 vs 单机对比:
#     仍漂 → 层切分 BUG (与硬件无关);   写出干净代码(像单机) → 跨 GPU 硬件。
# 两进程用不同 DS4_LOCK_FILE 绕实例锁; 专家 mmap 同文件→页缓存共享不翻倍; ctx/n 压小 + 引擎压力守卫兜底。
set -u
ROOT=/Users/fodelf/git/ds4-main
MODEL=$ROOT/gguf/ds4-mono-mixed.gguf
PORT=5601; CTX=2048; N=${1:-40}
PROMPT='<｜begin▁of▁sentence｜>// twoSum returns the indices of the two numbers in nums that add up to target.
func twoSum(nums []int, target int) []int {'
COMMON="DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=512 DS4_REPEAT_FREQ=${DS4_REPEAT_FREQ:-1} DS4_DBG_PEN=${DS4_DBG_PEN:-} DS4_MTP_SPEC_DISABLE=${DS4_MTP_SPEC_DISABLE:-} DS4_DBG_RP=${DS4_DBG_RP:-}"  # 无 math_safe; 惩罚/调试/spec旗子 env 可覆盖

pkill -f 'ds4 --role' 2>/dev/null; pkill -f "ds4 -m " 2>/dev/null; sleep 1
rm -f /tmp/sg_worker.log /tmp/sg_coord.out /tmp/sg_coord.log

echo "[1/2] 起 worker (本机 M4, 层 20:output, 换锁文件绕实例锁, listen 127.0.0.1:$PORT)" >&2
env DS4_LOCK_FILE=/tmp/ds4w.lock DS4_DIST_REVERSE_CONNECT=1 $COMMON \
  nohup "$ROOT/ds4" --role worker --listen 127.0.0.1 $PORT --coordinator 127.0.0.1 $PORT \
  --layers 20:output -m "$MODEL" -c $CTX --temp 0 --nothink > /tmp/sg_worker.log 2>&1 &
WPID=$!
trap 'kill $WPID 2>/dev/null; pkill -f "ds4 --role" 2>/dev/null' EXIT INT TERM

# 等 worker 就绪 (control listen)
ok=0
for _ in $(seq 1 120); do
  grep -qaE 'control listen|waiting for coordinator|backend initialized' /tmp/sg_worker.log 2>/dev/null && { ok=1; break; }
  grep -qaiE 'refusing|fatal|error|insufficient' /tmp/sg_worker.log 2>/dev/null && { echo "worker 起失败:"; tail -6 /tmp/sg_worker.log; exit 1; }
  kill -0 $WPID 2>/dev/null || { echo "worker 进程没了:"; tail -6 /tmp/sg_worker.log; exit 1; }
  sleep 1
done
[ "$ok" = 1 ] || { echo "worker 120s 未就绪:"; tail -8 /tmp/sg_worker.log; exit 1; }

echo "[2/2] 起 coordinator (本机 M4, 层 0:19, 拨 127.0.0.1:$PORT, 生成)" >&2
echo "---------- 同-GPU 双机 生成(原始输出) ----------"
env DS4_LOCK_FILE=/tmp/ds4c.lock DS4_DIST_REVERSE_CONNECT=1 $COMMON \
  perl -e 'alarm 500; exec @ARGV' \
  "$ROOT/ds4" --role coordinator --coordinator 127.0.0.1 $PORT --layers 0:19 \
  -m "$MODEL" -c $CTX --temp 0 -n $N -p "$PROMPT" --metal 2>/tmp/sg_coord.log | tee /tmp/sg_coord.out
echo ""
echo "---------- 结束 ----------"
grep -aE 't/s|prefill|generation' /tmp/sg_coord.log 2>/dev/null | tail -1 >&2
kill $WPID 2>/dev/null
