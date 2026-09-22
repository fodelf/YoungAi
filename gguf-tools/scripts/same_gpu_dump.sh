#!/bin/sh
# same_gpu_dump.sh — 同-GPU 双机(worker 也在 M4)对 prompt 做一次前向, coordinator dump 首 token logits。
# 配合单机 --dump-logits 对比: 定位 dist 切片 forward vs 单机 forward 的 logits 差幅。
set -u
ROOT=/Users/fodelf/git/ds4-main
MODEL=$ROOT/gguf/ds4-mono-mixed.gguf
PORT=5603; CTX=2048
DUMP=${DUMP:-/tmp/lg_dist.json}
PROMPT='<｜begin▁of▁sentence｜>// twoSum returns the indices of the two numbers in nums that add up to target.
func twoSum(nums []int, target int) []int {'
# (env 大扫除 2026-08-31: EXPERT_OFFLOAD=1 归 AUTO 判定; LOCK_FILE 换锁 env 口已无读取者)
# ★⚠实例锁现写死 /tmp/ds4.lock 且无 flag 绕口 → 本脚本"同机双进程"目前起不来第二个
# ds4(refusing to start), 待锁路径 flag 接线后才能复活。★
COMMON="--prefill-chunk 512"   # 无 strict-fp/repeat: 纯对比 forward 数值

pkill -f 'ds4 --role' 2>/dev/null; pkill -f "ds4 -m " 2>/dev/null; sleep 1
rm -f /tmp/sgd_worker.log "$DUMP"

echo "[1/2] 起 worker (本机 M4, 层 20:output, listen 127.0.0.1:$PORT)" >&2
nohup "$ROOT/ds4" --reverse-connect $COMMON --role worker --listen 127.0.0.1 $PORT --coordinator 127.0.0.1 $PORT \
  --layers 20:output -m "$MODEL" -c $CTX --temp 0 --nothink > /tmp/sgd_worker.log 2>&1 &
WPID=$!
trap 'kill $WPID 2>/dev/null; pkill -f "ds4 --role" 2>/dev/null' EXIT INT TERM
ok=0
for _ in $(seq 1 120); do
  grep -qaE 'control listen|waiting for coordinator|backend initialized' /tmp/sgd_worker.log 2>/dev/null && { ok=1; break; }
  grep -qaiE 'refusing|fatal|error|insufficient' /tmp/sgd_worker.log 2>/dev/null && { echo "worker 失败:"; tail -6 /tmp/sgd_worker.log; exit 1; }
  kill -0 $WPID 2>/dev/null || { echo "worker 没了:"; tail -6 /tmp/sgd_worker.log; exit 1; }
  sleep 1
done
[ "$ok" = 1 ] || { echo "worker 未就绪"; tail -8 /tmp/sgd_worker.log; exit 1; }

echo "[2/2] coordinator dump 首 token logits → $DUMP" >&2
perl -e 'alarm 400; exec @ARGV' \
  "$ROOT/ds4" --reverse-connect $COMMON --role coordinator --coordinator 127.0.0.1 $PORT --layers 0:19 \
  -m "$MODEL" -c $CTX --temp 0 --dump-logits "$DUMP" -p "$PROMPT" --metal 2>/tmp/sgd_coord.log
echo "DIST-DUMP-DONE rc=$? size=$(stat -f%z "$DUMP" 2>/dev/null || echo 0)" >&2
kill $WPID 2>/dev/null
