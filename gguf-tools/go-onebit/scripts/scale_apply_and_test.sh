#!/usr/bin/env bash
# scale_apply_and_test.sh — 杠杆① 收尾: scale 表 → patch M4 mono → 单机短测。
# 安全: M1 保留 pristine mono (回滚源); 单机短生成(-n 有限) + 外部内存看门狗
# (mono 单机持续生成有 kernel panic 前科, 短跑+看门狗兜底, 越线即杀)。
#
# 用法: scale_apply_and_test.sh            # patch + 测
#       scale_apply_and_test.sh test-only  # 跳过 patch (已 patch 过)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
M1=192.168.1.2; M1DIR=/Users/fodelf/ds4-main
MONO="$ROOT/gguf/ds4-mono-mixed.gguf"
PROMPT='<｜begin▁of▁sentence｜>// twoSum returns the indices of the two numbers in nums that add up to target.'
NPRED=${NPRED:-48}
KILL_GB=${KILL_GB:-14}

if [ "${1:-}" != "test-only" ]; then
  echo "[apply] 拉 scale 表 M1 → M4"
  rsync -a "$M1:/tmp/scales/" /tmp/scales/ || { echo "scale 表拉取失败"; exit 1; }
  n=$(ls /tmp/scales/L*.npz 2>/dev/null | wc -l | tr -d ' ')
  echo "[apply] $n 层 scale 表就位; dry-run 校验字节布局"
  python3 "$HERE/scale_apply.py" --gguf "$MONO" --scales /tmp/scales --layers 0-42 --dry-run || exit 1
  echo "[apply] 正式 patch (M1 保留 pristine 备份; 越线可从 M1 回滚)"
  python3 "$HERE/scale_apply.py" --gguf "$MONO" --scales /tmp/scales --layers 0-42 || exit 1
fi

echo "[test] 单机短测 (n=$NPRED, 内存看门狗 ${KILL_GB}G)"
# 外部内存看门狗: ds4 RSS 越 KILL_GB 立杀 (防 wired 暴涨 kernel panic)
( while true; do
    pid=$(pgrep -f "ds4 -m.*mono-mixed" | head -1); [ -z "$pid" ] && exit 0
    kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' '); g=$(( ${kb:-0} / 1048576 ))
    [ "$g" -gt "$KILL_GB" ] && { echo "[watchdog] RSS ${g}G > ${KILL_GB}G 杀"; kill -9 "$pid"; exit 1; }
    sleep 2
  done ) & WD=$!
env DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=2048 DS4_METAL_EXPERT_GATHER_THREADS=8 \
    DS4_METAL_NO_MODEL_WARMUP=1 DS4_MEM_BUDGET_MB=13000 \
    "$ROOT/ds4" -m "$MONO" --metal -c 4096 -n "$NPRED" --temp 0 --seed 1 -p "$PROMPT" 2>/tmp/scaletest_gen.log
kill "$WD" 2>/dev/null
echo "[test] 完成 (原样输出在上; 对比基线 twoSum 逐字)"
