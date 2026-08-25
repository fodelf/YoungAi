#!/bin/bash
# r28v2_alpha_probe.sh — 合并后 α 判决探针(2026-08-01)。
#
# 背景: r28v2 合并时沿用了冠军的 α=2.5, 但 fable5 早有实测"α 不可跨模型移植"
#       (v7 套 2.5 后 L03 路由一致 89.7%→78.6% 反而更差)。首跑输出退化重复,
#       α 是唯一不需重量化就能改的因子 —— 先用它做分离实验。
#
# 方法: route_alpha_set.py 是幂等绝对写(首次存 <model>.bias0.bin 裸态快照, 之后每次
#       从快照重算), 所以 α 可来回扫不累积。每个 α 跑同一 prompt 定性看输出形态。
#
# 用法: r28v2_alpha_probe.sh "<α列表>" ["<prompt>"] [ntok]
set -uo pipefail
ROOT="$HOME/ds4-main"
M="$ROOT/gguf/go-onebit/ds4-r28v2.gguf"
RB="$ROOT/gguf/go-onebit/r28v2/full/route_bias_r28.bin"
ALPHAS="${1:-0 1.0 2.5}"
PROMPT="${2:-写一个Go函数,计算两个整数之和}"
NTOK="${3:-60}"
S="$ROOT/gguf-tools/go-onebit/scripts"

[ -f "$M" ]  || { echo "模型缺: $M" >&2; exit 2; }
[ -f "$RB" ] || { echo "Δb 缺: $RB" >&2; exit 2; }

# 单次前向的 env: chunk=4 是 M1 Pro 上的硬约束(VQ gather scratch 50MB/专家,
# recommendedMax 10.67 GiB 减去 8.20 GiB backbone 只剩 ~2.4 GiB)
E="DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=4 DS4_VQ_SCRATCH_GB=2.2 \
DS4_VQ_GPU=1 DS4_MEM_BUDGET_MB=12000 DS4_ZCHAIN=$ROOT/gguf/go-onebit/r28v2/full/zchain.bin"

for A in $ALPHAS; do
    echo "########## α=$A ##########"
    "$(dirname "$0")/../calib/route_alpha_set" "$M" "$RB" "$A" 2>&1 | tail -2
    env $E "$ROOT/ds4" -m "$M" --ctx 4096 -p "$PROMPT" -n "$NTOK" --temp 0 2>/dev/null
    echo
done
echo "★探针完 — 模型当前 α=$(echo $ALPHAS | awk '{print $NF}'), 用 route_alpha_set.py 可改回★"
