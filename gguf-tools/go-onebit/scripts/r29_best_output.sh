#!/bin/bash
# r28v2_best_output.sh — 找 r28v2 这个模型能出的最好输出(2026-08-01 用户令"看看输出是最好的样子")。
#
# 两阶段, 不盲扫笛卡尔积:
#   阶段1  固定贪心, 扫 α 网格 → 已知 α=2.5 明显优于 α=0, 但从未扫过上界
#   阶段2  用阶段1 的每个 α 都跑一遍采样器组合, 逐条原样打印(判读交给人)
#
# α 用 route_alpha_set.py 幂等绝对写(从 <model>.bias0.bin 裸态快照重算), 来回扫不累积。
# 收尾一定把 α 设回 BEST_A, 别把模型留在扫描中间态。
#
# 用法: r28v2_best_output.sh [α列表] [ntok]
set -uo pipefail
ROOT="$HOME/ds4-main"
M="$ROOT/gguf/go-onebit/ds4-r29.gguf"
RB="$ROOT/gguf/go-onebit/r29/full/route_bias_r29.bin"
S="$ROOT/gguf-tools/go-onebit/scripts"
ALPHAS="${1:-1.5 2.5 4.0 6.0}"
NTOK="${2:-64}"
BEST_A="${BEST_A:-2.5}"

[ -f "$M" ] || { echo "模型缺: $M" >&2; exit 2; }
# ★必须在仓库根跑★: ds4 从别处启动时 metal/*.metal 解析不到,
# 表现为 "zchain GPU upload failed"(不是内存问题, 是 pipeline 建不起来)。
cd "$ROOT" || exit 2

# ★env 必须用数组★: 上一版 r28v2_alpha_probe.sh 把 env 串成含续行反斜杠的字符串,
# `env $E ./ds4` 展开后把 "\" 当成参数, ds4 静默失败(2>/dev/null 又吞了错误)⇒ 三个 α 全空输出。
ENVV=(DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=4 DS4_VQ_SCRATCH_GB=2.2
      DS4_VQ_GPU=1 DS4_MEM_BUDGET_MB=12000
      "DS4_ZCHAIN=$ROOT/gguf/go-onebit/r29/full/zchain.bin")

# prompt: 代码前缀在首轮分离实验里比 chat 问句表现好(前段连贯), 两种都跑
P_CODE='func add(a, b int) int {'
P_CHAT='写一个Go函数,计算两个整数之和'

run() {  # run <tag> <prompt> [额外env...]
    local tag="$1" prompt="$2"; shift 2
    echo "----- $tag -----"
    env "${ENVV[@]}" "$@" "$ROOT/ds4" -m "$M" --ctx 4096 -p "$prompt" -n "$NTOK" \
        --temp "${T:-0}" ${TOPP:+--top-p $TOPP} 2>/tmp/best.err || echo "[跑失败] $(tail -2 /tmp/best.err)"
    echo
}

echo "==================== 阶段1: α 网格(贪心, 代码前缀) ===================="
for A in $ALPHAS; do
    "$(dirname "$0")/../calib/route_alpha_set" "$M" "$RB" "$A" >/dev/null 2>&1
    T=0 TOPP= run "α=$A | greedy | code-prefix" "$P_CODE"
done

echo "==================== 阶段2: BEST_A=$BEST_A 下扫采样器 ===================="
"$(dirname "$0")/../calib/route_alpha_set" "$M" "$RB" "$BEST_A" >/dev/null 2>&1
T=0.7 TOPP=0.9 run "α=$BEST_A | t0.7/p0.9 | code-prefix" "$P_CODE"
T=0.7 TOPP=0.9 run "α=$BEST_A | t0.7/p0.9 +rep1.1 | code-prefix" "$P_CODE" DS4_REPEAT_FREQ=1.1 DS4_REPEAT_WINDOW=128
T=0.3 TOPP=0.9 run "α=$BEST_A | t0.3/p0.9 +rep1.1 | code-prefix" "$P_CODE" DS4_REPEAT_FREQ=1.1 DS4_REPEAT_WINDOW=128
T=0.7 TOPP=0.9 run "α=$BEST_A | t0.7/p0.9 +rep1.1 | chat"        "$P_CHAT" DS4_REPEAT_FREQ=1.1 DS4_REPEAT_WINDOW=128

"$(dirname "$0")/../calib/route_alpha_set" "$M" "$RB" "$BEST_A" 2>&1 | tail -1
echo "★扫完 — 模型 α 已固定回 ${BEST_A}★"
