#!/bin/bash
# dsq_gpu_timing_spark.sh — IQ2_XXS 编码器 GPU 计时(2026-08-19; CPU 对照 lane 已停用, 见文尾)。
# 两段: ①--dry-run(只加载模板+出计划, 作公共开销基线) ②GPU 生成一张专家张量。
# 生成耗时 = 总耗时 - 基线。张量默认 blk.20.ffn_gate_exps.weight
# (256 专家 x 4096x2048 → 8.4M 元素/专家, 每层两张 iq2_xxs + 一张 q2_k)。
# 用法: bash dsq_gpu_timing_spark.sh [张量名] [线程数]
set -uo pipefail
ROOT="$HOME/ds4-main"
T="${1:-blk.20.ffn_gate_exps.weight}"
NT="${2:-20}"
cd "$ROOT/gguf-tools"
ARGS=(--hf "$ROOT/hf/DeepSeek-V4-Flash-0731"
      --template "$ROOT/gguf/go-onebit/r30/template_head.gguf"
      --routed-w1 iq2_xxs --routed-w3 iq2_xxs --routed-w2 q2_k
      --attention-proj q2_k --attention q2_k --shared q2_k --output q2_k
      --dense q2_k --embedding q2_k --threads "$NT")

run(){ # $1=标签; 计时并打印
    local tag="$1"
    local t0=$(date +%s.%N)
    ./deepseek4-quantize "${ARGS[@]}" --compare-tensor "$T" >/dev/null 2>/tmp/dsq_t_$tag.log
    local t1=$(date +%s.%N)
    echo "$tag $(echo "$t1 - $t0" | bc)"
}

t0=$(date +%s.%N)
./deepseek4-quantize "${ARGS[@]}" --dry-run >/dev/null 2>&1
t1=$(date +%s.%N)
echo "baseline_dryrun $(echo "$t1 - $t0" | bc)"
run gpu   # GPU 编码是唯一路径(GPU-only 铁律 08-28), 无需开关
# ★CPU 对照 lane 停用(2026-08-31 env 大扫除)★: DS4Q_GPU=0 的 CPU 编码开关已无机制 ——
# GPU-only 铁律下 CPU 参考实现不再可选(历史实撞: CPU 参考路给过失真 40 倍且自洽的读数)。
# run cpu DS4Q_GPU=0
