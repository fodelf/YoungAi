#!/bin/bash
# dsq_gpu_compare_spark.sh — IQ2_XXS CUDA 编码器的逐字节对拍闸(2026-08-19)。
# 同一进程内: 每张专家张量先 GPU 编码, 再用 CPU 编码器编一遍逐字节比较
# (DS4Q_GPU_VERIFY=1 走 quants.c 里的 verify 分支), 输入完全相同 → 只测编码器本身。
# 量化配方与 quant_allq2_spark.sh 保持一致(无 imatrix → 由数据自合成, 两侧同值)。
#
# 用法: bash dsq_gpu_compare_spark.sh [张量名] [线程数]
set -uo pipefail
ROOT="$HOME/ds4-main"
T="${1:-blk.20.ffn_gate_exps.weight}"
NT="${2:-20}"
cd "$ROOT/gguf-tools"
export DS4Q_GPU_VERIFY=1
./deepseek4-quantize \
  --hf "$ROOT/hf/DeepSeek-V4-Flash-0731" \
  --template "$ROOT/gguf/go-onebit/r30/template_head.gguf" \
  --routed-w1 iq2_xxs --routed-w3 iq2_xxs --routed-w2 q2_k \
  --attention-proj q2_k --attention q2_k --shared q2_k --output q2_k \
  --dense q2_k --embedding q2_k \
  --threads "$NT" --compare-tensor "$T"
echo "COMPARE_EXIT=$?"
