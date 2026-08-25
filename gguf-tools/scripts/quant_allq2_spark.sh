#!/bin/bash
# quant_allq2_spark.sh — 全 q2 基座·零例外(2026-08-19 用户令"量化不要例外全部q2, 引擎不适配就改"):
#   专家 = IQ2_XXS(w1/w3) + Q2_K(w2); 其余一切 2D 权重(attn全家/compressor/indexer/shared/
#   输出头/embedding/router/hc_fn/dense) = Q2_K。1D norm/bias 数学上非矩阵, 保 f32。
#   行长非 256 倍数的张量由量化器按块约束自回落(装载侧 expect 报实型, 不静默)。
#   引擎适配 = f16 专线家族(embd/compressor/indexer) 装载时 q2→f16 影子(q8 repack 同类,
#   每 token 零成本); 其余原生 q2 kernel。
# 用法: OUT=... bash quant_allq2_spark.sh [额外 quantize 参数]
set -u
OUT="${OUT:-$HOME/ds4-main/gguf/ds4-allq2.gguf}"
HF="${HF:-$HOME/ds4-main/hf/DeepSeek-V4-Flash-0731}"   # 08-21 参数化: DSpark 压缩复用
cd ~/ds4-main/gguf-tools
exec ./deepseek4-quantize \
  --hf "$HF" \
  --template ~/ds4-main/gguf/go-onebit/r30/template_head.gguf \
  --out "$OUT" \
  --routed-w1 iq2_xxs --routed-w3 iq2_xxs --routed-w2 q2_k \
  --attention-proj q2_k --attention q2_k --shared q2_k --output q2_k \
  --dense q2_k --embedding q2_k \
  --threads 20 --overwrite "$@"
