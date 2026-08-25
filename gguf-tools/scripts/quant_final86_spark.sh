#!/bin/bash
# 定版 86G 配方(2026-08-17 用户令"正常量化到86g但要快"):
#   专家 = IQ2_XXS(w1/w3) + Q2_K(w2)  ← 用户 86G 设计的原档位, 官方同构, 引擎 LUT 特调
#   backbone = Q4_K(q/kv 投影+shared+输出头)  ← b4 实测的提速件(每 token 少读 ~1.7GB)
#   attn_output_a/b: 默认 q8_0; 2026-08-17晚 q4_K kernel 全家落地后用户批准 v2=q4_k
#     (AO=q4_k OUT=.../ds4-final86v2.gguf 跑, 体积 ~83.6GB, 行为门过前不覆盖前代)
#   indexer.attn_q_b 与 token_embd 保 f16(硬校验)
# 预算账: 专家 71.5 + backbone ~13.5 = ~85.1 GB (AO=q4_k 时 ~83.6)
set -u
AO="${AO:-q8_0}"                                   # attn_output_a/b 档位
OUT="${OUT:-$HOME/ds4-main/gguf/ds4-final86.gguf}"
cd ~/ds4-main/gguf-tools
ARGS=""
for L in $(seq 0 42); do
  ARGS="$ARGS --tensor-type blk.$L.indexer.attn_q_b.weight=f16"
  ARGS="$ARGS --tensor-type blk.$L.attn_output_a.weight=$AO"
  ARGS="$ARGS --tensor-type blk.$L.attn_output_b.weight=$AO"
done
exec ./deepseek4-quantize \
  --hf ~/ds4-main/hf/DeepSeek-V4-Flash-0731 \
  --template ~/ds4-main/gguf/go-onebit/r30/template_head.gguf \
  --out "$OUT" \
  --routed-w1 iq2_xxs --routed-w3 iq2_xxs --routed-w2 q2_k \
  --attention-proj q4_k --shared q4_k --output q4_k \
  $ARGS --threads 20 --overwrite "$@"
