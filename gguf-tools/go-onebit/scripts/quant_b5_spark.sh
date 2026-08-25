#!/bin/bash
# b5 = b4 配方 + attn_output_a/b 也 Q4_K(配套新写的 grouped_q4K/hc_expand_q4K kernel)
set -u
cd ~/ds4-main/gguf-tools
ARGS=""
for L in $(seq 0 42); do
  ARGS="$ARGS --tensor-type blk.$L.indexer.attn_q_b.weight=f16"
done
exec ./deepseek4-quantize \
  --hf ~/ds4-main/hf/DeepSeek-V4-Flash-0731 \
  --template ~/ds4-main/gguf/go-onebit/r30/template_head.gguf \
  --out ~/ds4-main/gguf/ds4-b5.gguf \
  --experts q2_k --attention-proj q4_k --shared q4_k --output q4_k \
  $ARGS --threads 20 --overwrite "$@"
