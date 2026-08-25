#!/bin/bash
# Spark backbone-q4k 配方(2026-08-17): 专家 q2_K + q/kv 投影、shared、输出头 q4_K。
# attn_output_a/b 保 q8_0(深度融合 kernel 未移植), indexer.attn_q_b 保 f16(引擎硬校验),
# token_embd 保 f16(引擎硬校验)。产物 98.04 GB。
# 教训: 脚本曾放 /tmp 被重启清掉 —— 战役脚本必须入 repo(铁律)。
set -u
cd ~/ds4-main/gguf-tools
ARGS=""
for L in $(seq 0 42); do
  ARGS="$ARGS --tensor-type blk.$L.indexer.attn_q_b.weight=f16"
  ARGS="$ARGS --tensor-type blk.$L.attn_output_a.weight=q8_0"
  ARGS="$ARGS --tensor-type blk.$L.attn_output_b.weight=q8_0"
done
exec ./deepseek4-quantize \
  --hf ~/ds4-main/hf/DeepSeek-V4-Flash-0731 \
  --template ~/ds4-main/gguf/go-onebit/r30/template_head.gguf \
  --out ~/ds4-main/gguf/ds4-b4.gguf \
  --experts q2_k --attention-proj q4_k --shared q4_k --output q4_k \
  $ARGS --threads 20 --overwrite "$@"
