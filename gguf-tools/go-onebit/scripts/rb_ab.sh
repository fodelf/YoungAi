#!/bin/bash
# rb_ab.sh — 路由偏置侧车 A/B 判决(2026-07-28, M1): rr code S=305 学生路由回放 × α 网格。
# 基线(无侧车)=agree 77.6/Σmin 0.7685/KL 0.5007; 神谕上限=82.9/0.7825/0.3815。
# 只读回放(BF_ONLY+TERM_MAXP=0, 反修族不进 env); VERDICT 汇总到 /tmp/rb_ab_summary.txt。
set -u
Q=/Users/fodelf/ds4-main/gguf-tools/go-onebit/quant
: > /tmp/rb_ab_summary.txt
ALPHAS=("$@"); [ ${#ALPHAS[@]} -gt 0 ] || ALPHAS=(0.5 1.0 2.0)
for A in "${ALPHAS[@]}"; do
  echo "[rb_ab] α=$A 起跑 $(date +%T)" >> /tmp/rb_ab_summary.txt
  ( cd "$Q" && env -i HOME="$HOME" PATH=/usr/bin:/bin \
      DS4_HF=/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base \
      DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_GO2B_HOT=1 \
      DS4_GO2B_HOT_TABLE=/Users/fodelf/ds4-main/gguf-tools/go-onebit/corpus/hot_v4.txt \
      DS4_CALIB_FULLSET=1 DS4_THREADS=8 \
      DS4_ANCHOR=/tmp/ds4quant_anchor_rr_rr_code_s305.bin DS4_NL=43 DS4_LCFG=g \
      DS4_COADAPT=1 DS4_LAYER_DIR=/Users/fodelf/ds4-main/gguf/go-onebit/layers \
      DS4_BF_ONLY=1 DS4_BF_TERM_MAXP=0 \
      DS4_ROUTE_BIAS=/tmp/route_bias_v4.bin DS4_ROUTE_BIAS_ALPHA="$A" \
      ./ds4quant_run /tmp/rr_code.ids 305 > "/tmp/rb_ab_a${A}.txt" 2>&1 )
  grep -aE "ROUTE_BIAS apply|VERDICT" "/tmp/rb_ab_a${A}.txt" | sed "s/^/α=$A /" >> /tmp/rb_ab_summary.txt
done
echo "RB_AB_DONE $(date +%T)" >> /tmp/rb_ab_summary.txt
