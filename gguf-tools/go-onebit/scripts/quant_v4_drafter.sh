#!/bin/bash
# v4 = v2 配方 + DSpark drafter(mtp 专家提 q4_K) — 投机接受率判别实验
set -u
EXTRA=""
for L in 0 1 2; do for W in gate up down; do
  EXTRA="$EXTRA --tensor-type mtp.$L.ffn_${W}_exps.weight=q4_k"
done; done
cd "$HOME/ds4-main"
AO=q4_k OUT="$HOME/ds4-main/gguf/ds4-final86v4.gguf" \
  exec bash gguf-tools/go-onebit/scripts/quant_final86_spark.sh --mtp-append 3 $EXTRA
