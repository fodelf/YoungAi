#!/bin/sh
# profile_corr.sh — R3-e: pin down why the 43-layer sidecar scores ~5× slower
# than the 22-layer one on the token-by-token perplexity path (async-submit
# already proved ~free on the paired decode bench). Three paired arms on the
# SAME short slice, expert-IO profiling on:
#   A 22-layer sidecar   B 43-layer EF sidecar   C bare
# (env 大扫除 2026-08-31: 逐相 IO profiling 诊断口 DS4_METAL_EXPERT_IO_PROFILE 已从引擎
#  删除 — gather/drain 行不再出现, 本脚本只剩 avg_nll/t-s 维度可比。)
#   usage: profile_corr.sh [EVAL=go_heldout_300.txt]
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
EVAL=${1:-$ROOT/gguf-tools/data/corpus/go_heldout_300.txt}
cd "$ROOT" || exit 1
for arm in "A22:gguf/ds4-go1b-corr-rrr-partial.gguf" "B43:gguf/ds4-go1b-corr-ef.gguf" "Cbare:"; do
    name=${arm%%:*}; corr=${arm#*:}
    CORRFLAG=""; [ -n "$corr" ] && CORRFLAG="--corr $corr"
    echo "=== $name ==="
    # shellcheck disable=SC2086
    ./ds4 -m gguf/ds4-go1b.gguf $CORRFLAG \
        --perplexity-file "$EVAL" --metal 2>&1 | \
        grep -aE 'avg_nll|profile|gather|drain|corr' | tail -12
done
echo PROFILE-DONE
