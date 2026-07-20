#!/bin/sh
# rank_sweep_m1.sh — sweep the z low-rank cap on ONE layer to test whether the
# four-loss activation-space z can capture the Go 1-bit error with enough rank.
# The eigen-decompose at high rank is single-machine slow, so this is meant to run
# detached on M1 (which holds all shards + cap) while M4 stays free.
#
# Fixed data (nx=512 distinct Go inputs, n_exp=64) so V is not input-starved up to
# rank~512. Watch corr_cos climb vs base_cos: steep climb => rank is the lever;
# plateau => linear activation z has a real ceiling (regularity lives elsewhere).
#
#   ssh M1 'nohup sh .../rank_sweep_m1.sh > /tmp/rank_sweep.log 2>&1 &'
set -u
ROOT="${ROOT:-/Users/fodelf/ds4-main}"
HF="$ROOT/hf/DeepSeek-V4-Flash-Base"; CAP="$ROOT/cap_m1"
cd "$ROOT/gguf-tools" || exit 1
LAYER="${LAYER:-8}"; NX="${NX:-512}"; NE="${NE:-64}"
echo "rank sweep: L=$LAYER nx=$NX n_exp=$NE  (col NF-1=base_cos NF=corr_cos)"
for R in 64 128 256 512; do
  echo "### rank=$R start $(date +%T) ###"
  ./calib_run --hf "$HF" --cap "$CAP" --layers "$LAYER" --solver hv \
     --nx "$NX" --n-experts "$NE" --maxrank "$R" --w-align 1 --threads 8 2>/dev/null \
     | awk 'NR>=5 && $1=='"$LAYER"' {print "rank='"$R"' -> "$0}'
  echo "rank=$R done $(date +%T)"
done
echo "SWEEP COMPLETE $(date +%T)"
