#!/bin/sh
# e2_sweep.sh — E2 aggregate-RRR ridge/feature sweep on one layer (M4 low
# layers / M1 high layers). Each row = one table line (TR/TE cos+rel).
# Evidence goal: does heavy ridge + full train pool + φ=ŷ lift HELD-OUT?
#   usage: e2_sweep.sh [LAYER=8] [NX=10240] [RANK=64]
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
GT=$ROOT/gguf-tools
L=${1:-8}
NX=${2:-10240}
RANK=${3:-64}
CAP=${CAP:-$ROOT/cap_m4}
HF=${HF:-$ROOT/hf/DeepSeek-V4-Flash-Base}
COMMON="--solver rrr --layers $L --nx $NX --rank $RANK --heldout-frac 0.17 --ntest 1024 --eig-iters 100 --threads 6 --lam-c 0.2 --cap $CAP --hf $HF"

cd "$GT" || exit 1
for cfg in \
    "--feat x    --lambda 1e-2" \
    "--feat x    --lambda 1e-1" \
    "--feat yhat --lambda 1e-2" \
    "--feat yhat --lambda 1e-1" \
    "--feat yhat --lambda 1e-1 --w-smooth 0.5" \
; do
    echo "=== L$L nx=$NX rank=$RANK :: $cfg ==="
    # shellcheck disable=SC2086
    ./calib_run $COMMON $cfg 2>&1 | grep -E "scalar-gain|TR base_cos|assembled" | tail -3
done
echo "=== E2-SWEEP-DONE L$L ==="
