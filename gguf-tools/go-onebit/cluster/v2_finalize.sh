#!/bin/sh
# v2_finalize.sh — after the two chunk-parallel captures finish on M1:
#   1. merge cap_m1_v2a + cap_m1_v2b -> cap_m1_v2 (row-concat, exact)
#   2. scp the M4-solvable layer slices (L0-12: ffn_in/route/route_w/routed)
#      into a v2 cap dir on M4 so the E5 M4 lane solves on the same gold.
# Run from M4.  usage: v2_finalize.sh [M4_CAP_OUT=$ROOT/cap_m4_v2]
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
ROOT_M1=${ROOT_M1:-/Users/fodelf/ds4-main}
M1=${M1:-192.168.1.2}
M4_CAP=${1:-$ROOT/cap_m4_v2}

echo "== merge on M1 =="
ssh "$M1" "cd $ROOT_M1/cap_work && ./venv/bin/python $ROOT_M1/gguf-tools/go-onebit/pyfwd/merge_caps.py \
    $ROOT_M1/cap_m1_v2a $ROOT_M1/cap_m1_v2b $ROOT_M1/cap_m1_v2 43" | tail -3 || exit 1

echo "== slice L0-12 -> $M4_CAP =="
mkdir -p "$M4_CAP"
L=0
while [ $L -le 12 ]; do
    for k in ffn_in route route_w routed; do
        scp -q "$M1:$ROOT_M1/cap_m1_v2/${k}_L${L}.npy" "$M4_CAP/" || exit 1
    done
    echo "L$L ok"
    L=$((L + 1))
done
scp -q "$M1:$ROOT_M1/cap_m1_v2/final_topk_idx.npy" "$M1:$ROOT_M1/cap_m1_v2/final_topk_val.npy" "$M4_CAP/" 2>/dev/null || true
df -h "$ROOT" | tail -1
echo "V2-FINALIZE DONE"
