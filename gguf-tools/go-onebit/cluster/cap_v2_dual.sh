#!/bin/sh
# cap_v2_dual.sh — teacher re-capture with the extended eight signals, split
# into N_PROC chunk-parallel processes on ONE host (chunks are independent
# 512-token sequences → row-concat merge is exact; see merge_caps.py).
# Run on the host that holds ALL HF shards (M1). Memory ≈ N_PROC × ~2 GiB.
#   usage: [N_PROC=2] [CHUNKS=24] [OUT_BASE=$ROOT/cap_m1_v2] cap_v2_dual.sh
# After both finish:
#   venv/bin/python pyfwd/merge_caps.py ${OUT_BASE}a ${OUT_BASE}b $OUT_BASE 43
set -u
ROOT=${ROOT:-/Users/fodelf/ds4-main}
CAPWORK=${CAPWORK:-$ROOT/cap_work}
HF=${HF:-$ROOT/hf/DeepSeek-V4-Flash-Base}
CORPUS=${CORPUS:-$CAPWORK/gocorpus_big.txt}
OUT_BASE=${OUT_BASE:-$ROOT/cap_m1_v2}
N_PROC=${N_PROC:-2}
CHUNKS=${CHUNKS:-24}

cd "$CAPWORK" || exit 1
PER=$((CHUNKS / N_PROC))
i=0
SFX="a b c d"
for s in $SFX; do
    [ $i -ge "$N_PROC" ] && break
    LO=$((i * PER)); HI=$(((i + 1) * PER))
    [ $((i + 1)) -eq "$N_PROC" ] && HI=$CHUNKS
    mkdir -p "${OUT_BASE}${s}"
    env DS4_HF="$HF" CORPUS="$CORPUS" OUTDIR="${OUT_BASE}${s}" \
        CHUNK_LO=$LO CHUNK_HI=$HI \
        nohup ./venv/bin/python capture_go_big.py > "/tmp/cap_v2${s}.log" 2>&1 &
    echo "lane $s: chunks [$LO,$HI) -> ${OUT_BASE}${s} (log /tmp/cap_v2${s}.log)"
    i=$((i + 1))
done
sleep 3
echo "capture procs: $(pgrep -f capture_go_big | wc -l | tr -d ' ')"
