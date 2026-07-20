#!/bin/sh
# e5_pump.sh — M1 side of the two-stage E5 pipeline: run the shard-dependent
# ŷ ASSEMBLY only (--dump-sel) for each given layer, landing compact
# sel_L{n}.bin files (~550MB) in SPOOL for the M4 consumer to pull+solve.
# Atomic: sel files appear via rename, so the puller never sees partials.
#   usage (on M1): e5_pump.sh "23 24 ... 42"
set -u
ROOT=${ROOT:-/Users/fodelf/ds4-main}
CAP=${CAP:-$ROOT/cap_m1_v2}
SPOOL=${SPOOL:-$ROOT/sel_spool}
NX=${NX:-10240}
LAYERS=${1:?layer list}
mkdir -p "$SPOOL"
cd "$ROOT/gguf-tools" || exit 1
for L in $LAYERS; do
    [ -f "$SPOOL/sel_L$L.bin" ] && continue
    # ENOSPC 三连教训：盘紧时等待（消费者删 sel 会回吐空间），失败会级联死泳道
    while [ "$(df -g "$SPOOL" | tail -1 | awk '{print $4}')" -lt 3 ]; do
        echo "pump: <3G free, waiting for consumers to drain" >&2; sleep 60
    done
    ./calib_run --solver rrr --nx "$NX" --heldout-frac 0.17 --ntest 1024 --threads 3 \
        --dump-sel "$SPOOL" --layers "$L" --cap "$CAP" \
        --hf "$ROOT/hf/DeepSeek-V4-Flash-Base" || echo "PUMP-FAIL L$L"
done
echo "PUMP-DONE"
