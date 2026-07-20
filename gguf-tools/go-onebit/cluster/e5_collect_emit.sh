#!/bin/sh
# e5_collect_emit.sh — collect M1's z files, emit the corr sidecar (φ=ŷ aware),
# verify round-trip, and place it next to the base model for auto-detection.
#   usage: [PHI=yhat] e5_collect_emit.sh [OUT=gguf/ds4-go1b-corr-rrr.gguf]
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
ROOT_M1=${ROOT_M1:-/Users/fodelf/ds4-main}
M1=${M1:-192.168.1.2}
ZDIR=${ZDIR:-$ROOT/zdump_rrr}
OUT=${1:-$ROOT/gguf/ds4-go1b-corr-rrr.gguf}
PHI=${PHI:-yhat}

scp -q "$M1:$ROOT_M1/zdump_rrr/z_L*.bin" "$ZDIR/" || true
N=$(ls "$ZDIR"/z_L*.bin 2>/dev/null | wc -l | tr -d ' ')
echo "z files: $N/43"
cd "$ROOT/gguf-tools" || exit 1
./emit_z --out "$OUT" --zdir "$ZDIR" --layers 43 --phi "$PHI" || exit 1
./emit_z --check "$OUT" --zdir "$ZDIR" || exit 1
ls -lh "$OUT"
echo "run: ./ds4 -m gguf/ds4-go1b.gguf --corr $OUT -p '<Go prompt>' -n 48 --temp 0 --metal"
