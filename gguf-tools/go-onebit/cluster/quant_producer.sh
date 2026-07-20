#!/bin/sh
# quant_producer.sh — M1 side of the cluster split-quantize: produce layer
# chunks into a local spool with at most MAX_AHEAD unconsumed parts (disk
# discipline: each part ~3-4G, M1 keeps ≤2). The coordinator (quant_dual.sh
# on M4) splices a part into the base file and deletes it; deletion is the
# consume signal that unblocks the next produce.
#   usage (on M1): quant_producer.sh "12-15 16-19 ..." [EXTRA_QUANT_ARGS...]
#   env: TMPL=/tmp/tmpl_hdr.gguf IMATRIX=/tmp/gostats_full.dat THREADS=6
#        SPOOL=~/ds4-main/quant_spool
set -u
ROOT=${ROOT:-/Users/fodelf/ds4-main}
CHUNKS=${1:?chunk list like "12-15 16-19"}; shift
EXTRA=${*:-"--experts go1b"}
TMPL=${TMPL:-/tmp/tmpl_hdr.gguf}
IMATRIX=${IMATRIX:-/tmp/gostats_full.dat}
SPOOL=${SPOOL:-$ROOT/quant_spool}
MAX_AHEAD=${MAX_AHEAD:-2}
mkdir -p "$SPOOL"
IM=""; [ -n "$IMATRIX" ] && [ -f "$IMATRIX" ] && IM="--imatrix $IMATRIX"
cd "$ROOT/gguf-tools" || exit 1
for CH in $CHUNKS; do
    [ -f "$SPOOL/part_$CH.done" ] && continue    # resume: already produced+unconsumed or consumed marker handled by coordinator
    while [ "$(ls "$SPOOL"/part_*.gguf 2>/dev/null | wc -l | tr -d ' ')" -ge "$MAX_AHEAD" ]; do
        sleep 30
    done
    echo "producing chunk $CH" >&2
    ./deepseek4-quantize --hf "$ROOT/hf/DeepSeek-V4-Flash-Base" --template "$TMPL" \
        $EXTRA $IM --layers "$CH" --manifest "$SPOOL/part_$CH.manifest.tmp" \
        --out "$SPOOL/part_$CH.gguf.tmp" --overwrite --threads "${THREADS:-6}" \
        > "/tmp/quant_ch_$CH.log" 2>&1 || { echo "PRODUCE-FAIL $CH"; exit 1; }
    mv "$SPOOL/part_$CH.manifest.tmp" "$SPOOL/part_$CH.manifest"
    mv "$SPOOL/part_$CH.gguf.tmp" "$SPOOL/part_$CH.gguf"   # atomic land: gguf LAST
    echo "chunk $CH landed" >&2
done
echo "PRODUCER-DONE"
