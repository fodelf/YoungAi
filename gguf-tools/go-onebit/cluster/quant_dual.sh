#!/bin/sh
# quant_dual.sh — service-registration cluster quantization, NO shared FS
# (user directive 2026-07-03: 双机集群量化用服务注册, 不要用 NFS).
#
# Registration: each host's capability record (which HF shards it holds, i.e.
# which LAYERS it can generate locally) lives in cluster/registry/<host>.cap;
# refresh_registry probes both hosts over ssh. Assignment reads the registry —
# add a third host by dropping in its record, no orchestrator change.
#
# Flow: assign layer ranges by shard locality -> both hosts run
# deepseek4-quantize --layers --manifest LOCALLY (local-disk HF reads, sparse
# holes for the peer share) -> peer packs its written ranges (quant_assemble
# pack) -> one scp -> coordinator splices (unpack) -> sha256 optional gate.
# Deterministic generation => assembled file is byte-identical to single-host.
#
# usage (on M4): quant_dual.sh OUT.gguf [EXTRA_QUANT_ARGS...]
#   env: TMPL=/tmp/tmpl_hdr.gguf IMATRIX= THREADS_M4=6 THREADS_M1=6
#        M4_LAYERS=0-11 M1_LAYERS=12-42 (defaults = shard locality registry)
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
ROOT_M1=${ROOT_M1:-/Users/fodelf/ds4-main}
M1=${M1:-192.168.1.2}
OUT=${1:?output gguf path}; shift
EXTRA=${*:-"--experts go1b"}
TMPL=${TMPL:-/tmp/tmpl_hdr.gguf}
TMPL_M1=${TMPL_M1:-/tmp/tmpl_hdr.gguf}
IMATRIX=${IMATRIX:-}
REG="$ROOT/gguf-tools/go-onebit/cluster/registry"
mkdir -p "$REG"

# ---- service registration: probe + record capabilities -----------------------
refresh_registry() {
    # local (M4): count local HF shards -> complete layers (shard k covers
    # layers ~ (k-1)*43/46; shards 1-13 => layers 0-11 complete)
    n4=$(ls "$ROOT"/hf/DeepSeek-V4-Flash-Base/model-*.safetensors 2>/dev/null | wc -l | tr -d ' ')
    f4=$(df -g "$ROOT" | tail -1 | awk '{print $4}')
    printf 'host=m4 shards=%s free_gb=%s layers=%s\n' "$n4" "$f4" \
        "$([ "$n4" -ge 13 ] && echo 0-11 || echo none)" > "$REG/m4.cap"
    n1=$(ssh -o BatchMode=yes "$M1" "ls $ROOT_M1/hf/DeepSeek-V4-Flash-Base/model-*.safetensors 2>/dev/null | wc -l" | tr -d ' ')
    f1=$(ssh -o BatchMode=yes "$M1" "df -g $ROOT_M1 | tail -1" | awk '{print $4}')
    printf 'host=m1 shards=%s free_gb=%s layers=%s\n' "$n1" "$f1" \
        "$([ "$n1" -ge 46 ] && echo 0-42 || echo partial)" > "$REG/m1.cap"
    cat "$REG"/*.cap >&2
}
refresh_registry

M4_LAYERS=${M4_LAYERS:-0-11}
# M1's share runs in ~4-layer CHUNKS (disk discipline: M1 keeps only one ~4G
# partial at a time — 19G free vs 33G whole-share; chunks are also the natural
# work-stealing granularity for a future third host).
M1_CHUNKS=${M1_CHUNKS:-"12-15 16-19 20-23 24-27 28-31 32-35 36-39 40-42"}
IM4=""; [ -n "$IMATRIX" ] && IM4="--imatrix $IMATRIX"
IM1=""; [ -n "$IMATRIX" ] && IM1="--imatrix /tmp/gostats_full.dat"

# Phase B: M1 producer loop in the background (both hosts quantize
# CONCURRENTLY — that is the whole point); spool caps M1 disk at ~2 parts.
echo "[quant_dual] launch M1 producer: chunks [$M1_CHUNKS]" >&2
ssh -o BatchMode=yes "$M1" "cd $ROOT_M1/gguf-tools && make deepseek4-quantize >/dev/null 2>&1; \
  nohup sh $ROOT_M1/gguf-tools/go-onebit/cluster/quant_producer.sh '$M1_CHUNKS' $EXTRA \
    < /dev/null > /tmp/quant_producer.log 2>&1 & disown; echo M1-PRODUCER-LAUNCHED" || exit 1

# Phase C: M4's own lane (foreground; its file is the splice base).
echo "[quant_dual] M4 lane: layers $M4_LAYERS" >&2
"$ROOT/gguf-tools/deepseek4-quantize" --hf "$ROOT/hf/DeepSeek-V4-Flash-Base" --template "$TMPL" \
    $EXTRA $IM4 --layers "$M4_LAYERS" --manifest /tmp/quant_m4.manifest \
    --out "$OUT" --overwrite --threads "${THREADS_M4:-6}" || exit 1
echo "[quant_dual] M4 lane done; splice loop" >&2

# Phase D: splice loop — pull each landed part, stream-splice, delete both
# sides (deletion = consume signal that unblocks the producer's next chunk).
SPOOL_M1=$ROOT_M1/quant_spool
for CH in $M1_CHUNKS; do
    until ssh -o BatchMode=yes "$M1" "[ -f $SPOOL_M1/part_$CH.gguf ]"; do
        if ! ssh -o BatchMode=yes "$M1" "pgrep -f quant_producer.sh >/dev/null || pgrep -f deepseek4-quantize >/dev/null"; then
            ssh -o BatchMode=yes "$M1" "[ -f $SPOOL_M1/part_$CH.gguf ]" && break
            echo "QUANT-DUAL-FAIL: producer died before chunk $CH (M1:/tmp/quant_producer.log)"; exit 1
        fi
        sleep 60
    done
    scp -o BatchMode=yes -q "$M1:$SPOOL_M1/part_$CH.manifest" "/tmp/part_$CH.manifest" || exit 1
    echo "[quant_dual] splice chunk $CH" >&2
    ssh -o BatchMode=yes "$M1" "python3 $ROOT_M1/gguf-tools/go-onebit/cluster/quant_assemble.py \
        pack $SPOOL_M1/part_$CH.manifest $SPOOL_M1/part_$CH.gguf /dev/stdout 2>/dev/null" \
      | python3 "$ROOT/gguf-tools/go-onebit/cluster/quant_assemble.py" unpack "/tmp/part_$CH.manifest" - "$OUT" \
        2>/dev/null | tail -1 || exit 1
    ssh -o BatchMode=yes "$M1" "rm -f $SPOOL_M1/part_$CH.gguf $SPOOL_M1/part_$CH.manifest" 2>/dev/null
done
echo "QUANT-DUAL-DONE $OUT"
