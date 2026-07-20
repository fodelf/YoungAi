#!/bin/sh
# gen_residual_cluster.sh — dual-host parallel go1b sparse residual generation.
#
# Each host runs emit_residual over the layer range whose HF shards it holds
# LOCALLY (M4 coordinator = shards for layers 0-11; M1 worker = all shards), so
# both read local SSD (fast) instead of one host pulling everything over NFS
# (~10x slower). The two partial residual GGUFs are then merged on M4.
#
# Usage: gen_residual_cluster.sh ACTIVE_EXPERTS_FILE OUT_GGUF
#   ACTIVE_EXPERTS_FILE  per-layer "L{n}: e0 e1 ..." Go-active expert lists
#   OUT_GGUF             merged sidecar written on M4 (this host)
set -e

ACTIVE=${1:?usage: gen_residual_cluster.sh ACTIVE_EXPERTS_FILE OUT_GGUF}
OUT=${2:?usage: gen_residual_cluster.sh ACTIVE_EXPERTS_FILE OUT_GGUF}

GO=/Users/fodelf/git/ds4-main/gguf-tools/go-onebit
M4_HF=/Users/fodelf/git/ds4-main/hf/DeepSeek-V4-Flash-Base
M1_HOST=192.168.1.2
M1_GO=/Users/fodelf/ds4-main/gguf-tools/go-onebit
M1_HF=/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base

M4_LAYERS=$(seq -s, 0 11)     # layers whose HF shards live on M4 locally
M1_LAYERS=$(seq -s, 12 42)    # remaining layers, generated on M1 locally
ACTIVE_BASE=$(basename "$ACTIVE")

echo "[cluster] sync emit_residual sources + active file -> M1"
ssh "$M1_HOST" "mkdir -p $M1_GO"
scp -q "$GO/emit_residual.c" "$GO/hf_read.c" "$GO/hf_read.h" \
       "$GO/onebit_quant.c" "$GO/onebit_quant.h" "$ACTIVE" "$M1_HOST:$M1_GO/"
ssh "$M1_HOST" "cd $M1_GO && cc -O2 -o emit_residual emit_residual.c hf_read.c onebit_quant.c"

echo "[cluster] launch M1 (layers 12-42, local HF) in background"
ssh "$M1_HOST" "cd $M1_GO && ./emit_residual --hf $M1_HF --out /tmp/res_m1.gguf \
    --layers $M1_LAYERS --active-experts $M1_GO/$ACTIVE_BASE" &
M1PID=$!

echo "[cluster] generate M4 (layers 0-11, local HF) here"
"$GO/emit_residual" --hf "$M4_HF" --out /tmp/res_m4.gguf \
    --layers "$M4_LAYERS" --active-experts "$ACTIVE"

echo "[cluster] wait for M1 ..."
wait "$M1PID"

echo "[cluster] copy M1 partial -> M4, merge"
scp -q "$M1_HOST:/tmp/res_m1.gguf" /tmp/res_m1.gguf
"$GO/merge_residual" --out "$OUT" /tmp/res_m4.gguf /tmp/res_m1.gguf
echo "[cluster] done -> $OUT"
