#!/bin/sh
# Dual-machine BALANCED z compute orchestration (run on M4 coordinator).
# Honors "idle machine joins remaining work" by giving each host ~half the layers; bounded by
# shard locality (each host computes only layers whose expert shards it has LOCALLY — the M4<->M1
# NFS is unreliable). To let M4 do layers >12 it must hold those HF shards locally.
#
# Default split here: M4 -> layers 0-21 (needs shards 1-22 + cap ffn_in_L0-21), M1 -> 22-42.
# PREREQ to use this split (one-time, frees disk on M4 by removing its go1b GGUF — M1 keeps a copy):
#   rm -f $ROOT/gguf/ds4-go1b.gguf                              # M1 has a copy; recopy before the run
#   scp -p '$M1:.../model-0001[4-9]*' '$M1:.../model-0002[0-2]*' $ROOT/hf/DeepSeek-V4-Flash-Base/  # shards 14-22
#   scp -q '$M1:.../cap_m1/ffn_in_L1[3-9].npy' '$M1:.../cap_m1/ffn_in_L2[0-1].npy' $ROOT/cap_m4/   # cap 13-21
# (each shard ~6 GiB; watch M4 free space). If M4 errors on a layer (missing shard), copy that shard.
set -e
M1="${1:-192.168.1.2}"; ROOT="${2:-/Users/fodelf/git/ds4-main}"
HF4="$ROOT/hf/DeepSeek-V4-Flash-Base"; CAP4="$ROOT/cap_m4"; Z4="$ROOT/zdump"
HF1=/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base; CAP1=/Users/fodelf/ds4-main/cap_m1; Z1=/Users/fodelf/ds4-main/zdump
M4_LAYERS="$(seq -s, 0 21)"; M1_LAYERS="$(seq -s, 22 42)"; NX=256
D="$(dirname "$0")"
echo "M4 -> $M4_LAYERS"; sh "$D/calib_detached.sh" "$ROOT/gguf-tools" "$M4_LAYERS" "$CAP4" "$HF4" "$Z4" "$NX" /tmp/calib_z_m4.log
echo "M1 -> $M1_LAYERS"; ssh "$M1" "sh /tmp/calib_detached.sh /Users/fodelf/ds4-main/gguf-tools '$M1_LAYERS' '$CAP1' '$HF1' '$Z1' $NX /tmp/calib_z.log"
echo "launched. poll with: sh $D/poll_z.sh ; finish with: sh $D/emit_and_run.sh"
