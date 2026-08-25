#!/bin/bash
# rsolve43.sh — 路由闭式侧车 43 层解算发射器(route_solve.py 驱动, 4-lane)。
set -uo pipefail
ROOT="$HOME/ds4-main"
ZL="$ROOT/gguf-tools/go-onebit/zlever"
OUT="${RS_OUT:-$ROOT/gguf/go-onebit/r30/route86}"
CAP="${RS_CAP:-/tmp/cap_allq2}"
ANC="${RS_ANC:-$ROOT/gguf/go-onebit/r30/anchor_cal12z_s2048.bin}"
HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-0731}"
mkdir -p "$OUT"
seq 0 42 | OPENBLAS_NUM_THREADS=4 xargs -P "${1:-4}" -I{} \
    python3 "$ZL/route_solve.py" "$HF" "$CAP" "$ANC" "$OUT" {} 1638
echo "[rsolve43] 收官: $(ls "$OUT"/zrec_route_L*.bin 2>/dev/null | wc -l)/43"
