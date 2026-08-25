#!/bin/bash
# zlever 活rank求解驱动: 层 × 预算(avg rank)列表(结果落 reports/, 逐行无缓冲)
set -u
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
L="${1:?层号}"; NEXP="${2:-8}"; BUDGETS="${3:-32 48}"
OUT="$HERE/reports/L$(printf %02d $L)_alloc.txt"; : > "$OUT"
python3 -u "$HERE/solve.py" "$ROOT/hf/DeepSeek-V4-Flash-0731" \
  "$ROOT/gguf/go-onebit/r30/full/layers" "$ROOT/gguf/go-onebit/r30/anchor_r30_s1716.bin" \
  "$L" "$NEXP" "$BUDGETS" 2>/dev/null | tee -a "$OUT"
echo "完 → $OUT"
