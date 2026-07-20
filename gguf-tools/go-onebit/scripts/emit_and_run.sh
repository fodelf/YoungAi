#!/bin/sh
# Final z integration (run on M4 after the dual z compute finishes all 43 layers):
#   1. collect M1's z dumps -> M4 zdump   2. emit corr sidecar   3. ensure go1b model present   4. run ds4 --corr
# Args: $1=M1_host(192.168.1.2) $2=ROOT(/Users/fodelf/git/ds4-main) $3=PROMPT
M1="${1:-192.168.1.2}"; ROOT="${2:-/Users/fodelf/git/ds4-main}"
PROMPT="${3:-Write a Go function that adds two integers.}"
# 1. gather M1's z (13-42 / 22-42) into M4 zdump
scp -p "$M1:/Users/fodelf/ds4-main/zdump/z_L*.bin" "$ROOT/zdump/" 2>/dev/null
N=$(ls "$ROOT"/zdump/z_L*.bin 2>/dev/null | wc -l | tr -d ' ')
echo "z files: $N/43"; [ "$N" -lt 43 ] && echo "WARNING: missing z layers (emit will skip them)"
# 2. emit corr sidecar (~90 MiB; does not touch the 45.6 GiB main model)
cd "$ROOT/gguf-tools" && ./emit_z --out "$ROOT/gguf/ds4-go1b-corr.gguf" --zdir "$ROOT/zdump" --layers 43 || exit 1
# 3. ensure the go1b model is present on M4 (it may have been freed to make disk for the dual z balance)
if [ ! -f "$ROOT/gguf/ds4-go1b.gguf" ]; then
  echo "recopy go1b model from $M1 ..."; scp -p "$M1:/Users/fodelf/ds4-main/gguf/ds4-go1b.gguf" "$ROOT/gguf/"
fi
# 4. run (auto-detects the sidecar next to -m, or pass --corr explicitly). Use repeat penalty vs greedy loops.
cd "$ROOT"
echo "=== ds4 go1b + z (--corr) ==="
./ds4 -m gguf/ds4-go1b.gguf --corr gguf/ds4-go1b-corr.gguf -p "$PROMPT" -n 48 --temp 0 --metal
