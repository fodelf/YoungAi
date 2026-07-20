#!/bin/sh
# z_finalize.sh — z 侧车收尾 (跑在 M4): 等 43 层 z bin 齐 → M1 emit_z 落盘
# gguf/sidecars/go-v3.gguf → 回传 M4 → nll_gate --corr 端到端判决 (bare 锚 3.4596)。
set -u
M1=192.168.1.2
M1ROOT=/Users/fodelf/ds4-main
ROOT=/Users/fodelf/git/ds4-main
SSH="ssh -o BatchMode=yes $M1"
ZD=$M1ROOT/zdump_v3

echo "[1] wait 43 z bins $(date +%H:%M:%S)"
while :; do
  N=$($SSH "ls $ZD 2>/dev/null | grep -c '^z_L.*bin'")
  echo "  z bins=$N/43 $(date +%H:%M:%S)"
  [ "$N" -ge 43 ] && break
  sleep 120
done

echo "[2] emit_z -> sidecar $(date +%H:%M:%S)"
$SSH "cd $M1ROOT/gguf-tools && mkdir -p $M1ROOT/gguf/sidecars && \
  ./emit_z --out $M1ROOT/gguf/sidecars/go-v3.gguf --zdir $ZD --layers 43 && \
  ls -la $M1ROOT/gguf/sidecars/go-v3.gguf" || { echo "EMIT-FAILED"; exit 1; }

echo "[3] ship sidecar + corr NLL 判决 $(date +%H:%M:%S)"
mkdir -p "$ROOT/gguf/sidecars"
scp -o BatchMode=yes -q "$M1:$M1ROOT/gguf/sidecars/go-v3.gguf" "$ROOT/gguf/sidecars/go-v3.gguf" || exit 1
cd "$ROOT"
echo "bare 锚: 3.4596"
MODEL=gguf/ds4-go1b-v3.gguf sh gguf-tools/go-onebit/scripts/nll_gate.sh gguf/sidecars/go-v3.gguf 2>&1 | tail -1
echo "Z-FINALIZE-DONE $(date +%H:%M:%S)"
