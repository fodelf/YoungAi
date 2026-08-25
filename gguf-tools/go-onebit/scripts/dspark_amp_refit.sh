#!/bin/bash
# dspark_amp_refit.sh — drafter 放大器扩锚重解(2026-08-21)。
# 885 行/块时 k=384 顶格 ⇒ 欠拟合; 本脚本用双提示拼接的大锚(~2500 行/块)重解三块。
# 段: 等大锚 → 三块 fit(产物进 <D>/big) → 成链 zchain_drafter_amp_big.bin
set -uo pipefail
ROOT="$HOME/ds4-main"
D="$ROOT/gguf/go-onebit/r30/dspark"
HF="$ROOT/hf/DeepSeek-V4-Flash-DSpark"
STUDENT="$ROOT/gguf/ds4-dspark-drafter3-q2.gguf"
cd "$ROOT"
until [ -s "$D/anchor_big.bin" ]; do sleep 20; done
sleep 5
mkdir -p "$D/big"
for B in 0 1 2; do
    timeout 3600 python3 -u gguf-tools/go-onebit/zlever/dspark_amp_fit.py \
        "$HF" "$STUDENT" "$D/anchor_big.bin" "$D/big" "$B" 2>&1 | grep -aE "锚行|数据|★"
done
"$(dirname "$0")/../calib/zrec_to_zchain" "$D/big" "$D/zchain_drafter_amp_big.bin" 3 2>&1 | tail -1
ls -l "$D/zchain_drafter_amp_big.bin"
