#!/bin/bash
# draft86.sh — drafter 高精终判(2026-08-20 凌晨): allq2 基座 + DSpark drafter 全家 q8_0
# (mtp.= 前缀 override; q2/q4 drafter acc≈1 已判, 官方 fp16 参考 acc 46%/位 → 精度是最后变量)。
# 排队: 等 /tmp/b328.log 出 B328_DONE 再启动(GPU 让位 328)。
set -uo pipefail
ROOT="$HOME/ds4-main"
OUT="$ROOT/gguf/ds4-allq2d.gguf"
LOG(){ echo "[draft86 $(date +%H:%M:%S)] $*"; }

until grep -q B328_DONE /tmp/b328.log 2>/dev/null; do sleep 30; done
LOG "①重铸 allq2+drafter(q8 全家)"
[ -f "$OUT" ] || OUT="$OUT" bash "$ROOT/gguf-tools/go-onebit/scripts/quant_allq2_spark.sh" \
    --mtp-append 3 --tensor-type mtp.=q8_0 || { LOG "★重铸失败★"; exit 2; }
LOG "②投机 A/B (acc 统计)"
cd "$ROOT"
env DS4_DSPARK_SPEC=1 DS4_DSPARK_STAT=1 timeout 900 ./ds4 --cuda -m "$OUT" \
    --zchain gguf/go-onebit/r30/full86/zchain_noge.bin \
    --temp 0 -n 128 -p "Write a Python quicksort function." </dev/null 2>&1 \
    | grep -aE "t/s|dspark-stat|armed" | tail -6
LOG "③纯解码对照"
timeout 900 ./ds4 --cuda -m "$OUT" --zchain gguf/go-onebit/r30/full86/zchain_noge.bin \
    -n 128 -p "Write a Python quicksort function." </dev/null 2>&1 | grep -aE "t/s"
LOG "draft86 收官"
