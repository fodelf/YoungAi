#!/bin/bash
# dspark86_spark.sh — DSpark 投机模型 全量q2+放大器 战役(2026-08-21 用户令"这个也要反修放大器")。
# 设计=amp86 全复刻, 锚换 DSpark 自源(0731 锚不代表本 checkpoint 激活)。
# 段: quant(quant_allq2_spark.sh HF参数化, 外部已跑) → anchor(DSpark HF FP前向)
#     → solve(43层乘性放大器 ELM, zlayer DS4_ZL_GGUF + amp_solve GPU) → chain → size
# 产物: gguf/go-onebit/r30/dspark/ ; 模型 gguf/ds4-dspark-allq2.gguf
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
ZL="$ROOT/gguf-tools/go-onebit/zlever"
G7="$ROOT/gguf/go-onebit/g7"
OUT="$ROOT/gguf/go-onebit/r30/dspark"
MDL="$ROOT/gguf/ds4-dspark-allq2.gguf"
HF="$ROOT/hf/DeepSeek-V4-Flash-DSpark"
ANCHOR="$OUT/anchor_dspark_cal12z_s2048.bin"
IDS="$G7/cal12z.ids"
ZC="$OUT/zchain_dspark_amp.bin"
LOG(){ echo "[dspark86 $(date +%H:%M:%S)] $*"; }
mkdir -p "$OUT"
ST="${1:-all}"

stage_anchor(){
    [ -s "$ANCHOR" ] && { LOG "anchor 已在, 跳过"; return 0; }
    LOG "anchor: DSpark HF FP 前向 S=2048(cal12z.ids)"
    cd "$ROOT/gguf-tools/go-onebit/quant"
    env DS4_HF="$HF" DS4_FP_ONLY=1 DS4_ANCHOR="$ANCHOR" DS4_THREADS=20 OPENBLAS_NUM_THREADS=1 \
        ./ds4quant_run "$IDS" 2048 || { LOG "★锚捕获失败★"; exit 2; }
    [ -s "$ANCHOR" ] || { LOG "★锚没落盘★"; exit 2; }
}

stage_solve(){
    [ -s "$MDL" ] || { LOG "★模型缺 $MDL(先跑 quant)★"; exit 3; }
    LOG "solve: 43层乘性放大器(ELM闭式, GPU)"
    for L in $(seq 0 42); do
        REC="$OUT/zrec_L$(printf %02d $L).bin"
        [ -s "$REC" ] && { LOG "L$L 已解, 跳过"; continue; }
        ZCF="$OUT/zcache_L$(printf %02d $L).npz"
        [ -s "$ZCF" ] || env DS4_ZL_GGUF="$MDL" DS4_ZL_NTOK=2048 DS4_ZL_GE=0 DS4_ZL_FTA=0 \
            DS4_ZL_ERF=0 DS4_ZL_SWLIM=60 DS4_ZL_GATE=99 \
            python3 -u "$ZL/zlayer.py" "$HF" "$OUT" "$ANCHOR" $L 1024 0 >"$OUT/zl_L$L.out" 2>&1 \
            || { LOG "★L$L zcache 失败(见 $OUT/zl_L$L.out)★"; exit 3; }
        python3 -u "$ZL/amp_solve.py" "$ANCHOR" "$ZCF" "$REC" 1638 >"$OUT/amp_L$L.out" 2>&1 \
            || { LOG "★L$L 解算失败(见 $OUT/amp_L$L.out)★"; exit 3; }
        rm -f "$ZCF"
        LOG "L$L ✓"
    done
}

stage_chain(){
    LOG "chain: zrec → DQZ2"
    python3 "$ZL/zrec_to_zchain.py" "$OUT" "$ZC" 43 || { LOG "★成链失败★"; exit 4; }
}

stage_size(){
    LOG "═══ 体积账 ═══"
    ls -l "$MDL" | awk '{printf "模型(全量q2): %.2f GB\n", $5/1e9}'
    [ -s "$ZC" ] && ls -l "$ZC" | awk '{printf "放大器侧车: %.1f MB\n", $5/1e6}'
    [ -s "$MDL" ] && [ -s "$ZC" ] && ls -l "$MDL" "$ZC" | awk '{s+=$5} END{printf "合计落地: %.2f GB\n", s/1e9}'
}

case "$ST" in
    anchor) stage_anchor ;;
    solve)  stage_solve ;;
    chain)  stage_chain ;;
    size)   stage_size ;;
    all)    stage_anchor && stage_solve && stage_chain && stage_size ;;
    *) echo "用法: $0 [anchor|solve|chain|size|all]"; exit 1 ;;
esac
LOG "dspark86 $ST 收官"
