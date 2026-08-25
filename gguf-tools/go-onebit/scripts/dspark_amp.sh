#!/bin/bash
# dspark_amp.sh — DSpark drafter 自己的反修放大器(用户四文件设计之第4件)。
# 四文件: ①gguf/ds4-allq2.gguf(全量q2主模型) ②主放大器 zchain ③drafter q2 gguf
#         ④本脚本产的 drafter 放大器 zchain(引擎 DS4_DRAFT_ZCHAIN 挂载)
#
# 段:
#   anchor — 引擎在**部署态**(q2 drafter + 真实投机流)捕获 drafter FFN 的
#            (Fin, 路由 idx/权重); DS4_DSPARK_ANCHOR 落盘。用学生捕获而非教师, 保证
#            解出来的修正贴的是真实推理分布。
#   solve  — dspark_amp_fit.py 按同一 Fin/路由离线重算教师(HF mxfp4)与学生(q2 gguf)
#            的 routed 输出, 差值上解乘性 ELM 闭式(zl.AMP/type7), 每 mtp 块一份 zrec。
#   chain  — zrec_to_zchain.py 合成 3 层链。
#   test   — 引擎四文件同挂, 报 acc/t/s(有无 drafter 放大器 A/B)。
#
# 用法: bash dspark_amp.sh [all|anchor|solve|chain|test]
set -uo pipefail
ROOT="$HOME/ds4-main"
OUT="$ROOT/gguf/go-onebit/r30/dspark"
MAIN="$ROOT/gguf/ds4-allq2.gguf"
MAIN_ZC="$ROOT/gguf/go-onebit/r30/full86/zchain_noge.bin"
STUDENT="$ROOT/gguf/ds4-dspark-drafter3-q2.gguf"
HF="$ROOT/hf/DeepSeek-V4-Flash-DSpark"
ANCHOR="$OUT/drafter_anchor.bin"
ZC="$OUT/zchain_drafter_amp.bin"
PROMPT="${PROMPT:-Write a Python function that reverses a string.}"
LOG(){ echo "[dspark_amp $(date +%H:%M:%S)] $*"; }
mkdir -p "$OUT"
cd "$ROOT"

stage_anchor(){
    [ -s "$ANCHOR" ] && { LOG "anchor 已在($(du -h "$ANCHOR" | cut -f1)), 跳过"; return 0; }
    LOG "anchor: 部署态捕获(q2 drafter, SPEC 开)"
    env DS4_DRAFT_GGUF="$STUDENT" DS4_DSPARK_SPEC=1 DS4_DSPARK_ANCHOR="$ANCHOR" \
        timeout 1800 ./ds4 --cuda -m "$MAIN" --zchain "$MAIN_ZC" --temp 0 -n 400 \
        -p "Write a Python function that reverses a string, then explain how it works and give three usage examples." \
        </dev/null 2>&1 | grep -aE "generation|avg_acc" | tail -2
    [ -s "$ANCHOR" ] || { LOG "★锚没落盘★"; exit 2; }
}

stage_solve(){
    for B in 0 1 2; do
        REC="$OUT/zrec_L0$B.bin"
        [ -s "$REC" ] && { LOG "block $B 已解, 跳过"; continue; }
        LOG "solve block $B"
        timeout 3600 python3 "$ROOT/gguf-tools/go-onebit/zlever/dspark_amp_fit.py" \
            "$HF" "$STUDENT" "$ANCHOR" "$OUT" "$B" || { LOG "★block $B 解算失败★"; exit 3; }
    done
}

stage_chain(){
    LOG "chain: 3 层 drafter 链"
    python3 "$ROOT/gguf-tools/go-onebit/zlever/zrec_to_zchain.py" "$OUT" "$ZC" 3 || exit 4
    ls -l "$ZC"
}

stage_test(){
    [ -s "$ZC" ] || { LOG "★缺 $ZC★"; exit 5; }
    for MODE in with without; do
        if [ "$MODE" = with ]; then EXTRA=(env DS4_DRAFT_ZCHAIN="$ZC"); else EXTRA=(env); fi
        printf "%-8s " "$MODE"
        "${EXTRA[@]}" DS4_DRAFT_GGUF="$STUDENT" DS4_DSPARK_SPEC=1 DS4_DSPARK_STAT=1 \
            timeout 900 ./ds4 --cuda -m "$MAIN" --zchain "$MAIN_ZC" --temp 0 -n 96 -p "$PROMPT" \
            </dev/null 2>&1 | grep -aE "avg_acc|generation|draft zchain" | tail -3 | tr '\n' ' ' | sed 's/ds4: //g'
        echo
    done
}

case "${1:-all}" in
    anchor) stage_anchor ;;
    solve)  stage_solve ;;
    chain)  stage_chain ;;
    test)   stage_test ;;
    all)    stage_anchor && stage_solve && stage_chain && stage_test ;;
    *) echo "用法: $0 [all|anchor|solve|chain|test]"; exit 1 ;;
esac
LOG "收官"
