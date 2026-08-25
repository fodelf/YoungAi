#!/bin/bash
# wt2cal_spark.sh — cal10 开源语料战役驱动(2026-08-19 用户令"尺子全面跟开源一样, 不自定义;
# 数据可少; 量化/反修每层<60s")。校准=wikitext-2-raw train 官方头部原样 N token(build_cal10.py,
# llama.cpp imatrix 同款), 评测=wt2 test(不变)。复用参数化既有脚本, 本驱动只做 env 编排:
#   base86p_spark.sh(锚现造+平权量化+wt2裸判) → zside_base86p.sh(纯z反修, 80/20 切分) → 终判。
# 用法: bash wt2cal_spark.sh [all|quant|zside|judge]   N 覆盖: WT2CAL_N=1536
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
R30="$ROOT/gguf/go-onebit/r30"
G7="$ROOT/gguf/go-onebit/g7"
N="${WT2CAL_N:-2048}"; CUT=$((N*8/10))
# ★TAG 参数化(08-19 二令: cal11=Bartowski calibration_datav3 小语料高覆盖, 全文等距 8 窗):
#   WT2CAL_TAG=calv3_cal11 WT2CAL_SRC=.../calibration_datav3.txt WT2CAL_CHUNKS=8
TAG="${WT2CAL_TAG:-cal10}"
SRC="${WT2CAL_SRC:-$ROOT/gguf-tools/go-onebit/corpus/wiki.train.raw}"
CHUNKS="${WT2CAL_CHUNKS:-1}"
if [ "$TAG" = cal10 ]; then IDS="$G7/wt2train_cal10.ids"; OUT="$R30/wt2cal"; else IDS="$G7/${TAG}.ids"; OUT="$R30/$TAG"; fi
export Q86_IDS="$IDS" Q86_S=$N Q86_NFIT=$N
export Q86_ANCHOR="$R30/anchor_${TAG}_s${N}.bin" Q86_OUT="$OUT"
LOG(){ echo "[wt2cal $(date +%H:%M:%S)] $*"; }
ST="${1:-all}"

[ -f "$Q86_IDS" ] || { python3 "$SC/build_cal10.py" "$N" "$SRC" "$Q86_IDS" "$CHUNKS" || { LOG "★$TAG 语料生成失败★"; exit 2; }; }

if [ "$ST" = all ] || [ "$ST" = quant ]; then
    LOG "锚+量化+裸判(基线尺)"
    JUDGE_DUMP=/tmp/${TAG}_base_wt2.bin bash "$SC/base86p_spark.sh" || exit $?
fi
if [ "$ST" = all ] || [ "$ST" = zside ]; then
    LOG "纯z反修(fit=0:$CUT ev=$CUT:$N)"
    ZS_LAYERS="$Q86_OUT/layers" ZS_ANCHOR="$Q86_ANCHOR" ZS_NTOK=$N \
        ZS_FIT="0:$CUT" ZS_EV="$CUT:$N" bash "$SC/zside_base86p.sh" 4 || exit $?
fi
if [ "$ST" = all ] || [ "$ST" = zside ] || [ "$ST" = judge ]; then
    LOG "反修后终判"
    JUDGE_DUMP=/tmp/${TAG}_pz_wt2.bin bash "$SC/base86p_spark.sh" || exit $?
fi
LOG "战役收官"
