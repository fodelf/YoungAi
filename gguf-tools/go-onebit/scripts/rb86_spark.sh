#!/bin/bash
# rb86_spark.sh — 路由偏置侧车战役(spark): 修基础量化 vs 开源 q2 的路由漂移差距。
# 依据: M10 神谕(钉 FP 路由 KL 0.4522→0.3823 全面超官方 q2 0.4207) — 差距全在路由。
# 链: ①捕 cal9 校准锚(M10 同域) ②B 回放 fit Δb(自由路由 vs FP 路由 margin 缺口)
#     ③wt2 盲判 α∈{0,2.5} 对比(0=基线复刻, 2.5=冠军定标)。
# 用法: bash rb86_spark.sh [anchor|fit|judge|all]
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
R30="$ROOT/gguf/go-onebit/r30"
G7="$ROOT/gguf/go-onebit/g7"
QBIN="$ROOT/gguf-tools/go-onebit/quant/ds4quant_run"
# ★2026-08-19 参数化(cal12 路由反修复用): RB_* env 可覆盖, 默认=原 cal9/en86 语义
CAL9_IDS="${RB_IDS:-$G7/wt2train_cal9.ids}"
CAL9_S="${RB_S:-2906}"
CAL9_ANCHOR="${RB_ANCHOR:-$R30/anchor_cal9_s2906.bin}"
LAYERS="${RB_LAYERS:-$R30/en86/layers}"
RB="${RB_OUT:-$R30/en86/route_bias_m10.bin}"
LCx=$(printf 'g%.0s' $(seq 1 43))
export DS4_HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-0731}"
export OPENBLAS_NUM_THREADS=1
LOG(){ echo "[rb86 $(date +%H:%M:%S)] $*" >&2; }

stage_anchor(){
    [ -f "$CAL9_ANCHOR" ] && { LOG "cal9 锚已在"; return 0; }
    cd "$ROOT/gguf-tools/go-onebit/quant"
    LOG "捕 cal9 锚 S=$CAL9_S (FP 前向)"
    DS4_FP_ONLY=1 DS4_ANCHOR="$CAL9_ANCHOR" DS4_THREADS=20 "$QBIN" "$CAL9_IDS" "$CAL9_S" \
        || { LOG "★锚捕获失败★"; exit 3; }
    [ -f "$CAL9_ANCHOR" ] || { LOG "★锚没落盘★"; exit 3; }
}

stage_fit(){   # B 回放 cal9, 自由路由(不设 ANCHOR_ROUTE), Δb 收官落盘
    cd "$ROOT/gguf-tools/go-onebit/quant"
    LOG "fit Δb (B 回放 cal9, 自由路由 vs 锚 FP 路由)"
    env -u DS4_ANCHOR_ROUTE \
        DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 DS4_CALIB_FULLSET=1 \
        DS4_EXPORT_BYTES=0 DS4_ANCHOR="$CAL9_ANCHOR" DS4_NFIT=1 DS4_THREADS=20 \
        DS4_LAYER_DIR="$LAYERS" DS4_LCFG="$LCx" DS4_VQ=1 DS4_TGT_ALPHA=1.0 \
        DS4_ROUTE_BIAS_FIT="$RB" \
        "$QBIN" "$CAL9_IDS" "$CAL9_S" 2>&1 | grep -E "ROUTE_BIAS|VERDICT|ops=" | tail -3
    [ -f "$RB" ] || { LOG "★Δb 没落盘★"; exit 4; }
    LOG "Δb ✓ $(ls -l "$RB" | awk '{print $5}') B"
}

judge_one(){   # $1=α (0=不挂 RB)
    local A="$1" ST="/tmp/rb86_wt2_a$1.bin" EXTRA=()
    [ "$A" != 0 ] && EXTRA=(DS4_ROUTE_BIAS="$RB" DS4_ROUTE_BIAS_ALPHA="$A" DS4_ROUTE_BIAS_MINCNT=8)
    cd "$ROOT/gguf-tools/go-onebit/quant"
    LOG "wt2 盲判 α=$A"
    env "${EXTRA[@]}" \
        DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 DS4_CALIB_FULLSET=1 \
        DS4_EXPORT_BYTES=0 DS4_ANCHOR="$R30/anchor_wt2_s2653.bin" DS4_NFIT=1 DS4_THREADS=20 \
        DS4_LAYER_DIR="$LAYERS" DS4_LCFG="$LCx" DS4_VQ=1 DS4_TGT_ALPHA=1.0 \
        DS4_DUMP_LOGITS="$ST" "$QBIN" "$G7/wt2.ids" 8000 2>&1 | tail -2
    cd "$ROOT"
    echo "══ wt2 五指标 α=$A ══"
    python3 "$SC/anchor_metrics.py" --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" --student "$ST" --tail 5
}

stage_judge(){ for A in ${RB_ALPHAS:-0 2.5}; do judge_one "$A"; done; }

case "${1:-all}" in
    anchor) stage_anchor ;;
    fit)    stage_fit ;;
    judge)  stage_judge ;;
    all)    stage_anchor && stage_fit && stage_judge ;;
    *) echo "用法: $0 [anchor|fit|judge|all]" >&2; exit 1 ;;
esac
