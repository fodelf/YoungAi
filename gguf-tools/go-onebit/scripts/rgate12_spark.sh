#!/bin/bash
# rgate12_spark.sh — 四支柱③ gate(x) 动态路由门 cal12 战役(2026-08-19 用户令"路由反修
# 要每层动态非写死α")。链: 校准链锚(已捕) → rgate.py solve 43层闭式 → wt2 链锚捕获 →
# patch 预测路由入判决锚 → DS4_ANCHOR_ROUTE=1 终判(可部署信息, 零FP偷看)。
# 对表: cal12+z 部署态 1.3894/0.7828/0.4747/79.1 | 神谕 1.3478/0.8029/0.3771/81.0 | 官方q2 0.4207。
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
R30="$ROOT/gguf/go-onebit/r30"
G7="$ROOT/gguf/go-onebit/g7"
Q="$ROOT/gguf-tools/go-onebit/quant"
LC=$(printf 'g%.0s' $(seq 1 43))
export DS4_HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-0731}" OPENBLAS_NUM_THREADS=1
LOG(){ echo "[rgate12 $(date +%H:%M:%S)] $*"; }

CH_CAL="$R30/cal12/chain_cal12z_s2048.bin"
CH_WT2="$R30/cal12/chain_wt2_s2653.bin"
RGD="$R30/cal12/rgate"
PATCHED="$R30/cal12/anchor_wt2_rgate.bin"

[ -f "$CH_CAL" ] || { LOG "★校准链锚缺 $CH_CAL — 先跑链态捕获★"; exit 2; }
LOG "solve: 43 层闭式 gate(x) (fit 0:1638, ev 1638:2048)"
python3 "$ROOT/gguf-tools/go-onebit/zlever/rgate.py" solve \
    "$R30/anchor_cal12z_s2048.bin" "$CH_CAL" "$RGD" 2048 1638 1638 2048 || { LOG "★solve 失败★"; exit 3; }

if [ ! -f "$CH_WT2" ]; then
    LOG "wt2 链锚捕获(部署链态, ~17min)"
    cd "$Q"
    env DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 DS4_CALIB_FULLSET=1 \
        DS4_EXPORT_BYTES=0 DS4_ANCHOR="$R30/anchor_wt2_s2653.bin" DS4_NFIT=1 DS4_THREADS=20 \
        DS4_LAYER_DIR="$R30/cal12/layers" DS4_LCFG="$LC" DS4_VQ=1 DS4_TGT_ALPHA=1.0 \
        DS4_CHAIN_ANCHOR="$CH_WT2" ./ds4quant_run "$G7/wt2.ids" 8000 2>&1 | tail -2
    cd "$ROOT"
    [ -f "$CH_WT2" ] || { LOG "★wt2 链锚没落盘★"; exit 4; }
fi

LOG "patch: 预测路由(纯链态信息)写入判决锚"
python3 "$ROOT/gguf-tools/go-onebit/zlever/rgate.py" patch \
    "$R30/anchor_wt2_s2653.bin" "$CH_WT2" "$RGD" "$PATCHED" 2653 || { LOG "★patch 失败★"; exit 5; }

LOG "终判: ANCHOR_ROUTE=1 × 预测路由锚"
cd "$Q"
env DS4_ANCHOR_ROUTE=1 DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 DS4_CALIB_FULLSET=1 \
    DS4_EXPORT_BYTES=0 DS4_ANCHOR="$PATCHED" DS4_NFIT=1 DS4_THREADS=20 \
    DS4_LAYER_DIR="$R30/cal12/layers" DS4_LCFG="$LC" DS4_VQ=1 DS4_TGT_ALPHA=1.0 \
    DS4_DUMP_LOGITS=/tmp/rgate12_wt2.bin ./ds4quant_run "$G7/wt2.ids" 8000 2>&1 | tail -2
cd "$ROOT"
echo "══ rgate12 终判五指标(对表: 部署态0.4747 | 神谕0.3771 | 官方q2 0.4207) ══"
"$(dirname "$0")/../calib/anchor_metrics" --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" \
    --student /tmp/rgate12_wt2.bin --tail 5
LOG "战役收官"
