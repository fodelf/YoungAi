#!/bin/bash
# q2z_wt2_bench.sh — q2z 反修模型的 wikitext-2 通用域尺(与 unsloth 官方 q2 表同域对比)。
# 口径(08-14 终判协议): 连续 2653 tok 单流无 BOS; 对 FP 相对指标(KL/top1/PPL比)可与官方表比,
# PPL 绝对值不可比。官方 IQ2_XXS 86.7GB: KLD 0.4207 / top1 77.92 / PPL比 1.3575。
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
R30="$ROOT/gguf/go-onebit/r30"
LAYERS="$R30/full/layers"
QBIN="$ROOT/gguf-tools/go-onebit/quant/ds4quant_run"
ANCHOR="$R30/anchor_wt2_s2653.bin"
IDS="$ROOT/gguf/go-onebit/g7/wt2.ids"
NL=43
export DS4_HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-0731}"
export OPENBLAS_NUM_THREADS=1
LOG(){ echo "[wt2 $(date +%H:%M:%S)] $*"; }

if [ ! -f "$ANCHOR" ]; then
    LOG "wt2 FP 锚生成(2653 tok)"
    cd "$ROOT/gguf-tools/go-onebit/quant"
    DS4_FP_ONLY=1 DS4_ANCHOR="$ANCHOR" DS4_NFIT=2653 DS4_THREADS=20 \
        "$QBIN" "$IDS" 2653 || { LOG "★锚失败★"; exit 3; }
fi
LOG "学生回放(量化+z侧车)"
cd "$ROOT/gguf-tools/go-onebit/quant"
env -u DS4_TUNE -u DS4_MINVOL -u DS4_VQ_RPLAN -u DS4_ZCHAIN \
    DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 DS4_CALIB_FULLSET=1 \
    DS4_EXPORT_BYTES=0 DS4_ANCHOR="$ANCHOR" DS4_NFIT=2653 DS4_THREADS=20 \
    DS4_LAYER_DIR="$LAYERS" DS4_LCFG=$(printf 'g%.0s' $(seq 1 $NL)) \
    DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_DUMP_LOGITS=/tmp/q2z_wt2_student.bin \
    "$QBIN" "$IDS" 2653 2>&1 | grep -E 'ops=|VERDICT' | tail -3
cd "$ROOT"
"$(dirname "$0")/../calib/anchor_metrics" --ref "$ANCHOR" --ids "$IDS" \
    --student /tmp/q2z_wt2_student.bin --fit 2653 || true
LOG "wt2 尺收官"
