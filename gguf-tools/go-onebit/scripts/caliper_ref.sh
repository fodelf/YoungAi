#!/bin/bash
# caliper_ref.sh — 校尺: 量化器参考前向对任意层件出 wt2 判决(2026-08-24 尺子事故)。
# 口径=base86p_spark.sh 裸判段逐字照抄; 二进制=ds4quant_run.old(08-22, 08-23 改动把 lfile 加载改坏待修)(历史对表 M10裸底0.4956/官方q2 0.4207 同尺)。
# 用法: caliper_ref.sh <层件目录> <输出logits> [线程=20]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LAYERS="$(realpath "${1:?层件目录}")"; OUT="${2:?输出logits}"; THR="${3:-20}"
R30="$ROOT/gguf/go-onebit/r30"; G7="$ROOT/gguf/go-onebit/g7"
N=$(ls "$LAYERS"/dql_vq_L*.bin 2>/dev/null | wc -l)
[ "$N" = 43 ] || { echo "层件不齐 $N/43" >&2; exit 2; }
export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=1073741824
LCx=$(printf "g%.0s" $(seq 1 43))
cd "$ROOT/gguf-tools/go-onebit/quant"
env DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-0731" OPENBLAS_NUM_THREADS=1 DS4_BF_MEMGB=8 \
    DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 DS4_CALIB_FULLSET=1 \
    DS4_EXPORT_BYTES=0 DS4_ANCHOR="$R30/anchor_wt2_s2653.bin" DS4_NFIT=1 DS4_THREADS="$THR" \
    DS4_LAYER_DIR="$LAYERS" DS4_LCFG="$LCx" DS4_VQ=1 DS4_TGT_ALPHA=1.0 \
    DS4_DUMP_LOGITS="$OUT" ./ds4quant_run.old "$G7/wt2.ids" 8000 2>&1 | tail -2
cd "$ROOT"
python3 gguf-tools/go-onebit/scripts/anchor_metrics.py --ref "$R30/anchor_wt2_s2653.bin" \
    --ids "$G7/wt2.ids" --student "$OUT" --tail 3
