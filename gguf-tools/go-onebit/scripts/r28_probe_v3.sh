#!/bin/bash
# r28_probe_v3.sh — v2 档 vs v3 档 同口径隔离探针(2026-08-01 用户令"打两个探针一浅一深")。
#
# 口径: DS4_MV_PROBE_L 隔离探针 — 上游走 FP 锚直通, 只评本层 ⇒ 干净的单层判决, 无上游
# 漂移污染, 两档可直接比。产物写临时目录, 不碰正式 layers。
#
# 探针对象(v3 变化最大的两层):
#   浅层 L05: v2=v12x256 h29  →  v3=v8x256 h22   (难度 Δ+0.0708, 高难度带)
#   深层 L41: v2=v32x256 h8   →  v3=v8x256 h25   (难度 Δ+0.0993, 全模型第二难, 升档最大)
#
# 用法: r28_probe_v3.sh [L05|L41|all]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
G7="$ROOT/gguf/go-onebit/g7"
TMP="$ROOT/gguf/go-onebit/probe_v3"
LOG(){ echo "[probe $(date +%H:%M:%S)] $*" >&2; }

WDOG(){ while true; do
    P=$(pgrep -nf "ds4quant_run .*rr_calib" || true); [ -n "$P" ] || { sleep 5; continue; }
    MB=$(footprint -p "$P" 2>/dev/null | grep -Eo 'Footprint: *[0-9.]+ *[KMG]B' | head -1 \
         | awk '{v=$2;u=$3; if(u=="GB")v*=1024; else if(u=="KB")v/=1024; printf "%d",v}' || true)
    [ -n "${MB:-}" ] && [ "$MB" -gt 11900 ] && { echo "[probe][wdog] ${MB}MB >11.9G 杀" >&2; kill -9 "$P" 2>/dev/null || true; }
    sleep 5; done }

probe(){
    local L=$1 DIM=$2 NC=$3 HOT=$4 W2D=$5 W2N=$6 TAG=$7
    local OUT="$TMP/L${L}_${TAG}"
    rm -rf "$OUT"; mkdir -p "$OUT/layers" "$OUT/ckpt"
    printf 'L=%d dim=%d nc=%d hot=%d w2dim=%d w2nc=%d\n' "$L" "$DIM" "$NC" "$HOT" "$W2D" "$W2N" > "$OUT/rplan.txt"
    cd "$ROOT/gguf-tools/go-onebit/quant"
    LOG "L${L} ${TAG}: vq${DIM}x${NC} h${HOT} w2 vq${W2D}x${W2N}"
    env DS4_ANCHOR="$G7/ds4quant_anchor_v5mini_s1716.bin" DS4_NFIT=933 DS4_THREADS="${DS4_THREADS:-6}" \
        DS4_MINVOL=1 DS4_MV_BASELINE=1 DS4_MV_COAD_BASE=1 DS4_TUNE=1 DS4_COADAPT=1 \
        DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_VQ_RPLAN="$OUT/rplan.txt" \
        DS4_MV_PROBE_L="$L" \
        DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="$ROOT/gguf-tools/go-onebit/corpus/prog_active_top64.txt" \
        DS4_PLAN="$OUT/plan.txt" DS4_CKPT_DIR="$OUT/ckpt" DS4_LAYER_DIR="$OUT/layers" \
        DS4_ZFILE="$OUT/zfile.bin" DS4_ZCHAIN="$OUT/zchain.bin" \
        ./ds4quant_run "$G7/rr_calib_prog_v5mini.ids" 1716 > "$OUT/probe.log" 2>&1
    LOG "L${L} ${TAG} rc=$?"
    grep -E "VQ_GATE L=0*${L} |^SEARCH L=${L} QT|L0*${L} ★体积★|L0*${L} ★贪心选" "$OUT/probe.log" | sed 's/ →.*//' >&2
    echo >&2
}

WDOG & WP=$!
trap 'kill $WP 2>/dev/null || true' EXIT

case "${1:-all}" in
  L05) probe 5 12 256 29 16 256 v2; probe 5 8 256 22 16 256 v3 ;;
  L41) probe 41 32 256 8 32 512 v2; probe 41 8 256 25 16 256 v3 ;;
  all) probe 5 12 256 29 16 256 v2; probe 5 8 256 22 16 256 v3
       probe 41 32 256 8 32 512 v2; probe 41 8 256 25 16 256 v3 ;;
  *) echo "用法: r28_probe_v3.sh [L05|L41|all]" >&2; exit 1 ;;
esac
LOG "探针收官 — 产物 $TMP"
