#!/bin/bash
# r28_probe_v4.sh — v2 静态表 vs v4 双信号表 的同口径隔离探针(2026-08-01 用户令"验证")。
#
# 口径: DS4_MV_PROBE_L 隔离探针 — 上游走 FP 锚直通, 只评本层 ⇒ 无上游漂移污染, 两档可直比。
# 全武装(DS4_TUNE 菜单: z变量/四损失/感知/向后 + coadapt), 两档条件完全一致, 只换档位。
#
# 三组验证, 各回答一个具体问题:
#   L05  v12x256 h29 (620.7M)  vs  v32x256 h106 (847.3M)
#        → "把钱从冷档密度挪到热专家" 这个策略对不对(v4 的核心赌注)
#   L41  v32x256 h8  (267.4M)  vs  v32x256 h141(1054.5M)
#        → 难度信号是否有效: L41 难度全模型第二高, v2 却给了最稀档
#   L01  v8x256  h31 (779.6M)  vs  v32x256 h4  (243.7M)
#        → 欠配风险: v4 把它砍到 1/3 体积, 实测会掉多少(审计预测误差贡献仅 2.4%)
#
# 用法: r28_probe_v4.sh [L05|L41|L01|all]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
G7="$ROOT/gguf/go-onebit/g7"
TMP="$ROOT/gguf/go-onebit/probe_v4"
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
        DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_VQ_RPLAN="$OUT/rplan.txt" DS4_MV_PROBE_L="$L" \
        DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="$ROOT/gguf-tools/go-onebit/corpus/prog_active_top64.txt" \
        DS4_PLAN="$OUT/plan.txt" DS4_CKPT_DIR="$OUT/ckpt" DS4_LAYER_DIR="$OUT/layers" \
        DS4_ZFILE="$OUT/zfile.bin" DS4_ZCHAIN="$OUT/zchain.bin" \
        ./ds4quant_run "$G7/rr_calib_prog_v5mini.ids" 1716 > "$OUT/probe.log" 2>&1
    local rc=$?
    local SZ=0; [ -f "$OUT/layers/dql_vq_L$(printf %02d $L).bin" ] && SZ=$(stat -f %z "$OUT/layers/dql_vq_L$(printf %02d $L).bin")
    printf '%s %s %s %s\n' "L$L" "$TAG" "$(grep -oE 'held relL2=[0-9.]+' "$OUT/probe.log" | tail -1 | cut -d= -f2)" "$SZ" >> "$TMP/results.txt"
    LOG "L${L} ${TAG} rc=$rc"
    grep -E "VQ_GATE L=0*${L} |held relL2=|★贪心选" "$OUT/probe.log" | sed 's/ →.*//' | tail -3 >&2
    echo >&2
}

mkdir -p "$TMP"; : > "$TMP/results.txt"
WDOG & WP=$!
trap 'kill $WP 2>/dev/null || true' EXIT

case "${1:-all}" in
  L05) probe 5 12 256 29 16 256 v2;  probe 5 32 256 106 32 512 v4 ;;
  L41) probe 41 32 256 8 32 512 v2;  probe 41 32 256 141 32 512 v4 ;;
  L01) probe 1 8 256 31 16 256 v2;   probe 1 32 256 4 32 512 v4 ;;
  all) probe 5 12 256 29 16 256 v2;  probe 5 32 256 106 32 512 v4
       probe 41 32 256 8 32 512 v2;  probe 41 32 256 141 32 512 v4
       probe 1 8 256 31 16 256 v2;   probe 1 32 256 4 32 512 v4 ;;
  *) echo "用法: r28_probe_v4.sh [L05|L41|L01|all]" >&2; exit 1 ;;
esac
LOG "探针收官 — 汇总:"
cat "$TMP/results.txt" >&2
