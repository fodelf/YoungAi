#!/bin/sh
# cap_ef2.sh — EF round-2 engine-trajectory capture, merged with the R3-d
# post-training corpus: student = Θ_fix + s1 ship sidecar, corpus = fresh
# go_mixed slice (harvested repos+issues, dual-chain routed). One batch prefill
# through the engine writes raw_{ffn_in,route,route_logits,route_w}_L* shards.
#   usage: cap_ef2.sh [out_dir] [corpus] [corr]
# Memory: engine capture is the ONLY big process (capture-class discipline);
# watchdog via --mem-budget-mb. Disk: ~11-12 GB for a 14k-token slice.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 1
OUT=${1:-cap_ef2}
CORPUS=${2:-gguf-tools/data/corpus/build/go_mixed_14k.txt}
CORR=${3:-gguf/ds4-go1b-corr-s1.gguf}   # '-' = bare student (round-1 capture)
[ -f "$CORPUS" ] || { echo "no corpus: $CORPUS" >&2; exit 1; }
[ "$CORR" = "-" ] || [ -f "$CORR" ] || { echo "no corr: $CORR" >&2; exit 1; }
CORR_ARGS="--corr $CORR"; [ "$CORR" = "-" ] && CORR_ARGS=""
mkdir -p "$OUT"
DISK_MIN_GB=${DISK_MIN_GB:-15}   # 25层全采~12G; 深层18层~5G 可降门 (DISK_MIN_GB=8)
df -g . | tail -1 | awk -v m="$DISK_MIN_GB" '{ if ($4+0 < m) { print "DISK-GUARD: <" m "G free, refuse capture"; exit 1 } }' || exit 1
# EF2 scope: shallow half + refresh (L0-24) — the not-yet-engine-calibrated
# base22 layers; deep half (L25-42) already judged decaying-value. ~12 GB.
# ★env 大扫除 2026-08-31: 层过滤 env(原 DS4_CAP_LAYERS)已从引擎删除, 且 --cap-layers
# flag 尚未接线(core_engine_api.c 有 setter 无 CLI 挂点) → 当前捕获=全 43 层, 盘账
# 从 ~12 GB 涨到 ~20 GB(DISK_MIN_GB 请按需调大)。恢复 L0-24 过滤要等 flag 接线。★
echo "cap_ef2: corpus=$CORPUS corr=$CORR out=$OUT layers=ALL(过滤 flag 未接线, 见头注) (progress -> stderr)" >&2
CTX=${CTX:-32768}   # 显式内存安全栏（ctx 直接定 context-buffer 大小）；只预检不自动膨胀
EST_TOK=$(( $(wc -c < "$CORPUS") / 3 ))
[ "$EST_TOK" -le "$CTX" ] || { echo "CORPUS-TOO-LONG: ~${EST_TOK}tok > ctx $CTX（加 CTX= 显式放大或裁语料）" >&2; exit 1; }
MODEL=${MODEL:-gguf/ds4-go1b.gguf}
./ds4 -m "$MODEL" $CORR_ARGS --cap-dir "$OUT" \
    --prompt-file "$CORPUS" -n 1 --temp 0 --metal --ctx "$CTX"
rc=$?
ls "$OUT" | wc -l | awk '{print "cap_ef2: " $1 " shards"}' >&2
echo "CAP-EF2-DONE rc=$rc"
