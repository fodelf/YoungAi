#!/bin/sh
# obase_v3_m4.sh — M4 双 lane: 从 v3 部署字节逐层产学生 obase (z 四损失解的 ŷ 输入).
# cap 三件套在 /tmp/cap_v3prep (由 M1 ship), 输出 obase_v3_L{L}.npy 同目录.
# 用法: sh obase_v3_m4.sh LANE_ID   (开 2 个即双 lane)
set -u
ROOT=/Users/fodelf/git/ds4-main
CAP=/tmp/cap_v3prep
DONE=/tmp/obase_v3_done; mkdir -p "$DONE"
LANE=${1:-1}
SWLIM=10.0   # HF config swiglu_limit 实值 (勿用默认)
LOG=/tmp/obase_v3_m4.log

wd() {
  while :; do
    for P in $(pgrep -f "obase_v3.py"); do
      R=$(ps -o rss= -p "$P" 2>/dev/null | tr -d " ")
      [ -n "$R" ] && [ "$R" -gt 3670016 ] && { echo "OBASE-WDKILL pid=$P rss=${R}KB" | tee -a "$LOG"; kill -9 "$P"; }
    done
    sleep 20
  done
}
[ "$LANE" = 1 ] && { wd & WD=$!; trap 'kill $WD 2>/dev/null' EXIT; }

while :; do
  L=""
  for c in $(seq 6 42); do
    [ -f "$DONE/L$c" ] && continue
    [ -f "$CAP/ffn_in_L$c.npy" ] && [ -f "$CAP/route_w_L$c.npy" ] || continue  # ship 未到先跳过
    mkdir "$DONE/claim_L$c" 2>/dev/null || continue
    L=$c; break
  done
  if [ -z "$L" ]; then
    N=$(ls "$DONE" | grep -c "^L")
    [ "$N" -ge 37 ] && { echo "obase lane$LANE: all done ($N)" | tee -a "$LOG"; break; }
    sleep 30; continue   # ship 还在路上
  fi
  echo "=== obase L$L lane$LANE $(date +%H:%M:%S) ===" | tee -a "$LOG"
  python3 -u "$ROOT/gguf-tools/go-onebit/quant/obase_v3.py" \
    --gguf "$ROOT/gguf/ds4-go1b-v3.gguf" --layer "$L" \
    --ffn-in "$CAP/ffn_in_L$L.npy" --route "$CAP/route_L$L.npy" --route-w "$CAP/route_w_L$L.npy" \
    --out "$CAP/obase_v3_L$L.npy" --swlim "$SWLIM" > "/tmp/obase_L$L.log" 2>&1 \
    && { grep -a OBASE-OK "/tmp/obase_L$L.log" | tee -a "$LOG"; touch "$DONE/L$L"; } \
    || { echo "obase L$L FAILED" | tee -a "$LOG"; tail -3 "/tmp/obase_L$L.log" | tee -a "$LOG"; }
  rm -rf "$DONE/claim_L$L"
done
