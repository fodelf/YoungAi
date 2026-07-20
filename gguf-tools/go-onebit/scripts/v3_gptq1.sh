#!/bin/sh
# v3_gptq1.sh — 新模型 v3: 逐层 GPTQ-1bit 符号重写驱动 (跑在 M1, 有 HF+cap).
# 一层一层重写+判决, 每层一行 SUMMARY 进 /tmp/v3_gptq1.log; done-marker 可续跑.
# 用法:
#   sh v3_gptq1.sh probe L [N]   # 干跑 L 层前 N 个专家(默认16), 只出判决表不写文件
#   sh v3_gptq1.sh layer L       # 重写 L 层全部专家 (--apply)
#   sh v3_gptq1.sh all           # L6..L42 顺序全跑 (cap 缺 L0-5), 续跑跳过已完成层
set -u
ROOT=/Users/fodelf/ds4-main
export DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-Base"
GGUF="$ROOT/gguf/ds4-go1b-v3.gguf"
PY="$ROOT/gguf-tools/go-onebit/quant/gptq1_rewrite.py"
DONE="$ROOT/zdump/v3_gptq1_done"; mkdir -p "$DONE"
LOG=/tmp/v3_gptq1.log

capfor() { # ffn_in/route 路径: cap_v2r2 优先, 缺层回落 cap_v2r1_deep
  L=$1; K=$2
  for d in "$ROOT/cap_v2r2" "$ROOT/cap_v2r1_deep"; do
    [ -f "$d/${K}_L${L}.npy" ] && { echo "$d/${K}_L${L}.npy"; return 0; }
  done
  return 1
}

run_layer() { # $1=L $2=extra args
  L=$1; shift
  FI=$(capfor "$L" ffn_in) || { echo "L$L: no ffn_in cap, skip" | tee -a "$LOG"; return 1; }
  RT=$(capfor "$L" route)  || { echo "L$L: no route cap, skip"  | tee -a "$LOG"; return 1; }
  python3 -u "$PY" --gguf "$GGUF" --layer "$L" --ffn-in "$FI" --route "$RT" \
    --json "/tmp/v3_gptq1_L${L}.json" "$@" 2>&1 | tee "/tmp/v3_gptq1_L${L}.log" | \
    grep -E "SUMMARY|experts \(|skip" | tee -a "$LOG"
}

watchdog() { # 盯住所有重写进程 RSS, 单进程>3.5G 杀该进程 (物理预期 <2G; 多lane总量仍受此界)
  while :; do
    for P in $(pgrep -f gptq1_rewrite.py); do
      R=$(ps -o rss= -p "$P" 2>/dev/null | tr -d " ")
      [ -n "$R" ] && [ "$R" -gt 3670016 ] && { echo "WDKILL pid=$P rss=${R}KB" | tee -a "$LOG"; kill -9 "$P"; }
    done
    sleep 20
  done
}

case "${1:?usage: probe L [N] | layer L | all | pool [N]}" in
  probe)  run_layer "${2:?L}" --experts "${3:-16}" ;;
  layer)  watchdog & WD=$!; run_layer "${2:?L}" --apply; kill $WD 2>/dev/null ;;
  pool) # 机内 N-lane 并行: 逐层原子认领(mkdir), 不同层写不同偏移区, 层内确定性=重跑幂等
    N=${2:-3}
    rm -rf "$DONE"/claim_L*   # 单调度器假设: 清陈旧认领
    export VECLIB_MAXIMUM_THREADS=4
    watchdog & WD=$!
    for wkr in $(seq 1 "$N"); do
      (
        while :; do
          L=""
          for c in $(seq 6 42); do
            [ -f "$DONE/L$c" ] && continue
            mkdir "$DONE/claim_L$c" 2>/dev/null || continue
            L=$c; break
          done
          [ -z "$L" ] && break
          echo "=== L$L lane$wkr $(date +%H:%M:%S) ===" | tee -a "$LOG"
          run_layer "$L" --apply && touch "$DONE/L$L"
          rm -rf "$DONE/claim_L$L"
        done
      ) &
    done
    wait
    kill $WD 2>/dev/null
    ls "$DONE" | grep -c "^L" | xargs echo "POOL-DONE layers:" | tee -a "$LOG"
    echo "ALL-DONE $(date +%H:%M:%S)" | tee -a "$LOG"
    ;;
  all)
    watchdog & WD=$!
    for L in $(seq 6 42); do
      [ -f "$DONE/L$L" ] && { echo "L$L done, skip" | tee -a "$LOG"; continue; }
      echo "=== L$L $(date +%H:%M:%S) ===" | tee -a "$LOG"
      run_layer "$L" --apply && touch "$DONE/L$L"
    done
    kill $WD 2>/dev/null
    echo "ALL-DONE $(date +%H:%M:%S)" | tee -a "$LOG"
    ;;
esac
