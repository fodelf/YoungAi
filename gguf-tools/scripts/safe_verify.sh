#!/bin/sh
# safe_verify.sh — 单机安全跑全 mono 的 offload 前向(perplexity/gen)。固化既有 proven flags,
# 治 2026-07-06 那次内核 panic(我漏了 NO_RESIDENCY + 乱设 MEM_BUDGET + 没 bound ctx → wired 暴涨饿死 watchdogd)。
# proven 依据: go2b_product.sh(已删, 见 git 历史) do_verify / mono_dual_run.sh。外加外部内存看门狗兜底(内部预算闸已证挡不住)。
# 用法: safe_verify.sh [slice_file] [ctx]     slice 默认 go_heldout_48.txt(快); 定音用 go_heldout_300.txt
set -u
ROOT=/Users/fodelf/git/ds4-main
MONO=$ROOT/gguf/ds4-mono-mixed.gguf
SLICE=${1:-$ROOT/gguf-tools/data/corpus/go_heldout_48.txt}
CTX=${2:-4096}
LOG=/tmp/safe_verify.log; : > "$LOG"

# ---- 外部内存看门狗(引擎内已有同款守卫, 这是双保险): 用系统压力等级(1正常/2警告/4临界),
#      不用 free%(offload 下 page cache 填满 free% 本就低但可回收→会误杀)。持续 critical ~6s 才杀。----
( crit=0; while :; do
    lvl=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 1)
    if [ "$lvl" -ge 4 ]; then crit=$((crit+1)); else crit=0; fi
    [ "$crit" -ge 3 ] && { echo "WATCHDOG-KILL pressure=critical x$crit → 杀 ds4" >>"$LOG"; pkill -f "ds4 -m $MONO"; break; }
    sleep 2
  done ) &
WD=$!

echo "=== safe_verify $(date +%H:%M:%S) slice=$(basename "$SLICE") ctx=$CTX ===" | tee -a "$LOG"
# proven flags: --no-residency(不 wire buffer) + --prefill-chunk(分块) + bound ctx; 不设 mem budget
# (env 大扫除 2026-08-31: EXPERT_OFFLOAD=1 归 AUTO 判定已删; PREFILL_CHUNK 迁 --prefill-chunk;
  perl -e 'alarm 600; exec @ARGV' \
  "$ROOT/ds4" -m "$MONO" --no-residency --prefill-chunk 512 --ctx "$CTX" --perplexity-file "$SLICE" --metal 2>&1 | tee -a "$LOG"
rc=$?
kill $WD 2>/dev/null
echo "VERIFY-EXIT-$rc  还原度=(5.6185-NLL)/(5.6185-0.5522)x100 [教师锚0.5522/v2裸0%]" | tee -a "$LOG"
