#!/bin/bash
# rr_after_l9_repair.sh — rr 污染事故修复收尾(2026-07-28, M1):
# 等修复重发洗完 L9(VQ_GATE L=09) → 截停反修(保 L10-42 昨夜产物) → 干净 rr 双语料终判。
# env 口径 = emit rr(81.6 基线)同款解释族(HF/VQ/热表), 反修族由 rr_verdict.sh 自清扫兜底。
# VERDICT 回写 /tmp/v4_bf.log 供监控转达。
set -u
LOG=/tmp/v4_bf.log
ROOT=/Users/fodelf/ds4-main
MARK=$(grep -n "L0-L9 修复重发" "$LOG" | tail -1 | cut -d: -f1)
[ -n "$MARK" ] || { echo "[rr_after] 修复标记缺失" >&2; exit 2; }

until tail -n +"$MARK" "$LOG" | grep -q "VQ_GATE L=09"; do sleep 20; done
sleep 5
echo "[rr_after] L9 洗净, 截停反修 $(date +%T)" >> "$LOG"
pkill -f campaign_v4; pkill -f "quant_layer.sh backfit"; pkill -f "ds4quant_run /tmp/rr_calib"
sleep 3

export DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-Base"
export DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_GO2B_HOT=1
export DS4_GO2B_HOT_TABLE="$ROOT/gguf-tools/go-onebit/corpus/hot_v4.txt"
export DS4_CALIB_FULLSET=1 DS4_THREADS=8

for RRIDS in /tmp/rr_hard.ids:64 /tmp/rr_code.ids:305; do
  RRF="${RRIDS%%:*}"; RRN="${RRIDS##*:}"
  TAG=$(basename "$RRF" .ids)
  echo "[rr_after] 干净 rr $TAG S=$RRN 起跑 $(date +%T)" >> "$LOG"
  bash "$ROOT/gguf-tools/go-onebit/scripts/rr_verdict.sh" "$RRF" "$RRN" \
      > "/tmp/rr_final_${TAG}.txt" 2>&1
  grep -aE "VERDICT" "/tmp/rr_final_${TAG}.txt" >> "$LOG" \
      || echo "[rr_after] $TAG 无 VERDICT 行(见 /tmp/rr_final_${TAG}.txt)" >> "$LOG"
done
echo "RR_ALL_DONE $(date +%T)" >> "$LOG"
