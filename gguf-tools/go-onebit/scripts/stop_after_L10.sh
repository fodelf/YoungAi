#!/bin/bash
# stop_after_L10.sh — L10 侧车修复截停器(2026-07-28, M1)。
# 背景: 污染 rr 击杀截尾 dql_vq_L10(表槽指零洞), L0-L9 修复范围漏掉它 →
# 重发 backfit 跑到 VQ_GATE L=10 出现即截停(L0-L9 幂等重打磨顺带)。
set -u
LOG=/tmp/v4_bf.log
MARK=$(grep -c "" "$LOG")
until tail -n +"$MARK" "$LOG" | grep -q "VQ_GATE L=10"; do sleep 20; done
sleep 5
echo "[stop_after_L10] L10 重导出完成, 截停 $(date +%T)" >> "$LOG"
pkill -f campaign_v4; pkill -f "quant_layer.sh backfit"; pkill -f "ds4quant_run /tmp/rr_calib"
echo "L10_REPAIR_DONE" >> "$LOG"
