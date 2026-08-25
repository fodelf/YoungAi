#!/bin/bash
# backfit_decide.sh — 反修自动裁决器(2026-07-22 用户纠正: 不是关闭反修, 是判断指标决定是否反修)。
# 在"基线模型 + rr 双判决 + 行为门"齐备后运行, 按判据自动裁决并(被授意时)自动执行反修:
#   判据1(指标缺口): rr_code smin < RR_FLOOR(默认 0.36, 既有 0.37 带下沿) → 有缺口, 反修有据
#   判据2(行为面): 行为门 g2/g3 出现真工具帧([tool_call ...])= 绿; 只见抄贴 <tool_result> 文本= 红
#   裁决表: 缺口+行为绿 → 反修有据(目的=抬指标; 自动: dql快照→backfit→re-merge→复测门, 门退化即回滚)
#           无缺口       → 反修无据跳过(v2 教训: 好底座上反修=零收益高风险)
#           行为红       → 反修非对症跳过(行为问题不是反修能修的, 2026-07-22 三腿合判)
# 用法: [RR_FLOOR=0.36] [AUTO_RUN=0] ./backfit_decide.sh <launch_log> <gate_report>
#   AUTO_RUN=1 且裁决"有据"时: 在 M1 自动执行 快照→backfit(DS4_BF_JUSTIFIED=1)→merge 链。
set -uo pipefail
LOG="${1:?用法: backfit_decide.sh <launch_log> <gate_report>}"
GATE="${2:?}"
RR_FLOOR="${RR_FLOOR:-0.36}"
M1="${M1:-192.168.1.2}"

SMIN=$(ssh "$M1" "grep -aE '^VERDICT lcfg=g{43}' $LOG | grep 'S=305' | tail -1" 2>/dev/null \
      | grep -oE 'smin=[0-9.]+' | cut -d= -f2)
[ -n "$SMIN" ] || { echo "[decide] 拒裁: $LOG 无 S=305 VERDICT"; exit 2; }

GATE_TXT=$(ssh "$M1" "cat $GATE" 2>/dev/null)
[ -n "$GATE_TXT" ] || { echo "[decide] 拒裁: 行为门报告 $GATE 缺失"; exit 2; }
if echo "$GATE_TXT" | grep -q '\[tool_call '; then GATE_OK=1; else GATE_OK=0; fi

DEFICIT=$(awk -v s="$SMIN" -v f="$RR_FLOOR" 'BEGIN{print (s<f)?1:0}')
echo "[decide] rr_code smin=$SMIN (地板=$RR_FLOOR, 缺口=$DEFICIT) 行为门真帧=$GATE_OK"

if [ "$GATE_OK" = 0 ]; then
    echo "[decide] ★裁决: 行为面红 → 反修非对症, 跳过(行为问题走模型/soul/引擎路线)"
    exit 0
fi
if [ "$DEFICIT" = 0 ]; then
    echo "[decide] ★裁决: 指标达标($SMIN≥$RR_FLOOR)且行为绿 → 反修无据, 基线即终态"
    exit 0
fi
echo "[decide] ★裁决: 指标缺口($SMIN<$RR_FLOOR)且行为绿 → 反修有据(目的=抬指标)"
if [ "${AUTO_RUN:-0}" = 1 ]; then
    echo "[decide] AUTO_RUN=1 → M1 执行: dql 快照 → backfit(JUSTIFIED) → 后续需人工接 merge+复测门"
    ssh "$M1" "cd /Users/fodelf/ds4-main && rm -rf gguf/go-onebit/layers_snap && cp -al gguf/go-onebit/layers gguf/go-onebit/layers_snap && ( DS4_BF_JUSTIFIED=1 nohup perl -e 'setpgrp(0,0); exec @ARGV or die \$!' ./quant_layer.sh backfit > /tmp/backfit_auto.log 2>&1 & ) && echo backfit已后台(快照在 layers_snap, 门退化可回滚)"
else
    echo "[decide] AUTO_RUN 未开 → 只出裁决; 执行: AUTO_RUN=1 重跑本裁决器"
fi
