#!/bin/bash
# champ86amp_chain.sh — 反修→sweep→双尺五指标 一条龙(2026-08-29 用户令"开始反修,sweep,看下五指标")
# 前置: champ86/layers_quant(量化态还原点, 43×4 类文件)+三锚(.layout 已随锚落)。
# 首跑以 /tmp/full_chain.sh 名义发车, 本文件为其入库版(脚本必须落 repo 铁律)。
set -u; cd ~/ds4-main
D2=gguf/go-onebit/vqhalf
L(){ echo "[chain $(date +%H:%M:%S)] $*"; }
L "②拷贝+③zlayer 反修发车(K=64, 从 layers_quant 重建工作区)"
SRCBASE=champ86/layers_quant bash gguf-tools/scripts/amp_clean_full.sh \
  champ86amp "" 64 "" "$D2/anchor_a_clean_s8192.bin" || { L "★反修失败★"; exit 1; }
L "④sweep 发车(推进+全层扫, 链闸=警告)"
bash gguf-tools/scripts/amp_campaign.sh champbf champ86amp full || { L "★sweep 失败★"; exit 1; }
L "⑤五指标: wt2 官方尺"
bash gguf-tools/scripts/caliper_ref.sh "$D2/champ86amp/layers" /tmp/qc_c86amp_wt2.bin 2>&1 | tail -12
L "⑤五指标: 判决份 8 域尺(裸+反修后)"
bash gguf-tools/scripts/amp_campaign.sh judge3 champ86amp
L "★全链收官★"
