#!/bin/bash
# en86_launch.sh — en86 战役 M1 本地启动器(唯一合法发车入口)。
# 教训(08-14 双实例互杀事故): 上一实例未死时重发 → 旧 wdog(低线)击杀新实例锚写出峰,
# 双方锚全截断。本启动器 = 清场→等灭→日志轮转→单实例发车(27G 兜底=锚写出合法峰~24G)。
set -u
R=/Users/fodelf/ds4-main
pkill -9 -f "r30_campaign.sh en86" 2>/dev/null; pkill -9 -f ds4quant_run 2>/dev/null; pkill -9 -f "zlever/zlayer.py" 2>/dev/null
for i in 1 2 3 4 5 6; do pgrep -f "r30_campaign.sh en86|ds4quant_run|zlever/zlayer.py" >/dev/null || break; sleep 2; done
pgrep -f "r30_campaign.sh en86|ds4quant_run|zlever/zlayer.py" >/dev/null && { echo "清场失败, 拒发"; exit 1; }
[ -f /tmp/en86.log ] && mv /tmp/en86.log "/tmp/en86_$(date +%H%M%S).old.log"
cd "$R"
nohup env WDOG_MB=27648 ZL_SWLIM=60 QBIN_OVERRIDE=$R/gguf-tools/go-onebit/quant/ds4quant_run.dchunk \
  bash gguf-tools/go-onebit/scripts/r30_campaign.sh en86 > /tmp/en86.log 2>&1 &
echo "en86 发车 PID=$! $(date +%F_%T) WDOG=27648MB"
