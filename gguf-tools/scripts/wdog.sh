#!/bin/bash
# wdog.sh — 独立内存看门狗(2026-08-16): r30_campaign.sh WDOG 同款, 可独立挂/换档。
# 用法: wdog.sh [MB阈值] [pgrep模式] [日志文件]
# 背景: progz86 首跑判决回放合法峰12.3G被默认11900误杀(en86先例=22528), 需运行中换档。
MB_LIM=${1:-22528}
PAT=${2:-[d]s4quant_run}
LOGF=${3:-/tmp/wdog.log}
while true; do
    P=$(pgrep -nf "$PAT" || true)
    [ -n "$P" ] || { sleep 5; continue; }
    MB=$(footprint -p "$P" 2>/dev/null | grep -Eo 'Footprint: *[0-9.]+ *[KMG]B' | head -1 \
         | awk '{v=$2;u=$3; if(u=="GB")v*=1024; else if(u=="KB")v/=1024; printf "%d",v}' || true)
    [ -n "${MB:-}" ] && [ "$MB" -gt "$MB_LIM" ] && { echo "[wdog $(date +%T)] ${MB}MB >${MB_LIM}MB 杀 $P" >> "$LOGF"; kill -9 "$P" 2>/dev/null || true; }
    sleep 5
done
