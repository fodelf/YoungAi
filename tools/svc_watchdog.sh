#!/bin/bash
# 12G 红线看门狗 — 由 svc.sh ensure_watchdog 拉起 (perl setpgrp 自立进程组, exec -a 改名
# svc_watchdog_marker)。argv: PORT M1 [LIMIT_GB=12]。超线双侧同杀 (coordinator kill + M1 pkill)。
# 为什么是独立文件: 旧内联版嵌在 '...\"...\\\"...' 五层转义里, macOS 系统 bash 3.2 对
# "$(awk "...\"...\"...")" 嵌套引号解析错乱 → g/rg 永远取空 → [ "" -gt 12 ] 恒假 →
# 红线从未生效 (2026-07-18 实证复现)。本文件只用 awk -v 单引号程序, 数学路径可单测。
PORT=${1:?port}; M1=${2:?m1 host}; LIMIT=${3:-12}
while true; do
  pid=$(pgrep -f "^\./ds4-server .*--port $PORT" | head -1)
  [ -z "$pid" ] && exit 0
  kb=$(ps -o rss= -p "$pid" | tr -d ' ')
  g=$(awk -v k="${kb:-0}" 'BEGIN{printf "%.0f", k/1048576}')
  rkb=$(ssh "$M1" "pgrep -f '^\./ds4 .*role worker' | head -1 | xargs -I{} ps -o rss= -p {}" 2>/dev/null | tr -d ' ')
  rg=$(awk -v k="${rkb:-0}" 'BEGIN{printf "%.0f", k/1048576}')
  if [ "${g:-0}" -gt "$LIMIT" ] || [ "${rg:-0}" -gt "$LIMIT" ]; then
    echo "[watchdog] red line: coord=${g}G worker=${rg}G limit=${LIMIT}G — killing both sides"
    kill "$pid"; ssh "$M1" "pkill -f 'role worker'"; exit 1
  fi
  sleep 30
done
