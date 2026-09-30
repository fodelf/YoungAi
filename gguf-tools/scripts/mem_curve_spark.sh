#!/bin/bash
# mem_curve_spark.sh — 服务端单条长请求的内存曲线(2026-09-29)。
#
# 为什么要它: 看门狗只记"杀"的那一刻(09-29 夜: 79k 上下文 MemAvailable 2314 MB 被杀; 前一夜 64.7k 时 1826 MB), 采样段的
# "采完 MemAvailable" 是请求结束、状态释放之后量的(稳定 8.9~9.0 GB), 请求【中途】的峰值一条读数都没有 ——
# 而按代码账(core_v41_forward.c: 状态按 cap 开, KV 满 1M 才 0.88 GB, 索引草稿翻倍长)算不出几 GB 的涨幅。
# 用法: mem_curve_spark.sh <日志路径> [看谁: 进程名, 默认 z_nightly_spark.sh]
#   每 5 s 一行: 时刻 MemAvailable(MB) ds4-server RSS(MB) 服务日志最新 gen= 数; 被看的脚本退出就收工。
#   RSS 与 MemAvailable 一起记是为了分主机/设备: GB10 统一内存上 cudaMalloc 的页不进进程 RSS, 只从 MemAvailable 里消失。
set -uo pipefail
OUT="${1:?日志路径}"; WATCH="${2:-z_nightly_spark.sh}"
SLOG="$HOME/ds4-server-1m.log"
echo "# t MemAvailable_MB rss_MB gen ctx_end" >>"$OUT"
while pgrep -f "$WATCH" >/dev/null; do
    a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    pid=$(pgrep -x ds4-server | head -1)
    rss=0; [ -n "$pid" ] && rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/"$pid"/status 2>/dev/null)
    last=$(grep -E "gen=[0-9]+ .*decoding chunk|prefill" "$SLOG" 2>/dev/null | tail -1)
    gen=$(echo "$last" | grep -oE "gen=[0-9]+" | head -1 | cut -d= -f2)
    ctx=$(echo "$last" | grep -oE "ctx=[0-9]+\.\.[0-9]+" | head -1 | cut -d. -f3)
    echo "$(date +%H:%M:%S) $a ${rss:-0} ${gen:-0} ${ctx:-0}" >>"$OUT"
    sleep 5
done
echo "# 收工 $(date +%H:%M:%S)" >>"$OUT"
