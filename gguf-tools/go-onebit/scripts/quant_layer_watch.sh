#!/bin/bash
# quant_layer_watch.sh — 非侵入式每层耗时观测(2026-08-19)。
# 量化器 stderr 只有 "generate_expert_tensor: layer L wid n/N experts" 没有时间戳,
# 这里 tail -F 流水线日志, 在每个 (层,张量) 完成时打一条带时间戳的行并算增量,
# 不动 amp86_spark.sh / quant_allq2_spark.sh 本身(禁改运行中脚本)。
# 用法: bash quant_layer_watch.sh [被观测日志] [输出]
set -uo pipefail
IN="${1:-/tmp/amp86_all.log}"
OUT="${2:-/tmp/amp86_layer_ts.log}"
: > "$OUT"
tail -F -n +1 "$IN" 2>/dev/null | awk -v out="$OUT" '
/generate_expert_tensor: layer .* experts$/ {
    # 只在 "n/N" 里 n==N 的收官行计时
    n = $(NF-1); split(n, a, "/");
    if (a[1] != a[2]) next;
    now = systime(); key = $3 " " $4;
    if (prev_t) d = now - prev_t; else d = 0;
    printf("%s layer=%s %s done  +%ds\n", strftime("%H:%M:%S"), $3, $4, d) >> out;
    fflush(out);
    prev_t = now;
}' &
echo "watching $IN -> $OUT (pid $!)"
