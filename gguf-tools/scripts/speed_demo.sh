#!/bin/sh
# speed_demo.sh — MINUTES-scale speed localization (never the NLL-sized run):
# three paired arms × 12 greedy tokens.
# (env 大扫除 2026-08-31: 逐相 IO profiling 诊断口 DS4_METAL_EXPERT_IO_PROFILE 已从引擎
#  删除 — gather/drain 行不再出现, 判决只剩 per-arm gen t/s。)
#   usage: speed_demo.sh   (arms fixed: 22-layer ship / 43-layer ef2 / bare)
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 1
P='<｜begin▁of▁sentence｜>package main

func Add(a, b int) int {'
for arm in "A22 gguf/ds4-go1b-corr-rrr-partial.gguf" "B43 gguf/ds4-go1b-corr-ef2.gguf" "C0 -"; do
    name=${arm%% *}; corr=${arm#* }
    echo "=== $name ==="
    if [ "$corr" = "-" ]; then
        ./ds4 -m gguf/ds4-go1b.gguf \
            -p "$P" -n 12 --temp 0 --metal > "/tmp/arm_$name.full" 2>&1
    else
        ./ds4 -m gguf/ds4-go1b.gguf --corr "$corr" \
            -p "$P" -n 12 --temp 0 --metal > "/tmp/arm_$name.full" 2>&1
    fi
    grep -a 't/s' "/tmp/arm_$name.full"
    awk '/ds4-io/ {for(i=1;i<=NF;i++){if($i~/^wall_ms=/){sub("wall_ms=","",$i);w+=$i;n++} if($i~/^drain_ms=/){sub("drain_ms=","",$i);d+=$i}}} END {if(n) printf "gather %.1fms drain %.1fms per-layer-event (n=%d)\n", w/n, d/n, n}' "/tmp/arm_$name.full"
done
echo SPEED-DEMO-DONE
