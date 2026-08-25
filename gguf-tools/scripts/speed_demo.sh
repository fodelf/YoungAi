#!/bin/sh
# speed_demo.sh — MINUTES-scale speed localization (never the NLL-sized run):
# three paired arms × 12 greedy tokens with expert-IO phase profiling.
# Verdict lines: per-arm gen t/s + per-layer gather/drain means.
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
        DS4_METAL_EXPERT_IO_PROFILE=1 ./ds4 -m gguf/ds4-go1b.gguf \
            -p "$P" -n 12 --temp 0 --metal > "/tmp/arm_$name.full" 2>&1
    else
        DS4_METAL_EXPERT_IO_PROFILE=1 ./ds4 -m gguf/ds4-go1b.gguf --corr "$corr" \
            -p "$P" -n 12 --temp 0 --metal > "/tmp/arm_$name.full" 2>&1
    fi
    grep -a 't/s' "/tmp/arm_$name.full"
    awk '/ds4-io/ {for(i=1;i<=NF;i++){if($i~/^wall_ms=/){sub("wall_ms=","",$i);w+=$i;n++} if($i~/^drain_ms=/){sub("drain_ms=","",$i);d+=$i}}} END {if(n) printf "gather %.1fms drain %.1fms per-layer-event (n=%d)\n", w/n, d/n, n}' "/tmp/arm_$name.full"
done
echo SPEED-DEMO-DONE
