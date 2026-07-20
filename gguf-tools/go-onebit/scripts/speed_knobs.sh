#!/bin/sh
# speed_knobs.sh — MINUTES-scale env-knob sweep on the BARE go1b decode path.
# The q2-era expert-IO knobs live in the shared A3 gather machinery (type-
# agnostic), so they apply to go1b streaming unmodified: zero-code levers.
# Arms: baseline / pread / pread+nocache / prefetch / pread+prefetch.
# One ds4 process at a time (instance discipline); 12 greedy tokens per arm.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || exit 1
MODEL=${MODEL:-gguf/ds4-go1b.gguf}
CORR=${CORR:-}
P='<｜begin▁of▁sentence｜>package main

func Add(a, b int) int {'
run_arm() {
    name=$1; shift
    echo "=== $name ===" >&2
    if [ -n "$CORR" ]; then set -- "$@" --corr "$CORR"; fi
    env DS4_METAL_EXPERT_IO_PROFILE=1 "$@" ./ds4 -m "$MODEL" \
        -p "$P" -n 12 --temp 0 --metal > "/tmp/knob_$name.full" 2>&1
    grep -a 't/s' "/tmp/knob_$name.full" | tail -1
    awk '/ds4-io/ {for(i=1;i<=NF;i++){if($i~/^wall_ms=/){sub("wall_ms=","",$i);w+=$i;n++} if($i~/^drain_ms=/){sub("drain_ms=","",$i);d+=$i}}} END {if(n) printf "  gather %.1fms drain %.1fms per-layer-event (n=%d)\n", w/n, d/n, n}' "/tmp/knob_$name.full"
}
run_arm base
run_arm pread          DS4_METAL_EXPERT_PREAD=1
run_arm pread_nocache  DS4_METAL_EXPERT_PREAD=1 DS4_METAL_EXPERT_PREAD_NOCACHE=1
run_arm prefetch       DS4_METAL_EXPERT_PREFETCH_AHEAD=1
run_arm pread_prefetch DS4_METAL_EXPERT_PREAD=1 DS4_METAL_EXPERT_PREFETCH_AHEAD=1
run_arm event_drain    DS4_METAL_EXPERT_EVENT_DRAIN=1
echo SPEED-KNOBS-DONE
