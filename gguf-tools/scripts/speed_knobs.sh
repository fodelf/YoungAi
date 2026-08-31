#!/bin/sh
# speed_knobs.sh — MINUTES-scale sweep on the BARE go1b decode path.
# ★env 大扫除 2026-08-31: 原扫描对象(PREAD/PREAD_NOCACHE/PREFETCH_AHEAD/EVENT_DRAIN
# 专家 IO env 旋钮族)与逐相 IO profiling 诊断口已整体从引擎删除, 行为写死 — 各臂无从
# 区分, 只保留 base 臂当纯速度基线; 死臂原样注释在文件尾, 供翻旧账。★
# One ds4 process at a time (instance discipline); 12 greedy tokens per arm.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 1
MODEL=${MODEL:-gguf/ds4-go1b.gguf}
CORR=${CORR:-}
P='<｜begin▁of▁sentence｜>package main

func Add(a, b int) int {'
run_arm() {
    name=$1; shift
    echo "=== $name ===" >&2
    if [ -n "$CORR" ]; then set -- "$@" --corr "$CORR"; fi
    ./ds4 -m "$MODEL" "$@" \
        -p "$P" -n 12 --temp 0 --metal > "/tmp/knob_$name.full" 2>&1
    grep -a 't/s' "/tmp/knob_$name.full" | tail -1
}
run_arm base
# 死臂存档(引擎旋钮已删, 见头注):
#   run_arm pread          DS4_METAL_EXPERT_PREAD=1
#   run_arm pread_nocache  DS4_METAL_EXPERT_PREAD=1 DS4_METAL_EXPERT_PREAD_NOCACHE=1
#   run_arm prefetch       DS4_METAL_EXPERT_PREFETCH_AHEAD=1
#   run_arm pread_prefetch DS4_METAL_EXPERT_PREAD=1 DS4_METAL_EXPERT_PREFETCH_AHEAD=1
#   run_arm event_drain    DS4_METAL_EXPERT_EVENT_DRAIN=1
echo SPEED-KNOBS-DONE
