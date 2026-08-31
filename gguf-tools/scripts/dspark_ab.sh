#!/bin/bash
# dspark_ab.sh — 四文件 A/B: drafter 放大器开/关 的 acc 与 t/s 对照(2026-08-21)。
# 用法: [ZC=<drafter_zchain>] [N=96] [R=3] bash dspark_ab.sh
# 交替跑(with/without 成对), 打印每轮 acc/t/s 与均值, 避免热态漂移造成的单边偏差。
set -uo pipefail
ROOT="$HOME/ds4-main"
D="$ROOT/gguf/go-onebit/r30/dspark"
ZC="${ZC:-$D/zchain_drafter_amp.bin}"
STUDENT="$ROOT/gguf/ds4-dspark-drafter3-q2.gguf"
MAIN="$ROOT/gguf/ds4-allq2.gguf"
MAIN_ZC="$ROOT/gguf/go-onebit/r30/full86/zchain_noge.bin"
N="${N:-96}"; R="${R:-3}"
PROMPT="${PROMPT:-Write a Python function that reverses a string.}"
cd "$ROOT"
[ -s "$ZC" ] || { echo "缺 drafter zchain: $ZC"; exit 1; }
declare -a A_ACC B_ACC A_TS B_TS
for i in $(seq 1 "$R"); do
    for M in with without; do
        DZC=""; [ "$M" = with ] && DZC="--draft-zchain $ZC"
        # (DSPARK_STAT 已删, avg_acc 腿失效, A/B 只看 t/s)
        timeout 1800 ./ds4 --cuda -m "$MAIN" --zchain "$MAIN_ZC" \
            --draft-gguf "$STUDENT" --spec $DZC --temp 0 -n "$N" -p "$PROMPT" \
            </dev/null >/dev/null 2>/tmp/ab_$M.log
        acc=$(grep -a avg_acc /tmp/ab_$M.log | tail -1 | sed 's/.*avg_acc=//')
        ts=$(grep -a generation /tmp/ab_$M.log | tail -1 | sed 's/.*generation: //; s/ t\/s//')
        printf "%-8s r%d  acc=%s  t/s=%s\n" "$M" "$i" "${acc:-?}" "${ts:-?}"
        if [ "$M" = with ]; then A_ACC+=("$acc"); A_TS+=("$ts"); else B_ACC+=("$acc"); B_TS+=("$ts"); fi
    done
done
mean(){ printf '%s\n' "$@" | awk '{s+=$1; n++} END{if(n) printf "%.3f", s/n; else printf "?"}'; }
echo "----"
echo "with    均值 acc=$(mean "${A_ACC[@]}")  t/s=$(mean "${A_TS[@]}")"
echo "without 均值 acc=$(mean "${B_ACC[@]}")  t/s=$(mean "${B_TS[@]}")"
