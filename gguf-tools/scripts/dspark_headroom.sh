#!/bin/bash
# dspark_headroom.sh — 放大器天花板判据(2026-08-21)。
# 逻辑: drafter 放大器只修"专家量化误差"。若 q4 教师 drafter 与 q2 学生 drafter 的
# 接受率本就接近 ⇒ 专家量化不是接受率瓶颈 ⇒ 放大器无论解得多准都没有肉可捞
# (此时问题不在算法, 在前提)。反之若差距明显 ⇒ 有头寸, 放大器该补上这段。
set -uo pipefail
ROOT="$HOME/ds4-main"
MAIN="$ROOT/gguf/ds4-allq2.gguf"
MAIN_ZC="$ROOT/gguf/go-onebit/r30/full86/zchain_noge.bin"
Q4="$ROOT/gguf/ds4-dspark-drafter3-q4.gguf"
Q2="$ROOT/gguf/ds4-dspark-drafter3-q2.gguf"
AMP="${AMP:-$ROOT/gguf/go-onebit/r30/dspark/zchain_drafter_cal12z.bin}"
N="${N:-96}"; R="${R:-2}"
PROMPT="${PROMPT:-Write a Python function that reverses a string.}"
cd "$ROOT"
run(){   # $1=标签 $2=drafter $3=zchain(空=无)
    local acc ts
    DZC=""; [ -n "${3:-}" ] && DZC="--draft-zchain $3"
    # (DSPARK_STAT 统计行已随诊断清退, avg_acc 腿失效, 判读只看 t/s)
    timeout 1800 ./ds4 --cuda -m "$MAIN" --zchain "$MAIN_ZC" \
        --draft-gguf "$2" --spec $DZC --temp 0 -n "$N" -p "$PROMPT" \
        </dev/null >/dev/null 2>/tmp/hr.log
    acc=$(grep -a avg_acc /tmp/hr.log | tail -1 | sed 's/.*avg_acc=//')
    ts=$(grep -a generation /tmp/hr.log | tail -1 | sed 's/.*generation: //; s/ t\/s//')
    printf "%-22s acc=%-6s t/s=%s\n" "$1" "${acc:-?}" "${ts:-?}"
}
for i in $(seq 1 "$R"); do
    run "q4 教师 drafter" "$Q4" ""
    run "q2 学生 drafter" "$Q2" ""
    [ -s "$AMP" ] && run "q2 + cal12z 放大器" "$Q2" "$AMP"
done
