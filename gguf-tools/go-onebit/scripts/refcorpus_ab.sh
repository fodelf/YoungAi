#!/bin/bash
# refcorpus_ab.sh — DS4_REF_CORPUS A/B 冒烟(四支柱分片消费端判决, ~8min):
# 同 prompt temp0 各跑 开/关 REF_CORPUS 两轮, 比三样:
#   ① copy-spec ref 档是否真触发(DS4_COPY_SPEC_LOG 诊断行)
#   ② 速度(gen t/s / 用时)
#   ③ 输出逐字一致性(免费门=lossless, 不一致=bug 信号)
# 复用 code1b_smoke.sh(看门狗内嵌)。输出: /tmp/refcorpus_ab.report
set -euo pipefail
cd "$(dirname "$0")/.."
# 绝对路径: code1b_smoke 会 cd 到 repo 根, 相对路径会 cannot open(首跑实证)
REF="${REF:-$PWD/corpus/build/ref_pillars/p2_method_core.txt:$PWD/corpus/build/ref_pillars/p1_go_code.txt}"
NPRED="${NPRED:-48}"
MODEL="${MODEL:-gguf/go-onebit/ds4-code1b-v2.gguf}"
RESID="${RESID:-gguf/sidecars/code-hot-res-v2.gguf}"
REPORT=/tmp/refcorpus_ab.report
: > "$REPORT"

P1='<｜begin▁of▁sentence｜>Cache penetration means queries for keys that exist in'
P2='<｜begin▁of▁sentence｜>// twoSum returns the indices of the two numbers in nums that add up to target.
func twoSum(nums []int, target int) []int {'

run() { # $1=tag $2=prompt $3=refenv
    local out=/tmp/ab_$1.out
    echo "[ab] $1 跑中..." >&2
    DS4_RESIDUAL="$RESID" DS4_COPY_SPEC_LOG=1 DS4_REF_CORPUS="$3" \
      PROMPT="$2" MODEL="$MODEL" NPRED="$NPRED" TIMEOUT_S=300 \
      ./scripts/code1b_smoke.sh > "$out" 2>/tmp/ab_$1.err || true
    {   echo "──── $1 ────"
        sed -n '/原始输出/,/引擎速度/p' "$out" | sed '1d;$d'
        grep -E 'ref-corpus|cs:|spec|prefill:.*generation' /tmp/ab_$1.err /tmp/code1b_smoke.log 2>/dev/null | tail -6
        grep -o '用时=[0-9]*s' "$out" || true
        echo
    } >> "$REPORT"
}

for pn in 1 2; do
    eval "P=\$P$pn"
    [ "${SKIP_OFF:-0}" = "1" ] || run "p${pn}_off" "$P" ""
    run "p${pn}_ref" "$P" "$REF"
    a=$(sed -n '/原始输出/,/引擎速度/p' /tmp/ab_p${pn}_off.out | sed '1d;$d')
    b=$(sed -n '/原始输出/,/引擎速度/p' /tmp/ab_p${pn}_ref.out | sed '1d;$d')
    if [ "$a" = "$b" ]; then echo "[p$pn] 输出逐字一致 ✓(lossless 闸过)" >> "$REPORT"
    else echo "[p$pn] ★输出不一致(bug 信号, 免费门应 lossless)★" >> "$REPORT"; fi
    echo >> "$REPORT"
done
echo "[ab] 完成 → $REPORT" >&2
