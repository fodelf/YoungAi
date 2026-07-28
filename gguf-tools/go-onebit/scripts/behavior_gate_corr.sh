#!/bin/bash
# behavior_gate_corr.sh — corr 快校准判决闸(2026-07-27)。
# 对给定 corr 侧车 + α: ①fork-flip×6(躲题 prompt 的 fork 位 top-2, 判 代码token 是否胜出)
# ②字节护栏(twoSum 16tok 与无 corr 基准逐字节 diff — 好行为不许被带坏; 允许差异时人工判读)。
# 用法: CORR=gguf/go-onebit/corr-v3.gguf ALPHA=0.5 ./behavior_gate_corr.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
CORR="${CORR:?corr gguf 相对路径}"
ALPHA="${ALPHA:-1.0}"
MODEL=gguf/go-onebit/ds4-vq22.gguf
RPT=$ROOT/gguf-tools/go-onebit/reports/behavior_gate_corr_a${ALPHA}_$(date +%F_%H%M).report
: > "$RPT"
cd "$ROOT"

run_fork() {  # run_fork <prompt文件> <标签>
  PROMPT="<｜begin▁of▁sentence｜>$(cat "$1")"$'\n' COORD_MODEL=$MODEL MODEL=$MODEL \
    CORR="$CORR" CORR_SCALE="$ALPHA" VQ_GPU=1 NPRED=2 CTX=2048 RUN_TIMEOUT=600 \
    DUMP_LP=/tmp/gate_corr_lp.json tools/dual_vq.sh > /tmp/gate_corr_run.log 2>&1
  python3 - "$2" >> "$RPT" <<'PEOF'
import json, sys
d = json.load(open('/tmp/gate_corr_lp.json'))
s = d['steps'][1] if len(d['steps']) > 1 else d['steps'][0]
top = s['top_logprobs'][:4]
sel = s['selected']['text']
code_win = not sel.strip().startswith('//')
print(f"[{sys.argv[1]}] fork选中 {sel!r} {'✓代码' if code_win else '✗注释'} | " +
      " | ".join(f"{t['token']['text']!r}:{t['logit']:.2f}" for t in top))
PEOF
}

echo "== corr=$CORR α=$ALPHA ==" >> "$RPT"
for K in 1 6 11 15 17 19; do
  run_fork /tmp/dodge_p$K.txt "Go/$K"
done

# 字节护栏: twoSum 16tok(与已记录基准 " \n    for i := 0; i < len(nums); i++ {" 对比)
PROMPT="<｜begin▁of▁sentence｜>// twoSum returns the indices of the two numbers in nums that add up to target.
func twoSum(nums []int, target int) []int {" COORD_MODEL=$MODEL MODEL=$MODEL \
  CORR="$CORR" CORR_SCALE="$ALPHA" VQ_GPU=1 NPRED=16 CTX=2048 RUN_TIMEOUT=600 \
  tools/dual_vq.sh > /tmp/gate_corr_ts.log 2>&1
TS=$(cat /tmp/dual_vq_coord.out)
REF=' 
    for i := 0; i < len(nums); i++ {'
if [ "$TS" = "$REF" ]; then echo "[护栏 twoSum] ✓逐字节一致" >> "$RPT"
else { echo "[护栏 twoSum] ★变化(人工判读):"; printf '%s\n' "$TS"; } >> "$RPT"; fi

cat "$RPT" >&2
echo "[gate] 归档 → $RPT" >&2
