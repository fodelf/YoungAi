#!/bin/bash
# spec_prof_spark.sh — 投机解码(--spec)一次生成的 nsys 剖面 + 逐核账(spark 本机跑, 2026-09-07)。
# 为什么不用 prof_bench_spark.sh/prof_decode_account.sh: 那两个按"每 token 一张 4 相图"切解码窗口, 投机路
# 一轮 = draft(drafter 3 块层批 5 token) + verify 批(≤4 token, 直发 ~3.5k 核) + 恢复, 没有逐 token 周期。
# 这里把整段生成(prefill 之后)的 kernel 时间按核名汇总, 再按 spec 账里的轮数折成 ms/轮, 看肉在哪个核。
# 用法: spec_prof_spark.sh <标签> [N=48] [工作区=champ86q4k] [drafter=gguf/ds4-dspark-ve-q4.gguf] [top=40] [mode=spec|plain] [fill=0]
# mode=plain: 不带 --spec 的纯解码剖面, "轮"=token(ms/轮 即 ms/token), 用来逐核对照投机 verify 批 vs 单 token 解码。
# fill>0(09-07 1M 战役): 改剖 ds4-bench 的合成上下文点(--fill-ctx fill, 前沿 fill+2048, 生成 N), 看 1M 处 verify 批的肉在哪个核。
# 产物: gguf/go-onebit/vqhalf/<工作区>/speed/specprof_<标签>.{nsys-rep,log,txt,_cuda_gpu_trace.csv}
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
TAG="${1:?标签}"; N="${2:-48}"; WS="${3:-champ86q4k}"; DRAFT="${4:-gguf/ds4-dspark-ve-q4.gguf}"; TOP="${5:-40}"; MODE="${6:-spec}"; FILL="${7:-0}"
SPEC_ARGS=(--draft-gguf "$DRAFT" --spec); [ "$MODE" = plain ] && SPEC_ARGS=()
SP="$ROOT/gguf/go-onebit/vqhalf/$WS/speed"; mkdir -p "$SP"
LOG(){ echo "[spec_prof $(date +%H:%M:%S)] $*"; }
[ -s "gguf/ds4-$WS.gguf" ] && [ -s "gguf/go-onebit/vqhalf/$WS/zchain.bin" ] || { LOG "★工作区 $WS 缺模型或链★"; exit 2; }
[ -s "$DRAFT" ] || { LOG "★drafter 缺 $DRAFT★"; exit 2; }
BUSY=$(for p in ds4 ds4-bench ds4-server ds4quant_run zlayer deepseek4-quantize; do pgrep -x "$p"; done)
[ -z "$BUSY" ] || { LOG "★机器非空: $BUSY★"; exit 3; }
REP="$SP/specprof_$TAG"; rm -f "$REP.nsys-rep" "$REP.sqlite" "${REP}_cuda_gpu_trace.csv"
PROMPT="Explain in plain words how a transformer language model generates text one token at a time, then describe two practical ways to make that generation faster without changing the model's output."
M=(--cuda -m "gguf/ds4-$WS.gguf" --zchain "gguf/go-onebit/vqhalf/$WS/zchain.bin" --mem-budget-mb 110000)
if [ "$FILL" -gt 0 ]; then
    CTX=$((FILL + 2048))
    [ -s speed-bench/readme_en_x80.txt ] || for i in $(seq 80); do cat README.md; echo; done > speed-bench/readme_en_x80.txt   # 英文语料现生成不入库
    CMD=(./ds4-bench "${M[@]}" --prefill-chunk 2048 --gen-tokens "$N" --prompt-file speed-bench/readme_en_x80.txt
         "${SPEC_ARGS[@]}" --fill-ctx "$FILL" --ctx-start "$CTX" --ctx-max "$CTX" --csv "$REP.csv")
else
    CMD=(./ds4 "${M[@]}" --temp 0 -n "$N" "${SPEC_ARGS[@]}" -p "$PROMPT")
fi
nsys profile -o "$REP" --force-overwrite=true --trace=cuda,nvtx --cuda-graph-trace=node "${CMD[@]}" > "$REP.txt" 2> "$REP.log"
RC=$?; LOG "rc=$RC $(grep -h 'generation:' "$REP.log" | tail -1) $([ "$FILL" -gt 0 ] && tail -1 "$REP.csv" 2>/dev/null)"
grep -h "spec 账\|spec 逐位" "$REP.log"
[ $RC -eq 0 ] || { tail -5 "$REP.log"; exit 4; }
nsys stats --report cuda_gpu_trace --format csv --force-export=true -o "$REP" "$REP.nsys-rep" > /dev/null 2>&1
CSV="${REP}_cuda_gpu_trace.csv"; [ -s "$CSV" ] || { LOG "★gpu_trace 导出失败★"; exit 5; }
ROUNDS=$(grep -h "spec 账" "$REP.log" | sed 's/.*投机轮 \([0-9]*\).*/\1/'); ROUNDS="${ROUNDS:-1}"
[ "$MODE" = plain ] && ROUNDS="$N"
PLAIN=$(grep -h "spec 账" "$REP.log" | sed 's/.*纯解码轮 \([0-9]*\).*/\1/'); PLAIN="${PLAIN:-0}"
LOG "生成段逐核账(prefill 之后; 投机轮 $ROUNDS, 纯解码轮 $PLAIN; ms/轮 = 总时长/投机轮数, 纯解码核也摊在里面)"
MARK=dspark_attn_kernel; [ "$MODE" = plain ] && MARK=q4k_hc_expand_kernel
bash "$(dirname "$0")/spec_prof_account.sh" "$CSV" "$ROUNDS" "$TOP" "$MARK"
echo "SPEC_PROF_${TAG}_END"
