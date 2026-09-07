#!/bin/bash
# prof_bench_spark.sh — ds4-bench 解码循环的 nsys 剖面(spark 本机跑)。
#
# 为什么要有它: prof_decode_spark.sh 剖的是 `ds4 -p` 的交互生成路(短提示, ctx 几十 token);
# bench 路(ds4_session_eval 逐 token + 前沿快照存取)在长上下文前沿的行为不同——
# 2026-09-05 图开后 bench 2048 涨 +6%, 4096 以上反而比图关低 0.4~0.9 t/s, 只在这条路露馅。
# 出两张表: kernel 汇总 + CUDA API 汇总(看 cudaGraphInstantiate/ExecUpdate/Launch 次数与耗时)。
# 用法: prof_bench_spark.sh [ws=champ86amp] [标签=bench_<ws>] [ctx=4096] [gen=64] [gguf]
# 产物: gguf/go-onebit/vqhalf/<ws>/speed/prof_<标签>.{nsys-rep,log,_cuda_gpu_kern_sum.csv,_cuda_api_sum.csv}
set -uo pipefail
ROOT="$HOME/ds4-main"; VQH="$ROOT/gguf/go-onebit/vqhalf"; cd "$ROOT" || exit 1
WS="${1:-champ86amp}"; TAG="${2:-bench_$WS}"; CTX="${3:-4096}"; GEN="${4:-64}"
# 09-06 加: $6 语料(默认 promessi_sposi.txt, 1M 用 x4 拼接版), $7 透传 ds4-bench 的额外参数
# (如 "--fill-ctx 1046528 --prefill-chunk 2048": 合成上下文直接在 1M 处剖面, 不必真灌)。
CORPUS="${6:-speed-bench/promessi_sposi.txt}"; EXTRA="${7:-}"
MDL="${5:-$ROOT/gguf/ds4-$WS.gguf}"; ZCH="$VQH/$WS/zchain.bin"
OUT="$VQH/$WS/speed"; mkdir -p "$OUT"
LOG(){ echo "[profb $(date +%H:%M:%S)] $*"; }
[ -s "$MDL" ] || { LOG "★模型缺 $MDL★"; exit 2; }
[ -s "$ZCH" ] || { LOG "★zchain 缺 $ZCH★"; exit 2; }
BUSY=$(for p in ds4 ds4-bench ds4-server ds4quant_run zlayer vq_merge_v4; do pgrep -x "$p"; done)
[ -z "$BUSY" ] || { LOG "★机器非空: $BUSY★"; exit 3; }
REP="$OUT/prof_$TAG"; rm -f "$REP.nsys-rep" "$REP.sqlite"
LOG "ctx=$CTX gen=$GEN → $REP"
nsys profile -o "$REP" --force-overwrite=true --trace=cuda,nvtx --cuda-graph-trace=node \
    ./ds4-bench --cuda -m "$MDL" --zchain "$ZCH" --mem-budget-mb 110000 \
    --prompt-file "$CORPUS" --ctx-start "$CTX" --ctx-max "$CTX" --step-incr 2048 $EXTRA \
    --gen-tokens "$GEN" --csv "$OUT/$TAG.csv" > "$REP.log" 2>&1
RC=$?; LOG "bench rc=$RC $(tail -1 "$OUT/$TAG.csv" 2>/dev/null)"
[ $RC -eq 0 ] || { tail -5 "$REP.log"; exit 4; }
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true -o "$REP" "$REP.nsys-rep" > /dev/null 2>&1
nsys stats --report cuda_api_sum --format csv -o "$REP" "$REP.nsys-rep" > /dev/null 2>&1
LOG "CUDA API 表(图相关行):"
grep -E "Name|Graph" "${REP}_cuda_api_sum.csv" | cut -c1-120 | head -12
LOG "kernel 表前 12:"
head -13 "${REP}_cuda_gpu_kern_sum.csv" | cut -c1-140
