#!/bin/bash
# prof_decode_spark.sh — 冠军态解码逐 kernel 出账(nsys, 在 spark 本机跑)。
#
# 速度改码的唯一依据: 先知道每 token 的 GPU 时间花在哪个 kernel, 再动手。跑一次贪心
# 32 token(前 1 token 是 prefill, 其余 31 token 是纯解码), nsys 抓 CUDA kernel 时间线,
# 用 cuda_gpu_kern_sum 报表按 kernel 汇总总时长/次数/均值。
# 用法: prof_decode_spark.sh [工作区=champ86amp] [标签=工作区名] [n=32] [GGUF=gguf/ds4-<工作区>.gguf]
# 产物: gguf/go-onebit/vqhalf/<工作区>/speed/prof_<标签>.{nsys-rep,kern.csv,log,txt}
# 内存账同 speed_champ_spark.sh(--mem-budget-mb 110000; nsys 自身开销 <2 GB)。
set -uo pipefail
ROOT="$HOME/ds4-main"; VQH="$ROOT/gguf/go-onebit/vqhalf"
WS="${1:-champ86amp}"; TAG="${2:-$WS}"; NTOK="${3:-32}"
MDL="${4:-$ROOT/gguf/ds4-$WS.gguf}"; ZCH="${5:-$VQH/$WS/zchain.bin}"   # $5=none ⇒ 不挂链(对照模型如 allq2)
OUT="$VQH/$WS/speed"; mkdir -p "$OUT"
LOG(){ echo "[prof $(date +%H:%M:%S)] $*"; }
cd "$ROOT" || exit 1
[ -s "$MDL" ] || { LOG "★合一 GGUF 缺 $MDL★"; exit 2; }
ZFLAG=(); if [ "$ZCH" != none ]; then [ -s "$ZCH" ] || { LOG "★zchain 缺 $ZCH★"; exit 2; }; ZFLAG=(--zchain "$ZCH"); fi
BUSY=$(for p in ds4 ds4-bench ds4-server ds4quant_run zlayer vq_merge_v4; do pgrep -x "$p"; done)
[ -z "$BUSY" ] || { LOG "★机器非空: $BUSY★"; exit 3; }
PROMPT="Explain in plain words how a Redis stream differs from a Redis list."
REP="$OUT/prof_$TAG"; rm -f "$REP.nsys-rep" "$REP.sqlite"
LOG "nsys 抓 $NTOK token 贪心解码 → $REP.nsys-rep"
# --cuda-graph-trace=node: token CUDA Graph 开着时 kernel 在图里, 默认(graph)口径整张图只算一条, 逐核账全没了
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node -o "$REP" --force-overwrite=true \
    ./ds4 --cuda -m "$MDL" "${ZFLAG[@]}" --mem-budget-mb 110000 --temp 0 -n "$NTOK" -p "$PROMPT" \
    > "$REP.txt" 2> "$REP.log"
RC=$?
grep -h "t/s\|abort\|refus\|error" "$REP.log" | cut -c1-200
LOG "ds4 rc=$RC 输出: $(head -c 300 "$REP.txt" | tr '\n' ' ')"
[ -s "$REP.nsys-rep" ] || { LOG "★无 nsys-rep★"; exit 4; }
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true -o "$REP" "$REP.nsys-rep" >/dev/null 2>&1
CSV="${REP}_cuda_gpu_kern_sum.csv"
[ -s "$CSV" ] || { LOG "★kern_sum 报表缺★"; exit 5; }
mv -f "$CSV" "$REP.kern.csv"
LOG "kernel 汇总(总时长降序, 前 30):"
# 列: Time(%),Total Time(ns),Instances,Avg(ns),Med(ns),Min(ns),Max(ns),StdDev(ns),Name
awk -F',' 'NR==1{next} {printf "%6.2f%% %9.2fms n=%-6s avg=%8.1fus  ", $1, $2/1e6, $3, $4/1e3; $1=$2=$3=$4=$5=$6=$7=$8=""; print substr($0,1,120)}' "$REP.kern.csv" | head -30
LOG "GPU kernel 总时长: $(awk -F',' 'NR>1{s+=$2} END{printf "%.1f ms", s/1e6}' "$REP.kern.csv") / $NTOK token"
