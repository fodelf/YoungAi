#!/bin/bash
# tokgraph_ab_spark.sh — 解码 token CUDA Graph 开/关的逐位对拍 + 速度 A/B(spark 本机跑)。
#
# 为什么不用 kernel_parity_spark.sh: --score-ids 走的是裸 token 通道(不过 token graph),
# 测不到图重放。这里走真正的生成路: 两个二进制各跑一遍 --dump-logprobs 贪心 N token
# (同 prompt/同种子/温 0), 逐字节比 JSON; 图重放若吃到错参数, 从某个位置起 logprobs 必分叉。
# 用法: tokgraph_ab_spark.sh <直发二进制> <开图二进制> [N=256] [标签] [提示文件]
#   第 5 参给了文件 ⇒ 用 --prompt-file + --ctx 8192 跑长提示(覆盖 n_comp>1024 的稀疏 indexer
#   路和 4096 原始窗满的态; 短提示对拍过不等于长上下文也过——图里烤进的计数参数只在这里露馅)。
# 产物: gguf/go-onebit/vqhalf/champ86amp/speed/tokgraph_<标签>_{off,on}.{json,log}
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
OFF="${1:?直发二进制}"; ON="${2:?开图二进制}"; N="${3:-256}"; TAG="${4:-$(date +%m%d_%H%M)}"; PFILE="${5:-}"
# $6 = 工作区(模型 gguf/ds4-<ws>.gguf + 链 vqhalf/<ws>/zchain.bin), 默认冠军 champ86amp; 09-06 加: Q4_K 骨架
# (champ86q4k)的 q4_K 核改动只有挂上该模型才会被执行, 对着冠军模型对拍等于没测。
WS="${6:-champ86amp}"
SP="$ROOT/gguf/go-onebit/vqhalf/$WS/speed"; mkdir -p "$SP"
[ -s "gguf/ds4-$WS.gguf" ] && [ -s "gguf/go-onebit/vqhalf/$WS/zchain.bin" ] || { echo "★工作区 $WS 缺模型或链★"; exit 2; }
M=(--cuda -m "gguf/ds4-$WS.gguf" --zchain "gguf/go-onebit/vqhalf/$WS/zchain.bin" --mem-budget-mb 110000 --temp 0 -n "$N")
PROMPT="Explain in plain words how a Redis stream differs from a Redis list, then give a short example of each."
if [ -n "$PFILE" ]; then
    [ -s "$PFILE" ] || { echo "★提示文件缺 $PFILE★"; exit 2; }
    P=(--prompt-file "$PFILE" --ctx 8192)
else
    P=(-p "$PROMPT")
fi
LOG(){ echo "[tokgraph $(date +%H:%M:%S)] $*"; }
BUSY=$(for p in ds4 ds4-bench ds4-server ds4quant_run zlayer; do pgrep -x "$p"; done)
[ -z "$BUSY" ] || { LOG "★机器非空: $BUSY★"; exit 3; }
run(){ # $1=二进制 $2=名
    "$1" "${M[@]}" --dump-logprobs "$SP/tokgraph_${TAG}_$2.json" "${P[@]}" > "$SP/tokgraph_${TAG}_$2.txt" 2> "$SP/tokgraph_${TAG}_$2.log"
    local rc=$?
    LOG "$2 rc=$rc $(grep -h 't/s' "$SP/tokgraph_${TAG}_$2.log" | tail -1) $(grep -ch 'token graph' "$SP/tokgraph_${TAG}_$2.log") 条图日志"
    grep -h "token graph\|graph capture\|graph instantiate" "$SP/tokgraph_${TAG}_$2.log" | head -3
    return $rc
}
run "$OFF" off || exit 4
run "$ON"  on  || exit 4
if cmp -s "$SP/tokgraph_${TAG}_off.json" "$SP/tokgraph_${TAG}_on.json"; then
    LOG "★逐字节全同: 开图 == 直发 (N=$N)★"
else
    LOG "★分叉★ 首个不同行:"
    diff <(tr ',' '\n' < "$SP/tokgraph_${TAG}_off.json") <(tr ',' '\n' < "$SP/tokgraph_${TAG}_on.json") | head -6
    LOG "文本: off=$(head -c 200 "$SP/tokgraph_${TAG}_off.txt" | tr '\n' ' ')"
    LOG "文本: on =$(head -c 200 "$SP/tokgraph_${TAG}_on.txt" | tr '\n' ' ')"
fi
