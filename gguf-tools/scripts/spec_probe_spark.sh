#!/bin/bash
# spec_probe_spark.sh — 投机解码(--spec + DSpark drafter)对纯解码的分钟级探针(spark 本机跑, 2026-09-07)。
# 同一二进制、同模型、同提示、温 0 各跑一遍(spec / plain), 打: 生成 t/s、spec 账(轮成本分解 + 逐位接受率,
# 引擎 ds4_spec_stats_print)、两路输出文本头(逐字节是否全同 —— 投机的贪心 verify 应与纯解码同轨; 不同轨 =
# 批/单 token 核累加序差异, 08-21 已定罪, 本战役要修到同轨)。
# 用法: spec_probe_spark.sh <标签> [N=128] [提示文件|-] [工作区=champ86q4k] [drafter=gguf/ds4-dspark-ve-q4.gguf] [二进制=./ds4]
# 产物: gguf/go-onebit/vqhalf/<工作区>/speed/spec_<标签>_{spec,plain}.{txt,log}
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
TAG="${1:?标签}"; N="${2:-128}"; PFILE="${3:--}"; WS="${4:-champ86q4k}"
DRAFT="${5:-gguf/ds4-dspark-ve-q4.gguf}"; BIN="${6:-./ds4}"
SP="$ROOT/gguf/go-onebit/vqhalf/$WS/speed"; mkdir -p "$SP"
LOG(){ echo "[spec_probe $(date +%H:%M:%S)] $*"; }
[ -s "gguf/ds4-$WS.gguf" ] && [ -s "gguf/go-onebit/vqhalf/$WS/zchain.bin" ] || { LOG "★工作区 $WS 缺模型或链★"; exit 2; }
[ -s "$DRAFT" ] || { LOG "★drafter 缺 $DRAFT★"; exit 2; }
BUSY=$(for p in ds4 ds4-bench ds4-server ds4quant_run zlayer deepseek4-quantize; do pgrep -x "$p"; done)
[ -z "$BUSY" ] || { LOG "★机器非空: $BUSY★"; exit 3; }
# 内存账: 主模型 84 GB(专家进 HBM arena) + drafter 11.4 GB(副 map 整体 cudaHostRegister) + KV/暂存 ⇒ ~97 GB,
# --mem-budget-mb 110000 由引擎 L1 静态闸把关(超预算拒绝启动), 121 GB 机器留 ~10 GB 给系统。
M=("$BIN" --cuda -m "gguf/ds4-$WS.gguf" --zchain "gguf/go-onebit/vqhalf/$WS/zchain.bin" --mem-budget-mb 110000 --temp 0 -n "$N")
PROMPT="Explain in plain words how a transformer language model generates text one token at a time, then describe two practical ways to make that generation faster without changing the model's output."
if [ "$PFILE" != "-" ]; then
    [ -s "$PFILE" ] || { LOG "★提示文件缺 $PFILE★"; exit 2; }
    P=(--prompt-file "$PFILE" --ctx 8192)
else
    P=(-p "$PROMPT")
fi
run(){ # $1=名 $2.. 附加参数
    local name="$1"; shift
    "${M[@]}" "$@" "${P[@]}" > "$SP/spec_${TAG}_$name.txt" 2> "$SP/spec_${TAG}_$name.log"
    local rc=$?
    LOG "$name rc=$rc $(grep -h 'generation:' "$SP/spec_${TAG}_$name.log" | tail -1)"
    grep -h "spec 账\|spec 逐位\|drafter armed\|draft gguf\|★" "$SP/spec_${TAG}_$name.log" | head -8
    LOG "$name 文本头: $(head -c 240 "$SP/spec_${TAG}_$name.txt" | tr '\n' ' ')"
    return $rc
}
run spec --draft-gguf "$DRAFT" --spec || exit 4
run plain || exit 4
if cmp -s "$SP/spec_${TAG}_spec.txt" "$SP/spec_${TAG}_plain.txt"; then
    LOG "★输出逐字节全同: spec == plain (N=$N)★"
else
    LOG "★输出分叉★ 首个不同处: $(cmp "$SP/spec_${TAG}_spec.txt" "$SP/spec_${TAG}_plain.txt" 2>&1 | head -1)"
fi
echo "SPEC_PROBE_${TAG}_END"
