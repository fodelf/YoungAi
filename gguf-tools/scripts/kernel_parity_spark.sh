#!/bin/bash
# kernel_parity_spark.sh — 引擎 kernel 改动的数值对拍(spark 本机跑)。
#
# 判"改 kernel 有没有改模型输出"不能只看贪心文本(2-bit MoE 路由近平局, 浮点重排就会翻
# token), 要看教师强制 logits: ①新二进制跑两遍逐字节全同(决定论) ②新/旧二进制各对 FP 锚
# 打五指标, 两者对锚的 KLD/Σmin/top1 应在同一噪声带内(同尺同语料, 差异=重排+路由翻转)。
# 用法: kernel_parity_spark.sh <旧二进制> <新二进制> [ids=g7/wt2.ids] [锚=r30/anchor_wt2_s2653.bin] [标签]
# 产物: gguf/go-onebit/vqhalf/champ86amp/speed/parity_<标签>_{old,new,new2}.bin + parity_<标签>.txt
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
OLD="${1:?旧二进制}"; NEW="${2:?新二进制}"
IDS="${3:-$ROOT/gguf/go-onebit/g7/wt2.ids}"; ANC="${4:-$ROOT/gguf/go-onebit/r30/anchor_wt2_s2653.bin}"
TAG="${5:-$(date +%m%d_%H%M)}"
# $6 = 工作区(模型 gguf/ds4-<ws>.gguf + 链 vqhalf/<ws>/zchain.bin), 默认冠军; 09-06 加: q4_K 核改动只在
# Q4_K 骨架模型(champ86q4k)上被执行, 对着冠军模型对拍等于没测。
WS="${6:-champ86amp}"
# $7 = 模式(09-06 加): decode(默认)=--score-ids 逐 token 解码路; prefill=--eval-ids 批路(512 token 块,
# 走 GEMM prefill 核)。批路 --eval-logits 出裸 f32 [S][V] 无头, --eval-no-bos 让它与 score-ids 喂同一
# id 流(首 id 原样, 不插 BOS) ⇒ 行与行对齐; 这里补 8 字节头(int32 S,V)转成 score 格式, 三个文件同构,
# anchor_metrics 的 --student / --ref-raw 都能直接吃。
MODE="${7:-decode}"
V=129280   # DS4_N_VOCAB(eval-logits 无头, 补头要它); score-ids 产物自带
SP="$ROOT/gguf/go-onebit/vqhalf/$WS/speed"; mkdir -p "$SP"
[ -s "gguf/ds4-$WS.gguf" ] && [ -s "gguf/go-onebit/vqhalf/$WS/zchain.bin" ] || { echo "★工作区 $WS 缺模型或链★"; exit 2; }
M=(--cuda -m "gguf/ds4-$WS.gguf" --zchain "gguf/go-onebit/vqhalf/$WS/zchain.bin" --mem-budget-mb 110000)
LOG(){ echo "[parity $(date +%H:%M:%S)] $*"; }
BUSY=$(for p in ds4 ds4-bench ds4-server ds4quant_run zlayer; do pgrep -x "$p"; done)
[ -z "$BUSY" ] || { LOG "★机器非空: $BUSY★"; exit 3; }
OUT="$SP/parity_$TAG.txt"; : > "$OUT"
score(){ # $1=二进制 $2=产物(score 格式: int32 S,V + f32[S*V])
    if [ "$MODE" = prefill ]; then
        "$1" "${M[@]}" --eval-ids "$IDS" --eval-no-bos --eval-logits "$2.raw" -n 1 -p x </dev/null > "$2.log" 2>&1 \
            || { LOG "★$1 eval-ids 失败(见 $2.log)★"; exit 4; }
        local n=$(( $(stat -c %s "$2.raw") / (V * 4) ))
        { perl -e 'print pack("V2", @ARGV)' "$n" "$V"; cat "$2.raw"; } > "$2"; rm -f "$2.raw"
    else
        "$1" "${M[@]}" --score-ids "$IDS" --score-out "$2" > "$2.log" 2>&1 || { LOG "★$1 score 失败(见 $2.log)★"; exit 4; }
    fi
}
LOG "ids=$(wc -l < "$IDS") 行, 锚=$(basename "$ANC"), 模式=$MODE"
score "$NEW" "$SP/parity_${TAG}_new.bin";  LOG "new 完"
score "$NEW" "$SP/parity_${TAG}_new2.bin"; LOG "new2 完"
if cmp -s "$SP/parity_${TAG}_new.bin" "$SP/parity_${TAG}_new2.bin"; then echo "决定论: 新二进制两遍逐字节全同 ✓" | tee -a "$OUT"
else echo "★决定论破: 新二进制两遍不同★" | tee -a "$OUT"; fi
score "$OLD" "$SP/parity_${TAG}_old.bin";  LOG "old 完"
for v in old new; do
    echo "== $v vs FP 锚 ==" | tee -a "$OUT"
    gguf-tools/bench/anchor_metrics --ref "$ANC" --ids "$IDS" --student "$SP/parity_${TAG}_$v.bin" 2>&1 | grep -v "^$" | tail -7 | tee -a "$OUT"
done
echo "== new vs old ==" | tee -a "$OUT"
gguf-tools/bench/anchor_metrics --ref-raw "$SP/parity_${TAG}_old.bin" --ids "$IDS" --student "$SP/parity_${TAG}_new.bin" 2>&1 | grep -v "^$" | tail -6 | tee -a "$OUT"
LOG "收官: $OUT"
