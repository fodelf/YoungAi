#!/usr/bin/env bash
# prefill_ttft_ruler.sh — 真实预填尺: 长提示进去, 量"到第一个字"的秒数与 t/s
#
# 为什么要另立一把尺: 原来的 prefill_2048_ruler.sh 走 --score-ids(教师强制逐位打分), 它需要**每个位置**
# 的 logits。而 CED(官方 §2.2/§3.2.2)的全部省法就是"提示的中间块不跑解码器段", 那些位置的 logits 就没有了 ——
# 于是 CED 在打分尺上一秒都省不下来, 量了等于没量。用户要的 300 t/s 是"长提示多快能开始吐字",
# 这把尺量的就是那个。
#
# 怎么读: 引擎自己打的 `[v41] prefill N token X.Xs (Y t/s)` 就是答案。
# 同时跑 --decoder-full(关 CED, 每块跑满 40 层)作对照, 两边的**生成文本**要拿来对比 ——
# CED 是官方认可的近似(报告原话 not mathematically equivalent), 不是逐位等价, 所以判据是
# "答案还对不对", 不是逐字节同。
#
# 出错会怎样: CED 那侧答非所问 / 复读 = 分界层的全局 KV 没写进去(v41_attention_kv_only 漏了),
# 表现是模型只看得见最后一块的上下文, 不报错。两边 t/s 一样 = ced_skip 没生效(块数只有 1, 把提示加长)。
#
# 用法: ./speed-bench/prefill_ttft_ruler.sh [提示文件] [截多少字符, 默认 40000] [块, 默认 512] [模型, 默认现役]

set -u
cd "$(dirname "$0")/.." || exit 1

SRC="${1:-speed-bench/promessi_sposi.txt}"
CHARS="${2:-40000}"
CHUNK="${3:-512}"
MODEL="${4:-gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-fp4.gguf}"   # 第 4 个参数换模型
AMP=gguf/v41/gr-fin-40-fp4
OUT=/tmp/ttft
P="$OUT/prompt.txt"

mkdir -p "$OUT"; rm -f "$OUT"/*.out "$OUT"/*.err
[ -f "$MODEL" ] || { echo "★没有模型 $MODEL★"; exit 1; }
[ -f "$SRC" ]   || { echo "★没有提示文件 $SRC★"; exit 1; }
head -c "$CHARS" "$SRC" > "$P"

run() {   # $1 = 标签, $2.. = 额外参数
  local tag="$1"; shift
  ./ds4 -m "$MODEL" --zchain "$AMP" --ctx 32768 --v41-chunk "$CHUNK" --temp 0 --seed 1 -n 16 \
        "$@" --prompt-file "$P" > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  printf "%-14s %s\n" "$tag" "$(grep -ao 'prefill [0-9]* token [0-9.]*s ([0-9.]* t/s)' "$OUT/$tag.err" | tail -1)"
}

echo "== 真实预填尺: $(wc -c < "$P") 字符 / 块 $CHUNK"
run ced                       # 默认 = CED 开(官方部署语义)
run full --decoder-full       # 对照 = 每块跑满 40 层

echo "== 生成文本(CED 是近似, 判据是答案还对不对, 不是逐字节同)"
echo "-- CED :"; head -c 300 "$OUT/ced.out";  echo
echo "-- FULL:"; head -c 300 "$OUT/full.out"; echo
