#!/usr/bin/env bash
# prefill_2048_ruler.sh — 速度战役每一轮都对这一把尺(2026-09-15 起)
#
# 干什么: 用金融判决料的前 2048 个 token 做一次教师强制打分(--score-ids, 解码路 100% 决定论),
# 同时拿到「预填秒数 / t.s」和「NLL / PPL」。前三轮(NVFP4 稠密 / 注意力分块 / 专家融合)
# 都是手打命令出的数, 没落盘 —— 这个脚本就是把那把尺固化下来, 以后每轮直接对。
#
# 为什么要两个数一起报: 09-15 撞过一次教训 —— 注意力核里无效 topk 槽读了 shared 残留,
# 0×NaN 让同一份输入温度 0 跑出四个 PPL, 于是"第三轮 PPL 反而更好"这种话是假账。
# 所以脚本默认跑 REPS 遍, 先证明 PPL 逐遍相同, 再谈快慢。
#
# 出错会怎样: PPL 逐遍不同 = 引擎又不确定了, 这一轮的质量读数一个都不许用, 先回去挖 bug;
# 看不到 "PPL(本段" = --score-nll 没生效或者提前失败, 去看 .err。
#
# 用法: ./speed-bench/prefill_2048_ruler.sh [标签] [跑几遍, 默认 3] [分块, 默认 512] [模型, 默认现役]

set -u
cd "$(dirname "$0")/.." || exit 1

TAG="${1:-run}"
REPS="${2:-3}"
CHUNK="${3:-512}"
NTOK=2048

MODEL="${4:-gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-fp4.gguf}"   # 第 4 个参数换模型(换新 GGUF 对尺用)
# ★第 5 个参数: 额外旗标(2026-09-16 decode.md D2)★
# 为什么要它: 这把尺默认按块 512 走**预填**核, 量不到解码路的核。改解码核(注意力/mHC/GEMV)要判质量,
# 得让它一个 token 一块地走 —— `"--decoder-full" 1` 就是这个用法(分块给 1 + 关 CED)。
# 不关 CED 会怎样: 块 1 时除最后一块外都只跑到分界层、不出 logits, NLL 直接是错的(不报错)。
BIN="${6:-ds4}"                                                # 第 6 个参数换二进制(对照 A/B 用)
EXTRA="${5:-}"
AMP=gguf/v41/gr-fin-40-fp4
IDS_SRC=gguf/v41judge/finj_n8192.ids
OUT=/tmp/prefill-ruler
IDS="$OUT/finj_n$NTOK.ids"

mkdir -p "$OUT"
[ -f "$MODEL" ]   || { echo "★没有模型 $MODEL★"; exit 1; }
[ -f "$IDS_SRC" ] || { echo "★没有判决料 $IDS_SRC★"; exit 1; }
[ -x ./"$BIN" ]   || { echo "★没有 ./$BIN, 先 make cuda-spark★"; exit 1; }

# 判决料前 2048 个 id(空白分隔, 一行一个或一行多个都行)
tr -s ' \t\n' '\n' < "$IDS_SRC" | grep -E '^[0-9]+$' | head -n "$NTOK" > "$IDS"
[ "$(wc -l < "$IDS")" = "$NTOK" ] || { echo "★判决料不足 $NTOK 个 id★"; exit 1; }

echo "== 预填尺 [$TAG] ${NTOK} token / 分块 $CHUNK / $REPS 遍"
# ★秒数取引擎自己打的那个(「完成 S=2048 … 34.4s」), 不取 wall clock —— wall 里有 110 GB
# 模型装载, 首跑冷页缓存能占 70 s, 把它算进预填就是自己骗自己。
for r in $(seq 1 "$REPS"); do
  ./"$BIN" -m "$MODEL" --zchain "$AMP" --ctx 32768 --v41-chunk "$CHUNK" $EXTRA \
        --score-ids "$IDS" --score-nll "$OUT/$TAG.nll$r.txt" --score-no-logits \
        > "$OUT/$TAG.out$r" 2> "$OUT/$TAG.err$r"
  SEC=$(grep -ao "完成 S=$NTOK[^\\n]*[0-9.]\+s" "$OUT/$TAG.err$r" | grep -o "[0-9.]*s$" | tr -d s)
  NLLPPL=$(grep -a "全序列平均" "$OUT/$TAG.err$r" | tail -1)
  NLL=$(echo "$NLLPPL" | grep -o "平均 [0-9.]*" | grep -o "[0-9.]*")
  PPL=$(echo "$NLLPPL" | grep -o "PPL [0-9.]*" | grep -o "[0-9.]*")
  TS=$([ -n "$SEC" ] && echo "scale=1; $NTOK / $SEC" | bc || echo "?")
  printf "遍 %-2s  预填 %7s s  %6s t/s   NLL %-9s PPL %s\n" "$r" "${SEC:-?}" "$TS" "${NLL:-?}" "${PPL:-?}"
done

echo "== NLL/PPL 是否逐遍相同(不同则本轮质量读数作废, 回去挖不确定性)"
for r in $(seq 1 "$REPS"); do grep -a "全序列平均" "$OUT/$TAG.err$r" | tail -1; done | sort -u
