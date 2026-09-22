#!/bin/bash
# v41_dspark_gen_ids.sh — 让引擎温 0 生成一段文本, 拼出"提示 + 生成"的 id 序列(一行一个 id), 给夹具链当料(2026-09-18)。
#
# 为什么要它: 投机在线赚的是**模型自己生成的文本**, 教师强制在它上面量的首位一致率才是线上 p1 的口径;
# 人写的语料(公告/快讯)是下界, 不是用户场景。提示走引擎自己的口径: 文件以 <｜begin▁of▁sentence｜> 开头
# 就当已渲染的裸文本续写(cli_gen.c is_rendered_chat_prompt), 否则套聊天模板(用户问一句, 模型答)。
# 用法: gguf-tools/scripts/v41_dspark_gen_ids.sh <提示文件> <生成 token 数> <输出 ids>
#   产物: <ids>(一行一个 id) <ids>.tok(提示 token 化) <ids>.txt(生成正文) <ids>.err(引擎日志, 含 [emit] 轨迹)
# 出错会怎样: 模型提前吐 EOS 就没那么多 token, 输出行数会少于 提示+N —— 下游按实际行数取 ntok, 别按 N 假设。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1
P="${1:?用法: $0 <提示文件> <生成 token 数> <输出 ids>}"
N="${2:?}"
OUT="${3:?}"
MODEL=gguf/v41/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf
ZCHAIN=gguf/v41/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine
[ -x ./ds4 ] || { echo "★没有 ./ds4★"; exit 1; }
# 提示的 token id(与生成那趟同一个提示、同一个渲染路 ⇒ 同一串 id)
./ds4 -m "$MODEL" --zchain "$ZCHAIN" --prompt-file "$P" --dump-tokens > "$OUT.tok" 2> "$OUT.tok.err" || { echo "★dump-tokens 失败★"; exit 2; }
# 生成: 温 0 + [emit] 逐 token 轨迹(位置 id)
./ds4 -m "$MODEL" --zchain "$ZCHAIN" --prompt-file "$P" -n "$N" --temp 0 --seed 1 --emit-trace > "$OUT.txt" 2> "$OUT.err" || { echo "★生成失败★"; exit 3; }
{ head -1 "$OUT.tok" | tr -d '[] ' | tr ',' '\n'; grep -a '^\[emit\] ' "$OUT.err" | awk '{print $3}'; } | grep -v '^$' > "$OUT"
NP=$(head -1 "$OUT.tok" | tr ',' '\n' | wc -l); NG=$(grep -ac '^\[emit\] ' "$OUT.err")
echo "$OUT: $(wc -l < "$OUT") ids = 提示 $NP + 生成 $NG; $(grep -a 'decode .*t/s' "$OUT.err" | tail -1)"
