#!/bin/bash
# spec_hdump_ab_spark.sh — 投机(verify 批) vs 纯解码 的逐层出口 hc 对拍(spark 本机跑, 2026-09-07)。
# 为什么: spec 与 plain 温 0 输出分叉, 核清单已对齐仍分叉 ⇒ 要定位到层。两条路各跑一次同 prompt, --eval-hdump 让
# prefill/解码/verify 批都按同格式逐行追加 h_L%02u.bin(每行 = 一个 token 的 4×4096 f32 hc), 另有
# L99 嵌入 / L96 主机侧 logits / L97 层 0 入环前 kv 行 / L98 层 0 环里原始行 / h_pos.bin 位置边车。
# ★按位置对齐(链 39 教训)★: spec 的行含被拒草稿(token 就不是 plain 的), 按行号对 plain 会把它误判成分叉;
# 同一位置取 spec 最后一次写出 = 提交行。先扫每个位置的 logits(L96)与层 0/末层出口, 首个不同的位置再逐层找凶手层。
# 用法: spec_hdump_ab_spark.sh <标签> <投机二进制> [纯解码二进制=./ds4] [n=3] [提示文件|-] [drafter]
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
TAG="${1:?标签}"; SPECBIN="${2:?投机二进制}"; BIN="${3:-./ds4}"; N="${4:-3}"; PFILE="${5:--}"
WS=champ86q4k; DRAFT="${6:-gguf/ds4-dspark-ve-q4.gguf}"
D="$ROOT/gguf/go-onebit/vqhalf/$WS/hdump_$TAG"; rm -rf "$D"; mkdir -p "$D/plain" "$D/spec"
LOG(){ echo "[hdump_ab $(date +%H:%M:%S)] $*"; }
BUSY=$(for p in ds4 ds4-bench ds4-server; do pgrep -x "$p"; done); [ -z "$BUSY" ] || { LOG "★机器非空: $BUSY★"; exit 3; }
M=(--cuda -m "gguf/ds4-$WS.gguf" --zchain "gguf/go-onebit/vqhalf/$WS/zchain.bin" --mem-budget-mb 110000 --temp 0 -n "$N")
PROMPT="Explain in plain words how a transformer language model generates text one token at a time."
if [ "$PFILE" != "-" ]; then P=(--prompt-file "$PFILE" --ctx 8192); else P=(-p "$PROMPT"); fi   # 长上下文对拍(09-07): 传提示文件
"$BIN" "${M[@]}" --eval-hdump "$D/plain" "${P[@]}" > "$D/plain.txt" 2> "$D/plain.log"; LOG "plain rc=$? $(grep -h generation: "$D/plain.log" | tail -1)"
"$SPECBIN" "${M[@]}" --eval-hdump "$D/spec" --draft-gguf "$DRAFT" --spec "${P[@]}" > "$D/spec.txt" 2> "$D/spec.log"; LOG "spec rc=$? $(grep -h 'spec 账' "$D/spec.log" | cut -c1-90)"
cmp -s "$D/plain.txt" "$D/spec.txt" && LOG "文本全同" || LOG "文本分叉: $(cmp "$D/plain.txt" "$D/spec.txt" 2>&1 | head -1)"
ROWB=$((4*4096*4))
RP=$(( $(stat -c %s "$D/plain/h_L00.bin") / ROWB )); RS=$(( $(stat -c %s "$D/spec/h_L00.bin") / ROWB ))
NP=$(( $(stat -c %s "$D/plain/h_pos.bin") / 4 )); NS=$(( $(stat -c %s "$D/spec/h_pos.bin") / 4 ))
PP=$((RP - NP)); PS=$((RS - NS))   # 逐层文件里 prefill 行数(生成行之前); L96/97/98/99 只有生成行
VB=$(( $(stat -c %s "$D/plain/h_L96.bin") / NP ))   # logits 行字节
LOG "行数 plain=$RP(生成 $NP) spec=$RS(生成 $NS, 含被拒草稿) prefill 行 $PP/$PS"
mapfile -t POSP < <(od -An -tu4 -v -w4 "$D/plain/h_pos.bin"); mapfile -t POSS < <(od -An -tu4 -v -w4 "$D/spec/h_pos.bin")
declare -A LAST; for j in "${!POSS[@]}"; do LAST[${POSS[$j]}]=$j; done   # 位置 → spec 最后一次写出的行(提交行)
# cmp_row <plain 文件> <行 p> <spec 文件> <行 s> <行字节>: 该行两边逐字节同 ⇒ 0
cmp_row(){ cmp -s <(dd if="$1" bs="$5" skip="$2" count=1 2>/dev/null) <(dd if="$3" bs="$5" skip="$4" count=1 2>/dev/null); }
bad_pos=-1; bad_i=-1; bad_j=-1; bad_what=""
for i in $(seq 0 $((NP-1))); do
    q=${POSP[$i]}; j=${LAST[$q]:-}
    [ -n "$j" ] || { LOG "位置 $q: spec 没提交到这里(生成到 ${POSS[$((NS-1))]}), 扫描止于此"; break; }
    for chk in "L96 logits" "L00 层0出口" "L42 末层出口"; do
        f=${chk%% *}
        if [ "$f" = L96 ]; then cmp_row "$D/plain/h_$f.bin" "$i" "$D/spec/h_$f.bin" "$j" "$VB"
        else cmp_row "$D/plain/h_$f.bin" "$((PP+i))" "$D/spec/h_$f.bin" "$((PS+j))" "$ROWB"; fi
        [ $? -eq 0 ] || { bad_pos=$q; bad_i=$i; bad_j=$j; bad_what="$chk"; break 2; }
    done
done
if [ $bad_pos -lt 0 ]; then LOG "★逐位置对拍: 全部 $NP 个生成位置 logits/层0/末层出口全同★"; echo "HDUMP_AB_${TAG}_END"; exit 0; fi
LOG "★首个不同位置 $bad_pos(生成第 $((bad_i+1)) 个 token, plain 行 $bad_i / spec 提交行 $bad_j): 先在 $bad_what 发现★"
# 该位置逐层找凶手: 嵌入(99) → 层 0 入环前 kv(97)/环里行(98) → 每层 [注意力出口(50+il), 层出口(il)] → logits(96)
first=""
cmp_row "$D/plain/h_L99.bin" "$bad_i" "$D/spec/h_L99.bin" "$bad_j" "$ROWB" || first="L99(嵌入 = token 本身不同)"
[ -n "$first" ] || cmp_row "$D/plain/h_L97.bin" "$bad_i" "$D/spec/h_L97.bin" "$bad_j" 2048 || first="L97(层 0 入环前 kv 行)"
[ -n "$first" ] || cmp_row "$D/plain/h_L98.bin" "$bad_i" "$D/spec/h_L98.bin" "$bad_j" 2048 || first="L98(层 0 环里原始行)"
if [ -z "$first" ]; then
    for il in $(seq 0 42); do
        for tag in $(printf "%02d" $((50+il))) $(printf "%02d" $il); do
            cmp_row "$D/plain/h_L$tag.bin" "$((PP+bad_i))" "$D/spec/h_L$tag.bin" "$((PS+bad_j))" "$ROWB" && continue
            first="L$tag($([ $tag -ge 50 ] && echo "第 $((10#$tag-50)) 层注意力出口" || echo "第 $((10#$tag)) 层出口"))"; break 2
        done
    done
fi
[ -n "$first" ] || { cmp_row "$D/plain/h_L96.bin" "$bad_i" "$D/spec/h_L96.bin" "$bad_j" "$VB" || first="L96(logits: 43 层出口全同, 输出头不同轨)"; }
LOG "位置 $bad_pos 首个不同: ${first:-无(只有扫描项不同?)}"
echo "HDUMP_AB_${TAG}_END"
