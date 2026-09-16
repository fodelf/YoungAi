#!/usr/bin/env bash
# d1_kv_ring_gate.sh — decode.md D1 的门: SWA 窗口改环之后, 温 0 输出必须与改造前**逐字节相同**。
#
# 这一刀改了什么: 每层的"最近 128 个 token 的 KV"原来存在一条排好序的线性缓冲里, 每算完一步就把
# 整段往前挪一行(每层两发 256 KB 拷贝, 40 层一步白搬 20 MB)。现在按官方的存法改成**环**——
# 位置 a 恒住在 a % 128 那一格, 新行直接盖掉 128 步之前那一行, 一个字节都不用挪。
#
# 为什么门是"逐字节同"而不是 NLL: 这一刀只改"某个键住在第几行", 读到的还是同一个键、同样的顺序、
# 同样的累加序 —— 数值上应当是恒等的。只要有一个字节不同, 就是行号映射错了(读到了别的位置的键),
# 不是"噪声带内", 不许放过。
#
# 出错会怎样(症状 → 真因):
#   短提示同、2K 提示分叉  ⇒ 环绕过一圈之后才错 = 取模那一步的边界(pos0 附近的行去了块区还是环)
#   纯解码同、投机分叉      ⇒ 回滚没把"没被接受的那几格"还原回去(v41_spec_rollback 的 ①)
#   两个二进制都跑不出来    ⇒ 先看 .err, 多半是 ds4.base 不在(下面会提示怎么造)
#
# 用法: ./speed-bench/d1_kv_ring_gate.sh [基线二进制, 默认 ds4.base] [模型] [生成几个 token, 默认 64]
#   基线二进制 = 改这一刀之前的 ./ds4, 改之前先 `cp ds4 ds4.base`。没有它就只能跑"新二进制自洽"那半张表。
set -u
cd "$(dirname "$0")/.." || exit 1

BASE="${1:-ds4.base}"
MODEL="${2:-gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-fp4-dspark.gguf}"
NGEN="${3:-64}"
AMP=gguf/v41/gr-fin-40-fp4
OUT=/tmp/d1-kv-ring        # 日志可以留 /tmp, 脚本必须在仓库里(铁律)
mkdir -p "$OUT"

[ -x ./ds4 ] || { echo "★没有 ./ds4, 先 make cuda-spark★"; exit 1; }
[ -f "$MODEL" ] || { echo "★没有模型 $MODEL★"; exit 1; }
# ★只用真场景提示★(用户令 2026-09-16): 3 token 的短提示退役 —— 那时可见键才几十个, 随上下文涨的
# 那几个核(indexer 打分/top-k/注意力)一个都量不到, 而用户要的速度是真对话里的速度。
head -c  8000 speed-bench/readme_en_x80.txt > "$OUT/p2k.txt"  || exit 1
head -c 50000 speed-bench/readme_en_x80.txt > "$OUT/p12k.txt" || exit 1

run () {   # $1=二进制 $2=标签 $3.. = 其余参数
  local bin=$1 tag=$2; shift 2
  ./"$bin" -m "$MODEL" --zchain "$AMP" --temp 0 --seed 1 "$@" > "$OUT/$tag.out" 2> "$OUT/$tag.err"
}

# 三个场景各自要抓的病: 2K = 环绕过很多圈; 12k = 稀疏路(indexer/top-k/压缩 KV)全量到; spec = 部分接受的回滚路
CASES="ctx2k:--no-dspark|--ctx|32768|--prompt-file|$OUT/p2k.txt
ctx12k:--no-dspark|--ctx|32768|--prompt-file|$OUT/p12k.txt
spec:--ctx|32768|--prompt-file|$OUT/p2k.txt"

fail=0
for c in $CASES; do
  tag="${c%%:*}"; args="${c#*:}"
  IFS='|' read -r -a A <<< "$args"
  echo "== $tag"
  if [ -x "./$BASE" ]; then
    run "$BASE" "base_$tag" -n "$NGEN" "${A[@]}"
  fi
  run ds4 "new_$tag" -n "$NGEN" "${A[@]}"
  if [ -x "./$BASE" ]; then
    if cmp -s "$OUT/base_$tag.out" "$OUT/new_$tag.out"; then echo "  逐字节同 ✓"
    else echo "  ★分叉★ $(cmp "$OUT/base_$tag.out" "$OUT/new_$tag.out" 2>&1 | head -1)"; fail=1; fi
  else
    echo "  (无基线 $BASE, 只跑新二进制)"
  fi
  grep -a -h "decode .* token\|DSpark:" "$OUT/new_$tag.err" | sed 's/^/  /'
done
echo
[ "$fail" = 0 ] && echo "D1 门: 全绿" || echo "D1 门: ★有分叉, 别报速度★"
exit "$fail"
