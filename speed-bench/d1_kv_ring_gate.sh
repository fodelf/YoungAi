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
MODE="${4:-gate}"     # gate = 全套门; bisect = 只做同轨定位(见文件末尾那一段)
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

# ★同轨定位(mtp-1.md M0′(d))★: 门报"分叉"之后, 下一个问题永远是"从哪一步开始的"。
# 办法: --v41-prof 下引擎逐 token 打一行 `[emit] <绝对位置> <token id>`, 两条路 diff 一下,
# 第一行不同的地方就是要查的那一步。再按 k 逐档跑(k=1 是最小的多 token 批), 就能分清:
#   k=1 就分叉 ⇒ 病在"批"本身(验证批那条核路与单 token 路算出来的 logits 不同);
#   k=1 不分叉、k=3 才分叉 ⇒ 病在"批大了以后"或**部分接受的回滚**(k=1 时要么全接受要么全拒, 回滚最简单);
#   位置本身对不上(不是 id 不同而是行数/位置错位) ⇒ 快照漏项, 状态被推歪了。
# 用法: ./speed-bench/d1_kv_ring_gate.sh ds4.base <模型> 64 bisect
if [ "$MODE" = bisect ] || [ "$MODE" = bisect12k ]; then
  P="$OUT/p2k.txt"; [ "$MODE" = bisect12k ] && P="$OUT/p12k.txt"
  echo "== 同轨定位: $(basename "$P") 提示, 纯解码 vs 投机(k=1 / k=3), 逐 token 比 [emit] 行"
  run ds4 bis_plain -n "$NGEN" --emit-trace --no-dspark --ctx 32768 --prompt-file "$P"
  grep -a "^\[emit\]" "$OUT/bis_plain.err" > "$OUT/ids_plain.txt"
  echo "  纯解码 $(wc -l < "$OUT/ids_plain.txt") 个 token"
  for k in 1 3; do
    run ds4 "bis_k$k" -n "$NGEN" --emit-trace --dspark --dspark-verify "$k" --ctx 32768 --prompt-file "$P"
    grep -a "^\[emit\]" "$OUT/bis_k$k.err" > "$OUT/ids_k$k.txt"
    echo "  投机 k=$k: $(wc -l < "$OUT/ids_k$k.txt") 个 token; $(grep -a -h 'DSpark:' "$OUT/bis_k$k.err" | tail -1)"
    # 第一处不同: 逐行比, 打印两边各三行上下文
    d=$(diff "$OUT/ids_plain.txt" "$OUT/ids_k$k.txt" | head -8)
    if [ -z "$d" ]; then echo "    ★逐 token 全同 ✓★"
    else echo "    ★第一处不同★"; echo "$d" | sed 's/^/      /'; fi
  done
  exit 0
fi

# ★MODE=graph(2026-09-18, fable5 09-18 立案): 解码整步 CUDA graph 的门★
# 三方 cmp: 基线二进制 / 新二进制 --no-graph(直发) / 新二进制默认(走图), 2K 与 12k 各一遍, 外加图路自洽两跑。
# 为什么门是逐字节: 图只是同一批核的另一种发法, 位置从设备槽读、组号/段数核里自算 —— 一个字节不同就是
# 某个核把位置读错了(桶上限当真值、垃圾槽被读、段数算岔), 不是噪声。t/s 一并打出(同机器状态三条路各一份)。
# 用法: ./speed-bench/d1_kv_ring_gate.sh ds4.base_pf <模型> 64 graph   (NGEN 给 128 可让 2K 提示跨 2048 那个桶边界)
if [ "$MODE" = graph ]; then
  fail=0
  for c in "ctx2k:$OUT/p2k.txt" "ctx12k:$OUT/p12k.txt"; do
    tag="${c%%:*}"; P="${c#*:}"
    echo "== $tag"
    [ -x "./$BASE" ] && run "$BASE" "gbase_$tag" -n "$NGEN" --no-dspark --ctx 32768 --prompt-file "$P"
    run ds4 "gdirect_$tag" -n "$NGEN" --no-dspark --no-graph --ctx 32768 --prompt-file "$P"
    run ds4 "ggraph_$tag"  -n "$NGEN" --no-dspark --ctx 32768 --prompt-file "$P"
    run ds4 "ggraph2_$tag" -n "$NGEN" --no-dspark --ctx 32768 --prompt-file "$P"
    for pair in "gdirect:ggraph" "ggraph:ggraph2" "gbase:gdirect"; do
      a="${pair%%:*}"; b="${pair#*:}"
      [ -f "$OUT/${a}_$tag.out" ] || continue
      if cmp -s "$OUT/${a}_$tag.out" "$OUT/${b}_$tag.out"; then echo "  $a == $b 逐字节同 ✓"
      else echo "  ★$a ≠ $b★ $(cmp "$OUT/${a}_$tag.out" "$OUT/${b}_$tag.out" 2>&1 | head -1)"; fail=1; fi
    done
    for t in gbase gdirect ggraph ggraph2; do
      [ -f "$OUT/${t}_$tag.err" ] && echo "  $t: $(grep -a -h 'prefill .* token\|decode .* token' "$OUT/${t}_$tag.err" | tr '\n' ' ') $(grep -a -h '\[graph\]' "$OUT/${t}_$tag.err" | tr '\n' ' ')"
    done
  done
  [ "$fail" = 0 ] && echo "门: 全绿" || echo "门: ★有分叉★"
  exit "$fail"
fi
# ★MODE=graphx: 跨桶重捕获的门★ —— 2K 提示(≈1846 token)生成 NGEN(给 ≥256)步, 位置越过 2048 那个桶边界时图要重捕获;
# 判据仍是走图 vs --no-graph 逐字节同, 外加日志里"位置桶"要出现两次(没出现第二次 = 没跨到, 把 NGEN 加大)。
if [ "$MODE" = graphx ]; then
  P="$OUT/p2k.txt"
  run ds4 "gxdirect" -n "$NGEN" --no-dspark --no-graph --ctx 32768 --prompt-file "$P"
  run ds4 "gxgraph"  -n "$NGEN" --no-dspark --ctx 32768 --prompt-file "$P"
  echo "  桶: $(grep -a -h '位置桶\|走图解了' "$OUT/gxgraph.err" | tr '\n' ' ')"
  echo "  $(grep -a -h 'decode .* token' "$OUT/gxdirect.err") (直发) | $(grep -a -h 'decode .* token' "$OUT/gxgraph.err") (图)"
  if cmp -s "$OUT/gxdirect.out" "$OUT/gxgraph.out"; then echo "  跨桶: 直发 == 图 逐字节同 ✓"; exit 0
  else echo "  ★跨桶分叉★ $(cmp "$OUT/gxdirect.out" "$OUT/gxgraph.out" 2>&1 | head -1)"; exit 1; fi
fi

# 四个场景各自要抓的病: 2K = 环绕过很多圈; 12k = 稀疏路(indexer/top-k/压缩 KV)全量到;
# spec/spec12k = 投机路(验证批的小批核 + 部分接受的回滚)。
# ★--dspark 要显式传★(2026-09-16): 投机默认已改成**关**(同轨没绿之前引擎默认路径必须是裸模型)。
CASES="ctx2k:--no-dspark|--ctx|32768|--prompt-file|$OUT/p2k.txt
ctx12k:--no-dspark|--ctx|32768|--prompt-file|$OUT/p12k.txt
spec:--dspark|--ctx|32768|--prompt-file|$OUT/p2k.txt
spec12k:--dspark|--ctx|32768|--prompt-file|$OUT/p12k.txt"

fail=0
for c in $CASES; do
  tag="${c%%:*}"; args="${c#*:}"
  IFS='|' read -r -a A <<< "$args"
  echo "== $tag"
  # ★投机场景不跟基线比★: 旧二进制不认识 `--dspark`(那时投机是默认开的), 喂给它直接 exit 2;
  # 而且投机路本来就在改, 跟旧二进制比没有意义。投机的判据是下面那张"同轨"表。
  usebase=1; case "$tag" in spec*) usebase=0;; esac
  if [ -x "./$BASE" ] && [ "$usebase" = 1 ]; then
    run "$BASE" "base_$tag" -n "$NGEN" "${A[@]}"
  fi
  run ds4 "new_$tag" -n "$NGEN" "${A[@]}"
  # ★自洽: 同一条路再跑一遍★(2026-09-16)。没有这一格, 下面"同轨分叉"根本判不了 ——
  # 这台机器上有一条 09-12 定罪、至今没了结的账: 103 GiB 注册映射的尾页偶发脏读(memory
  # spark_mmap_tail_glitch_backbone_first_cache), 它会让**同一条路**两跑也不同, 一次一个 token。
  # 所以顺序是: 先证"自己跟自己一样", 再去比"投机跟纯解码一样"。自洽不绿 = 先修引擎, 别查投机。
  run ds4 "rep_$tag" -n "$NGEN" "${A[@]}"
  if cmp -s "$OUT/new_$tag.out" "$OUT/rep_$tag.out"; then echo "  自洽(两跑同) ✓"
  else echo "  ★自洽就不绿★ $(cmp "$OUT/new_$tag.out" "$OUT/rep_$tag.out" 2>&1 | head -1)"; fail=1; fi
  if [ -x "./$BASE" ] && [ "$usebase" = 1 ]; then
    if cmp -s "$OUT/base_$tag.out" "$OUT/new_$tag.out"; then echo "  逐字节同 ✓"
    else echo "  ★与基线分叉★ $(cmp "$OUT/base_$tag.out" "$OUT/new_$tag.out" 2>&1 | head -1)"; fail=1; fi
  elif [ "$usebase" = 1 ]; then
    echo "  (无基线 $BASE, 只跑新二进制)"
  fi
  grep -a -h "decode .* token\|DSpark:\|一轮 " "$OUT/new_$tag.err" | sed 's/^/  /'
done

# ★同轨门(2026-09-16 mtp-1.md M1′)★: 同一个二进制, 同一份提示, 投机 vs 纯解码必须**逐字节同**。
# 这与上面那半张表是两件事: 上面比的是"新旧两个二进制的同一条路", 这里比的是"同一个二进制的两条路"。
# 为什么必须同: 投机的接受条件就是"主模型自己也会选这个 token", 所以它只该省时间、不该改一个字。
# 不同 = 验证批那条路与单 token 路算出来的 logits 不一样(核/分段/累加序), 或者回滚漏了状态。
# 出错会怎样: 分叉处两句往往都通顺(近平局翻面), 看输出根本发现不了 —— 只有 cmp 抓得到。
echo
echo "== 同轨(投机 == 纯解码, 同一个二进制)"
# ★只比公共前缀, 而且要**再去掉一个字节**★: 投机是按轮产出的, 最后一轮可能一次吐出好几个 token,
# 于是总数会比 -n 给的多几个(日志里"decode 65 token" vs "63 token" 就是这么来的)。
# 短的那份文件最后一个字节是生成结束时补的换行(run_v41_generation 的 fputc('\n')), 而长的那份
# 在同一位置是下一个 token 的首字符 —— 直接比 min(长度) 会在公共前缀的**最后一字节**上报假分叉。
# ★这条坑实撞过★(2026-09-16): 为此白查了两轮"第二个不同轨源", 而逐 token 的 id 序列其实完全一致。
# 要判的是"吐出来的 token 序列一样不一样", 不是文件一样长。
for pair in "p2k:spec:ctx2k" "p12k:spec12k:ctx12k"; do
  nm="${pair%%:*}"; rest="${pair#*:}"; sp="${rest%%:*}"; pl="${rest#*:}"
  A="$OUT/new_$sp.out"; B="$OUT/new_$pl.out"
  na=$(wc -c < "$A"); nb=$(wc -c < "$B"); n=$(( (na < nb ? na : nb) - 1 ))
  if [ "$n" -lt 16 ]; then echo "  $nm ★两边几乎没输出($na/$nb 字节), 判不了★"; fail=1
  elif cmp -s -n "$n" "$A" "$B"; then echo "  $nm 前 $n 字节逐字节同 ✓ (投机 $na / 纯解码 $nb 字节)"
  else echo "  $nm ★分叉★ $(cmp -n "$n" "$A" "$B" 2>&1 | head -1)"; fail=1; fi
done

echo
[ "$fail" = 0 ] && echo "门: 全绿" || echo "门: ★有分叉, 别报速度★"
exit "$fail"
