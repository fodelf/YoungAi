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
# 用法: ./speed-bench/d1_kv_ring_gate.sh [基线二进制, 默认 ds4.base] [模型] [生成几个 token, 默认 64] [模式, 默认 gate] [反修目录]
#   基线二进制 = 改这一刀之前的 ./ds4, 改之前先 `cp ds4 ds4.base`。没有它就只能跑"新二进制自洽"那半张表。
#   换配方要**成对换**: 第 2 个参数给 GGUF、第 5 个给它自己那份反修目录(gr_Lnn.bin)。反修是按某一份量化文件的
#   残差解的, 挂到别的文件上不报错但数值全错 —— 所以只换了模型没给第 5 个参数直接拒跑。
set -u
cd "$(dirname "$0")/.." || exit 1

BASE="${1:-ds4.base}"
MODEL="${2:-gguf/v41/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf}"
NGEN="${3:-64}"
MODE="${4:-gate}"     # gate = 全套门; bisect = 只做同轨定位(见文件末尾那一段)
AMP="${5:-gguf/v41/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine}"
OUT=/tmp/d1-kv-ring        # 日志可以留 /tmp, 脚本必须在仓库里(铁律)
mkdir -p "$OUT"
# 上下文没有参数(用户 2026-09-22 "不要任何写死的上下文, 上下文大小只有 1M 这一个选择"): 引擎从模型元数据
# deepseek4.context_length 读, --ctx 已不存在(传了直接拒)。以前这里写死 32768 是 V4 会话时代的遗留。

[ -x ./ds4 ] || { echo "★没有 ./ds4, 先 make cuda-spark★"; exit 1; }
[ -f "$MODEL" ] || { echo "★没有模型 $MODEL★"; exit 1; }
# 第 5 个参数给 none = 显式裸模型(不挂反修, 用来跟挂了的成对比); 忘了传仍走默认值, 配对门照样拦。
if [ "$AMP" != none ]; then
  [ -d "$AMP" ] || { echo "★没有反修目录 $AMP★"; exit 1; }
  # 配对门: 反修 manifest 记着它是在哪份 GGUF 上解的(gguf=…), 去掉 .gguf 后必须是模型名的前缀 ——
  # 同一底座的 -mtpnative/-dspark 变体只多了三塔, 反修通用; 换了底座(fp4 ↔ q4k)就是另一份残差, 拒跑。
  MF=$(grep -ao 'gguf=[^ ]*' "$AMP"/manifest* 2>/dev/null | head -1 | sed 's/^gguf=//; s/\.gguf$//')
  case "$(basename "$MODEL" .gguf)" in "$(basename "$MF")"*) ;; *) echo "★反修 $AMP 是在 $MF 上解的, 不配 $MODEL★"; exit 1;; esac
fi
# ★只用真场景提示★(用户令 2026-09-16): 3 token 的短提示退役 —— 那时可见键才几十个, 随上下文涨的
# 那几个核(indexer 打分/top-k/注意力)一个都量不到, 而用户要的速度是真对话里的速度。
head -c  8000 speed-bench/readme_en_x80.txt > "$OUT/p2k.txt"  || exit 1
head -c 50000 speed-bench/readme_en_x80.txt > "$OUT/p12k.txt" || exit 1
# p60k(≈6 万 token, 与 d0a_decode_profile.sh 的长提示同一刀): 给 online 模式的第 6 个参数用, 量长上下文档位上
# 投机 vs 纯解码 —— 12k 尺看不到随上下文涨的那笔税(sp.md: 10k → 71k 每步 +3.8 ms), 投机在那一档赚不赚只有量了才知道。
head -c 200000 speed-bench/readme_en_x80.txt > "$OUT/p60k.txt" || exit 1

# ★看门狗(2026-09-24 补)★: 这把尺每趟都装 113 GB 模型进 121 GB 的 spark, 以前一道闸都没有 —— 真挤到 MemAvailable → 0
# 不是报错, 是整机假死(08-24 实撞)。红线与 serve_1m_spark.sh 同一个数: 每 2 s 看一次, 连续两次 < 2500 MB 就杀本趟,
# .err 末尾留一行"★看门狗★"。杀了的那趟读数作废(输出不完整, 门会报分叉/缺数), 不许拿它判核。没有 /proc(Mac)就不看。
WD_KILL_MB=2500
run () {   # $1=二进制 $2=标签 $3.. = 其余参数
  local bin=$1 tag=$2; shift 2
  local z=(); [ "$AMP" != none ] && z=(--zchain "$AMP")
  # ${z[@]+"${z[@]}"}: set -u 下空数组直接展开在老 bash 上报 unbound, 这个写法两边都过
  ./"$bin" -m "$MODEL" ${z[@]+"${z[@]}"} --temp 0 --seed 1 "$@" > "$OUT/$tag.out" 2> "$OUT/$tag.err" &
  local pid=$! bad=0 a
  while kill -0 "$pid" 2>/dev/null; do
    if [ -r /proc/meminfo ]; then
      a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
      if [ "$a" -lt "$WD_KILL_MB" ]; then bad=$((bad+1)); else bad=0; fi
      if [ "$bad" -ge 2 ]; then
        echo "★看门狗: MemAvailable ${a} MB < ${WD_KILL_MB} 连续两次, 杀 $bin($tag)★" | tee -a "$OUT/$tag.err"
        kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; break
      fi
    fi
    sleep 2
  done
  wait "$pid"
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
  run ds4 bis_plain -n "$NGEN" --emit-trace --no-dspark --prompt-file "$P"
  grep -a "^\[emit\]" "$OUT/bis_plain.err" > "$OUT/ids_plain.txt"
  echo "  纯解码 $(wc -l < "$OUT/ids_plain.txt") 个 token"
  for k in 1 3; do
    run ds4 "bis_k$k" -n "$NGEN" --emit-trace --dspark --dspark-verify "$k" --prompt-file "$P"
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
    # 没基线就把上次留下的 gbase_* 删掉: 下面的三方 cmp 按"文件在不在"决定比不比, 旧文件(可能是另一个模型跑的)
    # 会被当基线比出一个假红(2026-09-19 q4k 门实撞: 拿 09-18 的 fp4 输出当基线, 报"gbase ≠ gdirect")。
    if [ -x "./$BASE" ]; then run "$BASE" "gbase_$tag" -n "$NGEN" --no-dspark --prompt-file "$P"
    else rm -f "$OUT/gbase_$tag.out" "$OUT/gbase_$tag.err"; fi
    run ds4 "gdirect_$tag" -n "$NGEN" --no-dspark --no-graph --prompt-file "$P"
    run ds4 "ggraph_$tag"  -n "$NGEN" --no-dspark --prompt-file "$P"
    run ds4 "ggraph2_$tag" -n "$NGEN" --no-dspark --prompt-file "$P"
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
# 第 6 个参数 = 换提示文件。★短提示那格必跑★(2026-09-23 实撞): 给 speed-bench/fin_chat_prompt.txt(105 token)+ NGEN 1200 ——
# 短提示预填几乎不长索引草稿, 捕获时要按桶上限扩容; 09-22 夜捕获先置了"捕获态"再扩容, 自己把自己拒掉, 服务端第一条
# 22 token 冒烟就撞上、整夜直发(每步 +3 ms)。2K 提示预填长出的余量碰巧够, 这格门一直绿, 所以它单独要一格。
# 日志里出现"捕获失败"直接判红: 那说明走图那趟其实在直发, 两边逐字节同也不代表图路是好的。
if [ "$MODE" = graphx ]; then
  P="${6:-$OUT/p2k.txt}"
  run ds4 "gxdirect" -n "$NGEN" --no-dspark --no-graph --prompt-file "$P"
  run ds4 "gxgraph"  -n "$NGEN" --no-dspark --prompt-file "$P"
  echo "  桶: $(grep -a -h '位置桶\|走图解了' "$OUT/gxgraph.err" | tr '\n' ' ')"
  echo "  $(grep -a -h 'decode .* token' "$OUT/gxdirect.err") (直发) | $(grep -a -h 'decode .* token' "$OUT/gxgraph.err") (图)"
  if grep -a -q '捕获失败' "$OUT/gxgraph.err"; then echo "  ★图路捕获失败, 实际走了直发★ $(grep -a -h '捕获' "$OUT/gxgraph.err" | head -2 | tr '\n' ' ')"; exit 1; fi
  if cmp -s "$OUT/gxdirect.out" "$OUT/gxgraph.out"; then echo "  跨桶: 直发 == 图 逐字节同 ✓"; exit 0
  else echo "  ★跨桶分叉★ $(cmp "$OUT/gxdirect.out" "$OUT/gxgraph.out" 2>&1 | head -1)"; exit 1; fi
fi

# ★MODE=online(2026-09-19): 同一条真提示上 投机 vs 纯解码 的整段 t/s + 调度器逐轮的账★
# 为什么单列一档: 门的 64 步只够判"同不同", 判不了"赚不赚" —— 投机是按整段文本赚钱的(歇轮、k 的分布、接受的分布都随文本走)。
# 提示 = speed-bench/fin_chat_prompt.txt(聊天口径, 引擎套模板; 模型写白酒龙头半年报, 出 1400 token 思考文本)。
# --emit-trace 让引擎每出一次草稿打一行 `[dspark] pos k val acc conf c0..c4`(core_v41_api.c), 这里按 conf[0] 分十档对实际
# 首位接受率 —— 校准表: 调度器的每一个判决(选 k、歇不歇)都是拿 conf 算的, conf 不准判决就是错的; 另打 k 的分布与逐位条件接受率。
# 用法: ./speed-bench/d1_kv_ring_gate.sh none <模型> 1400 online [反修目录] [提示文件, 默认金融提示] [nok]
#   第 6 个参数换提示(如 $OUT/p60k.txt 量长上下文档位); 第 7 个给 nok = 不跑下面钉死 k=1/k=3 的两趟 ——
#   那两趟只为解调度器常数(直发验 1 行 / 每多一行), 长提示上每趟预填要好几分钟, 常数又不随提示变, 不必每次付。
if [ "$MODE" = online ]; then
  P="${6:-speed-bench/fin_chat_prompt.txt}"; NOK="${7:-}"
  [ -f "$P" ] || { echo "★没有 $P★"; exit 1; }
  run ds4 onl_plain -n "$NGEN" --no-dspark --prompt-file "$P"
  run ds4 onl_spec  -n "$NGEN" --dspark --emit-trace --prompt-file "$P"
  echo "== 在线: $(basename "$P"), $NGEN token"
  echo "  纯解码: $(grep -a -h 'decode .* token' "$OUT/onl_plain.err") $(grep -a -h '稳态' "$OUT/onl_plain.err" | sed 's/.*⇒ //')"
  echo "  投机:   $(grep -a -h 'decode .* token' "$OUT/onl_spec.err") $(grep -a -h '稳态' "$OUT/onl_spec.err" | sed 's/.*⇒ //')"
  grep -a -h "DSpark:\|一轮 \|暂存换过指针\|走图解了" "$OUT/onl_spec.err" | sed 's/^/  /'
  A="$OUT/onl_spec.out"; B="$OUT/onl_plain.out"
  na=$(wc -c < "$A"); nb=$(wc -c < "$B"); n=$(( (na < nb ? na : nb) - 1 ))
  if cmp -s -n "$n" "$A" "$B"; then echo "  同轨: 前 $n 字节逐字节同 ✓ (投机 $na / 纯解码 $nb 字节)"
  else echo "  ★同轨分叉★ $(cmp -n "$n" "$A" "$B" 2>&1 | head -1)"; fi
  # 钉死 k=1 / k=3 各一趟: 两趟的"一轮 ms"解出 直发验 1 行 与 每多验一行 的钱(验证 1+k 行 = v1 + tok·k), 喂给陪审团与调度器常量
  if [ "$NOK" = nok ]; then echo "  (nok: 跳过钉死 k=1/k=3 两趟)"; else
  for k in 1 3; do
    run ds4 "onl_k$k" -n "$NGEN" --dspark --dspark-verify "$k" --prompt-file "$P"
    echo "  钉死 k=$k: $(grep -a -h 'decode .* token' "$OUT/onl_k$k.err") $(grep -a -h 'DSpark:\|一轮 ' "$OUT/onl_k$k.err" | tr '\n' ' ')"
  done
  fi
  echo "  调度器账(conf[0] 十分位档 → k≥1 的轮里实际首位接受率; 歇 = 预测比值 <1 的轮):"
  grep -a '^\[dspark\] pos' "$OUT/onl_spec.err" | awk '
    { k=$5; v=$7; a=$9; c0=$11; b=int(c0*10); if (b>9) b=9;
      n[b]++; if (k>=1) { n1[b]++; if (a>=1) hit[b]++ } sk[b]+=k; sa[b]+=a; if (v<1) rest[b]++;
      N++; if (k==0) k0++; kh[k]++;
      if (k>=1) { r1++; if (a>=1) h1++ } if (k>=2 && a>=1) { r2++; if (a>=2) h2++ } if (k>=3 && a>=2) { r3++; if (a>=3) h3++ } }
    END { for (b=0;b<10;b++) if (n[b]) printf "    conf0 %.1f~%.1f: %4d 轮(k≥1 %4d) 首位接受 %.3f 均k %.2f 均接受 %.2f 歇 %d\n",
                                       b/10, (b+1)/10, n[b], n1[b], n1[b]?hit[b]/n1[b]:0, sk[b]/n[b], sa[b]/n[b], rest[b];
          printf "    合计 %d 轮出草稿(其中 k=0 白跑 %d); 逐位条件接受 p1 %.3f(%d 轮) p2|1 %.3f(%d) p3|2 %.3f(%d); k 分布", N, k0, r1?h1/r1:0, r1, r2?h2/r2:0, r2, r3?h3/r3:0, r3;
          for (k=0;k<=5;k++) printf " %d:%d", k, kh[k]; printf "\n" }'
  exit 0
fi

# ★MODE=loop(2026-09-20): 温 0 贪心"打转"尺 —— 同一条金融提示上纯解码一趟, 数文本自己重复了多少★
# 为什么单列: 五指标是 teacher-forced(每一步都喂教师的正确前文), 看不见自由生成里"翻一个 token 之后一路错下去"
# 的病; 09-20 凌晨在这条提示上两档都从 40% 处开始循环"营收约？…净利润约？"(模型记不住数字, 贪心锁在占位符上)。
# 读数: "约？"次数 / 重复 4-gram 占比(按字; 全文里出现 ≥2 次的 4-gram 所占的位置比 —— 与 fable5 09-20 上午那张表同定义:
# gr-only 73%, gr+rb 76%; 中文正常长文 ~10%) / 最常见 4-gram / 每 300 字一段的"前文已出现过"占比(100% = 整段逐字抄前文,
# 一眼看出从哪段起进入死循环) / 解码 t/s。文本原样落 $OUT/loop_<模型>_<反修>.out, 判读一律看原文, 数字只是索引。
# 用法: ./speed-bench/d1_kv_ring_gate.sh none <模型> <生成上限, 0 = 不设上限> loop <反修目录|none> [提示文件, 默认金融提示]
#       (三档成对: 新+反修 / 新裸 / 现役+反修; 第 6 个参数给 $OUT/p2k.txt = 英文 readme 对照, 那条提示上正常长文不打转)
#       第 7 个参数 = 额外引擎参数(如 "--no-graph"), 输出文件名带 _x<参数去掉空格和横线>: 同一提示直发 1400 步与走图
#       逐字节 cmp, 就是"长程解码态有没有漂"的门(graph 模式的门只验了 128 步)。
if [ "$MODE" = loop ]; then
  P="${6:-speed-bench/fin_chat_prompt.txt}"; EXTRA="${7:-}"
  [ -f "$P" ] || { echo "★没有 $P★"; exit 1; }
  tag="loop_$(basename "$MODEL" .gguf)_$(basename "$AMP")_$(basename "$P" .txt)"
  [ -n "$EXTRA" ] && tag="${tag}_x$(echo "$EXTRA" | tr -d ' -')"
  # --emit-trace 只往 stderr 打 `[ptok] 位置 id`(套过模板的提示) 与 `[emit] 位置 id`(每个吐出的 token), 不改采样。
  # 留下精确 id 序列是为了 d1 模式(把这段文本喂教师锚)。★别用 --dump-tokens 拼提示★: 它按原文分词, 生成路却是
  # build_prompt 套了聊天模板的(09-20 实撞 75 vs 79), 铁律"捕获必须可复现·与部署同路" ⇒ id 只认引擎自己打的。
  # shellcheck disable=SC2086  # EXTRA 就是要按空格拆成多个参数
  # 生成数给 0 = 不传 -n = 生成到 EOS(判"停不停/复读"必须这么跑, 带上限只能说"到上限没停")
  NCAP=(); [ "$NGEN" -gt 0 ] && NCAP=(-n "$NGEN")
  run ds4 "$tag" ${NCAP[@]+"${NCAP[@]}"} --no-dspark --emit-trace --prompt-file "$P" $EXTRA
  grep -a -h '^\[ptok\] \|^\[emit\] ' "$OUT/$tag.err" | sort -n -k2 | awk '{print $3}' > "$OUT/$tag.ids"
  NP=$(grep -ac '^\[ptok\] ' "$OUT/$tag.err"); echo "$NP" > "$OUT/$tag.np"
  [ "$NP" -gt 0 ] || echo "  ★没有 [ptok] 行: ds4 二进制早于 09-20 夜的 --emit-trace 提示打点, ids 缺提示段★"
  echo "== 打转尺: $(basename "$MODEL") + $AMP, $(basename "$P"), $NGEN token → $OUT/$tag.out (ids $(wc -l < "$OUT/$tag.ids") = 提示 $NP + 生成 $(grep -ac '^\[emit\] ' "$OUT/$tag.err"))"
  echo "  $(grep -a -h 'decode .* token' "$OUT/$tag.err") $(grep -a -h '稳态' "$OUT/$tag.err" | sed 's/.*⇒ //')"
  echo "  字节 $(wc -c < "$OUT/$tag.out"), \"约？\" $(grep -ao '约？' "$OUT/$tag.out" | wc -l) 次"
  # 按字切(grep -o . 在 UTF-8 locale 下一行一个字, 中文不被拆成字节), 再在 awk 里拼 4-gram 计数。
  LC_ALL=C.utf8 grep -ao . "$OUT/$tag.out" | awk '
    { c[NR]=$0 }
    END { n=NR-3; if (n<1) { print "  文本太短, 不算 4-gram"; exit }
          for (i=1;i<=n;i++) { g[i]=c[i] c[i+1] c[i+2] c[i+3]; s=int((i-1)/300); tot[s]++; if (cnt[g[i]]++) seen[s]++ }
          for (i=1;i<=n;i++) if (cnt[g[i]]>1) rep++
          best=""; bc=0; for (k in cnt) if (cnt[k]>bc) { bc=cnt[k]; best=k }
          printf "  重复 4-gram 占比 %.1f%% (%d/%d), 最常见 \"%s\"×%d\n", 100*rep/n, rep, n, best, bc
          printf "  每 300 字一段, 前文已出现过的 4-gram 占比:"; for (s=0;s<=int((n-1)/300);s++) printf " %d", int(100*seen[s]/tot[s]+0.5); printf "\n" }'
  # 英文提示按词算(字级 4-gram 在英文里天然重复过半, 没有分辨力): 09-20 上午 readme 对照的读数 9.5% 就是词级。
  tr -s '[:space:]' '\n' < "$OUT/$tag.out" | awk '
    { w[NR]=$0 }
    END { n=NR-3; if (n<1) exit; for (i=1;i<=n;i++) { g[i]=w[i] " " w[i+1] " " w[i+2] " " w[i+3]; cnt[g[i]]++ }
          for (i=1;i<=n;i++) if (cnt[g[i]]>1) rep++; printf "  词级重复 4-gram 占比 %.1f%% (%d/%d 词)\n", 100*rep/n, rep, n }'
  # 第一个"约？"落在全文的百分之几(字节口径够用): 09-20 凌晨两档都是 40% 处起, 起点变了说明改的是"何时开始记不住", 不只是次数。
  b=$(grep -abo '约？' "$OUT/$tag.out" | head -1 | cut -d: -f1); t=$(wc -c < "$OUT/$tag.out")
  [ -n "$b" ] && echo "  第一个\"约？\"在 $((100 * b / t))% 处(字节 $b / $t)"
  exit 0
fi

# ★MODE=d1(2026-09-20 夜): "老师在复读位选谁" —— 判温 0 复读是模型本性还是量化/侧车的病★
# 前置: loop 模式落的 <tag>.ids(+ .np = 提示 token 数); 教师锚与学生 logits 由 v41_judge.sh <ids> <n> engine:<模型>:<反修目录> 产
#   在 gguf/v41judge/(teacher_<语料tag>_n<n>.bin / stu_<语料tag>_n<n>_eng_<模型>_amp_<反修>.bin)。
# 读法: 先在生成段 ids 里找复读周期 P(尾部 ids[i]==ids[i-P] 的最小 P), 把生成段切成 首次出现(第 1 遍)/第 2 遍/第 3 遍…;
#   anchor_metrics --row-out 每行 = 位置 kld smin same 老师命中 学生命中("命中" = argmax == 实际吐出的下一个 token)。
#   第 2 遍起"老师命中"高 = FP 给同样前文也会抄 ⇒ 复读是模型本性, 五指标奖励"像老师"就等于奖励复读;
#   "老师命中"低而"学生命中"高 = 老师会写别的, 学生自己锁死 ⇒ 量化/侧车的病。
# 用法: ./speed-bench/d1_kv_ring_gate.sh none <模型> 0 d1 none <ids> <teacher.bin> <student.bin>
if [ "$MODE" = d1 ]; then
  IDS="${6:?ids 文件}"; TB="${7:?教师 logits}"; SB="${8:?学生 logits}"
  NP=$(cat "${IDS%.ids}.np" 2>/dev/null); [ -n "$NP" ] || { echo "★缺 ${IDS%.ids}.np(提示 token 数), 先跑 loop 模式★"; exit 1; }
  [ -x ./gguf-tools/bench/anchor_metrics ] || make -C gguf-tools anchor_metrics || exit 1
  ./gguf-tools/bench/anchor_metrics --ref-raw "$TB" --ids "$IDS" --student "$SB" --row-out "$OUT/d1_rows.txt" | grep -E "==|Σmin|KLD|Same"
  awk -v NP="$NP" '
    FNR==NR { ids[NR-1]=$1; N=NR; next }
    { kld[$1]=$2; same[$1]=$4; rh[$1]=$5; sh[$1]=$6; next }
    END {
      NG=N-NP; for (j=0;j<NG;j++) g[j]=ids[NP+j]
      P=0; for (p=16; p<=int(NG/2) && !P; p++) { ok=1; for (i=NG-p;i<NG;i++) if (g[i]!=g[i-p]) { ok=0; break } if (ok) P=p }
      if (!P) { print "  尾部没有复读周期(P 16~" int(NG/2) " 内无解)"; exit }
      s=NG-P; while (s-1>=P && g[s-1]==g[s-1-P]) s--
      printf "  复读周期 P=%d token; 复读区 = 生成第 %d~%d token(%.1f 遍), 首次出现 = 第 %d~%d\n", P, s, NG-1, (NG-s)/P, s-P, s-1
      k=1; for (b=s-P; b<NG; b+=P) { e=b+P; if (e>NG) e=NG; n=0; a=0; c=0; d=0; q=0
        for (j=b;j<e;j++) { r=NP+j-1; if (r in same) { n++; a+=same[r]; c+=rh[r]; d+=sh[r]; q+=kld[r] } }
        printf "  第 %d 遍(生成 %d~%d, n=%d): 老师命中 %.1f%%  学生命中 %.1f%%  老师=学生 %.1f%%  KLD %.4f\n", k, b, e-1, n, n?100*c/n:0, n?100*d/n:0, n?100*a/n:0, n?q/n:0; k++ }
      printf "  第 2 遍开头 12 个 token 逐位 老师命中/学生命中/KLD:"; for (j=s; j<s+12 && j<NG; j++) { r=NP+j-1; printf " %d/%d/%.2f", rh[r], sh[r], kld[r] } printf "\n"
    }' "$IDS" "$OUT/d1_rows.txt"
  exit 0
fi

# ★MODE=dflt(2026-09-24): 投机翻成默认开之后的三格门★ —— 同一条真实请求(第 6 参数 = 提示 ids, 按 id 喂):
#   ①什么都不传(默认开): 输出 == 基线纯解码逐字节(第 7 参数 = 基线 .out, 例 base0924_plain.out), 且吐的 token 数恰好 = NGEN
#     (一轮接受多位时以前会越过上限, 服务端 max_tokens 是硬上限);
#   ②默认 + --temp 0.6: 不报错、走纯解码、日志有"投机只在贪心下成立"那一行(服务端温 1.0 的请求不能因默认值被拒);
#   ③显式 --dspark + --temp 0.6: 照旧硬拒(rc ≠ 0)。
# 用法: ./speed-bench/d1_kv_ring_gate.sh none <模型> 2048 dflt <反修目录> <提示 ids> <基线纯解码 .out>
if [ "$MODE" = dflt ]; then
  PIDS="${6:?提示 ids}"; BOUT="${7:?基线纯解码 .out}"
  fail=0
  run ds4 dflt_greedy -n "$NGEN" --emit-trace --gen-ids "$PIDS"
  ne=$(grep -ac '^\[emit\] ' "$OUT/dflt_greedy.err")
  echo "  ①默认: $(grep -a -h 'decode .* token\|DSpark:' "$OUT/dflt_greedy.err" | tr '\n' ' ')"
  grep -aq "DSpark: [1-9]" "$OUT/dflt_greedy.err" || { echo "  ★默认没走投机(没有 DSpark 汇总行)★"; fail=1; }
  [ "$ne" = "$NGEN" ] && echo "  吐 $ne token = 上限 ✓" || { echo "  ★吐 $ne token ≠ 上限 $NGEN★"; fail=1; }
  if cmp -s "$OUT/dflt_greedy.out" "$BOUT"; then echo "  == 基线纯解码逐字节同 ✓"; else echo "  ★与基线不同★ $(cmp "$OUT/dflt_greedy.out" "$BOUT" 2>&1 | head -1)"; fail=1; fi
  run ds4 dflt_temp -n 64 --temp 0.6 --gen-ids "$PIDS"; rc=$?
  if [ "$rc" = 0 ] && grep -aq "投机只在贪心下成立" "$OUT/dflt_temp.err"; then echo "  ②默认 + 采样: 走纯解码, 有日志 ✓"
  else echo "  ★默认 + 采样: rc=$rc, 日志: $(grep -a '投机\|dspark' "$OUT/dflt_temp.err" | head -2)★"; fail=1; fi
  run ds4 dflt_explicit -n 64 --dspark --temp 0.6 --gen-ids "$PIDS"; rc=$?
  if [ "$rc" != 0 ] && grep -aq "不能同开" "$OUT/dflt_explicit.err"; then echo "  ③显式 --dspark + 采样: 拒 ✓"
  else echo "  ★显式 --dspark + 采样没拒(rc=$rc)★"; fail=1; fi
  [ "$fail" = 0 ] && echo "门: 全绿" || echo "门: ★有红★"
  exit "$fail"
fi

# ★MODE=sim(2026-09-19): 调度器的离线陪审团 —— 接在 online 之后跑★
# ①从 online 那趟投机的 [emit] 轨迹(位置 + id)拼出整段真 id 序列(提示 token 走 --dump-tokens, 与生成同一条渲染路);
# ②教师强制取料(--dspark-capture): 每个位置都出一块草稿 + conf, 落 <pairs>.fix; ③gguf-tools/bench/dspark_sim 在主机上重放
# 任何调度策略 × 成本假设的整段 ms/token。为什么: 在线只看得到调度器自己走过的位置, 歇着的 58% 的步该不该歇, 只有取料能判。
# 成本(ms)取 online 那趟的实测: 走图一步与草稿一轮从日志解析; "直发验 1 行 42.5 / 每多一行 12.95" 是 09-18 钉死 k=1/k=3 两趟
# 解出来的(fable5 09-18 断档诊断对照表: 一轮 69.4 / 95.3 ms), 没法从混合 k 的均值里解出来, 写死在这。
# 用法: ./speed-bench/d1_kv_ring_gate.sh none <模型> 0 sim
#
# ★真实请求档(2026-09-24): 第 6 个参数给提示 ids 文件 ⇒ 在线四趟 + 取料 + 重放全在这条请求上做★
# 为什么: fin_chat_prompt 是测速基准(凭记忆写的数), 不是产品场景(铁律 09-21); 投机赚不赚全看接受率, 接受率随文本走,
# 所以只认真实请求。真实请求只有 id 是准的(文本重新分词拼不回去), 一律 --gen-ids 喂, 生成段 id 取纯解码那趟的 [emit]。
# 验证批的钱不再用 09-18 写死的 42.5/12.95: 钉死 k=1/k=3 两趟的"验证"项 V1/V3 解出(验证 1+k 行 = v1 + tok·k)
#   tok = (V3 − V1)/2, v1 = V1 − tok。09-23 的提速全是 n=1 核, 验证批核没动, 旧常数早过期。
# 第 7 个参数(可选) = 假设"验证批核补齐到 n=1 水平"时每多验一行的 ms(= 一行的专家账), 再重放一遍: 验 1 行 = 走图一步。
#   这一档回答的是"把验证批核做好以后投机能到多少", 不是现状。
# 第 8 个参数给 nocap = 只跑在线四趟(速度 + 同轨 + 验证批成本)就停: 改核后复测用, 取料(逐位 ~15 min)只在要重放时才付。
# 第 9 个参数(2026-09-24) = 跑哪个二进制(默认 ds4): 改核前后成对比必须同一机器状态两个二进制各跑一遍, 跨会话的 t/s 不可比。
# 用法: ./speed-bench/d1_kv_ring_gate.sh none <模型> <生成几个 token> sim <反修目录> <提示 ids> [理想每行 ms] [nocap] [二进制] [额外引擎参数]
# ★head 档(2026-09-24)★: 一条真实请求(id 文件)纯解码生成 NGEN 个 token, 把正文开头打出来 —— 看"模型开口说什么"(思考用什么语言、
# 电报体还是完整句)用的, 分钟级。第 7 个参数 = 追加的引擎参数(例 "--decoder-full" 做预填路 A/B), 第 8 个 = 输出标签。
# 用法: ./speed-bench/d1_kv_ring_gate.sh none <模型> <生成几个 token> head <反修目录> <提示 ids> [引擎参数] [标签]
if [ "$MODE" = head ] && [ -n "${6:-}" ]; then
  HX=(${7:-}); HT="head_${8:-$(basename "$6" .ids)}"
  [ -s "$6" ] || { echo "★没有 $6★"; exit 1; }
  run ds4 "$HT" -n "$NGEN" --no-dspark --gen-ids "$6" ${HX[@]+"${HX[@]}"}
  echo "== $HT($(basename "$6"), ${7:-默认}): $(grep -a -h 'prefill .* token' "$OUT/$HT.err" | tail -1)"
  head -c 600 "$OUT/$HT.out"; echo
  exit 0
fi
if [ "$MODE" = sim ] && [ -n "${6:-}" ]; then
  PIDS="$6"; TOKI="${7:-}"; NOCAP="${8:-}"; SBIN="${9:-ds4}"
  # 第 10 个参数(2026-09-24) = 四趟在线都追加的引擎参数(空格隔开), 例 "--no-graph"(同一二进制换一种发法做 A/B)
  SEXTRA=(${10:-})
  [ -x "./$SBIN" ] || { echo "★没有 ./$SBIN★"; exit 1; }
  [ -s "$PIDS" ] || { echo "★没有 $PIDS★"; exit 1; }
  [ "$NGEN" -gt 0 ] || { echo "★生成 token 数要 > 0★"; exit 1; }
  [ -x ./gguf-tools/bench/dspark_sim ] || make -C gguf-tools dspark_sim || exit 1
  t=rq_$(basename "$PIDS" .ids)
  # 第 8 个参数给 speconly(2026-09-24) = 只跑投机一趟: 同一条请求上比几种只影响投机的开关时用, 每种省下纯解码/k1/k3 三趟;
  # 同轨对上一次 nocap 留下的纯解码输出比(那些开关不改纯解码, 不用重跑它)。
  if [ "$NOCAP" = speconly ]; then
    [ -s "$OUT/${t}_plain.out" ] || { echo "★speconly 要先有 $OUT/${t}_plain.out(先对这条请求跑一次 nocap)★"; exit 1; }
    run "$SBIN" "${t}_spec" -n "$NGEN" --dspark --emit-trace --gen-ids "$PIDS" ${SEXTRA[@]+"${SEXTRA[@]}"}
    echo "== 真实请求 $(basename "$PIDS"), 只跑投机, 二进制 $SBIN ${SEXTRA[*]:-}"
    echo "  spec: $(grep -a -h 'decode .* token' "$OUT/${t}_spec.err") $(grep -a -h 'DSpark:\|一轮 \|子词表' "$OUT/${t}_spec.err" | tr '\n' ' ')"
    A="$OUT/${t}_spec.out"; B="$OUT/${t}_plain.out"
    na=$(wc -c < "$A"); nb=$(wc -c < "$B"); n=$(( (na < nb ? na : nb) - 1 ))
    if cmp -s -n "$n" "$A" "$B"; then echo "  同轨: 前 $n 字节逐字节同 ✓"; else echo "  ★同轨分叉★ $(cmp -n "$n" "$A" "$B" 2>&1 | head -1)"; fi
    exit 0
  fi
  run "$SBIN" "${t}_plain" -n "$NGEN" --no-dspark --emit-trace --gen-ids "$PIDS" ${SEXTRA[@]+"${SEXTRA[@]}"}
  run "$SBIN" "${t}_spec"  -n "$NGEN" --dspark --emit-trace --gen-ids "$PIDS" ${SEXTRA[@]+"${SEXTRA[@]}"}
  for k in 1 3; do run "$SBIN" "${t}_k$k" -n "$NGEN" --dspark --dspark-verify "$k" --gen-ids "$PIDS" ${SEXTRA[@]+"${SEXTRA[@]}"}; done
  echo "== 真实请求 $(basename "$PIDS"), 生成 $NGEN token, 二进制 $SBIN"
  for x in plain spec k1 k3; do
    echo "  $x: $(grep -a -h 'decode .* token' "$OUT/${t}_$x.err") $(grep -a -h '稳态' "$OUT/${t}_$x.err" | sed 's/.*⇒ //') $(grep -a -h 'DSpark:\|一轮 ' "$OUT/${t}_$x.err" | tr '\n' ' ')"
  done
  A="$OUT/${t}_spec.out"; B="$OUT/${t}_plain.out"
  na=$(wc -c < "$A"); nb=$(wc -c < "$B"); n=$(( (na < nb ? na : nb) - 1 ))
  if cmp -s -n "$n" "$A" "$B"; then echo "  同轨: 前 $n 字节逐字节同 ✓"; else echo "  ★同轨分叉, 下面的重放不作数★ $(cmp -n "$n" "$A" "$B" 2>&1 | head -1)"; fi
  for k in 1 3; do
    echo "  k=$k 验证项: $(grep -a -h '一轮 ' "$OUT/${t}_k$k.err" | sed 's/.*验证 \([0-9.]*\).*/\1/') ms"
  done
  [ "$NOCAP" = nocap ] && { echo "  (nocap: 不取料不重放)"; exit 0; }
  # 整段 id = 提示(原样) + 纯解码那趟吐的; 取料每个位置出一块草稿, 与这一串逐位比
  { tr -s ' \t' '\n\n' < "$PIDS" | grep -v '^$'; grep -a '^\[emit\] ' "$OUT/${t}_plain.err" | sort -n -k2 | awk '{print $3}'; } > "$OUT/$t.sim.ids"
  NP=$(tr -s ' \t' '\n\n' < "$PIDS" | grep -c .)
  echo "  取料: ids $(wc -l < "$OUT/$t.sim.ids") = 提示 $NP + 生成 $(( $(wc -l < "$OUT/$t.sim.ids") - NP ))"
  z=(); [ "$AMP" != none ] && z=(--zchain "$AMP")
  ./"$SBIN" -m "$MODEL" ${z[@]+"${z[@]}"} --score-ids "$OUT/$t.sim.ids" --dspark-capture "$OUT/$t.pairs.bin" --decoder-full \
      > /dev/null 2> "$OUT/$t.cap.err" || { echo "★取料失败★"; tail -3 "$OUT/$t.cap.err"; exit 3; }
  grep -a "首位 ↔ 底座" "$OUT/$t.cap.err"
  GMS=$(grep -a -h '稳态' "$OUT/${t}_plain.err" | sed 's/.*稳态 \([0-9.]*\) ms.*/\1/')
  DMS=$(grep -a -h '一轮 ' "$OUT/${t}_spec.err" | sed 's/.*草稿 \([0-9.]*\).*/\1/')
  V1=$(grep -a -h '一轮 ' "$OUT/${t}_k1.err" | sed 's/.*验证 \([0-9.]*\).*/\1/')
  V3=$(grep -a -h '一轮 ' "$OUT/${t}_k3.err" | sed 's/.*验证 \([0-9.]*\).*/\1/')
  [ -n "$GMS" ] && [ -n "$DMS" ] && [ -n "$V1" ] && [ -n "$V3" ] || { echo "★成本没解析出来(GMS=$GMS DMS=$DMS V1=$V1 V3=$V3), 看 .err★"; exit 4; }
  TOK=$(awk -v a="$V1" -v b="$V3" 'BEGIN{printf "%.2f", (b-a)/2}'); VV1=$(awk -v a="$V1" -v t="$TOK" 'BEGIN{printf "%.2f", a-t}')
  echo "  成本(实测): 走图一步 $GMS / 草稿一轮 $DMS / 验 1 行 $VV1 / 每多一行 $TOK ms"
  echo "== 重放: 现状成本"
  ./gguf-tools/bench/dspark_sim "$OUT/$t.pairs.bin.fix" "$OUT/$t.sim.ids" "$GMS" "$DMS" "$VV1" "$TOK" 4 "$NP" | tee "$OUT/$t.sim.txt"
  if [ -n "$TOKI" ]; then
    echo "== 重放: 假设验证批核补齐(验 1 行 = 走图一步 $GMS, 每多一行 $TOKI)"
    ./gguf-tools/bench/dspark_sim "$OUT/$t.pairs.bin.fix" "$OUT/$t.sim.ids" "$GMS" "$DMS" "$GMS" "$TOKI" 4 "$NP" | tee "$OUT/$t.sim_ideal.txt"
  fi
  exit 0
fi
if [ "$MODE" = sim ]; then
  P=speed-bench/fin_chat_prompt.txt
  [ -s "$OUT/onl_spec.err" ] || { echo "★先跑 online 模式★"; exit 1; }
  [ -x ./gguf-tools/bench/dspark_sim ] || make -C gguf-tools dspark_sim || exit 1
  ./ds4 -m "$MODEL" --zchain "$AMP" --prompt-file "$P" --dump-tokens > "$OUT/sim.tok" 2> "$OUT/sim.tok.err" || { echo "★dump-tokens 失败★"; exit 2; }
  { head -1 "$OUT/sim.tok" | tr -d '[] ' | tr ',' '\n'; grep -a '^\[emit\] ' "$OUT/onl_spec.err" | sort -n -k2 | awk '{print $3}'; } | grep -v '^$' > "$OUT/sim.ids"
  NP=$(head -1 "$OUT/sim.tok" | tr ',' '\n' | wc -l); NG=$(grep -ac '^\[emit\] ' "$OUT/onl_spec.err")
  echo "== sim: ids $(wc -l < "$OUT/sim.ids") = 提示 $NP + 生成 $NG"
  ./ds4 -m "$MODEL" --zchain "$AMP" --score-ids "$OUT/sim.ids" --dspark-capture "$OUT/sim_pairs.bin" --decoder-full \
      > /dev/null 2> "$OUT/sim_cap.err" || { echo "★取料失败★"; tail -3 "$OUT/sim_cap.err"; exit 3; }
  grep -a "首位 ↔ 底座" "$OUT/sim_cap.err"
  GMS=$(grep -a -h '稳态' "$OUT/onl_plain.err" | sed 's/.*稳态 \([0-9.]*\) ms.*/\1/'); DMS=$(grep -a -h '一轮 ' "$OUT/onl_spec.err" | sed 's/.*草稿 \([0-9.]*\).*/\1/')
  ./gguf-tools/bench/dspark_sim "$OUT/sim_pairs.bin.fix" "$OUT/sim.ids" "${GMS:-37.97}" "${DMS:-13.5}" 42.5 12.95 16 "$NP" | tee "$OUT/sim.txt"
  exit 0
fi

# 四个场景各自要抓的病: 2K = 环绕过很多圈; 12k = 稀疏路(indexer/top-k/压缩 KV)全量到;
# spec/spec12k = 投机路(验证批的小批核 + 部分接受的回滚)。
# ★--dspark 要显式传★(2026-09-16): 投机默认已改成**关**(同轨没绿之前引擎默认路径必须是裸模型)。
CASES="ctx2k:--no-dspark|--prompt-file|$OUT/p2k.txt
ctx12k:--no-dspark|--prompt-file|$OUT/p12k.txt
spec:--dspark|--prompt-file|$OUT/p2k.txt
spec12k:--dspark|--prompt-file|$OUT/p12k.txt"

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
  # ★投机必须真跑了才算★(2026-09-18 实撞): 草稿器在线静默失效(一个缓冲区太小)时一轮投机都没有, 输出当然与纯解码
  # 逐字节同 —— "同轨"是空真, 门照样全绿, 只有 t/s 和缺失的 DSpark 汇总行露馅。所以先看汇总行(轮数 > 0)。
  if ! grep -aq "DSpark: [1-9]" "$OUT/new_$sp.err"; then
    echo "  $nm ★投机一轮都没跑(没有 DSpark 汇总行), 同轨这格是空真, 判无效★"; fail=1
  else
    echo "  $nm $(grep -a "DSpark:" "$OUT/new_$sp.err" | tail -1 | sed 's/^\[v41\] //')"
  fi
  na=$(wc -c < "$A"); nb=$(wc -c < "$B"); n=$(( (na < nb ? na : nb) - 1 ))
  if [ "$n" -lt 16 ]; then echo "  $nm ★两边几乎没输出($na/$nb 字节), 判不了★"; fail=1
  elif cmp -s -n "$n" "$A" "$B"; then echo "  $nm 前 $n 字节逐字节同 ✓ (投机 $na / 纯解码 $nb 字节)"
  else echo "  $nm ★分叉★ $(cmp -n "$n" "$A" "$B" 2>&1 | head -1)"; fail=1; fi
done

echo
[ "$fail" = 0 ] && echo "门: 全绿" || echo "门: ★有分叉, 别报速度★"
exit "$fail"
