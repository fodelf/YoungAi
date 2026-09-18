#!/usr/bin/env bash
# d0a_decode_profile.sh — 段 1: 解码 124 ms/token 里, 除了读权重的那 25 ms, 剩下 100 ms 是谁吃的
#
# 背景(speed.md §3.2/§6): 解码每 token 必读 6.0 GB 权重, 板子实测带宽 240 GB/s ⇒ 字节账只值 25 ms,
# 也就是 40 t/s 的墙。引擎今天 8~9.6 t/s = 124 ms/token, 离墙 4 倍 —— 那 100 ms 从来没人量过,
# 前三轮全在优化预填, 解码一根手指没动(9.64 → 9.43)。
#
# 这个脚本干什么: 对一次短解码做两件事
#   ① nsys 采一段, 出逐核耗时表(cuda_gpu_kern_sum)与 API 表(cuda_api_sum) —— 核内 vs 主机往返
#   ② 引擎自己的 --v41-prof 逐层毫秒 —— 哪一层贵, engram 层(L01/L14)与源层(L02/08/14/20)是不是异常
# 两张表一对, 就能把 124 ms 拆成「核内耗时 / 发射间隙 / 主机往返 / engram 同步 pread」四项。
#
# 怎么读结果: 核时长之和 ≈ 总时长 ⇒ 瓶颈在核内(去融合、去改核形态);
# 核时长之和 ≪ 总时长 ⇒ 瓶颈是发射间隙/主机往返(上 CUDA graph, 这是段 2 的主菜);
# cudaMemcpy/cudaStreamSynchronize 占大头 ⇒ 主机往返, 逐项去掉。
#
# 出错会怎样: nsys 报 "CPU IP/backtrace sampling not supported" 是正常的(容器里没权限), 不影响 GPU 表;
# 出不来 kern_sum = 采样窗口里没跑到解码(提示太长, 时间全在预填), 把 -n 调大或提示调短。
#
# 用法: ./speed-bench/d0a_decode_profile.sh [生成几个 token, 默认 32] [nsys? yes/no, 默认 yes] [模型]
#                                          [长提示文件, 默认空=3 token 短提示] [截多少字符, 默认 40000]
#   第二个参数给 no = 只跑 ①(引擎逐层毫秒), 秒级出数 —— 每改一版解码核就对这一把, 不用等 nsys。
#   ★第 6/7/8 个参数(ncu 模式专用)★: 核正则 / 采几发 / 跳几发。例:
#     ./speed-bench/d0a_decode_profile.sh 8 ncu <模型> "" 40000 regex:v41_vq_gateup_kernel 6 200
#   ★第四个参数(2026-09-16 single-1.md)★: 给提示文件 = 在**长上下文**下分账。为什么要它: 短提示(3 token)
#   下可见键才 ~70 个, 注意力/indexer 那几个随上下文涨的核全量不到 —— 12k 提示下一步 211 ms 比短上下文
#   多出来的 157 ms, 在短尺上一个字都看不见。用户要的 40 t/s 是在真对话里要的, 那里这些核才是主菜。
#   长提示模式下表里前面几十步是**预填块**(发射数上万), 解码步在最后 NGEN 行, 看表尾。

set -u
cd "$(dirname "$0")/.." || exit 1

NGEN="${1:-32}"
DO_NSYS="${2:-yes}"
# ★第 9 个参数: 换二进制★(2026-09-16) —— 判"这一刀让哪个核变快/变慢"必须同机器状态两个二进制各跑一遍,
# 跨会话的 t/s 不可比(memory: 换 GGUF/换会话 = 换机器状态)。默认 ds4, 给 ds4.base 就是量改造前那份。
BIN="${9:-ds4}"
MODEL="${3:-gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-fp4.gguf}"
PSRC="${4:-}"
PCHARS="${5:-40000}"
AMP=gguf/v41/gr-fin-40-fp4
OUT=/tmp/d0a-decode-${BIN}
PROMPT="你好世界"     # 3 token: 预填几乎不占时间, 采到的基本全是解码

mkdir -p "$OUT"
[ -f "$MODEL" ] || { echo "★没有模型 $MODEL★"; exit 1; }
[ -x "./$BIN" ] || { echo "★没有 ./$BIN, 先 make cuda-spark★"; exit 1; }

# 提示: 默认短提示走 -p; 给了文件就截 PCHARS 字符走 --prompt-file(与 prefill_ttft_ruler.sh 同一口径)
PARG=(-p "$PROMPT")
if [ -n "$PSRC" ]; then
  [ -f "$PSRC" ] || { echo "★没有提示文件 $PSRC★"; exit 1; }
  head -c "$PCHARS" "$PSRC" > "$OUT/prompt.txt"
  PARG=(--ctx 32768 --prompt-file "$OUT/prompt.txt")
  echo "== 长提示模式: $PSRC 前 $PCHARS 字符"
fi
# ★仍然显式传 --no-dspark★: 投机的默认值 2026-09-16 已改成**关**(见 core_v41_api.c 那段注释),
# 但这把尺量的是"纯单 token 解码每步多少毫秒", 把它写死在命令行里, 以后默认值再变也量不错。
# 历史上 single.md §5 就把这个默认值写反了, 换个带三塔的文件就悄悄量成另一条路, 数字还长得挺像。
#
# ★第二个参数给 spec = 量投机路★(2026-09-16, mtp-1.md M0′(a)): nsys 照采, 但开 --dspark。
# 为什么要它: 投机一轮的成本得拆到核上才知道是"草稿器贵"还是"验证批的核形态贵" ——
# 之前这张表只能手工从 nsys 的 kern_sum 里扒, 而 kern_sum 把预填的核混在一起, 判不了。
# 读法: 表里 `<(unsigned int)6>` 这类模板实参就是验证批的 token 数, `<(unsigned int)1>` 是纯解码;
# 同一个核两份实例的 ms/step 一比, 就是"验证 k 位比验证 1 位贵几倍"——投机赢不赢全看这个数。
#
# ★第二个参数给 specno = 走投机路但不采 nsys★(2026-09-17, mtp-2.md M0): 只要引擎自己的探针
# (f16 范围看门狗 / moe-uniq / mtp-uniq / 一轮分账)时用它, 秒级出数, 不用等 nsys 采样与导出。
# ★第 10 个参数: 钉死验证几位(2026-09-17)★ 0 = 交给置信调度器。
# 为什么非要它: 调度器的成本常量是**编译期**量好的(core_draft_sched.c), 一旦核变快而常量还是旧的,
# 它会一律判"投机亏本"、一轮草稿都不出 —— 于是这把尺量不到任何验证批, 表里只有 n=1 那一档,
# 而"验证多一位多付多少"正是投机赢不赢的全部。实撞过一次: 新核落地后 40 个 token 里 0 轮草稿。
VERIFY_K="${10:-0}"
# ★第 12 个参数: 透传给引擎的额外参数★(2026-09-18): 例 "--no-graph" —— 解码整步 CUDA graph 的 A/B 必须是**同一个二进制**
# 两种发法(图 vs 直发), 换二进制比的是别的东西。多个参数用空格隔开。
EXTRA_ARGS=(${12:-})
SPEC=(--no-dspark)
IS_SPEC=0
if [ "$DO_NSYS" = spec ] || [ "$DO_NSYS" = specno ]; then
  SPEC=(--dspark); IS_SPEC=1
  [ "$VERIFY_K" != 0 ] && SPEC+=(--dspark-verify "$VERIFY_K")
  if [ "$DO_NSYS" = spec ]; then DO_NSYS=yes; else DO_NSYS=no; fi
  echo "== ★投机路(--dspark)★ 读表注意两件事:"
  echo "   ①一轮 = **两个 step**(草稿块一发 embed + 验证批一发 embed), 表里的 ms/step × 2 才是一轮;"
  echo "   ②'步的总毫秒'那一行是每次前向, 不是每 token —— 每 token 看引擎自己打的'一轮 … ⇒ ms/token'。"
fi

echo "== ① 引擎逐层毫秒(--v41-prof, 生成 $NGEN token)"
./"$BIN" -m "$MODEL" --zchain "$AMP" --v41-prof --temp 0 --seed 1 -n "$NGEN" \
      "${SPEC[@]}" "${EXTRA_ARGS[@]}" "${PARG[@]}" > "$OUT/prof.out" 2> "$OUT/prof.err"
grep -a "总 .* | 层(ms)" "$OUT/prof.err" | tail -3
# ★为什么判决取中位不取平均★(2026-09-15 撞的): spark 上 110 GB 模型 mmap 在 121 GB 机器里,
# kswapd/kcompactd 一直在回收页, 每轮总有几步被拖到 110 ms 以上。平均值被这些离群步拖着走,
# 同一份代码前后两跑能差 15% —— 拿它判核改得好不好, 判出来的全是页回收的脾气。中位数对离群免疫。
echo "-- 解码步的总毫秒分布(后 $NGEN 步):"
grep -ao "总 [0-9.]* ms" "$OUT/prof.err" | grep -o "[0-9.]*" | tail -n "$NGEN" | \
  awk '{s+=$1; if(NR==1||$1<mn)mn=$1; if($1>mx)mx=$1; a[NR]=$1}
       END{n=asort(a);
           md=(n%2)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2;
           printf "  步数 %d  ★中位 %.1f ms (= %.2f t/s)★  平均 %.1f  最小 %.1f  最大 %.1f\n",
                  n, md, 1000/md, s/n, mn, mx}'

[ "$IS_SPEC" = 1 ] && grep -a -h "DSpark:\|一轮 " "$OUT/prof.err" | sed 's/^/  /'
# ★引擎自带的三个探针(2026-09-17 mtp-2.md M0)★ 它们只在 --v41-prof 下出数, 都是"判下一刀该不该动"的依据:
#   [f16-range] 激活/中间量有没有超出 f16 的指数范围 ⇒ 定专家核张量核版走 f16 还是 TF32(mtp-2 §5.3)
#   [moe-uniq]  主干验证批的唯一专家数 ⇒ 验证多一位要多付多少专家字节(mtp-2 §2.2)
#   [mtp-uniq]  草稿塔的唯一专家数 ⇒ 塔专家并集那一刀值不值(mtp-2 §6.1)
grep -a -h "\[f16-range\]" "$OUT/prof.err" | tail -2 | sed 's/^/  /'
for tag in moe-uniq mtp-uniq; do
  grep -a -h "\[$tag\]" "$OUT/prof.err" | sed 's/.*n=\([0-9]*\).*唯一专家 \([0-9]*\) \/ \([0-9]*\).*/\1 \2 \3/' | \
    awk -v t="$tag" '{u[$1]+=$2; s[$1]+=$3; c[$1]++}
      END {for (n in c) printf "  [%s] n=%s: 唯一专家 %.2f / %d  (每 token %.2f 份, %d 层样本)\n",
                               t, n, u[n]/c[n], s[n]/c[n], (u[n]/c[n])/n, c[n]}' | sort
done
echo
echo "-- 生成的文本(温 0; 换核后拿它和上一版对, 变了就是数值动了)"
tail -c 400 "$OUT/prof.out"
echo

# ★ncu 模式(2026-09-16 single-1.md)★: 第二个参数给 ncu = 不采 nsys, 改用 Nsight Compute 问
# 两个大核**为什么**只跑到 110/165 GB/s。nsys 只告诉你"慢", ncu 才说得出"慢在哪条管道"。
#   为什么用 --replay-mode application: 默认的 kernel replay 要把核用到的设备内存备份一份,
#   本机模型 110+ GB, 直接失败(09-15 撞过)。application replay 是整个进程重跑一遍, 不备份。
#   为什么一次只问 2~3 个指标: 指标多了要多趟 replay, 每趟都是一次 110 GB 模型的加载。
#   ★GB10 上 dram__* 一律返回 n/a(统一内存, fable5 09-15 已撞)★ —— 别问它, 问 l1tex/LSU 与 stall 原因。
if [ "$DO_NSYS" = ncu ]; then
  NCU=$(command -v ncu || echo /usr/local/cuda/bin/ncu)
  [ -x "$NCU" ] || { echo "★没有 ncu★"; exit 1; }
  # ★核正则/采几发/跳几发走位置参数, 不走环境变量★(铁律: 脚本的 ${VAR:-默认} 也算 env 配置)
  # ★为什么必须能指定"跳几发"★(2026-09-16 实撞): 默认跳 200 发时, 3 字提示切成 6 token 的**预填**
  # 还没跑完(一次预填就发 600 多个核), 于是量到的全是预填, 而这把尺问的是解码。
  # 更要命的是正则同时匹配骨架核: 骨架核每步发 329 次, 6 发的名额一个不剩地被它占光, VQ 核一发没采到。
  # ⇒ 要量哪个核就只写哪个核, 跳几发按"预填发数 + 前几步解码"算(VQ gateup 每次前向 40 发)。
  KREGEX="${6:-regex:v41_vq_gateup_kernel}"
  NLAUNCH="${7:-6}"
  NSKIP="${8:-200}"
  # 第一趟: 访存管道的量(波前数 = LSU 真正付的访存次数; 扇区/请求 = 合并得好不好)
  M1=l1tex__data_pipe_lsu_wavefronts.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
  # 第二趟: warp 停在哪(long_scoreboard = 等全局访存返回; mio/lg_throttle = 访存指令发不出去 = 管道塞住)
  M2=smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
  # ★第三趟(2026-09-17 mtp-2.md §2.1)★: 这一发到底在忙什么 —— 发射槽用了几成、**指令总数**、各管道饱和度、
  # shared 查表的波前与 bank 冲突。为什么非要它: 前两趟只说"warp 停在等内存", 判不了"多一个 token 多付多少";
  # 指令数才是那个量。实测 n=5 验证批的指令数 = n=1 的 **5.00 倍(精确)**, 于是"按专家去重省字节"那条路
  # 从机理上就走不通(省的是读几遍, 省不掉每 token 都要乘一遍) —— 这一趟就是那个判决的来源。
  # 用法: 同一个核在 n=1(--no-dspark)与 n>1(第二参给 spec/specno)各跑一遍, 两张表一比就是边际成本的构成。
  M3=smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,l1tex__throughput.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,gpu__time_duration.sum
  # ★第四趟(2026-09-18)★: 其余停等原因。第二趟只问 long_scoreboard/mio/lg 三种, 位流整块读落地后 long_scoreboard
  # 仍 17~18, 可它已经解释不了时间 —— 得看 short_scoreboard(等 shared/shfl)、barrier(等 __syncthreads)、
  # not_selected(有活但没轮到)、wait(固定延迟)、math_pipe 各占多少, 才知道下一刀砍哪。
  # ★第 11 个参数: 跑哪几趟★(默认 "1 2 3"; 例 "2 4" 只问停等原因, 少装两次模型)。
  M4=smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,smsp__average_warps_issue_stalled_selected_per_issue_active.ratio
  # ★第五趟(2026-09-18)★: 命中率。位流整块读 + 跨块流水之后 long_scoreboard 仍 14~17, 热路上剩下的全局读只有每轮一条
  # 激活 LDG.128(应是 L1 命中)—— 若 L1 命中率低, 就是位流流过 L1 把激活挤掉了, 激活得搬进 shared。
  M5=l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum
  PASSES="${11:-1 2 3}"
  for pass in $PASSES; do
    eval "MM=\$M$pass"
    echo "== ② ncu 第 $pass 趟: $MM"
    # ★--clock-control none★: 别锁时钟。这台板子锁不了(统一内存 + 没有 root), 不传它 ncu 每趟都先
    # 试着锁一遍再警告, 白等; 而判决本来就是同机器状态两个二进制各两遍取中位, 不靠锁时钟。
    "$NCU" --replay-mode application --clock-control none --target-processes all -k "$KREGEX" \
      --launch-skip "$NSKIP" --launch-count "$NLAUNCH" --metrics "$MM" --csv \
      ./"$BIN" -m "$MODEL" --zchain "$AMP" --temp 0 --seed 1 -n 8 "${SPEC[@]}" "${EXTRA_ARGS[@]}" "${PARG[@]}" \
      > "$OUT/ncu$pass.csv" 2> "$OUT/ncu$pass.err"
    grep -a "^\"" "$OUT/ncu$pass.csv" | head -40 || tail -20 "$OUT/ncu$pass.err"
  done
  echo "== ③ SASS: 内层到底发了几条访存指令(注释里那句'一条非对齐 32 位读'是假设, 这里验它)"
  CUOBJ=$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)
  "$CUOBJ" -sass "./$BIN" > "$OUT/sass.txt" 2>/dev/null
  for k in v41_vq_row_dot v41_vq_gateup_kernel v41_fp4x32_gemv_kernel; do
    echo "-- $k 的访存指令直方图(LDG/LDS/LD 各几条)"
    awk -v k="$k" '/^\t\tFunction : / {inf = ($0 ~ k)} inf && /LDG|LDS|LD\./ {n=$0; sub(/^ *\/\*[0-9a-f]*\*\/ */,"",n); split(n,a," "); print a[1]}' \
      "$OUT/sass.txt" | sort | uniq -c | sort -rn | head -8
  done
  exit 0
fi

[ "$DO_NSYS" = yes ] || { echo "== ② nsys: 按参数跳过"; exit 0; }
# ★等 ① 段的进程真退干净再开 ②★(2026-09-17 实撞): 引擎有意留着单实例锁, 而 110 GB 的映射卸载要几秒 ——
# ① 段的 shell 已经返回, 内核还在收页。不等就是 ② 段直接
# "another ds4 process is already running; refusing to start", 而 nsys 照样采样、照样导出,
# 只是 sqlite 里**一个 CUDA 核都没有** ⇒ 后面三张表全空, 日志里却只有一行 SKIPPED, 很容易当成"表没出来"。
for _ in $(seq 1 60); do pgrep -f "[d]s4 -m gguf" >/dev/null || break; sleep 2; done
echo "== ② nsys 采样"
command -v nsys >/dev/null || { echo "★没有 nsys, ② 跳过★"; exit 0; }
rm -f "$OUT/dec.nsys-rep" "$OUT/dec.sqlite"
# ★--cuda-graph-trace=node★(2026-09-18 实撞): 默认按整张 graph 记一行, 图里的核一个都看不见 —— 解码整步 graph 落地后
# ③ 那张逐步表把 32 个解码步全算成一步、逐核合计里只剩预填的核。按节点记, 图里的核与直发的同名同列。
nsys profile -o "$OUT/dec" --force-overwrite true -t cuda --cuda-graph-trace=node \
  ./"$BIN" -m "$MODEL" --zchain "$AMP" --temp 0 --seed 1 -n "$NGEN" "${SPEC[@]}" "${EXTRA_ARGS[@]}" "${PARG[@]}" \
  > "$OUT/nsys.out" 2> "$OUT/nsys.err"

# ★删 sqlite 要在**第一条 nsys stats 之前**★(2026-09-16 实撞): 原来只在 ③ 前面删, 于是 ②
# 的 kern_sum/api_sum 复用了上一次的 sqlite("Existing SQLite export found"), 打出来的是上一版的账 ——
# 本次基线的 API 表里 cudaLaunchKernel 47443 发就是上一轮的残留。任何 nsys stats 之前先删干净。
rm -f "$OUT/dec.sqlite" "$OUT/trace_cuda_gpu_trace.csv"
echo "-- 逐核耗时(前 15)"
nsys stats --report cuda_gpu_kern_sum "$OUT/dec.nsys-rep" 2>/dev/null | head -25
echo "-- CUDA API(前 12): 这里出现 Synchronize/Memcpy 大头 = 主机往返"
nsys stats --report cuda_api_sum "$OUT/dec.nsys-rep" 2>/dev/null | head -20

# ③ 逐核时间线分账(2026-09-15 single.md §1 的尺): 按 v41_embed_kernel 切步, 每步求 墙钟/核忙/间隙/发射数,
#    再把骨架 GEMV / 专家 gateup·down / 注意力 / engram 各自的毫秒列出来, 末尾给专家核逐层 µs(长尾一眼看见)。
#    怎么读: 核忙 ≈ 墙钟 ⇒ 瓶颈在核内; 间隙大 ⇒ 发射/主机; 专家逐层里某几层 10 倍 ⇒ 那几层的 blob 没进设备缓存。
echo
echo "== ③ 逐核时间线分账"
# ★先删旧的 sqlite 与 csv★: nsys stats 见到已存在的导出就直接复用("Existing SQLite export found"),
# 于是这一版的核占比表里全是上一版的核 —— 2026-09-15 实撞, 差点按旧账去改核。
rm -f "$OUT/dec.sqlite" "$OUT/trace_cuda_gpu_trace.csv"
nsys stats --report cuda_gpu_trace --format csv -o "$OUT/trace" "$OUT/dec.nsys-rep" >/dev/null 2>&1
awk -F, -v ngen="$NGEN" '
# ★2026-09-16 single-1.md 扩表★: 原表只分 GEMV/专家/注意力/engram 四项, 于是"随上下文涨的那 157 ms"
# 全落进看不见的余项里。加 indexer 打分 / topk / mHC / D2D 拷贝四列, 再加一张**只统计解码步**的
# 逐核名次表 —— 长提示时 nsys 自带的 kern_sum 把预填的核混在一起, 那张表判不了解码。
NR>1 { st=$1+0; du=$2+0; line=$0;
  if (line ~ /v41_embed_kernel/) { step++; sstart[step]=st; lastend=st; }
  if (step>0) {
    nk[step]++; busy[step]+=du;
    if (st>lastend) gaps[step]+=(st-lastend);
    if (st+du>lastend) lastend=st+du;
    if (line ~ /vq_gateup/) { gi[step]++; gd[step,gi[step]]=du; }
    if (line ~ /vq_down/)   { di[step]++; dd[step,di[step]]=du; }
    if (line ~ /fp4x32_gemv_kernel<\(unsigned int\)1>/) { gemvsum[step]+=du; gemvn[step]++; }
    if (line ~ /sparse_attn/) att[step]+=du;
    if (line ~ /fp8blk_gemv/) eng[step]+=du;
    if (line ~ /indexer_score/) { idx[step]+=du; idxn[step]++; }
    if (line ~ /v41_topk_kernel|candidate_kernel/) tk[step]+=du;
    if (line ~ /hc_fused|hc_post|hc_pre|hc_split/) hc[step]+=du;
    if (line ~ /Memcpy|memcpy/) mc[step]+=du;
    # 解码步的逐核名次: 核名在第 21 列(Name)。★不能用 $NF★: 模板实参里有逗号
    # (`gemv_kernel<(unsigned int)1, (unsigned int)1>`), -F, 会把一个核名切成好几段。
    # 取 $21 再砍掉 "<" / "(" 之后的部分 ⇒ 同一个核的各个模板实例合并成一行(解码只有 NT=1 那份)。
    nm=$21; sub(/^ *void /,"",nm); sub(/[<(].*/,"",nm); gsub(/^[ "]+|[ "]+$/,"",nm);
    if (nm == "") nm = "[memcpy/memset]";
    kname[step,nk[step]]=nm; kdur[step,nk[step]]=du;
  }
}
END {
  print "step wall_ms busy_ms gap_ms launches gemv_ms(n) gateup_ms down_ms attn_ms engram_ms idx_ms(n) topk_ms hc_ms memcpy_ms";
  first = (step > ngen) ? step - ngen + 1 : 2;
  for (s=first; s<=step; s++) {
    wall=(s<step)?(sstart[s+1]-sstart[s])/1e6:0;
    gs=0; for(g=1;g<=gi[s];g++) gs+=gd[s,g];
    ds=0; for(d=1;d<=di[s];d++) ds+=dd[s,d];
    printf "%3d %7.1f %7.1f %6.1f %6d %6.1f(%d) %7.1f %6.1f %6.1f %6.1f %6.1f(%d) %6.1f %6.1f %6.1f\n",
      s, wall, busy[s]/1e6, gaps[s]/1e6, nk[s], gemvsum[s]/1e6, gemvn[s], gs/1e6, ds/1e6,
      att[s]/1e6, eng[s]/1e6, idx[s]/1e6, idxn[s], tk[s]/1e6, hc[s]/1e6, mc[s]/1e6;
  }
  # 解码步逐核合计(除去首尾两步: 首步冷缓存, 末步墙钟不完整)
  lo = first+1; hi = step-1; nsteps = hi-lo+1;
  if (nsteps > 0) {
    for (s=lo; s<=hi; s++) for (k=1; k<=nk[s]; k++) { tot[kname[s,k]] += kdur[s,k]; cnt[kname[s,k]]++; }
    printf "\n-- 解码步逐核合计(第 %d~%d 步, 每步平均; 长提示时这张表才是解码的真账)\n", lo, hi;
    printf "%-46s %9s %9s\n", "kernel", "ms/step", "launch/step";
    n = asorti(tot, order, "@val_num_desc");
    for (i=1; i<=n && i<=18; i++) { kk=order[i]; printf "%-46s %9.2f %9.1f\n", kk, tot[kk]/nsteps/1e6, cnt[kk]/nsteps; }
  }
  s=step-1; printf "\ngateup 第 %d 步逐层(us):", s; for(g=1;g<=gi[s];g++) printf " %d", gd[s,g]/1000; printf "\n";
  printf "down   第 %d 步逐层(us):", s; for(d=1;d<=di[s];d++) printf " %d", dd[s,d]/1000; printf "\n";
}' "$OUT/trace_cuda_gpu_trace.csv"

# ④ 按"步的类型"分类的逐核表(2026-09-17 mtp-2.md §1.2)。
# 为什么非要它: 投机一轮里有三种完全不同的步(草稿块 / 验证批 / 预填), 而 ③ 那张逐步表和 nsys 自带的
# kern_sum 都把它们混在一起 —— 于是**判不了"验证多一位多付多少"**, 而那个数就是投机赢不赢的全部。
# 分类办法: 每步按"出现了哪些核"认 —— 有 mtp_ 核的是草稿步; 有 nvfp4/cutlass 的是预填块;
# 其余按骨架 GEMV 的**模板实参**(= 这一批几个 token)分成 verify_n1(纯解码) / verify_n2 / …
# 读法: 同一个核在 verify_n1 与 verify_n3 两列的差 ÷ 2 = 每多一位的边际; 草稿步那一列是草稿器的全部账。
echo
echo "== ④ 按步类型分类的逐核表(草稿 / 验证 n= / 预填)"
awk -F, '
NR>1 { st=$1+0; du=$2+0; nm=$21; gsub(/"/,"",nm); sub(/^ *void /,"",nm);
  if (nm ~ /v41_embed_kernel/) { step++; sstart[step]=st; }
  if (step>0) { key=nm; base=key; sub(/[(<].*/,"",base);
     # ★NT 要从**整行**匹配, 不能从 $21★(2026-09-17 实撞): 核名里有逗号, -F, 把它切成好几列,
     # 列数还随核名长短变 ⇒ $21 有时不含 "<", 于是 ntstep 保留上一步的值, 把同一种 5 行验证步
     # 标成了 n1/n2/n3/n4/n5 五档(看着像"边际随批长", 其实是同一个量的五次采样)。
     if ($0 ~ /fp4x32_gemv_kernel<\(unsigned int\)[0-9]+/) { match($0,/fp4x32_gemv_kernel<\(unsigned int\)[0-9]+/);
        ntstep[step]=substr($0,RSTART+34,RLENGTH-34) }
     if (base ~ /mtp_/) mtpstep[step]=1;
     if (base ~ /nvfp4|cutlass|Kernel/) pf[step]=1;
     t[step,base]+=du; n[step,base]++; tot[step]+=du;
     if (!(base in seen)) { seen[base]=1; names[++nn]=base }
  }
}
END { for (s=1;s<=step;s++) { cls = mtpstep[s] ? "draft" : (pf[s] ? "prefill" : "verify_n" ntstep[s]); cnt[cls]++;
        for (i=1;i<=nn;i++) { b=names[i]; ct[cls,b]+=t[s,b]; cn[cls,b]+=n[s,b] } ctot[cls]+=tot[s];
        if (s<step) cwall[cls]+=(sstart[s+1]-sstart[s]) }
  m=asorti(cnt, ord);
  for (k=1;k<=m;k++) { c=ord[k];
     printf "\n-- %s: %d 步, 核忙 %.1f ms/步, 壁钟 %.1f ms/步\n", c, cnt[c], ctot[c]/cnt[c]/1e6, cwall[c]/cnt[c]/1e6;
     for (i=1;i<=nn;i++) { b=names[i]; v=ct[c,b]/cnt[c]/1e6;
        if (v>0.05) printf "   %-44s %8.2f ms %7.1f 发\n", b, v, cn[c,b]/cnt[c] } } }' \
  "$OUT/trace_cuda_gpu_trace.csv"
