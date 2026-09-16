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
MODEL="${3:-gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-fp4.gguf}"
PSRC="${4:-}"
PCHARS="${5:-40000}"
AMP=gguf/v41/gr-fin-40-fp4
OUT=/tmp/d0a-decode
PROMPT="你好世界"     # 3 token: 预填几乎不占时间, 采到的基本全是解码

mkdir -p "$OUT"
[ -f "$MODEL" ] || { echo "★没有模型 $MODEL★"; exit 1; }
[ -x ./ds4 ]    || { echo "★没有 ./ds4, 先 make cuda-spark★"; exit 1; }

# 提示: 默认短提示走 -p; 给了文件就截 PCHARS 字符走 --prompt-file(与 prefill_ttft_ruler.sh 同一口径)
PARG=(-p "$PROMPT")
if [ -n "$PSRC" ]; then
  [ -f "$PSRC" ] || { echo "★没有提示文件 $PSRC★"; exit 1; }
  head -c "$PCHARS" "$PSRC" > "$OUT/prompt.txt"
  PARG=(--ctx 32768 --prompt-file "$OUT/prompt.txt")
  echo "== 长提示模式: $PSRC 前 $PCHARS 字符"
fi
# ★--no-dspark 必须显式传★(2026-09-16): 投机解码的默认值是**开**(cli_diag.c: set_dspark(!no_dspark)),
# 带三塔的 GGUF 一喂进来就走投机路, 而这把尺量的是"纯单 token 解码每步多少毫秒"。不传的话换个文件
# 就悄悄量成另一条路, 数字还长得挺像 —— 历史上 single.md §5 就把这个默认值写反了。
SPEC=(--no-dspark)

echo "== ① 引擎逐层毫秒(--v41-prof, 生成 $NGEN token)"
./ds4 -m "$MODEL" --zchain "$AMP" --v41-prof --temp 0 --seed 1 -n "$NGEN" \
      "${SPEC[@]}" "${PARG[@]}" > "$OUT/prof.out" 2> "$OUT/prof.err"
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
  for pass in 1 2; do
    eval "MM=\$M$pass"
    echo "== ② ncu 第 $pass 趟: $MM"
    "$NCU" --replay-mode application --target-processes all -k "$KREGEX" \
      --launch-skip "$NSKIP" --launch-count "$NLAUNCH" --metrics "$MM" --csv \
      ./ds4 -m "$MODEL" --zchain "$AMP" --temp 0 --seed 1 -n 8 "${SPEC[@]}" "${PARG[@]}" \
      > "$OUT/ncu$pass.csv" 2> "$OUT/ncu$pass.err"
    grep -a "^\"" "$OUT/ncu$pass.csv" | head -40 || tail -20 "$OUT/ncu$pass.err"
  done
  echo "== ③ SASS: 内层到底发了几条访存指令(注释里那句'一条非对齐 32 位读'是假设, 这里验它)"
  CUOBJ=$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)
  "$CUOBJ" -sass ./ds4 > "$OUT/sass.txt" 2>/dev/null
  for k in v41_vq_row_dot v41_vq_gateup_kernel v41_fp4x32_gemv_kernel; do
    echo "-- $k 的访存指令直方图(LDG/LDS/LD 各几条)"
    awk -v k="$k" '/^\t\tFunction : / {inf = ($0 ~ k)} inf && /LDG|LDS|LD\./ {n=$0; sub(/^ *\/\*[0-9a-f]*\*\/ */,"",n); split(n,a," "); print a[1]}' \
      "$OUT/sass.txt" | sort | uniq -c | sort -rn | head -8
  done
  exit 0
fi

[ "$DO_NSYS" = yes ] || { echo "== ② nsys: 按参数跳过"; exit 0; }
echo "== ② nsys 采样"
command -v nsys >/dev/null || { echo "★没有 nsys, ② 跳过★"; exit 0; }
rm -f "$OUT/dec.nsys-rep" "$OUT/dec.sqlite"
nsys profile -o "$OUT/dec" --force-overwrite true -t cuda \
  ./ds4 -m "$MODEL" --zchain "$AMP" --temp 0 --seed 1 -n "$NGEN" "${SPEC[@]}" "${PARG[@]}" \
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
