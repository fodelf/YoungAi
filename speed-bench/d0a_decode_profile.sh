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
#   第二个参数给 no = 只跑 ①(引擎逐层毫秒), 秒级出数 —— 每改一版解码核就对这一把, 不用等 nsys。

set -u
cd "$(dirname "$0")/.." || exit 1

NGEN="${1:-32}"
DO_NSYS="${2:-yes}"
MODEL="${3:-gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-fp4.gguf}"
AMP=gguf/v41/gr-fin-40-fp4
OUT=/tmp/d0a-decode
PROMPT="你好世界"     # 3 token: 预填几乎不占时间, 采到的基本全是解码

mkdir -p "$OUT"
[ -f "$MODEL" ] || { echo "★没有模型 $MODEL★"; exit 1; }
[ -x ./ds4 ]    || { echo "★没有 ./ds4, 先 make cuda-spark★"; exit 1; }

echo "== ① 引擎逐层毫秒(--v41-prof, 生成 $NGEN token)"
./ds4 -m "$MODEL" --zchain "$AMP" --v41-prof --temp 0 --seed 1 -n "$NGEN" \
      -p "$PROMPT" > "$OUT/prof.out" 2> "$OUT/prof.err"
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

[ "$DO_NSYS" = yes ] || { echo "== ② nsys: 按参数跳过"; exit 0; }
echo "== ② nsys 采样"
command -v nsys >/dev/null || { echo "★没有 nsys, ② 跳过★"; exit 0; }
rm -f "$OUT/dec.nsys-rep" "$OUT/dec.sqlite"
nsys profile -o "$OUT/dec" --force-overwrite true -t cuda \
  ./ds4 -m "$MODEL" --zchain "$AMP" --temp 0 --seed 1 -n "$NGEN" -p "$PROMPT" \
  > "$OUT/nsys.out" 2> "$OUT/nsys.err"

echo "-- 逐核耗时(前 15)"
nsys stats --report cuda_gpu_kern_sum "$OUT/dec.nsys-rep" 2>/dev/null | head -25
echo "-- CUDA API(前 12): 这里出现 Synchronize/Memcpy 大头 = 主机往返"
nsys stats --report cuda_api_sum "$OUT/dec.nsys-rep" 2>/dev/null | head -20

# ③ 逐核时间线分账(2026-09-15 single.md §1 的尺): 按 v41_embed_kernel 切步, 每步求 墙钟/核忙/间隙/发射数,
#    再把骨架 GEMV / 专家 gateup·down / 注意力 / engram 各自的毫秒列出来, 末尾给专家核逐层 µs(长尾一眼看见)。
#    怎么读: 核忙 ≈ 墙钟 ⇒ 瓶颈在核内; 间隙大 ⇒ 发射/主机; 专家逐层里某几层 10 倍 ⇒ 那几层的 blob 没进设备缓存。
echo
echo "== ③ 逐核时间线分账"
nsys stats --report cuda_gpu_trace --format csv -o "$OUT/trace" "$OUT/dec.nsys-rep" >/dev/null 2>&1
awk -F, '
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
  }
}
END {
  print "step wall_ms busy_ms gap_ms launches gemv_ms(n) gateup_ms down_ms attn_ms engram_ms";
  for (s=2; s<=step; s++) {
    wall=(s<step)?(sstart[s+1]-sstart[s])/1e6:0;
    gs=0; for(g=1;g<=gi[s];g++) gs+=gd[s,g];
    ds=0; for(d=1;d<=di[s];d++) ds+=dd[s,d];
    printf "%3d %7.1f %7.1f %6.1f %6d %6.1f(%d) %7.1f %6.1f %6.1f %6.1f\n", s, wall, busy[s]/1e6, gaps[s]/1e6, nk[s], gemvsum[s]/1e6, gemvn[s], gs/1e6, ds/1e6, att[s]/1e6, eng[s]/1e6;
  }
  s=int(step/2); printf "gateup 第 %d 步逐层(us):", s; for(g=1;g<=gi[s];g++) printf " %d", gd[s,g]/1000; printf "\n";
  printf "down   第 %d 步逐层(us):", s; for(d=1;d<=di[s];d++) printf " %d", dd[s,d]/1000; printf "\n";
}' "$OUT/trace_cuda_gpu_trace.csv"
