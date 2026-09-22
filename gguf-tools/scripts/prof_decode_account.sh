#!/bin/bash
# prof_decode_account.sh — 从 nsys 剖面里只把【解码段】的 kernel 账拆出来(spark 本机跑)。
#
# 为什么要有它: prof_bench_spark.sh 的 cuda_gpu_kern_sum 把 prefill(2048 token 的 GEMM/
# dequant)和 decode(64 token)混在一张表里, 解码核的均值还被 prefill 同名核污染。真正
# 要看的是"每个解码 token 的 GPU 时间花在哪个核、发了几个核、核与核之间空了多久"。
# 做法: 导出 cuda_gpu_trace(逐次 kernel 事件), 以第 SKIP+1 次 vq_moe_gateup_fused2(L0)的
# 起点为解码窗口起点(前 SKIP 个 token 是 4 相图捕获, 不算稳态), 窗口到最后一个 kernel 结束。
# 用法: prof_decode_account.sh <prof.nsys-rep> [跳过 token=4] [解码 token 总数=64] [top=40]
# 输出: 每 token 周期/忙碌/空隙, 按每 token 时长排序的 kernel 表(含每 token 发射次数, GB/s 留给读者)。
set -uo pipefail
REP="${1:?nsys-rep}"; SKIP="${2:-4}"; NTOK="${3:-64}"; TOP="${4:-40}"
[ -s "$REP" ] || { echo "★缺 $REP★"; exit 2; }
BASE="${REP%.nsys-rep}"
CSV="${BASE}_cuda_gpu_trace.csv"
[ -s "$CSV" ] || nsys stats --report cuda_gpu_trace --format csv --force-export=true -o "$BASE" "$REP" >/dev/null 2>&1
[ -s "$CSV" ] || { echo "★gpu_trace 导出失败★"; exit 3; }
# 列(nsys 2024+): Start(ns),Duration(ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem,DymSMem,Bytes,Throughput,SrcMemKd,DstMemKd,Device,Ctx,GreenCtx,Strm,Name
# 名字含逗号(模板参数), 用 "最后一个引号段" 取名; 数值列在前面固定位置。
AWK=$(command -v gawk || command -v awk)   # asort 是 gawk 扩展(Ubuntu 默认 mawk 没有)
"$AWK" -F',' -v SKIP="$SKIP" -v NTOK="$NTOK" -v TOP="$TOP" '
NR==1 { for (i=1;i<=NF;i++) { if ($i=="Start (ns)") cs=i; if ($i=="Duration (ns)") cd=i; if ($i=="Name") cn=i; if ($i=="GrdX") cg=i }
        if (!cs||!cd) { cs=1; cd=2 } next }
{
    st=$cs+0; du=$cd+0;
    # 名字: 从第 cn 列到行尾拼回去(模板逗号被拆开了)
    nm=$cn; for (i=cn+1;i<=NF;i++) nm=nm","$i; gsub(/^"|"$/,"",nm);
    # 短名: 去掉参数表
    sub(/\(.*$/,"",nm); sub(/^void /,"",nm); sub(/^.*::/,"",nm);
    n++; S[n]=st; D[n]=du; N[n]=nm;
    if (nm=="vq_moe_gateup_fused2_kernel") { g++; if (g==SKIP*43+1) t0=st }   # 第 SKIP+1 个 token 的 L0
    if (st+du>tend) tend=st+du;
}
END {
    if (!t0) { print "★没找到解码起点(vq_moe_gateup_fused2)★"; exit 4 }
    ntk=NTOK-SKIP; win=(tend-t0)/1e6;
    # 忙碌并集(跨流合并区间)
    m=0; for (i=1;i<=n;i++) if (S[i]>=t0) { m++; A[m]=S[i]; B[m]=S[i]+D[i]; }
    # 简单插入排序太慢; 用 asort 索引
    for (i=1;i<=m;i++) key[i]=sprintf("%020d %d", A[i], i);
    cnt=asort(key);
    busy=0; ce=-1;
    for (k=1;k<=cnt;k++) { split(key[k],p," "); i=p[2]+0; a=A[i]; b=B[i];
        if (a>ce) { busy+=b-a; ce=b } else if (b>ce) { busy+=b-ce; ce=b } }
    busy/=1e6;
    printf "解码窗口: %d token, 周期 %.2f ms/token, GPU 忙 %.2f ms/token, 空隙 %.2f ms/token\n", ntk, win/ntk, busy/ntk, (win-busy)/ntk;
    for (i=1;i<=n;i++) if (S[i]>=t0) { T[N[i]]+=D[i]; C[N[i]]++; ksum+=D[i] }
    printf "kernel 时长总和 %.2f ms/token(含并发重叠), 每 token 发射 %d 个 kernel\n", ksum/1e6/ntk, int((m+ntk-1)/ntk);
    printf "%-52s %9s %8s %9s\n", "kernel", "ms/token", "发/token", "均值us";
    for (k in T) out[++o]=sprintf("%012.3f|%s", T[k]/1e6/ntk, k);
    asort(out);
    for (j=o; j>=1 && o-j<TOP; j--) { split(out[j],p,"|"); k=p[2]; printf "%-52s %9.3f %8.1f %9.1f\n", substr(k,1,52), T[k]/1e6/ntk, C[k]/ntk, T[k]/C[k]/1e3 }
}' "$CSV"
