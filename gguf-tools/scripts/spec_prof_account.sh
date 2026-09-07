#!/bin/bash
# spec_prof_account.sh — 投机剖面的逐核账(从 spec_prof_spark.sh 拆出, 可对已有 trace 重算)。
# 用法: spec_prof_account.sh <nsys cuda_gpu_trace.csv> <投机轮数> [top=40] [起点核=dspark_attn_kernel]
# 起点核: 生成段第一个只在生成里出现的核; 纯解码剖面(mode=plain)没有 dspark_*, 传 q4k_hc_expand_kernel(解码 out_b 路, prefill 走批核)。
# 两列: 增量 ms/轮 = 本核结束 − 上一核结束(流序), 是 PDL 下的真实成本; 时长 ms/轮 = nsys Duration 之和(含等前序的时间, 虚高)。
set -uo pipefail
CSV="${1:?trace csv}"; R="${2:?轮数}"; TOP="${3:-40}"; MARK="${4:-dspark_attn_kernel}"
AWK=$(command -v gawk || command -v awk)
# 生成段起点 = 第一次 drafter 前向的核(hc_mean_slot 只在 --spec 武装的 prefill/decode 出现; 用 dspark_attn 更准: 只在 draft 里)
"$AWK" -F',' -v R="$R" -v TOP="$TOP" -v MARK="$MARK" '
NR==1 { for (i=1;i<=NF;i++) { if ($i=="Start (ns)") cs=i; if ($i=="Duration (ns)") cd=i; if ($i=="Name") cn=i }
        if (!cs||!cd) { cs=1; cd=2 } next }
{
    st=$cs+0; du=$cd+0;
    nm=$cn; for (i=cn+1;i<=NF;i++) nm=nm","$i; gsub(/^"|"$/,"",nm);
    sub(/\(.*$/,"",nm); sub(/^void /,"",nm); sub(/^.*::/,"",nm);
    n++; S[n]=st; D[n]=du; N[n]=nm;
    if (nm==MARK && !t0) t0=st;
    if (st+du>tend) tend=st+du;
}
END {
    if (!t0) { print "★没找到生成段起点核 " MARK "★"; exit 4 }
    win=(tend-t0)/1e6;
    # key 里放原始行号(09-07 修: 此前放压缩后的序号再去索引原始数组 S/D/N, 增量列整列错位到别的核名上; 时长列没受影响)
    m=0; for (i=1;i<=n;i++) if (S[i]>=t0) { m++; key[m]=sprintf("%020d %d", S[i], i) }
    cnt=asort(key); busy=0; ce=-1;
    for (k=1;k<=cnt;k++) { split(key[k],p," "); i=p[2]+0; a=S[i]; b=S[i]+D[i];
        if (a>ce) { busy+=b-a; ce=b } else if (b>ce) { busy+=b-ce; ce=b } }
    busy/=1e6;
    printf "生成段: %.1f ms 墙钟 = %.1f ms/轮, GPU 忙 %.1f ms/轮, 空隙 %.1f ms/轮, 每轮 %d 个 kernel\n", win, win/R, busy/R, (win-busy)/R, int(m/R);
    # PDL 下核提前上 SM 在等待里等前序完成, nsys 的 Duration 把这段等待也算进本核 ⇒ 大核后面的核被虚高
    # (09-07 定罪: rows_q8_0 表 383 µs, 微基准同核 164; 差的 219 = 前序 grouped 215 的等待)。
    # 真实增量成本 = 本核结束时刻 − 上一核结束时刻(按起点排序的流序), 两列并列打印。
    for (i=1;i<=n;i++) if (S[i]>=t0) { T[N[i]]+=D[i]; C[N[i]]++ }
    pe=t0;
    for (k=1;k<=cnt;k++) { split(key[k],p," "); i=p[2]+0; e=S[i]+D[i]; inc=e-pe; if (inc<0) inc=0; X[N[i]]+=inc; if (e>pe) pe=e }
    printf "%-56s %9s %9s %8s %9s\n", "kernel", "增量ms/轮", "时长ms/轮", "发/轮", "均值us";
    for (k in T) out[++o]=sprintf("%012.3f|%s", X[k]/1e6/R, k);
    asort(out);
    for (j=o; j>=1 && o-j<TOP; j--) { split(out[j],p,"|"); k=p[2]; printf "%-56s %9.3f %9.3f %8.1f %9.1f\n", substr(k,1,56), X[k]/1e6/R, T[k]/1e6/R, C[k]/R, T[k]/C[k]/1e3 }
    # drafter 段单列(09-07): 一段 = hc_mean_slot_kernel(drafter 起点, main_x 取 HC 均值) 起, 到下一次 hc_mean_slot 之前的最后一个
    # dspark_* 核止; 段内按流序增量累计。verify 段 = 其余。两段各打小计, 看 draft 15 ms 里肉在哪。
    ind=0; dsum=0; pe=t0; delete DX; delete DC; delete DT;
    for (k=1;k<=cnt;k++) { split(key[k],p," "); i=p[2]+0; e=S[i]+D[i]; inc=e-pe; if (inc<0) inc=0;
        if (N[i]=="hc_mean_slot_kernel") ind=1;
        if (ind) { DX[N[i]]+=inc; DC[N[i]]++; DT[N[i]]+=D[i]; dsum+=inc; if (N[i] ~ /^dspark_(markov_bias_argmax|argmax)_kernel$/) last_arg=k }
        if (ind && N[i] !~ /^dspark_/ && N[i] !~ /^hc_mean_slot/ && last_arg && k>last_arg+2) { ind=0; last_arg=0 }
        if (e>pe) pe=e }
    printf "\n== drafter 段(hc_mean_slot → 末个 dspark argmax): 增量合计 %.1f ms/轮 ==\n", dsum/1e6/R;
    o2=0; for (k in DX) out2[++o2]=sprintf("%012.3f|%s", DX[k]/1e6/R, k);
    asort(out2);
    for (j=o2; j>=1 && o2-j<24; j--) { split(out2[j],p,"|"); k=p[2]; printf "%-56s %9.3f %9.3f %8.1f %9.1f\n", substr(k,1,56), DX[k]/1e6/R, DT[k]/1e6/R, DC[k]/R, DT[k]/DC[k]/1e3 }
}' "$CSV"
