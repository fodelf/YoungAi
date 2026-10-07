# d0a 第 ⑦ 张表 · 一步的逐核时间线(2026-10-07)。输入与 d0a_excl_timeline.awk 同一份 nsys cuda_gpu_trace csv。
# 为什么要它: ⑥ 只说"某类核独占几毫秒、重叠几毫秒", 动刀要知道重叠发生在**哪两发之间**(谁的尾巴拖着谁的头)。
# 用法:  gawk -v step=0  -f d0a_step_dump.awk trace.csv   → 逐步摘要(步号/壁钟/核数), 先用它找要看的那一步
#        gawk -v step=12 -f d0a_step_dump.awk trace.csv   → 第 12 步逐核: 起点 µs | 时长 | 独占 | 与前一发重叠 | 流 | 核名
#        加 -v agg=1 只打逐核名汇总(发数/均时长/均独占/均"被前一发压住"/均"压住后一发")。
# 步的切分照抄 ⑥: embed 核开新步。"重叠"按时间线真算(同一时刻 ≥2 发在跑), 不是简单看相邻两发。
function nm_of(s) { gsub(/"/, "", s); sub(/^ *void /, "", s); sub(/[(<].*/, "", s); return s == "" ? "[memcpy/memset]" : s; }
function dump_step(   n, i, j, o, t0, k, cnt, pend, prev_end, excl, ov, lastend, e, q, m, z, seg, tot, se, ee, cur, l, ends) {
  if (ne == 0) return;
  cur_step++;
  n = ne;
  # 按起点排序
  n = asorti(st, o, "@val_num_asc");
  t0 = st[o[1]]; lastend = t0;
  if (step == 0) {
    e = 0; for (i = 1; i <= n; i++) { j = o[i]; if (st[j] + du[j] > e) e = st[j] + du[j]; }
    printf("step %3d  壁钟 %8.3f ms  核 %4d  gu %2d  seg %2d  首核 %s\n", cur_step, (e - t0) / 1e6, n, ngu, nseg, nm[o[1]]);
    delete st; delete du; delete nm; delete sm; ne = 0; ngu = 0; nseg = 0; return;
  }
  if (cur_step != step) { delete st; delete du; delete nm; delete sm; ne = 0; ngu = 0; nseg = 0; return; }
  # 每一发的独占 = 它的 [s,e) 里没有别的发在跑的部分; 扫描线: 事件排序后逐段归属
  m = 0;
  for (i = 1; i <= n; i++) { j = o[i]; m++; et[m] = st[j]; ed[m] = 1; ei[m] = j; m++; et[m] = st[j] + du[j]; ed[m] = -1; ei[m] = j; }
  z = asorti(et, q, "@val_num_asc");
  tot = 0; prev_end = et[q[1]];
  for (i = 1; i <= z; i++) {
    k = q[i]; t = et[k];
    if (t > prev_end && tot == 1) { for (l in act) if (act[l]) { ex[l] += t - prev_end; break; } }
    act[ei[k]] += ed[k]; tot += ed[k]; if (!act[ei[k]]) delete act[ei[k]];
    prev_end = t;
  }
  # 与前一发(按起点序)的重叠 = max(0, 前一发终点 − 本发起点)
  if (!agg) printf("step %d(%s)  起点µs   时长µs  独占µs  被前发压µs 流 核名\n", cur_step, cls);
  for (i = 1; i <= n; i++) {
    j = o[i]; ov = (lastend > st[j]) ? (lastend - st[j]) / 1e3 : 0;
    if (!agg) printf("  %9.1f %8.1f %7.1f %8.1f  %2s %s\n", (st[j] - t0) / 1e3, du[j] / 1e3, ex[j] / 1e3, ov, sm[j], nm[j]);
    cnt[nm[j]]++; sdur[nm[j]] += du[j]; sex[nm[j]] += ex[j]; sov[nm[j]] += ov * 1e3;
    if (i > 1) sov_next[nm[o[i-1]]] += ov * 1e3;
    if (st[j] + du[j] > lastend) lastend = st[j] + du[j];
  }
  printf("\n逐核名汇总(step %d, 壁钟 %.3f ms):  发数  均时长µs  均独占µs  均被前发压µs  均压后发µs  时长和ms  独占和ms  核名\n", cur_step, (lastend - t0) / 1e6);
  for (k in cnt) printf("  %4d %9.1f %9.1f %11.1f %11.1f %9.3f %9.3f  %s\n", cnt[k], sdur[k] / cnt[k] / 1e3, sex[k] / cnt[k] / 1e3,
                        sov[k] / cnt[k] / 1e3, sov_next[k] / cnt[k] / 1e3, sdur[k] / 1e6, sex[k] / 1e6, k) | "sort -k7 -n -r";
  close("sort -k7 -n -r");
  delete st; delete du; delete nm; delete sm; ne = 0; ngu = 0; nseg = 0;
}
BEGIN { FS = ","; ne = 0; cur_step = 0; if (step == "") step = 0; }
NR > 1 {
  s = $1 + 0; d = $2 + 0; name = nm_of($21);
  if (name ~ /v41_(q4k_)?embed_kernel/) dump_step();
  ne++; st[ne] = s; du[ne] = d; nm[ne] = name; sm[ne] = $20;
  if (name ~ /vq_gu_persist/) ngu++;
  if (name ~ /attn_mma_seg/) nseg++;
  if (match($0, /(fp4x32|q4k)_gemv(1_stage|1_pipe)?_kernel<\(unsigned int\)[0-9]+/)) { v = substr($0, RSTART, RLENGTH); sub(/.*\)/, "", v); cls = "n" v; }
}
END { dump_step(); }
