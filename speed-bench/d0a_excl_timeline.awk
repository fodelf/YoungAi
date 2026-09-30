# d0a 第 ⑥ 张表 · 独占时间分账(2026-09-29 晚)。被 d0a_decode_profile.sh 引用, 不单独当脚本跑。
# 为什么要它: ③④ 两张表把每个核的时长直接相加, 而侧流并行 + PDL 提前发射之下核时长是**重叠**的
# (n=4 验证步核忙 85 ms 对壁钟 64 ms) —— 拿相加的账去估"某类核值几毫秒"会高估, 09-29 上午按它估 GEMV NT>1 值 3~4 ms,
# PDL 链式微基准一量, 4 行与 1 行只差 0.3 ms。这张表按 GPU 时间线扫一遍: 某一刻只有一类核在跑, 这段时间记给那一类(独占);
# 两类以上同时在跑记成"重叠"; 没核在跑记成空转。独占时间才是"把这类核砍成 0 能省下的上界"。
# 输入: nsys cuda_gpu_trace 的 csv(Start=$1, Duration=$2, Name=$21 的第一段; 核名里的逗号会把后面切乱, 第 21 列之前的列是稳的)。
# 步的切分与类型判法照抄 ④(embed 核开新步; 类型按 GEMV 模板实参 n / 草稿 / 预填)。
function cls_of(nm) {
  if (nm ~ /q4k_gemv1/) return "gemv";
  if (nm ~ /vq_gu_persist|vq_dn_persist|vq_gateup_kernel|vq_down_kernel/) return "expert";
  if (nm ~ /attn_mma_seg|sparse_attn|indexer_score|topk_kernel|candidate_kernel/) return "attn";
  if (nm ~ /fp8blk_gemv/) return "engram";
  if (nm ~ /hc_fused|hc_post|hc_mix|hc_pre|hc_split/) return "mhc";
  return "small";
}
function flush_step(   n, i, j, k, o, t, prev, dt, act, tot, one, c, key) {
  if (ne == 0) return;
  cls = mtp ? "draft" : (pf ? "prefill" : "verify_n" ntv);
  n = asorti(evt, o, "@val_num_asc");
  prev = evt[o[1]]; tot = 0;
  for (k in act) delete act[k];
  for (i = 1; i <= n; i++) {
    j = o[i]; t = evt[j]; dt = t - prev;
    if (dt > 0) {
      if (tot == 0) { idle[cls] += dt; }
      else {
        one = ""; c = 0;
        for (k in act) if (act[k] > 0) { c++; one = k; }
        if (c == 1) excl[cls, one] += dt; else mixed[cls] += dt;
        busy[cls] += dt;
        if (tot >= 2) ovl2[cls] += dt;
      }
    }
    act[evc[j]] += evd[j]; tot += evd[j]; prev = t;
  }
  nstep[cls]++; wall[cls] += (evt[o[n]] - evt[o[1]]);
  for (k in dsum) { dur[cls, k] += dsum[k]; delete dsum[k]; }
  delete evt; delete evc; delete evd; ne = 0; mtp = 0; pf = 0; ntv = "";
}
BEGIN { FS = ","; ne = 0; }
NR > 1 {
  st = $1 + 0; du = $2 + 0; nm = $21; gsub(/"/, "", nm); sub(/^ *void /, "", nm); sub(/[(<].*/, "", nm);
  if (nm == "") nm = "[memcpy/memset]";
  if (nm ~ /v41_(q4k_)?embed_kernel/) { flush_step(); step++; }
  if (step == 0) next;
  if (match($0, /(fp4x32|q4k)_gemv(1_stage|1_pipe)?_kernel<\(unsigned int\)[0-9]+/)) { v = substr($0, RSTART, RLENGTH); sub(/.*\)/, "", v); ntv = v; }
  if (nm ~ /mtp_/) mtp = 1;
  if (nm ~ /nvfp4|cutlass|Kernel|vqm_kernel|vqm_gather16|vqp_fused|q4k_to_bf16/) pf = 1;
  c = cls_of(nm);
  ne++; evt[ne] = st; evd[ne] = 1; evc[ne] = c;
  ne++; evt[ne] = st + du; evd[ne] = -1; evc[ne] = c;
  dsum[c] += du;
}
END {
  flush_step();
  split("gemv expert attn engram mhc small", CL, " ");
  m = asorti(nstep, ord);
  for (q = 1; q <= m; q++) {
    cls = ord[q]; ns = nstep[cls]; if (ns == 0) continue;
    printf "\n-- %s: %d 步 | 壁钟 %.1f ms/步 = 核忙 %.1f + 空转 %.1f;  核忙里 ≥2 核重叠 %.1f\n",
           cls, ns, wall[cls] / ns / 1e6, busy[cls] / ns / 1e6, idle[cls] / ns / 1e6, ovl2[cls] / ns / 1e6;
    printf "   %-8s %10s %10s   说明\n", "类", "独占ms", "时长和ms";
    for (i = 1; i <= 6; i++) { k = CL[i];
      printf "   %-8s %10.2f %10.2f   %s\n", k, excl[cls, k] / ns / 1e6, dur[cls, k] / ns / 1e6,
             (excl[cls, k] > 0 && dur[cls, k] > 1.3 * excl[cls, k]) ? "时长和被重叠虚高" : ""; }
    printf "   %-8s %10.2f            两类以上同时在跑, 归不到任何一类\n", "重叠", mixed[cls] / ns / 1e6;
  }
}
