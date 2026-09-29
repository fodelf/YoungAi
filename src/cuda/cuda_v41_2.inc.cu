/* cuda_v41_2.inc.cu — ds4_cuda.cu 分片: DeepSeek V4.1 批前向原语 ②(2026-09-12 战役 P2)。
 * RoPE / 激活量化(fp8·fp4 就地) / 压缩器池化 / indexer 打分·候选块·topk / 稀疏注意力。
 * 每个核都对着官方 model.py / kernel.py 的那一段写, 注释里标的是对应的官方函数。 */

/* indexer 三件(打分/候选块/topk)2026-09-18 挪到 cuda_v41_indexer.inc.cu(本片顶到 500 行)。 */

/* ---- RoPE(官方 precompute_freqs_cis + apply_rotary_emb): 相邻两元素 = 一个复数 ---- */
__device__ __forceinline__ static float v41_rope_freq(uint32_t i, uint32_t dim, float theta, uint32_t osl,
                                                      float factor, float beta_fast, float beta_slow) {
    float f = 1.0f / powf(theta, (float)(2u * i) / (float)dim);
    if (osl > 0u) {   /* YaRN ramp: corrected_dim(rot) = dim·ln(osl/(rot·2π)) / (2 ln θ) */
        const float lt = 2.0f * logf(theta);
        float low = floorf((float)dim * logf((float)osl / (beta_fast * 2.0f * (float)M_PI)) / lt);
        float high = ceilf((float)dim * logf((float)osl / (beta_slow * 2.0f * (float)M_PI)) / lt);
        low = fmaxf(low, 0.0f); high = fminf(high, (float)(dim - 1u));
        float ramp = ((float)i - low) / fmaxf(high - low, 1e-3f);
        ramp = fminf(fmaxf(ramp, 0.0f), 1.0f);
        const float smooth = 1.0f - ramp;
        f = f / factor * (1.0f - smooth) + f * smooth;
    }
    return f;
}
__global__ static void v41_rope_kernel(float *x, const int32_t *pos, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
                                       float theta, uint32_t osl, float factor, float bf, float bs, int inverse) {
    v41_pdl_wait();   /* PDL: 第一句就等上游(见 cuda_internal.cuh); 不经 PDL 发射时立即返回 */
    const uint32_t t = blockIdx.y, h = blockIdx.x;
    const uint32_t i = threadIdx.x;                 /* 复数下标 i < n_rot/2 */
    if (i >= n_rot / 2u) return;
    float *xr = x + ((uint64_t)t * n_head + h) * head_dim + (head_dim - n_rot) + 2u * i;
    const float ang = (float)pos[t] * v41_rope_freq(i, n_rot, theta, osl, factor, bf, bs);
    float c = cosf(ang), s = sinf(ang);
    if (inverse) s = -s;
    const float a = xr[0], b = xr[1];
    xr[0] = v41_bf16r(a * c - b * s);   /* 官方: x.float() 旋转后 copy_ 回 bf16 张量 */
    xr[1] = v41_bf16r(a * s + b * c);
}
int ds4_gpu_v41_rope_tensor(ds4_gpu_tensor *x, const ds4_gpu_tensor *pos, uint32_t n_tok, uint32_t n_head,
                            uint32_t head_dim, uint32_t n_rot, float theta, uint32_t original_seq_len,
                            float factor, float beta_fast, float beta_slow, bool inverse) {
    if (!x || !pos || (n_rot & 1u) || n_rot > 128u) return 0;
    v41_rope_kernel<<<dim3(n_head, n_tok), 64, 0, g_cur_stream>>>((float *)x->ptr, (const int32_t *)pos->ptr, n_head, head_dim, n_rot,
                                                                    theta, original_seq_len, factor, beta_fast, beta_slow, inverse ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "v41 rope");
}

/* 激活量化就地(act_quant)那一族已挪到 cuda_kv_pack.inc.cu —— 它与 KV 打包是同一件事
 * (同一套 scale 规则、同一张 FP4/FP8 表), 放在一起才不会两边各改各的。 */

/* ---- 压缩器池化(官方 Compressor.forward ratio>1, start_pos=0): 组内逐维 softmax 加权和 → bf16 ---- */
__global__ static void v41_compress_pool_kernel(float *out, const float *kv, const float *sc, uint32_t ratio, uint32_t dim) {
    const uint32_t g = blockIdx.x;
    for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
        float mx = -INFINITY;
        for (uint32_t t = 0; t < ratio; t++) mx = fmaxf(mx, sc[((uint64_t)g * ratio + t) * dim + d]);
        float den = 0.f, acc = 0.f;
        for (uint32_t t = 0; t < ratio; t++) {
            const float e = expf(sc[((uint64_t)g * ratio + t) * dim + d] - mx);
            den += e; acc += e * kv[((uint64_t)g * ratio + t) * dim + d];
        }
        out[(uint64_t)g * dim + d] = v41_bf16r(acc / den);
    }
}
int ds4_gpu_v41_compress_pool_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *score,
                                     uint32_t n_tok, uint32_t ratio, uint32_t dim) {
    if (!out || !kv || !score || ratio == 0) return 0;
    const uint32_t ng = n_tok / ratio;
    if (ng == 0) return 1;
    v41_compress_pool_kernel<<<ng, 256, 0, g_cur_stream>>>((float *)out->ptr, (const float *)kv->ptr, (const float *)score->ptr, ratio, dim);
    return cuda_ok(cudaGetLastError(), "v41 compress pool");
}

/* ★压缩源层的解码一步(graph 路, 2026-09-18 单行; 2026-09-22 扩成 n 行, 投机验证批进图)★
 * 为什么是一发而不是直发路那三件: 直发路的"两次主机偏移 memcpy + 主机判断再发池化核"三件的形状随位置变,
 * 一次捕获的图装不下; 这里 pos 从设备槽读, 每步同一发, 拓扑与相位无关。
 * ★与 v41_compress_pool_kernel 逐位同★: 池化那几行照抄(同 t 序、同 mx/den/acc 三步、同 bf16 舍点)。
 * 余行缓冲的格号 = pos % ratio 与直发路一致: 直发路预填后把 rem 行挪到头, rem = pos % ratio, 正是下一步该写的格。
 * 出错会怎样: 格号错一位 = 池化时组内 token 顺序错, softmax 逐维加权和变了 —— 不报错, 温 0 逐字节门会抓。
 * 本批 n 行按批内顺序逐行追加到第 (pos0%ratio + i) % ratio 格, 每凑满一组池化一次 →
 * pooled[j] / posg[j](j = 本批第几个完成的组, 最多 (ratio-1+n)/ratio 个; 没完成的行由打包核写进垃圾槽)。
 * 与直发路 v41_compress_source 的"线性追加 → 整批池化 → 余行挪到头"逐位等价: 每组吃的是同样几行、同样顺序, 批后余行同样落在
 * 0..rem−1 格; 池化那几行与单行版逐字同。同时把 [旧余行 | 本批 n 行] 线性存进 snap_*(与直发路 snap_cpre 同布局 ⇒ v41_spec_rollback
 * 一个字不改, 按 snap_cpend + keep − pend2 取行); snap_kv==NULL(纯解码 n=1 的图)不存。
 * 一 block 逐行走(n ≤ 8, 每行一次 __syncthreads: 上一组池化读完 0..ratio−1 格, 后面的行才许盖第 0 格)。 */
__global__ static void v41_compress_step_n_kernel(float *pooled, int32_t *posg, float *ckv_c, float *csc_c, float *snap_kv, float *snap_sc,
                                                  const float *ckv, const float *csc, const int32_t *posd, uint32_t ratio, uint32_t dim, uint32_t n) {
    v41_pdl_wait();   /* PDL: 第一句就等上游(见 cuda_internal.cuh); 不经 PDL 发射时立即返回 */
    const uint32_t pos0 = (uint32_t)posd[0], pend = pos0 % ratio;
    if (snap_kv)
        for (uint32_t r = 0; r < pend; r++)
            for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) { snap_kv[(uint64_t)r * dim + d] = ckv_c[(uint64_t)r * dim + d]; snap_sc[(uint64_t)r * dim + d] = csc_c[(uint64_t)r * dim + d]; }
    uint32_t j = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t slot = (pend + i) % ratio;
        __syncthreads();
        for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
            const float a = ckv[(uint64_t)i * dim + d], b = csc[(uint64_t)i * dim + d];
            ckv_c[(uint64_t)slot * dim + d] = a; csc_c[(uint64_t)slot * dim + d] = b;
            if (snap_kv) { snap_kv[(uint64_t)(pend + i) * dim + d] = a; snap_sc[(uint64_t)(pend + i) * dim + d] = b; }
        }
        if (slot + 1u != ratio) continue;          /* 没凑满: 只追加(slot 是 block 一致的, 分支不发散) */
        __syncthreads();
        if (threadIdx.x == 0) posg[j] = (int32_t)(pos0 + i + 1u - ratio);
        for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
            float mx = -INFINITY;
            for (uint32_t t = 0; t < ratio; t++) mx = fmaxf(mx, csc_c[(uint64_t)t * dim + d]);
            float den = 0.f, acc = 0.f;
            for (uint32_t t = 0; t < ratio; t++) {
                const float e = expf(csc_c[(uint64_t)t * dim + d] - mx);
                den += e; acc += e * ckv_c[(uint64_t)t * dim + d];
            }
            pooled[(uint64_t)j * dim + d] = v41_bf16r(acc / den);
        }
        j++;
    }
}
int ds4_gpu_v41_compress_step_n_tensor(ds4_gpu_tensor *pooled, ds4_gpu_tensor *posg, ds4_gpu_tensor *cpre_kv, ds4_gpu_tensor *cpre_sc,
                                       ds4_gpu_tensor *snap_kv, ds4_gpu_tensor *snap_sc, const ds4_gpu_tensor *ckv, const ds4_gpu_tensor *csc,
                                       const ds4_gpu_tensor *posd, uint32_t ratio, uint32_t dim, uint32_t n) {
    if (!pooled || !posg || !cpre_kv || !cpre_sc || !ckv || !csc || !posd || ratio < 2u || n == 0u || n > 8u) return 0;
    const uint32_t ngmax = (ratio - 1u + n) / ratio;
    if (cpre_kv->bytes < (uint64_t)ratio * dim * 4 || cpre_sc->bytes < (uint64_t)ratio * dim * 4) return 0;
    if (pooled->bytes < (uint64_t)ngmax * dim * 4 || posg->bytes < (uint64_t)ngmax * 4 || ckv->bytes < (uint64_t)n * dim * 4) return 0;
    if ((snap_kv != NULL) != (snap_sc != NULL)) return 0;
    if (snap_kv && (snap_kv->bytes < (uint64_t)(ratio - 1u + n) * dim * 4 || snap_sc->bytes < (uint64_t)(ratio - 1u + n) * dim * 4)) return 0;
    v41_compress_step_n_kernel<<<1, 256, 0, g_cur_stream>>>((float *)pooled->ptr, (int32_t *)posg->ptr, (float *)cpre_kv->ptr, (float *)cpre_sc->ptr,
        snap_kv ? (float *)snap_kv->ptr : NULL, snap_sc ? (float *)snap_sc->ptr : NULL,
        (const float *)ckv->ptr, (const float *)csc->ptr, (const int32_t *)posd->ptr, ratio, dim, n);
    return cuda_ok(cudaGetLastError(), "v41 compress step(n 行)");
}

/* ---- 稀疏注意力(官方 sparse_attn_kernel 语义): 一 block 一 (query, 8 头)(128 线程 = 4 warp, 每 warp 2 头), grid (n, 8),
 * 键序 = 窗口行(升序) 后接 topk 压缩行; 在线 softmax 逐键; p 先舍 bf16 再乘 v(官方 acc_s_cast);
 * 分母用未舍的 exp 和; sink 只进分母。q/k 值已是 bf16 格点, 点积 f32。
 * 第一版一 block 包 64 头(1024 线程): 解码 n=1 时整层只有 1 个 block 在一个 SM 上逐键 syncthreads, 0.5 ms/层 = 20 ms/token;
 * 拆成 8 个头组 block 后键行多读 8 次(L2 命中), 换 8 个 SM 并行。 ---- */
#ifndef V41_ATTN_HEADS_PER_BLOCK   /* split-K 分片(cuda_v41_attn_split.inc.cu)排在前面, 可能已经定义过 */
#define V41_ATTN_HEADS_PER_BLOCK 8u
#endif
#ifndef V41_ATTN_KTILE
#define V41_ATTN_KTILE 8u          /* 一次进 shared 的键行数: 8×512×4 B = 16 KB ⇒ 每 SM 能驻 6 个 block */
#endif
/* ★2026-09-15 按键分块重写(speed.md 第二轮; nsys 定罪: 这个核占预填 32.3%, 116 ms/层)★
 * 旧版逐键一轮: 每键两次 __syncthreads + 每键把整个累加器重缩放一遍(16 FMA/头/键)。
 * 现在一次搬 8 个键进 shared: 同步次数降到 1/8, 重缩放摊到 1/8(每块只按块内最大值缩一次) ——
 * 这正是 flash 式在线 softmax 的标准做法, 官方 sparse_attn_kernel 也是按 64 键一块做的
 * (我们用不了 64: 官方那份要 q_shared+kv_shared 各 64 KB = 128 KB 动态 shared, GB10 opt-in 上限只有 99 KB)。
 * ★数值★: 分块改变了在线 softmax 的分组, 与旧版不是逐位同(与官方的分组也不同, 官方 64 我们 8);
 * 判据因此是 NLL/五指标, 不是逐位 —— 这一条在 speed.md §6 D 段写明。
 * 无效 topk 槽(idx<0)不跳过而是记 -inf 分数, 让它在块内自然得到 p=0(跳过会打乱分块结构)。 */
__global__ static void v41_sparse_attn_kernel(float *o, const float *q, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                                              const float *sink, uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk,
                                              uint32_t n_head, uint32_t hd, float scale, uint32_t full_block, uint32_t ring,
                                              uint32_t win_lo) {
    const uint32_t i = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t per = hd / 32u;                 /* 512/32 = 16 维/lane */
    __shared__ float ks[V41_ATTN_KTILE][512];
    __shared__ int   kok[V41_ATTN_KTILE];          /* 这一槽是不是有效键 */
    __shared__ float ksc[V41_ATTN_KTILE][32];      /* 压缩行的 32 个缩放, 每行解一次(见 v41_ckv_get_s) */
    float qa[2][16], acc[2][16], mx[2], sum[2];
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * V41_ATTN_HEADS_PER_BLOCK + warp * 2u + hh;
        for (uint32_t e = 0; e < per; e++) { qa[hh][e] = q[((uint64_t)i * n_head + h) * hd + lane * per + e]; acc[hh][e] = 0.f; }
        mx[hh] = -1e30f; sum[hh] = 0.f;
    }
    const uint32_t p = pos0 + i;                   /* 绝对位置 */
    /* ★full_block(DSpark 草稿块)★: 本 chunk 的每一位看同一个键集合 = 整个窗口 + 块内全部 n 位
     * (官方 get_dspark_topk_idxs, 与 i 无关, 也不做因果截断 —— 5 个草稿位是一次出的)。
     * 主路(full_block=0)是因果的: 第 i 位只看到 [p+1-window, p]。 */
    const uint32_t last = full_block ? pos0 + full_block - 1u : p;
    uint32_t lo = full_block ? (pos0 > window ? pos0 - window : 0u)            /* 窗口 = 块前面那 window 个位置 */
                             : (p + 1u > window ? p + 1u - window : 0u);       /* 因果: 含自己那一位 */
    if (lo < win_lo) lo = win_lo;                  /* 环里 win_lo 之前的槽没写过(CED), 不读(官方 -1 屏蔽同义) */
    const uint32_t nwin = last - lo + 1u;
    const uint32_t nkeys = nwin + topk;
    for (uint32_t base = 0; base < nkeys; base += V41_ATTN_KTILE) {
        const uint32_t nt = (nkeys - base) < V41_ATTN_KTILE ? (nkeys - base) : V41_ATTN_KTILE;
        __syncthreads();                            /* 上一块的 ks 已被读完才能覆盖 */
        for (uint32_t t = threadIdx.x / 32u; t < nt; t += blockDim.x / 32u) {
            const uint32_t kk = base + t;
            /* 两种键来源两种存法: 窗口行还是 f32(它只有 2.6 MB, 打包不值), 压缩行是打包的 FP4 */
            const float *krow = NULL; const uint8_t *cpk = NULL;
            if (kk < nwin) krow = kvw + v41_win_row((int64_t)lo + kk, pos0, window, ring) * hd;
            else { const int32_t g = idx[(uint64_t)i * topk + (kk - nwin)];
                   if (g >= 0 && (uint32_t)g < ng) cpk = kvc + (uint64_t)g * DS4_V41_CKV_BYTES; }
            if (lane == 0) kok[t] = (krow || cpk) ? 1 : 0;
            if (cpk) ksc[t][lane] = ds4_e4m3fn_to_f32(cpk[DS4_V41_CKV_NIB + lane]);
            __syncwarp();
            /* ★无效槽必须写 0, 不能留着上一块的残留(2026-09-15 定罪)★: 下面算 acc_add 时对无效槽
             * 乘的是 p=0, 但 0 × 残留值只有在残留是有限值时才等于 0 —— shared 里上一个 block 留下的
             * 字节按 float 解释出来可能是 NaN/Inf, 于是 0×NaN = NaN。实撞后果: 位置 0 的 512 个 topk 槽
             * 几乎全无效 ⇒ token 0 的整份 hc(20480 个)在 L02 变非有限值, 而残留内容取决于哪个 block
             * 先用过这块 shared ⇒ 同一份输入温度 0 跑两遍结果不同。 */
            for (uint32_t d = lane; d < hd; d += 32u)
                ks[t][d] = krow ? krow[d] : (cpk ? v41_ckv_get_s(cpk, d, ksc[t]) : 0.f);   /* warp 内合并读 */
        }
        __syncthreads();
        for (int hh = 0; hh < 2; hh++) {
            float s[V41_ATTN_KTILE];
            float tm = -1e30f;
            for (uint32_t t = 0; t < nt; t++) {
                float d = 0.f;
                for (uint32_t e = 0; e < per; e++) d += qa[hh][e] * ks[t][lane * per + e];
                for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
                s[t] = kok[t] ? d * scale : -1e30f;
                tm = fmaxf(tm, s[t]);
            }
            const float nm = fmaxf(mx[hh], tm);
            const float rs = expf(mx[hh] - nm);     /* 整块只缩一次 */
            float acc_add[16];
            for (uint32_t e = 0; e < per; e++) acc_add[e] = 0.f;
            float ps = 0.f;
            for (uint32_t t = 0; t < nt; t++) {
                const float pv = expf(s[t] - nm);
                ps += pv;
                const float pb = v41_bf16r(pv);
                for (uint32_t e = 0; e < per; e++) acc_add[e] += pb * ks[t][lane * per + e];
            }
            sum[hh] = sum[hh] * rs + ps;
            for (uint32_t e = 0; e < per; e++) acc[hh][e] = acc[hh][e] * rs + acc_add[e];
            mx[hh] = nm;
        }
    }
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * V41_ATTN_HEADS_PER_BLOCK + warp * 2u + hh;
        const float den = sum[hh] + expf(sink[h] - mx[hh]);
        for (uint32_t e = 0; e < per; e++) o[((uint64_t)i * n_head + h) * hd + lane * per + e] = v41_bf16r(acc[hh][e] / den);
    }
}
int ds4_gpu_v41_sparse_attn_tensor(ds4_gpu_tensor *o, const ds4_gpu_tensor *q, const ds4_gpu_tensor *kv_win,
                                   const ds4_gpu_tensor *kv_comp, const ds4_gpu_tensor *idx,
                                   const void *model_map, uint64_t model_size, uint64_t sink_offset,
                                   uint32_t n_tok, uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk,
                                   uint32_t ratio,
                                   uint32_t n_head, uint32_t head_dim, float scale, int full_block, int ring,
                                   uint32_t win_lo, const ds4_gpu_tensor *posd, uint32_t pos_cap) {
    if (!o || !q || !kv_win || n_head != 64u || head_dim != 512u) { fprintf(stderr, "ds4: [v41] sparse attn 只实现 64 头×512\n"); return 0; }
    /* 钳位只有预填核实现(见 ds4_gpu_v41.h): 解码核(n≤8, 含 graph 路)进来时它必须是空操作, 否则停车而不是静默读脏槽。
     * 生成路把最后一块留够 window 个位置(core_v41_api.c), 所以解码时 p+1-window ≥ win_lo 恒成立。 */
    const uint32_t lo_first = pos0 + 1u > window ? pos0 + 1u - window : 0u;   /* 本批第一个 query 的自然下界(后面的只会更大) */
    const int clamp_matters = win_lo > lo_first;
    if (clamp_matters && n_tok <= 8u && !full_block) {
        fprintf(stderr, "ds4: ★[v41] sparse attn: 解码核不实现窗口钳位, 但 pos0 %u 的窗口下界 %u < win_lo %u —— 最后一块没留够 window 个位置?★\n",
                pos0, lo_first, win_lo);
        return 0;
    }
    /* ★graph 路(posd 非 NULL)只许走解码张量核版★: 标量 split 版在主机上按真键数定段长, 进不了一次捕获的图;
     * 张量核版抬不上 shared 时这里直接失败(不静默换核 —— 换了核就与直发路不是同一累加序, 逐字节门必分叉)。 */
    if (posd) {
        if (full_block || !ring || n_tok > 8u) { fprintf(stderr, "ds4: [v41] graph 路的注意力只收主路 n≤8 的形态\n"); return 0; }
        const float *sink = (const float *)cuda_model_range_ptr(model_map, sink_offset, (uint64_t)n_head * 4, "v41 sink");
        if (!sink) return 0;
        const int hasc = kv_comp && idx;   /* 纯窗口层(ratio 0)也走这里: 窗口范围同样随位置变 */
        if (!v41_attn_mma_decode((float *)o->ptr, (const float *)q->ptr, (const float *)kv_win->ptr,
                                 hasc ? (const uint8_t *)kv_comp->ptr : NULL, hasc ? (const int32_t *)idx->ptr : NULL,
                                 sink, n_tok, pos0, window, hasc ? ng : 0u, hasc ? topk : 0u, hasc ? ratio : 0u, n_head, head_dim, scale,
                                 (const int32_t *)posd->ptr, pos_cap, 0u, 1u)) {
            fprintf(stderr, "ds4: [v41] graph 路要的解码张量核注意力不可用\n"); return 0;
        }
        return 1;
    }
    if (kv_win->bytes < (uint64_t)(window + n_tok) * head_dim * 4) { fprintf(stderr, "ds4: [v41] 窗口缓冲不够 %u+%u 行\n", window, n_tok); return 0; }
    const float *sink = (const float *)cuda_model_range_ptr(model_map, sink_offset, (uint64_t)n_head * 4, "v41 sink");
    if (!sink) return 0;
    /* ★段 4★ 预填先走张量核版(cuda_sparse_attn_mma.inc.cu); 它自己判形状/块大小/shared, 不适用返回 0 回这里。
     * 解码(n_tok 小)恒走标量版: 那时 grid 只有 n_tok×4 个 block, 填不满 48 个 SM。 */
    /* ★解码(n_tok=1)先走 split-K★(single.md S4): 原核那时 grid 只有 8 个 block, 48 个 SM 里 40 个干等,
     * 键只能串着啃(实测 5.3 µs/键, 且随上下文线性涨)。split 把键切段铺满 SM; 键太少或形状不合它返回 0。
     * 草稿块(full_block)不走: 它的可见性是"块内全可见", 与 split 里按 pos0 算的因果窗口不是一回事。 */
    /* ★split-K 与 mma 两条快路只服务主路(历史段恒是环)★: 它们的调用条件是 !full_block, 而
     * full_block 只有 DSpark 草稿塔会给 —— 所以走到这两条路时 ring 必然是 1, 核里直接按环算。
     * 真要出现"非草稿的 full_block"或"非环的主路", 下面这个断言会先拦住, 不会静默读错行。 */
    if (!full_block && !ring) { fprintf(stderr, "ds4: [v41] sparse attn: 主路窗口必须是环(decode.md D1)\n"); return 0; }
    /* ★解码先试张量核版★(decode.md D2): 对一个 query, S = Q[64×512]·Kᵀ[512×nkeys] 是个真 GEMM,
     * 标量版付 5 条 shfl 才换 16 个 FMA(实测 69 GFLOP/s = 峰值的 0.4%)。键少/形状不合它自己返回 0。 */
    if (!full_block &&
        v41_attn_mma_decode((float *)o->ptr, (const float *)q->ptr, (const float *)kv_win->ptr,
                            kv_comp ? (const uint8_t *)kv_comp->ptr : NULL, idx ? (const int32_t *)idx->ptr : NULL,
                            sink, n_tok, pos0, window, ng, (kv_comp && idx) ? topk : 0u,
                            (kv_comp && idx) ? ratio : 0u, n_head, head_dim, scale, NULL, 0u, 0u, 1u))
        return 1;
    /* ★DSpark 草稿块也走张量核版(2026-09-29)★: 塔的注意力只有窗口(无压缩键)、块内全可见、窗口是线性段(ring=0)。
     * 原来落到下面的标量核: 一轮 3 塔 1.6 ms(每发 0.5 ms 算 5 位 × 133 键), 逐核表里草稿步的第三大项。草稿只提议、验证定输出,
     * 换核不改最终文本, 门 = 接受率直方图不退(d1 samp/dflt 的 DSpark 汇总行) + 陪审团。键太少/形状不合它自己返回 0 回标量核。 */
    if (full_block && !ring && !kv_comp && n_tok <= 8u &&
        v41_attn_mma_decode((float *)o->ptr, (const float *)q->ptr, (const float *)kv_win->ptr, NULL, NULL,
                            sink, n_tok, pos0, window, 0u, 0u, 0u, n_head, head_dim, scale, NULL, 0u, n_tok, 0u))
        return 1;
    if (!full_block &&
        v41_sparse_attn_split((float *)o->ptr, (const float *)q->ptr, (const float *)kv_win->ptr,
                              kv_comp ? (const uint8_t *)kv_comp->ptr : NULL, idx ? (const int32_t *)idx->ptr : NULL,
                              sink, n_tok, pos0, window, ng, (kv_comp && idx) ? topk : 0u, n_head, head_dim, scale))
        return 1;
    /* 草稿块(full_block)不走 mma 版: 那份是按因果窗口写的, 块内全可见的语义它没有(而且 n=5 也填不满张量核) */
    if (!full_block && ds4_sparse_attn_mma_launch((float *)o->ptr, (const float *)q->ptr, (const float *)kv_win->ptr,
                                   kv_comp ? (const uint8_t *)kv_comp->ptr : NULL, idx ? (const int32_t *)idx->ptr : NULL,
                                   sink, n_tok, pos0, window, ng, (kv_comp && idx) ? topk : 0u, n_head, head_dim, scale, win_lo))
        return 1;
    v41_sparse_attn_kernel<<<dim3(n_tok, n_head / V41_ATTN_HEADS_PER_BLOCK), V41_ATTN_HEADS_PER_BLOCK * 16u, 0, g_cur_stream>>>(
        (float *)o->ptr, (const float *)q->ptr, (const float *)kv_win->ptr,
        kv_comp ? (const uint8_t *)kv_comp->ptr : NULL, idx ? (const int32_t *)idx->ptr : NULL, sink, pos0, window, ng,
        (kv_comp && idx) ? topk : 0u, n_head, head_dim, scale, full_block ? n_tok : 0u, ring ? 1u : 0u, win_lo);
    return cuda_ok(cudaGetLastError(), "v41 sparse attn");
}
