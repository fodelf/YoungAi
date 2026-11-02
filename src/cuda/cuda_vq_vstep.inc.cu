/* cuda_vq_vstep.inc.cu — V 步(离散码字重选, 2026-11-02)的设备侧三件: ① 层码本近邻表 ② 按组一阶 Δ 选码 ③ 索引补丁(改 GPU 读的那份 blob 的索引位, 盘上不动)。
 *
 * 为什么: 专家 1.5 bit 的尾巴(判决份 KLD>2 的 177 位、鹈鹕"喉"位 0.012)对 ②/码本 δ/低秩 ③ 全不动 —— 它们都只动连续参数; PV-Tuning 在 ≤2 bit 上的收益
 *   来自重选每组的码字。本文件把"选哪个码字"做成设备上的一遍扫描 + 一遍补丁, 不落稠密权重、不改任何专家核(核照旧从 blob 读索引)。
 * ① knn: 层码本 nc 个 E4M3 8 维码字, 每个码字按欧氏距离取 K 个最近邻(不含自己) —— 候选只在近邻里, 是一阶线性化的信赖域(远跳一阶不可信)。
 * ② vselect: 每组(行 r, 列组 j)读当前码字 cur 与它的按组梯度 g[8](cuda_bwd_vgrad 累加的, 已含行增益), 在近邻里选 Δ = Σ g·(C_k − C_cur) 最小的; 写 Δ / 新码 / 旧码。
 *   主机按 Δ 排序只取最负的一小部分(vq_frac), 下一步重算梯度再选 —— 每步改一点是因为一阶预测只在小步内成立。
 * ③ code_patch: 记录 = {专家|矩阵<<16, 行, 列组, 新码|旧码<<16}; 12 位主流 + 13 位层的位平面按位改, 同一字节两组共用 ⇒ 32 位字 CAS 改位, 相邻组并发安全。
 *   改的是设备副本(训练/整模型常驻)或锁页暂存(流式层: 记表, v41_stage_fill 每次拷完再盖, 与 10-29 码本牵引同一套路); 整映射直读文件页的路改不了 ⇒ 返回 0 停车。
 *   改完按核的读法(vqb_code)逐条回读核对, 不同的条数交给调用方, 不是 0 就停 —— 不许一边说"补丁已挂"一边跑的是别的权重。
 * 出错会怎样: 12 位层给了 ≥4096 的码 ⇒ bad 置位返回 0; 载荷头不认(非 v3 / 形状不对) ⇒ bad; 近邻表 K > 32 拒。 */
#define VS_KMAX 32u

typedef struct { uint32_t ew, row, j, code; } vq_prec;   /* 与主机 pt_vstep 的记录同布局: ew = e | which<<16; code 低 16 新码, 高 16 旧码 */

__device__ __forceinline__ static void vs_decode8(const uint8_t *cw, float *o) {   /* 一个码字 8 个 E4M3 → f32(硬件转换, 与专家核同一条路) */
    const uint2 two = *(const uint2 *)cw;
    const __half2 h0 = v41_e4m3x2_to_half2(two.x & 0xffffu), h1 = v41_e4m3x2_to_half2(two.x >> 16);
    const __half2 h2 = v41_e4m3x2_to_half2(two.y & 0xffffu), h3 = v41_e4m3x2_to_half2(two.y >> 16);
    o[0] = __low2float(h0); o[1] = __high2float(h0); o[2] = __low2float(h1); o[3] = __high2float(h1);
    o[4] = __low2float(h2); o[5] = __high2float(h2); o[6] = __low2float(h3); o[7] = __high2float(h3);
}

/* ① 近邻表: 一线程一个码字, 全表扫一遍, 插入排序留 K 个最近(不含自己) */
__global__ static void vs_knn_kernel(const uint8_t *cb, uint32_t nc, uint32_t K, uint16_t *nbr) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nc) return;
    float ci[8]; vs_decode8(cb + (size_t)i * 8u, ci);
    float bd[VS_KMAX]; uint16_t bi[VS_KMAX];
    for (uint32_t k = 0; k < K; k++) { bd[k] = 3.4e38f; bi[k] = (uint16_t)i; }
    for (uint32_t m = 0; m < nc; m++) {
        if (m == i) continue;
        float cm[8]; vs_decode8(cb + (size_t)m * 8u, cm);
        float d = 0.f;
        #pragma unroll
        for (int t = 0; t < 8; t++) { const float x = cm[t] - ci[t]; d += x * x; }
        if (d >= bd[K - 1u]) continue;
        uint32_t p = K - 1u;
        while (p > 0u && bd[p - 1u] > d) { bd[p] = bd[p - 1u]; bi[p] = bi[p - 1u]; p--; }
        bd[p] = d; bi[p] = (uint16_t)m;
    }
    for (uint32_t k = 0; k < K; k++) nbr[(size_t)i * K + k] = bi[k];
}

int ds4_gpu_v41_vq_knn(const uint8_t *cb_e4m3, uint32_t nc, uint32_t K, uint16_t *nbr_out) {
    if (!cb_e4m3 || !nbr_out || !nc || nc > 65536u || !K || K > VS_KMAX) return 0;
    uint8_t *dcb = NULL; uint16_t *dn = NULL;
    int ok = cudaMalloc(&dcb, (size_t)nc * 8u) == cudaSuccess && cudaMalloc(&dn, (size_t)nc * K * 2u) == cudaSuccess &&
             cudaMemcpy(dcb, cb_e4m3, (size_t)nc * 8u, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        vs_knn_kernel<<<(nc + 127u) / 128u, 128, 0, g_cur_stream>>>(dcb, nc, K, dn);
        ok = cuda_ok(cudaGetLastError(), "vq knn") && cudaStreamSynchronize(g_cur_stream) == cudaSuccess &&
             cudaMemcpy(nbr_out, dn, (size_t)nc * K * 2u, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    if (dcb) cudaFree(dcb);
    if (dn) cudaFree(dn);
    return ok;
}

/* ② 选码: 一线程一组 */
__global__ static void vs_select_kernel(const float *dwm, const uint8_t *blob, uint32_t e, uint32_t which, uint32_t R, uint32_t C, const float *gov,
                                        const uint16_t *nbr, uint32_t K, float *delta, uint16_t *newc, uint16_t *curc, int *bad) {
    const uint32_t ng = R * (C >> 3), idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ng) return;
    const v41_vq_mat mm = v41_vq_open<1>(blob, (int)e, (int)which, R, C, gov);
    if (!mm.ok) { if (idx == 0) atomicExch(bad, 1); return; }
    const uint32_t r = idx / (C >> 3), j = idx % (C >> 3);
    const uint32_t cur = vqb_code<1>(mm, r, j);
    const float4 *gp = (const float4 *)(dwm + (uint64_t)r * C + (uint64_t)j * 8u);
    const float4 g0 = gp[0], g1 = gp[1];
    const float g[8] = { g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w };
    float cc[8]; vs_decode8(mm.cb + (size_t)cur * 8u, cc);
    float base = 0.f;
    #pragma unroll
    for (int t = 0; t < 8; t++) base += g[t] * cc[t];
    float best = 0.f; uint32_t bk = cur;
    for (uint32_t k = 0; k < K; k++) {
        const uint32_t cand = nbr[(size_t)cur * K + k];
        float ck[8]; vs_decode8(mm.cb + (size_t)cand * 8u, ck);
        float s = 0.f;
        #pragma unroll
        for (int t = 0; t < 8; t++) s += g[t] * ck[t];
        const float d = s - base;
        if (d < best) { best = d; bk = cand; }
    }
    delta[idx] = best; newc[idx] = (uint16_t)bk; curc[idx] = (uint16_t)cur;
}

/* dw_off = 该 (槽, 矩阵) 在 dw 里的 float 偏移(主机按 vg_slot_floats / vg_which_off 算); 输出三份主机数组各 R×C/8 个 */
int ds4_gpu_v41_vq_vselect(const ds4_gpu_tensor *dw, uint64_t dw_off, const void *model_map, uint64_t model_size, uint64_t blob_offset, uint64_t blob_bytes,
                           uint32_t e, uint32_t which, uint32_t R, uint32_t C, const float *gov_dev, const uint16_t *nbr_host, uint32_t nc, uint32_t K,
                           float *delta_out, uint16_t *code_out, uint16_t *cur_out) {
    if (!dw || !nbr_host || !K || K > VS_KMAX || (C & 7u)) return 0;
    const uint8_t *blob = (const uint8_t *)cuda_model_range_cached_ptr(model_map, blob_offset, blob_bytes);
    if (!blob && (g_model_registered || g_model_device_owned)) blob = (const uint8_t *)cuda_model_range_ptr(model_map, blob_offset, blob_bytes, "vstep blob");
    if (!blob) { fprintf(stderr, "ds4: [V步] 专家 blob 不在设备上(流式装载路不接选码)\n"); return 0; }
    const uint32_t ng = R * (C >> 3);
    uint16_t *dn = NULL, *dnew = NULL, *dcur = NULL; float *dd = NULL; int *bad = NULL;
    int ok = cudaMalloc(&dn, (size_t)nc * K * 2u) == cudaSuccess && cudaMalloc(&dnew, (size_t)ng * 2u) == cudaSuccess &&
             cudaMalloc(&dcur, (size_t)ng * 2u) == cudaSuccess && cudaMalloc(&dd, (size_t)ng * 4u) == cudaSuccess && cudaMalloc(&bad, 4) == cudaSuccess &&
             cudaMemcpy(dn, nbr_host, (size_t)nc * K * 2u, cudaMemcpyHostToDevice) == cudaSuccess && cudaMemset(bad, 0, 4) == cudaSuccess;
    int bad_h = 0;
    if (ok) {
        vs_select_kernel<<<(ng + 255u) / 256u, 256, 0, g_cur_stream>>>((const float *)dw->ptr + dw_off, blob, e, which, R, C, gov_dev, dn, K, dd, dnew, dcur, bad);
        ok = cuda_ok(cudaGetLastError(), "vq vselect") && cudaStreamSynchronize(g_cur_stream) == cudaSuccess &&
             cudaMemcpy(delta_out, dd, (size_t)ng * 4u, cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(code_out, dnew, (size_t)ng * 2u, cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(cur_out, dcur, (size_t)ng * 2u, cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(&bad_h, bad, 4, cudaMemcpyDeviceToHost) == cudaSuccess && !bad_h;
    }
    if (bad_h) fprintf(stderr, "ds4: [V步] 专家 %u 矩阵 %u 载荷头不认(非 v3 / 形状 %u×%u 不对)\n", e, which, R, C);
    if (dn) cudaFree(dn);
    if (dnew) cudaFree(dnew);
    if (dcur) cudaFree(dcur);
    if (dd) cudaFree(dd);
    if (bad) cudaFree(bad);
    return ok;
}

/* ③ 改位。12 位字段占 [bit, bit+12), 可能跨 2 个字节; 每个字节只改字段覆盖的那几位(与相邻组共用字节 ⇒ 位级 CAS) */
__device__ static void vs_set_bits(uint8_t *p, uint32_t mask, uint32_t val) {
    const uintptr_t a = (uintptr_t)p; uint32_t *w = (uint32_t *)(a & ~(uintptr_t)3u); const uint32_t sh = (uint32_t)(a & 3u) * 8u;
    const uint32_t m = mask << sh, v = (val & mask) << sh;
    uint32_t old = *w, assumed;
    do { assumed = old; const uint32_t nw = (assumed & ~m) | v; if (nw == assumed) break; old = atomicCAS(w, assumed, nw); } while (assumed != old);
}
__device__ static void vs_set_field12(uint8_t *rowp, uint32_t j, uint32_t v) {
    const uint32_t bit = j * 12u, b0 = bit >> 3, b1 = (bit + 11u) >> 3;
    for (uint32_t b = b0; b <= b1; b++) {
        const uint32_t lo = bit > 8u * b ? bit : 8u * b, hi = bit + 12u < 8u * b + 8u ? bit + 12u : 8u * b + 8u, n = hi - lo;
        vs_set_bits(rowp + b, ((1u << n) - 1u) << (lo - 8u * b), ((v >> (lo - bit)) & ((1u << n) - 1u)) << (lo - 8u * b));
    }
}
__global__ static void vs_patch_kernel(uint8_t *blob, const vq_prec *recs, uint32_t n, uint32_t IN, uint32_t MID, uint32_t OUT, int apply, uint32_t *mismatch, int *bad) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const vq_prec rc = recs[i];
    const uint32_t e = rc.ew & 0xffffu, which = rc.ew >> 16, R = which == 2u ? OUT : MID, C = which == 2u ? MID : IN, code = rc.code & 0xffffu;
    const v41_vq_mat mm = v41_vq_open<1>(blob, (int)e, (int)which, R, C, NULL);
    if (!mm.ok || rc.row >= R || rc.j >= mm.nidx_row || code >= mm.nc || (code >= 4096u && !mm.ex)) { atomicExch(bad, 1); return; }
    if (apply) {
        uint8_t *rowp = (uint8_t *)mm.ix + (((uint64_t)rc.row * mm.nidx_row * 12u) >> 3);
        vs_set_field12(rowp, rc.j, code & 0xfffu);
        if (mm.ex) vs_set_bits((uint8_t *)mm.ex + (size_t)rc.row * ((mm.nidx_row + 7u) >> 3) + (rc.j >> 3), 1u << (rc.j & 7u), ((code >> 12) & 1u) << (rc.j & 7u));
    } else if (vqb_code<1>(mm, rc.row, rc.j) != code) atomicAdd(mismatch, 1u);
}

/* 流式层(锁页暂存)的补丁表: 每层一份记录拷贝, 每次 v41_stage_fill 拷完这一层再盖一遍(主机侧按同一式子改位) */
#define V41_VSP_MAX 128u
static struct { const uint8_t *host; uint64_t bytes; vq_prec *recs; uint32_t n, IN, MID, OUT; } g_vsp[V41_VSP_MAX];
static uint32_t g_vsp_n;

static void vs_host_set_bits(uint8_t *p, uint32_t mask, uint32_t val) { *p = (uint8_t)((*p & ~mask) | (val & mask)); }
static int vs_host_apply(uint8_t *blob, const vq_prec *recs, uint32_t n, uint32_t IN, uint32_t MID, uint32_t OUT) {   /* 与 vs_patch_kernel 同式, 主机内存 */
    for (uint32_t i = 0; i < n; i++) {
        const vq_prec rc = recs[i];
        const uint32_t e = rc.ew & 0xffffu, which = rc.ew >> 16, R = which == 2u ? OUT : MID, C = which == 2u ? MID : IN, code = rc.code & 0xffffu;
        const uint64_t off = ds4vq_slot(blob, (int)e, (int)which);
        if (!off) return 0;
        const uint8_t *pay = blob + off;
        uint32_t flags; memcpy(&flags, pay + 16, 4);
        const uint32_t nidx_row = C / 8u, mrow = (nidx_row * 12u + 7u) / 8u;
        uint8_t *gr = (uint8_t *)pay + 32, *ix = gr + (size_t)R * 2u, *ex = (flags & 2u) ? ix + (size_t)R * mrow : NULL;
        if (rc.row >= R || rc.j >= nidx_row || (code >= 4096u && !ex)) return 0;
        uint8_t *rowp = ix + (((uint64_t)rc.row * nidx_row * 12u) >> 3);
        const uint32_t bit = rc.j * 12u, b0 = bit >> 3, b1 = (bit + 11u) >> 3, v = code & 0xfffu;
        for (uint32_t b = b0; b <= b1; b++) {
            const uint32_t lo = bit > 8u * b ? bit : 8u * b, hi = bit + 12u < 8u * b + 8u ? bit + 12u : 8u * b + 8u, nb = hi - lo;
            vs_host_set_bits(rowp + b, ((1u << nb) - 1u) << (lo - 8u * b), ((v >> (lo - bit)) & ((1u << nb) - 1u)) << (lo - 8u * b));
        }
        if (ex) vs_host_set_bits(ex + (size_t)rc.row * ((nidx_row + 7u) >> 3) + (rc.j >> 3), 1u << (rc.j & 7u), ((code >> 12) & 1u) << (rc.j & 7u));
    }
    return 1;
}
static void v41_vsp_stage_apply(uintptr_t reg, uint64_t bytes, uint8_t *stage) {
    for (uint32_t i = 0; i < g_vsp_n; i++) {
        const uintptr_t h = (uintptr_t)g_vsp[i].host;
        if (!(h >= reg && h + g_vsp[i].bytes <= reg + bytes)) continue;
        if (!vs_host_apply(stage + (h - reg), g_vsp[i].recs, g_vsp[i].n, g_vsp[i].IN, g_vsp[i].MID, g_vsp[i].OUT))
            fprintf(stderr, "ds4: ★[V步·补丁] 流式层 blob @%p 回盖失败★\n", (const void *)g_vsp[i].host);
    }
}

/* apply=1 盖补丁并逐条回读; recs 的 code 高 16 位是旧码(还原时调用方把新旧对调再调一次)。*mismatch = 回读与目标不同的条数(调用方不为 0 就停) */
int ds4_gpu_v41_vq_code_patch(const void *model_map, uint64_t model_size, uint64_t blob_offset, uint64_t blob_bytes, uint32_t IN, uint32_t MID, uint32_t OUT,
                              const uint32_t *recs, uint32_t n, uint32_t *mismatch) {
    if (!model_map || blob_offset > model_size || blob_bytes > model_size - blob_offset || !recs || !n) return 0;
    const uint8_t *hb = (const uint8_t *)model_map + blob_offset;
    if (!ds4vq_blob_ok(hb, (size_t)blob_bytes) || ds4vq_blob_ver(hb) != 3u) { fprintf(stderr, "ds4: [V步·补丁] blob @%llu 不是 v3\n", (unsigned long long)blob_offset); return 0; }
    *mismatch = 0;
    char *dev = (char *)cuda_model_range_cached_ptr(model_map, blob_offset, blob_bytes);
    if (!dev && g_model_device_owned && model_map == g_model_host_base) dev = (char *)cuda_model_ptr(model_map, blob_offset);
    if (dev) {
        vq_prec *dr = NULL; uint32_t *dm = NULL; int *bad = NULL, bad_h = 0;
        int ok = cudaMalloc(&dr, (size_t)n * sizeof(vq_prec)) == cudaSuccess && cudaMalloc(&dm, 4) == cudaSuccess && cudaMalloc(&bad, 4) == cudaSuccess &&
                 cudaDeviceSynchronize() == cudaSuccess && cudaMemcpy(dr, recs, (size_t)n * sizeof(vq_prec), cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemset(dm, 0, 4) == cudaSuccess && cudaMemset(bad, 0, 4) == cudaSuccess;
        if (ok) {
            vs_patch_kernel<<<(n + 255u) / 256u, 256, 0, g_cur_stream>>>((uint8_t *)dev, dr, n, IN, MID, OUT, 1, dm, bad);
            vs_patch_kernel<<<(n + 255u) / 256u, 256, 0, g_cur_stream>>>((uint8_t *)dev, dr, n, IN, MID, OUT, 0, dm, bad);
            ok = cuda_ok(cudaGetLastError(), "vq code patch") && cudaStreamSynchronize(g_cur_stream) == cudaSuccess &&
                 cudaMemcpy(mismatch, dm, 4, cudaMemcpyDeviceToHost) == cudaSuccess && cudaMemcpy(&bad_h, bad, 4, cudaMemcpyDeviceToHost) == cudaSuccess && !bad_h;
        }
        if (bad_h) fprintf(stderr, "ds4: ★[V步·补丁] 有记录载荷头不认 / 行列越界 / 12 位层给了 13 位码★\n");
        if (dr) cudaFree(dr);
        if (dm) cudaFree(dm);
        if (bad) cudaFree(bad);
        return ok;
    }
    if (g_model_registered || g_model_hmm_direct) {
        fprintf(stderr, "ds4: ★[V步·补丁] blob @%llu 走整映射路(GPU 直读文件页), 改不了索引 —— 要整模型常驻或封顶装载★\n", (unsigned long long)blob_offset);
        return 0;
    }
    /* 流式层: 记表(同一 blob 的记录追加), 暂存里正装着这一层就当场盖上并按主机读法核对 */
    uint32_t slot = g_vsp_n;
    for (uint32_t i = 0; i < g_vsp_n; i++) if (g_vsp[i].host == hb) slot = i;
    if (slot == g_vsp_n) {
        if (g_vsp_n == V41_VSP_MAX) { fprintf(stderr, "ds4: [V步·补丁] 表满(%u 层)\n", V41_VSP_MAX); return 0; }
        g_vsp[slot].host = hb; g_vsp[slot].bytes = blob_bytes; g_vsp[slot].recs = NULL; g_vsp[slot].n = 0; g_vsp[slot].IN = IN; g_vsp[slot].MID = MID; g_vsp[slot].OUT = OUT; g_vsp_n++;
    }
    vq_prec *nr = (vq_prec *)realloc(g_vsp[slot].recs, ((size_t)g_vsp[slot].n + n) * sizeof(vq_prec));
    if (!nr) return 0;
    memcpy(nr + g_vsp[slot].n, recs, (size_t)n * sizeof(vq_prec));
    g_vsp[slot].recs = nr; g_vsp[slot].n += n;
    if (g_v41_stage.valid) {
        const uintptr_t h = (uintptr_t)hb;
        if (h >= g_v41_stage.reg && h + blob_bytes <= g_v41_stage.reg + g_v41_stage.bytes) {
            (void)cudaDeviceSynchronize();
            if (!vs_host_apply(g_v41_stage.host + (h - g_v41_stage.reg), (const vq_prec *)recs, n, IN, MID, OUT)) return 0;
        }
    }
    return 1;
}
