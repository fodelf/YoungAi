/* cuda_embed_norm_kernels_4.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * embed/corr/repeat/norm/matmul(f16/f32) kernel 族 + kv fp8 rope store。
 */
__global__ static void fp8_kv_quantize_kernel(float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (off + tid < n_nope) {
            float q = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
            xr[off + tid] = q;
        }
        __syncthreads();
    }
}

/* kv 尾链三合一(2026-08-20 megakernel G1b): rope(rot尾段) → fp8(nope前段, 64线程树
 * 逐位照抄, barrier 全 block 陪跑) → store(全行 f16 往返)。decode n=1 单行单 block。 */
__global__ static void kv_rope_fp8_store_kernel(
        float *kv, float *raw, uint32_t raw_cap, uint32_t raw_row0,
        uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    /* 小批(投机 verify, 09-07): grid.x = token, 每 block 一行, 逐 token 数学与单 block 单行完全相同 */
    const uint32_t t = blockIdx.x;
    kv += (uint64_t)t * head_dim;
    const uint32_t raw_row = (raw_row0 + t) % raw_cap;
    const uint32_t pos = pos0 + t;
    const uint32_t tid = threadIdx.x;
    const uint32_t n_nope = head_dim - n_rot;
    /* 段B: rope(先于 fp8, 与原三发同序; 只动 [n_nope, head_dim)) */
    if (tid < (n_rot >> 1)) {
        const uint32_t i = tid * 2u;
        float corr0 = 0.0f, corr1 = 0.0f;
        if (ext_factor != 0.0f) {
            float denom = 2.0f * logf(freq_base);
            corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
            corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
            corr0 = fmaxf(0.0f, corr0);
            corr1 = fminf((float)(n_rot - 1), corr1);
        }
        float theta_extrap = (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        const float sn = sinf(theta) * mscale;
        float *tail = kv + n_nope;
        const float x0 = tail[i], x1 = tail[i + 1];
        tail[i] = x0 * c - x1 * sn;
        tail[i + 1] = x0 * sn + x1 * c;
    }
    __syncthreads();
    /* 段A: fp8(逐位照抄 fp8_kv_quantize_kernel; barrier 全 block 到达, 归约仍 64 线程同序) */
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (tid < 64u) {
            if (off + tid < n_nope) v = kv[off + tid];
            scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        }
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (tid < 64u && off + tid < n_nope) {
            float q = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
            kv[off + tid] = q;
        }
        __syncthreads();
    }
    /* 段C: store 全行(f16 往返, 语义同 store_raw_kv_batch_kernel n=1) */
    for (uint32_t d = tid; d < head_dim; d += blockDim.x)
        raw[(uint64_t)(raw_row % raw_cap) * head_dim + d] = __half2float(__float2half(kv[d]));
}

int ds4_gpu_kv_rope_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv, ds4_gpu_tensor *raw_cache,
        uint32_t raw_cap, uint32_t raw_row, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos, uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!kv || !raw_cache || raw_cap == 0 || n_rot > head_dim || (n_rot & 1u) ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    ds4_launch_pdl(kv_rope_fp8_store_kernel, 1, 256, 0, 0,
        (float *)kv->ptr, (float *)raw_cache->ptr, raw_cap, raw_row,
        head_dim, n_rot, pos, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "kv_rope_fp8_store launch");
}
/* 小批版: kv 连续 n_tok 行, 位置 pos0.., 环行 (pos0+t) % raw_cap; 一发 n_tok 个 block(逐 token 与单发同数值) */
int ds4_gpu_kv_rope_fp8_store_raw_batch_tensor(
        ds4_gpu_tensor *kv, ds4_gpu_tensor *raw_cache,
        uint32_t raw_cap, uint32_t pos0, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!kv || !raw_cache || raw_cap == 0 || n_tok == 0 || n_rot > head_dim || (n_rot & 1u) ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    ds4_launch_pdl(kv_rope_fp8_store_kernel, n_tok, 256, 0, 0,
        (float *)kv->ptr, (float *)raw_cache->ptr, raw_cap, pos0 % raw_cap,
        head_dim, n_rot, pos0, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "kv_rope_fp8_store batch launch");
}

