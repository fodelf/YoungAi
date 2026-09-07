struct ds4_metal_args_cpy {
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Typed copy/conversion between graph tensors. DS4 uses this for layout
// materialization and F32/F16 conversions at graph boundaries such as KV/cache
// packing and compressor pooling.
template<typename T0, typename T1>
kernel void kernel_cpy_t_t(
        constant ds4_metal_args_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tiitg/ntg[0];
    const int iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device T1 * dst_data = (device T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = iw0*ntg[0] + tiitg%ntg[0]; i00 < args.ne00; ) {
        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        dst_data[i00] = (T1) src[0];
        break;
    }
}

typedef decltype(kernel_cpy_t_t<float, float>) kernel_cpy_t;
// Host-visible copy/conversion variants used by the DS4 graph.
template [[host_name("kernel_cpy_f32_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<float, float>;
template [[host_name("kernel_cpy_f32_f16")]] kernel kernel_cpy_t kernel_cpy_t_t<float, half>;
template [[host_name("kernel_cpy_f16_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<half, float>;

// 压缩缓存行格式(ds4_gpu_core.h DS4_GPU_COMP_ROW_*): [448 维 f16][64 维 f32], 行 1152 B。两后端唯一格式:
// FP8 段 f16 精确, RoPE 段保 f32(转 f16 会丢位置精度)。
// → 连续 f16 行: flash-attention 的 K/V 打包(RoPE 段在此转 f16, 与打包 f32 缓存时同数值); n = 行数 × 512
kernel void kernel_dsv4_comp_rows_to_f16(
        constant uint32_t & n,
        device const char * src,
        device       half * dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    const uint r = gid / 512u, d = gid - r * 512u;
    device const char *row = src + (uint64_t)r * 1152u;
    dst[gid] = d < 448u ? ((device const half *)row)[d]
                        : half(((device const float *)(row + 896u))[d - 448u]);
}

// f32 [rows][512](暂存) → 压缩缓存行
kernel void kernel_dsv4_comp_rows_commit(
        constant uint32_t & n,
        device const float * src,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    const uint r = gid / 512u, d = gid - r * 512u;
    device char *row = dst + (uint64_t)r * 1152u;
    const float v = src[gid];
    if (d < 448u) ((device half *)row)[d] = half(v);
    else ((device float *)(row + 896u))[d - 448u] = v;
}
