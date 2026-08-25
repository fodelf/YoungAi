constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];

#ifdef DS4_METAL_HAS_TENSOR
template<
    short NR0, short NR1,
    typename SA, typename SA_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1>
kernel void kernel_mul_mm_mpp(
        constant ds4_metal_args_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;
    const int im = tgpig.z;
    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    threadgroup SA *sa = (threadgroup SA *)shmem;
    threadgroup SA *sb = sa + NR0*NK;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(sb, dextents<int32_t, 2>(NK, NR1));

    device const T1 *ptrB = (device const T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11/sizeof(T1);

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (!FC_mul_mm_bc_out || r0 + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    device const T0 *row_ptr = (device const T0 *)(srcA + args.nb01*(r0 + row) + offset0);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? (SA)row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos/(16*nl);
                    const short il = (k_pos/16)%nl;
                    device const block_q *row_ptr = (device const block_q *)(srcA + args.nb01*(r0 + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (SA)0;
                }
            }
        }
        for (int work = tiitg; work < NK*NR1; work += NUM_THREADS) {
            const int col = work/NK;
            const int k = work%NK;
            if ((!FC_mul_mm_bc_out && !FC_mul_mm_bc_inp) ||
                (r1 + col < N && loop_k + k < K)) {
                sb[col*NK + k] = (SA)ptrB[(uint64_t)(r1 + col)*strideB + loop_k + k];
            } else {
                sb[col*NK + k] = (SA)0;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(0, 0);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_batch = (device float *)dst + im*N*M;
    if (!FC_mul_mm_bc_out) {
        device float *dst_tile = dst_batch + r0 + (uint64_t)r1*M;
        auto tD = tensor(dst_tile, dextents<int32_t, 2>(NR0, NR1), array<int, 2>({1, M}));
        cT.store(tD);
    } else {
        auto tD = tensor(dst_batch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
        auto mD = tD.slice(r0, r1);
        cT.store(mD);
    }
}

typedef decltype(kernel_mul_mm_mpp<64, 32, half, half4x4, float4x4, 1, dequantize_f32, float, float4x4, float>) mul_mm_mpp_t;

template [[host_name("kernel_mul_mm_f16_f32_mpp")]]  kernel mul_mm_mpp_t kernel_mul_mm_mpp<64, 32, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;

// Retained Metal4/TensorOps dense prefill kernel.  The legacy MPP prototype
// staged both operands in threadgroup memory; this version stages only the
// model weight tile and lets MPP read the dense RHS activation matrix directly
// from device memory.  That direct-RHS shape was the clear win for DS4's large
// aligned F16/Q8_0 prompt matmuls.  The host selects the widest token tile that
// evenly divides the batch, with 128-token tiles retained after the 64-token
// retest was neutral or slower.
template<
    short NR1,
    typename SA, typename SA_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1>
kernel void kernel_mul_mm_mpp_direct_rhs(
        constant ds4_metal_args_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NR0 = 64;
    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;
    const int im = tgpig.z;
    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    threadgroup SA *sa = (threadgroup SA *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));

    device T1 *ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11/sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (r0 + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    device const T0 *row_ptr = (device const T0 *)(srcA + args.nb01*(r0 + row) + offset0);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? (SA)row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos/(16*nl);
                    const short il = (k_pos/16)%nl;
                    device const block_q *row_ptr = (device const block_q *)(srcA + args.nb01*(r0 + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (SA)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_batch = (device float *)dst + im*N*M;
    auto tD = tensor(dst_batch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
    auto mD = tD.slice(r0, r1);
    cT.store(mD);
}

typedef decltype(kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, float4x4, 1, dequantize_f32, float, float4x4, float>) mul_mm_mpp_direct_rhs_t;

template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs_n64")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<64, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs_n128")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<128, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, block_q8_0, 2, dequantize_q8_0, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs_n64")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<64, half, half4x4, block_q8_0, 2, dequantize_q8_0, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<128, half, half4x4, block_q8_0, 2, dequantize_q8_0, float, float4x4, float>;
#endif

// Tiled matrix-matrix kernel used for prompt batches larger than 8. DS4 uses
// this to turn prefill into large simdgroup matrix operations; each block_q
// contains 16*nl weights.
