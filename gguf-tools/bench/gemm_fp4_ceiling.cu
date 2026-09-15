/* gemm_fp4_ceiling.cu — S0 探针: 量本机(GB10/sm_121) cuBLASLt 在【V4.1 真实形状】上的
 * 窄精度 GEMM 天花板, 并回答 speed.md §2 的那一格: FP4 激活到底能不能用。
 *
 * 【为什么要它】speed.md 的预填账(§4.1)写着"专家 GEMM 走 FP4 tensor core", 但那是规格推的,
 * 不是量的。四档一起量才知道差在哪:
 *   ① BF16              —— 现役 cuBLAS 路的参照
 *   ② FP8×FP8 (MXFP8)   —— 官方 model.py 的激活口径(act_quant 1×32 ue8m0)+ 骨架权重
 *   ③ FP8×FP4           —— 官方 fp4_gemm 的口径(FP8 激活 × FP4 专家权重)
 *   ④ FP4×FP4 (MXFP4)   —— speed.md §2 要的"全 FP4"(激活也降到 E2M1)
 * ④ 比 ③ 快不了多少 ⇒ 全 FP4 只剩省激活暂存的意义, 质量风险不值得; 快很多 ⇒ 值得去过五指标门。
 *
 * 形状(config.json 实值, 不是估): dim 5120 / moe_inter 2304 / q_lora 1280 / 64 头 × 512 /
 * o_groups 8 × o_lora 1024。默认 M=4096 = speed.md P2 的预填块。
 *
 * 【坑】窄精度矩阵 cuBLASLt 只接受 TN 布局(A、B 都按 K 连续存), 正好与我们盘上的行主序一致;
 * 块缩放因子要求 m 补齐到 128、k/32 补齐到 4(Blackwell 的 swizzle 布局), 少分配不报错只越界读。
 * 本探针一律按补齐后的尺寸分配并清零 —— 我们量的是【速度】, 缩放值本身不参与判决。
 *
 * 用法: gemm_fp4_ceiling [M=4096] [iters=50]
 * 编译: nvcc -O3 -arch=sm_121 -o gemm_fp4_ceiling gemm_fp4_ceiling.cu -lcublasLt -lcublas
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <time.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define LK(x) do{ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ printf("cuBLASLt err %d @%d\n",(int)s,__LINE__); exit(1);} }while(0)

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static size_t up(size_t v, size_t m){ return (v + m - 1) / m * m; }

/* 一格测试 = 一个形状 × 一种精度组合 */
typedef struct {
    const char *name;          /* 打印用 */
    cudaDataType_t ta, tb;     /* A(权重) / B(激活) 元素类型 */
    cublasLtMatmulMatrixScale_t sa, sb;   /* 块缩放模式; SCALAR_32F = 不带块缩放 */
    int blk;                   /* 缩放块大小(元素); 0 = 无 */
} prec_t;

/* 返回 TFLOPS; 不支持返回 -1(把 cuBLASLt 的"没有可用算法"与真错误分开: 前者是本探针的答案之一) */
static double run_one(cublasLtHandle_t lt, cudaStream_t st, const prec_t *p,
                      int m, int n, int k, int iters, void *workspace, size_t wsbytes) {
    cublasLtMatmulDesc_t op = NULL;
    cublasLtMatrixLayout_t la = NULL, lb = NULL, ld = NULL;
    void *A = NULL, *B = NULL, *D = NULL, *SA = NULL, *SB = NULL;
    /* 每元素字节: FP4 半字节 ⇒ 按 k/2 算行字节 */
    const int a4 = (p->ta == CUDA_R_4F_E2M1), b4 = (p->tb == CUDA_R_4F_E2M1);
    const size_t abytes = (size_t)m * (a4 ? k / 2 : (p->ta == CUDA_R_16BF ? k * 2 : k));
    const size_t bbytes = (size_t)n * (b4 ? k / 2 : (p->tb == CUDA_R_16BF ? k * 2 : k));
    CK(cudaMalloc(&A, abytes)); CK(cudaMalloc(&B, bbytes));
    CK(cudaMalloc(&D, (size_t)m * n * 2));          /* 出口 bf16(与引擎同) */
    CK(cudaMemset(A, 0x11, abytes)); CK(cudaMemset(B, 0x11, bbytes));
    if (p->blk) {   /* 缩放张量按 Blackwell swizzle 补齐: m→128 倍数, k/blk→4 倍数 */
        const size_t sa_n = up((size_t)m, 128) * up((size_t)k / p->blk, 4);
        const size_t sb_n = up((size_t)n, 128) * up((size_t)k / p->blk, 4);
        CK(cudaMalloc(&SA, sa_n)); CK(cudaMalloc(&SB, sb_n));
        CK(cudaMemset(SA, 127, sa_n)); CK(cudaMemset(SB, 127, sb_n));   /* ue8m0 127 = 2^0 */
    }
    LK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA));
    LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB));
    if (p->blk) {
        LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &SA, sizeof SA));
        LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &SB, sizeof SB));
        LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &p->sa, sizeof p->sa));
        LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &p->sb, sizeof p->sb));
    }
    /* TN: A 存成 (k, m) 列主序 = 我们盘上的 [m][k] 行主序; B 同理 */
    LK(cublasLtMatrixLayoutCreate(&la, p->ta, k, m, k));
    LK(cublasLtMatrixLayoutCreate(&lb, p->tb, k, n, k));
    LK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, m, n, m));

    cublasLtMatmulPreference_t pref = NULL;
    LK(cublasLtMatmulPreferenceCreate(&pref));
    LK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsbytes, sizeof wsbytes));
    cublasLtMatmulHeuristicResult_t heur[1]; int nheur = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 1, heur, &nheur);
    double tf = -1.0;
    if (hs == CUBLAS_STATUS_SUCCESS && nheur > 0) {
        const float one = 1.f, zero = 0.f;
        cublasStatus_t rs = cublasLtMatmul(lt, op, &one, A, la, B, lb, &zero, D, ld, D, ld,
                                           &heur[0].algo, workspace, wsbytes, st);
        if (rs == CUBLAS_STATUS_SUCCESS) {
            CK(cudaStreamSynchronize(st));
            const double t0 = now_s();
            for (int i = 0; i < iters; i++)
                (void)cublasLtMatmul(lt, op, &one, A, la, B, lb, &zero, D, ld, D, ld,
                                     &heur[0].algo, workspace, wsbytes, st);
            CK(cudaStreamSynchronize(st));
            const double dt = now_s() - t0;
            tf = 2.0 * m * n * k * iters / dt * 1e-12;
        } else {
            printf("      (matmul 返回 %d)\n", (int)rs);
        }
    }
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(op);
    cudaFree(A); cudaFree(B); cudaFree(D); if (SA) cudaFree(SA); if (SB) cudaFree(SB);
    return tf;
}

/* ---- --check: 把 NVFP4 缩放张量的 swizzle 布局钉死 ----
 * 【为什么必须做】布局猜错 cuBLASLt 不报错, 只出一张"形状对、数值全错"的结果 —— 这正是仓里
 * "不报错只出假数"那一类。做法不靠读文档: 把 A 的 nibble 全设成 1.0、B 全设成 1.0、所有缩放设 1.0,
 * 则 C 每个元素 = k。再把 A 缩放张量的某一个字节改成 2.0, 那一块(16 个元素)的贡献从 16 变 32,
 * ⇒ 恰好一整行输出 +16。看是哪一行变了, 就反解出"缩放字节偏移 ↔ (行, k 块)"的映射。
 * 猜测公式(NVIDIA 128×4 swizzle): off(r,c) = ((r/128)*ceil(nkb/4) + c/4)*512 + (r%32)*16 + ((r%128)/32)*4 + c%4 */
static size_t nvfp4_scale_off(int r, int c, int nkb) {
    const int tiles_c = (nkb + 3) / 4;
    return (size_t)((r / 128) * tiles_c + c / 4) * 512 + (size_t)(r % 32) * 16 + (size_t)((r % 128) / 32) * 4 + (size_t)(c % 4);
}

static int layout_check(cublasLtHandle_t lt, cudaStream_t st, void *ws, size_t wsbytes) {
    const int m = 256, n = 8, k = 64, blk = 16, nkb = k / blk;
    const size_t sa_n = up((size_t)m, 128) * up((size_t)nkb, 4), sb_n = up((size_t)n, 128) * up((size_t)nkb, 4);
    unsigned char *hA = (unsigned char *)malloc((size_t)m * k / 2), *hB = (unsigned char *)malloc((size_t)n * k / 2);
    unsigned char *hSA = (unsigned char *)malloc(sa_n), *hSB = (unsigned char *)malloc(sb_n);
    memset(hA, 0x22, (size_t)m * k / 2);   /* E2M1 码 2 = 1.0, 两个一字节 */
    memset(hB, 0x22, (size_t)n * k / 2);
    memset(hSA, 0x38, sa_n); memset(hSB, 0x38, sb_n);   /* e4m3 0x38 = 1.0 */
    void *A, *B, *D, *SA, *SB;
    CK(cudaMalloc(&A, (size_t)m * k / 2)); CK(cudaMalloc(&B, (size_t)n * k / 2));
    CK(cudaMalloc(&D, (size_t)m * n * 4)); CK(cudaMalloc(&SA, sa_n)); CK(cudaMalloc(&SB, sb_n));
    CK(cudaMemcpy(A, hA, (size_t)m * k / 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(B, hB, (size_t)n * k / 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(SB, hSB, sb_n, cudaMemcpyHostToDevice));

    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, ld;
    LK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA));
    LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB));
    LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &SA, sizeof SA));
    LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &SB, sizeof SB));
    LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof sm));
    LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof sm));
    LK(cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, k, m, k));
    LK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, k, n, k));
    LK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, m, n, m));
    cublasLtMatmulPreference_t pref; LK(cublasLtMatmulPreferenceCreate(&pref));
    LK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsbytes, sizeof wsbytes));
    cublasLtMatmulHeuristicResult_t heur[1]; int nh = 0;
    if (cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 1, heur, &nh) != CUBLAS_STATUS_SUCCESS || nh == 0) {
        printf("★fp32 出口的 NVFP4 无算法, 布局验证换 bf16 出口重试★\n"); return 1;
    }
    float *hD = (float *)malloc((size_t)m * n * 4);
    const float one = 1.f, zero = 0.f;
    #define RUN() do { LK(cublasLtMatmul(lt, op, &one, A, la, B, lb, &zero, D, ld, D, ld, &heur[0].algo, ws, wsbytes, st)); \
                       CK(cudaStreamSynchronize(st)); CK(cudaMemcpy(hD, D, (size_t)m * n * 4, cudaMemcpyDeviceToHost)); } while (0)
    CK(cudaMemcpy(SA, hSA, sa_n, cudaMemcpyHostToDevice)); RUN();
    printf("--- NVFP4 布局验证(m=%d n=%d k=%d, A/B 全 1.0, 缩放全 1.0)\n", m, n, k);
    printf("  基线 C[0][0]=%.1f  (应 = k = %d)%s\n", hD[0], k, hD[0] == (float)k ? "  ✓" : "  ★对不上, nibble 或布局有问题★");
    if (hD[0] != (float)k) return 1;
    int bad = 0, probed = 0;
    const int rs[] = {0, 1, 31, 32, 127, 128, 200}, cs[] = {0, 1, 3, 2, 0, 1, 3};
    for (size_t t = 0; t < sizeof rs / sizeof rs[0]; t++) {
        const int r = rs[t], c = cs[t];
        if (r >= m || c >= nkb) continue;
        const size_t off = nvfp4_scale_off(r, c, nkb);
        memset(hSA, 0x38, sa_n); hSA[off] = 0x40;   /* e4m3 0x40 = 2.0 */
        CK(cudaMemcpy(SA, hSA, sa_n, cudaMemcpyHostToDevice)); RUN();
        int hit = -1, nhit = 0;
        for (int i = 0; i < m; i++) if (hD[i] != (float)k) { nhit++; if (hit < 0) hit = i; }
        probed++;
        const int ok = (nhit == 1 && hit == r && hD[r] == (float)(k + blk));
        if (!ok) bad++;
        printf("  预期(行 %3d, k块 %d) → 偏移 %4zu :  实测变的行 %d(共 %d 行, 值 %.1f)  %s\n",
               r, c, off, hit, nhit, hit >= 0 ? hD[hit] : 0.f, ok ? "✓" : "★不符★");
    }
    printf("  %d/%d 格吻合 ⇒ %s\n", probed - bad, probed,
           bad ? "★128×4 swizzle 公式不对, 生产侧不许照它写★" : "★缩放布局 = 128×4 swizzle, 公式可用★");
    free(hA); free(hB); free(hSA); free(hSB); free(hD);
    cudaFree(A); cudaFree(B); cudaFree(D); cudaFree(SA); cudaFree(SB);
    cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld); cublasLtMatmulDescDestroy(op);
    return bad ? 1 : 0;
    #undef RUN
}

int main(int argc, char **argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 4096;
    const int iters = argc > 2 ? atoi(argv[2]) : 50;
    const int only_check = (argc > 1 && !strcmp(argv[1], "--check"));
    int dev = 0; cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
    printf("%s  SM %d.%d × %d   M(预填块)=%d  ×%d 次\n",
           pr.name, pr.major, pr.minor, pr.multiProcessorCount, M, iters);

    cublasLtHandle_t lt; LK(cublasLtCreate(&lt));
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    const size_t wsbytes = 128u << 20;
    void *ws = NULL; CK(cudaMalloc(&ws, wsbytes));
    if (only_check) { const int rc = layout_check(lt, st, ws, wsbytes); cudaFree(ws); cublasLtDestroy(lt); return rc; }

    const prec_t precs[] = {
        { "BF16(参照)",   CUDA_R_16BF,   CUDA_R_16BF,   CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, 0 },
        { "FP8×FP8 MX",   CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, 32 },
        { "FP4权×FP8激活", CUDA_R_4F_E2M1, CUDA_R_8F_E4M3, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, 32 },
        { "FP4×FP4 MX",   CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, 32 },
        { "FP4×FP4 NV16", CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, 16 },
    };
    /* 形状全部来自 config.json 实值 */
    struct { const char *nm; int m, k; } shapes[] = {
        { "专家 w1/w3  [2304,5120]", 2304, 5120 },
        { "专家 w2     [5120,2304]", 5120, 2304 },
        { "骨架 wo_b   [5120,8192]", 5120, 8192 },
        { "骨架 wq_b   [32768,1280]", 32768, 1280 },
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        printf("\n%s   (m=%d k=%d n=%d, 单次 %.1f GFLOP)\n", shapes[s].nm, shapes[s].m, shapes[s].k, M,
               2.0 * shapes[s].m * shapes[s].k * M * 1e-9);
        for (size_t p = 0; p < sizeof precs / sizeof precs[0]; p++) {
            const double tf = run_one(lt, st, &precs[p], shapes[s].m, M, shapes[s].k, iters, ws, wsbytes);
            if (tf < 0) printf("  %-16s ★本机无可用算法(cuBLASLt 不支持这一组合)★\n", precs[p].name);
            else        printf("  %-16s %8.2f TFLOPS   %.3f ms/次\n", precs[p].name, tf,
                               2.0 * shapes[s].m * shapes[s].k * M / tf * 1e-12 * 1e3);
        }
    }
    cudaFree(ws); cublasLtDestroy(lt);
    return 0;
}
