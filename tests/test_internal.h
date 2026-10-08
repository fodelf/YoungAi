#ifndef DS4_TEST_INTERNAL_H
#define DS4_TEST_INTERNAL_H
/* ds4_test 拆分共享头 (重构阶段8): 原单文件 tests/ds4_test.c 按 suite 机械拆成
 * tests/t_*.c 后, 跨文件助手与各 suite 入口统一在这里声明。断言基元
 * (TEST_ASSERT/test_failures) 复用 server_tests_internal.h — 两组测试文件
 * 链进同一个 ds4_test 可执行, 失败计数必须是同一个。
 * GPU-only 声明包在 DS4_NO_GPU guard 里, 与各定义文件的整文件 guard 配对
 * (guard 外的 TU 用文件尾 dummy typedef 防空)。 */
#include "server_tests_internal.h"

/* t_engine.c(定义在 GPU guard 内): 运行器在无 GPU 构建里也引用它, 声明放
 * guard 外 — 该构建本就不链 GPU suite, 缺定义只在真被调用时才暴露(与拆分前
 * 隐式声明的行为一致, 但编译期干净)。 */
const char *test_model_path(void);
/* ds4_test 的带值参数 (t_main.c 解析; 默认值在各消费点)。 */
extern const char *g_test_model;
extern const char *g_test_vector_file;
extern const char *g_test_long_prompt;
extern const char *g_test_mpp_case;
extern const char *g_test_local_golden;
extern int g_test_keep_metal4;   /* --keep-metal4: logprob 向量测试期间不强制关 Metal4 */

#ifndef DS4_NO_GPU
#include "../ds4_gpu.h"
#include <math.h>

/* ---- t_engine.c: 共享引擎缓存 + env 保存恢复 + f16 转换 ---- */
ds4_engine *test_open_engine(bool quality);
ds4_engine *test_get_engine(bool quality);
void test_close_engines(void);
void test_close_engine(bool quality);
uint64_t test_round_up_u64(uint64_t n, uint64_t align);
uint16_t test_float_to_f16(float f);
float test_f16_to_f32(uint16_t h);

/* ---- t_metal_kernels.c: f16/q8_0 matvec+prefill 数值回归 ---- */
void test_fill_q8_0_weights(uint8_t *weights, uint32_t in_dim, uint32_t out_dim);
void test_metal_f16_matvec_fast_nr0_4(void);
void test_metal_f16_prefill_matmul(void);
void test_metal_q8_0_prefill_matmul(void);
void test_metal_q8_0_prefill_matmul_unaligned(void);

/* ---- t_metal_kernels_moe.c: rowslice TP + go1b/go2b routed-MoE ---- */
void test_metal_q8_0_rowslice_tp(void);
void test_metal_go1b_routed_moe(void);
void test_metal_go2b_routed_moe(void);

/* ---- t_metal_kernels_group.c: corr_apply(文件内 static) + 组入口 + 短 prefill ---- */
void test_metal_kernel_group(void);
void test_metal_short_prefill_ratio4(void);

/* ---- t_metal_v41_*.c: DeepSeek V4.1 Metal 原语数值回归(2026-10-08; 合成数据, 不要模型) ---- */
void test_v41_seed(uint64_t s);
uint32_t test_v41_rand_u32(void);
float test_v41_randf(void);
float test_v41_bf16r(float x);
int test_v41_model_begin(uint64_t bytes);
uint64_t test_v41_model_alloc(uint64_t bytes);
uint8_t *test_v41_model_ptr(uint64_t off);
const void *test_v41_model_map(void);
uint64_t test_v41_model_size(void);
int test_v41_map(void);
uint64_t test_v41_wbytes(uint32_t wt, uint64_t rows, uint64_t cols);
void test_v41_make_weights(uint32_t wt, uint64_t rows, uint64_t cols, uint8_t *dst, float *ref);
void test_v41_fill_bf16(float *x, uint64_t n, float scale);
ds4_gpu_tensor *test_v41_tensor(const void *src, uint64_t bytes);
int test_v41_read(const ds4_gpu_tensor *t, void *dst, uint64_t bytes);
int test_v41_cmp(const char *name, const float *got, const float *ref, uint64_t n, float tol, float floor_);
void test_v41_ref_matmul(const float *x, const float *w, float *out, uint32_t M, uint32_t N, uint32_t K, int round_out);
void test_metal_v41_dense(void);
void test_metal_v41_layer(void);
void test_metal_v41_attn(void);
void test_metal_v41_vq(void);
void test_metal_v41_bwd(void);
void test_metal_v41_bwd_attn(void);
void test_metal_v41_group(void);

/* ---- t_long_context.c(hex/token 字节助手被 t_vectors.c 复用) ---- */
char *test_read_file(const char *path);
bool test_hex_to_bytes(const char *hex, unsigned char *out, int cap, int *len);
bool test_token_bytes_equal(ds4_engine *engine, int token,
                            const unsigned char *want, int want_len);
void test_long_story_fact_recall(void);

/* ---- t_vectors.c: 官方向量夹具的类型与解析器(t_mpp.c 的 case 加载器复用) ---- */
#define TEST_VEC_MAX_STEPS 16
#define TEST_VEC_MAX_TOP 32
#define TEST_VEC_MAX_TOKEN_BYTES 128

typedef struct {
    unsigned char bytes[TEST_VEC_MAX_TOKEN_BYTES];
    int len;
    float logprob;
} test_vec_top;

typedef struct {
    unsigned char selected[TEST_VEC_MAX_TOKEN_BYTES];
    int selected_len;
    int ntop;
    test_vec_top top[TEST_VEC_MAX_TOP];
} test_vec_step;

typedef struct {
    char id[96];
    char prompt_path[512];
    int ctx;
    int nsteps;
    test_vec_step steps[TEST_VEC_MAX_STEPS];
} test_vec_case;

char *test_trim_line(char *line);
bool test_read_vector_case(FILE *fp, test_vec_case *vc);
bool test_fill_vector_case(FILE *fp, test_vec_case *vc);
void test_official_logprob_vectors(void);
void test_local_golden_vectors(void);

/* ---- t_mpp.c(top-k 助手被 t_vectors.c 复用) ---- */
void test_logits_topk(const float *logits, int n, int *out, int k);
bool test_topk_contains(const int *top, int k, int id);
void test_metal_mpp_equivalence(void);

/* ---- t_quality.c: 工具调用质量(要真模型) ---- */
void test_tool_call_quality(void);
#endif /* !DS4_NO_GPU */

/* ---- t_quality.c: 无 GPU 依赖段 ---- */
void test_server_unit_group(void);
void test_tp_allreduce(void);

/* ---- t_units.c: 无模型离线单测(采样/惩罚 API + rax 基数树) ---- */
void test_engine_units(void);
void test_rax_units(void);

#endif /* DS4_TEST_INTERNAL_H */
