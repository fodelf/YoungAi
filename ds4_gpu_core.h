#ifndef DS4_GPU_CORE_H
#define DS4_GPU_CORE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the DS4-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
typedef struct ds4_gpu_tensor ds4_gpu_tensor;

int ds4_gpu_init(void);
void ds4_gpu_cleanup(void);

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);
int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count);
/* 压缩缓存(attn comp)行格式 —— 两后端唯一的存储格式(09-07): [448 维 FP8 段 f16][64 维 RoPE 段 f32] = 1152 B/行。
 * 为什么: FP8 段是 e4m3 × 2^k(fp8_kv_quantize 的 scale 恒为 2 的幂), f16 精确可表示; RoPE 段是 f32, 转 f16 丢位置精度
 * (纯 f16 缓存实测 vs FP 锚 KL +0.9%), 所以保 f32。相对纯 f32 省 1.78×, 相对纯 f16 只多 12.5%, 数值与 f32 缓存逐位同。
 * head_dim/n_rot 钉死 512/64(core 分配时校验模型形状, 不符拒绝启动)。 */
#define DS4_GPU_COMP_ROW_NOPE  448u
#define DS4_GPU_COMP_ROW_ROT   64u
#define DS4_GPU_COMP_ROW_BYTES (DS4_GPU_COMP_ROW_NOPE * 2u + DS4_GPU_COMP_ROW_ROT * 4u)
/* f32 [rows][512] 行(暂存) → 压缩缓存行格式, 写到 dst 的 first_row 行起 */
int ds4_gpu_comp_rows_commit(ds4_gpu_tensor *dst, uint64_t first_row, const ds4_gpu_tensor *src, uint64_t rows);

int ds4_gpu_token_graph_begin(void);
void ds4_gpu_token_graph_set_pos(uint32_t pos);
int ds4_gpu_token_graph_end_launch(void);
/* decode 流水线: 命中预编码图直接发射 / GPU 忙时为 pos 预捕获下一图(CUDA-only) */
int ds4_gpu_token_graph_try_pending(int token, uint32_t pos, int need_logits);
/* 丢弃预捕获待发射的图(09-07): 下一步不是 eval(pos+1)(投机 verify 批插队/回退)时由核心侧连同主机计数器一起作废 */
int ds4_gpu_token_graph_pending_discard(void);
/* ===== 预发射(2026-09-07, CUDA): 贪心解码把下一图在本图跑完之前排进流 =====
 * 每个 token 图末尾 decode_argmax 把 logits 的 argmax 写进设备 token 槽; 图开头从槽取 token
 * (主机槽/设备槽二选一, 由每相位一份的 pinned 参数决定)。prelaunch(pos) 把已预捕获的 pos 图按
 * "取设备槽"发射, 主机不必等本图 logits 回传再采样再发射 ⇒ token 边界 GPU 零空转。
 * readback_async 把本图 logits + 设备槽 token 异步回传到 pinned 内存(排在本图之后、下一图之前);
 * readback_wait 等它落地。capable=0 的后端(Metal)走原同步路, core 不会调其余入口。 */
int ds4_gpu_decode_prelaunch_capable(void);
int ds4_gpu_decode_argmax_tensor(const ds4_gpu_tensor *logits, uint32_t n_vocab, int exclude_id);
int ds4_gpu_token_graph_prelaunch(uint32_t pos, int need_logits);
int ds4_gpu_token_graph_prelaunch_claim(uint32_t pos);   /* eval(pos) 开头认领已预发射的 pos, 认领即清 */
int ds4_gpu_decode_readback_async(const ds4_gpu_tensor *logits, uint64_t bytes,
                                  float *pinned_logits, int32_t *pinned_next_tok);
int ds4_gpu_decode_readback_wait(void);
void *ds4_gpu_host_alloc(uint64_t bytes);   /* pinned(CUDA)/普通 malloc(Metal) 主机内存 */
void ds4_gpu_host_free(void *p);
/* decode 双流并发: mark(主流,MoE 前) → begin(shared 段切侧流) → join(汇合) */
int ds4_gpu_side_mark(void);
int ds4_gpu_side_begin(void);
int ds4_gpu_side_main(void);
int ds4_gpu_dspark_hc_mean_tensor(ds4_gpu_tensor *dst, const ds4_gpu_tensor *hc,
                                  uint32_t n_embd, uint32_t n_hc, uint32_t slot, uint32_t n_tokens);
int ds4_gpu_dspark_attn_tensor(ds4_gpu_tensor *heads,
                               const void *model_map, uint64_t model_size, uint64_t sinks_offset,
                               const ds4_gpu_tensor *q, const ds4_gpu_tensor *win_kv,
                               const ds4_gpu_tensor *blk_kv,
                               uint32_t n_win, uint32_t blk, uint32_t n_head, uint32_t head_dim,
                               uint32_t win_base, uint32_t win_cap);
int ds4_gpu_dspark_win_scatter_tensor(ds4_gpu_tensor *win, const ds4_gpu_tensor *rows,
                                      uint32_t n, uint32_t pos0, uint32_t win_rows, uint32_t dim);
int ds4_gpu_dspark_argmax_only_tensor(ds4_gpu_tensor *out_id,
                                      const ds4_gpu_tensor *logits_row, uint32_t vocab);
/* DSpark 置信头: c_k = sigmoid(w·[x_k ; W1[prev_k]]) —— 调度器按前缀存活率 ∏c 选验证长度 */
int ds4_gpu_dspark_confidence_tensor(ds4_gpu_tensor *out_conf, const ds4_gpu_tensor *x,
                                     const void *model_map, uint64_t model_size,
                                     uint64_t conf_w_offset, uint64_t markov_w1_offset,
                                     const ds4_gpu_tensor *prev_ids,
                                     uint32_t dim, uint32_t rank, uint32_t vocab, uint32_t n_pos);

int ds4_gpu_dspark_markov_step_tensor(ds4_gpu_tensor *out_id, ds4_gpu_tensor *logits_row,
                                      const void *model_map, uint64_t model_size,
                                      uint64_t w1_offset, uint64_t w2_offset,
                                      const ds4_gpu_tensor *prev_id, uint32_t vocab, uint32_t rank);
int ds4_gpu_side_join(void);
int ds4_gpu_matmul_q4_K_pair_tensor(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
                                    const void *model_map, uint64_t model_size,
                                    uint64_t off0, uint64_t off1,
                                    uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
                                    const ds4_gpu_tensor *x);
int ds4_gpu_matmul_q2_K_pair_tensor(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
                                    const void *model_map, uint64_t model_size,
                                    uint64_t off0, uint64_t off1,
                                    uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
                                    const ds4_gpu_tensor *x);
int ds4_gpu_token_graph_precapture_begin(void);
int ds4_gpu_token_graph_precapture_end(uint32_t pos, int need_logits, int encode_ok);
int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_commands(void);
int ds4_gpu_end_commands(void);
int ds4_gpu_synchronize(void);

/* Tensor-parallel rendezvous. ds4_gpu_tp_signal_after_batch encodes a shared-
 * event signal at the tail of the current batch and returns the value to wait on
 * (0 on error); the caller flushes the batch so the GPU runs and fires it.
 * ds4_gpu_tp_host_wait blocks on the fast MTLSharedEvent path until that value,
 * making the just-encoded results host-visible without a full pipeline drain. */
uint64_t ds4_gpu_tp_signal_after_batch(void);
int ds4_gpu_tp_host_wait(uint64_t value);

/* GPU 跨度计时(verify 归因): span_begin/end 之间的 GPU 侧毫秒 */
void ds4_gpu_span_begin(void);
float ds4_gpu_span_end(void);
int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size);
/* 副模型(drafter gguf)整体 map 注册: 不动主模型状态, 只加一条 range */
int ds4_gpu_register_aux_model_map(const void *map, uint64_t size);
/* verify 批 CUDA 图: begin(slot=pos0&3) 开始捕获; end_launch(encode_ok) 收尾并发射。
 * 返回 1=已发射(图), 0=未启用(直发), -1=捕获失败(调用方需恢复状态后重编码直发)。 */
/* 数值护栏: 把张量里的非有限值(NaN/Inf)就地置 0 */
int ds4_gpu_sanitize_finite_tensor(ds4_gpu_tensor *t, uint64_t n_float);
/* 路由空槽消毒: selected 里的 -1/越界项归零并清其权重(避免 counts[-1] 越界与未写 mid 槽) */
int ds4_gpu_sanitize_router_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
                                   uint32_t n_pairs, uint32_t n_total_expert);
int ds4_gpu_batch_graph_begin(int slot);
int ds4_gpu_batch_graph_end_launch(int encode_ok);
/* 批版 pair 融合(同输入两矩阵一发): 激活量化一次 + 行数合并解并行度 + 权重驻留。
 * 返回 0 = 不适用(调用方回退两次单发)。 */
int ds4_gpu_matmul_q2_K_pair_batch_tensor(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
                                          const void *model_map, uint64_t model_size,
                                          uint64_t off0, uint64_t off1,
                                          uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
                                          const ds4_gpu_tensor *x, uint64_t n_tok);
int ds4_gpu_set_model_fd(int fd);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes);
/* Reduced-memory model loader. Identical to ds4_gpu_set_model_map_spans but each
 * span carries a resident flag: resident spans (backbone) are wired into the GPU
 * residency set; non-resident spans (routed experts) are still wrapped so the hot
 * path can resolve their buffers, but are kept out of the residency set so their
 * clean file-backed pages stay reclaimable under memory pressure. resident_flags
 * must be non-NULL with one entry per span. Used for routed-expert offload. */
int ds4_gpu_set_model_map_spans_split(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, const bool *resident_flags, uint32_t count, uint64_t max_tensor_bytes);
/* Dynamic routed-expert residency route. The host decides at load time whether
 * the fully-resident model fits the memory budget and pushes the verdict here so
 * the hot-path gather agrees (1 = stream/offload, 0 = keep resident).
 * CPU builds ignore it. */
void ds4_gpu_set_expert_offload(int enabled);
/* Survival flag for oversized single-host models (--no-residency): skip
 * MTLResidencySet wiring and view warmup so wired memory cannot balloon into
 * a kernel panic (2026-07-06 on record). Set before the model map. CUDA no-op. */
void ds4_gpu_set_no_residency(int on);
/* Strict IEEE-754 shader math (safe math mode + f32 raw KV + exp2/log2 RoPE)
 * for cross-GPU parity lanes. Must be set before ds4_gpu_init compiles the
 * shader library. Default off. CUDA no-op. */
void ds4_gpu_set_strict_fp(int on);
/* Metal 4 tensor API gate: 1 (default) = auto-probe per hardware generation;
 * 0 = force the legacy kernels (deterministic logprob-vector runs). Must be
 * set before ds4_gpu_init. CUDA no-op. */
void ds4_gpu_set_metal4_enabled(int on);
/* Resident routed-expert LRU pool (--expert-pool-*): mb==0 disables (default).
 * pinned_spec is the "L20:1,2;L21:7" whitelist text or NULL. CUDA no-op. */
void ds4_gpu_set_expert_pool(uint64_t mb, const char *pinned_spec, uint32_t auto_pin_top, uint32_t prefetch_top);
/* Frequency-pinned expert mlock cache (--expert-pin-*): file==NULL or
 * mlock_mb==0 disables; resid_mlock_mb wires the residual sidecar's slots of
 * the same pin set (0 = off). CUDA no-op. */
void ds4_gpu_set_expert_pin(const char *file, uint64_t mlock_mb, uint64_t resid_mlock_mb);
/* Dual-host expert fetch, client half: host!=NULL dials host:port (port 0 =
 * 5606); accept_port!=0 listens instead (reverse-established transport for
 * peers whose outbound connect is broken). CUDA no-op. */
void ds4_gpu_set_expert_fetch_client(const char *host, int port, int accept_port);
/* Predicted-expert peer staging (--expert-stage). Self-disables unless the
 * expert-fetch client is configured. CUDA no-op. */
void ds4_gpu_set_expert_stage(int on);
/* Device recommended max GPU working-set in bytes (0 if unknown / CPU build).
 * Used as the AUTO offload budget when DS4_MEM_BUDGET_MB is unset. */
uint64_t ds4_gpu_recommended_max_working_set_bytes(void);
/* Live GPU working-set (model wired + scratch). 0 on CPU build. */
uint64_t ds4_gpu_current_allocated_bytes(void);
/* P2.1 cross-layer router prediction prefetch (project.md): register one routed
 * MoE layer's router metadata so the backend can re-evaluate the next layer's
 * router on the CPU during decode and issue async read-ahead for the predicted
 * experts while the GPU is still computing the current layer.  gate_inp must be
 * the F16 [n_embd][n_expert] router matrix; probs_bias_offset is UINT64_MAX
 * when the model has no exp_probs_b bias.  Purely advisory: wrong predictions
 * waste read bandwidth but can never change inference results. */
/* hash_table_offset/k/rows describe the early layers' token-id hash routing
 * (ffn_gate_tid2eid I32 [k][n_vocab]): their expert set is an exact function
 * of the token id, so the backend can stage them with 100% accuracy as soon
 * as the token is known.  hash_table_offset == UINT64_MAX for score routing. */
int ds4_gpu_register_layer_router(const void *model_map, uint32_t layer, uint64_t gate_inp_offset, int gate_inp_is_f32, uint64_t probs_bias_offset, uint64_t gate_exps_offset, uint64_t up_exps_offset, uint64_t down_exps_offset, uint64_t gate_expert_bytes, uint64_t down_expert_bytes, uint32_t n_embd, uint32_t n_expert, uint64_t hash_table_offset, uint32_t hash_k, uint32_t hash_rows);
int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
/* 已装进设备副本的总字节。超预算的段是静默走主机映射的(每步 5~25 ms 长尾, 见 single.md S1),
 * 所以起跑时必须拿它跟"请求装的字节"对账并打印。CUDA 以外的后端返回 0。 */
uint64_t ds4_gpu_model_cache_bytes(void);
int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label);
/* CUDA decode: 启动期把 q8_0 权重 repack 成 scale/qs 分离平面(对齐 128bit 读)。
 * token graph capture 时 host dispatch 只跑一次, repack 表必须先建好。 */
int ds4_gpu_q8r_preload(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t in_dim, uint64_t out_dim);
int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
void ds4_gpu_set_quality(bool quality);
void ds4_gpu_print_memory_report(const char *label);


#ifdef __cplusplus
}
#endif

#endif
