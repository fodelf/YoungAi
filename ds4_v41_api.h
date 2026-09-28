/* ds4_v41_api.h — DeepSeek V4.1 公共出口(2026-09-12 战役 P2c)。由 ds4.h 包含, 单独拆出只为守 ds4.h 的 500 行线。
 *
 * V4.1 的会话/采样/服务接入还没做(P5), 现阶段两条路: ①--score-ids 分块增量前向出全位置 logits(对拍尺);
 * ②贪心生成(prefill 分块 + 逐 token 解码同一条前向)。no_engram / chunk 是对拍夹具(chunk 0 = 默认 512)。 */
#ifndef DS4_V41_API_H
#define DS4_V41_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* ds4_engine 的前置 typedef 来自 ds4.h(C99 不许重复 typedef) */
typedef int (*ds4_v41_emit_fn)(int token, void *ud);   /* 返回非 0 = 停止生成 */

int ds4_engine_is_v41(ds4_engine *e);
void ds4_engine_v41_set_prof(int on);   /* --v41-prof: 每次前向打逐层毫秒(每层同步一次, 只在查速度时开) */
void ds4_engine_v41_set_decoder_full(int on);   /* --decoder-full: 关 CED, 提示每块跑满全部层(精确路) */
void ds4_engine_v41_set_chunk(int n);           /* --v41-chunk: 预填分块大小(0 = 默认) */
void ds4_engine_v41_set_dspark(int mode);       /* 投机: 0 关 / 1 默认开(采样请求自动走纯解码) / 2 显式 --dspark(与采样同开硬拒); 见 core_v41_api.c */
void ds4_engine_v41_set_graph(int on);          /* --no-graph 关解码整步 CUDA graph(默认开; 只作 A/B 与定位, 输出逐字节同) */
void ds4_engine_v41_set_vq_group(int on);       /* --no-vq-group: 验证批/草稿塔的 VQ 专家核回逐对形态(默认分组核; 只作 A/B, 输出逐字节同) */
void ds4_engine_v41_set_emit_trace(int on);     /* --emit-trace: 逐 token 打 [emit] 位置+id(同轨定位) */
void ds4_engine_v41_set_block(unsigned b);      /* --dspark-block N: 钉死草稿块长(0=按元数据); 只作诊断 */
void ds4_engine_v41_set_verify_k(unsigned k);   /* --dspark-verify N: 每轮验证几位(0=默认 3); 诊断与逐档量字节账用 */
void ds4_engine_v41_set_amp_dir(const char *dir);   /* --zchain <dir>: V4.1 反修放大器目录(amp_Lnn.bin), 每层 MoE 出口 y += x·(B·A) */
void ds4_engine_v41_set_amp_scale(float s);         /* --zchain-scale β: 加载时把 A 乘 β(修正整体缩到 β 倍); ≤0 = 1.0 */
void ds4_engine_v41_set_posttrain_dir(const char *dir);   /* --posttrain <dir>: 三文件部署的第三件(后训练增益), 与 --zchain 的表逐元素相乘 */
void ds4_engine_v41_set_engram_dir(const char *dir);      /* --engram-dir <dir>: n-gram 表分片所在目录, 顶替 GGUF 里记的转换机绝对路径 */
/* --score-nll FILE / --score-topk K FILE / --score-no-logits: --score-ids 的三个小出口。
 * 与 V4 的 --eval-nll/--eval-topk 同一份实现(core_score_aux.c)、同一种字节。
 * 后训练一趟 5.8 万行, 全词表 logits 就是 30 GB —— 统一内存机器上写它 = 掏 GPU 内存(09-08 崩机)。 */
void ds4_engine_v41_set_score_aux(const char *nll_path, const char *topk_path, int topk,
                                  const char *rms_path, int skip_logits);
/* --score-ids 部署同路切分点 P(0 = 老口径): [0,P) 照生成路 CED 预填, [P,n) 跑满解码器。后训练取料用, 见 core_v41_api.c。 */
void ds4_engine_v41_set_score_split(int p);
int ds4_engine_v41_score_ids(ds4_engine *e, const int *ids, int n_ids, const char *out_path, int no_engram, int chunk);

/* --dspark-capture(mtp.md M6): 教师强制一位一块走完 ids, 每位出 (草稿器出口隐态, 主模型出口隐态) 一对,
 * 并直接量出"草稿器首位 ↔ 底座 argmax"的一致率(= 投机首位接受率 p1 的判决基线)。实现见 core_v41_dcap.c。 */
int ds4_engine_v41_dspark_capture(ds4_engine *e, const int *ids, int n_ids, const char *out_path);

/* --draft-amp <file>: 挂草稿器对齐边车(gguf-tools/amp/dspark_align 的产物)。只改草稿器的出口隐态,
 * 主模型一个字节不碰 —— 所以它**不可能**动五指标, 只动接受率。传 NULL/不传 = 不挂, 整条路恒等。 */
void ds4_engine_v41_set_draft_amp(const char *path);
void ds4_engine_v41_set_draft_amp_scale(float s);   /* --draft-amp-scale β: 诊断修正幅度 */
/* 没有 ctx 参数: V4.1 的上下文来自模型元数据(ds4_engine_v41_ctx), 调用方定不了边界, 只定生成上限 n_predict
 * (INT_MAX = 不设上限, 生成到 EOS 或上下文边界)。 */
int ds4_engine_v41_generate_argmax(ds4_engine *e, const int *prompt, int n_prompt, int n_predict,
                                   ds4_v41_emit_fn emit, void *ud);
/* 解码采样(2026-09-21, 113-1.md §4): 上面那条生成路的名字里的 argmax 是历史, 采样由这个设置面决定。
 * temperature ≤ 0(默认) = 裸 argmax: 图末尾的设备 argmax, 一个字节不变(门 = 温 0 输出逐字节回归)。
 * > 0 = 每步把末位 logits 行读回主机, 交给 V4 路**同一份**采样器 ds4_sample_logits(温度 / top-k / top-p / min-p / seed)
 * ⇒ 两条路同分布。只按模型自己的分布抽, 不动 logits(铁律"引擎不得改模型输出"); 频率/出现/序列复读惩罚还没接。
 * seed 0 = 按时钟(与 V4 CLI 同规则)。NULL = 回到裸 argmax。采样与显式 --dspark 不能同开: 生成路直接拒; 投机只是默认开着时, 采样请求走纯解码并打日志。
 * 惩罚(core_decode_penalty.c, 全部默认 0 = 不进那段代码): freq/presence = OpenAI 频率/出现惩罚(与 V4 路同式);
 * dry_multiplier > 0 开 DRY 序列复读惩罚(base 1.75 / allowed_length 2 是 llama.cpp 默认): 这是温 0 下也能挡死循环的唯一手段。
 * 任一惩罚非零时, 温 0 也走"读回 logits 行 → 罚 → argmax"这条路(argmax 由同一份采样器在温 0 时给出)。 */
typedef struct {
    float temperature, top_p, min_p; int top_k; uint64_t seed;
    float freq_penalty, presence_penalty;
    float dry_multiplier, dry_base; int dry_allowed_length;
} ds4_decode_sampling;
void ds4_engine_set_decode_sampling(const ds4_decode_sampling *sp);
/* 服务接入(2026-09-19, src/server/server_generate_v41.c): 预填每跑完一块回调一次("prefill_chunk", 已预填 token 数, 提示总数),
 * 与 V4 会话的 ds4_session_set_progress 同一种回调形状 —— 服务端拿它发 SSE 心跳与预填进度日志。NULL = 不回调。
 * ★返回非 0 = 调用方要求中止这趟前向★(2026-09-22): 13 万 token 的提示光预填就要 400 秒, 客户端在这期间
 * 挂断的话这 400 秒是纯浪费, 而本服务串行跑图 —— 后面排队的请求跟着一起超时。服务端在这里探对端还在不在。
 * ds4_engine_v41_ctx: V4.1 上下文 = 模型元数据 deepseek4.context_length(转换器从 HF max_position_embeddings 写入; 开模型之前
 * 是 0)。用户 2026-09-22 定"不要任何写死的上下文": 引擎里没有常量, 服务端 /v1/models 报的就是它, 每条请求的 max_tokens 按它钳;
 * CLI/服务端都不接受 --ctx。 */
typedef int (*ds4_v41_progress_fn)(void *ud, const char *event, int current, int total);
void ds4_engine_v41_set_progress(ds4_v41_progress_fn fn, void *ud);
int ds4_engine_v41_ctx(void);

/* 反修取料钩子(2026-09-13, C 反修驱动 gguf-tools/amp/v41_amp_run 用): 每层 MoE 出口、放大器应用前回调一次。
 * 全是主机内存、行主序: x[n][D] = MoE 输入(ffn_norm 出口, bf16 格点), y[n][D] = MoE 输出(bf16 格点, 还没加放大器),
 * sel[n][n_used] / rw[n][n_used] = 路由选中的专家与权重(路由 gate 不量化 ⇒ FP 靶那一遍可直接复用), pos0 = 本块起始位置,
 * clamp = SwiGLU 截断。alpha[n] = 本层 y 经 hc_post 进 hc、再经 hc_pre(用本层 ffn_pre 当 pre_mix)合成出来的系数
 * (= Σ_k pre[i][k]·post[i][k], 逐 token 标量)。★只有末层的 alpha 等于"y 到出口 hc_pre 的精确系数"★ —— 末层之后
 * 直接就是 norm→head, 中间层的 y 还要再穿过后面所有层, 那个系数只是一阶直通项。蒸馏靶(v41_kl_target)靠它把
 * 出口梯度折回本层输出。返回 0 继续; 1 = 取完了, 本次前向到此为止(跳过余下层与出口, 位置照常推进, 本块不出 logits);
 * <0 = 停车。序贯解算靠多遍: 第 k 遍挂着目录里已解的 L0..L(k−1)(--zchain 同一条加载/应用路)按块正常前向, 到第 k 层取
 * 料即停, 收齐全部行后离线解第 k 层 —— 与"一次前向里逐层解+挂"数学等价, 但不要求整批(8192 整批的缓冲把 avail 压到 6 GB)。
 * 为什么是主机内存而不是设备指针: 钩子方(解算器)与引擎各管各的显存与流, 一层 2×168 MB 拷贝零点几秒, 换来后端无关。
 *
 * ye[n][n_used][D](2026-09-13, 权重侧逐专家反修): 逐专家 down 输出, 【未乘路由权重】—— 引擎这一层算的
 * y[i][d] 就等于 Σ_k rw[i][k]·ye[i][k][d]。解逐专家逐通道增益 g 要它(y 对 g 逐输出通道独立线性)。
 * ★只有 prefill GEMM 路(块内 n>8)物化这个中间量★; 解码 gemv 路不物化, 那里 ye = NULL。取料走
 * --score-ids(chunk 512)本来就是 prefill 路, 与判决同路。一层一块 63 MB(n=512), 用不着就别读。
 * ysh[n][D]: 同一层 shared 专家的输出(bf16 格点, 未进 y 的加法)。引擎这一层出的
 * y = round_bf16(Σ_k rw[i][k]·ye[i][k][d] + ysh[i][d]) —— 逐位。权重侧解 g 只动 routed 那一半,
 * 所以靶必须先把 ysh 减掉; 同时这条恒等式就是取料的自检(对不上 = 配对错位, 只会出假账)。 */
typedef int (*ds4_v41_moe_hook_fn)(void *ud, int il, int pos0, int n, int D, int n_used, float clamp,
                                   const float *x, const float *y, const int *sel, const float *rw, const float *alpha,
                                   const float *ye, const float *ysh);
void ds4_engine_v41_set_moe_hook(ds4_v41_moe_hook_fn fn, void *ud);
/* 只在第 il 层回调(-1 = 每层)。取料的 D2H 发生在回调之前, 只要一层时不设它 = 每块白拷 39 层。 */
void ds4_engine_v41_set_moe_hook_layer(int il);

#ifdef __cplusplus
}
#endif
#endif
