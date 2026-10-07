/* core_v41_api.c — DeepSeek V4.1 对外出口(2026-09-12 战役 P2c): --score-ids 分块全位置 logits + 贪心生成。
 * 两者都走 core_v41_forward.c 的同一条增量前向; 解码 = n=1 的前向。采样/会话/服务接入是 P5。 */
#include "core_internal.h"
#include <time.h>
#include <unistd.h>

int ds4_engine_is_v41(ds4_engine *e) { (void)e; return DS4_MODEL_VARIANT == DS4_VARIANT_V41; }
/* 解码采样设置面(2026-09-21, 113-1.md §4; 语义见 ds4_v41_api.h)。全零 = 裸 argmax = 今天的路。 */
ds4_decode_sampling g_decode_sampling;   /* core_v41_sample.c 的惩罚路也读它 */
void ds4_engine_set_decode_sampling(const ds4_decode_sampling *sp) {
    if (sp) g_decode_sampling = *sp;
    else memset(&g_decode_sampling, 0, sizeof g_decode_sampling);
}
int g_ds4_v41_prof = 0;
/* ★--decoder-full★: 关掉 CED, 提示的每一块都跑满 40 层(精确路, 用来跟 CED 对质量)。
 * 默认是官方部署语义(CED 开), 见 core_v41_forward.c 的 ced_edge 注释。 */
int g_ds4_v41_decoder_full = 0;
void ds4_engine_v41_set_decoder_full(int on) { g_ds4_v41_decoder_full = on; }
/* ★--no-vq-group★: 验证批/草稿塔(n≥2)的 VQ 专家核默认走多 token 分组核(cuda_vq_group.inc.cu, 一 block 一个专家 × m 个 token);
 * 这个开关钉回"一 block 一对"的老形态, 只给同一二进制做 A/B —— 两条路输出逐字节同(乘加式同源), 差的只是时间。 */
int g_ds4_v41_vq_group = 1;
void ds4_engine_v41_set_vq_group(int on) { g_ds4_v41_vq_group = on; }
void ds4_engine_v41_set_prof(int on) { g_ds4_v41_prof = on; }
/* ★--dspark★: 投机解码(09-16 起默认关, 09-24 翻回默认开, 见下)。温 0 下两条路必须**逐字节同** ——
 * 投机的接受条件就是"主模型自己也会选这个 token", 所以它只省时间不改输出; 不同就是回滚漏了东西。
 *
 * ★为什么默认是关的(2026-09-16, 用户令"速度大幅度提升才改默认开启")★: 两条理由, 缺一条都不该关。
 * ①**同轨门今天不绿**: 同一份 2K 提示、温 0, 投机与纯解码的输出第二句就分叉(mtp-1.md §4)。
 *   引擎默认路径必须是裸模型的真值 —— 默认开着一个会改输出的近似路, 等于每个跑分都掺了别的东西。
 * ②**它今天还是亏的**: 一轮 188 ms 只产 1.96 个 token(96 ms/token), 而纯解码一步 52 ms。
 * 等 mtp-1.md 的 M1′(同轨)与 M2′/M3′/M4′(核形态)过门、且真比纯解码快之后, 再把默认翻回来。
 * `--no-dspark` 保留: 老脚本一路在传它, 现在是"再确认一次关", 不是错。 */
/* ★09-24 默认翻成开(用户令)★: 上面两条都已满足 —— ①真实 CFO 请求(14k 提示, 温 0, 2048 token)投机与纯解码逐字节同,
 * 每一刀都验; ②投机 39.27 t/s 对纯解码 28.5(fable5 09-24 两节)。
 * 三档: 0 = 关(--no-dspark); 1 = 默认开 —— 请求开了采样/惩罚时这一条自动走纯解码并打一行日志(服务端温 1.0 的请求不能因为
 * 默认值被拒); 2 = 显式 --dspark —— 与采样同开仍硬拒(用户两样都点名要, 给不了就得说, 不能悄悄只给一样)。 */
int g_ds4_v41_dspark = 1;
void ds4_engine_v41_set_dspark(int mode) { g_ds4_v41_dspark = mode; }
int ds4_engine_v41_dspark(void) { return g_ds4_v41_dspark; }
/* 单请求路上一趟的投机账(服务端监控页报草稿接受率用): generate 进门清零, 投机段收尾时写 */
static int g_v41_last_spec_rounds, g_v41_last_spec_off, g_v41_last_spec_acc;
void ds4_engine_v41_last_spec_stats(int *rounds, int *offered, int *accepted) {
    if (rounds) *rounds = g_v41_last_spec_rounds;
    if (offered) *offered = g_v41_last_spec_off;
    if (accepted) *accepted = g_v41_last_spec_acc;
}
/* --dspark-verify N: 每轮验证几位(0 = 用下面钉死的默认)。
 * 为什么要这个旋钮: ①同轨出问题时, k=1(验证批只有 2 行)是最小的多 token 批, 拿它跟 k=3 一比就知道
 * "病在批本身"还是"病在批大了以后"; ②字节账(mtp-1.md §3)要按 k 逐档量, 每档一个二进制是浪费。
 * 它不是兜底开关 —— 调度器(M5′)落地后由置信度定 k, 这个旋钮只作诊断与逐档量尺用。 */
/* --emit-trace: 逐 token 打 `[emit] <绝对位置> <token id>`(同轨定位, mtp-1.md M0′(d))。
 * ★为什么不搭 --v41-prof 的顺风车★: prof 会让前向**逐层 flush**(core_v41_forward.c), 时序一变,
 * 偶发的分叉可能就复现不出来了 —— 那样这把尺就在骗人。它只打这一行, 不改任何执行路径。 */
int g_ds4_v41_emit_trace = 0;
void ds4_engine_v41_set_emit_trace(int on) { g_ds4_v41_emit_trace = on; }
/* --dspark-block N: 把草稿块长钉成 N(0 = 用模型元数据里的 5)。**只作诊断**。
 * 为什么要它: 块长 5 + 4 个 noise 位 + 块内全可见, 是我们照官方 forward_embed/get_dspark_topk_idxs
 * 自己写的一段, 从来没有单独验过。拿 N=1 跑同一把取料尺(首位一致率)一比就知道那几位 noise 是在
 * 帮忙还是在捣乱 —— 训练时是带着它们训的, 所以 N=1 明显更准 = 我们这段写错了。
 * ★不是速度旋钮★: 块长是模型定的, 平时别动。 */
uint32_t g_ds4_v41_block = 0;
void ds4_engine_v41_set_block(uint32_t b) { g_ds4_v41_block = b; }
uint32_t g_ds4_v41_verify_k = 0;
void ds4_engine_v41_set_verify_k(uint32_t k) { g_ds4_v41_verify_k = k; }
const char *g_ds4_v41_draft_amp = NULL;
void ds4_engine_v41_set_draft_amp(const char *path) { g_ds4_v41_draft_amp = path; }
/* --draft-amp-scale β: 装载时 A *= β。用来分辨"方向错"还是"幅度过头" ——
 * β 小了就好转 = 幅度问题(过拟合/欠定); β 怎么调都不如不挂 = 目标函数选错了(L2 隐态 ≠ argmax)。 */
float g_ds4_v41_draft_amp_scale = 1.0f;
void ds4_engine_v41_set_draft_amp_scale(float s) { g_ds4_v41_draft_amp_scale = s; }
const char *g_ds4_v41_amp_dir = NULL;
#ifndef DS4_NO_GPU
const ds4_model *g_ds4_v41_model = NULL;
#endif
void ds4_engine_v41_set_amp_dir(const char *dir) { g_ds4_v41_amp_dir = (dir && dir[0]) ? dir : NULL; }
const char *g_ds4_v41_pt_dir = NULL;
/* 三文件部署第三件: 单独挂 ③ 也允许(用户要求"独立使用量化文件也可以"的对称面)。"单挂"的提示不在这里打:
 * 命令行解析时 ② 还没设(--zchain 要到打开引擎才转进 g_ds4_v41_amp_dir), 在这里查永远报"没挂反修件"(10-02 实撞,
 * ②③ 都挂上了还报)。改在插件真正装载时查, 见 core_v41_state.c v41_state_plugins。 */
void ds4_engine_v41_set_posttrain_dir(const char *dir) { g_ds4_v41_pt_dir = (dir && dir[0]) ? dir : NULL; }
float g_ds4_v41_amp_scale = 1.0f;
/* β ≤ 0 当"没传"处理(1.0 = 原样)。负 β 是把修正反向注入, 没有任何用途, 不留这条路。 */
void ds4_engine_v41_set_amp_scale(float s) { g_ds4_v41_amp_scale = s > 0.f ? s : 1.0f; }
ds4_v41_moe_hook_fn g_ds4_v41_hook = NULL;
void *g_ds4_v41_hook_ud = NULL;
void ds4_engine_v41_set_moe_hook(ds4_v41_moe_hook_fn fn, void *ud) { g_ds4_v41_hook = fn; g_ds4_v41_hook_ud = ud; }
/* 只在第 il 层回调(-1 = 每层都回调)。★不设这个就是每层白拷一遍★: 取料的 D2H(x/y/sel/rw/hc,
 * 外加 prefill 路的逐专家输出 63 MB/块)发生在回调【之前】, 而解算方通常只要一层 —— 40 层里
 * 39 层的拷贝全是白做的。实测 512 token 一块从 33 秒降到个位数。 */
int g_ds4_v41_hook_layer = -1;
void ds4_engine_v41_set_moe_hook_layer(int il) { g_ds4_v41_hook_layer = il; }
/* 服务接入的预填进度回调(2026-09-19): 服务端一条 15k token 的提示要预填两分钟, 流式客户端没有心跳就断线。
 * 只在生成路的预填块间回调, --score-ids 那条不回(它没有客户端在等)。 */
static ds4_v41_progress_fn g_v41_progress = NULL;
static void *g_v41_progress_ud = NULL;
void ds4_engine_v41_set_progress(ds4_v41_progress_fn fn, void *ud) { g_v41_progress = fn; g_v41_progress_ud = ud; }
int ds4_engine_v41_ctx(void) { return (int)g_ds4_v41.ctx; }   /* 模型元数据 deepseek4.context_length; 开模型之前是 0 */


#ifndef DS4_NO_GPU
/* --v41-chunk 也管生成路的预填分块(以前只管 --score-ids): 专家路每层要把 384 个专家全解一遍, 代价与块里有几个 token 无关, 块小就摊不开(实测账在 core_v41.h) */
int g_ds4_v41_chunk = 0;
void ds4_engine_v41_set_chunk(int n) { g_ds4_v41_chunk = n > 0 ? n : 0; }

/* 取下一个 token 的三条路(设备 argmax / 设备采样核 / 主机惩罚路)在 core_v41_sample.c(2026-09-28); 以前"采样 = 读回 517 KB 主机采样"
 * 的那版量过每步只贵 0.32 ms, 真正丢的是投机(采样下关掉, −32%), 所以采样进了设备核, 投机在采样下按拒绝采样走。 */

int ds4_engine_v41_generate_argmax(ds4_engine *e, const int *prompt, int n_prompt, int n_predict,
                                   ds4_v41_emit_fn emit, void *ud) {
    if (!e || !prompt || n_prompt < 1 || !ds4_engine_is_v41(e)) return 1;
    if (!e->metal_ready) { fprintf(stderr, "ds4: V4.1 前向需要 GPU 后端\n"); return 1; }
    g_v41_last_spec_rounds = g_v41_last_spec_off = g_v41_last_spec_acc = 0;   /* 预填就失败的趟也不能把上一趟的账留给监控 */
    const uint32_t np = (uint32_t)n_prompt;
    /* ★上下文只从模型元数据来(g_ds4_v41.ctx ← GGUF deepseek4.context_length)★(用户 2026-09-22): 以前这里收调用方传的
     * ctx_size(CLI --ctx / 服务端 --ctx, 默认 32768), 于是每个入口各配一个数(尺 32768 / 部署 1M / 判决 NTOK), 同一条请求
     * 换个入口就换一条边界。现在引擎里没有任何写死的上下文, 也没有参数能改它; 提示装不下就是装不下, 不"放大"也不"压回"。 */
    uint32_t ctx = g_ds4_v41.ctx;
    if (ctx == 0) { fprintf(stderr, "ds4: 模型元数据没有 deepseek4.context_length\n"); return 1; }
    if (np + 1u > ctx) { fprintf(stderr, "ds4: 提示 %u token 超过上下文 %u\n", np, ctx); return 1; }
    /* ★只按这一趟真正用得到的位置分配★(2026-09-22): 状态里几个大块是 cap_tok × ctx 的
     * (iscore f32 + cand u8 = ctx × 2.5 KB, 再加 kv 源层的 ctx/ratio 格), 1M 是边界, 不是这条请求能用到
     * 的长度。以前照边界分: 连一条 22 token 的请求都要吃 2.5 GiB 打分矩阵, 于是"上下文 1M"被误当成"每条请求都贵"。
     * 这一趟最多走到 np + n_predict 个位置(投机一轮会临时多推 ≤ 块长, 回滚前也要有地方放), 就分这么多。
     * ctx 仍然是硬边界: 生成到 st->ctx 就停(不设上限的请求 = 生成到 1M 边界或 EOS)。 */
    const uint64_t need = (uint64_t)np + (uint64_t)(n_predict > 0 ? n_predict : 0) + DS4_MTP_MAX_BLOCK + 2u;
    if ((uint64_t)ctx > need) ctx = (uint32_t)need;
    /* 调用方可以说"不设上限"(传 INT_MAX, 见 server_types2.h SERVER_NO_OUTPUT_CAP): 那就生成到 ctx 边界。
     * 这里把它钳成真实可用的步数, 后面按步数开的东西(hist)才不会照 INT_MAX 去要 8 GB。 */
    const int room = ctx > np ? (int)(ctx - np) : 0;
    if (n_predict > room) n_predict = room;
    const uint32_t ck = g_ds4_v41_chunk > 0 ? (uint32_t)g_ds4_v41_chunk : DS4_V41_CHUNK;
    const uint32_t cap = ck < np ? ck : np;
    ds4_v41_state st;
    if (!v41_state_alloc(&st, cap, ctx, DS4_MTP_MAX_BLOCK + 2u)) return 1;   /* 生成路: logits 只要解码/验证批那几行 + 预填末位 */
    st.head_last_only = 1;   /* 预填块只算末位 logits(core_v41.h) */
    ds4_gpu_tensor *am = ds4_gpu_tensor_alloc((uint64_t)(DS4_MTP_MAX_BLOCK + 2u) * 16u);   /* 设备槽: 每行 16 B(采样核 4 个 int; argmax 只用第 0 个) */
    const int eos = ds4_token_eos(e);
    int rc = 1;
    /* 三条路(core_v41_sample.c): 温度 > 0 且无惩罚 = 设备采样核(投机照走, 核里做拒绝采样); 任一惩罚非零 = 读回整行主机罚完采样
     * (惩罚看 token 史, 投机不接: 显式 --dspark 硬拒, 默认开则这一条走纯解码并出声 —— 静默改路 = 用户以为开着其实没开); 否则设备 argmax。 */
    const bool penal = g_decode_sampling.dry_multiplier > 0.f || g_decode_sampling.freq_penalty != 0.f || g_decode_sampling.presence_penalty != 0.f;
    const bool dev_sample = g_decode_sampling.temperature > 0.f && !penal;
    int dspark = g_ds4_v41_dspark;
    if (penal && dspark == 2) {
        fprintf(stderr, "ds4: ★复读惩罚(dry %.2f freq %.2f presence %.2f)与 --dspark 投机不能同开★(惩罚要按 token 史改 logits, 投机验证不接; 去掉 --dspark 或关惩罚)\n",
                (double)g_decode_sampling.dry_multiplier, (double)g_decode_sampling.freq_penalty, (double)g_decode_sampling.presence_penalty);
        if (am) ds4_gpu_tensor_free(am);
        v41_state_free(&st);
        return 1;
    }
    if (penal && dspark) {
        fprintf(stderr, "ds4: [v41] 本请求开了复读惩罚(dry %.2f freq %.2f presence %.2f), 投机不接惩罚 ⇒ 这一条走纯解码\n",
                (double)g_decode_sampling.dry_multiplier, (double)g_decode_sampling.freq_penalty, (double)g_decode_sampling.presence_penalty);
        dspark = 0;
    }
    float *rowbuf = penal ? xmalloc((size_t)DS4_N_VOCAB * 4u) : NULL;
    v41_hist hist = {NULL, 0, 0, NULL};
    if (penal) {
        hist.cap = (uint32_t)(n_predict > 0 ? n_predict : 0) + 2u;
        hist.tok = xmalloc((size_t)hist.cap * sizeof(int32_t));
        if (g_decode_sampling.dry_multiplier > 0.f) hist.brk = ds4_decode_breakers(e, DS4_N_VOCAB);
    }
    uint64_t rng = g_decode_sampling.seed ? g_decode_sampling.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());   /* 与 V4 CLI(cli_gen.c)同一条规则 */
    const bool sampling = dev_sample || penal;   /* 日志用: 这条请求不是裸 argmax */
    st.dev_sample = dev_sample ? 1 : 0;
    st.samp = (ds4_gpu_sample_params){ .temperature = g_decode_sampling.temperature, .top_p = g_decode_sampling.top_p,
                                       .min_p = g_decode_sampling.min_p, .top_k = g_decode_sampling.top_k, .seed = rng };
    do {
        if (!am) break;
        /* --emit-trace 连提示也打(`[ptok] 位置 id`): 提示是 build_prompt 套过聊天模板的(比 --dump-tokens 的原文分词多
         * BOS/角色/think 几个 token, 09-20 实撞 79 vs 75), 事后重新分词拼不回引擎真吃的序列; 教师锚/取料要的是这一份。 */
        if (g_ds4_v41_emit_trace) for (uint32_t i = 0; i < np; i++) fprintf(stderr, "[ptok] %u %d\n", i, prompt[i]);
        const double t0 = now_sec();
        bool ok = true, aborted = false;
        for (uint32_t c0 = 0; ok && c0 < np; ) {
            ok = v41_prefill_chunk(e, &st, prompt, np, &c0, cap);   /* 分块 / CED / 窗口尾规则在 core_v41_forward.c, 与并发路同一份 */
            /* 回调非 0 = 调用方不要这趟了(服务端: 客户端已挂断)。立刻停, 别把剩下的块算完 —— 13 万 token
             * 的提示预填 400 秒, 算给一个走掉的连接就是让后面排队的请求跟着超时(2026-09-22 早盘实撞)。 */
            if (ok && g_v41_progress &&
                g_v41_progress(g_v41_progress_ud, "prefill_chunk", (int)c0, (int)np) != 0) {
                fprintf(stderr, "[v41] 预填在 %u/%u 处按调用方要求中止\n", c0, np);
                aborted = true;
                break;
            }
        }
        if (!ok || aborted) break;
        const double t1 = now_sec();
        fprintf(stderr, "[v41] prefill %u token %.1fs (%.1f t/s)\n", np, t1 - t0, (double)np / (t1 - t0 + 1e-9));
        if (sampling) fprintf(stderr, "[v41] 解码采样(%s) temp %.2f top_p %.2f min_p %.2f top_k %d seed %llu%s dry %.2f/%.2f/%d freq %.2f presence %.2f\n",
                              dev_sample ? "设备核" : "主机惩罚路",
                              (double)g_decode_sampling.temperature, (double)g_decode_sampling.top_p, (double)g_decode_sampling.min_p,
                              g_decode_sampling.top_k, (unsigned long long)rng, penal ? " 惩罚" : "", (double)g_decode_sampling.dry_multiplier,
                              (double)g_decode_sampling.dry_base, g_decode_sampling.dry_allowed_length,
                              (double)g_decode_sampling.freq_penalty, (double)g_decode_sampling.presence_penalty);
        /* 末位 logits → argmax(或采样) → 逐 token 解码(n=1 前向) */
        int32_t tok32 = 0;
        if (!v41_next_token(&st, am, st.last_logit_row, rowbuf, &rng, &hist, &tok32)) break;   /* 末位的 logits 在哪一行由出口定(head_last_only ⇒ 第 0 行) */
        if (!rowbuf && !dev_sample) {   /* 设备 argmax 自检(一次): 与主机顺序扫对一下, 不同就报 —— 核错会静默吐错 token */
            float *row = xmalloc((size_t)DS4_N_VOCAB * 4);
            if (ds4_gpu_tensor_read(st.logits, (uint64_t)st.last_logit_row * DS4_N_VOCAB * 4, row, (uint64_t)DS4_N_VOCAB * 4)) {
                uint32_t best = 0; for (uint32_t v = 1; v < DS4_N_VOCAB; v++) if (row[v] > row[best]) best = v;
                if ((int32_t)best != tok32) fprintf(stderr, "ds4: ★V4.1 argmax 核 %d ≠ 主机 %u (logit %.4f vs %.4f)★\n", tok32, best, (double)row[tok32], (double)row[best]);
            }
            free(row);
        }
        int tok = (int)tok32, produced = 0;   /* produced = 交给 emit 的 token 数(含 EOS 那一位, 日志按 hit_eos 减掉); 兼作 n_predict 上限计数 */
        bool stop = false, hit_eos = false;    /* ★停只立标志不改 produced★(09-30 实撞: 投机接受路撞 EOS 曾写 produced=n_predict 跳外圈, 日志报上限 63 而真产出 42, 虚报 48 t/s) */
        /* ---- DSpark 投机解码(speed.md 段 6 D1) ----
         * 没带三塔/没带运行参数的 GGUF: dr.ready=0, 下面整段跳过, 走原来的单 token 环。
         * 一轮 = 草稿器出 block 位 → 主模型一次验证 1+k 位 → 逐位比贪心结果, 接受最长前缀。
         * ★温 0 下这与纯解码逐 token 是同一串输出★: 接受的条件就是"主模型自己也会选这个 token",
         * 不接受的位置全部回滚。所以它是纯粹的省时间, 不是近似 —— 门也就是逐字节同。
         * ★采样下(2026-09-28)★: 接受 ⇔ 均匀数 < 目标分布给草稿的概率, 拒绝从残差抽(设备核里做); 吐出 token 的边缘分布 = 纯解码采样的分布,
         * 但同 seed 下两条路的具体 token 不同(用了不同的硬币) —— 门是分布级(tests/cuda_sample_selftest.c), 不是逐字节。 */
        ds4_v41_draft dr;
        const bool spec = dspark && v41_draft_alloc(e, &dr);
        /* ★采样 + 投机: 草稿按塔的分布抽(硬币流 1), 验证核读塔 logits 做拒绝采样(接受 ⇔ u < min(1,p/q), 拒绝从 max(0,p−q) 抽)★(2026-09-29)
         * 判据不是在线 t/s(采样每趟另一篇文本, 同 seed 也不同, ±3 t/s 噪声比两方案的差大, 曾被它骗成"点质量占优"), 是取料路的接受率陪审团
         * (同一篇文本逐位置解析算, d1 accjury 档): 首位期望接受率 温 1.0 CFO 0.693 → 0.744, 大盘 0.634 → 0.757, 温 0.6 CFO 0.834 → 0.842
         * (点质量 E[p(argmax q)] → 分布草稿 E[Σmin(p,q)])。贪心不碰(温 0 仍 argmax, 逐字节门)。 */
        if (spec && dev_sample) { dr.dev_sample = 1; dr.samp = st.samp; dr.samp.stream = 1u; st.spec_q = dr.st.logits; }
        v41_sched cal; memset(&cal, 0, sizeof cal);   /* 调度器的本请求账: 接受率校准 ρ + 走图一步/草稿/验证 n 行的墙钟(core_draft_sched.c) */
        uint32_t spec_rounds = 0, spec_acc = 0, spec_off = 0, spec_hist[DS4_MTP_MAX_BLOCK + 1];
        /* 一轮的壁钟分账(2026-09-16): 投机赢不赢是个除法 —— 一轮的耗时要压到 E[接受+1] × 纯解码一步
         * 以下。实测一轮 118 ms 对预算 72 ms, 超 64%, 而这 118 从来没拆过。四项分开计, 就能分清
         * "草稿器自己太贵"(那接受率再高也救不回来)还是"验证/回滚的边角料吃掉了"。 */
        double ms_draft = 0, ms_verify = 0, ms_argmax = 0, ms_snap = 0;
        /* 没有"判亏本就歇几轮"的旋钮(2026-09-28 删): 它是个盘上扫出来的常量(09-16 定 16 → 09-19 陪审团 4 → 09-24 陪审团 0, 每次核变快
         * 就得回 dspark_sim 重扫), 而 0 = 机制关. 现在每一步都出草稿, 亏不亏由调度器按当轮 conf 判(k=0 的轮只白跑一次草稿 ~9.5 ms,
         * 那一步照走 n=1 的图). spec_skipped 只计数(日志), 不再让任何步跳过草稿。 */
        uint32_t spec_skipped = 0;
        for (uint32_t i = 0; i <= DS4_MTP_MAX_BLOCK; i++) spec_hist[i] = 0;
        /* 主机空隙分账(2026-10-07, --v41-prof): nsys 量到一轮里 GPU 空等主机 ~2.5 ms —— 草稿完→验证首核 1.2 ms、验证完→草稿首核 1.26 ms。
         * h_vd = 验证 wait 返回 → 草稿图 cudaGraphLaunch 返回(接受/回滚/emit/写槽/发图); h_dv = 草稿 step 返回 → 验证图发出(pick_k/起手/发图)。 */
        double h_vd = 0, h_dv = 0, t_prev_end = 0; uint32_t h_vdn = 0, h_dvn = 0;
        /* ★下一轮草稿在本轮收尾时先发(2026-10-07)★: 发出之后才做 emit/簿记, GPU 不空等主机; 轮顶只等。draft_r = v41_draft_launch 的返回值 */
        bool draft_inflight = false; int draft_r = 0; double t_dl0 = 0;
        while (!stop && produced < n_predict) {
            produced++;
            /* ★同轨定位用(mtp-1.md M0′(d))★: 逐 token 打"绝对位置 + token id"。
             * 为什么不看生成的文字: 文字把 token 边界抹掉了, 两条路差一个 token 可能只差半个词,
             * 而且分叉处往往两句都通顺(近平局翻面), 看不出来。把两条路的这几行 diff 一下,
             * 第一个不同的位置就是要查的那一步 —— 位置对得上、id 不同 = 那一步的 logits 不同(核);
             * 位置本身对不上 = 回滚把状态推歪了(快照漏项)。 */
            if (g_ds4_v41_emit_trace) fprintf(stderr, "[emit] %u %d\n", st.n_past, tok);
            const bool draft_now = spec;   /* 投机开着就每步出草稿(k 由调度器定, 可为 0) */
            /* ★解码整步 CUDA graph(core_decode_graph.c)★: 单 token 步(没开投机, 或投机这一步歇着)且暖过一步直发之后,
             * 一步 = 写槽 + 一发图 + 读设备 argmax。第一步仍走下面的直发(把平面副本/暂存那些懒分配全建好, 捕获态下不许分配)。
             * ★投机歇轮的步也走图★(2026-09-18): 图里的 hc_mean 核按设备位置把 main_hidden 落进环, 下次出草稿时按位置差补窗口,
             * 所以歇着的步不必再走直发付 5 ms 发射间隙 —— 以前 --dspark 一开, 连歇着的步都比纯解码慢(23.65 vs 25.9 t/s)。
             * ★先发图再 emit★: emit 是 fwrite+fflush 到文件, 实测 300 µs/步 —— 放在两步之间 GPU 就干等 300 µs,
             * 放在图跑着的时候做就是白赚。eos/上下文满的判断在发图之前(它们决定还发不发), 输出顺序与原来逐字相同。 */
            /* ★投机歇轮的步走图(2026-09-19 恢复)★: 09-18 第三版 2K 投机 + 图混跑 14 步 "illegal memory access" 已定罪 ——
             * 验证批(直发, n≤6)把后端暂存扩容换了指针, 图里烤死的旧指针失效; 现在 core_decode_graph.c 发图前对暂存代号,
             * 变了就重捕获(见 ds4_gpu_v41_scratch_generation)。fix4 那趟 58% 的步在歇, 每步直发比走图多付 3 ms。 */
            if (!draft_now && v41_graph_ready(&st) && tok != eos && st.n_past + 1 <= st.ctx) {
                int32_t nt = 0;
                const double ts0 = now_sec();
                if (!v41_graph_launch(e, &st, (int32_t)tok, 0)) { ok = false; break; }
                if (emit && emit(tok, ud) != 0) { (void)v41_graph_wait(e, &st, &nt); break; }   /* 图已发, 等完再走 */
                if (!v41_graph_wait(e, &st, &nt)) { ok = false; break; }
                if (spec) v41_sched_cost(&cal, 1u, (now_sec() - ts0) * 1e3);   /* 纯解码一步的墙钟 = 调度器的比价基准 */
                /* 惩罚路: 图末尾槽里的 token 不用, 读回这一步的 logits 行(row 0, n=1)在主机罚完采; 设备采样/argmax: nt 就是槽里的 token */
                if (rowbuf && !v41_next_token(&st, am, 0, rowbuf, &rng, &hist, &nt)) { ok = false; break; }
                tok = (int)nt;
                continue;
            }
            if (emit && emit(tok, ud) != 0) { hit_eos = tok == eos; break; }
            if (tok == eos) { hit_eos = true; break; }
            if (st.n_past + 1 > st.ctx) { fprintf(stderr, "\n[v41] 上下文满 %u\n", st.ctx); break; }
            uint32_t k = 0;
            int32_t batch[DS4_MTP_MAX_BLOCK + 1];
            batch[0] = (int32_t)tok;
            double tr0 = now_sec();
            float sched_val = 0.f;                 /* 调度器预测的"产出/成本"(k=0 的轮也有, 诊断行要打) */
            bool drafted = false, have_draft = false;
            const uint32_t pos_round = st.n_past;  /* 本轮块首位(tok)的绝对位置, 诊断行用 */
            if (draft_inflight) {   /* 上一轮收尾已发(或已试过发不出): 这里只等; 草稿的账从发出那一刻起算(含被它盖住的 emit 时间, GPU 为主) */
                draft_inflight = false; tr0 = t_dl0;
                if (draft_r == 1 && !v41_draft_wait(&dr)) { ok = false; break; }
                have_draft = draft_r != 0;
            } else if (draft_now) {   /* 本请求第一轮(没有上一轮收尾), 或上一轮不是投机轮 */
                dr.t_launched = 0.0;
                const int r = v41_draft_launch(e, &st, &dr, (int32_t)tok, st.n_past - 1u);
                if (r == 1 && t_prev_end > 0.0 && dr.t_launched > t_prev_end) { h_vd += dr.t_launched - t_prev_end; h_vdn++; }
                if (r == 1 && !v41_draft_wait(&dr)) { ok = false; break; }
                have_draft = r != 0;
            }
            if (have_draft) {
                /* ★验证几位由置信度定(mtp-1.md M5′, core_draft_sched.c)★
                 * 以前这里钉死 3。钉死的毛病在两头: 文本好猜时少赚(README 英文满打满算能接受 1.39/5),
                 * 难猜时白读专家(金融文本 0.78/3, 每多验一位就多读一份 1.61 GB 的专家权重)。
                 * 调度器拿草稿器自己报的 conf 算"再验一位期望多拿多少 token / 多付多少成本", 取最优。
                 * --dspark-verify N 仍可钉死一个 k(诊断与逐档量字节账用)。 */
                drafted = true;
                const uint32_t use = g_ds4_v41_verify_k ? g_ds4_v41_verify_k
                                                       : v41_draft_pick_k(dr.host_conf, dr.block, &sched_val, &cal);
                if (!g_ds4_v41_verify_k && sched_val < 1.0f) spec_skipped++;   /* 预测连草稿钱都赚不回来: 只计数, 本轮 k=0 */
                k = dr.block < use ? dr.block : use;
                for (uint32_t i = 0; i < k; i++) batch[i + 1u] = dr.host_ids[i + 1u];
            }
            const uint32_t nb = 1u + k;
            const double tr1 = now_sec();
            double tr2 = tr1, tr3;
            int32_t want[DS4_MTP_MAX_BLOCK + 1];   /* 逐位主模型的贪心结果: 第 i 位的 logits 预测的是 batch[i] 之后那一位 */
            bool walked_graph = false;             /* 这一发走了图(稳态成本)还是直发(暖身/捕获失败) */
            /* ★验证批走图★(2026-09-22, core_decode_graph.c): 这个 n 直发暖过之后, 快照 + 前向 + n 发 argmax + 读回一发图搞定;
             * 走不了(没暖/捕获失败)就按下面的直发路(快照 → 前向 → 逐位 argmax), 两条路输出逐字节同。 */
            if (k && v41_graph_batch_ready(&st, nb) && v41_graph_batch_launch(e, &st, batch, nb)) {
                h_dv += now_sec() - tr1; h_dvn++;
                if (!v41_graph_batch_wait(e, &st, want)) { ok = false; break; }
                tr3 = now_sec(); walked_graph = true; t_prev_end = tr3;
            } else if (!k && v41_graph_ready(&st)) {
                /* 草稿白跑的轮(调度器判 k=0): 这一步就是普通单 token 步, 走 n=1 的图(2026-09-28; 以前落到下面的直发, 每轮多付 ~4 ms
                 * 发射间隙 —— 贪心那条真实请求上 63/399 轮是 k=0)。eos/上下文满在上面已判过, 与纯解码分支同一条件。 */
                int32_t nt = 0;
                if (!v41_graph_launch(e, &st, (int32_t)tok, 1) || !v41_graph_wait(e, &st, &nt)) { ok = false; break; }
                want[0] = nt;
                tr3 = now_sec(); walked_graph = true;
            } else {
                if (k && !v41_spec_snapshot(&st, nb)) { ok = false; break; }
                tr2 = now_sec();
                if (!v41_forward(e, &st, batch, nb)) { ok = false; break; }
                if (k && !ds4_gpu_synchronize()) { ok = false; break; }   /* 分账要真壁钟, 不同步量到的是发射时间 */
                tr3 = now_sec();
                if (!v41_device_next(&st, am, 0u, nb, batch, want)) { ok = false; break; }   /* 采样核 / 逐行 argmax, 拼成 want[] */
            }
            /* 惩罚路: 投机已拒 ⇒ nb 恒 1, 只有 want[0]; 直发这一步(暖身步/捕获失败的重来路)也按主机路取 */
            if (rowbuf && !v41_next_token(&st, am, 0, rowbuf, &rng, &hist, &want[0])) { ok = false; break; }
            if (k) { ms_draft += (tr1 - tr0) * 1e3; ms_snap += (tr2 - tr1) * 1e3;
                     ms_verify += (tr3 - tr2) * 1e3; ms_argmax += (now_sec() - tr3) * 1e3; }
            if (drafted) {
                /* 调度器的账只记★稳态★成本: 每个 n 第一次直发是暖身(懒分配 + 首次触碰, 可能比走图慢几倍), 记进均值就会让那个 n 显得贵、
                 * 调度器躲着它、它永远攒不到走图的样本 —— 自证的陷阱。规则: 走了图就记; 直发但这个 n 下次仍走不了图(图关/捕获失败)也记
                 * (直发就是它的真实成本); 直发而下次能走图 = 暖身, 不记。草稿同理(core_v41_draft.c 的 last_warm)。 */
                const bool steady = walked_graph || !(k ? v41_graph_batch_ready(&st, nb) : v41_graph_ready(&st));
                if (!dr.last_warm) v41_sched_draft_cost(&cal, (tr1 - tr0) * 1e3);
                if (steady) v41_sched_cost(&cal, nb, (tr3 - tr2) * 1e3);
            }
            uint32_t a = 0;
            while (a < k && want[a] == batch[a + 1u]) a++;   /* 接受最长前缀(采样下 want 已按"接受 ⇒ 草稿 / 拒绝 ⇒ 残差"拼好, 同一句) */
            /* 每个出过草稿的轮都观测首位(k=0 的轮看吐出的 token 是否就是草稿首位), 否则校准会死锁在 ρ=0(见 core_v41.h) */
            if (drafted && !g_ds4_v41_verify_k) v41_sched_observe(&cal, dr.host_conf, k ? (a >= 1u) : (want[0] == dr.host_ids[1]));
            /* ★每出一次草稿打一行账(2026-09-19, --emit-trace 或 --v41-prof)★: 位置 / 调度器选的 k 与预测比值 / 实际接受 /
             * 五位的 sigmoid(conf)。k=0 的轮(草稿白跑、要歇 16 步)也打 —— 调度器"该不该歇"的判决只能拿这张表复核:
             * 每格 conf 对上实际接受率才算校准, 对不上就是调度器在按错的概率算账。 */
            if (drafted && (g_ds4_v41_prof || g_ds4_v41_emit_trace)) {
                fprintf(stderr, "[dspark] pos %u k %u val %.3f acc %u conf", pos_round, k, (double)sched_val, a);
                for (uint32_t i = 0; i < dr.block; i++) {
                    const float c = dr.host_conf[i];
                    fprintf(stderr, " %.3f", isfinite(c) ? 1.0 / (1.0 + exp(-(double)c)) : 0.0);
                }
                if (g_ds4_v41_prof) {   /* 草稿这几位 vs 主模型自己的几位 —— 看是"接近但不同"还是"完全不搭" */
                    fprintf(stderr, " | 草稿");
                    for (uint32_t i = 0; i < k; i++) fprintf(stderr, " %d", batch[i + 1u]);
                    fprintf(stderr, " | 主模型");
                    for (uint32_t i = 0; i < nb; i++) fprintf(stderr, " %d", want[i]);
                }
                fprintf(stderr, "\n");
            }
            if (k) {
                spec_rounds++; spec_acc += a; spec_off += k; spec_hist[a]++;
                const double tb0 = now_sec();
                if (!v41_spec_rollback(&st, 1u + a)) { ok = false; break; }
                ms_snap += (now_sec() - tb0) * 1e3;
            }
            const int tok_next = (int)want[a];
            /* ★先发下一轮草稿, 再 emit★(2026-10-07): nsys 量到验证完→下一轮草稿首核之间 GPU 空 1.26 ms, 大头是 emit(fwrite+fflush,
             * 每个接受 token 一次)与簿记。下一轮草稿只依赖回滚后的状态(已回滚)与 tok_next, 这里就能发; 会停的轮不发(多跑一次草稿无害但没必要)。
             * 发出之后的 emit 顺序与原来逐字相同(接受的几位 → 轮顶的 tok_next)。 */
            bool cont = spec && produced + a < n_predict && tok_next != eos && st.n_past + 1 <= st.ctx;
            for (uint32_t i = 0; cont && i < a; i++) if (batch[i + 1u] == eos) cont = false;
            if (cont) {
                t_dl0 = now_sec(); dr.t_launched = 0.0;
                draft_r = v41_draft_launch(e, &st, &dr, (int32_t)tok_next, st.n_past - 1u);
                if (draft_r == 1 && t_prev_end > 0.0 && dr.t_launched > t_prev_end) { h_vd += dr.t_launched - t_prev_end; h_vdn++; }
                draft_inflight = true;   /* 发不出(draft_r=0)也标上: 轮顶别再试第二次 */
            }
            for (uint32_t i = 0; i < a; i++) {   /* 白赚的那几位: 草稿与主模型一致, 直接吐 */
                /* ★到上限就停★(09-24): 一轮接受多位时以前会越过 n_predict 多吐几个(-n 2048 吐 2050)。温 0 下多出来的也是对的 token,
                 * 但服务端 max_tokens 是硬上限, 纯解码路恰好停在上限 —— 投机默认开之后两条路必须同一个上限语义。 */
                if (produced >= n_predict) break;
                produced++;
                const int t = (int)batch[i + 1u];
                /* 位置 = 回滚后的 n_past 减去还没吐的那几位(与纯解码那条打印的是同一个绝对位置口径) */
                if (g_ds4_v41_emit_trace) fprintf(stderr, "[emit] %u %d\n", st.n_past - a + i, t);
                if ((emit && emit(t, ud) != 0) || t == eos) { hit_eos = t == eos; stop = true; break; }
            }
            tok = tok_next;
        }
        if (draft_inflight && draft_r == 1) (void)v41_draft_wait(&dr);   /* 收尾: 最后一轮发了草稿但循环停了(emit 返回非零等), 等它完再释放 */
        if (spec_rounds) {
            fprintf(stderr, "\n[v41] DSpark: %u 轮(调度器判亏本歇了 %u 次), 平均接受 %.2f/%u 位; 直方图",
                    spec_rounds, spec_skipped, (double)spec_acc / spec_rounds, dr.block);
            for (uint32_t i = 0; i <= dr.block; i++) fprintf(stderr, " %u:%u", i, spec_hist[i]);
            const double R = (double)spec_rounds, ea = 1.0 + (double)spec_acc / R;
            const double per = (ms_draft + ms_snap + ms_verify + ms_argmax) / R;
            fprintf(stderr, "\n[v41] 一轮 %.1f ms = 草稿 %.1f + 快照回滚 %.1f + 验证 %.1f + argmax %.1f"
                            "; 产出 %.2f token ⇒ %.1f ms/token\n",
                    per, ms_draft / R, ms_snap / R, ms_verify / R, ms_argmax / R, ea, per / ea);
            {   /* 一直打(不只 prof): prof 模式自带逐段同步会把空隙量歪; 这一行只在投机轮数 > 0 时出, 一条请求一行 */
                double bp = 0, bl = 0, pb = 0, pe = 0, pc = 0, pa = 0, tc = 0; uint32_t bn = 0, nc = 0;
                v41_graph_batch_host(&st, &bp, &bl, &bn); v41_graph_batch_prep(&st, &pb, &pe, &pc, &pa); v41_graph_capture_cost(&st, &tc, &nc);
                const double gs = dr.gsteps ? (double)dr.gsteps : 1.0, bd = bn ? (double)bn : 1.0;
                fprintf(stderr, "[v41] 主机空隙/轮: 验证完→草稿图发出 %.2f ms(含 写槽 %.3f + cudaGraphLaunch(草稿图, 走图 %u 轮) %.2f; 其余 = 接受/回滚/emit)"
                                " | 草稿完→验证图发出 %.2f ms(含 起手 %.2f = 写槽 %.3f + engram 提交 %.3f + 图校验 %.3f + 置位 %.3f; cudaGraphLaunch %.2f); 样本 %u/%u 轮"
                                " | 捕获(一次性, 已摊在上面两项里): 验证图 %u 次共 %.0f ms, 草稿图 %u 次共 %.0f ms\n",
                        h_vdn ? h_vd / h_vdn * 1e3 : 0.0, dr.h_slots / gs * 1e3, dr.gsteps, dr.h_launch / gs * 1e3,
                        h_dvn ? h_dv / h_dvn * 1e3 : 0.0, bp / bd * 1e3, pb / bd * 1e3, pe / bd * 1e3, pc / bd * 1e3, pa / bd * 1e3, bl / bd * 1e3, h_vdn, h_dvn,
                        nc, tc * 1e3, dr.gcaps, dr.h_capture * 1e3);
            }
        }
        g_v41_last_spec_rounds = (int)spec_rounds; g_v41_last_spec_off = (int)spec_off; g_v41_last_spec_acc = (int)spec_acc;
        if (spec) v41_draft_free(&dr);
        const double t2 = now_sec();
        const int n_out = produced - (hit_eos ? 1 : 0);   /* 真吐出的 token 数, 与服务端 usage.completion_tokens 同口径 */ if (n_out > 0) fprintf(stderr, "\n[v41] decode %d token %.1fs (%.2f t/s)\n", n_out, t2 - t1, (double)n_out / (t2 - t1 + 1e-9));
        rc = ok ? 0 : 1;
    } while (0);
    free(rowbuf); free(hist.tok); free((void *)hist.brk);
    if (am) ds4_gpu_tensor_free(am);
    v41_state_free(&st);
    return rc;
}
#else
int ds4_engine_v41_generate_argmax(ds4_engine *e, const int *prompt, int n_prompt, int n_predict, ds4_v41_emit_fn emit, void *ud) {
    (void)e; (void)prompt; (void)n_prompt; (void)n_predict; (void)emit; (void)ud;
    fprintf(stderr, "ds4: V4.1 只有 GPU 路\n"); return 1;
}
#endif
