/* core_v41.h — DeepSeek V4.1 持久状态 + 分块增量前向(2026-09-12 战役 P2c)的内部声明。只被 core_internal.h 包含。
 *
 * 一个状态 = 一段上下文: 已处理 n_past 个位置, 每次 v41_forward 喂 n(≤ cap_tok) 个新 token, 位置 pos0=n_past 起,
 * 出这 n 个位置的 logits。prefill = 大块, 解码 = n=1, 同一条路(这是对拍能闭合的前提: 分块/逐 token 与整批逐位一致)。
 * 缓存(全 f32 行主序, 值落在官方量化格点上; 压到 fp4/fp8 字节是 P4 的事):
 *   win[il]      [(SWA+cap)][512]  每层窗口 KV: 前 SWA 行 = 位置 pos0-SWA..pos0-1, 后 n 行本 chunk; 层算完把尾 SWA 行挪到头
 *   comp_kv[src] [ctx/ratio][512]  kv 源层压缩 KV(rope+fp4 后), 按组追加; index_k[src] [ctx/ratio][128] 同步的 indexer 键
 *   cpre_*[src]  [(ratio+cap)][512] ratio>1 源层的池化输入余行(跨 chunk 没凑满一组的 token 留到下块)
 * 为什么不复用 ds4_gpu_graph: 那套是 V4 Flash 的状态机(比 4/128 压缩器暂存、indexer 自带缓存、spec 快照…), V4.1 的
 * 注意力侧几乎每件都不同(源层共享缓存、fp4 latent、wk 键、两级 topk、engram)。 */
#ifndef DS4_CORE_V41_H
#define DS4_CORE_V41_H
#ifndef DS4_NO_GPU

#define DS4_V41_CHUNK 512u          /* prefill 分块(缓冲按它分配; 对拍用 --v41-chunk 改小看自洽) */
#define DS4_V41_MAX_CTX_P2C 32768u  /* topk/候选块核用 shared 存整段组数, 48 KB ⇒ ratio-1 层 ≤ 48k 组; P4 换 radix select 再放开 */

typedef struct {
    uint32_t cap_tok, ctx;      /* 每块最多 token 数 / 位置容量 */
    uint32_t n_past;            /* 已入缓存的位置数 */
    uint32_t n, pos0;           /* 本 chunk: token 数, 起始绝对位置 */
    int32_t *hist;              /* 主机 token 历史 [ctx](engram 哈希回看 3 个 token 要跨 chunk) */
    ds4_gpu_tensor *tok, *pos, *posg;                    /* i32 [cap]: id / 绝对位置; i32 [cap]: 新组位置 g·ratio */
    ds4_gpu_tensor *hc, *hc2;   /* [cap][hc][E] 残差四路, 乒乓 */
    ds4_gpu_tensor *mix, *pre, *post, *comb, *pre_mix;   /* [cap][24] [cap][4] [cap][4] [cap][16] [cap][4] */
    ds4_gpu_tensor *x, *xn;     /* [cap][E] 子层输入 / 归一化后 */
    ds4_gpu_tensor *qr, *qrn, *q, *kv, *kvn;             /* [cap][1280] ×2, [cap][64·512], [cap][512] ×2 */
    ds4_gpu_tensor *ckv, *csc, *pooled, *latent, *ktmp;  /* [cap][512] ×4, [cap][128] */
    ds4_gpu_tensor *win[DS4_MAX_LAYER], *wintmp;         /* 窗口缓冲 / 平移暂存 [SWA][512] */
    ds4_gpu_tensor *comp_kv[DS4_MAX_LAYER], *index_k[DS4_MAX_LAYER];
    ds4_gpu_tensor *cpre_kv[DS4_MAX_LAYER], *cpre_sc[DS4_MAX_LAYER];
    uint32_t cpend[DS4_MAX_LAYER];   /* 源层余行数(= pos0 % ratio) */
    uint32_t ng_src[DS4_MAX_LAYER];  /* 源层缓存里的组数(含本 chunk 新完成的) */
    ds4_gpu_tensor *iq, *iw, *iscore, *cand, *idx;       /* [cap][32·128], [cap][32], [cap][ctx], u8 [cap][ctx], i32 [cap][512] */
    ds4_gpu_tensor *o, *low, *attn_out;                  /* [cap][64·512], [cap][8192], [cap][E] */
    ds4_gpu_tensor *glog, *sel, *rw, *routed;            /* [cap][384], i32 [cap][6], [cap][6], [cap][E] */
    ds4_gpu_tensor *sg, *su, *sh, *so, *y;               /* [cap][2304] ×3, [cap][E] ×2 */
    ds4_gpu_tensor *logits;     /* [cap][V] 本 chunk 全位置 */
    ds4_gpu_tensor *eraw, *erows, *ekv;                  /* engram: u8 [cap][24][264] 原始行字节, [cap][24·256] 解码行, [cap][(hc+1)·E] wkv 输出 */
    struct { int fd; uint64_t size; } eshard[DS4_V41_MAX_ENGRAM];   /* 表分片 fd(懒开; 行用并行 pread 拉, 不 mmap) */
    void *ejob;                 /* engram 取行后台任务(core_v41_engram.c 私有): 前向一开始发, 到 engram 层再收 */
    ds4_gpu_tensor *ampA[DS4_MAX_LAYER], *ampB[DS4_MAX_LAYER];   /* 反修放大器 A[K][D] / B[K][D](f32, 设备常驻), 无=NULL */
    uint32_t ampK[DS4_MAX_LAYER];
    ds4_gpu_tensor *ampT;       /* [cap][Kmax] 中间 T = x·B */
    int stop_early;             /* 反修钩子说"取完了": 本次前向跳过余下层与出口(core_v41_forward.c), 调用方不读 logits */
    const char *dump_prefix;    /* 非 NULL: 每层 MoE 入/出 落 <prefix>.x_Lnn.bin / .y_Lnn.bin(对拍夹具, 单块 n≤64 才开) */
    int no_engram;              /* 对拍夹具: 跳过 engram 层(与 Python --no-engram 同口径) */
    uint32_t idx_topk;          /* 本 chunk 最近一个 indexer 源层产出的 topk 宽度 */
    int16_t idx_owner;          /* 本 chunk 最近一个 indexer 源层(共享 topk 的来源), -1 无 */
    int16_t cand_owner;         /* 本 chunk 候选块掩码是否已产 */
} ds4_v41_state;

extern int g_ds4_v41_prof;   /* --v41-prof(core_v41_api.c) */
extern const char *g_ds4_v41_amp_dir;   /* --zchain <dir>(V4.1 形态: amp_Lnn.bin 目录), core_engine_open.c 转来 */
extern float g_ds4_v41_amp_scale;       /* --zchain-scale β: 加载时 A *= β(默认 1.0) */
extern const char *g_ds4_v41_pt_dir;    /* --posttrain <dir>: 三文件部署第三件(后训练增益目录), 与 ② 的表逐元素相乘 */
extern ds4_v41_moe_hook_fn g_ds4_v41_hook;   /* 反修钩子(ds4_engine_v41_set_moe_hook), 无=NULL */
extern void *g_ds4_v41_hook_ud;
extern int g_ds4_v41_hook_layer;   /* 钩子只在这一层回调(-1=每层); 见 ds4_engine_v41_set_moe_hook_layer */
bool v41_amp_load(ds4_v41_state *st, const char *dir, const char *pt_dir);   /* core_v41_amp.c: ②反修目录 + ③后训练目录 */
void v41_amp_free(ds4_v41_state *st);
int v41_amp_hook(ds4_v41_state *st, uint32_t il);        /* 钩子回调(挂了才动): 同步 → x/y/sel/rw 下主机 → 回调; 0 继续 / 1 提前结束(置 stop_early) / <0 失败 */
bool v41_amp_apply(ds4_v41_state *st, uint32_t il);      /* y += x·(B·A) → bf16(挂了该层才动) */
bool v41_state_alloc(ds4_v41_state *st, uint32_t cap_tok, uint32_t ctx);
void v41_state_free(ds4_v41_state *st);
bool v41_forward(ds4_engine *e, ds4_v41_state *st, const int32_t *ids, uint32_t n);   /* 追加 n 个 token, st->logits[n][V] */
bool v41_attention(ds4_engine *e, ds4_v41_state *st, uint32_t il);   /* core_v41_attn.c: st->xn → st->attn_out, 更新各缓存 */
bool v41_engram(ds4_engine *e, ds4_v41_state *st, uint32_t il);      /* core_v41_engram.c: 就地改 st->hc(engram 层才调) */
bool v41_engram_prefetch(ds4_engine *e, ds4_v41_state *st);          /* 前向开头: 所有 engram 层的行一次性后台并行 pread */
void v41_engram_close(ds4_v41_state *st);

#endif /* !DS4_NO_GPU */
#endif
