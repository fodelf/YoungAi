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

#define V41_EGATHER_THREADS 48u   /* engram 取行的常驻线程池大小(解码 n=1 时任务单元正好 2 层 × 24 行) */
#define V41_EDIO_ALIGN 4096u      /* O_DIRECT 的对齐粒度(逻辑块); 一行只有 264 B, 所以要读对齐超集 + 落脚点 */

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
    /* 投机验证的回滚快照(speed.md 段 6 D1): 验证批会把 1+k 个位置的 KV 全写进缓存, 其中没被接受的
     * 那几位必须撤销 —— 撤不掉的只有两处"破坏性平移"(窗口缓冲、压缩器余行), 所以前向前备份它们。
     * 其余(comp_kv/index_k 按组号写、ng_src/cpend 是计数)回退计数后下一轮直接覆盖, 不用备份。 */
    ds4_gpu_tensor *snap_win[DS4_MAX_LAYER], *snap_cpre_kv[DS4_MAX_LAYER], *snap_cpre_sc[DS4_MAX_LAYER];
    uint32_t snap_cpend[DS4_MAX_LAYER], snap_ng[DS4_MAX_LAYER], snap_past, snap_on;
    uint32_t cpend[DS4_MAX_LAYER];   /* 源层余行数(= pos0 % ratio) */
    uint32_t ng_src[DS4_MAX_LAYER];  /* 源层缓存里的组数(含本 chunk 新完成的) */
    ds4_gpu_tensor *iq, *iw, *iscore, *cand, *idx;       /* [cap][32·128], [cap][32], [cap][ctx], u8 [cap][ctx], i32 [cap][512] */
    ds4_gpu_tensor *o, *low, *attn_out;                  /* [cap][64·512], [cap][8192], [cap][E] */
    ds4_gpu_tensor *glog, *sel, *rw, *routed;            /* [cap][384], i32 [cap][6], [cap][6], [cap][E] */
    ds4_gpu_tensor *sg, *su, *sh, *so, *y;               /* [cap][2304] ×3, [cap][E] ×2 */
    ds4_gpu_tensor *logits;     /* [cap][V] 本 chunk 全位置 */
    ds4_gpu_tensor *eraw, *erows, *ekv;                  /* engram: u8 [cap][24][264] 原始行字节, [cap][24·256] 解码行, [cap][(hc+1)·E] wkv 输出 */
    struct { int fd; int dio; uint64_t size; } eshard[DS4_V41_MAX_ENGRAM];   /* 表分片 fd(懒开; 行用并行 pread 拉, 不 mmap) */
    void *ejob;                 /* engram 取行后台任务(core_v41_engram.c 私有): 前向一开始发, 到 engram 层再收 */
    ds4_gpu_tensor *ampA[DS4_MAX_LAYER], *ampB[DS4_MAX_LAYER];   /* 反修放大器 A[K][D] / B[K][D](f32, 设备常驻), 无=NULL */
    uint32_t ampK[DS4_MAX_LAYER];
    ds4_gpu_tensor *ampT;       /* [cap][Kmax] 中间 T = x·B */
    int stop_early;             /* 反修钩子说"取完了": 本次前向跳过余下层与出口(core_v41_forward.c), 调用方不读 logits */
    const char *dump_prefix;    /* 非 NULL: 每层 MoE 入/出 落 <prefix>.x_Lnn.bin / .y_Lnn.bin(对拍夹具, 单块 n≤64 才开) */
    int no_engram;              /* 对拍夹具: 跳过 engram 层(与 Python --no-engram 同口径) */
    int ced_skip;               /* ★CED★ 本 chunk 只跑编码器段 + 分界层的 KV 投影, 不跑解码器段, 不出 logits。
                                 * 官方 §2.2/§3.2.2: 解码器的全局 KV 就是编码器末态经分界层 wkv/wk 的投影 ——
                                 * 所以提示的中间块根本不需要跑后半 40% 的层, 它们的输出没有任何下游消费者。
                                 * 只有最后一块要跑满(它同时充当官方说的"解码器有界回放")。 */
    /* ---- DSpark(speed.md 段 6 D1) ---- */
    ds4_gpu_tensor *mainh;      /* [mainh_cap][n_target][E] 主前向顺手取的 main_hidden(只留本 chunk 末尾几行);
                                 * ★取的是目标层的**注意力输入**(engram 之后、层之前的 hc 四路均值), 不是层输出 ——
                                 * 取错不报错, 只是接受率掉到 1 附近(speed.md §7)。 */
    uint32_t mainh_cap, mainh_rows;
    int draft;                  /* 1 = 这个 state 是草稿塔的: 层权重走 weights.mtp.tower[il], 无 engram/无压缩 KV,
                                 * 注意力块内全可见, MoE 走逐专家 FP4。主状态恒 0。 */
    ds4_gpu_tensor *main_x;     /* 草稿态: [n][E] 块注意力的 KV 来源(main_norm(main_proj(main_hidden))) */
    const uint64_t *tower_exp_off[DS4_MTP_MAX_TOWERS];   /* 每塔 [3][n_expert] 专家张量偏移表 */
    uint32_t idx_topk;          /* 本 chunk 最近一个 indexer 源层产出的 topk 宽度 */
    int16_t idx_owner;          /* 本 chunk 最近一个 indexer 源层(共享 topk 的来源), -1 无 */
    int16_t cand_owner;         /* 本 chunk 候选块掩码是否已产 */
} ds4_v41_state;

#define DS4_MTP_MAX_BLOCK 8u   /* 一块最多几个草稿位(官方 5); 只是主机小数组的上限, 真值读元数据 */

/* DSpark 草稿器(core_v41_draft.c)。ready=0 = 这份 GGUF 没带三塔或没带运行参数 ⇒ 调用方走纯单 token 解码。 */
typedef struct {
    ds4_v41_state st;          /* 草稿塔自己的一套缓冲(draft=1, 层号 = 塔号) */
    ds4_gpu_tensor *mainx_raw; /* main_proj 出口(norm 前) */
    ds4_gpu_tensor *mk_embed;  /* [cap][rank] 每位的 markov embed(confidence 要) */
    ds4_gpu_tensor *mk_cur;    /* [rank] 当前这一位的(给 markov_head 当输入) */
    ds4_gpu_tensor *mk_bias;   /* [V] 当前这一位的 logit 偏置 */
    ds4_gpu_tensor *conf_in, *conf;   /* [cap][E+rank] / [cap] */
    ds4_gpu_tensor *ids, *ids_next;   /* i32 [cap]: [0]=真 token, [1..block]=草稿; argmax 的落点 */
    ds4_gpu_tensor *h;         /* [cap][E] 出口 hc_pre 的结果(confidence 也吃它) */
    uint64_t *exp_off[DS4_MTP_MAX_TOWERS];
    uint32_t block, ready;
    int32_t host_ids[DS4_MTP_MAX_BLOCK + 1];
    float host_conf[DS4_MTP_MAX_BLOCK];
} ds4_v41_draft;

bool v41_draft_alloc(ds4_engine *e, ds4_v41_draft *dr);
void v41_draft_free(ds4_v41_draft *dr);
/* 一轮草稿: 主前向刚推进了 rows 个已确认位置(mainh 里有它们的 main_hidden), 最后一位是 tok/pos_main。
 * 出 dr->host_ids[1..block](草稿 token)与 dr->host_conf[0..block-1](每位的条件接受概率)。 */
bool v41_draft_step(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int32_t tok, uint32_t pos_main, uint32_t rows);

/* engram 取行的一个工作单元(core_v41_engram.c 填, core_v41_epool.c 的池跑) */
typedef struct { const ds4_engine *e; const ds4_v41_state *st; uint64_t u0, u1; int err; uint8_t *bounce; } v41_eworker;
void *v41_eworker_run(void *arg);
uint32_t v41_epool_threads(void);                       /* 池里有几个线程(懒起); 0 = 起不来, 调用方自己同步做 */
bool v41_epool_submit(v41_eworker *w, uint32_t njob);   /* 提交一轮, 不阻塞 */
void v41_epool_wait(void);                              /* 等这一轮干完 */

extern int g_ds4_v41_prof;   /* --v41-prof(core_v41_api.c) */
extern int g_ds4_v41_dspark;         /* --no-dspark 关掉投机解码(对拍) */
extern int g_ds4_v41_decoder_full;   /* --decoder-full: 关 CED, 每块跑满 40 层(精确路) */
extern const char *g_ds4_v41_amp_dir;   /* --zchain <dir>(V4.1 形态: amp_Lnn.bin 目录), core_engine_open.c 转来 */
extern float g_ds4_v41_amp_scale;       /* --zchain-scale β: 加载时 A *= β(默认 1.0) */
extern const char *g_ds4_v41_pt_dir;    /* --posttrain <dir>: 三文件部署第三件(后训练增益目录), 与 ② 的表逐元素相乘 */
extern ds4_v41_moe_hook_fn g_ds4_v41_hook;   /* 反修钩子(ds4_engine_v41_set_moe_hook), 无=NULL */
extern void *g_ds4_v41_hook_ud;
extern int g_ds4_v41_hook_layer;   /* 钩子只在这一层回调(-1=每层); 见 ds4_engine_v41_set_moe_hook_layer */
bool v41_attention_kv_only(ds4_engine *e, ds4_v41_state *st, uint32_t il);   /* CED 分界层: 只写全局 KV */
bool v41_amp_load(ds4_v41_state *st, const char *dir, const char *pt_dir);   /* core_v41_amp.c: ②反修目录 + ③后训练目录 */
void v41_amp_free(ds4_v41_state *st);
int v41_amp_hook(ds4_v41_state *st, uint32_t il);        /* 钩子回调(挂了才动): 同步 → x/y/sel/rw 下主机 → 回调; 0 继续 / 1 提前结束(置 stop_early) / <0 失败 */
bool v41_amp_apply(ds4_v41_state *st, uint32_t il);      /* y += x·(B·A) → bf16(挂了该层才动) */
ds4_gpu_tensor *v41_alloc(uint64_t bytes, bool *ok);   /* 小工具: 分配失败只置 ok=false, 调用方一路攒到最后再判 */
bool v41_state_alloc(ds4_v41_state *st, uint32_t cap_tok, uint32_t ctx);
void v41_state_free(ds4_v41_state *st);
bool v41_forward(ds4_engine *e, ds4_v41_state *st, const int32_t *ids, uint32_t n);   /* 追加 n 个 token, st->logits[n][V] */
bool v41_spec_snapshot(ds4_v41_state *st);                      /* 验证批前: 备份会被破坏性平移的两处缓存 */
bool v41_spec_rollback(ds4_v41_state *st, uint32_t keep);       /* 验证批后: 只保留前 keep 个位置, 其余撤销 */
bool v41_layer(ds4_engine *e, ds4_v41_state *st, uint32_t il);        /* 一层(主干层或草稿塔, 看 st->draft) */
bool v41_attention(ds4_engine *e, ds4_v41_state *st, uint32_t il);   /* core_v41_attn.c: st->xn → st->attn_out, 更新各缓存 */
bool v41_draft_push_main(ds4_engine *e, ds4_v41_state *st, uint32_t rows);   /* core_v41_attn.c: 已确认位置的 main_x 进各塔窗口 */
bool v41_engram(ds4_engine *e, ds4_v41_state *st, uint32_t il);      /* core_v41_engram.c: 就地改 st->hc(engram 层才调) */
bool v41_engram_prefetch(ds4_engine *e, ds4_v41_state *st);          /* 前向开头: 所有 engram 层的行一次性后台并行 pread */
void v41_engram_close(ds4_v41_state *st);

#endif /* !DS4_NO_GPU */
#endif
