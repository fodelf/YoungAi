/* core_v41.h — DeepSeek V4.1 持久状态 + 分块增量前向(2026-09-12 战役 P2c)的内部声明。只被 core_internal.h 包含。
 *
 * 一个状态 = 一段上下文: 已处理 n_past 个位置, 每次 v41_forward 喂 n(≤ cap_tok) 个新 token, 位置 pos0=n_past 起,
 * 出这 n 个位置的 logits。prefill = 大块, 解码 = n=1, 同一条路(这是对拍能闭合的前提: 分块/逐 token 与整批逐位一致)。
 * 缓存(全 f32 行主序, 值落在官方量化格点上; 压到 fp4/fp8 字节是 P4 的事):
 *   win[il]      [(SWA+cap)][512]  每层窗口 KV。前 SWA 行是**环**(decode.md D1, 官方 window_kv_cache[pos % win]):
 *                位置 a 恒住在 a % SWA 那一格, 只装已提交的位置; 后 n 行是本批, 层算完由 win_commit 写进环。
 *                本批不直接进环, 是为了让投机验证批里没被接受的那几位不污染历史(回滚只还原几格, 不是整份 128 行)。
 *   comp_kv[src] [ctx/ratio][512]  kv 源层压缩 KV(rope+fp4 后), 按组追加; index_k[src] [ctx/ratio][128] 同步的 indexer 键
 *   cpre_*[src]  [(ratio+cap)][512] ratio>1 源层的池化输入余行(跨 chunk 没凑满一组的 token 留到下块)
 * 为什么不复用 ds4_gpu_graph: 那套是 V4 Flash 的状态机(比 4/128 压缩器暂存、indexer 自带缓存、spec 快照…), V4.1 的
 * 注意力侧几乎每件都不同(源层共享缓存、fp4 latent、wk 键、两级 topk、engram)。 */
#ifndef DS4_CORE_V41_H
#define DS4_CORE_V41_H
#ifndef DS4_NO_GPU

#define V41_EGATHER_THREADS 48u   /* engram 取行的常驻线程池大小(解码 n=1 时任务单元正好 2 层 × 24 行) */
#define V41_EDIO_ALIGN 4096u      /* O_DIRECT 的对齐粒度(逻辑块); 一行只有 264 B, 所以要读对齐超集 + 落脚点 */

/* prefill 分块(缓冲按它分配; 对拍用 --v41-chunk 改小看自洽)。
 * ★512 → 2048(2026-09-29)★: 一块要把本层 384 个专家的位流全解一遍(每层 2.55 GB), 块里每个专家平均只摊到 n/64 个 token ——
 * 块 512 时 8 个, 解码成本摊不开。12k 真实提示同机同趟(speed-bench/prefill_ttft_ruler.sh, 生成文本四档逐字节同):
 * 512 = 565 t/s / 1024 = 655 / 2048 = 715 / 4096 = 753; 内存(MemAvailable 最低)9193 / 7111 / 6580 / 3029 MB —— 4096 贴到看门狗
 * 红线(2500), 2048 是速度与内存的拐点。随块长的缓冲(hc/q/o/logits)见下面的内存账; logits 与出口头暂存已不随块长(head_last_only)。 */
#define DS4_V41_CHUNK 2048u
/* 上下文硬上限。历史: topk 与候选块两个核都把"整段组数"放 shared(48 KB ⇒ ratio-1 层最多 ~78k 个位置),
 * 所以这里一直写死 32768。topk 核 2026-09-16 换成固定 256 桶的 radix select, 候选块核 2026-09-21 把
 * [nb] 两个数组挪进全局暂存 —— 两道 shared 墙都没了, 上限改由内存账定。
 * ★配到 1M 到底花多少(2026-09-22 逐项按 v41_state_alloc 算, 不是按 free 的差值估)★:
 *   压缩 KV + 索引键   4 个 kv 源层 (ctx/ratio + 2) × 360 B  ⇒ 1M 档 944 MB = 0.879 GiB   ← 这才是"1M 上下文"的价钱
 *   logits             cap_tok × 129280 × 4                  ⇒ 0.247 GiB(与 ctx 无关)
 *   窗口环 40 层       (128 + cap_tok) × 128 × 4             ⇒ 0.012 GiB(与 ctx 无关)
 *   iscore / cand      索引打分草稿, 见下                     ⇒ 按这一趟真正走到的位置长, 不按 ctx
 * iscore [cap_tok][ng] f32 + cand [cap_tok][ng] u8 曾经按 ctx 一次开满(1M 档 2.0 + 0.5 GiB) —— 它不是 KV,
 * 是每次调用都重写的草稿(核里的行距就是当次的 ng, 见 cuda_v41_indexer.inc.cu), 开成最坏情况纯属浪费,
 * 而且正是"把上下文配大很贵"这个错觉的来源(1M 看着 3.6 GiB, 其中 2.5 GiB 是这块草稿)。
 * 现在走 v41_index_scratch_prepare 按需要的组数翻倍长: 一条写到 4 万位的请求只用 128 MiB + 32 MiB。
 * ★C2(2026-09-30)★ cand 改成候选块核写的紧凑列表 i32 [行][1+topk_blocks](与 ng 无关, 2048 行 16.8 MB), 打分草稿只按行块开(C1)。
 * 速度不受影响: 核只扫真实组数, 短上下文还是短上下文。
 * 调用方(crewAI)本来就按 1M 报上下文窗口, 服务端报少了会在长提示上直接 400。
 * ★上下文没有常量★(用户 2026-09-22 "不要任何写死的上下文, 上下文大小只有 1M 这一个选择"): 它是模型自己声明的
 * (GGUF deepseek4.context_length ← HF max_position_embeddings), 装载时读进 g_ds4_v41.ctx(ds4_internal.h), 引擎里没有
 * 第二个数 —— CLI/服务端不接受 --ctx, 脚本里不许出现上下文数, V4.1 前向的位置边界一律从它取。分配仍按"这一趟真正
 * 走到的位置"(上面的账), 所以上下文本身不花内存。 */

/* ★后训练反传的前向存档(2026-10-01, core_ptrain*.c)★: 训练前向时把反传要的几样按层抄一份; 推理时状态里这个指针恒 NULL,
 * 前向只多一次指针判断, 算式与次序一个不动。 */
typedef struct {
    ds4_gpu_tensor *moe_in[DS4_MAX_LAYER];   /* 该层 MoE 的输入 xn(ffn_norm 出口) [cap][E]: 放大器的梯度要它; NULL = 这层不存 */
    ds4_gpu_tensor *hc_in[DS4_MAX_LAYER];    /* 该层入口(engram 之前)的 hc, ★bf16 打包(u16)★ [cap][HC][E]: 逐层重算反传的起点
                                              * (hc 的值本来就在 bf16 格点上, 存高 16 位无损, 内存减半) */
    ds4_gpu_tensor *pm_in[DS4_MAX_LAYER];    /* 该层入口的 pre_mix [cap][HC](上一层 ffn 半层产的 pre) */
    ds4_gpu_tensor *idx[DS4_MAX_LAYER];      /* 该层注意力用的压缩组号 [cap][topk] i32(消费层复用源层的 topk —— 倒着重算时源层已不在手边) */
    uint32_t topk[DS4_MAX_LAYER], iratio[DS4_MAX_LAYER], ng[DS4_MAX_LAYER];   /* 与 idx 同一次前向里的 topk 宽度 / 来源层压缩比 / 可见组数 */
    ds4_gpu_tensor *ekv[DS4_MAX_LAYER];      /* engram 层的门输入 key|value [cap][(HC+1)·E], bf16 打包(前向舍过 bf16, 无损): 重算本层要先过门,
                                              * 而查表那一步要从盘上 pread 整段行, 倒着算时不重做 */
    ds4_gpu_tensor *sel[DS4_MAX_LAYER];      /* 该层路由选中的专家 [cap][top-k] i32(梯度检查的冻结选择用) */
    ds4_gpu_tensor *hpert; int32_t hpert_layer; float hpert_eps; int hpert_mid;   /* hc 边界检查: 第 hpert_layer 层入口(mid=1: 中段)hc += eps·hpert
                                              * (只给 pt_hccheck 用; NULL = 不动) */
    int replay;                               /* 1 = 冻结选择前向: 路由与 indexer 的选择照上面存的(基准前向那份), 只重算连续量。
                                              * 只给梯度检查用: 让有限差分只量反传覆盖的那部分(选择当常量), 不吃离散翻转的跳变 */
} v41_tsave;

typedef struct {
    uint32_t cap_tok, ctx;      /* 每块最多 token 数 / 位置容量 */
    v41_tsave *tsave;           /* 后训练前向存档(core_ptrain*.c), NULL = 推理 */
    uint32_t n_past;            /* 已入缓存的位置数 */
    uint32_t n, pos0;           /* 本 chunk: token 数, 起始绝对位置 */
    int32_t *hist;              /* 主机 token 历史 [ctx](engram 哈希回看 3 个 token 要跨 chunk) */
    ds4_gpu_tensor *tok, *pos, *posg;                    /* i32 [cap]: id / 绝对位置; i32 [cap]: 新组位置 g·ratio */
    int32_t *posg_pin[DS4_MAX_LAYER];   /* 压缩源层各自的 pinned 位置槽 [cap](零拷贝小核按流灌进 posg; 同步 memcpy 会把并发道串行化, 2026-09-30) */
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
    uint32_t snap_n;             /* 快照那一批有几位(= 验证批 1+k): 回滚只还原 [keep, snap_n) 那几格环 */
    uint32_t cpend[DS4_MAX_LAYER];   /* 源层余行数(= pos0 % ratio) */
    uint32_t ng_src[DS4_MAX_LAYER];  /* 源层缓存里的组数(含本 chunk 新完成的) */
    uint32_t win_from[DS4_MAX_LAYER];   /* ★本层窗口环里最早有效的绝对位置★(2026-09-21, bug.md §6.2): CED 跳过解码器段的块
                                 * 不写 L20~L39 的环, 那些位置的槽是脏的; 注意力核把窗口下界钳到它(官方 get_window_topk_idxs
                                 * 未填槽标 -1 屏蔽, 这是同一语义)。编码器层与 --decoder-full 恒 0。 */
    ds4_gpu_tensor *iq, *iw, *iscore, *cand, *idx;       /* [cap][32·128], [cap][32], f32 [isrows][iscap], i32 [icrows][1+topk_blocks], i32 [cap][512] */
    uint32_t iscap;             /* iscore 现在按几组分的(0 = 还没分)。按用到的组数长, 不按 ctx: v41_index_scratch_prepare */
    uint32_t isrows;            /* iscore 现在按几行分的: 预填块 = 行块 Rb 行(2026-09-30 C1), 解码 = 验证批那几行(见 v41_index_scratch_prepare) */
    uint32_t icrows;            /* cand 现在按几行分的: 候选列表要跨层活到 L36, 预填块按整块 n 行开(C2: 每行 1 + topk_blocks 个 i32, 与 ng 无关) */
    uint32_t iscap_gen;         /* 长一次 +1: 指针换了, 烤在解码图里的旧地址作废(core_decode_graph.c 按它重捕获) */
    uint64_t uid;               /* ★状态的单调序号(v41_state_alloc 分配, 进程内不重复)★(2026-10-06): 合批整步图的键原来只认状态指针, 请求态关掉再开
                                 * 常拿到同一个地址, 第三路某一步命中第一路残留的图(烤着已释放的缓冲)→ illegal memory access(训练器合批探针实撞;
                                 * 服务端并发采样 10-01 "首 token 就挂"同病)。core_v41_mgraph.c 的键连它一起比。 */
    ds4_gpu_tensor *o, *low, *attn_out;                  /* [cap][64·512], [cap][8192], [cap][E] */
    ds4_gpu_tensor *glog, *sel, *rw, *routed;            /* [cap][384], i32 [cap][6], [cap][6], [cap][E] */
    ds4_gpu_tensor *sg, *su, *sh, *so, *y;               /* [cap][2304] ×3, [cap][E] ×2 */
    ds4_gpu_tensor *logits;     /* [logits_rows][V]: 打分路 = cap 行(全位置); 生成路只开 DS4_MTP_MAX_BLOCK+2 行(见 head_last_only) */
    uint32_t logits_rows;
    /* ★生成路的预填块只算末位的 logits★(2026-09-29): 生成只读最后一位, 而出口头对整块走"q4_K 解成 bf16 暂存(1.3 GB) + cuBLAS"是
     * 白算 + 白占(块 2048 时 logits 1.06 GB + 暂存 1.3 GB)。置 1 时 v41_forward_body 对 n > DS4_V41_GEMV_MAX_TOK 的块只把末位
     * 归一化后的行拷进 xlast, 走解码同款的 q4_K GEMV 出 logits 第 0 行; last_logit_row 告诉调用方末位的 logits 在哪一行。
     * 解码/验证批(n ≤ 8)不受影响, 每行照算。打分路(--score-ids)每个位置都要 logits, 不置。 */
    int head_last_only;
    uint32_t last_logit_row;
    ds4_gpu_tensor *xlast;      /* [1][E] 末位归一化后的出口输入 */
    ds4_gpu_tensor *eraw[DS4_V41_MAX_ENGRAM], *erows, *ekv;   /* engram: u8 [cap][24][264] 原始行字节(每个 engram 层一份: graph 路
                                 * 两层的行在同一处一起上传, 不能共用一块), [cap][24·256] 解码行, [cap][(hc+1)·E] wkv 输出 */
    /* ---- 解码整步 CUDA graph(2026-09-18, core_decode_graph.c) ---- */
    int graph;                  /* 1 = 正在按"设备位置"口径发核(捕获中): 位置相关的核读 st->pos 槽, grid 按桶上限开,
                                 * 主机不做任何按位置的分支。只有 n=1 的主路会置它。 */
    uint32_t graph_pos_lo, graph_pos_cap;   /* 图的有效位置区间 [lo, cap](桶); 捕获时按它算各核的上限 */
    uint32_t n_direct1;         /* 直发跑过几次 n=1 前向: 第一次把懒分配(平面副本/暂存/核属性)全暖了, 之后才许捕获 */
    uint32_t n_direct_n[DS4_MTP_MAX_BLOCK + 2u];   /* 同上, 按批大小 n 各记一份(投机验证批 n=1+k 各自的暂存/核属性也是懒建的) */
    void *dgraph;               /* core_decode_graph.c 私有(图实例 + pinned 槽; n=1 纯解码图与 n=2.. 验证批图各一张) */
    int dev_sample;             /* ★设备采样(2026-09-28, core_v41_sample.c)★ 1 = 图/直发末尾用采样核替掉 argmax(温度 > 0 且无惩罚);
                                 * 投机在这条路上照走(核里做拒绝采样)。惩罚路仍读回整行在主机做, 恒 0。 */
    ds4_gpu_sample_params samp; /* 采样参数(每请求常量, 捕获进图: 温度 / top_k / top_p / min_p / seed) */
    const ds4_decode_sampling *psamp;   /* 主机惩罚路的采样面(并发: 每请求一份, core_v41_req.c); NULL = 进程全局 g_decode_sampling */
    const ds4_gpu_tensor *spec_q;   /* 投机 + 采样时: 草稿塔的 logits [block][V](第 i 行 = 验证第 i 行的草稿分布 q), 验证核按它做拒绝采样; NULL = 点质量草稿 */
    int egraph_err;             /* graph 里的 host 节点等 engram 取行时发现失败(host 节点没法报错, 只能记下来事后查) */
    int egraph_uploaded;        /* 捕获中: 位掩码, 第 ei 个 engram 层的 host 节点 + 上传小核已进图 */
    double eg_job_s, eg_wait_s, eg_enter_s; uint32_t eg_n;   /* engram 取行的账(只记 n=1 的解码步; 预填块一轮几百 ms 会把平均污染):
                                 * 提交→完成 / 到 engram 层时真等 / 图路: launch → host 节点开跑(= L0 时长 + 发射与回调延迟) */
    double eg_t_launch;         /* 图路: 本步 cudaGraphLaunch 的时刻(host 节点回调里量 enter 用) */
    int ced_done;               /* v41_forward_body 出口: 本块在 CED 分界层收工(不出 logits), 调用方只推进位置 */
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
    ds4_gpu_tensor *mainh;      /* [mainh_cap][n_target][E] 主前向顺手取的 main_hidden, ★按绝对位置定格的环★(位置 p 在第 p % cap 格;
                                 * 2026-09-18 起, 以前是"只留本 chunk 末尾几行", 歇过的位置进不了三塔窗口, 见 ds4_gpu_v41.h hc_mean 注释)。
                                 * ★取的是目标层的**注意力输入**(engram 之后、层之前的 hc 四路均值), 不是层输出 ——
                                 * 取错不报错, 只是接受率掉到 1 附近(speed.md §7)。 */
    uint32_t mainh_cap, mainh_wrote;   /* mainh_wrote: 本次前向写了几行(v41_layer 置, v41_forward 收账用) */
    int64_t mainh_end;          /* 环里最新一行的绝对位置(-1 = 还没有); 有效行 = 以它结尾、往前连续的 mainh_n 行 */
    uint32_t mainh_n;           /* ★为什么要记连续段★: CED 跳过解码器段的预填块不写 mainh、投机回滚要作废被拒的行 ——
                                 * 只有"以 mainh_end 结尾的连续 mainh_n 行"是真的, 草稿器补窗口只许从这段里取 */
    int draft;                  /* 1 = 这个 state 是草稿塔的: 层权重走 weights.mtp.tower[il], 无 engram/无压缩 KV,
                                 * 注意力块内全可见, MoE 按盘上形态走逐专家 FP4 或塔 VQ blob。主状态恒 0。 */
    ds4_gpu_tensor *main_x;     /* 草稿态: [n][E] 块注意力的 KV 来源(main_norm(main_proj(main_hidden))) */
    const uint64_t *tower_exp_off[DS4_MTP_MAX_TOWERS];   /* 每塔 [3][n_expert] 专家张量偏移表(逐专家 fp4x32 形态) */
    const ds4_tensor *tower_exps_vq[DS4_MTP_MAX_TOWERS]; /* 每塔一个 VQ blob(100 GB 配方形态); 与上面二选一, 非空即走主干融合核 */
    uint32_t idx_topk;          /* 本 chunk 最近一个 indexer 源层产出的 topk 宽度 */
    uint32_t idx_ratio;         /* 产出上面那个 topk 的那一层的压缩比。注意力拿它按**每个 query 自己的
                                 * 绝对位置**算段长 —— 段长一旦吃了批级的量, 投机与纯解码就分段不同、
                                 * 输出分叉(见 ds4_gpu_v41.h 的 ratio 注释, 那里记了踩过的两个坑)。 */
    int16_t idx_owner;          /* 本 chunk 最近一个 indexer 源层(共享 topk 的来源), -1 无 */
    int16_t cand_owner;         /* 本 chunk 候选块掩码是否已产 */
} ds4_v41_state;

/* DS4_MTP_MAX_BLOCK 定义在 ds4_gpu_v41.h —— 核侧(按专家并集的分组表)与主机侧(host_ids 等小数组)
 * 共用一份, 见那里的注释。 */
#define DS4_V41_DRAFT_GROWS 8u   /* 草稿图按"补窗口几行"分档的上限: 1 + 块长(5)= 6 是常态, 歇过一两轮到 8; 再多走直发 */
#define DS4_V41_MKCACHE_SLOTS 256u   /* markov 偏置缓存槽数(见 ds4_v41_draft.mk_cache 的账) */

/* DSpark 草稿器(core_v41_draft.c)。ready=0 = 这份 GGUF 没带三塔或没带运行参数 ⇒ 调用方走纯单 token 解码。 */
typedef struct {
    ds4_v41_state st;          /* 草稿塔自己的一套缓冲(draft=1, 层号 = 塔号) */
    ds4_gpu_tensor *mainx_raw; /* main_proj 出口(norm 前) */
    ds4_gpu_tensor *mk_embed;  /* [cap][rank] 每位的 markov embed(confidence 要) */
    ds4_gpu_tensor *mk_cur;    /* [rank] 当前这一位的(给 markov_head 当输入) */
    ds4_gpu_tensor *mk_bias;   /* [V] 当前这一位的 logit 偏置 */
    /* markov 偏置缓存(2026-10-07): 按 token id 缓存偏置向量, 命中就不读 66 MB 的 markov_head(ds4_gpu_v41.h 接口说明)。
     * 槽数的账(0908 请求生成序列里 id 的复现率): 64 槽 50% / 128 槽 55% / **256 槽 59%** / 512 槽 60% ⇒ 256(132 MB)。 */
    ds4_gpu_tensor *mk_cache, *mk_cache_ids, *mk_cache_next, *mk_hit;   /* [槽][V] f32 / [槽] id(−1 = 空) / [1] 轮换计数 / [1] 本位命中槽 */
    ds4_gpu_tensor *conf_in, *conf;   /* [cap][E+rank] / [cap] */
    ds4_gpu_tensor *ids, *ids_next;   /* i32 [cap]: [0]=真 token, [1..block]=草稿; argmax 的落点 */
    ds4_gpu_tensor *h;         /* [cap][E] 出口 hc_pre 的结果(confidence 也吃它) */
    uint64_t *exp_off[DS4_MTP_MAX_TOWERS];
    /* 草稿器对齐放大器(mtp.md M6): xn += xn·(Bᵀ·A), 把草稿器的出口隐态掰到主模型的那个。
     * 两边过同一个出口头 ⇒ 隐态对齐就是 logits 对齐。K=0 表示没挂, 整条路是恒等的。 */
    ds4_gpu_tensor *ampA, *ampB, *ampT;
    uint32_t ampK;
    ds4_gpu_tensor *mkE_dev, *mkH_dev; uint32_t mk_rows;   /* 蒸馏的偏置表(core_v41_draft_amp.c): 非 NULL 时偏置路读它们而不是 GGUF 原件 */
    ds4_gpu_tensor *mainh_lin; /* [SWA][n_target][E] 从主态 mainh 环里按位置顺序取出来的连续行(main_proj 的一批输入) */
    int64_t win_end;           /* 三塔窗口里最新一行的绝对位置(-1 = 空)。每轮草稿前把 (win_end, pos_main] 补进窗口 —— 缺多少补多少,
                                * 超过 128 就整窗重建; 这样歇过几轮、走过 graph 步都不会断档 */
    uint32_t block, ready;
    int32_t host_ids[DS4_MTP_MAX_BLOCK + 1];
    float host_conf[DS4_MTP_MAX_BLOCK];
    /* ★草稿一轮整段进 CUDA graph(2026-09-22)★: 按"补窗口几行"(rows = 1 + 上一轮接受数, 歇过几轮就更多)各一张图, 图里: 零拷贝灌
     * 位置/token → 补窗口(ring_rows/main_proj/推三塔) → 块前向 → markov 逐位 → confidence → 零拷贝读回 ids/conf。一轮草稿约 200 发核,
     * 直发时发射间隙 + 两次同步吃掉 11.6 ms 里的一大半。只在位置 ≥ 窗宽时走图(块注意力的 pos0 烤进图, 只有 lo = pos0−window 那一支
     * 与 pos0 无关); rows 超出档、暂存换过指针、这个 rows 还没直发暖过 ⇒ 直发。输出与直发逐字节同(同一批核换个发法)。 */
    void *gexec[DS4_V41_DRAFT_GROWS + 1u]; uint64_t ggen[DS4_V41_DRAFT_GROWS + 1u]; uint32_t gwarm[DS4_V41_DRAFT_GROWS + 1u];
    int32_t *p_tok, *p_bpos, *p_wpos, *p_first, *p_ids; float *p_onehot, *p_conf;   /* pinned 槽(零拷贝小核直接读/写) */
    ds4_gpu_tensor *firstd;    /* 设备 int: ring_rows 的起始行(图开头由零拷贝灌) */
    int cap_mode, graph_off;   /* 捕获中(主机写张量改走零拷贝) / 捕获失败过(之后一律直发) */
    uint32_t gsteps, gcaps;    /* 走图的轮数 / 捕获次数 */
    uint32_t rounds;           /* 出过几轮草稿(直发 + 走图) */
    int dev_sample;            /* 主路在采样(2026-09-29): 草稿逐位不取 argmax 而按同一套温度/截断从塔的分布抽(硬币流 1), 验证核读塔 logits 做拒绝采样 */
    ds4_gpu_sample_params samp;
    int last_warm;             /* 上一轮是暖身轮(本请求第一轮 / 这个 rows 档第一次直发, 付了懒分配与首次触碰): 调度器不把它的墙钟记进稳态成本 */
    double h_slots, h_launch, t_launched;   /* 主机空隙分账(2026-10-07, prof): 草稿图发前写槽 / cudaGraphLaunch 的累计秒; 本轮草稿图发出的时刻 */
    int pending;               /* 草稿图已发还没等(v41_draft_launch 返回 1 之后、v41_draft_wait 之前) */
    double h_capture;          /* 草稿图捕获的主机耗时累计秒(次数 = gcaps) */
} ds4_v41_draft;

bool v41_draft_alloc(ds4_engine *e, ds4_v41_draft *dr);
void v41_draft_free(ds4_v41_draft *dr);
bool v41_draft_amp_load(ds4_v41_draft *dr, const char *path);    /* core_v41_draft_amp.c: 出口对齐边车(单文件) */
bool v41_draft_amp_mount(ds4_v41_draft *dr, const char *path);   /* 文件 → 上面那个; 目录 → 蒸馏的件(塔件 + 出口件 + 偏置表) */
/* 一轮草稿: 主模型最后处理的位置是 pos_main(它的 main_hidden 已在 main_st->mainh 环里), tok 是还没进主模型的下一个 token。
 * 先把三塔窗口补到 pos_main(按 dr->win_end 算差, 从环里取), 再出块。
 * 出 dr->host_ids[1..block](草稿 token)与 dr->host_conf[0..block-1](每位的条件接受概率)。 */
bool v41_draft_step(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int32_t tok, uint32_t pos_main);
/* 发/等两半(2026-10-07): 单请求路在上一轮收尾时先把下一轮的草稿图发出去, 再做 emit/簿记(GPU 不空等主机); 合批路仍用 v41_draft_step。
 * launch 返回 0 = 出不了草稿 / 1 = 图已发(要 wait) / 2 = 直发已同步跑完(wait 空操作)。 */
int v41_draft_launch(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int32_t tok, uint32_t pos_main);
bool v41_draft_wait(ds4_v41_draft *dr);
/* 置信调度(core_draft_sched.c): 这一轮该验几位。0 = 一位都不值得验; *value_out = 预测的"产出/成本"比值,
 * < 1 表示连草稿钱都赚不回来 —— 调用方拿它决定下一轮还出不出草稿。 */
/* ★在线校准(2026-09-28)★: 采样下第 j 位被接受的概率 = 目标分布给草稿 token 的概率, 比 conf 头学的"贪心是否同选"低。
 * 不写死折扣: 本请求出过草稿的轮里 首位命中数 / 首位预测概率之和 = ρ, 乘到每位 σ(conf) 上(贪心下 ρ≈1 自动退回原式)。
 * ★每一轮都要观测, 包括 k=0 的轮★(09-28 实撞: 只在 k≥1 时观测, 第一轮首位被拒 ⇒ ρ=0 ⇒ 之后永远 k=0 ⇒ 再没有观测, 死锁在 21 t/s):
 * k≥1 看首位是否接受, k=0 看真实吐出的 token 是否等于草稿首位 —— 两者期望都是 p(草稿首位), 同一枚硬币。
 * 只由 token 史决定 ⇒ 温 0 可复现(与不接墙钟的理由同)。没有样本时 ρ = 1。 */
/* ★成本也是本请求自量的(2026-09-28)★: ms[n]/cnt[n] = 单 token 步(n=1)与验证 1+k 行(n=2..)的墙钟, draft_ms/draft_n = 草稿一轮;
 * 调度器按它们做加权最小二乘, 没有编译期常量(以前的 0.273/1.067/0.303 过期过三次)。 */
typedef struct {
    double pred1, real1;
    double ms[DS4_MTP_MAX_BLOCK + 2u]; uint32_t cnt[DS4_MTP_MAX_BLOCK + 2u];
    double draft_ms; uint32_t draft_n;
} v41_sched;
uint32_t v41_draft_pick_k(const float *conf, uint32_t block, float *value_out, const v41_sched *s);
void v41_sched_observe(v41_sched *s, const float *conf, int hit);   /* 一轮完: 记首位的预测 σ(conf[0]) 与命中(接受 / 吐出 == 草稿) */
void v41_sched_cost(v41_sched *s, uint32_t n_rows, double ms);      /* 一发的墙钟: n_rows=1 单 token 步, ≥2 验证批 */
void v41_sched_draft_cost(v41_sched *s, double ms);                 /* 草稿一轮的墙钟 */

/* ---- 取下一个 token(core_v41_sample.c, 2026-09-28) ----
 * 设备槽每行 4 个 int32(ds4_gpu_v41_sample_tensor 的口径; argmax 路只用 [0]): pick 把 n 行槽拼成 want[i] =
 * "第 i 行之后该是哪个 token" —— 采样 + 有草稿: 接受 ⇒ 草稿 token, 拒绝 ⇒ 残差样本(必 ≠ 草稿); 否则 = [0]。
 * 于是投机的"接受最长前缀 = want[a] == batch[a+1]"那一行对贪心与采样是同一句。 */
void v41_sample_pick(const int32_t *slot, const int32_t *batch, uint32_t n, int dev_sample, int32_t *want);
/* 直发: 对 logits 第 row0..row0+n-1 行发采样核(dev_sample)或逐行 argmax, 同步, 读回, 拼 want[n]。batch = 这 n 行的输入 token(草稿判据), 单行给 NULL */
bool v41_device_next(ds4_v41_state *st, ds4_gpu_tensor *am, uint32_t row0, uint32_t n, const int32_t *batch, int32_t *want);
extern ds4_decode_sampling g_decode_sampling;   /* ds4_engine_set_decode_sampling 设的每请求采样面(core_v41_api.c) */
/* 生成段 token 史(只在读回 logits 的惩罚路上记): 复读惩罚只看它, 不看提示。brk = 断点表(DRY 开时才建)。 */
typedef struct { int32_t *tok; uint32_t n, cap; const uint8_t *brk; } v41_hist;
/* rowbuf == NULL: 设备路(采样核 / argmax), 只读回 16 B; 非 NULL: 惩罚路, 把第 row 行读回主机, 罚完交给 V4 路同一份采样器 */
bool v41_next_token(ds4_v41_state *st, ds4_gpu_tensor *am, uint32_t row, float *rowbuf, uint64_t *rng, v41_hist *h, int32_t *out);

/* engram 取行的 io_uring 通道(core_v41_ering.c; Linux 才有, open 返回 NULL 就退回线程池)。
 * 一轮 = submit 先灌满队列(不等) → wait 边收边续发直到收齐(队列常满, 落脚点走空闲栈)。direct = fd 是 O_DIRECT 的(对齐超集进落脚点再拷)。
 * ★一轮的请求数 > cap 时 wait 里有真正的等盘★: 那样的轮不许在主线程上收(GPU 会跟着空转), 见 core_v41_engram.c 的取行线程。 */
typedef struct { int fd; int direct; uint32_t len; uint64_t off; uint8_t *dst; } v41_ering_req;
typedef struct v41_ering v41_ering;
v41_ering *v41_ering_open(uint32_t cap);
void v41_ering_close(v41_ering *r);
uint32_t v41_ering_cap(const v41_ering *r);   /* 队列深度 = 一次能在飞的请求数(落脚点数) */
bool v41_ering_submit(v41_ering *r, const v41_ering_req *reqs, uint32_t n);
bool v41_ering_wait(v41_ering *r);
/* engram 取行的一个工作单元(core_v41_engram.c 填, core_v41_epool.c 的池跑) */
typedef struct { const ds4_engine *e; const ds4_v41_state *st; uint64_t u0, u1; int err; uint8_t *bounce; } v41_eworker;
void *v41_eworker_run(void *arg);
uint32_t v41_epool_threads(void);                       /* 池里有几个线程(懒起); 0 = 起不来, 调用方自己同步做 */
bool v41_epool_submit(v41_eworker *w, uint32_t njob);   /* 提交一轮, 不阻塞 */
void v41_epool_wait(void);                              /* 等这一轮干完 */

extern int g_ds4_v41_prof;   /* --v41-prof(core_v41_api.c) */
extern int g_ds4_v41_dspark;         /* 投机: 0 关 / 1 默认开 / 2 显式 --dspark(见 core_v41_api.c); 装载期也读它定三塔的优先级 */
extern int g_ds4_v41_emit_trace;     /* --emit-trace: 逐 token 打 [emit] 位置+id(同轨定位) */
extern uint32_t g_ds4_v41_block;     /* --dspark-block N: 钉死草稿块长(0=按元数据); 只作诊断 */
extern uint32_t g_ds4_v41_verify_k;  /* --dspark-verify N: 每轮验证几位(0 = 引擎默认) */
extern int g_ds4_v41_decoder_full;   /* --decoder-full: 关 CED, 每块跑满 40 层(精确路) */
extern const char *g_ds4_v41_amp_dir;   /* --zchain <dir>(V4.1 形态: amp_Lnn.bin 目录), core_engine_open.c 转来 */
extern const char *g_ds4_v41_draft_amp; /* --draft-amp <file>: 草稿器对齐边车(mtp.md M6), 只动草稿器 */
extern float g_ds4_v41_draft_amp_scale;  /* --draft-amp-scale β: 装载时 A *= β(诊断幅度用) */
extern float g_ds4_v41_amp_scale;       /* --zchain-scale β: 加载时 A *= β(默认 1.0) */
extern const char *g_ds4_v41_pt_dir;    /* --posttrain <dir>: 三文件部署第三件(后训练增益目录), 与 ② 的表逐元素相乘 */
extern ds4_v41_moe_hook_fn g_ds4_v41_hook;   /* 反修钩子(ds4_engine_v41_set_moe_hook), 无=NULL */
extern void *g_ds4_v41_hook_ud;
extern int g_ds4_v41_hook_layer;   /* 钩子只在这一层回调(-1=每层); 见 ds4_engine_v41_set_moe_hook_layer */
bool v41_attention_kv_only(ds4_engine *e, ds4_v41_state *st, uint32_t il);   /* CED 分界层: 只写全局 KV */
extern const ds4_model *g_ds4_v41_model;   /* 引擎打开时指向 e->model(core_engine_open.c): 路由偏置侧车要按层找 exp_probs_b 张量(core_v41_amp.c) */
bool v41_amp_load(ds4_v41_state *st, const char *dir, const char *pt_dir);   /* core_v41_amp.c: ②反修目录 + ③后训练目录 */
void v41_rb_clear_all(void);   /* 全卸路由偏置侧车(设备表是进程级的); 裸底座状态开张前调 */
void v41_amp_free(ds4_v41_state *st);
int v41_amp_hook(ds4_v41_state *st, uint32_t il);        /* 钩子回调(挂了才动): 同步 → x/y/sel/rw 下主机 → 回调; 0 继续 / 1 提前结束(置 stop_early) / <0 失败 */
bool v41_amp_apply(ds4_v41_state *st, uint32_t il);      /* y += x·(B·A) → bf16(挂了该层才动) */
ds4_gpu_tensor *v41_alloc(uint64_t bytes, bool *ok);   /* 小工具: 分配失败只置 ok=false, 调用方一路攒到最后再判 */
/* logits_rows: 0 = 按 cap 开(打分路, 每个位置都要); 生成路给 DS4_MTP_MAX_BLOCK+2(解码 1 行 / 验证批 ≤ block+1 行 / 预填块末位 1 行) */
bool v41_state_alloc(ds4_v41_state *st, uint32_t cap_tok, uint32_t ctx, uint32_t logits_rows);
/* iscore 长到够放 rows_score 行 × ng_need 组(f32), cand 长到 rows_cand 行的候选列表(每行 1 + topk_blocks 个 i32, C2)(够了就是 no-op) */
bool v41_index_scratch_prepare(ds4_v41_state *st, uint32_t ng_need, uint32_t rows_score, uint32_t rows_cand);
void v41_state_free(ds4_v41_state *st);
/* ---- 并发(2026-09-30, batch.md; core_v41_state.c / core_v41_multi.c / core_v41_req.c) ----
 * 请求态预填完 v41_state_shrink: 稠密段的行放掉、注意力侧缩到 rcap 行, 只剩 KV + 几十 MB; 批态(v41_batch_rows_alloc)持有稠密段的
 * R 行缓冲 + 反修表; v41_multi_step 一步喂 R 个请求态各 1 个 token(稠密段拼行一次发, 注意力段逐请求用视图发)。 */
bool v41_state_shrink(ds4_v41_state *st, uint32_t rcap);
bool v41_state_clone(ds4_v41_state *dst, const ds4_v41_state *src);
/* ★合批纯解码一步最多几行(2026-10-10)★: 与 DS4_V41_GEMV_MAX_TOK(8, 解码小批核与预填 GEMM 的分界, 单请求路/验证批/打分路都按它分岔)分开 ——
 * 合批行数超过 8 时稠密段各核按行数自己走预填路, 不改任何 ≤ 8 行的路径。批态 / 合批步 / 生成驱动的数组按它开。 */
#define DS4_V41_MULTI_MAX 16u   /* 刚收缩的请求态 → 深拷一份(同提示多份采样共享预填, core_v41_req.c) */
bool v41_batch_rows_alloc(ds4_v41_state *st, uint32_t cap);
typedef struct ds4_v41_batch {
    ds4_v41_state rows;         /* 稠密段的行缓冲(批态): x/hc/…/logits 按 cap 行; 没有 KV */
    ds4_gpu_tensor *am;         /* [cap][16 B] 采样核 / argmax 落点 */
    float *onehot;              /* [cap][HC] pre_mix 的 one-hot(第 0 路), 每步灌进批态 */
    uint32_t cap;
    uint32_t warm_R;            /* 直发成功过的最大总行数(≤ 8 那档): 稠密段按总行数懒长的暂存(hc mix 合一核的段和等)只增不减, 捕获只许 R ≤ 它 */
    ds4_engine *e;
    void *mg;                   /* 合批整步图的缓存(core_v41_mgraph.c 私有) */
} ds4_v41_batch;
bool v41_batch_alloc(ds4_v41_batch *b, uint32_t cap);
void v41_batch_free(ds4_v41_batch *b);
typedef struct { ds4_gpu_tensor *tok, *pos, *xn, *erows, *qrn, *q, *kvn, *o; } v41_rowview;   /* 请求态指进批态行的八个视图 */
bool v41_attach(ds4_v41_state *m, const ds4_v41_state *B, uint32_t r, uint32_t rows, v41_rowview *v);   /* 挂视图 + 本步 n/pos0(core_v41_multi.c) */
void v41_detach(ds4_v41_state *m, v41_rowview *v);
bool v41_multi_body(ds4_engine *e, ds4_v41_batch *b, ds4_v41_state **m, const uint32_t *r0, const uint32_t *nr, uint32_t nm, uint32_t R);   /* embed → 40 层 → logits, 直发/捕获共用 */
void v41_multi_advance(ds4_v41_state *st, uint32_t n);   /* 一步之后的账: mainh 连续段 / 位置 / 暖身计数 */
/* 一步(直发): nm 路各 rows[i] 行(NULL = 各 1 行), tok 是拼接的输入 token(第 i 路的行从 Σrows[<i] 起); 前向 + 各路位置推进; logits 在 b->rows.logits 的对应行 */
bool v41_multi_step(ds4_engine *e, ds4_v41_batch *b, ds4_v41_state **m, const int32_t *tok, const uint32_t *rows, uint32_t nm);
/* 合批整步图(core_v41_mgraph.c): ready = 这个 (成员集, 各路行数, 各路位置桶) 有图或能捕(各路这个行数直发暖过); round = 发图 + 等 + 各路取 want(设备 argmax/采样核)
 * + 位置推进(快照记账在里面, 回滚仍是 v41_spec_rollback)。返回 false = 没走图(状态没动), 调用方走直发。 */
bool v41_multi_graph_ready(ds4_v41_batch *b, ds4_v41_state **m, const uint32_t *nr, uint32_t nm);
bool v41_multi_graph_round(ds4_engine *e, ds4_v41_batch *b, ds4_v41_state **m, const int32_t *tok, const uint32_t *nr, uint32_t nm, int32_t *want);
void v41_mgraph_free(ds4_v41_batch *b);
bool v41_multi_pick(ds4_v41_batch *b, ds4_v41_state *m, uint32_t row, float *rowbuf, uint64_t *rng, v41_hist *h, int32_t *out);   /* 第 row 行按 m 的采样面取 token */
bool v41_multi_pick_rows(ds4_v41_batch *b, ds4_v41_state *m, uint32_t row0, uint32_t rows, const int32_t *batch, int32_t *want);   /* 验证批 rows 行一次取(设备路) */
/* 一块预填(单请求路 generate_argmax 与并发路 ds4_v41_req_prefill_step 共用: 分块 / CED / 窗口尾规则只写这一处, core_v41_forward.c) */
bool v41_prefill_chunk(ds4_engine *e, ds4_v41_state *st, const int32_t *prompt, uint32_t np, uint32_t *c0, uint32_t cap);
bool v41_hc_half(ds4_engine *e, ds4_v41_state *st, const ds4_layer_weights *l, bool attn_half);   /* core_v41_forward.c: 半层入口三件 */
bool v41_moe(const ds4_model *m, const ds4_layer_weights *l, ds4_v41_state *st, uint32_t il);      /* core_v41_forward.c: MoE(含反修应用) */
bool v41_engram_rows(ds4_engine *e, ds4_v41_state *st, uint32_t il);    /* core_v41_engram.c: 收行 → 上传 → 解码行进 st->erows(按请求) */
bool v41_engram_apply(ds4_engine *e, ds4_v41_state *st, uint32_t il);   /* core_v41_engram.c: erows → wkv → 门 → 就地改 st->hc(可按批) */
extern int g_ds4_v41_chunk;   /* --v41-chunk(core_v41_api.c): 预填分块, 0 = DS4_V41_CHUNK */
extern int g_ds4_v41_lanes;   /* 合批缓存段按路分流(core_v41_multi.c); --no-lanes 关(A/B 用) */
bool v41_forward(ds4_engine *e, ds4_v41_state *st, const int32_t *ids, uint32_t n);   /* 追加 n 个 token, st->logits[n][V] */
/* 前向的主体(embed → 40 层 → 出口 head), 不含输入上传/engram 预取/末尾同步/位置推进 —— graph 捕获与直发共用这一段。
 * 直发 = v41_forward 全套; 捕获 = core_decode_graph.c 自己做输入(pinned 槽)与收尾。 */
bool v41_forward_body(ds4_engine *e, ds4_v41_state *st);
/* 解码整步 graph(core_decode_graph.c): ready = 这一步能走图(标志开、n=1 主路、暖过、无探针/钩子);
 * step = 走图解一步: 写槽 → 发 → 等 → 出下一个 token(设备 argmax), 主机状态照直发那样推进。 */
/* engram 在 graph 路的三步(core_v41_engram.c): 发图前 arm(写本步序号) → 发图后 serve(主机收 pread、逐层置位) → sync 后 err */
bool v41_engram_graph_arm(ds4_v41_state *st);
bool v41_engram_graph_serve(ds4_v41_state *st);
int v41_engram_graph_err(const ds4_v41_state *st);
bool v41_graph_ready(const ds4_v41_state *st);
bool v41_graph_step(ds4_engine *e, ds4_v41_state *st, int32_t tok, int32_t *next_tok);
/* step 的两半: launch 发出这一步的图后立刻返回(调用方趁 GPU 跑着去 emit 当前 token), wait 等它跑完取下一个 token。
 * after_draft = 这一步前面刚跑过一轮草稿(调度器判 k=0): 照样走图, 但不进"稳态 ms/步"的账(那本账只认纯解码步)。 */
bool v41_graph_launch(ds4_engine *e, ds4_v41_state *st, int32_t tok, int after_draft);
bool v41_graph_wait(ds4_engine *e, ds4_v41_state *st, int32_t *next_tok);
/* ★投机验证批的整步图(2026-09-22)★: n = 1+k 行按 n 各一张图, 快照(窗口环按层 / 压缩器余行在追加核里)进图, 主机只记账;
 * ready = 这个 n 直发暖过 + 图开着; launch 返回 false = 这一批走不了图(捕获失败/形状不合), 调用方按直发跑, 状态没动。
 * wait 出 n 行的设备 argmax(next[i] = 第 i 行的贪心 token), 位置推进 n; 回滚仍是 v41_spec_rollback(host 直发)。 */
bool v41_graph_batch_ready(const ds4_v41_state *st, uint32_t n);
bool v41_graph_batch_launch(ds4_engine *e, ds4_v41_state *st, const int32_t *ids, uint32_t n);
bool v41_graph_batch_wait(ds4_engine *e, ds4_v41_state *st, int32_t *next);
void v41_graph_batch_host(const ds4_v41_state *st, double *prep, double *launch, uint32_t *n);   /* 验证批发图的主机账(prof 分账行) */
bool v41_graph_allowed(const ds4_v41_state *st);   /* 主路允许走图(不要求 n=1 暖过): 草稿图的门 */
void v41_graph_batch_prep(const ds4_v41_state *st, double *t_begin, double *t_engram, double *t_check, double *t_arm);   /* 起手四段(累计秒) */
void v41_graph_capture_cost(const ds4_v41_state *st, double *t_capture, uint32_t *n_capture);   /* 图捕获的主机耗时(累计秒)/次数 */
void v41_graph_free(ds4_v41_state *st);
extern int g_ds4_v41_graph;   /* --no-graph 清零(core_v41_api.c); 默认开 */
bool v41_spec_snapshot(ds4_v41_state *st, uint32_t n);          /* 验证批前: 备份这 n 位会盖掉的环格 + 压缩器余行 */
bool v41_spec_rollback(ds4_v41_state *st, uint32_t keep);       /* 验证批后: 只保留前 keep 个位置, 其余撤销 */
bool v41_layer(ds4_engine *e, ds4_v41_state *st, uint32_t il);        /* 一层(主干层或草稿塔, 看 st->draft) */
bool v41_attention(ds4_engine *e, ds4_v41_state *st, uint32_t il);   /* core_v41_attn.c: st->xn → st->attn_out, 更新各缓存 = 下面三段拼起来 */
bool v41_attn_in(ds4_engine *e, ds4_v41_state *st, uint32_t il);     /* 投影进(按行): xn → qr/qrn/q(rope) + kv/kvn; 合批时在批态上一次发 */
bool v41_attn_cache(ds4_engine *e, ds4_v41_state *st, uint32_t il);  /* 缓存段(按请求): kvn 进窗口 / 压缩源 / 索引源 / 稀疏注意力 → o / 环提交 */
bool v41_attn_out(ds4_engine *e, ds4_v41_state *st, uint32_t il);    /* 投影出(按行): o 逆 rope → wo_a → wo_b → attn_out */
/* 稠密投影按【盘上登记类型】分发(FP8 原件精度 / q4_K 100GB 配方 / fp4x32) —— 实现与所以然在 core_v41_attn.c。
 * ★骨架的每一处矩阵乘都要走这里★: 直接调某个具体格式的核, 换配方时就是拿错解码器读对的字节, 不报错只出假数。 */
bool v41_tproj(const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *w, uint64_t in_dim, uint64_t out_dim,
               const ds4_gpu_tensor *x, uint32_t n, int round_out);
bool v41_tproj_grouped(const ds4_model *m, ds4_gpu_tensor *low, const ds4_tensor *w, uint32_t n_groups,
                       uint64_t group_dim, uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n, int round_out);
bool v41_embed(const ds4_model *m, ds4_gpu_tensor *out, const ds4_gpu_tensor *tok, const ds4_tensor *w,
               uint64_t n_vocab, uint32_t n, uint64_t dim);
bool v41_draft_push_main(ds4_engine *e, ds4_v41_state *st, uint32_t rows);   /* core_v41_attn.c: 已确认位置的 main_x 进各塔窗口 */
bool v41_engram(ds4_engine *e, ds4_v41_state *st, uint32_t il);      /* core_v41_engram.c: 就地改 st->hc(engram 层才调) */
bool v41_engram_prefetch(ds4_engine *e, ds4_v41_state *st);          /* 前向开头: 所有 engram 层的行一次性后台并行 pread */
void v41_engram_close(ds4_v41_state *st);

#endif /* !DS4_NO_GPU */
#endif
