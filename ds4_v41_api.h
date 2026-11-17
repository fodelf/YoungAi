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
void ds4_engine_v41_set_replay(int k);          /* --decoder-replay K: 预填最后一块的解码器段只回放末尾 K 行(默认 140 = 官方 keep_tokens; 0 = 整块) */
void ds4_engine_v41_set_dspark(int mode);       /* 投机: 0 关 / 1 默认开(采样请求自动走纯解码) / 2 显式 --dspark(与采样同开硬拒); 见 core_v41_api.c */
void ds4_engine_v41_set_graph(int on);          /* --no-graph 关解码整步 CUDA graph(默认开; 只作 A/B 与定位, 输出逐字节同) */
void ds4_engine_v41_set_vq_group(int on);       /* --no-vq-group: 验证批/草稿塔的 VQ 专家核回逐对形态(默认分组核; 只作 A/B, 输出逐字节同) */
void ds4_engine_v41_set_emit_trace(int on);     /* --emit-trace: 逐 token 打 [emit] 位置+id(同轨定位) */
void ds4_engine_v41_set_block(unsigned b);      /* --dspark-block N: 钉死草稿块长(0=按元数据); 只作诊断 */
void ds4_engine_v41_set_verify_k(unsigned k);   /* --dspark-verify N: 每轮验证几位(0=默认 3); 诊断与逐档量字节账用 */
void ds4_engine_v41_set_amp_dir(const char *dir);   /* --zchain <dir>: V4.1 反修放大器目录(amp_Lnn.bin), 每层 MoE 出口 y += x·(B·A) */
void ds4_engine_v41_set_amp_scale(float s);         /* --zchain-scale β: 加载时把 A 乘 β(修正整体缩到 β 倍); ≤0 = 1.0 */
void ds4_engine_v41_set_posttrain_dir(const char *dir);   /* --posttrain <dir>: 三文件部署的第三件(后训练增益), 与 --zchain 的表逐元素相乘 */
/* 服务里热切换 ②③(目录给 NULL/空 = 不挂): 先用试探状态真装一遍, 装不上就保持原样返回 -1(err 写原因)。
 * ★调用时不许有活着的 V4.1 状态★(进程级的路由偏置/增益表会被换掉), 见 core_v41_state.c */
int ds4_engine_v41_switch_plugins(const char *amp_dir, const char *pt_dir, char *err, size_t errn);
/* 当前挂着的 ②③ 目录(没挂 = NULL), 给服务端报给页面 */
const char *ds4_engine_v41_amp_dir(void);
const char *ds4_engine_v41_posttrain_dir(void);
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
/* n_prompt(2026-09-29): 前 n_prompt 个 id 按分块预填(提示段不逐位取料, 14k 提示逐位要十几分钟), 从第 n_prompt 个起一位一块; 0 = 从头逐位。
 * 顺带出★接受率陪审团★: 采样面温度 > 0 时, 逐位置按同一套温度算 主模型分布 p 与 草稿塔首位分布 q, 报 平均 p(argmax q)(点质量草稿的期望
 * 首位接受率)与 平均 Σmin(p,q)(草稿按分布抽的期望首位接受率) —— 同一段文本上的解析量, 不吃采样噪声(在线 t/s 每趟是另一篇文本, seed 间 ±3 t/s)。 */
int ds4_engine_v41_dspark_capture(ds4_engine *e, const int *ids, int n_ids, const char *out_path, int n_prompt);

/* --draft-amp <file>: 挂草稿器对齐边车(gguf-tools/amp/dspark_align 的产物)。只改草稿器的出口隐态,
 * 主模型一个字节不碰 —— 所以它**不可能**动五指标, 只动接受率。传 NULL/不传 = 不挂, 整条路恒等。 */
void ds4_engine_v41_set_draft_amp(const char *path);
void ds4_engine_v41_set_draft_amp_scale(float s);   /* --draft-amp-scale β: 诊断修正幅度 */
/* ---- 视觉(2026-10-13; src/core/core_v41_vision.c / core_v41_image.c; 官方 image_processor.py + vision.py) ----
 * 用法: 提示里每张图放一个占位 token(ds4_engine_v41_image_token_id, 渲染文本里是 <｜deepseek_image｜>), 分词后调
 * ds4_v41_images_prepare 把占位展开成图片 span(每张 ≤ max_image_tokens 位)并做好预处理, 拿展开后的 token 去开请求 / 生成,
 * 图片对象挂上(ds4_v41_req_set_images / generate_argmax 的 imgs), 预填前引擎跑一次视觉塔。
 * 没带视觉塔的 GGUF(deepseek4.vision.block_count 缺/0): has_vision = 0, prepare 直接报错 —— 不假装看见。 */
typedef struct { const unsigned char *data; size_t len; } ds4_v41_image_src;   /* 图片原始字节(jpeg/png/gif/bmp 等 stb_image 认的格式) */
struct ds4_v41_images;
struct ds4_v41_req;   /* 完整声明在下面的并发段; 这里先引名, 免得原型参数里的 struct 变成只在原型作用域里的新类型 */
int ds4_engine_v41_has_vision(ds4_engine *e);
int ds4_engine_v41_image_token_id(ds4_engine *e);   /* -1 = 没视觉塔 */
struct ds4_v41_images *ds4_v41_images_prepare(ds4_engine *e, const int *prompt, int n_prompt, const ds4_v41_image_src *imgs, int n_imgs,
                                              int **out_tokens, int *out_n, char *err, size_t errn);   /* out_tokens 是 malloc 的, 调用方 free */
void ds4_v41_images_free(struct ds4_v41_images *imgs);
uint64_t ds4_v41_images_bytes(const struct ds4_v41_images *imgs);   /* 预填期额外峰值(服务端准入账) */
int ds4_v41_req_set_images(struct ds4_v41_req *r, struct ds4_v41_images *imgs);   /* 开请求之后、第一块预填之前调; 1 = 挂上 */
int ds4_engine_v41_vision_golden(ds4_engine *e, const char *image_path, const char *golden_path);   /* --vision-golden: 对官方金标四道比(core_v41_vision_gate.c); 0 = 过 */

/* 没有 ctx 参数: V4.1 的上下文来自模型元数据(ds4_engine_v41_ctx), 调用方定不了边界, 只定生成上限 n_predict
 * (INT_MAX = 不设上限, 生成到 EOS 或上下文边界)。imgs = 提示里图片的展开结果(ds4_v41_images_prepare), 无图 NULL。 */
int ds4_engine_v41_generate_argmax(ds4_engine *e, const int *prompt, int n_prompt, struct ds4_v41_images *imgs, int n_predict,
                                   ds4_v41_emit_fn emit, void *ud);
/* --ptrain <配置>(2026-10-01, 后训练 ③ 第八版 = 上下文蒸馏; src/core/core_ptrain*.c): 读复盘块 + 自出问答 → 教师 top-K →
 * 反传训练 ③ 的低秩放大器 → 落 amp_Lnn.bin + base.fnv(引擎 --posttrain 直接挂)。配置是 key=value 文本, 字段见 core_ptrain_data.c。
 * 只接 --zchain ②, 不接 --posttrain(起点恒为 ①+②)。返回 0 = 落盘成功。 */
int ds4_engine_ptrain(ds4_engine *e, const char *spec);
/* --restore-b <配置>(2026-10-28, 反修段 B 的还原训练; src/core/core_restoreb.c): 靶 = 官方 API 逐位 top-20, 只在量化丢掉能量的缺口位
 * (第 0 步逐位 KL 的 rb_gap 分位以上)出梯度; 起点 rb_init 可扩秩。借 ③ 的装料/反传/落盘件, ③ 本身不动。返回 0 = 落盘成功。 */
int ds4_engine_restoreb(ds4_engine *e, const char *spec);
/* --draft-train <配置>(2026-10-07, 草稿器蒸馏; src/core/core_draft_kd*.c): 让 DSpark 三塔改盯部署底座 —— 教师强制走底座自己采样的真实请求续写,
 * 块内各位对底座同位置分布做 KL, 反传训三塔 MoE 出口的低秩件 + 出口对齐件; 落 tower_Tn.bin + exit.dspa + base.fnv, 引擎 --draft-amp <目录> 直接挂。
 * 底座/塔原权重/头全冻结, 验证侧不动 ⇒ 只动接受率。配置是 key=value 文本(字段见 core_draft_kd.c)。返回 0 = 落盘成功。 */
int ds4_engine_draft_train(ds4_engine *e, const char *spec);
/* 解码采样(2026-09-21, 113-1.md §4): 上面那条生成路的名字里的 argmax 是历史, 采样由这个设置面决定。
 * temperature ≤ 0(默认) = 裸 argmax: 图末尾的设备 argmax, 一个字节不变(门 = 温 0 输出逐字节回归)。
 * > 0 = ★设备采样核★(2026-09-28, src/cuda/cuda_v41_sample.inc.cu): 温度 / top-k / top-p / min-p 与 V4 路采样器 ds4_sample_logits
 * 同一套语义, 在图末尾替掉 argmax, 主机只读 16 B; 投机照走(核里做拒绝采样: 接受 ⇔ 均匀数 < 目标分布给草稿的概率, 拒绝从残差抽,
 * 吐出 token 的边缘分布 = 纯解码采样的分布)。只按模型自己的分布抽, 不动 logits(铁律"引擎不得改模型输出")。
 * seed 0 = 按时钟(与 V4 CLI 同规则)。NULL = 回到裸 argmax。同 seed 下投机与纯解码吐的具体 token 不同(硬币不同), 分布相同。
 * 惩罚(core_decode_penalty.c, 全部默认 0 = 不进那段代码): freq/presence = OpenAI 频率/出现惩罚(与 V4 路同式);
 * dry_multiplier > 0 开 DRY 序列复读惩罚(base 1.75 / allowed_length 2 是 llama.cpp 默认): 这是温 0 下也能挡死循环的唯一手段。
 * 任一惩罚非零时, 温 0 也走"读回 logits 行 → 罚 → argmax"这条路(argmax 由同一份采样器在温 0 时给出); 惩罚要按 token 史改 logits,
 * 投机不接它: 显式 --dspark + 惩罚 直接拒, 投机只是默认开着时该请求走纯解码并打日志。 */
/* greedy_fn: 工具语法位贪心(2026-10-11)。取每个 token 之前问一次 greedy_fn(greedy_ud, ahead, n) —— "调用方已收到的文本后面再接
 * ahead[0..n) 这几个 token, 下一位是不是工具调用的协议语法(DSML 标签 / 参数头 / JSON 标点)", 非 0 = 这一位取 argmax, 0 = 照采样。
 * 跟着采样面走, 单请求路与并发请求态读的是同一份, 不分路。只在采样路生效(温 0 本来就是 argmax); NULL = 不问。
 * 规则与实现见 core_v41_sample.c v41_sample_pick。 */
typedef int (*ds4_greedy_fn)(void *ud, const int32_t *ahead, uint32_t n);
/* force_fn: 调用方指定下一位(2026-10-17, 思考段死环强制收口)。问法与 greedy_fn 同(已收到的 + ahead[0..n) 之后), 返回 ≥ 0 = 这一位就取它,
 * 不看 logits; < 0 = 照常(先 greedy_fn 再采样)。只在"前面的草稿全被接受"的行上问, 强制的那位就是这一轮的落点(与草稿不同则后面的行作废)。
 * 引擎自己不判任何事 —— 什么时候强制、强制成什么, 全在调用方(服务端 server_think_loop.c); NULL = 不问。 */
typedef int32_t (*ds4_force_fn)(void *ud, const int32_t *ahead, uint32_t n);
typedef struct {
    float temperature, top_p, min_p; int top_k; uint64_t seed;
    float freq_penalty, presence_penalty;
    float dry_multiplier, dry_base; int dry_allowed_length;
    ds4_greedy_fn greedy_fn; void *greedy_ud;
    ds4_force_fn force_fn; void *force_ud;
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
/* KV 前缀复用(2026-10-11, core_v41_kvreuse.c): 默认开; --no-kv-reuse 关(只作 A/B 与精确门)。
 * last_reused = 上一趟 generate 从检查点接上的 token 数(0 = 整段预填), 服务端拿它报 usage 的 cached_tokens 与监控页。 */
void ds4_engine_v41_set_kv_reuse(int on);
int ds4_engine_v41_last_reused(void);
/* 切换插件作废掉的那个检查点的提示 token(malloc 的, 调用方 free; 没有 / 带图片 = NULL, n=0)。服务端切完空闲时拿它用新插件重新预填
 * 一遍(ds4_engine_v41_generate_argmax n_predict=0 就只预填 + 存检查点), 用户切完再发消息直接接上。取一次就清。 */
int32_t *ds4_engine_v41_kv_stale_prompt(uint32_t *n);
/* 检查点磁盘离载(2026-11-15, core_v41_ckpt_disk.c): 服务端 --kv-disk-dir 开。dir=NULL/不开 = 旧行为。
 * budget_mb 0 = 默认 4096; 文件按 tok 序列哈希 + LRU 预算管理。开/关是引擎级(进程内只有一份索引)。
 * open 返回 0 成功; 失败写 err。offload = 换 agent/会话边界时把两槽整份写到磁盘、再放显存 —— 回到旧对话时
 * 引擎自己从磁盘装进请求态(restore 优先 GPU 槽，槽没了就查磁盘)。 */
int ds4_engine_v41_kv_disk_open(const char *dir, uint64_t budget_mb, char *err, size_t errn);
void ds4_engine_v41_kv_disk_close(void);
void ds4_engine_v41_kv_disk_clear(void);   /* 全清(切插件同款) */
void ds4_engine_v41_kv_disk_conversation_set(const char *conv);   /* 请求/agent 会话号, CK store 尾 trailer 记 */
int ds4_engine_v41_kv_disk_erase_conversation(const char *conv);  /* agent 关闭: 只删它归属的 CK, 别的 agent 留 */
void ds4_engine_v41_kv_offload(void);
/* 上一趟 generate 逐步的 logits 指纹(只有主机惩罚路记; 见 core_v41_sample.c), 返回步数; 服务端 --trace 时写进 trace 给精确门比 */
int ds4_engine_v41_logit_sums(const double **out);

/* ---- 并发(2026-09-30, batch.md; 实现 src/core/core_v41_req.c): 请求态 + 批态 + 多状态一步 ----
 * 服务端调度器(server_sched_v41.c)拿这几个出口把 N 条请求合成一次前向: 每条请求 open 后一块一块 prefill_step(块间调度器去发心跳、
 * 探客户端、让别的请求解码), 预填完的请求每步与其他请求一起 multi_step(各喂 1 个 token, 稠密段拼行一次发, 权重每步只读一遍)。
 * 没有会话 / KV 复用 / 投机(第二期)。温 0 下每路输出与单请求路逐字节同(同一份预填与取 token 代码)。
 * 出错会怎样: multi_step 非 0 = 这一步的所有请求都没推进(状态没动), 调用方按错误收尾, 不重试。 */
struct ds4_v41_req;
struct ds4_v41_batch;
struct ds4_v41_batch *ds4_v41_batch_open(ds4_engine *e, int cap);   /* cap = 一步最多合几行(解码小批核路上限 8) */
void ds4_v41_batch_close(struct ds4_v41_batch *b);
struct ds4_v41_req *ds4_v41_req_open(ds4_engine *e, const int *prompt, int n_prompt, int n_predict, const ds4_decode_sampling *sp);
int ds4_v41_req_prefill_step(struct ds4_v41_req *r);   /* 跑一块预填: 1 = 预填完(首个 token 在 req_next), 0 = 还有块, -1 = 失败 */
/* 共享预填(2026-10-09): 在 src 预填完之前挂一个同提示、自己采样面的分身; src 最后一块出完 logits 时分身按自己的采样面取首个 token,
 * 收缩后深拷 src 的 KV ⇒ 分身直接是预填完的请求态(不要再 prefill_step)。src 预填失败或先关, 分身停在没预填完(multi_step 会拒)。
 * 与各自预填逐字节同(预填是确定的), 省掉 G−1 遍预填。分身各自 close。 */
struct ds4_v41_req *ds4_v41_req_fork(struct ds4_v41_req *src, const ds4_decode_sampling *sp);
void ds4_v41_req_progress(const struct ds4_v41_req *r, int *c0, int *np);
int ds4_v41_req_next(const struct ds4_v41_req *r);    /* 还没进模型的下一个 token(= 刚生成的那个) */
int ds4_v41_req_pos(const struct ds4_v41_req *r);     /* 已进缓存的位置数 */
int ds4_v41_req_room(const struct ds4_v41_req *r);    /* 还能再进几个位置(0 = 上下文满, 别再喂) */
int ds4_v41_multi_step(struct ds4_v41_batch *b, struct ds4_v41_req **r, int n);   /* 纯解码一步: 各请求喂自己的 next, 出各自的新 next(outq 1 个); 非 0 失败 */
/* 投机一轮(2026-09-30 傍晚): 各请求各出草稿, 验证行拼进一次前向, 各自接受/回滚; 之后 req_take 拿这一轮吐出的 token(接受的草稿 + 新 next, 按序 emit)。
 * 没三塔 / 惩罚路 / 引擎投机关 = 退成纯解码一步(outq 1 个)。上下文不够放一个验证批的请求这一轮也只走 1 行。 */
int ds4_v41_multi_round(struct ds4_v41_batch *b, struct ds4_v41_req **r, int n);
int ds4_v41_req_take(struct ds4_v41_req *r, int *out, int max);   /* 取这一轮吐出的 token(最后一个 = 新 next), 取过即清; 返回个数 */
void ds4_v41_req_spec_stats(const struct ds4_v41_req *r, int *rounds, int *offered, int *accepted);   /* 投机轮数 / 出过的草稿位 / 接受的草稿位 */
/* 单请求路(ds4_engine_v41_generate_argmax)上一趟的同一组投机账(服务监控页用); 每次 generate 进门清零, 没投机全 0 */
void ds4_engine_v41_last_spec_stats(int *rounds, int *offered, int *accepted);
int ds4_engine_v41_dspark(void);   /* 投机开关现值(ds4_engine_v41_set_dspark 设的那个): 0 关 / 1 默认开 / 2 显式开 */
void ds4_v41_req_close(struct ds4_v41_req *r);
/* 合批生成驱动(2026-10-09, core_v41_gen.c): 作业 = 已渲染的提示 ids + 自己的采样面 + 上限 token 数(含 EOS 那一位)。至多 bcap 路一起解码;
 * refill = 0 整组跑完才开下一组(训练器探针/采样, 与 10-06 的组划分逐字节同), 1 = 哪路写完立刻补下一个作业(出题)。相邻同提示作业共享预填。
 * done 每个作业回调一次(text 只在回调内有效; eos = 1 以 EOS 收口, 0 = 到上限或上下文满)。返回 false = 中途失败, 没回调的作业没写完。
 * tok[0..ntok) = 采出来的 token id 原样(含收口的 EOS), 同样只在回调内有效 —— 训练器步 2 要在学生自己采的那串 id 上算损失,
 * 文本再分词不保证逐 id 回得去(一个采出来的 token 解码后可能被规范分词拆成两个), 位置就对不上。 */
typedef struct { const int *ids; uint32_t len; uint32_t max_tok; ds4_decode_sampling sp; } ds4_v41_gen_job;
typedef void (*ds4_v41_gen_done_fn)(void *ud, uint32_t job, const char *text, size_t len, const int *tok, int eos, uint32_t ntok);
bool ds4_v41_gen_run(ds4_engine *e, const ds4_v41_gen_job *jobs, uint32_t n, uint32_t bcap, int refill, ds4_v41_gen_done_fn done, void *ud);
void ds4_engine_v41_set_lanes(int on);              /* --no-lanes: 合批的缓存段不按路分流(默认分; 只作 A/B, 输出逐字节同) */
uint64_t ds4_v41_req_prefill_bytes(int n_prompt);   /* 预填期设备峰值字节(行缓冲 + KV + 索引草稿), 服务端准入用 */
uint64_t ds4_v41_req_resident_bytes(void);          /* 预填完收缩后的常驻字节(KV 按模型上下文) */

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
/* --score-ids 的出口 logits 回调(2026-10-21, 反修端到端择优): 每块前向完、logits 读回主机后回调一次, 行 [row0, row0+nrow) 的
 * 全词表 f32 logits(未 softmax, 行主序 [nrow][vocab]); 回调返回非 0 = 这趟停车。设了回调时 out_path 可以给 NULL(不落文件);
 * 回调只在这一块的生命期内有效, 要留的数据自己拷。CED 块 / 钩子早停的块没有 logits, 不回调。NULL = 不回调。 */
typedef int (*ds4_v41_logits_fn)(void *ud, int row0, int nrow, const float *logits, int vocab);
void ds4_engine_v41_set_logits_hook(ds4_v41_logits_fn fn, void *ud);

#ifdef __cplusplus
}
#endif
#endif
