/* core_ptrain.h — 后训练 ③(上下文蒸馏 + 答案 NLL, 2026-10-01 起; 10-10 收口成一份料一条命令)的内部声明。只被 core_ptrain*.c 包含。
 *
 * 一句话: 料是 jsonl, 一行一题 {"messages":[user, assistant], "context"?: 材料} 或一段原文 {"text": "..."}(与 unsloth 的两种料一一对应:
 *   conversational = messages, continued pretraining = text; 10-10 对齐, 训练器不再有别的入口, 文档要不要先出成问答由用户在料上决定)。
 *   带 context 的题: 教师 = 部署模型(①+②)把材料放进上下文答题, 学生 = 同一模型 + ③ 不看材料答同一题, 在答案位上让学生分布逼近教师(余量桶 KL);
 *   不带 context 的题: 目标 = 答案 token 本身(one-hot, 同一个 KL 核算出来就是交叉熵)。
 *   text 行: 不走聊天模板, 序列 = BOS + 原文 + EOS, 每个 token 都是目标(one-hot); 超 maxlen 的按 token 切成连续几段各自成样本(unsloth 是截断丢尾, 这里不丢)。
 *   另一份固定的保持料(hold= 的 jsonl, 通用题 + 部署态自己的回答): 教师 = 不看任何材料的部署态自己, 把 ③ 在通用问题上钉住(10-01 实撞: 没有它局部性崩)。
 * 梯度反传到 ③ 的低秩放大器(每层 MoE 输出 y += xn·B·A, 与 ② 同构, 引擎 --posttrain 直接挂)。
 *
 * ★一轮两步(10-11, docs/posttrain.md)★: 上面是步 1(记忆: 在给定答案上逼近教师/金标)。sample_n > 0 时每轮再走步 2(泛化, core_ptrain_onpolicy.c):
 *   学生(挂当前 ③、不看材料)对本轮每道训练题自己采 sample_n 份答案 → 教师(看材料的部署态; 不带材料的题 = 把这条问答当示范放进上下文)
 *   给学生采出来的那串 token 逐位打 top-K → 同一个 KL 核在学生自己的分布上训。为什么: 只在一种写法上训, 梯度走的是"看到这串补那串"的捷径,
 *   换种问法/换种语言就不点火; 学生自己的采样就是"另一种写法", 教师告诉它每一位该怎么走(SDFT 2601.19897 / GRIN 2608.25243)。
 *   判决多一个数: eval= 给的整份 jsonl 全部当留出(换了形式的题), 它的损失 = 泛化; 选轮有它就按它选。
 *
 * ★10-10 砍掉的东西(用户: "本末倒置, 往定制化场景跑偏了")★: 账本硬目标 / 自采样 REINFORCE / 锚 KL / 锚行合并 / 采样器 / 自定义探针 / 部署态答案缓存 /
 * 清单 .list + .qa 私有格式。训练目标只剩上面两种, 判决只剩留出损失 + wt2 门(脚本)。
 *
 * 为什么不用解算器: 闭式解只能把"答案该往哪挪"线性化到一层的输出上, 答案位的激活不像拟合行就不点火(10-01 六趟复盘 ③ 的结构病);
 * 文献里文档写进权重的方案(KMs/PD/Cartridges/SDFT)都是梯度训练 + 教师分布当目标。 */
#ifndef DS4_CORE_PTRAIN_H
#define DS4_CORE_PTRAIN_H
#include "core_internal.h"   /* 守卫外: CPU 构建的 ds4_engine_ptrain 空壳也要 ds4_engine 类型 */
#ifndef DS4_NO_GPU

/* 训练配置(--ptrain <spec>, 每行 key=value, # 起注释; 解析在 core_ptrain_data.c) */
typedef struct {
    char data[1024], out[1024];
    char hold[1024];               /* 保持料 jsonl(可空, 空就打一行警告): 题不带 context, 答案 = 部署态自己答的, 教师 = 部署态自己 */
    char init[1024];               /* 非空 = 放大器 A/B 从这个 ③ 目录的 amp_Lnn.bin 读起(续训 / 在非零点做梯度检查); 空 = A=0、B 随机 */
    char evalf[1024];              /* eval=: 泛化留出 jsonl(料格式同 data), 整份不进训练; 空 = 没有泛化指标, 选轮退回同形式留出 */
    char evalkl[1024];             /* eval_kl=: 同形式留出(eval 1)从这份文件来, data= 整份进训练、不再五取一。为什么(11-02 V 步 v06/v09): 实体料一只动物三种问法, 按行五取一
                                    * 留出降了 5% 却一点没迁到没练过的实体 —— 降的是"背同实体的别种问法"; 留出要按实体整只切, 调用方(vstep_spark.sh prep)切好给这里 */
    char teacher_fp[1024];         /* teacher=fp:<表>: 教师 = 出厂原件的参考前向(v41_teacher.py --seqs --topk 出的 top-K 表, core_ptrain_fp.c), 每道题按整条教师序列
                                    * 逐 id 对上才填, 对不上就停(不回退到部署态教师); 这时 one-hot 行也全改成向 FP 蒸馏。空 = 教师是部署态自己(老路逐位同) */
    char dumpids[1024];            /* dumpids=<文件>: 装料后把每道题的教师序列(去重, 一行一串 id)写出去就退出 —— 拿去给 FP 参考前向打分 */
    uint32_t sample_n;             /* 步 2 每题采几份(0 = 不走步 2, 整趟与 10-10 版逐位同) */
    uint32_t expand;               /* expand=N(缺省 4): 装料时让部署态把每道训练题改写成 N 组同格式 jsonl 新问答一起训, 缓存 <data>.expandN.jsonl(0 = 关, 老路逐位同), core_ptrain_expand.c */
    uint32_t layer_lo, layer_hi;   /* 训练哪几层的放大器(含两端); 只训末层时出口反传是闭式的, 不用穿注意力 */
    uint32_t rank, topk, epochs, batch, maxlen, probe_n, probe_tok, seed;
    uint32_t budget_s;             /* budget_s=N(11-02, 缺省 0 = 不限): 训练墙上时间预算(秒, 从第一步起算, 不含装载/第 0 步评估)。到了就把当前批当本轮最后一批:
                                    * 照常轮末评估 + 落 ckpt, 然后不再开下一轮。为什么是硬闸不是算料: 用户定"本地一轮 ≤ 1 小时", 料长/步速的账总有漂移(ent04 账 ~100 分钟/轮实跑 62),
                                    * 闸在训练器里才保证超时不是靠人盯。与 max_steps 不同: max_steps 是计时模式(不评估不存盘), 这里产物照样能挂 */
    uint32_t vq_l[DS4_MAX_LAYER], vq_nl;   /* vq_layers=a-b(11-02 V 步, core_ptrain_vstep.c): 给了就不训低秩, 改成端到端重选这几层专家权重的离散码字(每轮一步), 产物 = 索引补丁 */
    uint32_t vq_hot, vq_knn, vq_group, vq_bs, vq_check;   /* 每层每步最热几个专家(缺省 4) / 码本近邻数(16) / 每趟梯度几层一组(5) / 每步批几题(≥batch, 缺省 16) / 1 = 第 1 轮先做一阶预测对实测的机制审计 */
    float vq_frac;                 /* 每步每矩阵最多改多少份组(缺省 0.005): 一阶线性化只在小步内成立, PV-Tuning 同一取舍 */
    float vq_tail;                 /* vq_tail=τ(11-02 晚): V 步选码/审计趟按行尾权 w = min(16, max(1, KL/τ))(ds4_gpu_bwd_kl_tail); 0 = 关。留出评估不加权 */
    uint32_t vq_docs;              /* vq_docs=N(11-02 晚, 缺省 4): V 步每轮的"前缀文档份"带 N 篇(BOS 开头的长行: agent 头行那种 10k 前缀 + 2k 靶), 轮转; ids 流仍每轮 1 篇一份 */
    char vq_init[1024];            /* vq_init=<③ 目录>(11-02 深夜): V 步起点 = 这份索引补丁盖上后的态(续训 / 跨层叠); 它的记录并进本趟日志 ⇒ 每个 ckpt 自包含可单独挂 */
    uint32_t data_eval;            /* data_eval=1(11-02): 给了 eval= 时 data 的料也照常 FNV 五取一留出(缺省 0 = 老规矩不留出) —— V 步要知识留出与编程判决份两边都把门 */
    uint32_t ctxlen;               /* ctxlen=N(10-25, 缺省 0 = maxlen): 样本最长可到 N 位; 超过 maxlen 的 one-hot 样本按"前缀只预填、尾窗才反传"过 ——
                                    * 前 sn−maxlen 位当普通预填进 KV 缓存(不存档不反传), 最后 maxlen 位存档 + 反传, 损失只算尾窗里的目标。
                                    * 为什么: 反传缓冲按 maxlen 分(每层 hc_in 存档 + 几十个 n×E×4), 4096 窗 10 GB+ 分不出(10-25 两次实撞), 而鹈鹕
                                    * "path 少一半"是一两万位设计推理的累积, 靶必须在长上下文里 —— 前缀给上下文, 尾窗给梯度, 内存只按尾窗长。 */
    uint32_t doc_tail;             /* doc_tail=K(10-25, 缺省 0 = 全训; 只在前缀形态): 每篇 text 只把末 K 段当训练样本, 前面的原文只当上下文(首个样本整段预填, 不反传)。
                                    * 为什么: 图形轨迹的 </think> 与 SVG 正文在末 4~5 段(正文中位 1.4 万字符), 快速验证时前面两万位思考只要给上下文, 预填约是训练速度的 4~5 倍 */
    uint32_t eval_n;               /* eval_n=N(10-25, 缺省 0 = 不限): 料里的留出按整篇等间隔抽, 约 N 段, 其余留出段标 eval=3(不训也不评, 同一条原文的段不漏进训练)。
                                    * 为什么: 留出按整条原文 FNV 五取一, 整条轨迹 2.8 万位切 28 段全进留出 ⇒ 82 条轨迹留出 649 段, 第 0 步评估跑了 40 分钟还没完(10-25 18:4x) */
    uint32_t gradcheck;            /* 1 = 训练前先做有限差分梯度检查(每个训练层的 A 沿梯度方向 ±ε), 2 = 查完就退出 */
    uint32_t prof;                 /* 1 = 整步分段计时(每段前后同步, 只作诊断: 会拖慢), 每 5 步打一次累计表(pt_prof_print) */
    uint32_t max_steps;            /* >0 = 只训这么多步就退出: 不做第 0 步评估/探针、不评估不存盘 —— 专给计时用(kdprof 段), 产物不能挂 */
    uint32_t epoch_tok;            /* >0 = 一轮只过打乱序里连续的一段(各题行数含提示累计到它即收), 下一轮接着取, 取完整个打乱序才重新打乱
                                    * ⇒ 一个周期内每题恰好一次(10-03 用户定: 全层全量一轮 30 分钟低于带宽地板, 改成 10 分钟一轮的切片); 0 = 一轮 = 全量过一遍 */
    uint32_t contcheck;            /* 1 = 按篇接力自检(前缀形态): 第一篇 ≥3 段的训练原文, 段 0、1 接力前向 + 反传, 段 2 先接力再清状态从头预填各算一次损失, 比完就退出 */
    uint32_t rccheck;              /* 1 = 重算对拍: 第一道训练题前向 + 反传一次, 每层重算后补算层出口, 与前向存档的下一层入口比, 打完就退出 */
    uint32_t gcl[16], ngcl;        /* gclayers=a/b/..: 放大器梯度检查只查这几层(不给 = 训练段全部) */
    uint8_t hcmid[16];             /* hccheck 里层号后跟 m(如 32m) = 扰动点放在该层中段(注意力半层出口、ffn 半层入口) */
    uint32_t hcl[16], nhcl;        /* hccheck=a/b/..: 逐层定位反传错在哪 —— 在第 k 层入口沿解析梯度扰动 hc, 有限差分对解析(冻结选择), 打完就退出 */
    uint32_t diag;                 /* 1 = 逐位诊断(core_ptrain_diag.c): 每道题答案逐位并排 教师 / 挂 ③ / 部署态 前几名, 写 out/diag.txt, 不训练就退出 */
    uint32_t teachercheck;         /* 1 = 教师表全部重算, 每题一块算两遍(必须逐位同) + 切三块算一遍(分块等价), 打完就退出(10-11: 超长续算改分块后的门); 2 加逐位解码参照; 3 再加 W=8(解码核)/W=9(预填核)对解码 —— 预填路与解码路差在哪的门 */
    uint32_t packcheck;            /* 1 = 合批的门(①一题一包与单题路逐位同 ②一包两份的有限差分), 2 = STE 尖峰普查(单题路按题长扫 L8 有限差分); 打完就退出(core_ptrain_diag.c) */
    uint32_t nopack;               /* 1 = 训练/评估不合批(一题一题过, 10-02 之前的路; 只作对照) */
    int32_t dbg_layer;             /* ≥0 = 该层 ffn 半层反传逐步打梯度范数(第一题那几行, core_ptrain_layer.c pt_dbg), 定位哪一步把梯度算大; 缺省 -1 */
    uint32_t poison;               /* 1 = 毒值测试: 每层反传前把梯度暂存整块、状态第 n 行以后、本层窗口填 NaN, 哪层先出 NaN = 那层读了没写过的内存 */
    float lr, clip, init_std;
} pt_cfg;

/* 一个教师上下文块(同一份 context 的题共一块): 教师序列 = 材料 + 空行 + 问题, 同块的题共享前 lcp 个 token(前缀快照的依据)。
 * hold = 保持料块(没有材料): 教师提示 = 学生提示, 教师分布就是部署态自己。sft = 不带材料的普通题与 text 原文段(目标 one-hot, 不要教师)。 */
typedef struct { char *name; char *text; uint32_t lcp; int hold, sft; } pt_chunk;

/* 一道题: 学生序列 = 部署同款聊天渲染(无 system, 不思考) + 答案 + EOS; 教师序列 = 块 + 题的同款渲染 + 同一串答案。
 * raw(text 行): 学生序列 = BOS + 原文段(+ 末段 EOS), sa0 = 1, 没有问句 —— 探针不抽它(只有 BOS 的提示生成不出可比的东西), 评估照常算留出 NLL */
typedef struct {
    uint32_t chunk;
    int eval;                       /* 1 = 留出(措辞没进训练); 2 = eval= 文件的题(换了形式的整组留出, 量泛化) */
    int raw;                        /* 1 = text 原文段 */
    uint32_t doc;                   /* raw 段属于第几行原文(pt_data.ndoc 计数): 前缀形态按篇接力, 打乱时整篇的段按顺序挨着(core_ptrain.c pt_group_docs) */
    char *q, *a;
    int32_t *sids; uint32_t sn, sa0;   /* 学生 ids / 长度 / 答案第一个 token 的下标 */
    int32_t *tids; uint32_t tn, ta0;   /* 教师 ids / 长度 / 答案起点 */
    uint32_t m;                     /* 答案 token 数(含 EOS) = 要对齐的分布行数 */
    int32_t *top_id; float *top_p, *top_rest;   /* 教师 [m][K] / [m][K] / [m]; NULL = 还没算 */
    int sft;                        /* 1 = 表是答案 token 的 one-hot(同一个 KL 核算出来就是交叉熵, 不过教师信念这一道; 装料时填好, 不进教师缓存) */
    int src;                        /* 0 = data 文件, 1 = hold 文件, 2 = eval 文件: 教师缓存按文件各一份(<文件>.teacher.bin), 同一份保持料换料重训永远命中 */
    int32_t *dids; uint32_t dn;     /* 步 2 的教师提示(不带材料的题才有: 这条问答当示范放进上下文; 带材料的题步 2 直接用 tids 的提示段) */
    uint64_t fpkey;                 /* FP 教师表的键(10-26): 0 = 自己的 tids 整条; 前缀形态的原文段(tids 是整篇的前缀) = 整篇序列 BOS+原文(+EOS) 的 FNV ——
                                     * 一篇只导一条教师序列、打一次表, 各段按"表序列以 tids 开头"对上、取自己那 m 行(core_ptrain_fp.c); 按段的窗去重会把一篇算 doc_tail 遍 */
    uint32_t l0;                    /* 答案位里开头几位只当上下文、不进损失(10-26 晚, 分支续写料: text 行带 "prefix" = 学生自己吐的思考前缀, 靶只是官方从那里续写的部分)。
                                     * 这几位在前向里(有 logits 行), 表里也有(one-hot 占位), 只是 pt_loss_rows 从第 l0 位起算 —— 老料全 0, 逐位不变 */
} pt_sample;

typedef struct {
    pt_chunk *ch; uint32_t nch;
    pt_sample *s; uint32_t ns;
    uint32_t ndoc;                  /* 读进来的 text 行数(pt_sample.doc 的上界) */
    int32_t **docids; uint32_t *docn;   /* 每篇原文的整条序列(按 pt_sample.doc 下标; 只前缀形态有, 否则 NULL/0): dumpids 按它导整篇 */
} pt_data;
uint64_t pt_ids_fnv(const int32_t *ids, uint32_t n);   /* 教师序列的键(core_ptrain_fp.c) */

/* splitmix64: 采样种子由 (seed, 轮, 题号, 份号) 这类整数定, 两趟同配置逐份同种子(步 2 与改写都用它) */
static inline uint64_t pt_mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

/* core_ptrain_expand.c: 多形式扩展。d 里现有的训练题(src 0, 非留出, 非原文段)每道让部署态改写成 c->expand 种问法, 答案照抄,
 * 写成与料同格式的 jsonl 到 cache(已有就不生成); 调用方再把它当 src 0 装进来并把这些行全标成训练(改写的是训练题的形式, 不许落到留出那边) */
bool pt_expand(ds4_engine *e, const pt_cfg *c, const pt_data *d, const char *cache);

/* core_ptrain_data.c */
bool pt_cfg_load(pt_cfg *c, const char *path);
bool pt_parse_ids_line(const char *ln, int32_t **ids, uint32_t *n);   /* ids 行 {"ids":[…]}(core_ptrain_ids.c, 11-02): 是 ids 行才返回 true; 坏行直接停 */
bool pt_add_ids(ds4_engine *e, const pt_cfg *c, pt_data *d, int src, const int32_t *ids, uint32_t n);
bool pt_add_tokens(ds4_engine *e, const pt_cfg *c, pt_data *d, int src, int eval, int32_t first, const int32_t *tok, uint32_t T, uint32_t np);   /* 原文段切段建样本(core_ptrain_data.c) */
int pt_is_eval(const char *q);   /* 去空白 FNV 五取一(core_ptrain_data.c) */
bool pt_parse_line(const char *ln, char **sys, char **ctx, char **q, char **a, char **txt, char **pfx, int *eos, const char **err);   /* 料行解析(唯一一份);
                                    * eos: text 行的 "eos" 字段(缺省 1 = 末尾补 EOS 当目标; false = 原文没写完, 不补 —— 撞长度上限的轨迹末尾补 EOS 是假靶)
                                    * pfx: text 行可选的 "prefix"(没有给 ""): 排在 text 前面、只当上下文不当靶的原文(分支续写料的学生前缀, 见 pt_sample.l0) */
bool pt_data_load(ds4_engine *e, const pt_cfg *c, pt_data *d);
/* 前缀形态的块界对齐粒度 = max(2, 各层压缩比)(层表里读; V4.1 层表压缩比只有 0/1/2, 所以就是"取偶"): 前缀长/段界都取它的倍数,
 * 组界不跨前缀/尾窗(尾窗压缩器反传只对本窗自己的组求导, core_ptrain_layer.c pt_compress_bwd 的 pg0 = pfx/ratio)。 */
uint32_t pt_pfx_align(void);
void pt_data_free(pt_data *d);
void pt_data_lcp(pt_data *d);   /* 每块的教师公共前缀(装料末尾算一次; 步 2 新造的样本同样要) */
uint32_t pt_chunk_of(pt_data *d, const char *ctx, int hold, int raw);
bool pt_teacher(ds4_engine *e, const pt_cfg *c, pt_data *d);   /* 教师 top-K(块前缀快照; 结果缓存到 <料文件>.teacher.bin), core_ptrain_teacher.c */
bool pt_teacher_fresh(ds4_engine *e, const pt_cfg *c, pt_data *d);   /* 同一趟教师前向, 不读不写缓存(步 2 的采样答案每轮都是新串) */
/* core_ptrain_fp.c: FP 教师。dump = 每道题的教师序列去重写成一行一串 id(dumpids=); fp = 读 v41_teacher_seqs.py 的 top-K 表填全部样本(teacher=fp:) */
bool pt_dump_ids(const pt_cfg *c, const pt_data *d);
bool pt_teacher_fp(const pt_cfg *c, pt_data *d);
void pt_state_reset(ds4_v41_state *st);   /* 状态回到"位置 0、空缓存"(缓冲不动): 一条新序列从头算 */

/* 第二阶段逐层反传的缓冲(core_ptrain_layer.c): 一次只活一层的重算中间量 + 梯度。按 maxlen 行分一次, 每层复用。 */
typedef struct {
    ds4_gpu_tensor *mixa, *posta, *comba, *prea, *xa, *xna, *qr, *kv, *o, *hcin, *mixf, *postf, *combf, *pref, *xf, *xnf;   /* q 与中段 hc 直接读状态里的(见 core_ptrain_layer.c) */
    ds4_gpu_tensor *g_hcm, *g_hcin, *g_y, *g_xnf, *g_xf, *g_pre_a, *g_postf, *g_combf, *g_attn, *g_posta, *g_comba,
                   *g_low, *g_o, *g_q, *g_kvn, *g_qrn, *g_qr, *g_kv, *g_xna, *g_xa, *g_prein, *g_sh, *g_sg, *g_su, *g_rw, *g_glog;
    /* 分界层以下(训练段碰到 kv 源层 / engram 层才分配): */
    ds4_gpu_tensor *gcomp[DS4_MAX_LAYER];   /* 源层压缩行的梯度 [n][512]: 读取层倒着算时累加, 轮到源层时经压缩器反传进它的 xn */
    ds4_gpu_tensor *posg[DS4_MAX_LAYER];    /* 源层压缩行的位置 g·ratio(i32 [n]), 压缩行 RoPE 的反向用 */
    ds4_gpu_tensor *cckv, *ccsc, *cpool, *cgpool, *cgkv, *cgsc;   /* 压缩器重算 / 反传暂存 [n][512] */
    ds4_gpu_tensor *ekvf, *hpre;            /* engram: 门输入解包 [n][(HC+1)·E] / 门之前的 hc [n][HC][E] */
    int ready;
} pt_layer_buf;

/* prof 分段(pt_run.tm 的下标): 逐层反传里四段(core_ptrain_layer.c) + 每题外围四段(core_ptrain_bwd.c) + 每步两段(core_ptrain.c) */
enum { PT_TM_RECOMP, PT_TM_ROUTED, PT_TM_ATTN, PT_TM_LREST,      /* 重算 / routed 专家反向 / 注意力半层反向 / 层内其余 */
       PT_TM_FWD, PT_TM_LOSS, PT_TM_EXIT, PT_TM_LAYERS, PT_TM_Q,  /* 前向(带存档) / 教师表上传 + KL / 出口反传(head + RMSNorm + hc_pre) / 逐层反传墙钟 / 每题墙钟 */
       PT_TM_ADAM, PT_TM_STEP };                                  /* 梯度范数 + Adam / 每步墙钟(批内各题 + Adam) */
double pt_tick(const pt_cfg *c);   /* prof 开着: 同步 GPU 后取墙钟; 关着: 0(零开销), core_ptrain_layer.c */

/* 合批(core_ptrain_pack.c, 10-02): 几道题首尾相接进批态 r->st(总行数 ≤ maxlen), 逐 token 的段(embed/hc/norm/投影/MoE/出口)整批一次发,
 * 每遍专家权重只读一次摊给几道题; 注意力缓存段(窗口/压缩源/indexer/稀疏注意力)与它的反传、压缩器反传按题分段。
 * 每题一个请求槽(ds4_v41_state): 行字段是批态行的视图(v41_attach), 缓存字段也是批态自己缓冲的视图 —— 按行的(窗口/压缩器余行/索引中间量)
 * 从第 r0 行切, 按组的(压缩行/索引键)从 g0 组切; 各题位置都从 0 起, 块区正好首尾相接(窗口的块区铺满 [SWA, SWA+R)), 不另开池。
 * 为什么: 一题两三百 token 摊到 ~365 个专家每个只 ~4 个, 每遍要把 ~100 GB 专家位流读满一遍(10-02: vqm 一层 11 ms = 232 GB/s 贴带宽墙),
 * 一题一题过就是"每题三遍全量读"的地板; 合批后同一遍读喂几道题。 */
#define PT_PACK_MAX 8u
typedef struct {
    uint32_t nb, R;                                    /* 本包几道题 / 合计行数; nb = 0 = 单题路(批态自己就是那道题) */
    uint32_t r0[PT_PACK_MAX], n[PT_PACK_MAX];          /* 各题在批态里的起始行 / 行数 */
    uint32_t g0[PT_PACK_MAX][DS4_MAX_LAYER];           /* 各题在源层压缩行里的起始组号(前面各题 n/ratio 之和) */
    uint32_t ng[PT_PACK_MAX][DS4_MAX_LAYER], topk[PT_PACK_MAX][DS4_MAX_LAYER], iratio[PT_PACK_MAX][DS4_MAX_LAYER];   /* 前向存档(按题) */
    ds4_v41_state *rq;                                 /* 请求槽 [PT_PACK_MAX](堆上: 一个状态几十 KB) */
    v41_rowview rv[PT_PACK_MAX];                       /* 各槽挂进批态的行视图(整包前向 + 反传期间一直挂着) */
    int ready;
} pt_pack;

/* 行视图(第 r0 行起 rows 行, 每行 rowf 个 f32) */
static inline ds4_gpu_tensor *pt_rows(const ds4_gpu_tensor *t, uint32_t r0, uint32_t rows, uint64_t rowf) {
    return ds4_gpu_tensor_view(t, (uint64_t)r0 * rowf * 4, (uint64_t)rows * rowf * 4);
}

/* core_ptrain_bwd.c: 一个样本的前向 + 损失 + 反传(梯度累加进 gA/gB); grad=0 只算损失(评估) */
typedef struct {
    ds4_v41_state st;
    v41_tsave save;
    uint32_t K;      /* A/B 的总行数 = k2 + rank */
    uint32_t k2;     /* 前 k2 行是挂着的 ② 的低秩件(冻结: 梯度每步清零, 落盘不写; 11-01 ③ 叠 ②); ② 无低秩 / 训 ② 本身时 0 */
    ds4_gpu_tensor *gA[DS4_MAX_LAYER], *gB[DS4_MAX_LAYER], *mA[DS4_MAX_LAYER], *vA[DS4_MAX_LAYER], *mB[DS4_MAX_LAYER], *vB[DS4_MAX_LAYER];
    ds4_gpu_tensor *tid, *tp, *trest, *loss;   /* 一个样本的教师表 / 逐行损失(设备) */
    ds4_gpu_tensor *gxn, *gx, *ghc, *gpre, *gy, *T, *gT;   /* 反传暂存(按 maxlen 行); ghc/gpre 在多层模式下 = 逐层往下传的 g_hc / g_pre_mix */
    float *loss_h;
    /* 决策位读数(10-22, docs/think_close.md §4): dec_on 期间 pt_loss_rows 把目标是 dec_ids(EOS / </think>)的位单独累计 —— 这几位的 KL 与教师给它的 p。
     * 为什么: 留出 KL 是逐位平权, 一道题几百位里决策位只有一位, 量化把它压 4~22 倍在总 KL 里看不见(10-17 的 ③ 留出 −46% 而生成纹丝不动) */
    int dec_on; int32_t dec_ids[2]; double dec_kl, dec_tp; uint32_t dec_n;
    /* 尾位计数(11-02 晚): tail_on 期间 pt_loss_rows 数 KL>1 / KL>2 的位 —— V 步每轮在判决份上打出来, 与判决尺的尾位数(c5f2 634 / 177)直接可比;
     * 为什么: 均值四趟都动尾巴一位不动, 轮内只看均值分不清"尾位修了均值付了代价"和"什么都没修" */
    int tail_on; uint32_t tail_n, tail_n1, tail_n2;
    /* 逐题决策位 KL(10-22 下午): dec_last = pt_loss_rows 刚算的那道题决策位 KL 和(没有决策位 = -1), dec_slot[i] = 包里第 i 题的(单题路放 [0]);
     * pt_run_list 按题抄进 dec_each → heldout_eNN.tsv 第 5 列。为什么: 24 位的均值两趟差 0.433/0.329 而整体留出几乎同, 不看逐题分不清是尺的噪声还是真变化 */
    double dec_last, dec_slot[PT_PACK_MAX];
    pt_layer_buf lb;                /* 多层模式(训练层不止末层)才分配 */
    uint32_t pfx;                   /* 当前样本的前缀长(ctxlen 形态, 见 pt_cfg.ctxlen): 尾窗块首位的绝对位置; 0 = 老形态/合批 */
    ds4_gpu_tensor *ring_snap[DS4_MAX_LAYER];   /* ctxlen > maxlen 才分: 前缀预填完每层窗口环头 [0,SWA) 的快照, 尾窗前向的环提交会盖掉它, 逐层重算前还原 */
    ds4_gpu_tensor *ring_post[DS4_MAX_LAYER];   /* 同上, 尾窗前向刚提交完的环头: 反传重算把环头还原成了前缀那份, 反传完再拷回来, 状态才是"整段 [0, sn) 都进了缓存" */
    const pt_sample *cont_s;        /* ★按篇接力★: 状态里此刻正好是这道样本的整段 [0, sn)(前向 + 反传完、环头已拷回); 下一道是它的接续
                                     * (前缀 == 它的整段)就不清状态、不重新预填前缀。NULL = 状态不可接(合批/诊断用过, 或这道失败)。
                                     * 代价: 前缀的 KV 是上一步参数下算的(批末 Adam 已更新过一次), 和 Transformer-XL 的段间缓存同一种近似 */
    double tm[11]; uint32_t tm_n, tm_steps;   /* prof 累计秒数(下标 PT_TM_*), 训练题数, 优化步数 */
    ds4_gpu_tensor *hcap; int32_t hcap_layer; int hcap_mid;   /* hc 边界检查: 反传经过第 hcap_layer 层时把该层入口(mid=1: 中段)梯度抄进 hcap */
    pt_pack pk;                     /* 合批(多层模式才开; 梯度检查/重算对拍/hc 检查仍走单题路, pk.nb = 0) */
    ds4_gpu_tensor *vg_dw, *vg_slot[DS4_MAX_LAYER];   /* V 步(core_ptrain_vstep.c): 按组梯度缓冲 + 每层"专家 → 槽"表(NULL = 该层不收); 逐层反传前设给 ds4_gpu_bwd_moe_vgrad */
} pt_run;
/* 这道题能不能进训练/评估(教师给了表、不超批态行数; 长样本在 ctxlen 内走前缀形态): 与 pt_step_sample 的跳过条件同一个。
 * 前缀形态只许 one-hot 或原文段(raw): 它们的表每行都是"整篇前缀下的分布"(one-hot / 整篇打的 FP 表, core_ptrain_fp.c), 窄窗的部署态教师表不是(data.c pt_add_sample 的 lim)。
 * 10-26 实撞: 只认 sft 时, 挂 FP 教师的 v3 趟 503 段原文里长于 maxlen 的全被跳过 —— 步 10 批 KL 0.0000 梯度 0, 1.9 s/步, "训练前 64 题 0.0000"。 */
static inline bool pt_runnable(const pt_run *r, const pt_sample *s) {
    return s->top_id && (s->sn <= r->st.cap_tok || ((s->sft || s->raw) && s->sn + 8u <= r->st.ctx));   /* st.ctx = max(ctxlen, maxlen) + 8 */
}
/* 一个样本答案位的损失 + logits 梯度(就地写进 logits 第 row 行起 m−off 行; *glog = 那几行的视图, 调用方 free)。
 * off = 答案位里跳过前几位(前缀形态: 落在前缀里的目标没有 logits 行), 表指针按它偏移; 老路传 0 */
bool pt_loss_rows(const pt_cfg *c, pt_run *r, ds4_gpu_tensor *logits, uint32_t row, const pt_sample *s, uint32_t off, double *loss, uint32_t *ntok, ds4_gpu_tensor **glog);
bool pt_run_alloc(ds4_engine *e, const pt_cfg *c, pt_run *r);
void pt_run_free(pt_run *r);
bool pt_step_sample(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *s, int grad, double *loss_sum, uint32_t *ntok);
bool pt_save_amp(const pt_cfg *c, pt_run *r, const char *dir);   /* ③ 落盘: amp_Lnn.bin(f32) + base.fnv */
bool pt_eval(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d, int which, uint32_t cap_n, double *kl, int hold, double *li, uint32_t *ni, double *di);   /* core_ptrain.c; V 步的轮末评估也用 */
bool pt_vstep_run(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d, const uint32_t *ord, uint32_t ntr, double kl_ev0, FILE *lf, double t0);   /* core_ptrain_vstep.c */
long pt_mem_mb(void);   /* 可用内存 MB(/proc/meminfo; 没有 = -1), core_ptrain.c */
/* 一个优化步(core_ptrain.c): 全局梯度范数 → 裁剪系数 → 各训练层 A/B 的 Adam(step 从 1 起)。步 1 与步 2 共用, 同一个 Adam 状态 */
bool pt_opt_step(const pt_cfg *c, pt_run *r, uint32_t step, double *gnorm, int *clipped);
/* 步 2(core_ptrain_onpolicy.c): qs[0..nq) = 本轮步 1 过的训练题号; 当前 ③ 存到 out/live 挂上去采样 → 教师打分 → 训, 步数接着 *step 往下记 */
bool pt_onpolicy(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d, const uint32_t *qs, uint32_t nq, uint32_t ep, uint32_t *step, FILE *lf, double t0);
bool pt_diag(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d);   /* diag=1: 逐位诊断, core_ptrain_diag.c(会把 ③ 的 A 置零, 跑完只能退出) */

/* core_ptrain_layer.c: 多层模式。alloc 分逐层缓冲与前向存档(层入口 hc/pre_mix/组号, engram 层另存门输入); bwd 从 r->ghc/r->gpre
 * (末层输出处的梯度)起, 第 hi 层到第 lo 层逐层"重算本层 → 倒推梯度", 放大器梯度累加进 gA/gB。训练段必须到末层(hi = 最后一层):
 * 出口梯度直接灌进第 hi 层, 中间没有别的层。分界层以下的压缩器与 engram 反传见 core_ptrain_layer.c 头注释。 */
bool pt_layer_alloc(const pt_cfg *c, pt_run *r);
void pt_layer_free(pt_run *r);
void pt_bwd_park(pt_run *r);                         /* 步 2 采样/教师期间放掉反传专用缓冲(合批槽 + 逐层缓冲 + 前向存档) */
bool pt_bwd_unpark(const pt_cfg *c, pt_run *r);      /* 步 2 训之前按原分配式分回来 */
bool pt_layers_bwd(ds4_engine *e, const pt_cfg *c, pt_run *r, uint32_t n);

/* core_ptrain_pack.c: 合批。alloc/free 开关请求槽; step = 一包(ss[0..nb), 总行数 ≤ maxlen)前向 + 各题损失(+ 反传),
 * loss_each/ntok_each 按题出(与 pt_step_sample 同口径: KL 和 / 答案 token 数)。
 * list = 一串题(训练的一批 / 评估的一组)装包后逐包过(不合批时一题一题过), 和进 loss_sum/ntok, 逐题进 loss_each/ntok_each(可 NULL)。
 * warm = 预热探底(装一个最满的包走一趟前向 + 反传); check = packcheck 门与普查(在 core_ptrain_diag.c)。 */
bool pt_pack_alloc(const pt_cfg *c, pt_run *r);
void pt_pack_free(pt_run *r);
bool pt_pack_step(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *const *ss, uint32_t nb, int grad, double *loss_each, uint32_t *ntok_each);
bool pt_run_list(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *const *ss, uint32_t n, int grad,
                 double *loss_sum, uint32_t *ntok, double *loss_each, uint32_t *ntok_each, double *dec_each);
bool pt_pack_warm(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d, uint32_t *rows);
bool pt_pack_check(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d);

/* core_ptrain_probe.c: 贪心探针(留出题, 部署态与挂 ③ 并排原样输出; 1 路一批, 与单请求路逐字节同)。pt_probes = probe_n 道料的留出题
 * + 至多 PT_PROBE_HOLD 道保持料留出题, 部署态答案在 base[] 里跨轮共用, 原样写 out/probe_<tag>.txt。 */
#define PT_PROBE_HOLD 2u
#define PT_PROBE_SLOTS(c) ((c)->probe_n * 2u + PT_PROBE_HOLD)   /* 料留出 probe_n + 保持料 + 泛化留出 probe_n(base[] 与选题缓冲的大小) */
/* 收口探针(10-11)的生成上限 = 参考答案 token 数的这么多倍: 正常收口的答案不会比参考答案长 4 倍, 到了还没 EOS 就是不收口(B 臂 e02 实撞 37k token)。
 * 上限按每道题的参考答案定, 不是全局常数(长答案题给长上限, 短答案题不白等) */
#define PT_STOP_RATIO 4u
/* base_stop: 部署态在这组题上的收口数, 一趟内只算一次(进来 <0 = 还没算); stop_k/stop_n = 这一轮挂 ③ 的收口数 / 题数(e00 没 init 时 = 部署态的) */
bool pt_probes(ds4_engine *e, const pt_cfg *c, const pt_data *d, char **base, int *base_stop, const char *pt_dir, const char *tag, uint32_t *stop_k, uint32_t *stop_n);

#endif
#endif
