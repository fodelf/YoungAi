/* core_ptrain.h — 后训练 ③ 第八版(上下文蒸馏, 2026-10-01)的内部声明。只被 core_ptrain*.c 包含。
 *
 * 一句话: 教师 = 部署模型(①+②)把复盘块放进上下文答题, 学生 = 同一模型 + ③ 不看块答同一题; 在答案位上让学生的分布逼近
 * 教师的(余量桶 KL), 梯度反传到 ③ 的低秩放大器(每层 MoE 输出 y += xn·B·A, 与 ② 同构, 引擎 --posttrain 直接挂)。
 * 料(qa 文件)由 z_nightly_spark.sh kdgen 段让模型读块自出; 本模块只做: 分词 → 教师 top-K → 训练 → 留出评估 → 落 ③。
 *
 * 为什么不用解算器: 闭式解只能把"答案该往哪挪"线性化到一层的输出上, 答案位的激活不像拟合行就不点火(10-01 六趟复盘 ③ 的结构病);
 * 文献里文档写进权重的方案(KMs/PD/Cartridges/SDFT)都是梯度训练 + 教师分布当目标。 */
#ifndef DS4_CORE_PTRAIN_H
#define DS4_CORE_PTRAIN_H
#include "core_internal.h"   /* 守卫外: CPU 构建的 ds4_engine_ptrain 空壳也要 ds4_engine 类型 */
#ifndef DS4_NO_GPU

/* 训练配置(--ptrain <spec>, 每行 key=value, # 起注释; 解析在 core_ptrain.c) */
typedef struct {
    char data[1024], out[1024];
    char init[1024];               /* 非空 = 放大器 A/B 从这个 ③ 目录的 amp_Lnn.bin 读起(续训 / 在非零点做梯度检查); 空 = A=0、B 随机 */
    char probe_q[1024];            /* 非空 = 自定义探针题文件(一行一题): 第 0 步与每轮末, 部署态 / 挂 ③ 各贪心答一遍(看指定问法翻没翻、邻居伤没伤) */
    char teacher[1024];            /* 教师表按题缓存文件(core_ptrain_teacher.c); 空 = out/teacher.bin。每日叠加训练各日共用一份: 回放的旧题永远命中 */
    char anchor[1024];             /* 非空 = 清单里 kl 行(自采样的锚)的教师挂这份 ③ 答同一串 token(信任域: 锚 = 本轮起点自己); 空 = 部署态。
                                    * 为什么不默认锚部署态: 部署态在决策题上就是没学过规则的那个, 锚它 = 一边训一边把学到的规则拉回去(10-04 奖励回路) */
    uint32_t sample_n;             /* >0 = 自定义探针题每道按模型卡配方(温 1 / top_p 1 / min_p 0, 种子固定)抽这么多份答案 → out/sample_<tag>.txt(pt_probes):
                                    * 奖励回路的采样器, 结算归脚本(kd_domain/<域>.sh <域>_reward)。只在落盘的 ③ 上抽: epochs=0 时第 0 步, 否则每轮末 */
    char probe_base[1024];         /* 非空 = 自定义探针的部署态答案缓存文件(跨趟): 部署态 = ①+② 各轮不变, 奖励回路每轮重新装载再算 20 道是白算(10-06) */
    uint32_t probe_batch, sample_batch;   /* 生成侧一次合几路(1..8; core_ptrain_probe.c): 探针缺省 1 —— 行数变 ⇒ 稠密段末位舍入变 ⇒ 决策数字那一位的近平局翻面
                                           * (10-06 门: 挂 ③ 20 道里 3 道目标价不同), 1 路与单请求路逐字节同, 贪心读数才与 0001~0010 可比; 采样缺省 8(分布级同, 只求快) */
    uint32_t eval0, probe0;        /* 第 0 步要不要评估 / 探针(缺省都 1)。奖励回路第 2 轮起的起点 = 上一轮末同一份 ③, 它的轮末评估/探针就是这一步的读数,
                                    * 再算一遍纯是重复(10-06: 一轮 40 分钟里白算 5 分多); epochs=0 的趟两者强制为 1(那一趟的产物就是第 0 步的探针与采样) */
    uint32_t layer_lo, layer_hi;   /* 训练哪几层的放大器(含两端); 第一阶段只有末层 —— 出口反传是闭式的, 不用穿注意力 */
    uint32_t rank, topk, epochs, batch, maxlen, probe_n, probe_tok, seed;
    uint32_t gradcheck;            /* 1 = 训练前先做有限差分梯度检查(每个训练层的 A 沿梯度方向 ±ε), 2 = 查完就退出 */
    uint32_t prof;                 /* 1 = 整步分段计时(每段前后同步, 只作诊断: 会拖慢), 每 5 步打一次累计表(pt_prof_print) */
    uint32_t max_steps;            /* >0 = 只训这么多步就退出: 不做第 0 步评估/探针、不评估不存盘 —— 专给计时用(kdprof 段), 产物不能挂 */
    uint32_t epoch_tok;            /* >0 = 一轮只过打乱序里连续的一段(各题行数含提示累计到它即收), 下一轮接着取, 取完整个打乱序才重新打乱
                                    * ⇒ 一个周期内每题恰好一次(10-03 用户定: 全层全量一轮 30 分钟低于带宽地板, 改成 10 分钟一轮的切片); 0 = 一轮 = 全量过一遍 */
    uint32_t rccheck;              /* 1 = 重算对拍: 第一道训练题前向 + 反传一次, 每层重算后补算层出口, 与前向存档的下一层入口比, 打完就退出 */
    uint32_t gcl[16], ngcl;        /* gclayers=a/b/..: 放大器梯度检查只查这几层(不给 = 训练段全部) */
    uint8_t hcmid[16];             /* hccheck 里层号后跟 m(如 32m) = 扰动点放在该层中段(注意力半层出口、ffn 半层入口) */
    uint32_t hcl[16], nhcl;        /* hccheck=a/b/..: 逐层定位反传错在哪 —— 在第 k 层入口沿解析梯度扰动 hc, 有限差分对解析(冻结选择), 打完就退出 */
    uint32_t diag;                 /* 1 = 逐位诊断(core_ptrain_diag.c): 每道题答案逐位并排 教师 / 挂 ③ / 部署态 前几名, 写 out/diag.txt, 不训练就退出 */
    uint32_t packcheck;            /* 1 = 合批的门(①一题一包与单题路逐位同 ②一包两份的有限差分), 2 = STE 尖峰普查(单题路按题长扫 L8 有限差分); 打完就退出(core_ptrain_diag.c) */
    uint32_t nopack;               /* 1 = 训练/评估不合批(一题一题过, 10-02 之前的路; 只作对照) */
    int32_t dbg_layer;             /* ≥0 = 该层 ffn 半层反传逐步打梯度范数(第一题那几行, core_ptrain_layer.c pt_dbg), 定位哪一步把梯度算大; 缺省 -1 */
    uint32_t poison;               /* 1 = 毒值测试: 每层反传前把梯度暂存整块、状态第 n 行以后、本层窗口填 NaN, 哪层先出 NaN = 那层读了没写过的内存 */
    float lr, clip, init_std;
    float hard_num, hard_txt;      /* 硬目标题(清单第 4 列 hard, 答案是代码算的)的逐 token 权重: 含数字的 token(决策数字) / 其余 token(10-04, 见 pt_add_sample) */
    float rft_scale;               /* 自采样行(清单第 4 列 rft)的权重 = (这份答案的奖励 − 同题各份的均值) × rft_scale(pt_rft_group) */
} pt_cfg;

/* 一个教师上下文块(复盘的一段): 教师序列 = 块正文 + 空行 + 问题, 同块的题共享前 lcp 个 token(前缀快照的依据) */
/* hold = 空块(保持料): 教师提示 = 学生提示, 教师分布就是部署态自己 —— 把 ③ 在通用问题上钉住 */
typedef struct { char *name; char *text; uint32_t lcp; int hold; } pt_chunk;

/* 一道题: 学生序列 = 部署同款聊天渲染(无 system, 不思考) + 答案 + EOS; 教师序列 = 块 + 题的同款渲染 + 同一串答案 */
typedef struct {
    uint32_t chunk;
    int eval;                       /* 1 = 留出(措辞没进训练) */
    char *q, *a;
    int32_t *sids; uint32_t sn, sa0;   /* 学生 ids / 长度 / 答案第一个 token 的下标 */
    int32_t *tids; uint32_t tn, ta0;   /* 教师 ids / 长度 / 答案起点 */
    uint32_t m;                     /* 答案 token 数(含 EOS) = 要对齐的分布行数 */
    int32_t *top_id; float *top_p, *top_rest;   /* 教师 [m][K] / [m][K] / [m]; NULL = 还没算 */
    int hard; float *w;             /* hard = 教师表是答案 token 的 one-hot(同一个 KL 核算出来就是交叉熵, 不过教师信念这一道): 1 = 账本硬目标(清单第 4 列 hard,
                                     * 答案是代码算的; w = 含数字 hard_num / 其余 hard_txt), 2 = 自采样(第 4 列 rft; w = 全 token 同一个组内优势, 可为负 = 往下压, 见 pt_rft_group);
                                     * 0 = 教师表; w = NULL 全 1, kl 锚行(第 4 列 kl)的 w = 全 token 同一个 β */
    float r; int nocache;           /* r = 自采样行的原始奖励(.qa 里 #W 行 / 清单第 5 列); nocache = kl 锚行: 教师挂 anchor= 的 ③, 一次性, 不读不写教师缓存 */
} pt_sample;

typedef struct {
    pt_chunk *ch; uint32_t nch;
    pt_sample *s; uint32_t ns;
} pt_data;

/* core_ptrain_data.c */
bool pt_cfg_load(pt_cfg *c, const char *path);
uint32_t pt_userq_load(const char *path, char ***out);   /* probe_q 文件 → 题数(空路径 = 0, 读不了 = UINT32_MAX) */
bool pt_data_load(ds4_engine *e, const pt_cfg *c, pt_data *d);
void pt_data_free(pt_data *d);
bool pt_teacher(ds4_engine *e, const pt_cfg *c, pt_data *d);   /* 教师 top-K(块前缀快照; 结果缓存到 out/teacher.bin), core_ptrain_teacher.c */
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
    uint32_t K;
    ds4_gpu_tensor *gA[DS4_MAX_LAYER], *gB[DS4_MAX_LAYER], *mA[DS4_MAX_LAYER], *vA[DS4_MAX_LAYER], *mB[DS4_MAX_LAYER], *vB[DS4_MAX_LAYER];
    ds4_gpu_tensor *tid, *tp, *trest, *tw, *loss;   /* 一个样本的教师表 / 逐 token 权重(硬目标题) / 逐行损失(设备) */
    ds4_gpu_tensor *gxn, *gx, *ghc, *gpre, *gy, *T, *gT;   /* 反传暂存(按 maxlen 行); ghc/gpre 在多层模式下 = 逐层往下传的 g_hc / g_pre_mix */
    float *loss_h;
    pt_layer_buf lb;                /* 多层模式(训练层不止末层)才分配 */
    double tm[11]; uint32_t tm_n, tm_steps;   /* prof 累计秒数(下标 PT_TM_*), 训练题数, 优化步数 */
    ds4_gpu_tensor *hcap; int32_t hcap_layer; int hcap_mid;   /* hc 边界检查: 反传经过第 hcap_layer 层时把该层入口(mid=1: 中段)梯度抄进 hcap */
    pt_pack pk;                     /* 合批(多层模式才开; 梯度检查/重算对拍/hc 检查仍走单题路, pk.nb = 0) */
} pt_run;
/* 这道题能不能进训练/评估(教师给了表、不超批态行数): 与 pt_step_sample 的跳过条件同一个 */
static inline bool pt_runnable(const pt_run *r, const pt_sample *s) { return s->top_id && s->sn <= r->st.cap_tok; }
bool pt_run_alloc(ds4_engine *e, const pt_cfg *c, pt_run *r);
void pt_run_free(pt_run *r);
bool pt_step_sample(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *s, int grad, double *loss_sum, uint32_t *ntok);
bool pt_save_amp(const pt_cfg *c, pt_run *r, const char *dir);   /* ③ 落盘: amp_Lnn.bin(f32) + base.fnv */
bool pt_diag(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d);   /* diag=1: 逐位诊断, core_ptrain_diag.c(会把 ③ 的 A 置零, 跑完只能退出) */

/* core_ptrain_layer.c: 多层模式。alloc 分逐层缓冲与前向存档(层入口 hc/pre_mix/组号, engram 层另存门输入); bwd 从 r->ghc/r->gpre
 * (末层输出处的梯度)起, 第 hi 层到第 lo 层逐层"重算本层 → 倒推梯度", 放大器梯度累加进 gA/gB。训练段必须到末层(hi = 最后一层):
 * 出口梯度直接灌进第 hi 层, 中间没有别的层。分界层以下的压缩器与 engram 反传见 core_ptrain_layer.c 头注释。 */
bool pt_layer_alloc(const pt_cfg *c, pt_run *r);
void pt_layer_free(pt_run *r);
bool pt_layers_bwd(ds4_engine *e, const pt_cfg *c, pt_run *r, uint32_t n);

/* core_ptrain_pack.c: 合批。alloc/free 开关请求槽; step = 一包(ss[0..nb), 总行数 ≤ maxlen)前向 + 各题损失(+ 反传),
 * loss_each/ntok_each 按题出(与 pt_step_sample 同口径: KL 和 / 答案 token 数)。
 * list = 一串题(训练的一批 / 评估的一组)装包后逐包过(不合批时一题一题过), 和进 loss_sum/ntok, 逐题进 loss_each/ntok_each(可 NULL)。
 * warm = 预热探底(装一个最满的包走一趟前向 + 反传); check = packcheck 门与普查(在 core_ptrain_diag.c)。 */
bool pt_pack_alloc(const pt_cfg *c, pt_run *r);
void pt_pack_free(pt_run *r);
bool pt_pack_step(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *const *ss, uint32_t nb, int grad, double *loss_each, uint32_t *ntok_each);
bool pt_run_list(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *const *ss, uint32_t n, int grad,
                 double *loss_sum, uint32_t *ntok, double *loss_each, uint32_t *ntok_each);
bool pt_pack_warm(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d, uint32_t *rows);
bool pt_pack_check(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d);
bool pt_eval_anchor(ds4_engine *e, const pt_cfg *c, pt_run *r, pt_data *d, double *kl, uint32_t *nq);   /* kl 锚行上的裸 KL(权重摘掉), nq = 行数 */

/* core_ptrain_probe.c: 生成侧(合批, 10-06)。自定义探针题(probe_q= 文件): 部署同款渲染(无 system、不思考)后贪心答; 部署态答案算一次存进 ub
 * (probe_base= 可跨趟缓存), 之后各轮只算挂 ③ 的。pt_probes = 留出题探针(probe_n 道 + 至多 PT_PROBE_HOLD 道保持料留出题, 部署态在 base[] 里跨轮)
 * + 自定义题探针 + 奖励回路的采样(sample_n), 原样写 out/probe_<tag>.txt / out/sample_<tag>.txt。 */
typedef struct { char **q, **ub; uint32_t n; } pt_userq;
#define PT_PROBE_HOLD 2u
bool pt_probes(ds4_engine *e, const pt_cfg *c, const pt_data *d, char **base, pt_userq *uq, const char *pt_dir, const char *tag);

#endif
#endif
