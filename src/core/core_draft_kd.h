/* core_draft_kd.h — 草稿器蒸馏(2026-10-07, --draft-train <配置>)的内部声明。只被 core_draft_kd*.c 包含。
 *
 * 一句话: DSpark 三塔是照原始 FP 模型训的, 部署的是量化 + 反修的底座, 两边同一位置的分布有偏 —— 采样下首位期望接受率
 * Σmin(p,q) 只有 0.74~0.76(09-29 陪审团; q = p 时它是 1)。这里让草稿器改盯**部署底座**: 教师强制走底座自己采样出来的真实请求
 * 续写, 块内 B 位各对底座同位置的分布做 KL(top-K + 余量桶, 与后训练 ③ 同一个核), 梯度穿过三塔反传到两种低秩件:
 *   塔件: 每塔 MoE 出口 y += xn·B·A(与 ③ 同构, 挂在草稿态 st.ampA[T]);  出口件: xn += xn·BᵀA(现成 --draft-amp 槽)。
 * 底座、塔原权重、出口头、markov 表全冻结; 验证侧一个字节不动 ⇒ 贪心输出仍逐字节等于纯解码, 采样下边缘仍恰是 p, 动的只有接受率。
 * 料 = 提示 + 底座采样的续写(ids 文件): 草稿器的靶本来就是底座在自己轨迹上的分布, 这不是 ③ 那条"自吐轨迹当拟合料"的病。
 * 底座只跑一遍(core_draft_kd_cap.c): 每份文本的 main_x(三塔窗口的 KV 来源)与教师表缓存到盘, 之后每轮只动三塔。
 * 判据: 第 0 轮(件为零)陪审团首位 Σmin 必须复现 d1 accjury 在同一份 ids 上的数(接线门); 留出文本五位 Σmin 升; 部署门在脚本里。 */
#ifndef DS4_CORE_DRAFT_KD_H
#define DS4_CORE_DRAFT_KD_H
#include "core_internal.h"
#ifndef DS4_NO_GPU

typedef struct {
    char texts[1024], out[1024], cache[1024], init[1024];
    uint32_t rank, rank_exit;      /* 塔件秩(0 = 不训塔件) / 出口件秩(0 = 不训出口件) */
    uint32_t topk, epochs, batch_blocks, logits_rows, seed, gradcheck, max_steps, prof, prompt_tail;
    uint32_t loss_kl;              /* 0 = 总变差(缺省; 目标 = 1 − 期望接受率, 教师 top-K 到 4096) / 1 = 前向 KL(③ 的核, K ≤ 128; 只作对照, 10-07 实撞见 cuda_draft_attn) */
    uint32_t markov;               /* 1 = markov 偏置表(embd [Vm][R] + head [V][R], 原件 f32/bf16 复制成 f32 设备表)也当训练变量(与上下文无关, 跨请求可迁移) */
    float lr, clip, init_std, temp;   /* temp: 损失与陪审团的温度(缺省 1 = 模型卡配方) */
    float lr_markov;               /* 偏置表的学习率(缺省 = lr) */
} dk_cfg;

/* 一份文本(ids = 提示 + 续写, n_prompt = 提示长)。缓存: main_x[n][E] f32、教师 top-K [n][K] + 余量 [n];
 * 留出文本另存 [n−n_prompt][V] f32 教师 logits(陪审团要全词表)。
 * 块 i(吃 ids[i], 首位坐在位置 i, 窗口 = 位置 < i 的 main_x, 第 j 位的教师 = 底座第 i+j 行)的可用区间 [b_lo, b_hi)。 */
typedef struct {
    char path[1024]; int eval; uint32_t n_prompt;
    int32_t *ids; uint32_t n;
    int fd; uint64_t o_mx, o_tid, o_tp, o_trest, o_tlog;   /* 缓存文件只按需 pread(每批几十 MB), 不整份进内存: 12 份文本整份进来 8 GB, 10-07 实撞看门狗 */
    uint32_t b_lo, b_hi;
} dk_text;
typedef struct { dk_text *t; uint32_t nt; } dk_data;
bool dk_text_pread(const dk_text *t, uint64_t off, void *buf, uint64_t bytes);   /* core_draft_kd_cap.c */

/* 一塔反传的缓冲(一次只活一塔: 重算中间量 + 梯度, R 行) */
typedef struct {
    ds4_gpu_tensor *mixa, *posta, *comba, *prea, *xa, *xna, *qr, *kv, *o, *hcin, *mixf, *postf, *combf, *pref, *xf, *xnf;
    ds4_gpu_tensor *g_hcm, *g_hcin, *g_y, *g_xnf, *g_xf, *g_pre_a, *g_postf, *g_combf, *g_attn, *g_posta, *g_comba, *g_low, *g_o, *g_q,
                   *g_kvn, *g_qrn, *g_qr, *g_kv, *g_xna, *g_xa, *g_prein, *g_sh, *g_sg, *g_su, *g_rw, *g_glog, *g_blk;
} dk_layer_buf;

typedef struct {
    ds4_v41_state st;              /* 草稿态(draft=1, 层号 = 塔号), R 行; st.ampA/B[T] = 塔件, st.ampT 中间 */
    v41_tsave save;                /* 只用 sel[塔号]: 梯度检查的冻结选择前向(replay=1 时各塔路由照基准前向存的选, 有限差分才不吃并列边翻转) */
    uint32_t R, B, NT, K, Kx, Lr;  /* 行数 / 块长 / 塔数 / 塔件秩 / 出口件秩 / 出口头一次算几行 */
    ds4_gpu_tensor *gA[DS4_MTP_MAX_TOWERS], *gB[DS4_MTP_MAX_TOWERS], *mA[DS4_MTP_MAX_TOWERS], *vA[DS4_MTP_MAX_TOWERS],
                   *mB[DS4_MTP_MAX_TOWERS], *vB[DS4_MTP_MAX_TOWERS];
    ds4_gpu_tensor *xA, *xB, *gxA, *gxB, *mxA, *vxA, *mxB, *vxB, *xT, *gxT;   /* 出口件 [Kx][E] + Adam + 中间 [R][Kx] */
    ds4_gpu_tensor *gT;            /* 塔件反向的中间 gT = gy·Aᵀ [R][K](T = x·Bᵀ 借 st.ampT) */
    ds4_gpu_tensor *mkE, *mkH, *gmkE, *gmkH, *mE, *vE, *mH, *vH, *ge;   /* markov 偏置表 [Vm][Rk] / [V][Rk] + 梯度 + Adam + 反向中间 ge [Lr][Rk]; NULL = 不训 */
    uint32_t Vm, Rk;
    ds4_gpu_tensor *hist[DS4_MTP_MAX_TOWERS]; uint32_t hist_rows;   /* 每塔历史窗口 KV [hist_rows][HD](本批) */
    uint32_t hbase, nb;            /* 本批: hist 第 0 行的绝对位置 / 块数 */
    ds4_gpu_tensor *mx_dev, *posh, *bpos;   /* 本批历史 main_x [hist_rows][E] / 历史位置 i32 [hist_rows] / 块首位 i32 [nb] */
    ds4_gpu_tensor *lse[DS4_MTP_MAX_TOWERS];                        /* 每塔注意力的 lse [R][NH](反向用) */
    ds4_gpu_tensor *hc_in[DS4_MTP_MAX_TOWERS], *pm_in[DS4_MTP_MAX_TOWERS];   /* 每塔入口 hc [R][HC][E] f32 / pre_mix [R][HC](反传重算起点) */
    ds4_gpu_tensor *h, *xn_pre, *g_xn, *g_xn_pre, *g_h, *ghc, *gpre;   /* 出口链 */
    ds4_gpu_tensor *mk_ids, *mk_emb, *mk_bias;                     /* markov: 各行"前一个真 token" i32 [R] / embed [R][rank] / 偏置 [Lr][V] */
    ds4_gpu_tensor *tid, *tp, *trest, *loss, *tlog_dev, *jury;    /* 教师表 [R][K] / 逐行损失 [R] / 留出教师 logits [Lr][V] / 陪审团 [Lr][4] */
    float *loss_h, *jury_h; int32_t *bpos_h, *mk_h;
    float *mx_h, *tt_p, *tt_rest, *tl_h; int32_t *tt_id;   /* 主机暂存: 本批历史 main_x / 教师表连续段 [nb+B][K] / 留出教师 logits 连续段 */
    uint32_t tl_rows;
    dk_layer_buf lb;
    double tm[8]; uint32_t tm_n;   /* prof: 窗口 KV / 塔前向 / 出口 / 出口反传 / 塔反传 / 整批 */
} dk_run;
enum { DK_TM_WIN, DK_TM_TOWERS, DK_TM_EXIT, DK_TM_EXITB, DK_TM_TOWERB, DK_TM_BATCH };

/* core_draft_kd.c */
bool dk_cfg_load(dk_cfg *c, const char *path);
/* core_draft_kd_cap.c: 料清单 + 缓存(没有就跑底座取) */
bool dk_data_load(ds4_engine *e, const dk_cfg *c, dk_data *d);
void dk_data_free(dk_data *d);
/* core_draft_kd_fwd.c */
bool dk_run_alloc(ds4_engine *e, const dk_cfg *c, dk_run *r);
void dk_run_free(dk_run *r);
bool dk_tower_fwd(ds4_engine *e, dk_run *r, uint32_t T);   /* 一塔前向(st->hc 入 → st->hc 出), 块注意力走批量核; 反传重算也用它 */
/* 一批: 文本 t 的块 [i0, i0+nb)。grad=1 反传(梯度累加进 gA/gB/gxA/gxB); jury 非 NULL 时(留出文本)累加陪审团 [B][3] 与行数 */
bool dk_batch(ds4_engine *e, const dk_cfg *c, dk_run *r, const dk_text *t, uint32_t i0, uint32_t nb, int grad,
              double *loss_sum, uint32_t *nrows, double *jury, uint32_t *jrows);
/* core_draft_kd_bwd.c */
bool dk_layer_alloc(dk_run *r);
void dk_layer_free(dk_run *r);
bool dk_exit_bwd(ds4_engine *e, dk_run *r);                 /* g_xn(头的转置乘已填) → 出口件梯度 → out_norm → hc_pre → r->ghc/gpre */
bool dk_towers_bwd(ds4_engine *e, dk_run *r, uint32_t nb);  /* 从末塔到塔 0 逐塔重算 + 倒推, 塔件梯度累加进 gA/gB */
bool dk_save(const dk_cfg *c, dk_run *r, const char *dir);  /* 件落盘: tower_Tn.bin({E,K,1}+A+B) + exit.dspa + base.fnv */

#endif
#endif
