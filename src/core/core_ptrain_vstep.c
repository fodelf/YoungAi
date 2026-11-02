/* core_ptrain_vstep.c — V 步: 端到端重选专家权重的离散码字(2026-11-02)。总述见 core_ptrain.h 的 vq_* 字段。
 *
 * 为什么: 1.5 bit 专家丢的是尾巴(判决份 KLD>2 的位、鹈鹕"喉"位 0.012 对 FP 0.88), 两周里 ②(行增益/路由偏置)、码本 δ(10-29)、低秩 ③(六轮)全只动连续参数,
 *   均值动 1%、尾巴 0。PV-Tuning 在 ≤2 bit 上的结论: 收益在"V 步"= 按一阶线性化重选每组码字, 每步只改一小部分, 下一步重算梯度再选。这里就是那一步。
 * 一轮(= cfg 的一个 epoch)做什么:
 *   ① 取 vq_bs 道训练题当这一轮的批; 路由计数(第 1 轮先单独过一遍反传数, 之后用上一轮攒的)挑每层最热的 vq_hot 个专家(收益 ∝ 路由份额, 10-21 拆账);
 *   ② 每 vq_group 层一组: 开 dw 缓冲(槽 = 组内层 × 热门专家), 反传过这一批把 gate/up/down 的按组梯度攒进去(cuda_bwd_vgrad);
 *   ③ 每矩阵按组在层码本的 vq_knn 个近邻里选一阶 Δ 最小的码字(cuda_vq_vstep), Δ<0 的按 Δ 排序只取最负的 vq_frac 份;
 *   ④ 第 1 轮 vq_check=1 先做机制审计: 只盖全局最负的 64 组, 同一批不带梯度再算一次损失, 比"一阶预测 ΣΔ"与"实测 ΔL"(符号必须同, 量级同数量级), 反了就停;
 *   ⑤ 盖补丁(改 GPU 读的那份 blob 的索引位, 盘上不动, 逐条回读 ≠0 停车) → 留出评估 → 落 <out>/ckpt_vNN/vqpatch_Lnn.bin(记录: 专家|矩阵, 行, 列组, 新码|旧码)。
 * 不训 ③ 的低秩(A 留在 0, 梯度攒了也不步进、每趟清零), 这一轮产物只有索引补丁 —— 一件事一个产物, 读数不混。
 * 出错会怎样: 层 blob 不是 v3 / 近邻表建不出 / 回读不同 ⇒ 返回 false 停车; budget_s 到了按轮收(与 ③ 同一个闸)。 */
#include "core_ptrain.h"
#include "../common/ds4_gr_fnv.h"
#include "../../vq_fmt.h"
#include <sys/stat.h>
#include <math.h>
#ifndef DS4_NO_GPU

typedef struct { uint32_t ew, row, j, code; } pt_vrec;   /* 与 cuda_vq_vstep.inc.cu 的 vq_prec 同布局 */
typedef struct { pt_vrec *v; uint32_t n, cap; } pt_vlog;
typedef struct { float d; uint32_t idx; } pt_vcand;
typedef struct { float d; uint32_t il; pt_vrec r; } pt_vmove;

static int vcand_cmp(const void *a, const void *b) { const float x = ((const pt_vcand *)a)->d, y = ((const pt_vcand *)b)->d; return x < y ? -1 : x > y; }
static int vmove_cmp(const void *a, const void *b) { const float x = ((const pt_vmove *)a)->d, y = ((const pt_vmove *)b)->d; return x < y ? -1 : x > y; }
static int vrec_key_cmp(const void *a, const void *b) {   /* 键 (ew,row,j) 升序; 同键按原序(stable: 调用方把序号塞在 code 的高位之外另存, 见 vlog_save) */
    const pt_vrec *x = a, *y = b;
    if (x->ew != y->ew) return x->ew < y->ew ? -1 : 1;
    if (x->row != y->row) return x->row < y->row ? -1 : 1;
    return x->j < y->j ? -1 : x->j > y->j;
}

static const ds4_tensor *vq_blob(uint32_t il) { char nm[64]; snprintf(nm, sizeof nm, "blk.%u.ffn_exps_vq.blob", il); return model_find_tensor(g_ds4_v41_model, nm); }

static void vlog_push(pt_vlog *l, pt_vrec r) {
    if (l->n == l->cap) { l->cap = l->cap ? l->cap * 2u : 4096u; l->v = realloc(l->v, (size_t)l->cap * sizeof(pt_vrec)); if (!l->v) { fprintf(stderr, "ds4: [V步] 补丁日志内存\n"); exit(1); } }
    l->v[l->n++] = r;
}

/* 落盘: 同一组多次改过只留一条 —— 旧码取最早一次(盘上原样), 新码取最后一次; 文件 = 头 {'VQP1', 层, 条数, IN, MID, OUT} + 记录 */
static bool vlog_save(const pt_vlog *l, const char *dir, uint32_t il, uint32_t IN, uint32_t MID, uint32_t OUT) {
    if (!l->n) return true;
    pt_vrec *s = xmalloc((size_t)l->n * sizeof(pt_vrec)); uint32_t *ordr = xmalloc((size_t)l->n * 4u);
    for (uint32_t i = 0; i < l->n; i++) { s[i] = l->v[i]; ordr[i] = i; }
    /* 稳定排序: 先按键 qsort, 同键段内再按原序号冒泡(段很短) */
    typedef struct { pt_vrec r; uint32_t o; } kr; kr *k = xmalloc((size_t)l->n * sizeof(kr));
    for (uint32_t i = 0; i < l->n; i++) { k[i].r = s[i]; k[i].o = i; }
    qsort(k, l->n, sizeof(kr), vrec_key_cmp);
    uint32_t out_n = 0;
    for (uint32_t i = 0; i < l->n;) {
        uint32_t j = i; while (j < l->n && !vrec_key_cmp(&k[i], &k[j])) j++;
        uint32_t first = i, last = i;
        for (uint32_t t = i; t < j; t++) { if (k[t].o < k[first].o) first = t; if (k[t].o > k[last].o) last = t; }
        pt_vrec r = k[last].r; r.code = (r.code & 0xffffu) | (k[first].r.code & 0xffff0000u);
        if ((r.code & 0xffffu) != (r.code >> 16)) s[out_n++] = r;   /* 改回原码的不写 */
        i = j;
    }
    free(k); free(ordr);
    char p[4200]; snprintf(p, sizeof p, "%s/vqpatch_L%02u.bin", dir, il);
    FILE *f = fopen(p, "wb");
    const uint32_t hd[6] = { 0x31505156u, il, out_n, IN, MID, OUT };
    const bool ok = f && fwrite(hd, 4, 6, f) == 6 && (out_n == 0 || fwrite(s, sizeof(pt_vrec), out_n, f) == out_n);
    if (f) fclose(f);
    free(s);
    if (!ok) fprintf(stderr, "ds4: [V步] 写不了 %s\n", p);
    return ok;
}

/* 一趟: 这一轮的批按 c->batch 分块过 pt_run_list; grad=1 攒 dw(放大器梯度也攒了, 过完清零不步进)。
 * 返回 Σ_题 (该题答案位平均 KL) / c->batch —— 与梯度同一缩放(pt_loss_rows: 每题 1/(m·batch)), 一阶预测 ΣΔ 才能和它直接比。
 * 11-02 实撞: 第一版按 token 求和的 ls 算, 审计"预测 −0.0030 实测 −0.358, 比值 118"= 平均每题 118 个答案位, 不是机制错 */
static bool vs_pass(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d, const uint32_t *bi, uint32_t nb, int grad, double *loss) {
    const pt_sample **bs = xmalloc((size_t)c->batch * sizeof *bs);
    double *li = xmalloc((size_t)c->batch * 8u); uint32_t *ni = xmalloc((size_t)c->batch * 4u);
    double tot = 0;
    /* 尾权只在 V 步自己的趟里开(选码梯度 / 审计 / 本批前后), 留出评估(pt_eval)仍是老口径 —— 门与历史读数可比 */
    if (!ds4_gpu_bwd_kl_tail(c->vq_tail, 16.f)) { fprintf(stderr, "ds4: [V步] 这个后端不支持尾权(vq_tail)\n"); free(bs); free(li); free(ni); return false; }
    for (uint32_t b0 = 0; b0 < nb; b0 += c->batch) {
        const uint32_t b1 = b0 + c->batch < nb ? b0 + c->batch : nb;
        for (uint32_t b = b0; b < b1; b++) { bs[b - b0] = &d->s[bi[b]]; li[b - b0] = 0; ni[b - b0] = 0; }
        double ls = 0; uint32_t nt = 0;
        if (!pt_run_list(e, c, r, bs, b1 - b0, grad, &ls, &nt, li, ni, NULL)) { ds4_gpu_bwd_kl_tail(0.f, 0.f); free(bs); free(li); free(ni); return false; }
        for (uint32_t b = b0; b < b1; b++) if (ni[b - b0]) tot += li[b - b0] / (double)ni[b - b0];
    }
    ds4_gpu_bwd_kl_tail(0.f, 0.f);
    free(bs); free(li); free(ni);
    if (grad) for (uint32_t il = c->layer_lo; il <= c->layer_hi; il++)
        if (!ds4_gpu_tensor_fill_f32(r->gA[il], 0.f, (uint64_t)r->K * DS4_N_EMBD) || !ds4_gpu_tensor_fill_f32(r->gB[il], 0.f, (uint64_t)r->K * DS4_N_EMBD)) return false;
    *loss = tot / (double)c->batch;
    return true;
}

static bool vs_patch(const ds4_model *m, uint32_t il, const pt_vrec *recs, uint32_t n, uint32_t E, uint32_t FF) {
    if (!n) return true;
    const ds4_tensor *t = vq_blob(il); uint32_t mis = 0;
    if (!t || !ds4_gpu_v41_vq_code_patch(m->map, m->size, t->abs_offset, t->bytes, E, FF, E, (const uint32_t *)recs, n, &mis)) { fprintf(stderr, "ds4: ★[V步] L%02u 补丁盖不上★\n", il); return false; }
    if (mis) { fprintf(stderr, "ds4: ★[V步] L%02u 补丁回读 %u/%u 条与目标不同, 停车★\n", il, mis, n); return false; }
    return true;
}

bool pt_vstep_run(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d, const uint32_t *ord, uint32_t ntr, double kl_ev0, FILE *lf, double t0) {
    const ds4_model *m = g_ds4_v41_model;
    const uint32_t E = DS4_N_EMBD, FF = DS4_N_FF_EXP, NE = DS4_N_EXPERT, NL = c->vq_nl, K = c->vq_knn, HOT = c->vq_hot, GRP = c->vq_group ? c->vq_group : 1u;
    if (!m || !NL || !HOT || !K || K > 32u || c->vq_bs < c->batch) { fprintf(stderr, "ds4: [V步] 配置不全(vq_layers / vq_hot / vq_knn≤32 / vq_bs≥batch)\n"); return false; }
    const uint64_t slot_floats = 2ull * FF * E + (uint64_t)E * FF;
    /* 每层: 码本近邻表 */
    uint16_t *nbr[DS4_MAX_LAYER] = { 0 }; uint32_t ncl[DS4_MAX_LAYER] = { 0 };
    for (uint32_t i = 0; i < NL; i++) {
        const uint32_t il = c->vq_l[i]; const ds4_tensor *t = vq_blob(il); uint64_t cbo = 0;
        if (!t || !ds4vq_v3_codebook((const uint8_t *)m->map + t->abs_offset, (size_t)t->bytes, &cbo, &ncl[il])) { fprintf(stderr, "ds4: [V步] L%02u 专家 blob 不是 v3\n", il); return false; }
        nbr[il] = xmalloc((size_t)ncl[il] * K * 2u);
        if (!ds4_gpu_v41_vq_knn((const uint8_t *)m->map + t->abs_offset + cbo, ncl[il], K, nbr[il])) { fprintf(stderr, "ds4: [V步] L%02u 近邻表失败\n", il); return false; }
    }
    fprintf(stderr, "ds4: [V步] %u 层(L%u..), 每层热门 %u 专家, 近邻 %u, 每步改 %.3g 份, 批 %u 题, 每组 %u 层; 槽 %.0f MB × %u; 尾权 τ=%.3g%s\n",
            NL, c->vq_l[0], HOT, K, (double)c->vq_frac, c->vq_bs, GRP, (double)slot_floats * 4 / 1e6, HOT * GRP, (double)c->vq_tail, c->vq_tail > 0.f ? "(本批损失为加权口径)" : "(关)");
    uint32_t *counts = xmalloc_zeroed((size_t)DS4_MAX_LAYER * NE, 4);
    pt_vlog vlog[DS4_MAX_LAYER]; memset(vlog, 0, sizeof vlog);
    /* vq_init=<③ 目录>(11-02 深夜): 起点 = 这份补丁盖上后的 GPU 态(盘上仍不动), 记录先并进日志 ⇒ 落盘的每个 ckpt 自包含(旧码 = 盘上原样, 单独挂就是全部),
     * 不用合并两份补丁; 起点动的层不在 vq_layers 里照样原样存进去(存盘按"有记录的层")。知识留出的第 0 步在它之前算的是裸态, 这里按起点态重算, 门才是对起点 */
    if (c->vq_init[0]) {
        if (v41_vqpatch_load(c->vq_init) < 0) { fprintf(stderr, "ds4: ★[V步] 起点补丁 %s 挂不上★\n", c->vq_init); return false; }
        uint64_t seeded = 0;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) { uint32_t n = 0; const pt_vrec *v = v41_vqpatch_records(il, &n); for (uint32_t q = 0; q < n; q++) vlog_push(&vlog[il], v[q]); seeded += n; }
        const double kl_bare = kl_ev0;
        if (!pt_eval(e, c, r, d, 1, UINT32_MAX, &kl_ev0, 0, NULL, NULL, NULL)) return false;
        fprintf(stderr, "ds4: [V步] 起点 = %s(%llu 组已换码, 并入日志); 知识留出 裸态 %.4f → 起点态 %.4f\n", c->vq_init, (unsigned long long)seeded, kl_bare, kl_ev0);
    }
    /* 样本分两类(11-02 用户: "编程指标不掉或提升 + 细节还原", 两个目标都写进选码门): 流段(fpkey ≠ 0 —— ids 行 / 长原文的接力段, 判决尺那种编程料)与短题(其余, 实体知识行)。
     * 每轮: 第 0 份 = 一篇流的全部段按篇序(接力, 整篇 1.5 万位全是靶), 其余份 = 短题按 batch 切; 门 = 全部份都为负 ⇒ 一组码字必须编程份和知识份同时说"该换"。 */
    uint32_t *pool_doc = xmalloc((size_t)ntr * 4u), npd = 0, *pool_short = xmalloc((size_t)ntr * 4u), nps = 0;
    /* 流 = 整篇长过 maxlen 的原文(被切成多段接力); 短题 = 整篇装得下一窗的(fpkey 不能当判据: 前缀形态下短行也带整篇键, 11-02 实撞 315 短行全被当成流, 份数退成 1) */
    for (uint32_t i = 0; i < ntr; i++) { const pt_sample *s = &d->s[ord[i]]; if (s->fpkey && d->docn[s->doc] > c->maxlen) pool_doc[npd++] = ord[i]; else pool_short[nps++] = ord[i]; }
    uint32_t doc_st[256], doc_n[256], ndocs = 0;   /* 文档按篇分组(样本号升序 = 篇内段序) */
    if (npd) {
        for (uint32_t i = 1; i < npd; i++) { const uint32_t v = pool_doc[i]; uint32_t j = i; while (j && pool_doc[j - 1] > v) { pool_doc[j] = pool_doc[j - 1]; j--; } pool_doc[j] = v; }
        for (uint32_t i = 0; i < npd; i++) { if (!ndocs || d->s[pool_doc[i]].doc != d->s[pool_doc[doc_st[ndocs - 1]]].doc) { if (ndocs == 256) break; doc_st[ndocs] = i; doc_n[ndocs] = 0; ndocs++; } doc_n[ndocs - 1]++; }
    }
    /* 文档两类(11-02 晚): ids 流(判决尺那种裸 token 流, 第 0 个 id 顶 BOS ⇒ sids[0] ≠ BOS)每轮 1 篇一份; 前缀文档(BOS 开头的长行: agent 头行 10k 前缀 + 官方思考头 2k 靶)
     * 每轮 vq_docs 篇一份 —— 一篇只有 1~3 段, 单篇一份靶位太少选不出码字(v01 教训: 份内靶位少 = 噪声赢家) */
    uint32_t sdoc[256], pdoc[256], nsd = 0, npdc = 0;
    for (uint32_t k = 0; k < ndocs; k++) { if (d->s[pool_doc[doc_st[k]]].sids[0] == e->vocab.bos_id) pdoc[npdc++] = k; else sdoc[nsd++] = k; }
    const uint32_t NDOC = c->vq_docs ? c->vq_docs : 1u;
    const uint32_t Ss = nps ? c->vq_bs / c->batch : 0, S = Ss + (nsd ? 1u : 0u) + (npdc ? 1u : 0u);
    if (!S || (nps && c->vq_bs % c->batch)) { fprintf(stderr, "ds4: [V步] 料里既没有流段也没有短题, 或 vq_bs %u 不是 batch %u 的整数倍\n", c->vq_bs, c->batch); return false; }
    uint32_t *sl = xmalloc(((size_t)c->vq_bs + npd + 1u) * 4u), sh_off[258], cur = 0, pcur = 0; uint64_t rs = 0x9E3779B97F4A7C15ull ^ c->seed;
    uint32_t nevg = 0; for (uint32_t i = 0; i < d->ns; i++) nevg += d->s[i].eval == 2 && d->s[i].top_id != NULL;
    fprintf(stderr, "ds4: [V步] 料: 流 %u 篇 + 前缀文档 %u 篇(共 %u 段) + 短题 %u; 每轮 %u 份(流 %u + 前缀文档 %u × %u 篇 + 短题 %u × %u 题); 泛化留出(eval=)%u 段\n",
            nsd, npdc, npd, nps, S, nsd ? 1u : 0u, npdc ? 1u : 0u, NDOC, Ss, c->batch, nevg);
    uint32_t *counts_doc = xmalloc_zeroed((size_t)DS4_MAX_LAYER * NE, 4);   /* 流份 / 短题份各自计数, 归一后相加挑热门(流 1.5 万位会把短题的 2 千位淹掉) */
    float *delta = NULL; uint16_t *ncode = NULL, *ccode = NULL; pt_vcand *cand = NULL; uint64_t cap_g = 0;
    pt_vmove *moves = NULL; uint32_t nmv = 0, capmv = 0;
    bool ok = true; uint64_t moved_all = 0;
    double frac = c->vq_frac, kl_prev = kl_ev0, kg_prev = 0;   /* 信赖域: 一轮两边留出任一不降就整轮回退、份额减半(下一轮重选) */
    /* 判决份门改配对(11-02 晚, 用户 "先试一试"): 16 段逐段与部署态配对, D = Σ(部署_i − 本轮_i)/Σm_i, SE 照 pt_paired_gain(题当独立单位, 比值估计线性化);
     * 门 = D ≥ −2·SE(不显著比部署态差)。为什么: v14 两轮判决份 +0.0001~0.0002 全被"一位不许高"拦下, 而 16 段对全尺 15360 位只是抽样, 同向升才算退, 异号乱跳是抽样噪声 */
    /* pt_eval 的逐题数组按入选顺序只填前 k 项(k = 判决份段数), 其余必须是 0 才能用 ni>0 筛 —— v15 轮 1 实撞: 没清零读到 408 "段"、D 天文数字、SE inf */
    double *kg_li0 = xmalloc_zeroed(d->ns + 1, 8), *kg_li = xmalloc_zeroed(d->ns + 1, 8); uint32_t *kg_ni0 = xmalloc_zeroed(d->ns + 1, 4), *kg_ni = xmalloc_zeroed(d->ns + 1, 4);
    r->tail_on = 1; r->tail_n = r->tail_n1 = r->tail_n2 = 0;
    if (nevg && !pt_eval(e, c, r, d, 2, UINT32_MAX, &kg_prev, 0, kg_li0, kg_ni0, NULL)) return false;
    r->tail_on = 0;
    const double kg0 = kg_prev; const uint32_t t1_0 = r->tail_n1, t2_0 = r->tail_n2;
    if (nevg) fprintf(stderr, "ds4: [V步] 第 0 步泛化留出(判决份) %.4f; 尾位 KL>2 %u / KL>1 %u(共 %u 位)\n", kg_prev, t2_0, t1_0, r->tail_n);
    for (uint32_t rd = 1; rd <= c->epochs && ok; rd++) {
        /* ① 批: 份 0 = 第 (rd−1) mod 篇数 篇的全部段; 其余份 = 短题轮转取(取完打乱, 种子固定) */
        uint32_t k = 0, sh = 0;
        if (nsd) { const uint32_t dk = sdoc[(rd - 1u) % nsd]; sh_off[0] = 0; for (uint32_t i = 0; i < doc_n[dk]; i++) sl[k++] = pool_doc[doc_st[dk] + i]; sh = 1; }
        if (npdc) {   /* 前缀文档份: 接着上轮的位置取 NDOC 篇(轮转, 不打乱: 篇序 = 料序) */
            sh_off[sh++] = k;
            for (uint32_t q = 0; q < NDOC && q < npdc; q++) { const uint32_t dk = pdoc[pcur]; pcur = (pcur + 1u) % npdc; for (uint32_t i = 0; i < doc_n[dk]; i++) sl[k++] = pool_doc[doc_st[dk] + i]; }
        }
        for (uint32_t s = 0; s < Ss; s++) {
            sh_off[sh++] = k;
            for (uint32_t i = 0; i < c->batch; i++) {
                if (cur >= nps) { for (uint32_t q = nps; q > 1; q--) { rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27; const uint32_t j = (uint32_t)((rs * 2685821657736338717ull) % q); const uint32_t t = pool_short[q - 1]; pool_short[q - 1] = pool_short[j]; pool_short[j] = t; } cur = 0; }
                sl[k++] = pool_short[cur++];
            }
        }
        sh_off[S] = k;
        double l0 = 0;
        memset(counts, 0, (size_t)DS4_MAX_LAYER * NE * 4u); memset(counts_doc, 0, (size_t)DS4_MAX_LAYER * NE * 4u);
        if (rd == 1) for (uint32_t s = 0; s < S && ok; s++) {   /* 只为数路由(第 1 轮没有上一轮的账) */
            double ls = 0; ds4_gpu_bwd_moe_count(s == 0 && nsd ? counts_doc : counts, DS4_MAX_LAYER, NE);
            ok = vs_pass(e, c, r, d, sl + sh_off[s], sh_off[s + 1] - sh_off[s], 1, &ls);
        }
        if (!ok) break;
        uint32_t hot[DS4_MAX_LAYER][64];
        for (uint32_t i = 0; i < NL; i++) {   /* 每层: 流份与短题份各归一再相加, 取前 HOT 个 */
            const uint32_t il = c->vq_l[i]; const uint32_t *cd = counts_doc + (size_t)il * NE, *cs = counts + (size_t)il * NE;
            double td = 0, ts = 0; for (uint32_t x = 0; x < NE; x++) { td += cd[x]; ts += cs[x]; }
            for (uint32_t h = 0; h < HOT; h++) {
                uint32_t best = UINT32_MAX; double bs = -1;
                for (uint32_t x = 0; x < NE; x++) {
                    bool used = false; for (uint32_t y = 0; y < h; y++) used |= hot[il][y] == x;
                    const double sc = (td > 0 ? cd[x] / td : 0) + (ts > 0 ? cs[x] / ts : 0);
                    if (!used && sc > bs) { bs = sc; best = x; }
                }
                hot[il][h] = best;
            }
        }
        nmv = 0; double pred_all = 0; uint64_t scanned = 0, agree_n = 0;
        /* ②③ 分组攒梯度 + 选码。★批切成 S 份(每份 = 一个 batch), 每份单独攒梯度、单独选码, 只留 S 份全为负且全选同一码字的组, 排序按最弱的那份★
         * 为什么(11-02 两次实撞): v01 一份梯度取 1% = 17.7 万组本批 −、留出 +17%; v03 两半一致门仍过 10% 的组, 十层 17.7 万组留出 +14% ——
         * 几十万个离散自由度对着两千个答案位挑, 极值全是噪声赢家(winner's curse); 份数越多、门越严, 同样的 GPU 代价(总样本数不变), 只多几次选码和主机归并。 */
        for (uint32_t g0 = 0; g0 < NL && ok; g0 += GRP) {
            const uint32_t g1 = g0 + GRP < NL ? g0 + GRP : NL, nsl = (g1 - g0) * HOT, nmat = nsl * 3u;
            ds4_gpu_tensor *dw = ds4_gpu_tensor_alloc(slot_floats * nsl * 4u);
            if (!dw) { fprintf(stderr, "ds4: [V步] dw %.1f GB 分不出\n", (double)slot_floats * nsl * 4 / 1e9); ok = false; break; }
            int32_t *so = xmalloc((size_t)NE * 4u);
            for (uint32_t i = g0; i < g1; i++) {
                const uint32_t il = c->vq_l[i];
                for (uint32_t x = 0; x < NE; x++) so[x] = -1;
                for (uint32_t h = 0; h < HOT; h++) so[hot[il][h]] = (int32_t)((i - g0) * HOT + h);
                r->vg_slot[il] = ds4_gpu_tensor_alloc((uint64_t)NE * 4u);
                if (!r->vg_slot[il] || !ds4_gpu_tensor_write(r->vg_slot[il], 0, so, (uint64_t)NE * 4u)) ok = false;
            }
            free(so);
            /* 组内每矩阵的归并区: 最弱 Δ / Δ 和 / 第一份选的码 / 全份一致标志 */
            float **dmax = xmalloc_zeroed(nmat, sizeof(float *)), **dsum = xmalloc_zeroed(nmat, sizeof(float *)); uint16_t **code0 = xmalloc_zeroed(nmat, sizeof(uint16_t *)); uint8_t **agree = xmalloc_zeroed(nmat, sizeof(uint8_t *));
            memset(counts, 0, (size_t)DS4_MAX_LAYER * NE * 4u); memset(counts_doc, 0, (size_t)DS4_MAX_LAYER * NE * 4u);   /* 这一趟的计数给下一轮挑专家 */
            l0 = 0;
            for (uint32_t s = 0; s < S && ok; s++) {
                double ls = 0;
                if (!ds4_gpu_tensor_fill_f32(dw, 0.f, slot_floats * nsl)) { ok = false; break; }
                ds4_gpu_bwd_moe_count(s == 0 && nsd ? counts_doc : counts, DS4_MAX_LAYER, NE);
                r->vg_dw = dw; ok = vs_pass(e, c, r, d, sl + sh_off[s], sh_off[s + 1] - sh_off[s], 1, &ls); r->vg_dw = NULL;
                if (!ok) break;
                l0 += ls;
                for (uint32_t i = g0; i < g1 && ok; i++) {
                    const uint32_t il = c->vq_l[i]; const ds4_tensor *t = vq_blob(il);
                    for (uint32_t h = 0; h < HOT && ok; h++) {
                        const uint32_t ex = hot[il][h], slot = (i - g0) * HOT + h;
                        for (uint32_t which = 0; which < 3u && ok; which++) {
                            const uint32_t R = which == 2u ? E : FF, C = which == 2u ? FF : E, ng = R * (C / 8u), mi = slot * 3u + which;
                            if ((uint64_t)ng > cap_g) { cap_g = ng; free(delta); free(ncode); free(ccode); free(cand); delta = xmalloc((size_t)ng * 4u); ncode = xmalloc((size_t)ng * 2u); ccode = xmalloc((size_t)ng * 2u); cand = xmalloc((size_t)ng * sizeof(pt_vcand)); }
                            if (!dmax[mi]) { dmax[mi] = xmalloc((size_t)ng * 4u); dsum[mi] = xmalloc_zeroed(ng, 4); code0[mi] = xmalloc((size_t)ng * 2u); agree[mi] = xmalloc((size_t)ng); memset(agree[mi], 1, ng); for (uint32_t g = 0; g < ng; g++) dmax[mi][g] = -3.4e38f; }
                            const uint64_t off = (uint64_t)slot * slot_floats + (which == 0u ? 0ull : which == 1u ? (uint64_t)FF * E : 2ull * FF * E);
                            if (!ds4_gpu_v41_vq_vselect(dw, off, m->map, m->size, t->abs_offset, t->bytes, ex, which, R, C, NULL, nbr[il], ncl[il], K, delta, ncode, ccode)) { ok = false; break; }
                            float *dm = dmax[mi], *ds = dsum[mi]; uint16_t *c0 = code0[mi]; uint8_t *ag = agree[mi];
                            if (s == 0) for (uint32_t g = 0; g < ng; g++) { c0[g] = ncode[g]; dm[g] = delta[g]; ds[g] = delta[g]; ag[g] = delta[g] < 0.f; }
                            else for (uint32_t g = 0; g < ng; g++) { ds[g] += delta[g]; if (delta[g] > dm[g]) dm[g] = delta[g]; if (!(delta[g] < 0.f) || ncode[g] != c0[g]) ag[g] = 0; }
                            if (s == S - 1u) {   /* 最后一份: 归并出候选 */
                                uint32_t nc = 0;
                                for (uint32_t g = 0; g < ng; g++) if (ag[g]) { cand[nc].d = dm[g]; cand[nc].idx = g; nc++; }
                                agree_n += nc;
                                qsort(cand, nc, sizeof(pt_vcand), vcand_cmp);
                                uint32_t take = (uint32_t)((double)ng * (double)frac); if (take > nc) take = nc;
                                for (uint32_t q = 0; q < take; q++) {
                                    if (nmv == capmv) { capmv = capmv ? capmv * 2u : 65536u; moves = realloc(moves, (size_t)capmv * sizeof(pt_vmove)); if (!moves) { fprintf(stderr, "ds4: [V步] 候选内存\n"); exit(1); } }
                                    const uint32_t gidx = cand[q].idx;
                                    pt_vmove mv; mv.d = ds[gidx]; mv.il = il;   /* 预测用整批(各份之和), 排序用的是最弱一份 */
                                    mv.r.ew = ex | (which << 16); mv.r.row = gidx / (C / 8u); mv.r.j = gidx % (C / 8u); mv.r.code = (uint32_t)c0[gidx] | ((uint32_t)ccode[gidx] << 16);
                                    moves[nmv++] = mv; pred_all += mv.d;
                                }
                                scanned += ng;
                            }
                        }
                    }
                }
            }
            for (uint32_t i = g0; i < g1; i++) { if (r->vg_slot[c->vq_l[i]]) ds4_gpu_tensor_free(r->vg_slot[c->vq_l[i]]); r->vg_slot[c->vq_l[i]] = NULL; }
            for (uint32_t mi = 0; mi < nmat; mi++) { free(dmax[mi]); free(dsum[mi]); free(code0[mi]); free(agree[mi]); }
            free(dmax); free(dsum); free(code0); free(agree);
            ds4_gpu_tensor_free(dw);
        }
        if (!ok) break;
        ds4_gpu_bwd_moe_count(NULL, 0, 0);
        qsort(moves, nmv, sizeof(pt_vmove), vmove_cmp);
        uint32_t applied = 0;
        /* ④ 机制审计: 只盖最负的 64 组, 实测 ΔL 对一阶预测 */
        if (rd == 1 && c->vq_check && nmv) {
            /* 基线另起一趟不带梯度的(b0[s] 按份), 盖完也按份量(b1[s]): 实测 = Σ(b1−b0) 与一阶预测同口径, 带/不带梯度前向若有系统差在"基线差"一列里现形,
             * 哪一份在升一眼能看 —— v07(11-02, batch 8)审计 −0.0025 对 +0.00055 只有总数, 分不清是梯度错、前向分叉还是某一份过冲 */
            /* 两档累计: 64 组(与 v02~v08 的历史读数同口径)只打不判, 2048 组才判停。v08 实撞: 基线差 0、流份(1.5 万靶位)−0.00023 符号对、三个短题份(各几十个靶位)
             * +0.0005 / +0.00097 / −0.0007 乱跳 —— 64 组的一阶信号 −0.0025 在路由翻转的噪声地板之下(MoE top-k 离散: 权重一动少数 token 换专家, 损失跳 ±1e-3,
             * 靶位多的份平均掉、少的份平均不掉); v06 整轮 7 万组预测 −0.249 实测 −0.227(0.91), 规模上去就对得上。 */
            static const uint32_t scale[2] = { 64u, 2048u };
            double l0n = 0, b0[258], b1[258], pred = 0, meas = 0;
            for (uint32_t s = 0; s < S && ok; s++) { ok = vs_pass(e, c, r, d, sl + sh_off[s], sh_off[s + 1] - sh_off[s], 0, &b0[s]); l0n += b0[s]; }
            for (uint32_t sc = 0; sc < 2u && ok; sc++) {
                const uint32_t na = nmv < scale[sc] ? nmv : scale[sc]; double l1 = 0;
                if (na <= applied) break;
                for (uint32_t i = 0; i < NL && ok; i++) {
                    const uint32_t il = c->vq_l[i]; pt_vrec tmp[2048]; uint32_t n = 0;
                    for (uint32_t q = applied; q < na; q++) if (moves[q].il == il) { tmp[n++] = moves[q].r; pred += moves[q].d; }
                    ok = vs_patch(m, il, tmp, n, E, FF);
                    for (uint32_t q = 0; q < n; q++) vlog_push(&vlog[il], tmp[q]);
                }
                for (uint32_t s = 0; s < S && ok; s++) { ok = vs_pass(e, c, r, d, sl + sh_off[s], sh_off[s + 1] - sh_off[s], 0, &b1[s]); l1 += b1[s]; }
                if (!ok) break;
                meas = l1 - l0n; applied = na;
                fprintf(stderr, "ds4: [V步·审计] 最负 %u 组: 一阶预测 ΔL %.6g, 实测 %.6g(%.6g → %.6g), 比值 %.3g; 基线差(带梯度 − 不带) %.3g; 各份实测 Δ:", na, pred, meas, l0n, l1, pred != 0 ? meas / pred : 0.0, l0 - l0n);
                for (uint32_t s = 0; s < S; s++) fprintf(stderr, " %.3g", b1[s] - b0[s]);
                fprintf(stderr, "%s\n", meas > 0 && sc ? " ★实测反而升: 梯度/补丁/读法有错, 停车★" : meas > 0 ? " (64 组在噪声地板下, 看 2048 组)" : "");
                if (lf) { fprintf(lf, "vcheck n %u pred %.6g meas %.6g base_gap %.6g\n", na, pred, meas, l0 - l0n); fflush(lf); }
            }
            if (!ok) break;
            if (meas > 0) { ok = false; break; }
        }
        /* ⑤ 其余补丁 */
        for (uint32_t i = 0; i < NL && ok; i++) {
            const uint32_t il = c->vq_l[i]; uint32_t n = 0;
            for (uint32_t q = applied; q < nmv; q++) n += moves[q].il == il;
            if (!n) continue;
            pt_vrec *tmp = xmalloc((size_t)n * sizeof(pt_vrec)); uint32_t k = 0;
            for (uint32_t q = applied; q < nmv; q++) if (moves[q].il == il) tmp[k++] = moves[q].r;
            ok = vs_patch(m, il, tmp, n, E, FF);
            for (uint32_t q = 0; q < n; q++) vlog_push(&vlog[il], tmp[q]);
            free(tmp);
        }
        if (!ok) break;
        moved_all += nmv;
        double kl = 0, l2 = 0;
        if (!vs_pass(e, c, r, d, sl, sh_off[S], 0, &l2)) { ok = false; break; }   /* 盖完本批再算一遍: 与留出并排, 分得清"过拟合"(本批降留出升)和"过冲"(两个都升) */
        if (!pt_eval(e, c, r, d, 1, UINT32_MAX, &kl, 0, NULL, NULL, NULL)) { ok = false; break; }
        double kg = 0;
        r->tail_on = 1; r->tail_n = r->tail_n1 = r->tail_n2 = 0;
        if (nevg && !pt_eval(e, c, r, d, 2, UINT32_MAX, &kg, 0, kg_li, kg_ni, NULL)) { ok = false; break; }
        r->tail_on = 0;
        double kgD = 0, kgSE = 0;   /* 对部署态的配对降幅 ± SE(正 = 比部署态好) */
        if (nevg) {
            double sd = 0, sm = 0, ss = 0; uint32_t q = 0;
            for (uint32_t i = 0; i < d->ns; i++) if (kg_ni0[i] && kg_ni[i]) { sd += kg_li0[i] - kg_li[i]; sm += kg_ni0[i]; q++; }
            kgD = sm > 0 ? sd / sm : 0;
            for (uint32_t i = 0; i < d->ns; i++) if (kg_ni0[i] && kg_ni[i]) { const double u = kg_li0[i] - kg_li[i] - kgD * kg_ni0[i]; ss += u * u; }
            kgSE = (q > 1 && sm > 0) ? sqrt(ss * q / (q - 1)) / sm : 0;
            fprintf(stderr, "ds4: [ptrain] V 步轮 %u 泛化留出(判决份) %.4f(上轮 %.4f, 第 0 步 %.4f, %+.2f%%); 对部署态配对 %+.5f ± %.5f(%u 段, %s); 尾位 KL>2 %u / KL>1 %u(第 0 步 %u / %u)\n",
                    rd, kg, kg_prev, kg0, kg0 > 0 ? (kg / kg0 - 1.0) * 100.0 : 0.0, kgD, kgSE, q, kgD < -2.0 * kgSE ? "显著退" : "不显著退", r->tail_n2, r->tail_n1, t2_0, t1_0);
        }
        fprintf(stderr, "ds4: [ptrain] ★V 步轮 %u 末: 扫 %llu 组, %u 份全同意 %llu, 改 %u 组(累计 %llu, 份额 %.4g); 一阶预测 ΣΔ %.5g, 本批损失 %.5f → %.5f(%+.2f%%); 留出损失 %.4f(上轮 %.4f, 第 0 步 %.4f, %+.1f%%), %.0f s★\n",
                rd, (unsigned long long)scanned, S, (unsigned long long)agree_n, nmv, (unsigned long long)moved_all, frac, pred_all, l0, l2, l0 > 0 ? (l2 / l0 - 1.0) * 100.0 : 0.0,
                kl, kl_prev, kl_ev0, kl_ev0 > 0 ? (kl / kl_ev0 - 1.0) * 100.0 : 0.0, now_sec() - t0);
        if (lf) { fprintf(lf, "vround %u eval_kl %.5f moved %u pred %.6g batch_loss %.5f after %.5f frac %.5g\n", rd, kl, nmv, pred_all, l0, l2, frac); fflush(lf); }
        /* 信赖域: 知识留出必须逐轮降(它是目标); 编程判决份只要求不比第 0 步(部署态)差 —— 用户的门是"编程不掉", 对照物是部署态。
         * v09 实撞: 轮 3 判决份恰好读到抖动带低点 0.1815, 之后轮 4/5 的 0.1819/0.1820(仍比部署态 0.1831 低 0.6%)都被当成退而回退, 知识那边两轮各 −0.4% 的真收益全扔了;
         * 判决份 16 段的轮间抖动 ±0.0005, 对上一轮接受值做棘轮就是拿噪声当门。 */
        if (kl > kl_prev || (nevg && kgD < -2.0 * kgSE)) {
            for (uint32_t i = 0; i < NL && ok; i++) {
                const uint32_t il = c->vq_l[i]; uint32_t n = 0;
                for (uint32_t q = 0; q < nmv; q++) n += moves[q].il == il;
                if (!n) continue;
                pt_vrec *tmp = xmalloc((size_t)n * sizeof(pt_vrec)); uint32_t k = 0;
                for (uint32_t q = 0; q < nmv; q++) if (moves[q].il == il) { tmp[k] = moves[q].r; tmp[k].code = (tmp[k].code >> 16) | (tmp[k].code << 16); k++; }
                ok = vs_patch(m, il, tmp, n, E, FF);
                vlog[il].n -= n;   /* 这一轮追加的记录正好在尾部 */
                free(tmp);
            }
            moved_all -= nmv; frac *= 0.5;
            fprintf(stderr, "ds4: [ptrain] V 步轮 %u 回退(%s%s): %u 组写回旧码, 份额减半到 %.4g\n", rd,   /* 写明哪把尺触发: v09 轮 4 是编程判决份 +0.0004 触发, 旧句只打知识留出两数, 读成"0.5409 > 0.5431"自相矛盾 */
                    kl > kl_prev ? "知识留出没降 " : "", (nevg && kgD < -2.0 * kgSE) ? "编程判决份显著比部署态差" : "", nmv, frac);
            if (lf) { fprintf(lf, "vround %u revert frac %.5g\n", rd, frac); fflush(lf); }
            if (!ok) break;
            if (c->budget_s && now_sec() - t0 >= (double)c->budget_s) { fprintf(stderr, "ds4: [ptrain] budget_s=%u 到了: V 步到此收\n", c->budget_s); break; }
            continue;
        }
        kl_prev = kl; if (nevg) kg_prev = kg;
        char dir[1200]; snprintf(dir, sizeof dir, "%s/ckpt_v%02u", c->out, rd); mkdir(dir, 0755);
        for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) ok = vlog_save(&vlog[il], dir, il, E, FF, E);   /* 有记录的层都存(vq_init 带进来的别层记录也在) */
        if (ok) {   /* 底座指纹(与 pt_save_amp 同一式): 补丁是在这一份 ② 挂着时选出来的, 引擎挂时核对(core_v41_amp.c v41_pt_base_ok) */
            unsigned nfp = 0;
            const uint64_t h = (g_ds4_v41_amp_dir && g_ds4_v41_amp_dir[0]) ? ds4_gr_dir_fnv(g_ds4_v41_amp_dir, DS4_N_LAYER, &nfp) : DS4_GR_FNV_SEED;
            char p[1300]; snprintf(p, sizeof p, "%s/base.fnv", dir);
            FILE *f = fopen(p, "w"); ok = f && fprintf(f, "%016llx %u\n", (unsigned long long)h, nfp) > 0; if (f) fclose(f);
        }
        if (ok) fprintf(stderr, "ds4: [ptrain] V 步补丁已落 %s(vqpatch_Lnn.bin + base.fnv); 挂法: --zchain ② --posttrain %s\n", dir, dir);
        if (c->budget_s && now_sec() - t0 >= (double)c->budget_s) { fprintf(stderr, "ds4: [ptrain] budget_s=%u 到了: V 步到此收\n", c->budget_s); break; }
    }
    ds4_gpu_bwd_moe_count(NULL, 0, 0);
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) { free(nbr[il]); free(vlog[il].v); }
    free(counts); free(counts_doc); free(sl); free(pool_doc); free(pool_short); free(delta); free(ncode); free(ccode); free(cand); free(moves);
    free(kg_li0); free(kg_li); free(kg_ni0); free(kg_ni);
    return ok;
}
#endif /* !DS4_NO_GPU */
