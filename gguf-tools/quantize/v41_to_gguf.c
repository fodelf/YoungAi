/* v41_to_gguf.c — V4.1 量化目录(v41_quantize 产出的 safetensors) → 引擎单 GGUF(2026-09-12, 战役 P0)。
 *
 * 【干什么】把 09-12 落盘的 1.5 bpw 文件(routed 专家 VQ dim8×nc4096 三件 + 骨架 FP4 1×32 + bf16 小张量)
 * 按引擎的 blk.N.* 命名与 GGUF 类型写成一个文件, 引擎才能 mmap 跑它。纯搬字节: 不重新量化,
 * 数值与 Python 判决过的文件逐位同 —— 这是后面"引擎 logits 对拍 Python 学生"判决成立的前提。
 *
 * 【映射】
 *   routed 专家 → blk.L.ffn_exps_vq.blob(type 42, DQVL ver 2: nexp 真值 384; 每矿阵 DQVQ 载荷 =
 *                 [头 16][码本 nc×dim f16][行增益 rows f16][12bit 索引位流(行字节对齐 = 连续流)][8 B 尾垫])
 *                 码本三矩阵共用一份, 但 DQVQ 每载荷自带 ⇒ 每专家写三份(+50 MB/层, 先换实现简单; 去重后置)
 *   FP4 1×32 权重(weight I8 + scale ue8m0) → type 43 fp4x32(16 B nibble + 1 B scale 交织块)
 *   bf16 小张量(norm/hc/gate/indexer wk/k_norm/proj) → f32;  f32 照抄(compressor wkv/wgate, sink, hc)
 *   engram wkv(FP8 E4M3 + 32×32 块 ue8m0) → f16(幅值可表示, 逐元素检查不溢出)
 *   engram 表(203 GB)不进文件: 元数据记真实分片路径 + 字节偏移, 引擎 mmap 按行读
 *   engram 哈希常量 ← v41_engram_consts.py 产物(EGRC), 作 I32/I64 张量嵌入
 *   vision / aligner / image_start|end|newline / gate.bias_vl 不要; mtp 三塔留 P5
 * 用法: v41_to_gguf <量化目录> <engram_consts.bin> <out.gguf> */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include "../../src/common/ds4_st41.h"
#include "../../src/common/ds4_gguf_write.h"
#include "../../src/common/ds4_quantfmt.h"
#include "../../src/common/ds4_fp8.h"
#include "../../src/common/ds4_float.h"
#include "v41_cfg.h"

enum { J_F32_BF16, J_F32_COPY, J_BF16_COPY, J_FP8BLK, J_FP4X32, J_VQBLOB, J_RAW };
typedef struct { int kind; char src[192]; int layer; const void *raw; int s2; } job_t;
static job_t *g_jobs; static int g_nj, g_cj;
static v41_st S; static v41_cfg C; static ds4gw W;
/* ★第二个源: 原始 HF 目录(2026-09-17, mtp-1.md 的 M6 真因)★
 * DSpark 三塔在**原始 HF 里是 FP8**(E4M3 + 32×32 块缩放), 而我们吃的量化目录里它们已经被量化器
 * 连同主干骨架一起压成了 FP4(I8 + 1×32) —— 于是"三塔 FP4 全透传"实际透传的是一份被压过的草稿器。
 * 实撞的代价: 草稿器对**它本来就该对齐的 FP 原模型**的首位一致率只有 0.5023(gguf-tools/bench/dspark_agree),
 * 最容易的那一档(FP 锚 top1−top2 ≥ 6)也只有 0.814, 而底座是 0.977 —— 信息就在它的输入里, 它却答不出来。
 * 三塔骨架总共才 ~260 MB, 还原成 FP8 只多 ~240 MB, 这笔精度不该省。
 * ★只有 mtp.* 走这个源★: 主干是有意量化的, 那是产品本身。 */
static v41_st S2; static int g_has_s2 = 0, g_plan_s2 = 0;
static uint64_t g_out_bytes;

static void die(const char *m) { fprintf(stderr, "★%s★\n", m); exit(1); }
static const v41_st_ent *need_in(const char *name, int s2) {
    const v41_st_ent *e = v41_st_find(s2 ? &S2 : &S, name);
    if (!e) { fprintf(stderr, "★缺张量 %s(源: %s)★\n", name, s2 ? "原始 HF" : "量化目录"); exit(1); }
    return e;
}
static const v41_st_ent *need(const char *name) { return need_in(name, g_plan_s2); }
static const uint8_t *sdata(const v41_st_ent *e, int s2) { return v41_st_data(s2 ? &S2 : &S, e); }
static int add_job(int kind, const char *gname, uint32_t type, uint32_t nd, const uint64_t *ne, uint64_t nbytes, const char *src, int layer, const void *raw) {
    if (g_nj == g_cj) { g_cj = g_cj ? g_cj * 2 : 4096; g_jobs = (job_t *)realloc(g_jobs, sizeof(job_t) * g_cj); }
    job_t *j = &g_jobs[g_nj++];
    memset(j, 0, sizeof *j); j->kind = kind; j->layer = layer; j->raw = raw;
    if (src) snprintf(j->src, sizeof j->src, "%s", src);
    j->s2 = g_plan_s2;
    return ds4gw_tensor(&W, gname, type, nd, ne, nbytes);
}

/* ---- 登记: 按源张量 dtype 决定目标类型 ---- */
static void plan_small(const char *gname, const char *src) {            /* bf16/f32 小张量 → f32 */
    const v41_st_ent *e = need(src);
    uint64_t ne[4]; for (int i = 0; i < e->nd; i++) ne[i] = (uint64_t)e->shape[e->nd - 1 - i];
    int kind = !strcmp(e->dtype, "BF16") ? J_F32_BF16 : !strcmp(e->dtype, "F32") ? J_F32_COPY : -1;
    if (kind < 0) { fprintf(stderr, "★%s dtype %s 不是 BF16/F32★\n", src, e->dtype); exit(1); }
    add_job(kind, gname, DS4_GGT_F32, (uint32_t)e->nd, ne, (uint64_t)v41_st_numel(e) * 4, src, -1, NULL);
}
/* ★2026-09-15 clear.md C1: BF16 原样存, 不再展开成 f32★
 * 这几个矩阵(路由 gate / compressor / indexer 投影)在 HF 原件里本来就是 BF16, 是我们的转换器
 * 把它们展成 f32 才让每 token 多读了 0.20 GB。存回 BF16 = **拿回官方精度**, 逐位无损, 字节减半。
 * 为什么不顺手压 FP4: 这几个矩阵决定"选哪 6 个专家"和"哪些位置进候选池", 位宽一动行为就跳 ——
 * 要压得单独过五指标门(clear.md C1 的 A/B 项), 不许搭这一刀的便车。 */
static void plan_bf16(const char *gname, const char *src) {
    const v41_st_ent *e = need(src);
    if (strcmp(e->dtype, "BF16")) { fprintf(stderr, "★%s dtype %s 不是 BF16★\n", src, e->dtype); exit(1); }
    uint64_t ne[4]; for (int i = 0; i < e->nd; i++) ne[i] = (uint64_t)e->shape[e->nd - 1 - i];
    add_job(J_BF16_COPY, gname, DS4_GGT_BF16, (uint32_t)e->nd, ne, (uint64_t)v41_st_numel(e) * 2, src, -1, NULL);
}
static void plan_fp4(const char *gname, const char *src) {             /* FP4 1×32 → fp4x32 */
    const v41_st_ent *e = need(src);
    char sn[192]; snprintf(sn, sizeof sn, "%s", src); char *dot = strrchr(sn, '.'); strcpy(dot, ".scale"); need(sn);
    if (e->nd != 2 || strcmp(e->dtype, "I8")) { fprintf(stderr, "★%s 不是 FP4 打包 I8 二维★\n", src); exit(1); }
    uint64_t rows = (uint64_t)e->shape[0], cols = (uint64_t)e->shape[1] * 2;
    if (cols % 32) { fprintf(stderr, "★%s 列 %llu 非 32 倍★\n", src, (unsigned long long)cols); exit(1); }
    uint64_t ne[2] = {cols, rows};
    add_job(J_FP4X32, gname, DS4_GGT_FP4X32, 2, ne, rows * cols / 32 * 17, src, -1, NULL);
}
/* ★2026-09-15 clear.md C1: engram wkv 原样存 FP8, 不再展开成 f16★
 * 这张表 6144×25600, 两层共 0.315 GB。原来转换器把 e4m3×块缩放算成 f16 落盘 —— 字节翻倍(0.63 GB),
 * **而且是有损的**: f16 只有 10 位尾数, e4m3 的 3 位尾数乘上 2 的幂之后不一定落在 f16 格点上,
 * 大值还可能溢出(老代码专门数了 over/nan 就是防这个)。原样存反而更准, 且每 token 少读 0.31 GB。
 * 布局: 先 rows×cols 个 e4m3 字节, 紧跟 ceil(rows/32)×ceil(cols/32) 个 ue8m0 缩放字节。 */
static void plan_fp8blk(const char *gname, const char *src) {
    const v41_st_ent *e = need(src);
    if (e->nd != 2 || strcmp(e->dtype, "F8_E4M3")) { fprintf(stderr, "★%s 不是 F8_E4M3 二维★\n", src); exit(1); }
    char sn[192]; snprintf(sn, sizeof sn, "%s", src); strcpy(strrchr(sn, '.'), ".scale");
    const v41_st_ent *se = need(sn);
    uint64_t rows = (uint64_t)e->shape[0], cols = (uint64_t)e->shape[1];
    if ((uint64_t)se->shape[0] != (rows + 31) / 32 || (uint64_t)se->shape[1] != (cols + 31) / 32)
        { fprintf(stderr, "★%s 32×32 scale 形状不配★\n", src); exit(1); }
    uint64_t ne[2] = {cols, rows};
    add_job(J_FP8BLK, gname, DS4_GGT_FP8_32X32, 2, ne, rows * cols + ((rows + 31) / 32) * ((cols + 31) / 32), src, -1, NULL);
}
static uint64_t vq_payload_bytes(int L, int e, const char *m, uint64_t *rows, uint64_t *cols) {
    char n[192];
    snprintf(n, sizeof n, "layers.%d.ffn.experts.%d.vq.cb", L, e); const v41_st_ent *cb = need(n);
    snprintf(n, sizeof n, "layers.%d.ffn.experts.%d.%s.vq.idx", L, e, m); const v41_st_ent *ix = need(n);
    snprintf(n, sizeof n, "layers.%d.ffn.experts.%d.%s.vq.gain", L, e, m); const v41_st_ent *g = need(n);
    uint64_t nc = (uint64_t)cb->shape[0], dim = (uint64_t)cb->shape[1];
    int nbit = 0; while ((1ull << nbit) < nc) nbit++;
    *rows = (uint64_t)ix->shape[0];
    *cols = (uint64_t)ix->shape[1] * 8 / (uint64_t)nbit * dim;
    if ((uint64_t)g->shape[0] != *rows) die("vq.gain 行数与 idx 不符");
    return 16 + nc * dim * 2 + *rows * 2 + (uint64_t)ix->shape[0] * (uint64_t)ix->shape[1] + 8;
}
static void plan_vqblob(int L) {
    const int nexp = C.n_routed_experts;
    uint64_t total = 16 + (uint64_t)nexp * 3 * 8;
    static const char *mats[3] = {"w1", "w3", "w2"};        /* which: 0=w1 1=w3 2=w2 (vq_fmt.h) */
    for (int e = 0; e < nexp; e++) for (int w = 0; w < 3; w++) { uint64_t r, c; total += vq_payload_bytes(L, e, mats[w], &r, &c); }
    char gn[96]; snprintf(gn, sizeof gn, "blk.%d.ffn_exps_vq.blob", L);
    uint64_t ne[1] = {total};
    add_job(J_VQBLOB, gn, DS4_GGT_VQBLOB, 1, ne, total, NULL, L, NULL);
}
static int in_list(const int *v, int n, int x) { for (int i = 0; i < n; i++) if (v[i] == x) return 1; return 0; }

static void plan_layer(int L) {
    char g[96], s[192];
#define SMALL(gf, sf) do { snprintf(g, sizeof g, gf, L); snprintf(s, sizeof s, sf, L); plan_small(g, s); } while (0)
#define BF16(gf, sf)  do { snprintf(g, sizeof g, gf, L); snprintf(s, sizeof s, sf, L); plan_bf16(g, s); } while (0)
#define FP4(gf, sf)   do { snprintf(g, sizeof g, gf, L); snprintf(s, sizeof s, sf, L); plan_fp4(g, s); } while (0)
    SMALL("blk.%d.hc_attn_fn.weight", "layers.%d.hc_attn_fn");
    SMALL("blk.%d.hc_attn_scale.weight", "layers.%d.hc_attn_scale");
    SMALL("blk.%d.hc_attn_base.weight", "layers.%d.hc_attn_base");
    SMALL("blk.%d.attn_norm.weight", "layers.%d.attn_norm.weight");
    FP4("blk.%d.attn_q_a.weight", "layers.%d.attn.wq_a.weight");
    SMALL("blk.%d.attn_q_a_norm.weight", "layers.%d.attn.q_norm.weight");
    FP4("blk.%d.attn_q_b.weight", "layers.%d.attn.wq_b.weight");
    FP4("blk.%d.attn_kv.weight", "layers.%d.attn.wkv.weight");
    SMALL("blk.%d.attn_kv_a_norm.weight", "layers.%d.attn.kv_norm.weight");
    SMALL("blk.%d.attn_sinks.weight", "layers.%d.attn.attn_sink");
    FP4("blk.%d.attn_output_a.weight", "layers.%d.attn.wo_a.weight");
    FP4("blk.%d.attn_output_b.weight", "layers.%d.attn.wo_b.weight");
    const int ratio = C.compress_ratios[L];
    if (in_list(C.kv_source_layers, C.n_kv_source, L)) {
        BF16("blk.%d.attn_compressor_kv.weight", "layers.%d.attn.compressor.wkv.weight");
        if (ratio > 1) BF16("blk.%d.attn_compressor_gate.weight", "layers.%d.attn.compressor.wgate.weight");
        SMALL("blk.%d.attn_compressor_norm.weight", "layers.%d.attn.compressor.norm.weight");
        BF16("blk.%d.indexer.wk.weight", "layers.%d.attn.indexer.wk.weight");
        SMALL("blk.%d.indexer.k_norm.weight", "layers.%d.attn.indexer.k_norm.weight");
    }
    if (in_list(C.index_source_layers, C.n_index_source, L)) {
        FP4("blk.%d.indexer.attn_q_b.weight", "layers.%d.attn.indexer.wq_b.weight");
        BF16("blk.%d.indexer.proj.weight", "layers.%d.attn.indexer.weights_proj.weight");
    }
    SMALL("blk.%d.hc_ffn_fn.weight", "layers.%d.hc_ffn_fn");
    SMALL("blk.%d.hc_ffn_scale.weight", "layers.%d.hc_ffn_scale");
    SMALL("blk.%d.hc_ffn_base.weight", "layers.%d.hc_ffn_base");
    SMALL("blk.%d.ffn_norm.weight", "layers.%d.ffn_norm.weight");
    BF16("blk.%d.ffn_gate_inp.weight", "layers.%d.ffn.gate.weight");
    SMALL("blk.%d.exp_probs_b.bias", "layers.%d.ffn.gate.bias");
    plan_vqblob(L);
    FP4("blk.%d.ffn_gate_shexp.weight", "layers.%d.ffn.shared_experts.w1.weight");
    FP4("blk.%d.ffn_up_shexp.weight", "layers.%d.ffn.shared_experts.w3.weight");
    FP4("blk.%d.ffn_down_shexp.weight", "layers.%d.ffn.shared_experts.w2.weight");
    if (in_list(C.engram_layer_ids, C.n_engram, L)) {
        snprintf(g, sizeof g, "blk.%d.engram_wkv.weight", L); snprintf(s, sizeof s, "layers.%d.engram.wkv.weight", L); plan_fp8blk(g, s);
        SMALL("blk.%d.engram_q.weight", "layers.%d.engram.q_weight");
        SMALL("blk.%d.engram_k.weight", "layers.%d.engram.k_weight");
    }
#undef SMALL
#undef BF16
#undef FP4
}

/* ---- DSpark 三塔(speed.md 段 6 S1, 2026-09-15) ----
 * 官方 model.py: `mtp.0/1/2` 是三个 SWA(128) 草稿块, 各 128 个专家 top-3。结构与普通层一模一样
 * (hc 六件 + attn 五投影 + ffn gate/专家/shared), 外加五个只此一份的头:
 *   mtp.0.main_proj / main_norm  —— 主模型 L37/38/39 的注意力输入(hc 四路均值)拼成 15360 维再投回 5120,
 *                                   得到 main_x, 它是块注意力的主 KV 来源。取错位置不报错, 只是接受率掉到 1 附近。
 *   mtp.2.markov_head.embed/head —— 秩 256 的 markov 头, 按块内前一个草稿 token 给 logit 加偏置。
 *   mtp.2.confidence_head.proj   —— 出每一位的条件接受概率, 调度器据此决定每轮验证几个。
 *   mtp.2.norm                   —— 三塔共用的出口 norm(官方 self.mtp[-1] 借主模型的 embed/head, GGUF 不再存一份)。
 * ★专家走 FP4 直透, 不做 VQ★(speed.md §2: "草稿质量 = 接受率 = 速度, 不省这里")。
 * ★不要的★: gate.bias_vl(视觉路), 与文件头第 16 行同一条规矩。 */
/* 塔数与每塔专家数一律从 config.json 读(C.n_mtp_layers / C.dspark_n_routed_experts) —— 写死数字
 * 就是把模型架构焊进工具, 换一版模型静默出错(铁律 2026-09-15: 禁魔数硬编码)。 */
/* ★三塔一律从**原始 HF** 取, 按原生精度存(2026-09-17)★ —— 见 S2 那段注释的所以然。
 * MFP8 = 原件的 F8_E4M3 + 32×32 缩放, 原样搬(attn 五投影 / shared 专家三件 / main_proj);
 * MBF16 = 原件的 BF16, 原样搬(路由 gate / markov 两件 / confidence) —— 原来 plan_small 把它展成 f32,
 *   字节翻倍而值没变, 草稿器每轮为此白读 130 MB。★norm 不在此列★: 引擎的 rms_norm 核把权重当 f32 读
 *   (cuda_v41_1.inc.cu), 存成 BF16 它照样按 f32 解释 —— 不报错, 直接出垃圾(实撞: 首位一致率 0.035)。
 *   norm 一律走 MSMALL(BF16→f32, 值无损), 只有过"按类型分发"那条路的矩阵才许存 BF16;
 * MFP4 = 原件出厂就是 FP4 的那一类(128 个路由专家), 照旧;
 * MSMALL = 原件就是 F32 的(sink / gate.bias / hc_*)。
 * ★哪个张量是哪一类不许猜★: plan_* 里都按源 dtype 校验, 对不上直接停车。 */
static void plan_mtp(int T) {
    char g[96], s[192];
    if (!g_has_s2) die("三塔要从原始 HF 取原生精度: 第 5 个参数给 HF 目录(见 --help)");
    g_plan_s2 = 1;
#define MSMALL(gf, sf) do { snprintf(g, sizeof g, gf, T); snprintf(s, sizeof s, sf, T); plan_small(g, s); } while (0)
#define MBF16(gf, sf)  do { snprintf(g, sizeof g, gf, T); snprintf(s, sizeof s, sf, T); plan_bf16(g, s); } while (0)
#define MFP8(gf, sf)   do { snprintf(g, sizeof g, gf, T); snprintf(s, sizeof s, sf, T); plan_fp8blk(g, s); } while (0)
#define MFP4(gf, sf)   do { snprintf(g, sizeof g, gf, T); snprintf(s, sizeof s, sf, T); plan_fp4(g, s); } while (0)
    MSMALL("mtp.%d.hc_attn_fn.weight", "mtp.%d.hc_attn_fn");
    MSMALL("mtp.%d.hc_attn_scale.weight", "mtp.%d.hc_attn_scale");
    MSMALL("mtp.%d.hc_attn_base.weight", "mtp.%d.hc_attn_base");
    MSMALL("mtp.%d.attn_norm.weight", "mtp.%d.attn_norm.weight");
    MFP8("mtp.%d.attn_q_a.weight", "mtp.%d.attn.wq_a.weight");
    MSMALL("mtp.%d.attn_q_a_norm.weight", "mtp.%d.attn.q_norm.weight");
    MFP8("mtp.%d.attn_q_b.weight", "mtp.%d.attn.wq_b.weight");
    MFP8("mtp.%d.attn_kv.weight", "mtp.%d.attn.wkv.weight");
    MSMALL("mtp.%d.attn_kv_a_norm.weight", "mtp.%d.attn.kv_norm.weight");
    MSMALL("mtp.%d.attn_sinks.weight", "mtp.%d.attn.attn_sink");
    MFP8("mtp.%d.attn_output_a.weight", "mtp.%d.attn.wo_a.weight");
    MFP8("mtp.%d.attn_output_b.weight", "mtp.%d.attn.wo_b.weight");
    MSMALL("mtp.%d.hc_ffn_fn.weight", "mtp.%d.hc_ffn_fn");
    MSMALL("mtp.%d.hc_ffn_scale.weight", "mtp.%d.hc_ffn_scale");
    MSMALL("mtp.%d.hc_ffn_base.weight", "mtp.%d.hc_ffn_base");
    MSMALL("mtp.%d.ffn_norm.weight", "mtp.%d.ffn_norm.weight");
    MBF16("mtp.%d.ffn_gate_inp.weight", "mtp.%d.ffn.gate.weight");
    MSMALL("mtp.%d.exp_probs_b.bias", "mtp.%d.ffn.gate.bias");
    MFP8("mtp.%d.ffn_gate_shexp.weight", "mtp.%d.ffn.shared_experts.w1.weight");
    MFP8("mtp.%d.ffn_up_shexp.weight", "mtp.%d.ffn.shared_experts.w3.weight");
    MFP8("mtp.%d.ffn_down_shexp.weight", "mtp.%d.ffn.shared_experts.w2.weight");
    for (int e = 0; e < C.dspark_n_routed_experts; e++) {
        snprintf(g, sizeof g, "mtp.%d.ffn_exp.%d.gate.weight", T, e); snprintf(s, sizeof s, "mtp.%d.ffn.experts.%d.w1.weight", T, e); plan_fp4(g, s);
        snprintf(g, sizeof g, "mtp.%d.ffn_exp.%d.up.weight",   T, e); snprintf(s, sizeof s, "mtp.%d.ffn.experts.%d.w3.weight", T, e); plan_fp4(g, s);
        snprintf(g, sizeof g, "mtp.%d.ffn_exp.%d.down.weight", T, e); snprintf(s, sizeof s, "mtp.%d.ffn.experts.%d.w2.weight", T, e); plan_fp4(g, s);
    }
    if (T == 0) {
        plan_fp8blk("mtp.main_proj.weight", "mtp.0.main_proj.weight");   /* 原件 F8_E4M3 32×32 */
        plan_small("mtp.main_norm.weight", "mtp.0.main_norm.weight");
    }
    if (T == 2) {
        plan_bf16("mtp.markov_embd.weight", "mtp.2.markov_head.embed.weight");
        plan_bf16("mtp.markov_head.weight", "mtp.2.markov_head.head.weight");
        plan_bf16("mtp.confidence.weight", "mtp.2.confidence_head.proj.weight");
        plan_small("mtp.out_norm.weight", "mtp.2.norm.weight");
    }
    g_plan_s2 = 0;
#undef MBF16
#undef MFP8
#undef MSMALL
#undef MFP4
}

/* ---- 生成字节 ---- */
static uint8_t *g_buf; static uint64_t g_cap;
static uint8_t *buf(uint64_t n) { if (n > g_cap) { g_cap = n + (n >> 3); g_buf = (uint8_t *)realloc(g_buf, g_cap); if (!g_buf) die("内存不够"); } return g_buf; }

static uint64_t gen_small(const job_t *j) {
    const v41_st_ent *e = need_in(j->src, j->s2); const uint8_t *p = sdata(e, j->s2); if (!p) exit(1);
    uint64_t n = (uint64_t)v41_st_numel(e); float *o = (float *)buf(n * 4);
    if (j->kind == J_BF16_COPY) { memcpy(o, p, n * 2); return n * 2; }   /* 原样搬 2 字节, 一位都不动 */
    if (j->kind == J_F32_COPY) memcpy(o, p, n * 4);
    else for (uint64_t i = 0; i < n; i++) { uint16_t h; memcpy(&h, p + 2 * i, 2); o[i] = ds4_bf16_to_f32(h); }
    return n * 4;
}
static uint64_t gen_fp4(const job_t *j) {
    const v41_st_ent *e = need_in(j->src, j->s2); const uint8_t *w = sdata(e, j->s2);
    char sn[192]; snprintf(sn, sizeof sn, "%s", j->src); strcpy(strrchr(sn, '.'), ".scale");
    const v41_st_ent *se = need_in(sn, j->s2); const uint8_t *sc = sdata(se, j->s2);
    if (!w || !sc) exit(1);
    uint64_t rows = (uint64_t)e->shape[0], cb = (uint64_t)e->shape[1] / 16;   /* 每行块数 = cols/32 */
    if ((uint64_t)se->shape[0] != rows || (uint64_t)se->shape[1] != cb) { fprintf(stderr, "★%s scale 形状不配★\n", j->src); exit(1); }
    uint8_t *o = buf(rows * cb * 17);
    for (uint64_t r = 0; r < rows; r++)
        for (uint64_t b = 0; b < cb; b++) {
            uint8_t *d = o + (r * cb + b) * 17;
            memcpy(d, w + r * (cb * 16) + b * 16, 16);
            d[16] = sc[r * cb + b];
        }
    return rows * cb * 17;
}
static uint64_t gen_fp8blk(const job_t *j) {
    const v41_st_ent *e = need_in(j->src, j->s2); const uint8_t *w = sdata(e, j->s2);
    char sn[192]; snprintf(sn, sizeof sn, "%s", j->src); strcpy(strrchr(sn, '.'), ".scale");
    const v41_st_ent *se = need_in(sn, j->s2); const uint8_t *sc = sdata(se, j->s2);
    if (!w || !sc) exit(1);
    uint64_t rows = (uint64_t)e->shape[0], cols = (uint64_t)e->shape[1];
    uint64_t sr = (rows + 31) / 32, scn = (cols + 31) / 32, n = rows * cols + sr * scn;
    uint8_t *o = buf(n);
    memcpy(o, w, rows * cols);                 /* e4m3 平面, 一位不动 */
    memcpy(o + rows * cols, sc, sr * scn);     /* ue8m0 缩放平面 */
    return n;
}
static uint64_t gen_vqblob(const job_t *j) {
    const int L = j->layer, nexp = C.n_routed_experts;
    static const char *mats[3] = {"w1", "w3", "w2"};
    uint64_t total = 16 + (uint64_t)nexp * 3 * 8;
    for (int e = 0; e < nexp; e++) for (int w = 0; w < 3; w++) { uint64_t r, c; total += vq_payload_bytes(L, e, mats[w], &r, &c); }
    uint8_t *o = buf(total); memset(o, 0, 16 + (uint64_t)nexp * 3 * 8);
    uint32_t hdr[4] = {0x4C565144u, 2u, (uint32_t)L, (uint32_t)nexp}; memcpy(o, hdr, 16);
    uint64_t *tab = (uint64_t *)(o + 16); uint64_t off = 16 + (uint64_t)nexp * 3 * 8;
    char n[192];
    for (int e = 0; e < nexp; e++) {
        snprintf(n, sizeof n, "layers.%d.ffn.experts.%d.vq.cb", L, e); const v41_st_ent *cb = need(n); const uint8_t *cbp = v41_st_data(&S, cb);
        for (int w = 0; w < 3; w++) {
            snprintf(n, sizeof n, "layers.%d.ffn.experts.%d.%s.vq.idx", L, e, mats[w]); const v41_st_ent *ix = need(n); const uint8_t *ixp = v41_st_data(&S, ix);
            snprintf(n, sizeof n, "layers.%d.ffn.experts.%d.%s.vq.gain", L, e, mats[w]); const v41_st_ent *g = need(n); const uint8_t *gp = v41_st_data(&S, g);
            if (!cbp || !ixp || !gp) exit(1);
            uint64_t rows, cols; vq_payload_bytes(L, e, mats[w], &rows, &cols);
            uint16_t dim = (uint16_t)cb->shape[1], nc = (uint16_t)cb->shape[0];
            uint8_t *p = o + off; tab[e * 3 + w] = off;
            memcpy(p, &(uint32_t){0x51565144u}, 4); memcpy(p + 4, &dim, 2); memcpy(p + 6, &nc, 2);
            uint32_t r32 = (uint32_t)rows, c32 = (uint32_t)cols; memcpy(p + 8, &r32, 4); memcpy(p + 12, &c32, 4);
            p += 16; memcpy(p, cbp, (size_t)nc * dim * 2); p += (size_t)nc * dim * 2;
            memcpy(p, gp, rows * 2); p += rows * 2;
            uint64_t ixb = (uint64_t)ix->shape[0] * (uint64_t)ix->shape[1]; memcpy(p, ixp, ixb); p += ixb;
            memset(p, 0, 8); p += 8;
            off = (uint64_t)(p - o);
        }
    }
    if (off != total) die("vq blob 字节账不平");
    return total;
}

/* tokenizer 常量(v41_tokenizer_consts.py 产物 TOKC) → tokenizer.ggml.tokens / merges 两个字串数组 +
 * model/pre/bos/eos。引擎 vocab_load 只吃这几个键; pre="joyai-llm" 是 core_bpe.c 写死的切分规则名。 */
static void emit_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) die("打不开 tokenizer_consts.bin");
    uint32_t h[4]; if (fread(h, 4, 4, f) != 4 || h[0] != 0x434B4F54u || h[1] != 1) die("tokenizer_consts.bin 魔数/版本不对(TOKC v1)");
    uint32_t nt = h[2], nm = h[3];
    if ((int)nt != C.vocab_size) { fprintf(stderr, "★tokenizer token 数 %u != vocab_size %d★\n", nt, C.vocab_size); exit(1); }
    char **tok = (char **)malloc(sizeof(char *) * nt), **mg = (char **)malloc(sizeof(char *) * nm);
    for (uint32_t i = 0; i < nt + nm; i++) {
        uint32_t L; if (fread(&L, 4, 1, f) != 1 || L > 4096) die("tokenizer_consts.bin 读坏");
        char *s = (char *)malloc(L + 1); if (fread(s, 1, L, f) != L) die("tokenizer_consts.bin 读不满"); s[L] = 0;
        if (i < nt) tok[i] = s; else mg[i - nt] = s;
    }
    fclose(f);
    ds4gw_kv_str(&W, "tokenizer.ggml.model", "gpt2");
    ds4gw_kv_str(&W, "tokenizer.ggml.pre", "joyai-llm");
    ds4gw_kv_arr_str(&W, "tokenizer.ggml.tokens", (const char *const *)tok, nt);
    ds4gw_kv_arr_str(&W, "tokenizer.ggml.merges", (const char *const *)mg, nm);
    ds4gw_kv_u32(&W, "tokenizer.ggml.bos_token_id", 0);
    ds4gw_kv_u32(&W, "tokenizer.ggml.eos_token_id", 1);
    fprintf(stderr, "[tokenizer] %u token / %u merges 进 KV\n", nt, nm);
    for (uint32_t i = 0; i < nt; i++) free(tok[i]); for (uint32_t i = 0; i < nm; i++) free(mg[i]); free(tok); free(mg);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "用法: v41_to_gguf <量化目录> <engram_consts.bin> <tokenizer_consts.bin> <out.gguf> [原始HF目录]\n"
                        "  第 5 个参数 = 原始 HF 目录, **带 DSpark 三塔时必给**: 三塔在原件里是 FP8/BF16,\n"
                        "  而量化目录里已经被量化器压成 FP4 —— 那是草稿器的精度, 不该跟主干一起压(见 plan_mtp 注释)。\n");
        return 2;
    }
    const char *qdir = argv[1], *egp = argv[2], *tkp = argv[3], *out = argv[4];
    const char *hfdir = argc > 5 ? argv[5] : NULL;
    if (v41_st_open(&S, qdir)) return 1;
    if (hfdir) {
        if (v41_st_open(&S2, hfdir)) { fprintf(stderr, "★打不开原始 HF 目录 %s★\n", hfdir); return 1; }
        g_has_s2 = 1;
        fprintf(stderr, "[源] 主干 ← %s; ★三塔 ← %s(原生精度)★\n", qdir, hfdir);
    }
    char cfgp[4200]; snprintf(cfgp, sizeof cfgp, "%s/inference/config.json", qdir); v41_cfg_load(&C, cfgp);
    fprintf(stderr, "[配置] %d 层 dim %d 专家 %d top-%d 路由 %s ×%.2f; 压缩比 %d 项; kv源 %d 索引源 %d engram %d 层\n",
            C.n_layers, C.dim, C.n_routed_experts, C.n_activated_experts, C.score_func, C.route_scale, C.n_compress_ratios, C.n_kv_source, C.n_index_source, C.n_engram);
    /* engram 常量 */
    FILE *ef = fopen(egp, "rb"); if (!ef) die("打不开 engram_consts.bin");
    uint32_t eh[8]; if (fread(eh, 4, 8, ef) != 8 || eh[0] != 0x43524745u) die("engram_consts.bin 魔数不对(EGRC)");
    uint32_t ev = eh[1], evocab = eh[2], ecv = eh[3], enl = eh[4], eng = eh[5], enh = eh[6], epad = eh[7];
    if (ev != 1 || (int)evocab != C.vocab_size || (int)ecv != C.engram_compressed_vocab_size || (int)enl != C.n_engram || (int)eng != C.engram_max_ngram_size || (int)enh != C.engram_n_heads) die("engram_consts.bin 与 config 不配");
    uint64_t n_tm = evocab, n_mul = (uint64_t)enl * eng, n_pr = (uint64_t)enl * (eng - 1) * enh, n_off = (uint64_t)enl * (eng - 1) * enh;
    int32_t *tm = (int32_t *)malloc(n_tm * 4); int64_t *mul = (int64_t *)malloc(n_mul * 8), *pr = (int64_t *)malloc(n_pr * 8), *offs = (int64_t *)malloc(n_off * 8);
    if (fread(tm, 4, n_tm, ef) != n_tm || fread(mul, 8, n_mul, ef) != n_mul || fread(pr, 8, n_pr, ef) != n_pr || fread(offs, 8, n_off, ef) != n_off) die("engram_consts.bin 读不满");
    fclose(ef);

    if (ds4gw_begin(&W, out)) return 1;
    /* ---- 元数据: 引擎 core_validate 要的全套 deepseek4.* + V4.1 新键 ---- */
    ds4gw_kv_str(&W, "general.architecture", "deepseek4");
    ds4gw_kv_str(&W, "general.name", "DeepSeek V4.1 Flash");
    ds4gw_kv_u32(&W, "general.alignment", 32);
    ds4gw_kv_str(&W, "deepseek4.variant", "v41_flash");
    ds4gw_kv_str(&W, "deepseek4.recipe", "routed=vq8x4096 skeleton=fp4x32 engram=sidecar nocal (v41_quantize 2026-09-12)");
    emit_tokenizer(tkp);
    ds4gw_kv_u32(&W, "deepseek4.block_count", (uint32_t)C.n_layers);
    ds4gw_kv_u64(&W, "deepseek4.context_length", 1048576);
    ds4gw_kv_u32(&W, "deepseek4.embedding_length", (uint32_t)C.dim);
    ds4gw_kv_u32(&W, "deepseek4.vocab_size", (uint32_t)C.vocab_size);
    ds4gw_kv_u32(&W, "deepseek4.attention.head_count", (uint32_t)C.n_heads);
    ds4gw_kv_u32(&W, "deepseek4.attention.head_count_kv", 1);
    ds4gw_kv_u32(&W, "deepseek4.attention.key_length", (uint32_t)C.head_dim);
    ds4gw_kv_u32(&W, "deepseek4.attention.value_length", (uint32_t)C.head_dim);
    ds4gw_kv_u32(&W, "deepseek4.rope.dimension_count", (uint32_t)C.rope_head_dim);
    ds4gw_kv_u32(&W, "deepseek4.attention.q_lora_rank", (uint32_t)C.q_lora_rank);
    ds4gw_kv_u32(&W, "deepseek4.attention.output_lora_rank", (uint32_t)C.o_lora_rank);
    ds4gw_kv_u32(&W, "deepseek4.attention.output_group_count", (uint32_t)C.o_groups);
    ds4gw_kv_u32(&W, "deepseek4.expert_count", (uint32_t)C.n_routed_experts);
    /* 引擎按元数据认三塔, 不写死数字(铁律: 引擎禁版本名/魔数硬编码) */
    ds4gw_kv_u32(&W, "deepseek4.mtp.tower_count", (uint32_t)C.n_mtp_layers);
    ds4gw_kv_u32(&W, "deepseek4.mtp.expert_count", (uint32_t)C.dspark_n_routed_experts);
    /* 草稿器跑起来还要这五件(speed.md 段 6 D1): 每塔选几个专家 / 一块出几位 / 占位 token 的 id /
     * markov 头的秩 / 主模型哪几层的注意力输入拼成 main_x。少一件引擎就只能猜, 所以全部落元数据。 */
    ds4gw_kv_u32(&W, "deepseek4.mtp.expert_used_count", (uint32_t)C.dspark_n_activated_experts);
    ds4gw_kv_u32(&W, "deepseek4.mtp.block_size", (uint32_t)C.dspark_block_size);
    ds4gw_kv_u32(&W, "deepseek4.mtp.noise_token_id", (uint32_t)C.dspark_noise_token_id);
    ds4gw_kv_u32(&W, "deepseek4.mtp.markov_rank", (uint32_t)C.dspark_markov_rank);
    ds4gw_kv_arr_i32(&W, "deepseek4.mtp.target_layers", C.dspark_target_layer_ids, (uint64_t)C.n_dspark_target);
    ds4gw_kv_u32(&W, "deepseek4.expert_used_count", (uint32_t)C.n_activated_experts);
    ds4gw_kv_u32(&W, "deepseek4.expert_feed_forward_length", (uint32_t)C.moe_inter_dim);
    ds4gw_kv_u32(&W, "deepseek4.expert_shared_count", (uint32_t)C.n_shared_experts);
    ds4gw_kv_u32(&W, "deepseek4.hash_layer_count", 0);
    ds4gw_kv_u32(&W, "deepseek4.attention.sliding_window", (uint32_t)C.window_size);
    ds4gw_kv_u32(&W, "deepseek4.attention.indexer.head_count", (uint32_t)C.index_n_heads);
    ds4gw_kv_u32(&W, "deepseek4.attention.indexer.key_length", (uint32_t)C.index_head_dim);
    ds4gw_kv_u32(&W, "deepseek4.attention.indexer.top_k", (uint32_t)C.index_topk);
    ds4gw_kv_u32(&W, "deepseek4.hyper_connection.count", (uint32_t)C.hc_mult);
    ds4gw_kv_u32(&W, "deepseek4.hyper_connection.sinkhorn_iterations", (uint32_t)C.hc_sinkhorn_iters);
    ds4gw_kv_f32(&W, "deepseek4.hyper_connection.epsilon", C.hc_eps);
    ds4gw_kv_f32(&W, "deepseek4.attention.layer_norm_rms_epsilon", C.norm_eps);
    ds4gw_kv_f32(&W, "deepseek4.expert_weights_scale", C.route_scale);
    ds4gw_kv_bool(&W, "deepseek4.expert_weights_norm", 1);
    ds4gw_kv_str(&W, "deepseek4.expert_gating_func", C.score_func);
    ds4gw_kv_f32(&W, "deepseek4.rope.freq_base", C.rope_theta);
    ds4gw_kv_f32(&W, "deepseek4.rope.scaling.factor", C.rope_factor);
    ds4gw_kv_f32(&W, "deepseek4.rope.scaling.yarn_beta_fast", (float)C.beta_fast);
    ds4gw_kv_f32(&W, "deepseek4.rope.scaling.yarn_beta_slow", (float)C.beta_slow);
    ds4gw_kv_u64(&W, "deepseek4.rope.scaling.original_context_length", (uint64_t)C.original_seq_len);
    ds4gw_kv_f32(&W, "deepseek4.attention.compress_rope_freq_base", C.compress_rope_theta);
    { int32_t cr[V41_MAX_LAYERS]; for (int i = 0; i < C.n_compress_ratios; i++) cr[i] = C.compress_ratios[i];
      ds4gw_kv_arr_i32(&W, "deepseek4.attention.compress_ratios", cr, (uint64_t)C.n_compress_ratios); }
    { float sw[V41_MAX_LAYERS]; for (int i = 0; i < C.n_layers; i++) sw[i] = C.swiglu_limit;
      ds4gw_kv_arr_f32(&W, "deepseek4.swiglu_clamp_exp", sw, (uint64_t)C.n_layers); }
    { int32_t v[V41_MAX_LIST]; for (int i = 0; i < C.n_kv_source; i++) v[i] = C.kv_source_layers[i];
      ds4gw_kv_arr_i32(&W, "deepseek4.attention.kv_source_layers", v, (uint64_t)C.n_kv_source);
      for (int i = 0; i < C.n_index_source; i++) v[i] = C.index_source_layers[i];
      ds4gw_kv_arr_i32(&W, "deepseek4.attention.index_source_layers", v, (uint64_t)C.n_index_source); }
    ds4gw_kv_u32(&W, "deepseek4.attention.candidate.source_layer", (uint32_t)C.candidate_source_layer);
    ds4gw_kv_u32(&W, "deepseek4.attention.candidate.topk_blocks", (uint32_t)C.candidate_topk_blocks);
    ds4gw_kv_u32(&W, "deepseek4.attention.candidate.block_size", (uint32_t)C.candidate_block_size);
    { int32_t v[V41_MAX_LIST]; for (int i = 0; i < C.n_engram; i++) v[i] = C.engram_layer_ids[i];
      ds4gw_kv_arr_i32(&W, "deepseek4.engram.layer_ids", v, (uint64_t)C.n_engram);
      uint64_t rows[V41_MAX_LIST]; for (int i = 0; i < C.n_engram; i++) rows[i] = (uint64_t)C.engram_num_embeddings[i];
      ds4gw_kv_arr_u64(&W, "deepseek4.engram.num_embeddings", rows, (uint64_t)C.n_engram); }
    ds4gw_kv_u32(&W, "deepseek4.engram.max_ngram_size", (uint32_t)C.engram_max_ngram_size);
    ds4gw_kv_u32(&W, "deepseek4.engram.head_count", (uint32_t)C.engram_n_heads);
    ds4gw_kv_u32(&W, "deepseek4.engram.head_dim", (uint32_t)C.engram_head_dim);
    ds4gw_kv_u32(&W, "deepseek4.engram.vocab_size", (uint32_t)C.engram_vocab_size);
    ds4gw_kv_u32(&W, "deepseek4.engram.compressed_vocab_size", (uint32_t)C.engram_compressed_vocab_size);
    ds4gw_kv_u32(&W, "deepseek4.engram.pad_id_compressed", epad);
    /* engram 表: 真实分片路径 + 绝对字节偏移(引擎 mmap 该文件, 行 i 的 256 B 权重在 weight_offset+i*256) */
    for (int i = 0; i < C.n_engram; i++) {
        char n[128], k[128]; int L = C.engram_layer_ids[i];
        snprintf(n, sizeof n, "layers.%d.engram.embed.weight", L); const v41_st_ent *ew = need(n);
        snprintf(n, sizeof n, "layers.%d.engram.embed.scale", L); const v41_st_ent *es = need(n);
        if (ew->shard != es->shard) die("engram 表 weight/scale 不在同一分片");
        char rp[PATH_MAX]; if (!realpath(S.sh[ew->shard].path, rp)) die("engram 分片 realpath 失败");
        if (ew->shape[0] != C.engram_num_embeddings[i] || ew->shape[1] != C.engram_head_dim) die("engram 表形状与 config 不配");
        snprintf(k, sizeof k, "deepseek4.engram.%d.table_path", i); ds4gw_kv_str(&W, k, rp);
        snprintf(k, sizeof k, "deepseek4.engram.%d.weight_offset", i); ds4gw_kv_u64(&W, k, S.sh[ew->shard].data0 + ew->off);
        snprintf(k, sizeof k, "deepseek4.engram.%d.scale_offset", i); ds4gw_kv_u64(&W, k, S.sh[es->shard].data0 + es->off);
        fprintf(stderr, "[engram] L%d 表 %lld×%lld @ %s +%llu\n", L, (long long)ew->shape[0], (long long)ew->shape[1], rp, (unsigned long long)(S.sh[ew->shard].data0 + ew->off));
    }
    /* ---- 张量登记 ---- */
    plan_fp4("token_embd.weight", "embed.weight");
    for (int L = 0; L < C.n_layers; L++) plan_layer(L);
    for (int T = 0; T < C.n_mtp_layers; T++) plan_mtp(T);   /* DSpark 三塔(段 6 S1) */
    plan_small("output_norm.weight", "norm.weight");
    plan_fp4("output.weight", "head.weight");
    { uint64_t ne[3];
      ne[0] = n_tm; add_job(J_RAW, "engram.token_map", 26, 1, ne, n_tm * 4, NULL, -1, tm);
      ne[0] = eng; ne[1] = enl; add_job(J_RAW, "engram.multipliers", 27, 2, ne, n_mul * 8, NULL, -1, mul);
      ne[0] = enh; ne[1] = eng - 1; ne[2] = enl; add_job(J_RAW, "engram.primes", 27, 3, ne, n_pr * 8, NULL, -1, pr);
      ne[0] = (uint64_t)(eng - 1) * enh; ne[1] = enl; add_job(J_RAW, "engram.offsets", 27, 2, ne, n_off * 8, NULL, -1, offs); }
    fprintf(stderr, "[登记] %d 张量, 数据区 %.3f GB\n", W.nt, (double)W.total / 1e9);
    if (ds4gw_header(&W)) return 1;
    /* ---- 逐张量生成+写 ---- */
    int last_layer = -1;
    for (int i = 0; i < g_nj; i++) {
        const job_t *j = &g_jobs[i]; uint64_t n = 0; const void *d = g_buf;
        switch (j->kind) {
            case J_F32_BF16: case J_F32_COPY: case J_BF16_COPY: n = gen_small(j); d = g_buf; break;
            case J_FP4X32: n = gen_fp4(j); d = g_buf; break;
            case J_FP8BLK: n = gen_fp8blk(j); d = g_buf; break;
            case J_VQBLOB: n = gen_vqblob(j); d = g_buf; break;
            case J_RAW: n = W.t[i].nbytes; d = j->raw; break;
        }
        if (ds4gw_write(&W, i, d, n)) return 1;
        g_out_bytes += n;
        if (j->kind == J_VQBLOB) { fprintf(stderr, "[L%02d] 专家 blob %.3f GB, 累计 %.2f GB\n", j->layer, (double)n / 1e9, (double)g_out_bytes / 1e9); fflush(stderr); }
        if (j->layer >= 0 && j->layer != last_layer) { last_layer = j->layer; v41_st_release_idle(&S); }
    }
    if (ds4gw_end(&W)) return 1;
    fprintf(stderr, "[完成] %s: %d 张量, %.3f GB 数据(十进制)\n", out, W.nt, (double)g_out_bytes / 1e9);
    v41_st_close(&S);
    return 0;
}
