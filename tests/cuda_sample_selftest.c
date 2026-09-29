/* cuda_sample_selftest.c — 设备采样核(src/cuda/cuda_v41_sample.inc.cu)的分布门(2026-09-28)。只要 CUDA 设备, 不要模型。
 *
 * 门是分布级的, 不是逐字节: 采样路没有"应该吐哪个 token"的金标, 有的是"吐每个 token 的频率必须等于目标分布"。
 * 三件事各一组配方(温度 / top_k / top_p / min_p):
 *   ① 全分布样本: 抽 N 次(位置 0..N-1 当随机数计数器), 保留集外频率必须恰为 0(截断是精确的), 保留集内前 32 个 token 的
 *      频率与截断后归一的概率之差 ≤ 5σ(σ = √(p(1−p)/N)), 全表 |Σ|f−p|| 的一半(总变差) ≤ 界。
 *   ② 投机的边缘分布: 固定草稿 d, 吐出 = 接受 ? d : 残差样本, 它的频率同样必须等于 p(拒绝采样的正确性就是这一句);
 *      接受率必须 ≈ p(d)。
 *   ③ 确定性: 同参数同位置再抽一遍, 四个字逐位相同(图路与直发路同一个核 ⇒ 这就是"同一条请求两种发法出同一串 token")。
 * 出错会怎样: 保留集外出现样本 = 门槛键算错(基数选择/键变换); 前 32 项某项超 5σ = Gumbel 键或随机数偏; 投机边缘不等于 p =
 * 接受硬币或残差抽错(投机路会悄悄改分布, 采样下从文本上看不出来 —— 这就是为什么要这道门)。 */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define V 129280u          /* 与 V4.1 词表同大, 让线程跨步/桶统计的形状与生产一致 */
#define ROWS 64u           /* 一发核出 64 行(位置连续), 攒够 N 个样本 */
#define N 262144u          /* 样本数 */

typedef struct { float temperature, top_p, min_p; int top_k; const char *name; } recipe;

static uint64_t rs = 0x1234567ull;
static float rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 40) & 0xFFFFFFu) / 16777216.0f; }

/* 主机金标: 与 ds4_sample_logits 同一套语义算"保留集 + 归一概率"(double) */
static int cmp_desc(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (y > x) - (y < x); }
static void reference(const float *l, const recipe *r, double *p /* [V] 截断归一后的概率, 0 = 不在保留集 */) {
    double M = -INFINITY; for (uint32_t i = 0; i < V; i++) if (l[i] > M) M = l[i];
    static double m[V], sorted[V];
    double cut_k = -INFINITY;
    if (r->top_k > 0 && (uint32_t)r->top_k < V) {
        for (uint32_t i = 0; i < V; i++) sorted[i] = l[i];
        qsort(sorted, V, sizeof sorted[0], cmp_desc);
        cut_k = sorted[r->top_k - 1];
    }
    double Z = 0.0;
    for (uint32_t i = 0; i < V; i++) { m[i] = l[i] >= cut_k ? exp(((double)l[i] - M) / r->temperature) : 0.0; Z += m[i]; }
    double cut_p = -INFINITY;
    if (r->top_p < 1.0f) {   /* 从大到小累计首次 ≥ top_p·Z 的那一项的 logit(并列一起保留) */
        uint32_t n = 0; for (uint32_t i = 0; i < V; i++) if (m[i] > 0.0) sorted[n++] = l[i];
        qsort(sorted, n, sizeof sorted[0], cmp_desc);
        double cum = 0.0;
        for (uint32_t i = 0; i < n; i++) { cum += exp((sorted[i] - M) / r->temperature); if (cum >= (double)r->top_p * Z) { cut_p = sorted[i]; break; } }
    }
    const double cut_m = r->min_p > 0.0f ? M + r->temperature * log((double)r->min_p) : -INFINITY;
    double ZK = 0.0;
    for (uint32_t i = 0; i < V; i++) { if (m[i] <= 0.0 || l[i] < cut_p || l[i] < cut_m) m[i] = 0.0; ZK += m[i]; }
    for (uint32_t i = 0; i < V; i++) p[i] = m[i] / ZK;
}

static int check_freq(const char *what, const uint32_t *cnt, const double *p, uint32_t n_samples) {
    int bad = 0; double tv = 0.0; uint32_t outside = 0;
    for (uint32_t i = 0; i < V; i++) { tv += fabs((double)cnt[i] / n_samples - p[i]); if (p[i] == 0.0 && cnt[i]) outside += cnt[i]; }
    tv *= 0.5;
    /* 前 32 项按 p 排序逐项 5σ */
    uint32_t top[32]; uint32_t nt = 0;
    for (uint32_t i = 0; i < V; i++) {
        if (p[i] <= 0.0) continue;
        uint32_t j;
        if (nt < 32u) j = nt++;
        else { if (p[i] <= p[top[31]]) continue; j = 31u; }
        while (j > 0 && p[top[j - 1]] < p[i]) { top[j] = top[j - 1]; j--; }
        top[j] = i;
    }
    double worst = 0.0; uint32_t worst_i = 0;
    for (uint32_t t = 0; t < nt; t++) {
        const uint32_t i = top[t];
        const double f = (double)cnt[i] / n_samples, sd = sqrt(p[i] * (1.0 - p[i]) / n_samples);
        const double z = sd > 0.0 ? fabs(f - p[i]) / sd : 0.0;
        if (z > worst) { worst = z; worst_i = i; }
    }
    /* 总变差的期望 ≈ Σ√(p(1−p)/N)/2 之类, 对 12 万项的长尾偏大; 用 0.05 作粗界(核对不上时它会是 0.3~1) */
    if (outside || worst > 5.0 || tv > 0.05) bad = 1;
    printf("  %-14s 保留集外 %u 个样本, 前 %u 项最差 %.2fσ(词 %u: 频率 %.5f vs p %.5f), 总变差 %.4f %s\n", what, outside, nt, worst, worst_i,
           (double)cnt[worst_i] / n_samples, p[worst_i], tv, bad ? "★红★" : "✓");
    return bad;
}

int main(void) {
    if (!ds4_gpu_init()) { fprintf(stderr, "★没有 CUDA 设备★\n"); return 1; }
    static float l[V]; static double p[V]; static uint32_t cnt[V], cnt_spec[V];
    /* 合成 logits: 长尾 + 几十个尖峰, 像真实出口的形状(最大项概率几成) */
    for (uint32_t i = 0; i < V; i++) l[i] = -6.0f * rnd() - 4.0f;
    for (uint32_t t = 0; t < 40; t++) l[(uint32_t)(rnd() * V)] = 2.0f + 6.0f * rnd();
    l[777] = 9.5f; l[4242] = 8.8f; l[100000] = 8.0f;
    l[5] = -INFINITY;   /* 非有限项: 两边都要跳过 */
    const recipe R[] = {
        { 1.0f, 1.0f, 0.0f, 0, "温1 全表" },
        { 0.7f, 0.9f, 0.05f, 0, "温0.7 p.9 m.05" },
        { 1.3f, 0.95f, 0.0f, 40, "温1.3 p.95 k40" },
        { 1.0f, 1.0f, 0.0f, 1, "k1(退化贪心)" },
    };
    /* 草稿分布 q: p 的 logits 加噪(q ≠ p 但相近, 像草稿塔): ④ 草稿从 q 抽、验证核读 q, 吐出边缘仍须是 p, 接受率 ≈ Σmin(p,q) */
    static float lq[V]; static double q[V]; static uint32_t cnt_q[V];
    for (uint32_t i = 0; i < V; i++) lq[i] = isfinite(l[i]) ? l[i] + 1.5f * (rnd() - 0.5f) : l[i];
    ds4_gpu_tensor *tl = ds4_gpu_tensor_alloc((uint64_t)ROWS * V * 4u), *tq = ds4_gpu_tensor_alloc((uint64_t)ROWS * V * 4u);
    ds4_gpu_tensor *tpos = ds4_gpu_tensor_alloc(ROWS * 4u), *ttok = ds4_gpu_tensor_alloc(ROWS * 4u), *tout = ds4_gpu_tensor_alloc(ROWS * 16u);
    ds4_gpu_tensor *ttokq = ds4_gpu_tensor_alloc(ROWS * 4u), *toutq = ds4_gpu_tensor_alloc(ROWS * 16u);
    if (!tl || !tq || !tpos || !ttok || !tout || !ttokq || !toutq) { fprintf(stderr, "★分配失败★\n"); return 1; }
    for (uint32_t r = 0; r < ROWS; r++)
        if (!ds4_gpu_tensor_write(tl, (uint64_t)r * V * 4u, l, (uint64_t)V * 4u) || !ds4_gpu_tensor_write(tq, (uint64_t)r * V * 4u, lq, (uint64_t)V * 4u)) return 1;
    int32_t pos[ROWS], tok[ROWS], out[ROWS * 4], out2[ROWS * 4], noneq[ROWS], outq[ROWS * 4];
    for (uint32_t i = 0; i < ROWS; i++) noneq[i] = -1;   /* 抽草稿那一发: 没有草稿 */
    if (!ds4_gpu_tensor_write(ttokq, 0, noneq, sizeof noneq)) return 1;
    int fail = 0;
    const int32_t d = 4242;   /* 点质量草稿: 第二大的项(接受率 = p(d) 几成, 拒绝路也能量到) */
    for (size_t ri = 0; ri < sizeof R / sizeof R[0]; ri++) {
        const recipe *r = &R[ri];
        reference(l, r, p); reference(lq, r, q);
        ds4_gpu_sample_params sp = { r->temperature, r->top_p, r->min_p, r->top_k, 0x9E37ull + (uint64_t)ri, 0u };
        ds4_gpu_sample_params spq = sp; spq.stream = 1u;
        memset(cnt, 0, sizeof cnt); memset(cnt_spec, 0, sizeof cnt_spec); memset(cnt_q, 0, sizeof cnt_q);
        uint32_t nacc = 0, ndet = 0, naccq = 0;
        for (uint32_t s = 0; s < N; s += ROWS) {
            for (uint32_t i = 0; i < ROWS; i++) { pos[i] = (int32_t)(s + i); tok[i] = d; }   /* 每行草稿 = 下一行的输入 = d(末行无) */
            if (!ds4_gpu_tensor_write(tpos, 0, pos, sizeof pos) || !ds4_gpu_tensor_write(ttok, 0, tok, sizeof tok)) return 1;
            if (!ds4_gpu_v41_sample_tensor(tout, tl, 0u, ROWS, V, tpos, ttok, &sp, NULL) || !ds4_gpu_synchronize() ||
                !ds4_gpu_tensor_read(tout, 0, out, sizeof out)) { fprintf(stderr, "★采样核失败★\n"); return 1; }
            if (s == 0) {   /* ③ 确定性: 同一发再来一遍 */
                if (!ds4_gpu_v41_sample_tensor(tout, tl, 0u, ROWS, V, tpos, ttok, &sp, NULL) || !ds4_gpu_synchronize() ||
                    !ds4_gpu_tensor_read(tout, 0, out2, sizeof out2)) return 1;
                ndet = memcmp(out, out2, sizeof out) == 0 ? 1u : 0u;
            }
            for (uint32_t i = 0; i < ROWS; i++) {
                const int32_t *o = out + 4u * i;
                if (o[0] < 0 || (uint32_t)o[0] >= V) { fprintf(stderr, "★样本越界 %d★\n", o[0]); return 1; }
                cnt[o[0]]++;
                if (i + 1u < ROWS) { cnt_spec[o[1] ? d : o[2]]++; nacc += o[1] ? 1u : 0u; }
            }
            /* ④ 草稿从 q 抽(流 1, 与引擎草稿塔同一发核), 填成下一行的输入, 再按 q 验证 */
            if (!ds4_gpu_v41_sample_tensor(toutq, tq, 0u, ROWS, V, tpos, ttokq, &spq, NULL) || !ds4_gpu_synchronize() ||
                !ds4_gpu_tensor_read(toutq, 0, outq, sizeof outq)) return 1;
            tok[0] = -1; for (uint32_t i = 0; i + 1u < ROWS; i++) tok[i + 1u] = outq[4u * i];
            if (!ds4_gpu_tensor_write(ttok, 0, tok, sizeof tok)) return 1;
            if (!ds4_gpu_v41_sample_tensor(tout, tl, 0u, ROWS, V, tpos, ttok, &sp, tq) || !ds4_gpu_synchronize() ||
                !ds4_gpu_tensor_read(tout, 0, out, sizeof out)) return 1;
            for (uint32_t i = 0; i + 1u < ROWS; i++) { const int32_t *o = out + 4u * i; cnt_q[o[1] ? tok[i + 1u] : o[2]]++; naccq += o[1] ? 1u : 0u; }
        }
        const uint32_t n_spec = N / ROWS * (ROWS - 1u);
        printf("== %s: 保留集 %d 项(核报), 确定性 %s\n", r->name, out[3], ndet ? "✓" : "★两遍不同★");
        fail |= !ndet;
        fail |= check_freq("① 全分布", cnt, p, N);
        fail |= check_freq("② 点质量草稿边缘", cnt_spec, p, n_spec);
        const double acc = (double)nacc / n_spec, sd = sqrt(p[d] * (1.0 - p[d]) / n_spec);
        const int accbad = fabs(acc - p[d]) > 5.0 * sd + 1e-9;
        printf("  接受率 %.5f vs p(草稿) %.5f (%.2fσ) %s\n", acc, p[d], sd > 0 ? fabs(acc - p[d]) / sd : 0.0, accbad ? "★红★" : "✓");
        fail |= accbad;
        fail |= check_freq("④ 分布草稿边缘", cnt_q, p, n_spec);
        double emin = 0.0; for (uint32_t i = 0; i < V; i++) emin += p[i] < q[i] ? p[i] : q[i];
        const double accq = (double)naccq / n_spec, sdq = sqrt(emin * (1.0 - emin) / n_spec);
        const int accqbad = fabs(accq - emin) > 5.0 * sdq + 1e-9;
        printf("  分布草稿接受率 %.5f vs Σmin(p,q) %.5f (%.2fσ) %s\n", accq, emin, sdq > 0 ? fabs(accq - emin) / sdq : 0.0, accqbad ? "★红★" : "✓");
        fail |= accqbad;
    }
    printf(fail ? "★采样核分布门: 有红★\n" : "采样核分布门: 全绿\n");
    return fail;
}
