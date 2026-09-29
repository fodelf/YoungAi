/* core_draft_sched.c — 投机解码的置信调度器(mtp-1.md M5′, 2026-09-16; 2026-09-28 成本改为引擎自量): 每一轮验证几位,
 * 由草稿器自己报的置信度与**本请求实测的成本**决定, 而不是钉死一个 k, 也不是钉死三个成本常量。
 *
 * 为什么需要它: 验证第 j 位要多付一份专家字节, 而它只有在"前 j−1 位全中"时才可能兑现。
 * 草稿器的 confidence 头出的就是这个条件概率(官方报告 §2.4.3 原话: per-position conditional
 * acceptance probabilities, 用来估前缀存活率)。09-16 盘上实测: 同一个二进制, README 英文均接受 1.39/5, 金融难文本只有 0.78/3。
 *
 * 判据(纯粹的除法, 不玄):
 *   一轮产出 = 1 + Σ_{j≤k} a_j,  其中 a_j = ∏_{i≤j} ρ·σ(conf_i) = 第 j 位被验到时的前缀存活率
 *   一轮成本 = 草稿 + 验证(1+k 行)   (毫秒, 本请求实测)
 *   选 k = argmax 产出/成本; 与"纯解码一步产 1 个"(1/走图一步 ms)比, 比值 > 1 才值得投机。
 *
 * ★成本为什么改成引擎自量(2026-09-28, 用户铁律"不得为了速度写死任何硬编码")★: 以前三个数(草稿/验 1 行/每多一行, 以纯解码一步为 1)是
 * 盘上量的编译期常量, 核一变快常量还是旧的, 调度器就拿旧账把投机整个挡死 —— 09-17 实撞(新核落地后 40 个 token 里 0 轮草稿),
 * 之后 09-19、09-24 各重标了两次。现在每请求自己记: 走图一步 / 草稿一轮 / 验证 n 行 各自的墙钟均值, 验证成本按 n 做加权最小二乘
 * (验 n 行 = v1 + tok·(n−1); 验 1 行就是一个单 token 步, 所以 n=1 那个点 = 纯解码步的均值), 没量到的 n 用拟合线。
 * 起步没有样本时先验最长的一块(k = block)量一次, 再补一个 n=2 的点, 之后就全按实测走。
 *
 * ★为什么以前不接墙钟, 现在可以★: 09-16 的顾虑是"温 0 下输出不许随时钟变"(V4 08-21 撞过: 那版的批路不逐字节, 用墙钟仲裁 k 就等于用
 * 时钟改输出)。V4.1 的投机 == 纯解码 逐字节(d1 门每改必验), k 只影响快慢不影响输出, 所以 k 随墙钟变是安全的; 变的只是速度的可复现性。
 * ρ(接受率校准)仍只由 token 史定。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU
#include <math.h>

static float v41_sched_sigmoid(float c) { return isfinite(c) ? 1.0f / (1.0f + expf(-c)) : 0.0f; }   /* 非有限 conf = 没把握 */

/* ★采样下的在线校准(2026-09-28, 声明见 core_v41.h)★: conf 头学的是"贪心会不会同选", 采样下接受概率 = 目标分布给草稿的概率, 系统性偏低。
 * ρ = 本请求出过草稿的轮里 首位命中数 / 首位 σ(conf) 之和, 乘到每位 σ 上(封顶 1)。贪心下 ρ 在 1 附近, 是对 conf 头本身的校正。
 * 每一轮都观测(k=0 的轮看吐出的 token 是否就是草稿首位), 否则首轮被拒就死锁在 ρ=0。 */
void v41_sched_observe(v41_sched *s, const float *conf, int hit) {
    if (!s) return;
    s->pred1 += (double)v41_sched_sigmoid(conf[0]);
    s->real1 += hit ? 1.0 : 0.0;
}

/* 成本样本: n_rows = 1 是一个单 token 步(纯解码步或 k=0 轮的图步), ≥ 2 是验证 1+k 行; ms = 这一发的墙钟(发到等完) */
void v41_sched_cost(v41_sched *s, uint32_t n_rows, double ms) {
    if (!s || n_rows == 0u || n_rows > DS4_MTP_MAX_BLOCK + 1u || !(ms > 0.0)) return;
    s->ms[n_rows] += ms; s->cnt[n_rows]++;
}
void v41_sched_draft_cost(v41_sched *s, double ms) { if (s && ms > 0.0) { s->draft_ms += ms; s->draft_n++; } }

/* 验证 n 行的毫秒: 按已有样本(每个 n 一个均值, 计数为权)做加权最小二乘 ms(n) = v1 + tok·(n−1); 只有一个 x 时退化成常数。
 * 返回 false = 一个样本都没有。 */
static bool v41_sched_fit(const v41_sched *s, double *v1, double *tok) {
    double W = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    uint32_t nx = 0; double x0 = 0;
    for (uint32_t n = 1; n <= DS4_MTP_MAX_BLOCK + 1u; n++) {
        if (!s->cnt[n]) continue;
        const double w = (double)s->cnt[n], x = (double)(n - 1u), y = s->ms[n] / w;
        if (nx == 0) x0 = x;
        if (nx == 0 || x != x0) nx++;
        W += w; Sx += w * x; Sy += w * y; Sxx += w * x * x; Sxy += w * x * y;
    }
    if (W <= 0.0) return false;
    const double den = W * Sxx - Sx * Sx;
    if (nx < 2u || den <= 0.0) { *v1 = Sy / W; *tok = 0.0; return true; }   /* 一种 n: 只能当常数 */
    *tok = (W * Sxy - Sx * Sy) / den;
    *v1 = (Sy - *tok * Sx) / W;
    return true;
}

/* conf[0..block-1] = 草稿器每位的置信 logit(不是概率, sigmoid 之后才是)。
 * 返回这一轮该验几位(0 = 一位都不值得验, 本轮草稿白跑); *value_out 给出预测的"产出/成本"比值(以纯解码一步的产出率为 1),
 * 调用方拿它计数"判亏本"的轮。 */
uint32_t v41_draft_pick_k(const float *conf, uint32_t block, float *value_out, const v41_sched *s) {
    if (value_out) *value_out = 1.0f;
    if (!s || block == 0u) return block;
    /* 起步: 还没有验证样本 ⇒ 先验最长的一块量一次; 只有一个 n≥2 的点 ⇒ 补一个 n=2(或最长)的点, 拟合线才有斜率 */
    uint32_t nmeas = 0, first = 0;
    for (uint32_t n = 2; n <= block + 1u; n++) if (s->cnt[n]) { if (!nmeas) first = n; nmeas++; }
    if (nmeas == 0u) return block;
    if (nmeas == 1u && !s->cnt[1]) return first == 2u ? block : 1u;
    double v1, tok;
    if (!v41_sched_fit(s, &v1, &tok)) return block;
    /* 纯解码一步的毫秒: 有真实的单 token 步就用它(k=0 轮走图的步也算), 否则用拟合线在 n=1 的值(验 1 行 ≈ 一步) */
    const double step = s->cnt[1] ? s->ms[1] / (double)s->cnt[1] : v1;
    const double draft = s->draft_n ? s->draft_ms / (double)s->draft_n : 0.0;
    if (!(step > 0.0)) return block;
    const float rho = s->pred1 > 0.0 ? (float)(s->real1 / s->pred1) : 1.0f;
    float surv = 1.0f, sum = 0.0f;
    double best = 1.0 / (draft + v1);   /* k=0: 草稿白跑, 只走一个单 token 步 */
    uint32_t bestk = 0;
    for (uint32_t j = 0; j < block; j++) {
        const float pj = rho * v41_sched_sigmoid(conf[j]);
        surv *= pj < 1.0f ? pj : 1.0f;
        sum += surv;
        const double cost = draft + v1 + tok * (double)(j + 1u);   /* 验 1+(j+1) 行 */
        const double val = (1.0 + (double)sum) / (cost > 0.0 ? cost : step);
        if (val > best) { best = val; bestk = j + 1u; }
    }
    if (value_out) *value_out = (float)(best * step);   /* 1.0 = 与纯解码一步持平 */
    return bestk;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_draft_sched_nonempty_tu;
