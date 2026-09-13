/* finetune_sft_target.inc.c — 一步低秩 SFT 的靶: 把"本应写出的 token 的 NLL"
 * 对 routed 的梯度算出来。被 finetune_solve.c 以 --mode sft 包含(单 TU)。
 *
 * ★为什么换这个靶★(2026-09-10, 前两轮定罪之后):
 *   ① routed 靶  = 教师−学生的 routed 差 → 43 层 held-out 全负。真因是离散路由:
 *      读了上下文后每层 1.3~1.7 个专家被换掉, 一个作用在 x 上的平滑低秩加性映射
 *      原理上表示不了这种跳变。
 *   ② rte 靶     = 教师−学生的路由 logits 差 → 拟合到 +12~21%, 但前 6 专家重合率
 *      纹丝不动(±0.004): 拿回的那 15% 不足以让任何一个专家换位。
 *   两轮的共同病根: 靶是"教师−学生的中间层差", 那是代理目标, 而且教师那份差里
 *   大半是它在抄事后上下文里的次日价格 —— 学生的 x 里根本没有这个信息, 解不出来。
 *   ③ 这里的靶 = 损失本身对 routed 的梯度。判据和目标函数变成同一个数(NLL),
 *      不会再出现"内部指标好、端到端不动"那种测量分叉。梯度是连续量, 天然适合
 *      低秩连续映射。
 *
 * 链条(最后一层闭式, 不需要自动微分 —— ds4 也没有反向):
 *   dL/dlogits        = p − onehot(t)
 *   dL/d(output_norm) = Wᵀ·dL/dlogits = Σ_v p_v·W[v] − W[t]
 *   dL/d(output_embd) = Jᵀ_rmsnorm · 上
 *   dL/d(routed_42)   = α · 上,  α = Σ_c w_c·split_c(逐 token 正标量)
 *
 * ★三处近似, 都可测量, 不是拍脑袋★:
 *   1. top-K: 只取 p 的前 K 项。实测 K=64 覆盖 97.3% 概率质量, 丢掉的 2.7% 分散在
 *      12.9 万个方向上近似正交, 求和后大部分互相抵消。K 调大即可收紧, 覆盖率随
 *      topk 文件一起落盘, 糙到什么程度是能读出来的。
 *   2. RMSNorm 雅可比只取主项 (g⊙v)/rms, 丢掉沿 x 的投影项 x·(x·(g⊙v))/(d·rms³)。
 *      高维下该项量级 ~1/d。rms 是逐 token 正标量, 与 α 一起吸收进步长 η。
 *   3. α 省略 = 最小二乘里给每个样本均匀权重, 只改各样本相对权重, 不改靶方向。
 *   ★这三项都只影响幅度/权重, 不改梯度方向★ —— 所以"方向对不对"这一判决不受它们
 *   影响; 真要收紧, 逐个补回来即可(引擎顺手落 output_embd 就能把 2、3 做精确)。 */

/* 引擎 --eval-topk 的产物: 头 <magic 'ETGD', K, S, VOCAB>, 之后每行
 * <row u32><tgt i32><tgt_p f32><mass f32><ids i32[K]><ps f32[K]>。 */
typedef struct {
    uint32_t *row;      /* [n] 行号(与 --eval-ids 的 ids 行号同一口径) */
    int      *tgt;      /* [n] 目标 token; <0 = 末位没有下一个 token */
    float    *mass;     /* [n] top-K 覆盖的概率质量 */
    int      *ids;      /* [n*K] */
    float    *ps;       /* [n*K] */
    uint32_t  n, K, S, vocab;
} sft_topk;

static void sft_topk_read(const char *path, sft_topk *o) {
    FILE *f = fopen(path, "rb");
    if (!f) die("--topk 打不开: %s", path);
    uint32_t hd[4];
    if (fread(hd, 4, 4, f) != 4 || hd[0] != 0x44475445u)
        die("%s: 不是 --eval-topk 产物(magic 对不上)", path);
    o->K = hd[1]; o->S = hd[2]; o->vocab = hd[3];
    if (!o->K || o->K > 4096u) die("%s: K=%u 不合理", path, o->K);
    const size_t rec = 4 + 4 + 4 + 4 + (size_t)o->K * 8;
    long pos = ftell(f);
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f) - pos;
    fseek(f, pos, SEEK_SET);
    if (sz <= 0 || (size_t)sz % rec) die("%s: 记录区 %ld 字节不是 %zu 的整数倍", path, sz, rec);
    o->n = (uint32_t)((size_t)sz / rec);
    o->row = malloc((size_t)o->n * 4);
    o->tgt = malloc((size_t)o->n * 4);
    o->mass = malloc((size_t)o->n * 4);
    o->ids = malloc((size_t)o->n * o->K * 4);
    o->ps  = malloc((size_t)o->n * o->K * 4);
    if (!o->row || !o->tgt || !o->mass || !o->ids || !o->ps) die("topk 内存不足");
    double msum = 0;
    for (uint32_t i = 0; i < o->n; i++) {
        if (fread(&o->row[i], 4, 1, f) != 1 || fread(&o->tgt[i], 4, 1, f) != 1) die("%s 截断", path);
        float tp;
        if (fread(&tp, 4, 1, f) != 1 || fread(&o->mass[i], 4, 1, f) != 1) die("%s 截断", path);
        (void)tp;
        if (fread(o->ids + (size_t)i * o->K, 4, o->K, f) != o->K ||
            fread(o->ps  + (size_t)i * o->K, 4, o->K, f) != o->K) die("%s 截断", path);
        msum += o->mass[i];
    }
    fclose(f);
    printf("topk: %u 行 K=%u S=%u vocab=%u, 平均覆盖概率质量 %.4f\n",
           o->n, o->K, o->S, o->vocab, o->n ? msum / o->n : 0.0);
}

static void sft_topk_free(sft_topk *o) {
    free(o->row); free(o->tgt); free(o->mass); free(o->ids); free(o->ps);
}

/* 解 output.weight 的第 v 行(Q8_0, 每行 DIM 元素 = DIM/32 块 × 34 字节)到 out[DIM]。
 * dequant 走 src/common 那一份唯一实现 —— 工具侧禁止再抄第二份(CLAUDE.md 铁律)。 */
static void sft_deq_vocab_row(const uint8_t *base, uint32_t v, float *out) {
    const uint64_t nblk = DIM / 32u;
    ds4_deq_q8_0(base + (uint64_t)v * nblk * 34u, nblk, out);
}

/* 逐行算靶 R[i] = −η · onorm ⊙ (Σ_v p_v·W[v] − W[t]);  X[i] = 学生第 42 层的 x。
 * 返回实际用上的行数(tgt<0 的行没有目标 token, 跳过)。 */
/* keep[] 非 NULL 时只收 keep[row]!=0 的行。★为什么要过滤★: SFT 的 loss 只该落在
 * completion 上 —— 每条样本是 [当日材料][本应写的报告], 材料段是 prompt, 模型不需要
 * 学着去"预测材料"。不滤的话材料段(15690 行里 4792 行)照样进正规方程, 白白稀释靶,
 * 而且教出来的是"更会背材料"而不是"更会写对报告"。 */
static int sft_build_target(const sft_topk *tk, const uint8_t *wq8, const float *onorm,
                            const float *Xall, long xrows, int xshift,
                            float eta, float *X, float *R, const unsigned char *keep) {
    float *wrow = malloc((size_t)DIM * sizeof(float));
    double *acc = malloc((size_t)DIM * sizeof(double));
    if (!wrow || !acc) die("靶缓冲内存不足");
    int m = 0;
    for (uint32_t i = 0; i < tk->n; i++) {
        if (tk->tgt[i] < 0) continue;
        if (keep && !keep[tk->row[i]]) continue;
        const long xr = (long)tk->row[i] - xshift;
        if (xr < 0 || xr >= xrows) continue;          /* 捕获行号外: 静默跳过会出假账, 这里只收对得上的 */
        for (uint32_t j = 0; j < DIM; j++) acc[j] = 0.0;
        for (uint32_t q = 0; q < tk->K; q++) {
            const float p = tk->ps[(size_t)i * tk->K + q];
            if (p <= 0.0f) continue;
            sft_deq_vocab_row(wq8, (uint32_t)tk->ids[(size_t)i * tk->K + q], wrow);
            for (uint32_t j = 0; j < DIM; j++) acc[j] += (double)p * wrow[j];
        }
        sft_deq_vocab_row(wq8, (uint32_t)tk->tgt[i], wrow);
        float *rd = R + (size_t)m * DIM, *xd = X + (size_t)m * DIM;
        const float *xs = Xall + (size_t)xr * DIM;
        for (uint32_t j = 0; j < DIM; j++) {
            /* 负梯度方向: 要降 NLL, 所以靶取 −η·g(ds4_z_solve 解 min‖U z Vᵀx − R‖²) */
            rd[j] = (float)(-(double)eta * onorm[j] * (acc[j] - (double)wrow[j]));
            xd[j] = xs[j];
        }
        m++;
    }
    free(wrow); free(acc);
    return m;
}
