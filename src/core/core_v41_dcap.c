/* core_v41_dcap.c — DSpark 草稿器的对齐取料(mtp.md M6, 2026-09-16)。
 *
 * 要解决的问题: 草稿器是照**原始 FP 模型**训的, 而我们部署的是量化+反修的底座。在线首位接受率
 * **0.58~0.64**(2K README), 而底座对原模型的 Same top 是 **0.72**(金融主尺 j 8192, V4.1 教师)。
 * "草稿器本身是好的, 是底座漂移把它的收益吃掉了"是**待验的假设**, 不是结论 —— 这份取料配
 * gguf-tools/bench/dspark_agree 的三个一致率就是验它的尺(2026-09-17 那次判决用的料错位、锚又是
 * V4 的, 两个数都不算, 见下面"口径"一段与 fable5 09-18)。
 *
 * 摘掉这个天花板的办法: 让草稿器改盯**部署底座**说话。两边的 logits 都由**同一个出口头**算
 * (官方 forward_head 借主模型的 head), 所以只要把草稿器喂给头的那个隐态掰到主模型喂给头的那个,
 * logits 自然就对齐了 —— 于是这件事塌缩成一个最普通的最小二乘:
 *
 *     min ‖ X·(I + BᵀA) − Y ‖²,  X = 草稿器的出口隐态, Y = 主模型的出口隐态(同一位置)
 *
 * 形式与反修放大器一模一样(y += x·(B·A)), 所以引擎侧能直接复用 ds4_gpu_v41_amp_apply_tensor,
 * 产物也是几十 MB 的边车, **主模型一个字节不碰, 不重转 GGUF**。这一片只负责取料。
 *
 * 怎么取: 教师强制走一遍 token 序列, 一次一位(块 1), 第 i 步(i ≥ 1)两件事按这个顺序 ——
 *   ①草稿: 主模型已经处理到位置 i-1, 拿它的 main_hidden(i-1) 与"下一个真 token" ids[i] 出一块
 *     (块首位坐在位置 i), 首位草稿预测的是位置 i+1 → 出口隐态 X 与草稿首位
 *   ②主模型再处理位置 i → 出口隐态 Y 与它的 argmax(同样预测位置 i+1)
 * 两个 token 一比就是**首位接受率 p1 的直接测量**, 也就是 M6 的判决基线。
 *
 * ★口径必须与在线一字不差(2026-09-18 实撞, 这是第一版的 bug)★: 在线一轮是
 *   v41_draft_step(tok = 刚采出、还没进主模型的那个 token, pos_main = 主模型最后处理的位置)
 * 教师强制下"还没进主模型的下一个 token"就是 ids[i], main_hidden 是位置 i-1 的。第一版写成了
 * "先让主模型吃 ids[i], 再用 main_hidden(i) 和 tok = ids[i] 出草稿" —— 等于把 ids[i] 在位置 i+1
 * 又摆了一遍, 草稿器猜的是"ids[i] ids[i] 后面是什么", 却拿去跟主模型对位置 i+1 的预测比。
 * 不报错, 只是三个一致率(①③)全量低了, 顺带把两次"草稿器对齐"的料也喂错了(判负两次的真因)。
 * 症状: 首位一致率(0.56)与在线 p1(0.58~0.64)看着同量级, 所以没被发现 —— 要抓它只能逐式对官方
 * generate 环: model(x[:, i], i) → forward_spec(output_ids, main_hidden, i), output_ids 是**下一位**。
 *
 * ★为什么必须一位一块★: 草稿器的窗口装的是"已确认位置的 main_x 投影", 要一位一位推进;
 * 而 main_hidden 只留最后几行。块开大就取不到每个位置的那一份, 不报错, 只会取到错位的料。
 * ★为什么要 --decoder-full★: 块 1 时 CED 会让非末块只跑到分界层、不出 logits, Y 直接是错的, 也不报错。
 *
 * 出错会怎样: p1 量出来接近 0 = 草稿器根本没接上(main_hidden 取错层/窗口没推进);
 * p1 与在线生成时量到的对不上 = 取料与部署不同路, 那样解出来的放大器是假账(铁律: 捕获必须与部署同路)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU

/* 盘上格式: 16 B 头 + 每位置 [X f32 D][Y f32 D] + 尾部 [主模型 token i32][草稿首位 i32] × n。
 * 头里写 D 与 n, 解算器按它读 —— 两边各写各的尺寸迟早对不上, 而错位不会报错。
 * pos0 = 第 0 对对应的**主模型位置**(它的 argmax 预测 pos0+1): 位置 0 没有可配的草稿(草稿要从
 * main_hidden(0) 出, 首位预测的是位置 2), 所以第 0 对是位置 1, pos0 = 1。锚(FP logits)的行号
 * = pos0 + 对号, 一致率工具按它对行; 旧文件这一格是 0(那批料本身就是错位的, 不再用)。
 * 2026-09-29 起 pos0 = n_prompt(提示段按块预填时第 0 对是位置 n_prompt; 老口径 n_prompt=1 不变)。 */
typedef struct { char magic[4]; uint32_t d, n, pos0; } v41_dcap_hdr;

/* ★金标夹具的料(2026-09-18)★ <out>.fix: 20 B 头 {"DFIX", n_target, D, block, n} + 每对
 * [草稿器吃的 main_hidden f32 n_target×D][块首位 token i32][它出的 block 位草稿 i32][block 个 conf f32]。
 * 夹具 gguf-tools/scripts/v41_dspark_fixture.py 拿**官方** mtp.* 三塔喂同一份 main_hidden, 逐位比草稿 id。
 * 为什么非要这一份: 修过口径的三个一致率说草稿器在最好猜的位置也只中 76%(底座同档 97%), 而
 * "我们的塔算错了"与"量化隐态把塔带偏了"两种病在 p1 上长得一模一样, 只有同一份输入喂两套实现才分得开。 */
typedef struct { char magic[4]; uint32_t n_target, d, block, n; } v41_dfix_hdr;

/* 接受率陪审团(2026-09-29): 同一位置上 主模型分布 p = softmax(lm/T) 与 草稿塔首位分布 q = softmax(ld/T)(全词表, 与部署默认 top_p 1 / min_p 0 同),
 * 点质量草稿的期望接受率 = p(argmax q), 草稿按分布抽的期望接受率 = Σmin(p,q)。主机 double 算, 是判官不是生产路。 */
static void v41_dcap_jury(const float *lm, const float *ld, uint32_t V, double T, double *pm, double *qm, double *acc_pm, double *acc_dist) {
    double Mm = -INFINITY, Md = -INFINITY; uint32_t qi = 0;
    for (uint32_t i = 0; i < V; i++) {
        if (lm[i] > Mm) Mm = lm[i];
        if (ld[i] > Md) { Md = ld[i]; qi = i; }
    }
    double Zm = 0.0, Zd = 0.0;
    for (uint32_t i = 0; i < V; i++) {
        pm[i] = isfinite(lm[i]) ? exp(((double)lm[i] - Mm) / T) : 0.0; Zm += pm[i];
        qm[i] = isfinite(ld[i]) ? exp(((double)ld[i] - Md) / T) : 0.0; Zd += qm[i];
    }
    double smin = 0.0;
    for (uint32_t i = 0; i < V; i++) { const double p = pm[i] / Zm, q = qm[i] / Zd; smin += p < q ? p : q; }
    *acc_pm = pm[qi] / Zm; *acc_dist = smin;
}

int ds4_engine_v41_dspark_capture(ds4_engine *e, const int *ids, int n_ids, const char *out_path, int n_prompt) {
    if (!e || !ids || n_ids < 3 || !ds4_engine_is_v41(e)) return 1;   /* 位置 0 只暖主模型, 至少要 1 对 */
    if (!e->metal_ready) { fprintf(stderr, "ds4: V4.1 取料需要 GPU 后端\n"); return 1; }
    const uint32_t D = DS4_N_EMBD;
    uint32_t ctx = (uint32_t)n_ids + 1;
    if (ctx > g_ds4_v41.ctx) { fprintf(stderr, "ds4: 取料序列 %d 超过上下文 %u(模型元数据)\n", n_ids, g_ds4_v41.ctx); return 1; }
    if (n_prompt < 1 || n_prompt > n_ids - 2) n_prompt = 1;   /* 0/越界 = 老口径: 位置 0 暖主模型, 从 1 起逐位 */
    ds4_v41_state st;
    const uint32_t cap = n_prompt > 1 ? (DS4_V41_CHUNK < (uint32_t)n_prompt ? DS4_V41_CHUNK : (uint32_t)n_prompt) : 1u;
    if (!v41_state_alloc(&st, cap, ctx, 0)) return 1;   /* cap=1: 一位一块, 见文件头; 提示段按块预填时 cap = 块; logits 按 cap 开(取料路不走末位捷径) */
    ds4_v41_draft dr;
    if (!v41_draft_alloc(e, &dr)) { fprintf(stderr, "ds4: 这份 GGUF 没带 DSpark 三塔, 取不了料\n"); v41_state_free(&st); return 1; }
    FILE *fo = fopen(out_path, "wb");
    ds4_gpu_tensor *am = ds4_gpu_tensor_alloc(16);
    float *xrow = xmalloc((size_t)D * 4), *yrow = xmalloc((size_t)D * 4);
    int32_t *mtok = xmalloc((size_t)n_ids * 4), *dtok = xmalloc((size_t)n_ids * 4);
    const uint32_t NT = g_ds4_v41.n_mtp_target, B = dr.block;
    float *mh = xmalloc((size_t)NT * D * 4);
    const double T = g_decode_sampling.temperature > 0.f ? (double)g_decode_sampling.temperature : 0.0;   /* > 0 才出陪审团 */
    float *lm = T > 0.0 ? xmalloc((size_t)DS4_N_VOCAB * 4) : NULL, *ld = T > 0.0 ? xmalloc((size_t)DS4_N_VOCAB * 4) : NULL;
    double *pm = T > 0.0 ? xmalloc((size_t)DS4_N_VOCAB * 8) : NULL, *qm = T > 0.0 ? xmalloc((size_t)DS4_N_VOCAB * 8) : NULL;
    double sum_pm = 0.0, sum_dist = 0.0;
    char pfix[4400];
    snprintf(pfix, sizeof pfix, "%s.fix", out_path);
    FILE *ff = fopen(pfix, "wb");
    int rc = 1;
    uint32_t nw = 0, hit = 0;
    do {
        if (!fo || !am || !ff) { fprintf(stderr, "ds4: 取料落盘/暂存分配失败\n"); break; }
        v41_dcap_hdr h = { { 'D','C','A','P' }, D, 0, (uint32_t)n_prompt };
        if (fwrite(&h, sizeof h, 1, fo) != 1) break;
        v41_dfix_hdr hf = { { 'D','F','I','X' }, NT, D, B, 0 };
        if (fwrite(&hf, sizeof hf, 1, ff) != 1) break;
        bool ok = true;
        /* 位置 0..n_prompt-1: 主模型按块预填(n_prompt=1 就是老口径的"位置 0 只暖主模型"), 让 main_hidden 环就位; 这一段没有可配的草稿。
         * 末块至少留 window 个位置(与生成路的预填同规则, core_v41_api.c), 草稿器第一轮补窗口才补得齐。 */
        for (int c0 = 0; ok && c0 < n_prompt; ) {
            uint32_t nc = (uint32_t)(n_prompt - c0) < cap ? (uint32_t)(n_prompt - c0) : cap;
            const uint32_t rest = (uint32_t)(n_prompt - c0) - nc;
            if (rest > 0u && rest < DS4_N_SWA && nc > DS4_N_SWA) nc -= DS4_N_SWA - rest;
            int32_t chunk[DS4_V41_CHUNK];
            for (uint32_t j = 0; j < nc; j++) chunk[j] = (int32_t)ids[c0 + (int)j];
            ok = v41_forward(e, &st, chunk, nc);
            c0 += (int)nc;
        }
        if (!ok) break;
        for (int i = n_prompt; i < n_ids - 1 && ok; i++) {
            const int32_t tok = (int32_t)ids[i];
            /* ①草稿(在线同一口径): tok = 还没进主模型的下一个 token, main_hidden = 主模型最后处理的位置 i-1。
             * 块首位坐在位置 i, 首位草稿预测位置 i+1; 出口隐态第 0 行就是喂给出口头出这一位的那行。 */
            if (!v41_draft_step(e, &st, &dr, tok, (uint32_t)(i - 1))) { ok = false; break; }
            if (!ds4_gpu_synchronize() || !ds4_gpu_tensor_read(dr.st.xn, 0, xrow, (uint64_t)D * 4)) { ok = false; break; }
            if (ld && !ds4_gpu_tensor_read(dr.st.logits, 0, ld, (uint64_t)DS4_N_VOCAB * 4)) { ok = false; break; }   /* 塔首位 logits(+markov 偏置) = q */
            const int32_t guess = dr.host_ids[1];
            /* 夹具料: 这一轮草稿吃的 main_hidden(= st.mainh 第 0 行, 主模型位置 i-1 的三层拼接)+ 首位 token + 全部草稿/conf */
            if (!ds4_gpu_tensor_read(st.mainh, 0, mh, (uint64_t)NT * D * 4) ||
                fwrite(mh, 4, (size_t)NT * D, ff) != (size_t)NT * D || fwrite(&tok, 4, 1, ff) != 1 ||
                fwrite(dr.host_ids + 1, 4, B, ff) != B || fwrite(dr.host_conf, 4, B, ff) != B) { ok = false; break; }
            /* ②主模型处理位置 i → 出口隐态 Y 与 argmax, 预测的同样是位置 i+1 */
            if (!v41_forward(e, &st, &tok, 1u)) { ok = false; break; }
            int32_t want = 0;
            if (!ds4_gpu_v41_argmax_tensor(am, st.logits, 0u, DS4_N_VOCAB) || !ds4_gpu_synchronize() ||
                !ds4_gpu_tensor_read(am, 0, &want, 4) ||
                !ds4_gpu_tensor_read(st.xn, 0, yrow, (uint64_t)D * 4)) { ok = false; break; }
            if (lm) {   /* 陪审团: 同一位置的 p(主模型) 与 q(塔首位), 两种草稿方案的期望首位接受率 */
                double apm, adist;
                if (!ds4_gpu_tensor_read(st.logits, 0, lm, (uint64_t)DS4_N_VOCAB * 4)) { ok = false; break; }
                v41_dcap_jury(lm, ld, DS4_N_VOCAB, T, pm, qm, &apm, &adist);
                sum_pm += apm; sum_dist += adist;
            }
            /* ③同一个目标位置(i+1)的两份, 配对落盘: 第 nw 对 = 主模型位置 i = pos0 + nw */
            if (fwrite(xrow, 4, D, fo) != D || fwrite(yrow, 4, D, fo) != D) { ok = false; break; }
            mtok[nw] = want; dtok[nw] = guess;
            if (want == guess) hit++;
            nw++;
            if ((nw % 64u) == 0u)
                fprintf(stderr, "[dcap] %u/%d 位置, 首位一致 %.3f\r", nw, n_ids - 1 - n_prompt, (double)hit / (double)nw);
        }
        if (!ok) { fprintf(stderr, "\nds4: 取料在第 %u 位失败\n", nw); break; }
        if (fwrite(mtok, 4, nw, fo) != nw || fwrite(dtok, 4, nw, fo) != nw) break;
        h.n = nw;
        if (fseek(fo, 0, SEEK_SET) != 0 || fwrite(&h, sizeof h, 1, fo) != 1) break;
        hf.n = nw;
        if (fseek(ff, 0, SEEK_SET) != 0 || fwrite(&hf, sizeof hf, 1, ff) != 1) break;
        fprintf(stderr, "\n[dcap] 落盘 %s: %u 位置 × 2 × %u f32(第 0 对 = 主模型位置 %u); 夹具料 %s(%u 对 × main_hidden %u×%u + %u 位草稿)\n",
                out_path, nw, D, (unsigned)n_prompt, pfix, nw, NT, D, B);
        /* ★这一行就是 M6 的判决基线★: 草稿器首位 ↔ 部署底座 argmax 的一致率。
         * 它与在线生成时 spec 账里的 p1 应当同量级 —— 差很多就说明取料与部署不同路, 先别解。 */
        fprintf(stderr, "[dcap] ★草稿器首位 ↔ 底座 argmax 一致率 = %.4f (n=%u)★\n", (double)hit / (double)(nw ? nw : 1), nw);
        if (lm && nw)
            fprintf(stderr, "[dcap] ★接受率陪审团 温 %.2f: 点质量草稿 E[p(argmax q)] = %.4f, 分布草稿 E[Σmin(p,q)] = %.4f (n=%u, 提示段 %d 个位置不计)★\n",
                    T, sum_pm / (double)nw, sum_dist / (double)nw, nw, n_prompt);
        /* ★顺带把出口度量也取走(mtp-1.md M6′)★: 解算侧要按"头怎么看这个维度"加权, 而不是按隐态的
         * 欧氏距离 —— 09-16 那次判负(留出一致率不升反降)的真因就是这个度量选错了。
         * 落成 <out>.hdiag(D 个 f32 = 每一列的平方和)。写不出来只警告不停车: 老口径(纯 L2)还能解。 */
        {
            ds4_gpu_tensor *cn = ds4_gpu_tensor_alloc((uint64_t)D * 4);
            float *hd = xmalloc((size_t)D * 4);
            char p2[4400];
            snprintf(p2, sizeof p2, "%s.hdiag", out_path);
            /* 出口头按盘上类型挑列平方和核(fp4x32 / q4_K 骨架各一支): 读错格式不报错, 只出一整列假度量 */
            const ds4_tensor *ho = e->weights.output;
            const int okcn = cn && (ho->type == DS4_TENSOR_Q4_K
                ? ds4_gpu_v41_head_colnorm_q4k_tensor(cn, e->model.map, e->model.size, ho->abs_offset, DS4_N_VOCAB, D)
                : ds4_gpu_v41_head_colnorm_tensor(cn, e->model.map, e->model.size, ho->abs_offset, DS4_N_VOCAB, D));
            if (okcn && ds4_gpu_synchronize() && ds4_gpu_tensor_read(cn, 0, hd, (uint64_t)D * 4)) {
                FILE *f2 = fopen(p2, "wb");
                if (f2) {
                    if (fwrite(hd, 4, D, f2) == D) fprintf(stderr, "[dcap] 出口度量落盘 %s (%u 列平方和)\n", p2, D);
                    fclose(f2);
                }
            } else fprintf(stderr, "ds4: ★出口度量没取到, 解算只能走纯 L2(那条 09-16 判过负)★\n");
            if (cn) ds4_gpu_tensor_free(cn);
            free(hd);
        }
        rc = 0;
    } while (0);
    if (fo) fclose(fo);
    if (ff) fclose(ff);
    if (am) ds4_gpu_tensor_free(am);
    free(xrow); free(yrow); free(mtok); free(dtok); free(mh); free(lm); free(ld); free(pm); free(qm);
    v41_draft_free(&dr);
    v41_state_free(&st);
    return rc;
}
#else
int ds4_engine_v41_dspark_capture(ds4_engine *e, const int *ids, int n_ids, const char *out_path, int n_prompt) {
    (void)e; (void)ids; (void)n_ids; (void)out_path; (void)n_prompt;
    fprintf(stderr, "ds4: V4.1 只有 GPU 路\n"); return 1;
}
#endif
