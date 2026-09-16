/* core_v41_dcap.c — DSpark 草稿器的对齐取料(mtp.md M6, 2026-09-16)。
 *
 * 要解决的问题: 草稿器是照**原始 FP 模型**训的, 而我们部署的是量化+反修的底座。实测首位接受率
 * **0.61**, 而底座对原模型的 Same top 是 **0.72** —— 0.72 × 0.85(草稿器自身命中) ≈ 0.61, 对得上。
 * 也就是说草稿器本身是好的, **是底座漂移把它的收益吃掉了**, 而 0.72 就是它的天花板。
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
 * 怎么取: 教师强制走一遍 token 序列, 一次一位(块 1) ——
 *   ①主模型前向位置 i → 出口隐态 Y[i] 与它的 argmax(它预测的是位置 i+1)
 *   ②草稿器从位置 i 起一轮 → 出口隐态 X[i] 与它的首位草稿(同样预测位置 i+1)
 * 两个 token 一比就是**首位接受率 p1 的直接测量**, 也就是 M6 的判决基线。
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
 * 头里写 D 与 n, 解算器按它读 —— 两边各写各的尺寸迟早对不上, 而错位不会报错。 */
typedef struct { char magic[4]; uint32_t d, n, rsv; } v41_dcap_hdr;

int ds4_engine_v41_dspark_capture(ds4_engine *e, const int *ids, int n_ids, const char *out_path) {
    if (!e || !ids || n_ids < 2 || !ds4_engine_is_v41(e)) return 1;
    if (!e->metal_ready) { fprintf(stderr, "ds4: V4.1 取料需要 GPU 后端\n"); return 1; }
    const uint32_t D = DS4_N_EMBD;
    uint32_t ctx = (uint32_t)n_ids + 1;
    if (ctx > DS4_V41_MAX_CTX_P2C) { fprintf(stderr, "ds4: 取料序列 %d 超上下文上限 %u\n", n_ids, DS4_V41_MAX_CTX_P2C); return 1; }
    ds4_v41_state st;
    if (!v41_state_alloc(&st, 1u, ctx)) return 1;   /* cap=1: 一位一块, 见文件头 */
    ds4_v41_draft dr;
    if (!v41_draft_alloc(e, &dr)) { fprintf(stderr, "ds4: 这份 GGUF 没带 DSpark 三塔, 取不了料\n"); v41_state_free(&st); return 1; }
    FILE *fo = fopen(out_path, "wb");
    ds4_gpu_tensor *am = ds4_gpu_tensor_alloc(16);
    float *xrow = xmalloc((size_t)D * 4), *yrow = xmalloc((size_t)D * 4);
    int32_t *mtok = xmalloc((size_t)n_ids * 4), *dtok = xmalloc((size_t)n_ids * 4);
    int rc = 1;
    uint32_t nw = 0, hit = 0;
    do {
        if (!fo || !am) { fprintf(stderr, "ds4: 取料落盘/暂存分配失败\n"); break; }
        v41_dcap_hdr h = { { 'D','C','A','P' }, D, 0, 0 };
        if (fwrite(&h, sizeof h, 1, fo) != 1) break;
        bool ok = true;
        for (int i = 0; i < n_ids - 1 && ok; i++) {
            const int32_t tok = (int32_t)ids[i];
            if (!v41_forward(e, &st, &tok, 1u)) { ok = false; break; }
            /* 主模型在位置 i 的出口隐态与 argmax —— 它预测的是位置 i+1 */
            int32_t want = 0;
            if (!ds4_gpu_v41_argmax_tensor(am, st.logits, 0u, DS4_N_VOCAB) || !ds4_gpu_synchronize() ||
                !ds4_gpu_tensor_read(am, 0, &want, 4) ||
                !ds4_gpu_tensor_read(st.xn, 0, yrow, (uint64_t)D * 4)) { ok = false; break; }
            /* 草稿器从位置 i 起出一块; 它的出口隐态第 0 行同样是"预测位置 i+1"的那一行 */
            if (!v41_draft_step(e, &st, &dr, tok, (uint32_t)i, 1u)) { ok = false; break; }
            if (!ds4_gpu_synchronize() || !ds4_gpu_tensor_read(dr.st.xn, 0, xrow, (uint64_t)D * 4)) { ok = false; break; }
            if (fwrite(xrow, 4, D, fo) != D || fwrite(yrow, 4, D, fo) != D) { ok = false; break; }
            mtok[nw] = want; dtok[nw] = dr.host_ids[1];
            if (want == dr.host_ids[1]) hit++;
            nw++;
            if ((nw % 64u) == 0u)
                fprintf(stderr, "[dcap] %u/%d 位置, 首位一致 %.3f\r", nw, n_ids - 1, (double)hit / (double)nw);
        }
        if (!ok) { fprintf(stderr, "\nds4: 取料在第 %u 位失败\n", nw); break; }
        if (fwrite(mtok, 4, nw, fo) != nw || fwrite(dtok, 4, nw, fo) != nw) break;
        h.n = nw;
        if (fseek(fo, 0, SEEK_SET) != 0 || fwrite(&h, sizeof h, 1, fo) != 1) break;
        fprintf(stderr, "\n[dcap] 落盘 %s: %u 位置 × 2 × %u f32\n", out_path, nw, D);
        /* ★这一行就是 M6 的判决基线★: 草稿器首位 ↔ 部署底座 argmax 的一致率。
         * 它与在线生成时 spec 账里的 p1 应当同量级 —— 差很多就说明取料与部署不同路, 先别解。 */
        fprintf(stderr, "[dcap] ★草稿器首位 ↔ 底座 argmax 一致率 = %.4f (n=%u)★\n", (double)hit / (double)(nw ? nw : 1), nw);
        rc = 0;
    } while (0);
    if (fo) fclose(fo);
    if (am) ds4_gpu_tensor_free(am);
    free(xrow); free(yrow); free(mtok); free(dtok);
    v41_draft_free(&dr);
    v41_state_free(&st);
    return rc;
}
#else
int ds4_engine_v41_dspark_capture(ds4_engine *e, const int *ids, int n_ids, const char *out_path) {
    (void)e; (void)ids; (void)n_ids; (void)out_path;
    fprintf(stderr, "ds4: V4.1 只有 GPU 路\n"); return 1;
}
#endif
