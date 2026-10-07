/* core_v41_draft_amp.c — 草稿器的件: 挂载(2026-10-07 从 core_v41_draft.c 拆出, 那片顶到 500 行)。
 *
 * 两种东西: ①09-16 的出口对齐边车(单个 DSPA 文件, xn += xn·BᵀA); ②草稿器蒸馏(src/core/core_draft_kd*.c)的件目录 ——
 * exit.dspa(同①) + tower_Tn.bin(塔件, 装进草稿态 st.ampA[塔号], v41_moe 草稿分支应用 y += xn·B·A) + markov_embd/head.bin(偏置表,
 * 顶替 GGUF 里原件的两张, v41_draft_block 的偏置路改读设备表; confidence 头仍吃原件的行) + base.fnv(训练时挂的 ② 指纹)。
 * ★只改草稿器★: 主模型一个字节不碰, 贪心输出仍逐字节等于纯解码, 采样下边缘仍恰是 p, 动的只有接受率。
 * 出错会怎样: 件是对着 ①+② 训的, 指纹对不上就停车 —— 挂到别的底座上不报错, 只是接受率掉。 */
#include "core_internal.h"
#include "src/common/ds4_gr_fnv.h"
#include <sys/stat.h>
#ifndef DS4_NO_GPU

/* 挂草稿器对齐边车(mtp.md M6; gguf-tools/amp/dspark_align 的产物)。
 * 盘上: 16 B 头 {"DSPA", D, K} + A[K][D] f32 + B[K][D] f32。
 * ★只改草稿器★: 主模型一个字节不碰, 所以它**不可能**动五指标, 只动接受率 —— 这是它能独立发车的前提。
 * 出错会怎样: 文件在但 D 对不上 = 不是这个模型解的, 直接停车(挂上去只会让草稿变垃圾, 而且不报错)。 */
bool v41_draft_amp_load(ds4_v41_draft *dr, const char *path) {
    struct { char magic[4]; uint32_t d, k, rsv; } h;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ds4: [v41] --draft-amp 打不开 %s\n", path); return false; }
    bool ok = fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, "DSPA", 4);
    if (ok && h.d != DS4_N_EMBD) {
        fprintf(stderr, "ds4: [v41] --draft-amp 维度 %u ≠ %u, 不是这个模型解的\n", h.d, (unsigned)DS4_N_EMBD);
        ok = false;
    }
    if (!ok) { fclose(f); fprintf(stderr, "ds4: [v41] --draft-amp %s 不是对齐边车\n", path); return false; }
    const uint64_t nb = (uint64_t)h.k * h.d * 4;
    float *buf = xmalloc((size_t)nb);
    bool alloc_ok = true;
    dr->ampA = v41_alloc(nb, &alloc_ok);
    dr->ampB = v41_alloc(nb, &alloc_ok);
    dr->ampT = v41_alloc((uint64_t)(dr->st.cap_tok ? dr->st.cap_tok : 8u) * h.k * 4, &alloc_ok);
    ok = alloc_ok && fread(buf, 1, (size_t)nb, f) == nb;
    if (ok && g_ds4_v41_draft_amp_scale != 1.0f)
        for (uint64_t i = 0; i < nb / 4; i++) buf[i] *= g_ds4_v41_draft_amp_scale;
    if (ok) ok = ds4_gpu_tensor_write(dr->ampA, 0, buf, nb) != 0;
    if (ok) ok = fread(buf, 1, (size_t)nb, f) == nb && ds4_gpu_tensor_write(dr->ampB, 0, buf, nb);
    free(buf); fclose(f);
    if (!ok) { fprintf(stderr, "ds4: [v41] --draft-amp 读取失败\n"); return false; }
    dr->ampK = h.k;
    fprintf(stderr, "ds4: [v41] 草稿器对齐边车已挂: K=%u D=%u β=%.3f (%s)\n", h.k, h.d, (double)g_ds4_v41_draft_amp_scale, path);
    return true;
}

/* ★草稿器蒸馏的件目录(2026-10-07, src/core/core_draft_kd*.c 的产物)★: <dir>/exit.dspa(出口件, 上面那个格式) + <dir>/tower_Tn.bin
 * (塔件 {E,K,1} + A[K][E] + B[K][E], 与 ③ 的 amp_Lnn.bin 同格式, 装进草稿态 st.ampA[塔号], v41_moe 的草稿分支应用 y += xn·B·A)
 * + base.fnv(训练时挂的 ② 目录指纹)。★件是对着 ①+② 这个底座训的★: 指纹对不上就停车 —— 挂到别的底座上不报错, 只是接受率掉。
 * 只改草稿器: 主模型一个字节不碰, 贪心输出仍逐字节等于纯解码, 采样下边缘仍恰是 p。 */
static bool v41_draft_amp_dir_load(ds4_v41_draft *dr, const char *dir) {
    const uint32_t E = DS4_N_EMBD, NT = g_ds4_v41.mtp_towers;
    char p[4200]; struct stat sb;
    snprintf(p, sizeof p, "%s/base.fnv", dir);
    FILE *f = fopen(p, "r");
    unsigned long long want = 0; unsigned wn = 0, hn = 0;
    if (!f || fscanf(f, "%llx %u", &want, &wn) != 2) { if (f) fclose(f); fprintf(stderr, "ds4: [v41] 草稿器件目录 %s 没有 base.fnv, 核对不了底座, 停车\n", dir); return false; }
    fclose(f);
    const uint64_t have = (g_ds4_v41_amp_dir && g_ds4_v41_amp_dir[0]) ? ds4_gr_dir_fnv(g_ds4_v41_amp_dir, DS4_N_LAYER, &hn) : DS4_GR_FNV_SEED;
    if (have != (uint64_t)want || hn != wn) {
        fprintf(stderr, "ds4: ★[v41] 草稿器件 %s 是对着另一份 ② 训的(指纹 %016llx/%u, 现挂 %016llx/%u), 停车★\n", dir, want, wn, (unsigned long long)have, hn);
        return false;
    }
    snprintf(p, sizeof p, "%s/exit.dspa", dir);
    if (stat(p, &sb) == 0 && !v41_draft_amp_load(dr, p)) return false;
    uint32_t kmax = 0, nt = 0;
    for (uint32_t T = 0; T < NT; T++) {
        snprintf(p, sizeof p, "%s/tower_T%u.bin", dir, T);
        f = fopen(p, "rb");
        if (!f) continue;
        int32_t hd[3] = { 0, 0, 0 };
        if (fread(hd, 4, 3, f) != 3 || hd[0] != (int32_t)E || hd[1] <= 0 || hd[2] != 1) { fclose(f); fprintf(stderr, "ds4: [v41] 塔件 %s 头不对(要 E %u, f32)\n", p, E); return false; }
        const uint32_t K = (uint32_t)hd[1]; const uint64_t nel = (uint64_t)K * E;
        float *buf = xmalloc((size_t)nel * 4);
        bool ok = true;
        dr->st.ampA[T] = v41_alloc(nel * 4, &ok); dr->st.ampB[T] = v41_alloc(nel * 4, &ok);
        ok = ok && fread(buf, 1, (size_t)nel * 4, f) == nel * 4;
        if (ok && g_ds4_v41_draft_amp_scale != 1.0f) for (uint64_t i = 0; i < nel; i++) buf[i] *= g_ds4_v41_draft_amp_scale;
        ok = ok && ds4_gpu_tensor_write(dr->st.ampA[T], 0, buf, nel * 4) && fread(buf, 1, (size_t)nel * 4, f) == nel * 4 && ds4_gpu_tensor_write(dr->st.ampB[T], 0, buf, nel * 4);
        free(buf); fclose(f);
        if (!ok) { fprintf(stderr, "ds4: [v41] 塔件 %s 读取失败\n", p); return false; }
        dr->st.ampK[T] = K; if (K > kmax) kmax = K; nt++;
    }
    /* 偏置表(markov_embd.bin {Vm,R,1} / markov_head.bin {V,R,1} + f32, 训练器 dk_save 落的): 两张都在才顶替原件, 只有一张 = 件目录坏了, 停车 */
    int nm = 0;
    for (int w = 0; w < 2; w++) {
        snprintf(p, sizeof p, "%s/%s", dir, w ? "markov_head.bin" : "markov_embd.bin");
        f = fopen(p, "rb");
        if (!f) continue;
        int32_t hd[3] = { 0, 0, 0 };
        const uint32_t R = g_ds4_v41.mtp_markov_rank;
        if (fread(hd, 4, 3, f) != 3 || hd[1] != (int32_t)R || hd[2] != 1 || hd[0] <= 0 || (w && hd[0] != (int32_t)DS4_N_VOCAB)) { fclose(f); fprintf(stderr, "ds4: [v41] 偏置表 %s 头不对\n", p); return false; }
        const uint64_t nel = (uint64_t)hd[0] * R; float *buf = xmalloc((size_t)nel * 4); bool ok = true;
        ds4_gpu_tensor **dst = w ? &dr->mkH_dev : &dr->mkE_dev;
        *dst = v41_alloc(nel * 4, &ok);
        ok = ok && fread(buf, 1, (size_t)nel * 4, f) == nel * 4 && ds4_gpu_tensor_write(*dst, 0, buf, nel * 4);
        free(buf); fclose(f);
        if (!ok) { fprintf(stderr, "ds4: [v41] 偏置表 %s 读取失败\n", p); return false; }
        if (!w) dr->mk_rows = (uint32_t)hd[0];
        nm++;
    }
    if (nm == 1) { fprintf(stderr, "ds4: [v41] 件目录 %s 的偏置表只有一张, 停车\n", dir); return false; }
    if (!nt && !dr->ampK && !nm) { fprintf(stderr, "ds4: [v41] 草稿器件目录 %s 里没有 tower_Tn.bin / exit.dspa / markov_*.bin\n", dir); return false; }
    if (kmax) { bool ok = true; dr->st.ampT = v41_alloc((uint64_t)dr->st.cap_tok * kmax * 4, &ok); if (!ok) return false; }
    fprintf(stderr, "ds4: [v41] 草稿器件已挂: %u 个塔件(K≤%u) + 出口件 K=%u%s, β=%.3f (%s)\n", nt, kmax, dr->ampK, nm ? " + 偏置表" : "", (double)g_ds4_v41_draft_amp_scale, dir);
    return true;
}

/* --draft-amp 的分发: 文件 = 09-16 的出口对齐边车; 目录 = 草稿器蒸馏的件(出口件 + 塔件 + 偏置表) */
bool v41_draft_amp_mount(ds4_v41_draft *dr, const char *path) {
    struct stat sb;
    const bool isdir = stat(path, &sb) == 0 && S_ISDIR(sb.st_mode);
    return isdir ? v41_draft_amp_dir_load(dr, path) : v41_draft_amp_load(dr, path);
}

#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_draft_amp_nonempty_tu;
