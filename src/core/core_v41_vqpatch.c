/* core_v41_vqpatch.c — V 步索引补丁的装载(2026-11-02): ③ 目录里的 vqpatch_Lnn.bin 让这一层若干专家权重组换码字(core_ptrain_vstep.c 训出来的)。
 *
 * 为什么是补丁不是新 GGUF: 盘上底座一个字节不动(可逆、不加体积), 改的只是 GPU 读的那份 blob 的索引位(ds4_gpu_v41_vq_code_patch:
 *   设备副本直接改, 流式层记表每次拷完再盖), 专家核照旧从 blob 读 —— 不挂时与原样逐字节同。
 * 文件: 头 6×u32 {'VQP1', 层, 条数, IN, MID, OUT} + 条数 × {专家|矩阵<<16, 行, 列组, 新码|旧码<<16}。IN/MID/OUT 与本模型形状对不上 ⇒ 拒(那是别的模型的补丁)。
 * 卸: 装另一组或装空时先把上一组按旧码写回(记录就在手里), 再装新的; 回读 ≠0 一律停车。
 * 出错会怎样: 头不对 / 形状不对 / 这台机器的装载路改不了索引(整映射直读文件页) ⇒ 返回 <0, 插件挂不上就停 —— 不许悄悄按原索引跑, 那是另一个模型。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU

#define VQP_MAGIC 0x31505156u   /* 'VQP1' */
typedef struct { uint32_t ew, row, j, code; } vqp_rec;
static struct { vqp_rec *v; uint32_t n; } g_vqp[DS4_MAX_LAYER];   /* 此刻挂着的记录(还原用) */
static uint32_t g_vqp_layers;

static const ds4_tensor *vqp_blob(uint32_t il) { char nm[64]; snprintf(nm, sizeof nm, "blk.%u.ffn_exps_vq.blob", il); return g_ds4_v41_model ? model_find_tensor(g_ds4_v41_model, nm) : NULL; }

static bool vqp_apply(uint32_t il, const vqp_rec *v, uint32_t n, bool restore) {
    const ds4_model *m = g_ds4_v41_model; const ds4_tensor *t = vqp_blob(il);
    if (!t || !n) return t != NULL;
    vqp_rec *w = xmalloc((size_t)n * sizeof(vqp_rec));
    for (uint32_t i = 0; i < n; i++) { w[i] = v[i]; if (restore) w[i].code = (v[i].code >> 16) | (v[i].code << 16); }   /* 还原 = 新旧对调 */
    uint32_t mis = 0;
    const bool ok = ds4_gpu_v41_vq_code_patch(m->map, m->size, t->abs_offset, t->bytes, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD, (const uint32_t *)w, n, &mis) && mis == 0;
    if (!ok) fprintf(stderr, "ds4: ★[V步·补丁] L%02u %s失败(回读不同 %u/%u 条)★\n", il, restore ? "还原" : "装载", mis, n);
    free(w);
    return ok;
}

const void *v41_vqpatch_records(uint32_t il, uint32_t *n) { *n = il < DS4_MAX_LAYER ? g_vqp[il].n : 0; return *n ? g_vqp[il].v : NULL; }

/* dir = NULL/"" = 只还原。返回挂上的层数, <0 = 失败 */
int v41_vqpatch_load(const char *dir) {
    bool ok = true;
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) if (g_vqp[il].n) { ok = vqp_apply(il, g_vqp[il].v, g_vqp[il].n, true) && ok; free(g_vqp[il].v); g_vqp[il].v = NULL; g_vqp[il].n = 0; }
    if (g_vqp_layers) fprintf(stderr, "ds4: [V步·补丁] 还原 %u 层%s\n", g_vqp_layers, ok ? "" : " ★有层还原失败★");
    g_vqp_layers = 0;
    if (!ok) return -1;
    int n = 0; uint64_t recs = 0;
    for (uint32_t il = 0; dir && dir[0] && il < DS4_N_LAYER; il++) {
        char p[4200]; snprintf(p, sizeof p, "%s/vqpatch_L%02u.bin", dir, il);
        FILE *f = fopen(p, "rb");
        if (!f) continue;
        uint32_t hd[6] = { 0 };
        bool good = fread(hd, 4, 6, f) == 6 && hd[0] == VQP_MAGIC && hd[1] == il && hd[3] == DS4_N_EMBD && hd[4] == DS4_N_FF_EXP && hd[5] == DS4_N_EMBD;
        vqp_rec *v = good && hd[2] ? xmalloc((size_t)hd[2] * sizeof(vqp_rec)) : NULL;
        if (good && hd[2]) good = fread(v, sizeof(vqp_rec), hd[2], f) == hd[2];
        fclose(f);
        if (!good) { fprintf(stderr, "ds4: ★V步补丁 %s 头或长度不对(要 'VQP1' / 层 %u / 形状 %u,%u,%u)★\n", p, il, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD); free(v); return -1; }
        if (hd[2] && !vqp_apply(il, v, hd[2], false)) { free(v); return -1; }
        g_vqp[il].v = v; g_vqp[il].n = hd[2]; recs += hd[2]; n++;
    }
    g_vqp_layers = (uint32_t)n;
    if (n) fprintf(stderr, "ds4: [反修·V步补丁] %d 层共 %llu 组换了码字(盘上 %.2f MB) ← %s\n", n, (unsigned long long)recs, (double)recs * 16 / 1e6, dir);
    return n;
}
#endif /* !DS4_NO_GPU */
