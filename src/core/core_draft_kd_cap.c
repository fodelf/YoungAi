/* core_draft_kd_cap.c — 草稿器蒸馏: 料清单、底座取料与缓存(2026-10-07)。总述见 core_draft_kd.h。
 *
 * 料的形状: 清单每行 `<ids 文件> <train|eval> <n_prompt>`(# 起注释); ids 一行一个 token id = 提示 + 底座采样的续写
 * (d1_kv_ring_gate.sh accjury 段 / z_nightly_spark.sh dkgen 段的产物), n_prompt = 提示长(块只在续写区取, 见 b_lo/b_hi)。
 * 取料 = 整份文本按部署分块(DS4_V41_CHUNK)跑一遍底座, 每块 **跑满 40 层**(ced_skip=0): 三塔要的 main_hidden 在 L37/38/39 的注意力输入,
 * CED 跳过解码器段的块根本不产它; 生成路的每一步本来就是跑满的, 所以续写区的 main_x 与部署逐位同一个函数。
 * 每份文本落: main_x[n][E](= main_norm(main_proj(main_hidden)), 窗口 KV 的唯一来源)、教师 top-K + 余量(底座在每个位置的分布,
 * 温 1 = 模型卡配方)、留出文本另加 [n−n_prompt][V] f32 全词表教师 logits(陪审团的 Σmin 要全词表, top-K 代替会把尾巴的质量算丢)。
 * 缓存按 (ids, K, 留出与否) 的指纹命名, 料不变第二趟直接读; 底座换了(另一份 ② 或量化)要换缓存目录 —— 这里不核对, 落 base.fnv 的是件。
 * ★缓存不整份进内存★(10-07 实撞: 12 份文本 8 GB 主机内存, 第 11 份取料时 MemAvailable 掉到 2383 MB 被看门狗杀): 训练/评估每批只 pread 要的几行(历史 main_x
 * ≤ 256 行、教师表 nb+B 行、留出 logits 几十行), 页缓存自己管。
 * 出错会怎样: main_x 取错层/取成层输出不报错, 只表现为第 0 轮陪审团对不上 accjury(core_draft_kd.c 的接线门就是为它设的)。 */
#include "core_draft_kd.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#ifndef DS4_NO_GPU

#define DK_CACHE_VER 1u
typedef struct { char magic[4]; uint32_t ver, n, n_prompt, E, K, V, has_tlog, pad; } dk_cache_hdr;

static uint64_t dk_key(const dk_text *t, uint32_t K) {
    uint64_t h = 1469598103934665603ull;
    #define DK_H(x) do { h ^= (uint64_t)(uint32_t)(x); h *= 1099511628211ull; } while (0)
    DK_H(DK_CACHE_VER); DK_H(K); DK_H(t->eval); DK_H(t->n); DK_H(t->n_prompt);
    for (uint32_t i = 0; i < t->n; i++) DK_H(t->ids[i]);
    #undef DK_H
    return h;
}

static bool dk_cache_path(const dk_cfg *c, const dk_text *t, char *p, size_t np) {
    mkdir(c->cache, 0755);
    snprintf(p, np, "%s/%016llx.dkc", c->cache, (unsigned long long)dk_key(t, c->topk));
    return true;
}

/* 取料时的主机暂存(落盘即放; 训练/评估按需 pread 缓存文件) */
typedef struct { float *mx, *tp, *trest, *tlog; int32_t *tid; } dk_host;
static void dk_host_alloc(dk_host *h, const dk_text *t, uint32_t E, uint32_t K, uint32_t V) {
    h->mx = xmalloc((size_t)t->n * E * 4); h->tid = xmalloc((size_t)t->n * K * 4); h->tp = xmalloc((size_t)t->n * K * 4);
    h->trest = xmalloc((size_t)t->n * 4);
    h->tlog = t->eval ? xmalloc((size_t)(t->n - t->n_prompt) * V * 4) : NULL;
}
static void dk_host_free(dk_host *h) { free(h->mx); free(h->tid); free(h->tp); free(h->trest); free(h->tlog); memset(h, 0, sizeof *h); }

bool dk_text_pread(const dk_text *t, uint64_t off, void *buf, uint64_t bytes) {
    uint8_t *b = buf;
    while (bytes) {
        const ssize_t r = pread(t->fd, b, (size_t)bytes, (off_t)off);
        if (r <= 0) { fprintf(stderr, "ds4: [dk] 缓存读失败 %s @%llu\n", t->path, (unsigned long long)off); return false; }
        b += r; off += (uint64_t)r; bytes -= (uint64_t)r;
    }
    return true;
}

/* 打开缓存并核对头, 记各段偏移(不读正文) */
static bool dk_cache_open(const dk_cfg *c, dk_text *t, int quiet) {
    char p[1200]; dk_cache_path(c, t, p, sizeof p);
    const int fd = open(p, O_RDONLY);
    if (fd < 0) return false;
    const uint32_t E = DS4_N_EMBD, V = DS4_N_VOCAB, K = c->topk;
    dk_cache_hdr h;
    bool ok = pread(fd, &h, sizeof h, 0) == (ssize_t)sizeof h && !memcmp(h.magic, "DKC1", 4) && h.ver == DK_CACHE_VER && h.n == t->n &&
              h.n_prompt == t->n_prompt && h.E == E && h.K == K && h.V == V && h.has_tlog == (t->eval ? 1u : 0u);
    if (!ok) { close(fd); return false; }
    t->fd = fd;
    t->o_mx = sizeof h; t->o_tid = t->o_mx + (uint64_t)t->n * E * 4; t->o_tp = t->o_tid + (uint64_t)t->n * K * 4;
    t->o_trest = t->o_tp + (uint64_t)t->n * K * 4; t->o_tlog = t->o_trest + (uint64_t)t->n * 4;
    if (!quiet) fprintf(stderr, "ds4: [dk] 缓存命中 %s ← %s(%u token, 提示 %u%s)\n", p, t->path, t->n, t->n_prompt, t->eval ? ", 留出含全词表教师" : "");
    return true;
}

static bool dk_cache_write(const dk_cfg *c, const dk_text *t, const dk_host *h) {
    char p[1200]; dk_cache_path(c, t, p, sizeof p);
    FILE *f = fopen(p, "wb");
    if (!f) { fprintf(stderr, "ds4: [dk] 缓存写不了 %s\n", p); return false; }
    const uint32_t E = DS4_N_EMBD, V = DS4_N_VOCAB, K = c->topk;
    dk_cache_hdr hd = { { 'D', 'K', 'C', '1' }, DK_CACHE_VER, t->n, t->n_prompt, E, K, V, t->eval ? 1u : 0u, 0u };
    bool ok = fwrite(&hd, sizeof hd, 1, f) == 1 && fwrite(h->mx, 4, (size_t)t->n * E, f) == (size_t)t->n * E &&
              fwrite(h->tid, 4, (size_t)t->n * K, f) == (size_t)t->n * K && fwrite(h->tp, 4, (size_t)t->n * K, f) == (size_t)t->n * K &&
              fwrite(h->trest, 4, t->n, f) == t->n;
    if (ok && h->tlog) ok = fwrite(h->tlog, 4, (size_t)(t->n - t->n_prompt) * V, f) == (size_t)(t->n - t->n_prompt) * V;
    fclose(f);
    if (ok) fprintf(stderr, "ds4: [dk] 缓存落 %s\n", p);
    else fprintf(stderr, "ds4: [dk] 缓存写坏 %s(盘满?)\n", p);
    return ok;
}

/* 底座跑一遍: main_x + 教师表(+ 留出的全词表 logits)。状态的 mainh 环按整份文本开(位置 p 在第 p 格), 每块的行就是连续的。 */
static bool dk_capture(ds4_engine *e, const dk_cfg *c, dk_text *t) {
    const ds4_model *m = &e->model;
    const uint32_t n = t->n, E = DS4_N_EMBD, V = DS4_N_VOCAB, K = c->topk, NT = g_ds4_v41.n_mtp_target;
    const uint32_t cap = n < DS4_V41_CHUNK ? n : DS4_V41_CHUNK;
    if (!NT || !e->weights.mtp.main_proj || !e->weights.mtp.main_norm) { fprintf(stderr, "ds4: [dk] 这份 GGUF 没带三塔, 取不了料\n"); return false; }
    ds4_v41_state st;
    if (!v41_state_alloc(&st, cap, n + 8u, cap)) return false;   /* logits 每个位置都要 */
    bool ok = true;
    if (st.mainh) ds4_gpu_tensor_free(st.mainh);
    st.mainh_cap = n; st.mainh = v41_alloc((uint64_t)n * NT * E * 4, &ok); st.mainh_end = -1; st.mainh_n = 0;
    ds4_gpu_tensor *raw = v41_alloc((uint64_t)cap * E * 4, &ok), *mxd = v41_alloc((uint64_t)cap * E * 4, &ok);
    ds4_gpu_tensor *tid = v41_alloc((uint64_t)cap * K * 4, &ok), *tp = v41_alloc((uint64_t)cap * K * 4, &ok), *tr = v41_alloc((uint64_t)cap * 4, &ok);
    dk_host hb; dk_host_alloc(&hb, t, E, K, V);
    st.ced_skip = 0; st.head_last_only = 0;   /* 每块跑满 40 层: 三塔的 main_hidden 在解码器段, CED 块不产它 */
    const double t0 = now_sec();
    for (uint32_t c0 = 0; ok && c0 < n; ) {
        const uint32_t nc = n - c0 < cap ? n - c0 : cap;
        ok = v41_forward(e, &st, t->ids + c0, nc);
        ds4_gpu_tensor *view = ok ? ds4_gpu_tensor_view(st.mainh, (uint64_t)c0 * NT * E * 4, (uint64_t)nc * NT * E * 4) : NULL;
        ok = ok && view && v41_tproj(m, raw, e->weights.mtp.main_proj, (uint64_t)E * NT, E, view, nc, 1) &&
             ds4_gpu_v41_rms_norm_tensor(mxd, raw, m->map, m->size, e->weights.mtp.main_norm->abs_offset, E, nc, DS4_RMS_EPS) &&
             ds4_gpu_synchronize() && ds4_gpu_tensor_read(mxd, 0, hb.mx + (size_t)c0 * E, (uint64_t)nc * E * 4);
        if (view) ds4_gpu_tensor_free(view);
        if (ok && hb.tlog && c0 + nc > t->n_prompt) {   /* 留出: 先抄全词表 logits(top-K 核会改写这些行) */
            const uint32_t r0 = c0 > t->n_prompt ? c0 : t->n_prompt, mm = c0 + nc - r0;
            ok = ds4_gpu_tensor_read(st.logits, (uint64_t)(r0 - c0) * V * 4, hb.tlog + (size_t)(r0 - t->n_prompt) * V, (uint64_t)mm * V * 4);
        }
        ok = ok && ds4_gpu_bwd_topk_tensor(tid, tp, tr, st.logits, 0, nc, V, K) && ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(tid, 0, hb.tid + (size_t)c0 * K, (uint64_t)nc * K * 4) && ds4_gpu_tensor_read(tp, 0, hb.tp + (size_t)c0 * K, (uint64_t)nc * K * 4) &&
             ds4_gpu_tensor_read(tr, 0, hb.trest + c0, (uint64_t)nc * 4);
        c0 += nc;
        fprintf(stderr, "ds4: [dk] 取料 %s: %u/%u 位置 %.0f s\r", t->path, c0, n, now_sec() - t0);
    }
    fprintf(stderr, "\n");
    if (raw) ds4_gpu_tensor_free(raw);
    if (mxd) ds4_gpu_tensor_free(mxd);
    if (tid) ds4_gpu_tensor_free(tid);
    if (tp) ds4_gpu_tensor_free(tp);
    if (tr) ds4_gpu_tensor_free(tr);
    v41_state_free(&st);
    if (ok) ok = dk_cache_write(c, t, &hb);
    dk_host_free(&hb);   /* 落盘即放: 训练按需 pread */
    if (!ok) fprintf(stderr, "ds4: [dk] 取料失败 %s\n", t->path);
    return ok;
}

static bool dk_read_ids(dk_text *t) {
    FILE *f = fopen(t->path, "r");
    if (!f) { fprintf(stderr, "ds4: [dk] 读不了 ids %s\n", t->path); return false; }
    uint32_t cap = 4096, n = 0; int32_t *ids = xmalloc((size_t)cap * 4); int v;
    while (fscanf(f, "%d", &v) == 1) { if (n == cap) { cap *= 2; ids = realloc(ids, (size_t)cap * 4); } ids[n++] = v; }
    fclose(f);
    t->ids = ids; t->n = n;
    return n > 0;
}

bool dk_data_load(ds4_engine *e, const dk_cfg *c, dk_data *d) {
    memset(d, 0, sizeof *d);
    FILE *f = fopen(c->texts, "r");
    if (!f) { fprintf(stderr, "ds4: [dk] 读不了料清单 %s\n", c->texts); return false; }
    const uint32_t B = g_ds4_v41.mtp_block;
    char ln[2048]; uint32_t cap = 16;
    d->t = xmalloc_zeroed(cap, sizeof(dk_text));
    while (fgets(ln, sizeof ln, f)) {
        char path[1024], kind[32]; unsigned np = 0;
        if (ln[0] == '#' || sscanf(ln, "%1023s %31s %u", path, kind, &np) != 3) continue;
        if (d->nt == cap) { cap *= 2; d->t = realloc(d->t, (size_t)cap * sizeof(dk_text)); memset(d->t + d->nt, 0, (size_t)(cap - d->nt) * sizeof(dk_text)); }
        dk_text *t = &d->t[d->nt];
        snprintf(t->path, sizeof t->path, "%s", path);
        t->eval = !strcmp(kind, "eval"); t->n_prompt = np;
        if (!dk_read_ids(t)) { fclose(f); return false; }
        if (t->n_prompt < 1u || t->n_prompt + B + 1u > t->n) { fprintf(stderr, "ds4: [dk] %s: 提示 %u / 总长 %u 不成形(续写至少 %u 个 token)\n", path, t->n_prompt, t->n, B + 1u); fclose(f); return false; }
        /* 块 i 吃 ids[i], 窗口要 i−1 的 main_x(i ≥ 1), 第 B−1 位的教师是底座第 i+B−1 行 ⇒ i ≤ n−B。训练可带提示尾巴(prompt_tail), 留出只认续写区 */
        t->b_lo = t->eval ? t->n_prompt : (t->n_prompt > c->prompt_tail + 1u ? t->n_prompt - c->prompt_tail : 1u);
        t->b_hi = t->n - B + 1u;
        d->nt++;
    }
    fclose(f);
    if (!d->nt) { fprintf(stderr, "ds4: [dk] 料清单 %s 里没有文本\n", c->texts); return false; }
    for (uint32_t i = 0; i < d->nt; i++) {
        dk_text *t = &d->t[i];
        t->fd = -1;
        if (dk_cache_open(c, t, 0)) continue;
        if (!dk_capture(e, c, t) || !dk_cache_open(c, t, 1)) return false;
    }
    uint32_t ntr = 0, nev = 0;
    for (uint32_t i = 0; i < d->nt; i++) { const dk_text *t = &d->t[i]; if (t->eval) nev += t->b_hi - t->b_lo; else ntr += t->b_hi - t->b_lo; }
    fprintf(stderr, "ds4: [dk] 料: %u 份文本, 训练块 %u / 留出块 %u(块长 %u)\n", d->nt, ntr, nev, B);
    return true;
}

void dk_data_free(dk_data *d) {
    for (uint32_t i = 0; i < d->nt; i++) { dk_text *t = &d->t[i]; free(t->ids); if (t->fd >= 0) close(t->fd); t->fd = -1; }
    free(d->t); d->t = NULL; d->nt = 0;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_draft_kd_cap_nonempty_tu;
