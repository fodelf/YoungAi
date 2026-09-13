/* core_v41_engram.c — DeepSeek V4.1 engram 层(2026-09-12 战役 P2b), 逐式对照官方 engram.py NgramHashState.forward
 * 与 model.py Engram.forward。
 *
 * 链路: 压缩 id(token_map) → 每位置 4-gram 滚动 XOR 哈希(乘子/素数/偏移来自 GGUF 常量张量, 转换器从 tokenizer
 * 算好的) → 24 行(3 种 n-gram × 8 头) → 从 203 GB 表(原 HF 分片, 在盘)并行 pread 原始行字节(256 B fp8 + 8 B ue8m0)
 * → 上 GPU dequant 成 bf16 格点 → wkv(f16 [25600][6144]) → key(4 路)|value → 门(CUDA 核) → hc 就地更新。
 * 取行(P4): 两个 engram 层的行在前向一开始就由 48 个线程一次性 pread(一线程一行, NVMe 吃并发), 与前面几层的 GPU 算重叠;
 * 到 engram 层只收结果。第一版 mmap 逐行页错误串行: 512 token 两层吃 16.6 s/38 s; 第二版每层各自 16 线程: 解码 16 ms/层。
 * 回看的 3 个 token 可能在上一块 —— 取自 st->hist(整段 token 历史), 按绝对位置索引。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU

#define V41_EGATHER_THREADS 48u

static const void *v41_tensor_host(const ds4_model *m, const ds4_tensor *t) { return tensor_data(m, t); }

static bool v41_engram_open_shard(ds4_v41_state *st, uint32_t ei) {
    if (st->eshard[ei].fd >= 0) return true;
    const char *path = g_ds4_v41.engram_table_path[ei];
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "ds4: engram 表打不开 %s\n", path); return false; }
    struct stat sb; if (fstat(fd, &sb) != 0) { close(fd); return false; }
    st->eshard[ei].fd = fd; st->eshard[ei].size = (uint64_t)sb.st_size;
    return true;
}

/* 官方 NgramHashState.forward(无 image mask): tokens[shift] = compressed[p-shift](p-shift<0 → pad);
 * products[k] = tokens[k]·mult[layer][k]; rolling = products[0]; for i=1..G-1: rolling ^= products[i];
 * hash_i = rolling % primes[layer][i-1][head] + offsets[layer][(i-1)·heads+head]。p 是绝对位置。 */
static void v41_engram_hash(const ds4_engine *e, const ds4_v41_state *st, uint32_t ei, uint32_t p, int64_t *rows /*[cols]*/) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    const ds4_weights *W = &e->weights;
    const int32_t *tmap = (const int32_t *)v41_tensor_host(&e->model, W->engram_token_map);
    const int64_t *mult = (const int64_t *)v41_tensor_host(&e->model, W->engram_multipliers);
    const int64_t *prim = (const int64_t *)v41_tensor_host(&e->model, W->engram_primes);
    const int64_t *offs = (const int64_t *)v41_tensor_host(&e->model, W->engram_offsets);
    const uint32_t G = v->engram_max_ngram, H = v->engram_heads;
    int64_t prod[8];
    for (uint32_t k = 0; k < G; k++) {
        int64_t cid;
        if ((int64_t)p - (int64_t)k < 0) cid = (int64_t)v->engram_pad;
        else { int32_t tok = st->hist[p - k]; cid = (tok >= 0 && (uint32_t)tok < DS4_N_VOCAB) ? tmap[tok] : (int64_t)v->engram_pad; }
        prod[k] = (int64_t)((uint64_t)cid * (uint64_t)mult[(uint64_t)ei * G + k]);   /* 官方 int64 乘(乘子有界不溢出) */
    }
    int64_t rolling = prod[0];
    for (uint32_t i = 1; i < G; i++) {
        rolling ^= prod[i];
        for (uint32_t h = 0; h < H; h++) {
            const int64_t pr = prim[((uint64_t)ei * (G - 1) + (i - 1)) * H + h];
            int64_t r = rolling % pr; if (r < 0) r += pr;          /* torch 对非负数 %, 这里保险 */
            rows[(i - 1) * H + h] = r + offs[(uint64_t)ei * (G - 1) * H + (i - 1) * H + h];
        }
    }
}

/* 后台取行任务: 工作单元 = (engram 层 ei, 位置 p, 列 c) 一行; 线程按单元区间切, 同 (ei,p) 的哈希只算一次 */
typedef struct {
    const ds4_engine *e; const ds4_v41_state *st; uint64_t u0, u1; int err;
} v41_eworker;
typedef struct {
    uint32_t n_eng, n, cols, HD, nsc;
    uint8_t *raw[DS4_V41_MAX_ENGRAM];      /* [n][cols][HD+nsc] */
    int64_t *rows[DS4_V41_MAX_ENGRAM];     /* [n][cols] */
    pthread_t th[V41_EGATHER_THREADS]; v41_eworker w[V41_EGATHER_THREADS]; uint32_t nth;
    int started, joined, err;
} v41_ejob;

static void *v41_eworker_run(void *arg) {
    v41_eworker *wk = (v41_eworker *)arg;
    const ds4_v41_state *st = wk->st; const v41_ejob *J = (const v41_ejob *)st->ejob; const ds4_v41_cfg *v = &g_ds4_v41;
    const uint32_t stride = J->HD + J->nsc;
    int64_t rows[64]; uint32_t last_ei = UINT32_MAX, last_p = UINT32_MAX;
    for (uint64_t u = wk->u0; u < wk->u1; u++) {
        const uint32_t ei = (uint32_t)(u / ((uint64_t)J->n * J->cols)), rem = (uint32_t)(u % ((uint64_t)J->n * J->cols));
        const uint32_t p = rem / J->cols, c = rem % J->cols;
        if (ei != last_ei || p != last_p) {
            v41_engram_hash(wk->e, st, ei, st->pos0 + p, rows);
            memcpy(J->rows[ei] + (size_t)p * J->cols, rows, (size_t)J->cols * sizeof(int64_t));
            last_ei = ei; last_p = p;
        }
        const int64_t r = rows[c];
        if (r < 0 || (uint64_t)r >= v->engram_rows[ei]) { wk->err = 1; return NULL; }
        uint8_t *dst = J->raw[ei] + ((size_t)p * J->cols + c) * stride;
        const int fd = st->eshard[ei].fd;
        if (pread(fd, dst, J->HD, (off_t)(v->engram_weight_off[ei] + (uint64_t)r * J->HD)) != (ssize_t)J->HD) { wk->err = 2; return NULL; }
        if (pread(fd, dst + J->HD, J->nsc, (off_t)(v->engram_scale_off[ei] + (uint64_t)r * J->nsc)) != (ssize_t)J->nsc) { wk->err = 2; return NULL; }
    }
    return NULL;
}

static void v41_ejob_free(ds4_v41_state *st) {
    v41_ejob *J = (v41_ejob *)st->ejob;
    if (!J) return;
    if (J->started && !J->joined) for (uint32_t t = 0; t < J->nth; t++) pthread_join(J->th[t], NULL);
    for (uint32_t i = 0; i < DS4_V41_MAX_ENGRAM; i++) { free(J->raw[i]); free(J->rows[i]); }
    free(J); st->ejob = NULL;
}

bool v41_engram_prefetch(ds4_engine *e, ds4_v41_state *st) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    v41_ejob_free(st);
    if (v->n_engram == 0) return true;
    v41_ejob *J = xmalloc(sizeof *J); memset(J, 0, sizeof *J);
    J->n_eng = v->n_engram; J->n = st->n; J->cols = (v->engram_max_ngram - 1) * v->engram_heads; J->HD = v->engram_head_dim; J->nsc = J->HD / 32u;
    st->ejob = J;
    for (uint32_t ei = 0; ei < J->n_eng; ei++) {
        if (!v41_engram_open_shard(st, ei)) return false;
        if (v->engram_weight_off[ei] + v->engram_rows[ei] * J->HD > st->eshard[ei].size ||
            v->engram_scale_off[ei] + v->engram_rows[ei] * J->nsc > st->eshard[ei].size) { fprintf(stderr, "ds4: engram 表偏移越界\n"); return false; }
        J->raw[ei] = xmalloc((size_t)J->n * J->cols * (J->HD + J->nsc));
        J->rows[ei] = xmalloc((size_t)J->n * J->cols * sizeof(int64_t));
    }
    const uint64_t U = (uint64_t)J->n_eng * J->n * J->cols;
    J->nth = U < V41_EGATHER_THREADS ? (uint32_t)U : V41_EGATHER_THREADS;
    for (uint32_t t = 0; t < J->nth; t++) {
        J->w[t] = (v41_eworker){ e, st, U * t / J->nth, U * (t + 1) / J->nth, 0 };
        if (pthread_create(&J->th[t], NULL, v41_eworker_run, &J->w[t]) != 0) {   /* 起不来就本线程同步做 */
            v41_eworker_run(&J->w[t]); J->th[t] = pthread_self();
        }
    }
    J->started = 1;
    return true;
}

static bool v41_ejob_wait(ds4_v41_state *st) {
    v41_ejob *J = (v41_ejob *)st->ejob;
    if (!J || !J->started) return false;
    if (!J->joined) {
        for (uint32_t t = 0; t < J->nth; t++) { if (!pthread_equal(J->th[t], pthread_self())) pthread_join(J->th[t], NULL); if (J->w[t].err) J->err = J->w[t].err; }
        J->joined = 1;
    }
    if (J->err) fprintf(stderr, "ds4: engram 取行失败(%s)\n", J->err == 1 ? "行号越界" : "pread 短读");
    return J->err == 0;
}

void v41_engram_close(ds4_v41_state *st) {
    v41_ejob_free(st);
    for (uint32_t i = 0; i < DS4_V41_MAX_ENGRAM; i++) {
        if (st->eshard[i].fd >= 0) close(st->eshard[i].fd);
        st->eshard[i].fd = -1;
    }
    if (st->eraw) { ds4_gpu_tensor_free(st->eraw); st->eraw = NULL; }
    if (st->erows) { ds4_gpu_tensor_free(st->erows); st->erows = NULL; }
    if (st->ekv) { ds4_gpu_tensor_free(st->ekv); st->ekv = NULL; }
}

bool v41_engram(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    const int16_t ei = v->engram_index_of[il];
    if (ei < 0) return true;
    if (!st->ejob && !v41_engram_prefetch(e, st)) return false;   /* 没预取(不该发生)就现取 */
    if (!v41_ejob_wait(st)) return false;
    const v41_ejob *J = (const v41_ejob *)st->ejob;
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const uint32_t n = st->n, E = DS4_N_EMBD, HC = DS4_N_HC, HD = J->HD, cols = J->cols, stride = J->HD + J->nsc;
    const uint64_t in_dim = (uint64_t)cols * HD, out_dim = (uint64_t)(HC + 1) * E;
    if (!st->erows) {
        st->eraw = ds4_gpu_tensor_alloc((uint64_t)st->cap_tok * cols * stride);
        st->erows = ds4_gpu_tensor_alloc((uint64_t)st->cap_tok * in_dim * 4);
        st->ekv = ds4_gpu_tensor_alloc((uint64_t)st->cap_tok * out_dim * 4);
    }
    if (!st->eraw || !st->erows || !st->ekv) return false;
    if (st->dump_prefix) {   /* 对拍夹具: 行号落 <prefix>.erows_Lnn.txt(每行一个位置, 24 个全局行号), 与 Python hash_ids 直接 diff */
        char p[4400]; snprintf(p, sizeof p, "%s.erows_L%02u.txt", st->dump_prefix, il);
        FILE *df = fopen(p, "w");
        if (df) { for (uint32_t q = 0; q < n; q++) for (uint32_t c = 0; c < cols; c++) fprintf(df, "%lld%c", (long long)J->rows[ei][(size_t)q * cols + c], c + 1 == cols ? '\n' : ' '); fclose(df); }
    }
    if (!ds4_gpu_tensor_write(st->eraw, 0, J->raw[ei], (uint64_t)n * cols * stride)) return false;
    if (!ds4_gpu_v41_engram_rows_tensor(st->erows, st->eraw, n * cols, HD)) return false;
    /* wkv(f16 权重, fp8 线性 → bf16 输出) → 门 → hc 就地 */
    if (!ds4_gpu_matmul_f16_tensor(st->ekv, m->map, m->size, l->engram_wkv->abs_offset, in_dim, out_dim, st->erows, n)) return false;
    if (!ds4_gpu_v41_round_bf16_tensor(st->ekv, (uint64_t)n * out_dim)) return false;
    if (!ds4_gpu_v41_engram_gate_tensor(st->hc, st->ekv, m->map, m->size, l->engram_q->abs_offset, l->engram_k->abs_offset, E, HC, n, DS4_RMS_EPS)) return false;
    if (st->dump_prefix) {   /* 对拍夹具: engram 后的 hc [n][HC][E] 落 <prefix>.hce_Lnn.bin(对 Python engram 模块输出) */
        char p[4400]; snprintf(p, sizeof p, "%s.hce_L%02u.bin", st->dump_prefix, il);
        float *buf = xmalloc((size_t)n * HC * E * 4); ds4_gpu_synchronize();
        if (ds4_gpu_tensor_read(st->hc, 0, buf, (uint64_t)n * HC * E * 4)) { FILE *f = fopen(p, "wb"); if (f) { fwrite(buf, 4, (size_t)n * HC * E, f); fclose(f); } }
        free(buf);
    }
    return true;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_engram_nonempty_tu;
