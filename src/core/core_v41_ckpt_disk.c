/* core_v41_ckpt_disk.c — V4.1 KV 检查点离载到磁盘(2026-11-15; core_v41_ckpt.h 的方案落地)。
 *
 * 文件格式: 80 B 头(魔数/版本/槽/位置/模型上下文/图片位/created/last_used/mainh 账/logits 字节/层数)
 * + 61 × 44 B 层账(ckv/idk/cpk/cps 字节 + ng_src/cpend/win_from)
 * + n×4 B 提示 token + n×8 B 图片指纹 + 逐层显存缓冲(win 固定 SWA 行, 压缩 KV/索引 K/余行按层账, mainh, logits 一行)。
 *
 * 为什么用主机回读: ds4_gpu_tensor_read/write 是两后端唯一的同步进出; 这里按 8 MiB 块流式过一遍，
 * 不在主机里装整份 0.4~0.9 GB 检查点 —— 内存余量只有 5 GB 级时，整份 staging 本身就是另一条 OOM。
 *
 * 磁盘预算与 LRU: 文件头记 last_used; 开时扫目录压到预算内，写新文件前也先删最久没用的，
 * 直到"已用 + 新文件" ≤ 预算; 单文件大于预算 ⇒ 不存(返回失败，GPU 不动)。
 * 为什么按对话而不是按文件: 检查点文件是 tok 序列的哈希命名，回旧对话时用"提示前 n 位哈希相同"找候选。
 * 不做什么: 本文件不碰 K 的显存释放 —— 离载由 core_v41_kvreuse.c 调 store 后自己 free。
 * 出错会怎样: 目录打不开/文件读写失败 ⇒ 返回失败; 调用方走冷回收，不留下半份文件索引。 */
#ifndef DS4_NO_GPU

#include "core_internal.h"
#include "core_v41_ckpt.h"

#include <dirent.h>

#define V41CKPT_MAGIC   "DSV41CK"
#define V41CKPT_VERSION 1u
#define V41CKPT_HDR     80u
#define V41CKPT_META_PER 44u
#define V41CKPT_META    (V41CKPT_META_PER * DS4_MAX_LAYER)
#define V41CKPT_NAME_LEN 28u
#define V41CKPT_STAGE   (8ull * 1024ull * 1024ull)
#define V41CKPT_DEFAULT_MB 4096ull

typedef struct {
    char *path;
    char hash[17];
    uint32_t slot;
    uint32_t n;
    uint32_t flags;
    uint64_t bytes;
    uint64_t last_used;
} v41ckpt_disk_entry;

static int g_on;
static char *g_dir;
static uint64_t g_budget;
static v41ckpt_disk_entry *g_entry;
static int g_len, g_cap;
static uint64_t g_used;
static uint8_t *g_stage;

static void le32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static uint32_t get32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static void le64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static uint64_t get64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

/* 文件名哈希: 16 hex 的 FNV-1a(非密码学，只做缓存键; 回读时还要逐字节比 tok/sig 才认) */
static uint64_t v41ck_hash(const void *ptr, size_t n) {
    const uint8_t *p = ptr;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
static void hash_hex(uint64_t h, char out[17]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 16; i++) { out[i] = hex[(h >> (4 * (15 - i))) & 15u]; }
    out[16] = '\0';
}
static char *path_join(const char *dir, const char *name) {
    const size_t d = strlen(dir), n = strlen(name);
    char *p = xmalloc(d + 1 + n + 1);
    memcpy(p, dir, d);
    size_t k = d;
    if (!d || dir[d - 1] != '/') p[k++] = '/';
    memcpy(p + k, name, n);
    p[k + n] = '\0';
    return p;
}

static bool entry_push(char *path, const char *hash, uint32_t slot, uint32_t n, uint32_t flags, uint64_t bytes, uint64_t last_used) {
    if (g_len >= g_cap) {
        g_cap = g_cap ? g_cap * 2 : 8;
        g_entry = xrealloc(g_entry, (size_t)g_cap * sizeof *g_entry);   /* core xrealloc? */
    }
    v41ckpt_disk_entry *e = &g_entry[g_len++];
    e->path = path; memcpy(e->hash, hash, 17); e->slot = slot; e->n = n; e->flags = flags;
    e->bytes = bytes; e->last_used = last_used;
    g_used += bytes;
    return true;
}

static void entry_remove(int i, bool unlink_file) {
    if (i < 0 || i >= g_len) return;
    v41ckpt_disk_entry *e = &g_entry[i];
    if (unlink_file) unlink(e->path);
    g_used -= e->bytes;
    free(e->path);
    if (i != g_len - 1) memmove(e, &g_entry[i + 1], (size_t)(g_len - i - 1) * sizeof *g_entry);
    g_len--;
}

/* 预算不足: 删最久没用的文件，直到 已用+extra ≤ 预算; 只剩单文件也装不下 ⇒ 失败 */
static bool trim_lru(uint64_t extra) {
    while (g_used + extra > g_budget && g_len > 0) {
        int victim = 0;
        for (int i = 1; i < g_len; i++) {
            const uint64_t a = g_entry[i].last_used, b = g_entry[victim].last_used;
            if (a < b || (a == b && g_entry[i].n < g_entry[victim].n)) victim = i;   /* LRU 同秒按短的先删(少丢长复用) */
        }
        entry_remove(victim, true);
    }
    return g_used + extra <= g_budget;
}

static void stage_ensure(void) {
    if (!g_stage) g_stage = xmalloc((size_t)V41CKPT_STAGE);
}

/* 张量 → 文件(流式): 8 MiB 块回读后立刻写，主机里最多只有一块 */
static bool tensor_to_file(FILE *fp, ds4_gpu_tensor *t, uint64_t bytes) {
    if (!bytes) return true;
    if (!t || bytes > ds4_gpu_tensor_bytes(t)) return false;
    stage_ensure();
    for (uint64_t off = 0; off < bytes; off += V41CKPT_STAGE) {
        const uint64_t n = bytes - off < V41CKPT_STAGE ? bytes - off : V41CKPT_STAGE;
        if (!ds4_gpu_tensor_read(t, off, g_stage, n)) return false;
        if (fwrite(g_stage, 1, (size_t)n, fp) != (size_t)n) return false;
    }
    return true;
}
static bool file_to_tensor(FILE *fp, ds4_gpu_tensor *t, uint64_t bytes) {
    if (!bytes) return true;
    if (!t || bytes > ds4_gpu_tensor_bytes(t)) return false;
    stage_ensure();
    for (uint64_t off = 0; off < bytes; off += V41CKPT_STAGE) {
        const uint64_t n = bytes - off < V41CKPT_STAGE ? bytes - off : V41CKPT_STAGE;
        if (fread(g_stage, 1, (size_t)n, fp) != (size_t)n) return false;
        if (!ds4_gpu_tensor_write(t, off, g_stage, n)) return false;
    }
    return true;
}

static uint64_t win_bytes(void) { return (uint64_t)DS4_N_SWA * DS4_N_HEAD_DIM * 4u; }

/* 一份检查点的文件字节 = 头+层账+tok/sig+每层缓冲(只写有效份)+mainh+logits 一行 */
static uint64_t file_bytes(const ckpt *k) {
    uint64_t b = V41CKPT_HDR + V41CKPT_META + (uint64_t)k->n * 12u;
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4u, wb = win_bytes();
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        b += wb;
        if (!k->ckv[il]) continue;   /* 非压缩源层: ckv/idk 全 NULL ⇒ 文件 0 字节 */
        b += (uint64_t)k->ng_src[il] * DS4_V41_CKV_BYTES + (uint64_t)k->ng_src[il] * DS4_V41_IDXK_BYTES;
        if (k->cpk[il] && k->cps[il] && k->cpend[il]) b += (uint64_t)k->cpend[il] * rowb * 2u;
    }
    b += k->mainh_bytes + (uint64_t)DS4_N_VOCAB * 4u;
    return b;
}

static bool write_header(FILE *fp, uint32_t slot, uint32_t n, uint32_t ctx, uint32_t flags, uint32_t layer_count,
                         uint64_t created, uint64_t last_used, uint64_t mainh_bytes, int64_t mainh_end,
                         uint32_t mainh_n, uint64_t logits_bytes) {
    uint8_t h[V41CKPT_HDR] = {0};
    memcpy(h, V41CKPT_MAGIC, 8);
    le32(h + 8, V41CKPT_VERSION); le32(h + 12, slot); le32(h + 16, n); le32(h + 20, ctx);
    le32(h + 24, flags); le32(h + 28, 0);
    le64(h + 32, created); le64(h + 40, last_used); le64(h + 48, mainh_bytes);
    le64(h + 56, (uint64_t)mainh_end); le32(h + 64, mainh_n); le64(h + 68, logits_bytes);
    le32(h + 76, layer_count);
    return fwrite(h, 1, V41CKPT_HDR, fp) == V41CKPT_HDR;
}

static bool write_meta(FILE *fp, const ckpt *k, uint32_t il, uint32_t ckv_b, uint32_t idk_b,
                       uint32_t cpk_b, uint32_t cps_b) {
    uint8_t m[V41CKPT_META_PER] = {0};
    le64(m + 0, ckv_b); le64(m + 8, idk_b); le64(m + 16, cpk_b); le64(m + 24, cps_b);
    le32(m + 32, il < DS4_N_LAYER ? k->ng_src[il] : 0);
    le32(m + 36, il < DS4_N_LAYER ? k->cpend[il] : 0);
    le32(m + 40, il < DS4_N_LAYER ? k->win_from[il] : 0);
    return fwrite(m, 1, V41CKPT_META_PER, fp) == V41CKPT_META_PER;
}

static bool write_tensors(FILE *fp, const ckpt *k) {
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4u, wb = win_bytes();
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!tensor_to_file(fp, k->win[il], wb)) return false;
        if (!k->ckv[il]) continue;
        if (!tensor_to_file(fp, k->ckv[il], (uint64_t)k->ng_src[il] * DS4_V41_CKV_BYTES)) return false;
        if (!tensor_to_file(fp, k->idk[il], (uint64_t)k->ng_src[il] * DS4_V41_IDXK_BYTES)) return false;
        if (k->cpk[il] && k->cps[il] && k->cpend[il]) {
            const uint64_t pb = (uint64_t)k->cpend[il] * rowb;
            if (!tensor_to_file(fp, k->cpk[il], pb) || !tensor_to_file(fp, k->cps[il], pb)) return false;
        }
    }
    if (!tensor_to_file(fp, k->mainh, k->mainh_bytes)) return false;
    return tensor_to_file(fp, k->logits, (uint64_t)DS4_N_VOCAB * 4u);
}

/* 回读时先扫目录建索引; 只读头，不读 tok/sig/缓冲 —— 索引本身只有 path+hash+n+slot+bytes+last */
static bool scan_dir(void) {
    DIR *d = opendir(g_dir);
    if (!d) return false;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        const char *name = de->d_name;
        if (strlen(name) != V41CKPT_NAME_LEN || strncmp(name, "v41ck_", 6) != 0 ||
            strcmp(name + 24, ".ckp") != 0) continue;
        char hash[17];
        for (int i = 0; i < 16; i++) {
            char c = name[6 + i];
            if (!isxdigit((unsigned char)c)) goto skip;
            hash[i] = (char)tolower((unsigned char)c);
        }
        hash[16] = '\0';
        const char *path = path_join(g_dir, name);
        struct stat st;
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) { free((void *)path); goto skip; }
        FILE *fp = fopen(path, "rb");
        if (!fp) { free((void *)path); goto skip; }
        uint8_t h[V41CKPT_HDR];
        bool ok = fread(h, 1, V41CKPT_HDR, fp) == V41CKPT_HDR && memcmp(h, V41CKPT_MAGIC, 8) == 0 &&
                  get32(h + 8) == V41CKPT_VERSION;
        uint32_t slot = 0, n = 0, flags = 0;
        uint64_t last_used = 0;
        if (ok) {
            slot = get32(h + 12); n = get32(h + 16); flags = get32(h + 24);
            last_used = get64(h + 40);
            if (!last_used) last_used = get64(h + 32);
            ok = n > 0 && slot < 2 && (uint64_t)st.st_size >= V41CKPT_HDR + V41CKPT_META + (uint64_t)n * 12u;
        }
        fclose(fp);
        if (!ok || entry_push((char *)path, hash, slot, n, flags, (uint64_t)st.st_size, last_used) == false) {
            free((void *)path); goto skip;
        }
        continue;
skip:;
    }
    closedir(d);
    return true;
}

/* 回读头 + 层账(层账在文件里连续 61 份，只取本模型层数的前 DS4_N_LAYER 份) */
static bool read_header_meta(FILE *fp, uint32_t *slot, uint32_t *n, uint32_t *ctx, uint32_t *flags,
                             uint32_t *layer_count, uint64_t *last_used, uint64_t *mainh_bytes,
                             int64_t *mainh_end, uint32_t *mainh_n, uint64_t *logits_bytes,
                             uint8_t meta[V41CKPT_META]) {
    uint8_t h[V41CKPT_HDR];
    if (fread(h, 1, V41CKPT_HDR, fp) != V41CKPT_HDR || memcmp(h, V41CKPT_MAGIC, 8) != 0) return false;
    if (get32(h + 8) != V41CKPT_VERSION) return false;
    *slot = get32(h + 12); *n = get32(h + 16); *ctx = get32(h + 20); *flags = get32(h + 24);
    *last_used = get64(h + 40); *mainh_bytes = get64(h + 48);
    *mainh_end = (int64_t)get64(h + 56); *mainh_n = get32(h + 64); *logits_bytes = get64(h + 68);
    *layer_count = get32(h + 76);
    return fread(meta, 1, V41CKPT_META, fp) == V41CKPT_META;
}

static bool file_bytes_match(FILE *fp, uint64_t off, const uint8_t *src, uint64_t bytes, bool require_zero) {
    if (!bytes) return true;
    stage_ensure();
    if (fseeko(fp, (off_t)off, SEEK_SET) != 0) return false;
    for (uint64_t done = 0; done < bytes; done += V41CKPT_STAGE) {
        const uint64_t n = bytes - done < V41CKPT_STAGE ? bytes - done : V41CKPT_STAGE;
        if (fread(g_stage, 1, (size_t)n, fp) != (size_t)n) return false;
        if (require_zero) {
            for (uint64_t i = 0; i < n; i++) if (g_stage[i]) return false;
        } else if (memcmp(g_stage, src + done, (size_t)n) != 0) return false;
    }
    return true;
}

static void touch_file(FILE *fp, uint64_t now) {
    if (!fp || fseeko(fp, 40, SEEK_SET) != 0) return;
    uint8_t b[8]; le64(b, now); (void)fwrite(b, 1, 8, fp);
}

bool v41_ckpt_disk_open(const char *dir, uint64_t budget_mb, char *err, size_t errn) {
    if (err && errn) err[0] = '\0';
    v41_ckpt_disk_close();
    if (!dir || !dir[0]) { if (err && errn) snprintf(err, errn, "checkpoint disk dir empty"); return false; }
    if (mkdir(dir, 0775) != 0 && errno != EEXIST) {
        if (err && errn) snprintf(err, errn, "cannot create checkpoint dir %s: %s", dir, strerror(errno));
        return false;
    }
    { size_t dlen = strlen(dir) + 1; g_dir = xmalloc(dlen); memcpy(g_dir, dir, dlen); }
    if (!budget_mb) budget_mb = V41CKPT_DEFAULT_MB;
    g_budget = budget_mb * 1024ull * 1024ull;
    if (!scan_dir() || !trim_lru(0)) {
        if (err && errn) snprintf(err, errn, "cannot scan/budget checkpoint dir %s", dir);
        v41_ckpt_disk_close();
        return false;
    }
    g_on = 1;
    return true;
}

void v41_ckpt_disk_close(void) {
    for (int i = g_len - 1; i >= 0; i--) entry_remove(i, false);   /* 只关索引, 不删已落盘的文件 */
    free(g_entry); g_entry = NULL; g_len = g_cap = 0; g_used = 0;
    free(g_stage); g_stage = NULL;
    free(g_dir); g_dir = NULL; g_budget = 0; g_on = 0;
}

bool v41_ckpt_disk_enabled(void) { return g_on != 0; }

void v41_ckpt_disk_clear(void) {
    for (int i = g_len - 1; i >= 0; i--) entry_remove(i, true);
    /* 目录和索引还活着: 切插件后新检查点照样能存新文件(旧文件已删) */
}

bool v41_ckpt_disk_store(const ckpt *k, uint32_t slot, char *err, size_t errn) {
    if (err && errn) err[0] = '\0';
    if (!g_on || !k || !k->valid || !k->n || slot >= 2) return false;
    const uint64_t fb = file_bytes(k);
    if (fb > g_budget) {
        if (err && errn) snprintf(err, errn, "checkpoint %.0f MB > disk budget %.0f MB", (double)fb / 1048576.0, (double)g_budget / 1048576.0);
        return false;
    }
    uint64_t h = v41ck_hash(k->tok, (size_t)k->n * 4u);
    char hash[17]; hash_hex(h, hash);
    char name[V41CKPT_NAME_LEN + 1];
    snprintf(name, sizeof name, "v41ck_%s_%u.ckp", hash, slot);
    char *path = path_join(g_dir, name);
    /* 同键重存: 旧索引的字节不再算(文件马上被 wb 截断)，预算只按新文件算 */
    for (int i = 0; i < g_len; i++) if (!strcmp(g_entry[i].path, path)) { entry_remove(i, false); break; }
    if (!trim_lru(fb)) {
        if (err && errn) snprintf(err, errn, "checkpoint disk budget full (need %.0f MB)", (double)fb / 1048576.0);
        free(path);
        return false;
    }
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        if (err && errn) snprintf(err, errn, "cannot write checkpoint %s", path);
        free(path); return false;
    }
    uint32_t flags = 0;
    if (k->sig) for (uint32_t i = 0; i < k->n; i++) if (k->sig[i]) { flags = 1; break; }
    const uint64_t now = (uint64_t)time(NULL);
    bool ok = write_header(fp, slot, k->n, (uint32_t)g_ds4_v41.ctx, flags, DS4_N_LAYER, now, now,
                           k->mainh_bytes, k->mainh_end, k->mainh_n, (uint64_t)DS4_N_VOCAB * 4u);
    for (uint32_t il = 0; ok && il < DS4_MAX_LAYER; il++) {
        uint32_t ckv = 0, idk = 0, cpk = 0, cps = 0;
        if (il < DS4_N_LAYER && k->ckv[il]) {
            ckv = k->ng_src[il] * DS4_V41_CKV_BYTES; idk = k->ng_src[il] * DS4_V41_IDXK_BYTES;
            if (k->cpk[il] && k->cpend[il]) cpk = k->cpend[il] * DS4_N_HEAD_DIM * 4u, cps = cpk;
        }
        ok = write_meta(fp, k, il, ckv, idk, cpk, cps);
    }
    if (ok) ok = fwrite(k->tok, 1, (size_t)k->n * 4u, fp) == (size_t)k->n * 4u &&
                  fwrite(k->sig, 1, (size_t)k->n * 8u, fp) == (size_t)k->n * 8u;
    if (ok) ok = write_tensors(fp, k);
    if (ok) ok = fflush(fp) == 0;
    fclose(fp);
    if (!ok || !ds4_gpu_synchronize()) {
        unlink(path);
        if (err && errn) snprintf(err, errn, "checkpoint write failed %s", path);
        free(path); return false;
    }
    entry_push(path, hash, slot, k->n, flags, fb, now);
    return true;
}

/* 候选: 按"本条提示的前 n 位字节哈希"与文件哈希名比; 取 n 最长的那份(槽 1 通常 ⊇ 槽 0)。
 * 哈希非密码学 ⇒ 装之前还要逐字节比文件里的 tok/sig。 */
static const v41ckpt_disk_entry *find_candidate(const int32_t *prompt, uint32_t np) {
    const v41ckpt_disk_entry *best = NULL;
    for (int i = 0; i < g_len; i++) {
        const v41ckpt_disk_entry *e = &g_entry[i];
        if (!e->n || e->n > np || (best && e->n <= best->n)) continue;
        uint64_t h = v41ck_hash(prompt, (size_t)e->n * 4u);
        char hex[17]; hash_hex(h, hex);
        if (strcmp(hex, e->hash) == 0) best = e;
    }
    return best;
}

uint32_t v41_ckpt_disk_restore_state(ds4_v41_state *st, const int32_t *prompt, uint32_t np,
                                     const uint64_t *sig, char *err, size_t errn) {
    if (err && errn) err[0] = '\0';
    if (!g_on || !st || !prompt || np == 0) return 0;
    const v41ckpt_disk_entry *e = find_candidate(prompt, np);
    if (!e || e->n > st->ctx) return 0;
    FILE *fp = fopen(e->path, "rb");
    if (!fp) return 0;
    uint32_t slot = 0, n = 0, ctx = 0, flags = 0, layer_count = 0, mainh_n = 0;
    uint64_t last_used = 0, mainh_bytes = 0, logits_bytes = 0;
    int64_t mainh_end = 0;
    uint8_t meta[V41CKPT_META];
    bool ok = read_header_meta(fp, &slot, &n, &ctx, &flags, &layer_count, &last_used, &mainh_bytes,
                               &mainh_end, &mainh_n, &logits_bytes, meta);
    ok = ok && n == e->n && slot == e->slot && ctx == (uint32_t)g_ds4_v41.ctx && layer_count == DS4_N_LAYER;
    if (ok) {
        const uint64_t tok_off = V41CKPT_HDR + V41CKPT_META, sig_off = tok_off + (uint64_t)n * 4u;
        ok = file_bytes_match(fp, tok_off, (const uint8_t *)prompt, (uint64_t)n * 4u, false);
        if (ok && sig) ok = file_bytes_match(fp, sig_off, (const uint8_t *)sig, (uint64_t)n * 8u, false);
        else if (ok) ok = file_bytes_match(fp, sig_off, NULL, (uint64_t)n * 8u, true);   /* 无图 ⇒ 文件里指纹全 0 */
    }
    if (!ok) { fclose(fp); return 0; }
    /* 装进请求态: 只写已开的缓冲，不新分 GPU —— 请求态自己的容量比检查点那份大 */
    const uint64_t wb = win_bytes();
    uint64_t off = V41CKPT_HDR + V41CKPT_META + (uint64_t)n * 12u;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const uint64_t ckv = get64(meta + il * V41CKPT_META_PER + 0), idk = get64(meta + il * V41CKPT_META_PER + 8);
        const uint64_t cpk = get64(meta + il * V41CKPT_META_PER + 16), cps = get64(meta + il * V41CKPT_META_PER + 24);
        ok = file_to_tensor(fp, st->win[il], wb); off += wb;
        if (ok && ckv) {
            if (!st->comp_kv[il]) ok = false;
            else { ok = file_to_tensor(fp, st->comp_kv[il], ckv); off += ckv; }
        }
        if (ok && idk) {
            if (!st->index_k[il]) ok = false;
            else { ok = file_to_tensor(fp, st->index_k[il], idk); off += idk; }
        }
        if (ok && cpk) {
            if (!st->cpre_kv[il] || !st->cpre_sc[il]) ok = false;
            else {
                ok = file_to_tensor(fp, st->cpre_kv[il], cpk) && file_to_tensor(fp, st->cpre_sc[il], cps);
                off += cpk + cps;
            }
        }
    }
    if (ok && mainh_bytes) {
        if (!st->mainh || mainh_bytes > ds4_gpu_tensor_bytes(st->mainh)) ok = false;
        else { ok = file_to_tensor(fp, st->mainh, mainh_bytes); off += mainh_bytes; }
    }
    if (ok) {
        if (logits_bytes != (uint64_t)DS4_N_VOCAB * 4u || !st->logits) ok = false;
        else ok = file_to_tensor(fp, st->logits, logits_bytes);
    }
    if (ok) ok = ds4_gpu_synchronize();
    fclose(fp);
    if (!ok) { if (err && errn) snprintf(err, errn, "checkpoint restore failed %s", e->path); return 0; }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        st->ng_src[il] = get32(meta + il * V41CKPT_META_PER + 32);
        st->cpend[il] = get32(meta + il * V41CKPT_META_PER + 36);
        st->win_from[il] = get32(meta + il * V41CKPT_META_PER + 40);
    }
    if (st->mainh) { st->mainh_end = mainh_end; st->mainh_n = mainh_n; }
    memcpy(st->hist, prompt, (size_t)n * 4u);
    for (uint32_t i = 0; i < n; i++) st->himg[i] = sig ? sig[i] != 0 : 0;
    st->n_past = n; st->last_logit_row = 0;   /* 等长复用: 末位 logits 在第 0 行(core_v41_sample.c 从它采) */
    /* 刚被用过 ⇒ LRU 的 last_used 往前挪(文件头也写，重启扫描后不倒退) */
    const uint64_t now = (uint64_t)time(NULL);
    ((v41ckpt_disk_entry *)e)->last_used = now;
    FILE *tfp = fopen(e->path, "r+b");
    if (tfp) { touch_file(tfp, now); fflush(tfp); fclose(tfp); }
    return n;
}

#endif /* !DS4_NO_GPU */
