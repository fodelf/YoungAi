/* ---------------- 对称特征分解: 循环 Jacobi(只用在 r×r 的小核上, r≤KMAX+64) ----------------
 * 出参: 特征值降序 w[r], 特征向量按列存 E[r][r](E[i*r+c] = 第 c 个向量的第 i 分量)。 */
static void jacobi_eig(double *A, int n, double *w, double *E) {
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) E[(size_t)i * n + j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 60; sweep++) {
        double off = 0.0;
        for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++) off += A[(size_t)p * n + q] * A[(size_t)p * n + q];
        double diag = 0.0;
        for (int p = 0; p < n; p++) diag += A[(size_t)p * n + p] * A[(size_t)p * n + p];
        if (off <= 1e-30 * (diag + 1e-300)) break;
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double apq = A[(size_t)p * n + q];
                if (fabs(apq) < 1e-300) continue;
                double app = A[(size_t)p * n + p], aqq = A[(size_t)q * n + q];
                double theta = (aqq - app) / (2.0 * apq);
                double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; k++) {
                    double akp = A[(size_t)k * n + p], akq = A[(size_t)k * n + q];
                    A[(size_t)k * n + p] = c * akp - s * akq;
                    A[(size_t)k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; k++) {
                    double apk = A[(size_t)p * n + k], aqk = A[(size_t)q * n + k];
                    A[(size_t)p * n + k] = c * apk - s * aqk;
                    A[(size_t)q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; k++) {
                    double ekp = E[(size_t)k * n + p], ekq = E[(size_t)k * n + q];
                    E[(size_t)k * n + p] = c * ekp - s * ekq;
                    E[(size_t)k * n + q] = s * ekp + c * ekq;
                }
            }
        }
    }
    for (int i = 0; i < n; i++) w[i] = A[(size_t)i * n + i];
    for (int a = 0; a < n; a++) {                     /* 降序(选择排序, n≤1088) */
        int best = a;
        for (int b = a + 1; b < n; b++) if (w[b] > w[best]) best = b;
        if (best == a) continue;
        double t = w[a]; w[a] = w[best]; w[best] = t;
        for (int k = 0; k < n; k++) { double e = E[(size_t)k * n + a]; E[(size_t)k * n + a] = E[(size_t)k * n + best]; E[(size_t)k * n + best] = e; }
    }
}

/* ---------------- 最小 npz(ZIP_STORED) 读写 ----------------
 * zcache_LXX.npz 是 np.savez 产物, 也要能被 amp_solve/rec_fidelity/各 probe 脚本读回去,
 * 所以读要吃 numpy 写的 ZIP64 局部头, 写要能被 python zipfile 校验通过(CRC32 必填)。
 * 读的部分与 calib/rec_fidelity.c 的解析器同源(那边全升 f64, 这里保留原 dtype 省内存)。 */
static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) { return p[0] | (p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static void wr16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void wr32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }

static uint32_t crc_tab[256];
static void crc_init(void) {
    if (crc_tab[1]) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_tab[i] = c;
    }
}
static uint32_t crc_upd(uint32_t crc, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    crc = ~crc;
    for (size_t i = 0; i < n; i++) crc = crc_tab[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

typedef struct { const uint8_t *data; char dt[8]; long long d0, d1, nd; } npy_t;

/* 在 npz 映像里找 name.npy, 出参指向未压缩数据(零拷贝). 找不到返回 -1. */
static int npz_find(const uint8_t *buf, long long sz, const char *name, npy_t *out) {
    char want[128]; snprintf(want, sizeof want, "%s.npy", name);
    long long off = 0;
    while (off + 30 <= sz) {
        if (rd32(buf + off) != 0x04034b50u) break;
        uint16_t meth = rd16(buf + off + 8), nlen = rd16(buf + off + 26), xlen = rd16(buf + off + 28);
        long long csz = rd32(buf + off + 18);
        const char *nm = (const char *)buf + off + 30;
        long long pay = off + 30 + nlen + xlen;
        if (csz == 0xFFFFFFFFLL) {   /* numpy 用 force_zip64 写, 真尺寸在 extra id=0x0001 */
            const uint8_t *x = buf + off + 30 + nlen, *xe = x + xlen;
            while (x + 4 <= xe) {
                uint16_t id = rd16(x), l = rd16(x + 2);
                if (id == 1 && l >= 16) { memcpy(&csz, x + 12, 8); break; }
                x += 4 + l;
            }
        }
        if ((long long)strlen(want) == nlen && !memcmp(nm, want, (size_t)nlen)) {
            if (meth != 0) die("npz %s: 压缩条目不支持(np.savez 应为 STORED)", name);
            const uint8_t *p = buf + pay;
            if (memcmp(p, "\x93NUMPY", 6)) die("npz %s: 非 npy 头", name);
            int maj = p[6];
            uint32_t hl = maj >= 2 ? rd32(p + 8) : rd16(p + 8);
            const char *hdr = (const char *)p + (maj >= 2 ? 12 : 10);
            memset(out, 0, sizeof *out);
            out->data = p + (maj >= 2 ? 12 : 10) + hl;
            { const char *q = strstr(hdr, "'descr':"); if (!q) die("npz %s: 无 descr", name);
              q = strchr(q + 8, '\''); const char *e = strchr(q + 1, '\'');
              size_t l = (size_t)(e - q - 1); if (l > 7) l = 7; memcpy(out->dt, q + 1, l); }
            { const char *q = strstr(hdr, "'shape':"); if (!q) die("npz %s: 无 shape", name);
              q = strchr(q, '(') + 1; out->d0 = 1; out->d1 = 1; out->nd = 0;
              while (*q && *q != ')') {
                  while (*q == ' ' || *q == ',') q++;
                  if (*q == ')') break;
                  long long v = strtoll(q, (char **)&q, 10);
                  if (out->nd == 0) out->d0 = v; else if (out->nd == 1) out->d1 = v;
                  out->nd++;
              } }
            if (strstr(hdr, "'fortran_order': True")) die("npz %s: fortran_order 不支持", name);
            return 0;
        }
        off = pay + csz;
    }
    return -1;
}
/* 取字段并转成目标类型(数据在 npz 里可能非对齐, 一律 memcpy 逐元素搬) */
static void npz_to_f32(const npy_t *a, float *out, long long n) {
    if (!strcmp(a->dt, "<f4")) { memcpy(out, a->data, (size_t)n * 4); return; }
    if (!strcmp(a->dt, "<f8")) { for (long long i = 0; i < n; i++) { double v; memcpy(&v, a->data + i * 8, 8); out[i] = (float)v; } return; }
    if (!strcmp(a->dt, "<f2")) { for (long long i = 0; i < n; i++) { uint16_t v; memcpy(&v, a->data + i * 2, 2); out[i] = f16_to_f32(v); } return; }
    die("npz: dtype %s 不能转 f32", a->dt);
}
static void npz_to_i32(const npy_t *a, int *out, long long n) {
    if (!strcmp(a->dt, "<i4")) { for (long long i = 0; i < n; i++) { int32_t v; memcpy(&v, a->data + i * 4, 4); out[i] = v; } return; }
    if (!strcmp(a->dt, "<i8")) { for (long long i = 0; i < n; i++) { int64_t v; memcpy(&v, a->data + i * 8, 8); out[i] = (int)v; } return; }
    die("npz: dtype %s 不能转 i32", a->dt);
}

/* --- npz 写: 顺序追加 STORED 条目, 收尾写中央目录 --- */
typedef struct { char name[64]; uint32_t crc, size, off; } zent_t;
typedef struct { FILE *f; zent_t e[16]; int n; long long pos; } zwr_t;

static void npy_header(char *hdr, size_t cap, const char *dt, long long d0, long long d1, int nd, size_t *hlen) {
    char body[256];
    if (nd == 1) snprintf(body, sizeof body, "{'descr': '%s', 'fortran_order': False, 'shape': (%lld,), }", dt, d0);
    else snprintf(body, sizeof body, "{'descr': '%s', 'fortran_order': False, 'shape': (%lld, %lld), }", dt, d0, d1);
    size_t need = 10 + strlen(body) + 1;
    size_t pad = (64 - (need % 64)) % 64;
    size_t hl = strlen(body) + pad + 1;
    if (10 + hl > cap) die("npy 头太长");
    memcpy(hdr, "\x93NUMPY\x01\x00", 8);
    wr16((uint8_t *)hdr + 8, (uint32_t)hl);
    memcpy(hdr + 10, body, strlen(body));
    memset(hdr + 10 + strlen(body), ' ', pad);
    hdr[10 + hl - 1] = '\n';
    *hlen = 10 + hl;
}
/* data=NULL 时写 n 个零字节的数据体(pDY 是全零, 不必真开 169MB) */
static void zw_add(zwr_t *z, const char *name, const char *dt, long long d0, long long d1, int nd,
                   const void *data, size_t esz) {
    char hdr[256]; size_t hlen;
    npy_header(hdr, sizeof hdr, dt, d0, d1, nd, &hlen);
    size_t nel = (size_t)d0 * (size_t)(nd > 1 ? d1 : 1);
    size_t dsz = nel * esz, total = hlen + dsz;
    if (z->pos + (long long)total + 4096 > 0xF0000000LL)
        die("zcache 超过 4GB —— 本实现只写 ZIP32, 拒绝写出会被 numpy 读坏的文件");
    uint32_t crc = 0;
    crc = crc_upd(crc, hdr, hlen);
    if (data) crc = crc_upd(crc, data, dsz);
    else { static const uint8_t zbuf[65536] = {0}; size_t left = dsz;
           while (left) { size_t c = left > sizeof zbuf ? sizeof zbuf : left; crc = crc_upd(crc, zbuf, c); left -= c; } }
    char fn[64]; snprintf(fn, sizeof fn, "%s.npy", name);
    uint8_t lh[30] = {0};
    wr32(lh, 0x04034b50u); wr16(lh + 4, 20); wr16(lh + 6, 0); wr16(lh + 8, 0);
    wr16(lh + 10, 0); wr16(lh + 12, 0x21);          /* 固定时间戳: 产物字节可复现 */
    wr32(lh + 14, crc); wr32(lh + 18, (uint32_t)total); wr32(lh + 22, (uint32_t)total);
    wr16(lh + 26, (uint32_t)strlen(fn)); wr16(lh + 28, 0);
    zent_t *e = &z->e[z->n++];
    snprintf(e->name, sizeof e->name, "%s", fn);
    e->crc = crc; e->size = (uint32_t)total; e->off = (uint32_t)z->pos;
    fwrite(lh, 1, 30, z->f); fwrite(fn, 1, strlen(fn), z->f); fwrite(hdr, 1, hlen, z->f);
    if (data) fwrite(data, 1, dsz, z->f);
    else { static const uint8_t zbuf[65536] = {0}; size_t left = dsz;
           while (left) { size_t c = left > sizeof zbuf ? sizeof zbuf : left; fwrite(zbuf, 1, c, z->f); left -= c; } }
    z->pos += 30 + (long long)strlen(fn) + (long long)total;
}
static void zw_finish(zwr_t *z) {
    long long cd = z->pos;
    for (int i = 0; i < z->n; i++) {
        zent_t *e = &z->e[i];
        uint8_t ch[46] = {0};
        wr32(ch, 0x02014b50u); wr16(ch + 4, 20); wr16(ch + 6, 20); wr16(ch + 8, 0); wr16(ch + 10, 0);
        wr16(ch + 12, 0); wr16(ch + 14, 0x21);
        wr32(ch + 16, e->crc); wr32(ch + 20, e->size); wr32(ch + 24, e->size);
        wr16(ch + 28, (uint32_t)strlen(e->name)); wr16(ch + 30, 0); wr16(ch + 32, 0);
        wr16(ch + 34, 0); wr16(ch + 36, 0); wr32(ch + 38, 0); wr32(ch + 42, e->off);
        fwrite(ch, 1, 46, z->f); fwrite(e->name, 1, strlen(e->name), z->f);
        z->pos += 46 + (long long)strlen(e->name);
    }
    uint8_t eo[22] = {0};
    wr32(eo, 0x06054b50u); wr16(eo + 4, 0); wr16(eo + 6, 0);
    wr16(eo + 8, (uint32_t)z->n); wr16(eo + 10, (uint32_t)z->n);
    wr32(eo + 12, (uint32_t)(z->pos - cd)); wr32(eo + 16, (uint32_t)cd); wr16(eo + 20, 0);
    fwrite(eo, 1, 22, z->f);
}

/* ---------------- VQ 侧车(dql_vq_LXX.bin)反量化 ----------------
 * 逐式对应 probe_behavior_spectrum.vq_dequant:
 *   槽表在 blob+16, 每专家 3 个 u64 偏移(w1,w3,w2 顺序);
 *   载荷 = magic|dim,nc(u16)|rows,cols(u32) | 码本 f16[nc*dim] | 行乘子 f16[rows] | 索引位流。
 *   out[r][c] = cb[idx[r*(cols/dim) + c/dim]][c%dim] * gr[r]
 * 位流是小端 bit order(numpy unpackbits bitorder='little'): 第 i 个索引取 bit 偏移 i*nbit。
 *
 * 与运行时那份的关系(2026-08-25 二期核对): quant/vq_qc.h:265 的 vq_unpack_dequant ——
 * 就是 ds4quant_run.c 的 bytes_moe 用的那个 —— 与这里【数值同式】: 位宽推法、3 字节窗
 * 取索引、码本×行乘子全一样, 只有行号的写法差异(`r=(i*dim)/cols` vs `r=i/(cols/dim)`,
 * cols 整除 dim 时恒等)。本文件这份多了两处【只读 guard】: 索引 ≥nc 时钳到 nc-1、
 * 位流读取按 blob 尾部截断补零 —— 侧车是外部产物, 越界会读飞。 */
#define VQ_MAGIC 0x51565144u

static uint64_t vq_slot(const uint8_t *blob, size_t bsz, int e, int w) {
    size_t off = 16 + (size_t)(e * 3 + w) * 8;
    if (off + 8 > bsz) die("vq_slot 越界: e=%d w=%d", e, w);
    uint64_t v; memcpy(&v, blob + off, 8); return v;
}
/* 返回 malloc 的 f32 [rows*cols]; off==0 视为无该槽(返回 NULL, 与 py `if off else None` 同) */
static float *vq_dequant(const uint8_t *blob, size_t bsz, uint64_t off, long *R_out, long *C_out) {
    if (!off) return NULL;
    if (off + 16 > bsz) die("vq 载荷越界 off=%llu", (unsigned long long)off);
    uint32_t magic; memcpy(&magic, blob + off, 4);
    if (magic != VQ_MAGIC) die("vq 载荷 magic 不对: 0x%08x (assert)", magic);
    uint16_t dim, nc; memcpy(&dim, blob + off + 4, 2); memcpy(&nc, blob + off + 6, 2);
    uint32_t rows, cols; memcpy(&rows, blob + off + 8, 4); memcpy(&cols, blob + off + 12, 4);
    size_t p = (size_t)off + 16;
    size_t ncb = (size_t)nc * dim;
    float *cb = (float *)xmalloc(ncb * sizeof(float));
    for (size_t i = 0; i < ncb; i++) { uint16_t h; memcpy(&h, blob + p + i * 2, 2); cb[i] = f16_to_f32(h); }
    p += ncb * 2;
    float *gr = (float *)xmalloc((size_t)rows * sizeof(float));
    for (size_t i = 0; i < rows; i++) { uint16_t h; memcpy(&h, blob + p + i * 2, 2); gr[i] = f16_to_f32(h); }
    p += (size_t)rows * 2;
    int nbit = 1; while ((1 << nbit) < (int)nc) nbit++;     /* = max(1,(nc-1).bit_length()) */
    size_t nidx = (size_t)rows * cols / dim;
    size_t per = cols / dim;
    float *W = (float *)xmalloc((size_t)rows * cols * sizeof(float));
    for (size_t i = 0; i < nidx; i++) {
        uint32_t id;
        if (nbit == 8) { id = (p + i < bsz) ? blob[p + i] : 0u; }
        else {
            uint64_t bit = (uint64_t)i * (uint64_t)nbit, by = bit >> 3;
            uint32_t v = 0;
            for (int k = 0; k < 3; k++) if (p + by + k < bsz) v |= (uint32_t)blob[p + by + k] << (8 * k);
            id = (v >> (bit & 7)) & ((1u << nbit) - 1u);
        }
        if (id >= nc) id = nc - 1;
        size_t r = i / per;
        float g = gr[r];
        const float *c = cb + (size_t)id * dim;
        float *w = W + i * dim;
        for (int j = 0; j < dim; j++) w[j] = c[j] * g;
    }
    free(cb); free(gr);
    *R_out = (long)rows; *C_out = (long)cols;
    return W;
}

/* ================= GGUF 标量模式(DS4_ZL_GGUF) =================
 * py 侧: gguf.GGUFReader 取张量 + gguf.quants.dequantize 反量化, 专家沿【外维】连续,
 *        所以"第 e 个专家"= 该张量原始字节按专家数均分后的第 e 段(py: per=db.size//nexp)。
 * 这里对齐的是【数值结果】不是实现 —— py 走 gguf-py 的 numpy 向量化路, 这里走标量循环。
 *
 * 【来源, 不重写数值表】
 *   - GGUF v3 头解析: 逐式抄 quant/vq_merge_v4.c 的 rdstr/skipv/parse_header
 *     (同一份小端假设; 那边有"同参数下与 vq_merge_v4.py 逐字节 md5 相同"的金标)。
 *   - q2_K: 抄 ds4.c:4352 的 deq_q2K_row_f32(注释里写明与 CUDA host_deq_q2k_block
 *     同式且已对拍); 索引式 qpos/shift 一字未改。
 *   - iq2_xxs 网格: 用 gguf-tools/quants.c:714 那张 kgrid[256](IQ2_XXS 编码器建表用的
 *     同一张表), 由 2bit 四元组现场展开成 8×int8 = 2*l+1 —— 与 llama.cpp 的
 *     iq2xxs_grid 逐字节同, 所以不另抄一份 2048 字节的常量。
 *     符号表 ksigns_iq2xs 也不抄: 它就是 "popcount 为奇数则置 bit7", 现场算。
 *   - q4_K/q8_0: llama.cpp 标准式(get_scale_min_k4 / y=d·q), 与 gguf-py 的
 *     Q4_K.get_scale_min / Q8_0.dequantize_blocks 逐式同。
 * 【范围】只实现这条产线真出现过的类型(全q2 底座 = 专家 w1/w3 IQ2_XXS + w2 Q2_K,
 *   见 scripts/quant_allq2_spark.sh), 外加 f32/f16/bf16/q8_0/q4_K 这几个零成本的。
 *   碰到别的类型直接停车 —— py 那边 gguf-py 也是抛异常, 不静默出垃圾。 */

/* ★重构阶段2(2026-08-25)★ 上面注释描述的实现已升格为全仓唯一
 * src/common/{ds4_gguf,ds4_quantfmt}(逐式搬移, tests/fixtures/quantfmt/ 金标
 * 逐字节回归)。此处只剩 die 口径 shim: 库返回错误码, 停车与否由本工具定。 */
#include "../../src/common/ds4_quantfmt.c"
#include "../../src/common/ds4_gguf.c"

enum { GGT_F32 = DS4_GGT_F32, GGT_F16 = DS4_GGT_F16, GGT_Q8_0 = DS4_GGT_Q8_0,
       GGT_Q2_K = DS4_GGT_Q2_K, GGT_Q4_K = DS4_GGT_Q4_K,
       GGT_IQ2_XXS = DS4_GGT_IQ2_XXS, GGT_BF16 = DS4_GGT_BF16 };
typedef ds4_gguf gg_ctx;
typedef ds4_gguf_tensor gg_tensor;

static void gg_type_geom(uint32_t ty, uint64_t *blk, uint64_t *tsz) {
    if (!ds4_ggt_geom(ty, blk, tsz))
        die("GGUF 张量类型 %u 未实现 dequant — .py 侧 gguf-py 同样会抛, 拒跑", ty);
}

static void gg_dequant(uint32_t ty, const uint8_t *src, uint64_t nelem, float *out) {
    uint64_t blk, tsz;
    if (!ds4_ggt_geom(ty, &blk, &tsz)) die("GGUF dequant: 类型 %u 未实现", ty);
    if (nelem % blk) die("GGUF dequant: 元素数 %llu 不是块 %llu 的整数倍",
                         (unsigned long long)nelem, (unsigned long long)blk);
    if (ds4_deq_bytes(ty, src, nelem, out)) die("GGUF dequant: 类型 %u 失败", ty);
}

static void gg_open(gg_ctx *g, const char *path) {
    char err[256];
    if (ds4_gguf_open(g, path, err, sizeof err)) die("DS4_ZL_GGUF %s", err);
}

/* py 的 _gg_expert(l,nm,e): 张量 blk.<L>.ffn_{gate,up,down}_exps.weight,
 * shape=[内维 ne0, 行 ne1, 专家 ne2] → 每专家 [rows=ne[-2], cols=ne[0]],
 * 字节按专家数均分(per = 总字节/nexp)。返回 malloc 的 f32 [rows*cols]。 */
static float *gg_expert(const gg_ctx *g, int L, const char *nm, int e, long *R_out, long *C_out) {
    static const char *TN[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };
    int wi = !strcmp(nm, "w1") ? 0 : (!strcmp(nm, "w3") ? 1 : 2);
    char want[128];
    snprintf(want, sizeof want, "blk.%d.%s.weight", L, TN[wi]);
    const gg_tensor *t = NULL;
    for (int i = 0; i < g->nt; i++) if (!strcmp(g->t[i].name, want)) { t = &g->t[i]; break; }
    if (!t) die("KeyError: GGUF 里没有张量 %s", want);
    if (t->nd < 3) die("%s 维数 %u < 3 — 没有专家外维, 口径不明拒跑", want, t->nd);
    const uint64_t nexp = t->ne[t->nd - 1], rows = t->ne[t->nd - 2], cols = t->ne[0];
    if ((uint64_t)e >= nexp) die("%s: 专家 %d ≥ %llu", want, e, (unsigned long long)nexp);
    uint64_t nel = 1;
    for (uint32_t d2 = 0; d2 < t->nd; d2++) nel *= t->ne[d2];
    if (nel != rows * cols * nexp) die("%s: 维数 %u 有中间维, py 的 rows/cols 取法会错位, 拒跑", want, t->nd);
    uint64_t blk, tsz; gg_type_geom(t->type, &blk, &tsz);
    if (nel % blk) die("%s: 元素数 %llu 不是块 %llu 整数倍", want, (unsigned long long)nel, (unsigned long long)blk);
    const uint64_t nbytes = nel / blk * tsz;
    if (nbytes % nexp) die("%s: 字节 %llu 不能按 %llu 专家均分(py 的 per=db.size//nexp 会截断)",
                           want, (unsigned long long)nbytes, (unsigned long long)nexp);
    const uint64_t per = nbytes / nexp;
    const uint64_t base = g->data0 + t->off + (uint64_t)e * per;
    if (base + per > g->msz) die("%s 专家 %d 数据越界(off=%llu)", want, e, (unsigned long long)base);
    float *w = (float *)xmalloc((size_t)rows * cols * sizeof(float));
    gg_dequant(t->type, g->map + base, rows * cols, w);
    *R_out = (long)rows; *C_out = (long)cols;
    return w;
}

/* ---------------- 锚(DQA2)读取: probe_layer_behavior.anchor_layer 逐式 ---------------- */
typedef struct { int S, HCM, DIM, NL, VOCAB, NACT; } ameta_t;
static void anchor_layer(const char *ap, int L, int ntok, float **fin, int **ridx, float **rw, ameta_t *m) {
    FILE *f = fopen(ap, "rb");
    if (!f) die("锚打不开: %s", ap);
    uint32_t hd[8];
    if (fread(hd, 4, 8, f) != 8) die("锚头截断: %s", ap);
    if (hd[0] != 0x32415144u) die("%s 不是 DQA2 锚(magic 0x%08x)", ap, hd[0]);
    m->S = (int)hd[1]; m->HCM = (int)hd[2]; m->DIM = (int)hd[3];
    m->NL = (int)hd[4]; m->VOCAB = (int)hd[5]; m->NACT = (int)hd[6];
    if (m->DIM != D) die("锚 DIM=%d ≠ %d — .py 写死 D=4096, 口径不明拒跑", m->DIM, D);
    if (ntok > m->S) die("DS4_ZL_NTOK=%d > 锚 S=%d", ntok, m->S);
    long long fin_off = 40;
    long long ridx_off = fin_off + (long long)m->NL * m->S * m->DIM * 4;
    long long rw_off = ridx_off + (long long)m->NL * m->S * m->NACT * 4;
    *fin = (float *)xmalloc((size_t)ntok * D * sizeof(float));
    *ridx = (int *)xmalloc((size_t)ntok * m->NACT * sizeof(int));
    *rw = (float *)xmalloc((size_t)ntok * m->NACT * sizeof(float));
    if (fseeko(f, (off_t)(fin_off + (long long)L * m->S * m->DIM * 4), SEEK_SET) ||
        fread(*fin, 4, (size_t)ntok * D, f) != (size_t)ntok * D) die("锚 fin 读不满 L=%d", L);
    if (fseeko(f, (off_t)(ridx_off + (long long)L * m->S * m->NACT * 4), SEEK_SET) ||
        fread(*ridx, 4, (size_t)ntok * m->NACT, f) != (size_t)ntok * m->NACT) die("锚 ridx 读不满 L=%d", L);
    if (fseeko(f, (off_t)(rw_off + (long long)L * m->S * m->NACT * 4), SEEK_SET) ||
        fread(*rw, 4, (size_t)ntok * m->NACT, f) != (size_t)ntok * m->NACT) die("锚 rw 读不满 L=%d", L);
    fclose(f);
}

/* ---------------- swiglu / φ 提升 ----------------
 * ★分歧点★ ds4quant_fwd.c 的 dq_expert_fp 只夹 gate 的上侧(gg>swlim), .py 的 swiglu 是
 * 【双侧】夹(g 和 u 都夹到 [-lim,lim])。这里按 .py 写 —— 解算目标必须和 .py 的目标一致,
 * 否则 held 挽回率没法对拍。内层 exp 的 ±60 夹在 lim>0 时是死代码(g 已在 ±lim 内),
 * 只有 DS4_ZL_SWLIM=0(关截断)时才起作用, 照抄 CPU 路的 probe_layer_behavior.swiglu。 */
static float zl_swiglu1(float g, float u, float lim) {
    if (lim > 0) {
        if (g > lim) g = lim; else if (g < -lim) g = -lim;
        if (u > lim) u = lim; else if (u < -lim) u = -lim;
    }
    float gc = g > 60.0f ? 60.0f : (g < -60.0f ? -60.0f : g);
    return (g / (1.0f + expf(-gc))) * u;
}

/* zl_phi(M) = [M, M⊙M/rms, relu(M)], rms=sqrt(mean(M²,axis=1))+1e-6, 结果 f32。
 * 两个入口的中间精度不同, 照抄 .py:
 *   from_f64: py 传的是 f64 的 Xa —— 全程 f64 算, 最后一次性 astype(f32);
 *   from_f32: py 传的是 X[ev].astype(f32) —— 平方/除法都在 f32 上做。 */
typedef struct { const double *M; float *out; int d; } phi64_ctx;
static void phi64_worker(void *vc, int t0, int t1) {
    phi64_ctx *c = (phi64_ctx *)vc;
    for (int t = t0; t < t1; t++) {
        const double *m = c->M + (size_t)t * c->d;
        float *o = c->out + (size_t)t * 3 * c->d;
        double ss = 0;
        for (int j = 0; j < c->d; j++) ss += m[j] * m[j];
        double n = sqrt(ss / c->d) + 1e-6;
        for (int j = 0; j < c->d; j++) {
            o[j] = (float)m[j];
            o[c->d + j] = (float)((m[j] * m[j]) / n);
            o[2 * c->d + j] = (float)(m[j] > 0 ? m[j] : 0);
        }
    }
}
typedef struct { const float *M; float *out; int d; } phi32_ctx;
static void phi32_worker(void *vc, int t0, int t1) {
    phi32_ctx *c = (phi32_ctx *)vc;
    for (int t = t0; t < t1; t++) {
        const float *m = c->M + (size_t)t * c->d;
        float *o = c->out + (size_t)t * 3 * c->d;
        double ss = 0;                       /* 平方在 f32(与 numpy 同), 求和用 f64 累加器 */
        for (int j = 0; j < c->d; j++) { float sq = m[j] * m[j]; ss += (double)sq; }
        float n = (float)sqrtf((float)(ss / c->d)) + 1e-6f;
        for (int j = 0; j < c->d; j++) {
            o[j] = m[j];
            o[c->d + j] = (m[j] * m[j]) / n;
            o[2 * c->d + j] = m[j] > 0 ? m[j] : 0.0f;
        }
    }
}

