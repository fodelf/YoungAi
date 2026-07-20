/* npy.c — minimal NumPy .npy reader. See npy.h.
 *
 * Layout of a .npy file:
 *   6 bytes  magic   "\x93NUMPY"
 *   1 byte   major version
 *   1 byte   minor version
 *   header_len: v1.x -> 2 bytes LE, v2.x+ -> 4 bytes LE
 *   header:  ASCII Python-dict literal of header_len bytes, space-padded,
 *            terminated by '\n', e.g.
 *              {'descr': '<f2', 'fortran_order': False, 'shape': (12288, 4096), }
 *   data:    raw element bytes, C-order (unless fortran_order), little-endian.
 *
 * Host is assumed little-endian (darwin/linux on x86-64 / arm64), matching the
 * project's supported platforms; big-endian descr ('>') is therefore rejected.
 */
#include "npy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- IEEE-754 half (binary16) -> float (binary32), self-contained ---- */
/* Handles normals, subnormals, +/-0, +/-inf and NaN exactly. */
static float npy_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16; /* sign bit to fp32 position */
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;   /* 5-bit exponent           */
    uint32_t mant = (uint32_t)(h & 0x03FFu);       /* 10-bit mantissa          */
    uint32_t bits;

    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;                            /* +/- zero */
        } else {
            /* subnormal: renormalize into an fp32 normal.
             * fp32 exponent for value mant * 2^-24 with implicit leading 1
             * recovered by shifting the mantissa up to bit 10. */
            uint32_t e = 127u - 15u + 1u;           /* = 113 */
            while ((mant & 0x0400u) == 0u) {        /* shift until bit10 set */
                mant <<= 1;
                e--;
            }
            mant &= 0x03FFu;                        /* drop the implicit 1 */
            bits = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        /* inf (mant==0) or NaN (mant!=0); keep the mantissa payload */
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        /* normal: rebias exponent (15 -> 127), widen mantissa (10 -> 23) */
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/* ---- header dictionary parsing ---- */

static int rd_u8(FILE *f, unsigned *v) {
    int c = fgetc(f);
    if (c == EOF) return -1;
    *v = (unsigned)c;
    return 0;
}

/* Parse the ASCII dict header into *m. Returns 0 on success, -1 on error.
 * Big-endian descr is rejected; fortran_order is recorded (not rejected here). */
static int npy_parse_dict(const char *h, npy_meta *m) {
    memset(m, 0, sizeof(*m));

    /* ---- descr ('<f2' style: [byteorder] kind itemsize) ---- */
    const char *p = strstr(h, "'descr'");
    if (!p) return -1;
    p = strchr(p + 7, ':');
    if (!p) return -1;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    char quote = *p;
    if (quote != '\'' && quote != '"') return -1;
    p++;
    const char *dend = strchr(p, quote);
    if (!dend || dend == p) return -1;

    const char *d = p;
    if (*d == '<' || *d == '>' || *d == '=' || *d == '|') {
        if (*d == '>') return -1;                   /* big-endian unsupported */
        d++;
    }
    if (d >= dend) return -1;
    m->kind = *d++;
    if (d >= dend) return -1;
    int itemsize = 0;
    while (d < dend && *d >= '0' && *d <= '9') {
        itemsize = itemsize * 10 + (*d - '0');
        d++;
    }
    if (itemsize <= 0) return -1;
    m->itemsize = itemsize;

    /* ---- fortran_order ---- */
    const char *fp = strstr(h, "'fortran_order'");
    if (!fp) return -1;
    fp = strchr(fp + 15, ':');
    if (!fp) return -1;
    fp++;
    while (*fp == ' ' || *fp == '\t') fp++;
    if (strncmp(fp, "True", 4) == 0)        m->fortran = 1;
    else if (strncmp(fp, "False", 5) == 0)  m->fortran = 0;
    else return -1;

    /* ---- shape (tuple of decimal ints, possibly empty for 0-d) ---- */
    const char *s = strstr(h, "'shape'");
    if (!s) return -1;
    s = strchr(s + 7, ':');
    if (!s) return -1;
    s = strchr(s, '(');
    if (!s) return -1;
    s++;
    m->ndim  = 0;
    m->count = 1;
    while (*s && *s != ')') {
        while (*s == ' ' || *s == '\t' || *s == ',') s++;
        if (*s == ')' || *s == '\0') break;
        if (*s >= '0' && *s <= '9') {
            int64_t v = 0;
            while (*s >= '0' && *s <= '9') {
                v = v * 10 + (*s - '0');
                s++;
            }
            if (m->ndim >= 8) return -1;
            m->shape[m->ndim++] = v;
            m->count *= v;
        } else {
            return -1;                              /* unexpected token */
        }
    }
    if (*s != ')') return -1;
    return 0;
}

/* Read magic+version+header, fill *m, and (if data_off) report the byte offset
 * where raw data begins. Leaves the stream positioned at data start. */
static int npy_parse_header(FILE *f, npy_meta *m, long *data_off) {
    unsigned char magic[6];
    static const unsigned char ref[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
    if (fread(magic, 1, 6, f) != 6) return -1;
    if (memcmp(magic, ref, 6) != 0) return -1;

    unsigned major, minor;
    if (rd_u8(f, &major) || rd_u8(f, &minor)) return -1;

    size_t hlen;
    if (major == 1) {
        unsigned b0, b1;
        if (rd_u8(f, &b0) || rd_u8(f, &b1)) return -1;
        hlen = (size_t)b0 | ((size_t)b1 << 8);
    } else if (major >= 2) {
        unsigned b0, b1, b2, b3;
        if (rd_u8(f, &b0) || rd_u8(f, &b1) || rd_u8(f, &b2) || rd_u8(f, &b3)) return -1;
        hlen = (size_t)b0 | ((size_t)b1 << 8) | ((size_t)b2 << 16) | ((size_t)b3 << 24);
    } else {
        return -1;
    }

    char *hdr = (char *)malloc(hlen + 1);
    if (!hdr) return -1;
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); return -1; }
    hdr[hlen] = '\0';

    if (data_off) *data_off = ftell(f);
    int rc = npy_parse_dict(hdr, m);
    free(hdr);
    return rc;
}

int npy_stat(const char *path, npy_meta *out) {
    if (!path || !out) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int rc = npy_parse_header(f, out, NULL);
    fclose(f);
    return rc;
}

/* Which (kind,itemsize) pairs npy_read_f32 can convert. */
static int npy_supported(char kind, int isz) {
    switch (kind) {
        case 'f': return isz == 2 || isz == 4 || isz == 8;
        case 'i': return isz == 1 || isz == 2 || isz == 4 || isz == 8;
        case 'u': return isz == 1 || isz == 2 || isz == 4 || isz == 8;
        case 'b': return isz == 1;                  /* numpy bool */
        default:  return 0;
    }
}

/* Convert one little-endian element at p to float32. */
static float npy_elem_to_f32(const unsigned char *p, char kind, int isz) {
    switch (kind) {
        case 'f':
            if (isz == 2) { uint16_t v; memcpy(&v, p, 2); return npy_half_to_float(v); }
            if (isz == 4) { float    v; memcpy(&v, p, 4); return v; }
            if (isz == 8) { double   v; memcpy(&v, p, 8); return (float)v; }
            break;
        case 'i':
            if (isz == 1) { int8_t   v; memcpy(&v, p, 1); return (float)v; }
            if (isz == 2) { int16_t  v; memcpy(&v, p, 2); return (float)v; }
            if (isz == 4) { int32_t  v; memcpy(&v, p, 4); return (float)v; }
            if (isz == 8) { int64_t  v; memcpy(&v, p, 8); return (float)v; }
            break;
        case 'u':
        case 'b':
            if (isz == 1) { uint8_t  v; memcpy(&v, p, 1); return (float)v; }
            if (isz == 2) { uint16_t v; memcpy(&v, p, 2); return (float)v; }
            if (isz == 4) { uint32_t v; memcpy(&v, p, 4); return (float)v; }
            if (isz == 8) { uint64_t v; memcpy(&v, p, 8); return (float)v; }
            break;
        default:
            break;
    }
    return 0.0f; /* unreachable: callers gate on npy_supported() */
}

float *npy_read_f32(const char *path, npy_meta *out) {
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    npy_meta m;
    long data_off = 0;
    if (npy_parse_header(f, &m, &data_off) != 0) { fclose(f); return NULL; }
    if (m.fortran) { fclose(f); return NULL; }                 /* C-order only */
    if (!npy_supported(m.kind, m.itemsize)) { fclose(f); return NULL; }
    if (m.count < 0) { fclose(f); return NULL; }
    if ((uint64_t)m.count > (uint64_t)(SIZE_MAX / sizeof(float))) { fclose(f); return NULL; }

    float *buf = (float *)malloc((size_t)m.count * sizeof(float));
    if (!buf) { fclose(f); return NULL; }

    if (fseek(f, data_off, SEEK_SET) != 0) { free(buf); fclose(f); return NULL; }

    const size_t isz = (size_t)m.itemsize;
    unsigned char chunk[65536];
    size_t per = sizeof(chunk) / isz;               /* elements per read */
    if (per == 0) per = 1;

    int64_t i = 0;
    while (i < m.count) {
        int64_t remain = m.count - i;
        size_t want = (remain < (int64_t)per) ? (size_t)remain : per;
        size_t got = fread(chunk, isz, want, f);
        if (got != want) { free(buf); fclose(f); return NULL; }
        for (size_t k = 0; k < got; k++) {
            buf[i + (int64_t)k] = npy_elem_to_f32(chunk + k * isz, m.kind, m.itemsize);
        }
        i += (int64_t)got;
    }

    fclose(f);
    if (out) *out = m;
    return buf;
}

/* ======================================================================== */
#ifdef NPY_TEST
#include <math.h>

#define SCRATCH "/private/tmp/claude-501/-Users-fodelf-git-ds4-main/" \
                "24503593-c406-4203-a34b-b2d8ea433b47/scratchpad"

static int g_fail = 0;
#define CHECK(cond, msg) do {                                   \
        if (cond) { printf("  PASS: %s\n", msg); }              \
        else { printf("  FAIL: %s\n", msg); g_fail = 1; }       \
    } while (0)

/* Emit a v1.0 .npy by hand (magic + version + 2-byte len + padded header + data). */
static int write_npy(const char *path, const char *descr,
                     const int64_t *shape, int ndim,
                     const void *data, size_t nbytes) {
    char dict[256];
    int n = snprintf(dict, sizeof(dict),
                     "{'descr': '%s', 'fortran_order': False, 'shape': (", descr);
    for (int i = 0; i < ndim; i++)
        n += snprintf(dict + n, sizeof(dict) - n, "%lld, ", (long long)shape[i]);
    n += snprintf(dict + n, sizeof(dict) - n, "), }");

    /* pad with spaces so (10 + header_len) is a multiple of 64, header ends '\n' */
    int base = 10 + n + 1;
    int pad  = (64 - (base % 64)) % 64;
    for (int i = 0; i < pad && n < (int)sizeof(dict) - 1; i++) dict[n++] = ' ';
    dict[n++] = '\n';
    int headerlen = n;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    unsigned char hdr[8] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
    unsigned char hl[2]  = { (unsigned char)(headerlen & 0xff),
                             (unsigned char)((headerlen >> 8) & 0xff) };
    fwrite(hdr, 1, 8, f);
    fwrite(hl, 1, 2, f);
    fwrite(dict, 1, (size_t)headerlen, f);
    if (nbytes) fwrite(data, 1, nbytes, f);
    fclose(f);
    return 0;
}

int main(void) {
    printf("npy self-test\n");

    /* (1) round-trip a 2x3 float32 array, assert exact + shape */
    printf("[1] f32 2x3 round-trip\n");
    {
        const char *path = SCRATCH "/npy_test_f32.npy";
        float src[6] = { 1.5f, -2.0f, 3.25f, 4.0f, 5.5f, 6.75f };
        int64_t shape[2] = { 2, 3 };
        CHECK(write_npy(path, "<f4", shape, 2, src, sizeof(src)) == 0, "write f32 npy");

        npy_meta m;
        float *got = npy_read_f32(path, &m);
        CHECK(got != NULL, "read f32 npy");
        if (got) {
            CHECK(m.ndim == 2 && m.shape[0] == 2 && m.shape[1] == 3, "shape == (2,3)");
            CHECK(m.kind == 'f' && m.itemsize == 4, "kind 'f' itemsize 4");
            CHECK(m.count == 6, "count == 6");
            int exact = 1;
            for (int i = 0; i < 6; i++) if (got[i] != src[i]) exact = 0;
            CHECK(exact, "values exact");
            free(got);
        }
    }

    /* (2) fp16: 1.0=0x3C00, 2.0=0x4000, -1.5=0xBE00 -> ~equal (1e-3) */
    printf("[2] f16 known values\n");
    {
        const char *path = SCRATCH "/npy_test_f16.npy";
        uint16_t src[3] = { 0x3C00, 0x4000, 0xBE00 };
        float    exp[3] = { 1.0f, 2.0f, -1.5f };
        int64_t shape[1] = { 3 };
        CHECK(write_npy(path, "<f2", shape, 1, src, sizeof(src)) == 0, "write f16 npy");

        npy_meta m;
        float *got = npy_read_f32(path, &m);
        CHECK(got != NULL, "read f16 npy");
        if (got) {
            CHECK(m.ndim == 1 && m.shape[0] == 3 && m.count == 3, "shape == (3,)");
            CHECK(m.kind == 'f' && m.itemsize == 2, "kind 'f' itemsize 2");
            int ok = 1;
            for (int i = 0; i < 3; i++) if (fabsf(got[i] - exp[i]) > 1e-3f) ok = 0;
            CHECK(ok, "values ~equal (1e-3)");
            printf("    -> [%g, %g, %g]\n", got[0], got[1], got[2]);
            free(got);
        }
    }

    /* (3) stat the real (large) capture file — header only, NO data read.
     * External-data test: SKIP (not FAIL) when the NFS mount is absent or
     * unreadable — a self-test must not depend on another machine's export. */
    printf("[3] stat real ffn_in_L0.npy (header only)\n");
    {
        const char *path = "/private/tmp/m1_ds4/cap_m1/ffn_in_L0.npy";
        FILE *probe = fopen(path, "rb");
        if (!probe) {
            printf("  SKIP: %s not readable (NFS mount absent) — external-data test skipped\n", path);
            goto after_real_stat;
        }
        fclose(probe);
        npy_meta m;
        int rc = npy_stat(path, &m);
        CHECK(rc == 0, "npy_stat ok");
        if (rc == 0) {
            printf("    ndim=%d shape=[", m.ndim);
            for (int i = 0; i < m.ndim; i++)
                printf("%lld%s", (long long)m.shape[i], i + 1 < m.ndim ? "," : "");
            printf("] kind='%c' itemsize=%d fortran=%d count=%lld\n",
                   m.kind, m.itemsize, m.fortran, (long long)m.count);
            CHECK(m.ndim == 2 && m.shape[0] == 12288 && m.shape[1] == 4096,
                  "shape == [12288,4096]");
            CHECK(m.kind == 'f' && m.itemsize == 2, "kind 'f' itemsize 2");
        }
    }
after_real_stat:

    printf("%s\n", g_fail ? "RESULT: FAIL" : "RESULT: ALL PASS");
    return g_fail;
}
#endif /* NPY_TEST */
