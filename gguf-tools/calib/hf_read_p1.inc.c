/* hf_read.c — minimal self-contained safetensors reader.  See hf_read.h.
 *
 * Shard layout (one .safetensors file):
 *   [u64 little-endian header_len][JSON header of header_len bytes][raw data]
 * Header JSON: name -> {"dtype":..,"shape":[..],"data_offsets":[begin,end]},
 * with offsets relative to the start of the data block (= 8 + header_len).
 * A sharded model adds model.safetensors.index.json: {"weight_map":{name:file}}.
 *
 * The dtype-conversion and FP8 128x128 block-dequant math below is replicated
 * from deepseek4-quantize.c (tensor_to_f32 / dequant_fp8_weight) so this module
 * stays standalone. */
#define _POSIX_C_SOURCE 200809L

#include "hf_read.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

/* ============================ data structures ============================ */

typedef struct {
    char     dtype[16];   /* "F16" | "BF16" | "F32" | "F8_E4M3" | "F8_E8M0" ... */
    int      ndim;
    int64_t  shape[8];
    uint64_t begin, end;  /* relative to the shard's data block */
} st_info;

typedef struct {
    char   *name;         /* owned */
    st_info info;
} st_entry;

typedef struct {
    char     *file;       /* shard filename (owned) */
    char     *path;       /* full path (owned) */
    int       loaded;
    FILE     *fp;         /* kept open once loaded */
    uint64_t  data_base;  /* 8 + header_len */
    st_entry *tensors;    /* sorted by name once loaded */
    int       n_tensors, cap_tensors;
} st_shard;

typedef struct {
    char *name;           /* tensor name (owned) */
    int   shard_idx;
} weight_ref;

struct hf_db {
    char       *hf_dir;
    weight_ref *weights;  /* sorted by name */
    int         n_weights, cap_weights;
    st_shard   *shards;
    int         n_shards, cap_shards;
};

/* =============================== numeric =============================== */

static float f32_from_bits(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

/* f16/bf16/e4m3/e8m0 标量: 换 src/common 共享实现(批2 收敛, 原地四份人肉副本清掉)。
 * 本地名保留成薄包装, 调用点零改动。 */
#include "../../src/common/ds4_float.h"
#include "../../src/common/ds4_fp8.h"

static float f16_to_f32(uint16_t bits)  { return ds4_f16_to_f32(bits); }
static float bf16_to_f32(uint16_t bits) { return ds4_bf16_to_f32(bits); }

/* ★E4M3 语义陷阱★共享库把 0x7f/0xff(NaN 槽)解成 NaN(IEEE 语义, 见 ds4_fp8.h 头注);
 * 本文件旧实现一直解成 0.0f —— 含 NaN 权重的读数是既定口径, 显式包一层保旧语义。 */
static float e4m3fn_to_f32(uint8_t x) {
    float v = ds4_e4m3fn_to_f32(x);
    return isnan(v) ? 0.0f : v;
}

static float e8m0_to_f32(uint8_t e) { return ds4_e8m0_to_f32(e); }

static uint16_t load_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static float load_f32_le(const uint8_t *p) {
    uint32_t b = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                 ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return f32_from_bits(b);
}

/* ============================ tiny JSON scan ============================ */
/* All scanners take (p,e); return the new cursor, or NULL on malformed input.
 * Safetensors headers are machine-generated and well-formed; we still bound
 * everything by `e` and handle string escapes when skipping. */

static const char *js_ws(const char *p, const char *e) {
    while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

/* Parse a JSON string at *pp (must point at '"'); return malloc'd unescaped,
 * NUL-terminated copy and advance *pp past the closing quote.  NULL on error. */
static char *js_str(const char **pp, const char *e) {
    const char *p = *pp;
    if (p >= e || *p != '"') return NULL;
    p++;
    size_t cap = 32, len = 0;
    char *out = malloc(cap);
    if (!out) return NULL;
    while (p < e && *p != '"') {
        char c = *p++;
        if (c == '\\') {
            if (p >= e) { free(out); return NULL; }
            char esc = *p++;
            switch (esc) {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'u': if (e - p < 4) { free(out); return NULL; } p += 4; c = '?'; break;
                default:  c = esc; break;  /* " \ / and anything else literal */
            }
        }
        if (len + 1 >= cap) {
            cap *= 2;
            char *n2 = realloc(out, cap);
            if (!n2) { free(out); return NULL; }
            out = n2;
        }
        out[len++] = c;
    }
    if (p >= e || *p != '"') { free(out); return NULL; }
    out[len] = '\0';
    *pp = p + 1;
    return out;
}

/* Skip one complete JSON value (string/number/bool/null/object/array). */
static const char *js_skip(const char *p, const char *e) {
    p = js_ws(p, e);
    if (p >= e) return NULL;
    if (*p == '"') {
        p++;
        while (p < e && *p != '"') { if (*p == '\\') { p++; if (p >= e) return NULL; } p++; }
        return (p < e) ? p + 1 : NULL;
    }
    if (*p == '{' || *p == '[') {
        char open = *p, close = (open == '{') ? '}' : ']';
        int depth = 0;
        while (p < e) {
            char c = *p;
            if (c == '"') {                       /* skip nested string */
                p++;
                while (p < e && *p != '"') { if (*p == '\\') { p++; if (p >= e) return NULL; } p++; }
                if (p >= e) return NULL;
                p++;
                continue;
            }
            if (c == open) depth++;
            else if (c == close) { depth--; if (depth == 0) return p + 1; }
            p++;
        }
        return NULL;
    }
    /* number / true / false / null */
    while (p < e && *p != ',' && *p != '}' && *p != ']' &&
           *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
    return p;
}

/* Parse a signed integer at p; store in *out; return cursor or NULL. */
static const char *js_int(const char *p, const char *e, int64_t *out) {
    p = js_ws(p, e);
    int neg = 0;
    if (p < e && *p == '-') { neg = 1; p++; }
    int64_t v = 0, got = 0;
    while (p < e && *p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; got = 1; }
    if (!got) return NULL;
    *out = neg ? -v : v;
    return p;
}

/* Parse one tensor entry object {"dtype":..,"shape":[..],"data_offsets":[a,b]}
 * starting at (or before, modulo ws) '{'.  Returns cursor after '}' or NULL. */
static const char *parse_tensor_obj(const char *p, const char *e, st_info *info) {
    memset(info, 0, sizeof(*info));
    p = js_ws(p, e);
    if (p >= e || *p != '{') return NULL;
    p++;
    p = js_ws(p, e);
    if (p < e && *p == '}') return p + 1;
    for (;;) {
        char *fk = js_str(&p, e);
        if (!fk) return NULL;
        p = js_ws(p, e);
        if (p >= e || *p != ':') { free(fk); return NULL; }
        p++; p = js_ws(p, e);
        int rc = 0;
        if (strcmp(fk, "dtype") == 0) {
            char *dt = js_str(&p, e);
            if (!dt) rc = -1;
            else { snprintf(info->dtype, sizeof(info->dtype), "%s", dt); free(dt); }
        } else if (strcmp(fk, "shape") == 0) {
            if (p >= e || *p != '[') rc = -1;
            else {
                p++; p = js_ws(p, e);
                if (p < e && *p == ']') p++;
                else for (;;) {
                    int64_t d; const char *np = js_int(p, e, &d);
                    if (!np) { rc = -1; break; }
                    p = np;
                    if (info->ndim < 8) info->shape[info->ndim++] = d;
                    p = js_ws(p, e);
                    if (p < e && *p == ',') { p++; p = js_ws(p, e); continue; }
                    if (p < e && *p == ']') { p++; break; }
                    rc = -1; break;
                }
            }
        } else if (strcmp(fk, "data_offsets") == 0) {
            if (p >= e || *p != '[') rc = -1;
            else {
                p++; p = js_ws(p, e);
                int64_t off[2] = {0, 0}; int no = 0;
                if (p < e && *p == ']') p++;
                else for (;;) {
                    int64_t d; const char *np = js_int(p, e, &d);
                    if (!np) { rc = -1; break; }
                    p = np;
                    if (no < 2) off[no] = d;
                    no++;
                    p = js_ws(p, e);
                    if (p < e && *p == ',') { p++; p = js_ws(p, e); continue; }
                    if (p < e && *p == ']') { p++; break; }
                    rc = -1; break;
                }
                info->begin = (uint64_t)off[0];
                info->end   = (uint64_t)off[1];
            }
        } else {
            const char *np = js_skip(p, e);
            if (!np) rc = -1; else p = np;
        }
        free(fk);
        if (rc) return NULL;
        p = js_ws(p, e);
        if (p < e && *p == ',') { p++; p = js_ws(p, e); continue; }
        if (p < e && *p == '}') return p + 1;
        return NULL;
    }
}

/* ============================== containers ============================== */

static void shard_add_tensor(st_shard *s, char *name /*owned*/, const st_info *info) {
    if (s->n_tensors == s->cap_tensors) {
        s->cap_tensors = s->cap_tensors ? s->cap_tensors * 2 : 256;
        s->tensors = realloc(s->tensors, (size_t)s->cap_tensors * sizeof(*s->tensors));
    }
    s->tensors[s->n_tensors].name = name;
    s->tensors[s->n_tensors].info = *info;
    s->n_tensors++;
}

static int db_find_or_create_shard(hf_db *db, const char *file) {
    for (int i = 0; i < db->n_shards; i++)
        if (strcmp(db->shards[i].file, file) == 0) return i;
    if (db->n_shards == db->cap_shards) {
        db->cap_shards = db->cap_shards ? db->cap_shards * 2 : 16;
        db->shards = realloc(db->shards, (size_t)db->cap_shards * sizeof(*db->shards));
    }
    st_shard *s = &db->shards[db->n_shards];
    memset(s, 0, sizeof(*s));
    s->file = strdup(file);
    size_t la = strlen(db->hf_dir), lb = strlen(file);
    s->path = malloc(la + 1 + lb + 1);
    memcpy(s->path, db->hf_dir, la);
    s->path[la] = '/';
    memcpy(s->path + la + 1, file, lb + 1);
    if (!s->file || !s->path) return -1;
    return db->n_shards++;
}

static void db_add_weight(hf_db *db, char *name /*owned*/, int shard_idx) {
    if (db->n_weights == db->cap_weights) {
        db->cap_weights = db->cap_weights ? db->cap_weights * 2 : 4096;
        db->weights = realloc(db->weights, (size_t)db->cap_weights * sizeof(*db->weights));
    }
    db->weights[db->n_weights].name = name;
    db->weights[db->n_weights].shard_idx = shard_idx;
    db->n_weights++;
}

static int cmp_wref(const void *a, const void *b) {
    return strcmp(((const weight_ref *)a)->name, ((const weight_ref *)b)->name);
}
static int cmp_sent(const void *a, const void *b) {
    return strcmp(((const st_entry *)a)->name, ((const st_entry *)b)->name);
}

/* ============================= shard loading ============================= */

static int parse_header(st_shard *s, const char *h, size_t hlen) {
    const char *p = h, *e = h + hlen;
    p = js_ws(p, e);
    if (p >= e || *p != '{') return -1;
    p++; p = js_ws(p, e);
    if (p < e && *p == '}') return 0;
    for (;;) {
        char *name = js_str(&p, e);
        if (!name) return -1;
        p = js_ws(p, e);
        if (p >= e || *p != ':') { free(name); return -1; }
        p++;
        if (strcmp(name, "__metadata__") == 0) {
            free(name);
            const char *np = js_skip(p, e);
            if (!np) return -1;
            p = np;
        } else {
            st_info info;
            const char *np = parse_tensor_obj(p, e, &info);
            if (!np) { free(name); return -1; }
            p = np;
            shard_add_tensor(s, name, &info);  /* takes ownership of name */
        }
        p = js_ws(p, e);
        if (p < e && *p == ',') { p++; p = js_ws(p, e); continue; }
        if (p < e && *p == '}') break;
        return -1;
    }
    return 0;
}

