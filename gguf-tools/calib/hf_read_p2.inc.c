/* Lazily read & parse the shard's JSON header only (never the data block). */
static int shard_load(st_shard *s) {
    if (s->loaded) return 0;
    FILE *fp = fopen(s->path, "rb");
    if (!fp) { fprintf(stderr, "hf_read: cannot open shard %s\n", s->path); return -1; }
    uint8_t hb[8];
    if (fread(hb, 1, 8, fp) != 8) { fclose(fp); return -1; }
    uint64_t hlen = 0;
    for (int i = 0; i < 8; i++) hlen |= (uint64_t)hb[i] << (8 * i);
    if (hlen == 0 || hlen > ((uint64_t)1 << 31)) { fclose(fp); return -1; } /* sanity */
    char *header = malloc((size_t)hlen + 1);
    if (!header) { fclose(fp); return -1; }
    if (fread(header, 1, (size_t)hlen, fp) != (size_t)hlen) { free(header); fclose(fp); return -1; }
    header[hlen] = '\0';
    s->data_base = 8 + hlen;
    if (parse_header(s, header, (size_t)hlen) != 0) { free(header); fclose(fp); return -1; }
    free(header);
    qsort(s->tensors, (size_t)s->n_tensors, sizeof(*s->tensors), cmp_sent);
    s->fp = fp;
    s->loaded = 1;
    return 0;
}

static int lookup_tensor(hf_db *db, const char *name, st_shard **s_out, st_info **info_out) {
    weight_ref keyw; keyw.name = (char *)name; keyw.shard_idx = 0;
    weight_ref *w = bsearch(&keyw, db->weights, (size_t)db->n_weights,
                            sizeof(*db->weights), cmp_wref);
    if (!w) return -1;
    st_shard *s = &db->shards[w->shard_idx];
    if (shard_load(s) != 0) return -1;
    st_entry keye; keye.name = (char *)name;
    st_entry *t = bsearch(&keye, s->tensors, (size_t)s->n_tensors,
                          sizeof(*s->tensors), cmp_sent);
    if (!t) return -1;
    if (s_out) *s_out = s;
    if (info_out) *info_out = &t->info;
    return 0;
}

/* fseek+fread ONLY this tensor's [begin,end) slice into a fresh buffer. */
static int read_tensor_raw(hf_db *db, const char *name, st_info *info_out,
                           uint8_t **data_out, size_t *nbytes_out) {
    st_shard *s = NULL; st_info *info = NULL;
    if (lookup_tensor(db, name, &s, &info) != 0) return -1;
    size_t nb = (size_t)(info->end - info->begin);
    uint8_t *buf = malloc(nb ? nb : 1);
    if (!buf) return -1;
    if (fseeko(s->fp, (off_t)(s->data_base + info->begin), SEEK_SET) != 0) { free(buf); return -1; }
    if (nb && fread(buf, 1, nb, s->fp) != nb) { free(buf); return -1; }
    if (info_out) *info_out = *info;
    *data_out = buf;
    *nbytes_out = nb;
    return 0;
}

/* name "...X.weight" -> "...X.scale".  Returns 0 on success. */
static int make_scale_name(const char *name, char *buf, size_t cap) {
    size_t L = strlen(name);
    const char *suf = ".weight";
    size_t ls = strlen(suf);
    if (L < ls || strcmp(name + L - ls, suf) != 0) return -1;
    if (L - ls + sizeof(".scale") > cap) return -1;
    memcpy(buf, name, L - ls);
    memcpy(buf + (L - ls), ".scale", sizeof(".scale"));  /* includes NUL */
    return 0;
}

/* ================================ public ================================ */

static char *read_whole_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseeko(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    off_t sz = ftello(fp);
    if (sz < 0 || fseeko(fp, 0, SEEK_SET) != 0) { fclose(fp); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    if ((off_t)fread(buf, 1, (size_t)sz, fp) != sz) { free(buf); fclose(fp); return NULL; }
    buf[sz] = '\0';
    fclose(fp);
    if (len_out) *len_out = (size_t)sz;
    return buf;
}

/* Parse index.json: top-level object, find "weight_map":{name:file}. */
static int parse_index(hf_db *db, const char *text, size_t len) {
    const char *p = text, *e = text + len;
    p = js_ws(p, e);
    if (p >= e || *p != '{') return -1;
    p++; p = js_ws(p, e);
    if (p < e && *p == '}') return -1;  /* no weight_map */
    for (;;) {
        char *key = js_str(&p, e);
        if (!key) return -1;
        p = js_ws(p, e);
        if (p >= e || *p != ':') { free(key); return -1; }
        p++;
        if (strcmp(key, "weight_map") == 0) {
            free(key);
            p = js_ws(p, e);
            if (p >= e || *p != '{') return -1;
            p++; p = js_ws(p, e);
            if (p < e && *p == '}') { p++; }
            else for (;;) {
                char *tname = js_str(&p, e);
                if (!tname) return -1;
                p = js_ws(p, e);
                if (p >= e || *p != ':') { free(tname); return -1; }
                p++; p = js_ws(p, e);
                char *fname = js_str(&p, e);
                if (!fname) { free(tname); return -1; }
                int si = db_find_or_create_shard(db, fname);
                free(fname);
                if (si < 0) { free(tname); return -1; }
                db_add_weight(db, tname, si);  /* takes ownership of tname */
                p = js_ws(p, e);
                if (p < e && *p == ',') { p++; p = js_ws(p, e); continue; }
                if (p < e && *p == '}') { p++; break; }
                return -1;
            }
        } else {
            free(key);
            const char *np = js_skip(p, e);
            if (!np) return -1;
            p = np;
        }
        p = js_ws(p, e);
        if (p < e && *p == ',') { p++; p = js_ws(p, e); continue; }
        if (p < e && *p == '}') break;
        return -1;
    }
    return 0;
}

hf_db *hf_open(const char *hf_dir) {
    if (!hf_dir) return NULL;
    hf_db *db = calloc(1, sizeof(*db));
    if (!db) return NULL;
    db->hf_dir = strdup(hf_dir);
    if (!db->hf_dir) { free(db); return NULL; }

    size_t la = strlen(hf_dir);
    const char *idx_rel = "model.safetensors.index.json";
    char *idx = malloc(la + 1 + strlen(idx_rel) + 1);
    if (!idx) { hf_close(db); return NULL; }
    memcpy(idx, hf_dir, la); idx[la] = '/'; strcpy(idx + la + 1, idx_rel);

    size_t len = 0;
    char *text = read_whole_file(idx, &len);
    free(idx);

    if (text) {
        int rc = parse_index(db, text, len);
        free(text);
        if (rc != 0) { hf_close(db); return NULL; }
    } else {
        /* Fallback: a single non-sharded model.safetensors. */
        int si = db_find_or_create_shard(db, "model.safetensors");
        if (si < 0 || shard_load(&db->shards[si]) != 0) { hf_close(db); return NULL; }
        st_shard *s = &db->shards[si];
        for (int i = 0; i < s->n_tensors; i++) {
            char *nm = strdup(s->tensors[i].name);
            if (!nm) { hf_close(db); return NULL; }
            db_add_weight(db, nm, si);
        }
    }

    if (db->n_weights == 0) { hf_close(db); return NULL; }
    qsort(db->weights, (size_t)db->n_weights, sizeof(*db->weights), cmp_wref);
    return db;
}

int hf_has(hf_db *db, const char *name) {
    if (!db || !name) return 0;
    weight_ref keyw; keyw.name = (char *)name; keyw.shard_idx = 0;
    return bsearch(&keyw, db->weights, (size_t)db->n_weights,
                   sizeof(*db->weights), cmp_wref) != NULL;
}

int hf_meta(hf_db *db, const char *name, int64_t shape_out[8], int *ndim_out,
            char dtype_out[16]) {
    if (!db || !name) return -1;
    st_info *info = NULL;
    if (lookup_tensor(db, name, NULL, &info) != 0) return -1;
    if (ndim_out) *ndim_out = info->ndim;
    if (shape_out) for (int i = 0; i < 8; i++) shape_out[i] = (i < info->ndim) ? info->shape[i] : 0;
    if (dtype_out) snprintf(dtype_out, 16, "%s", info->dtype);
    return 0;
}

float *hf_read_f32(hf_db *db, const char *name, int64_t *n_out) {
    if (!db || !name) return NULL;
    st_info info; uint8_t *data = NULL; size_t nbytes = 0;
    if (read_tensor_raw(db, name, &info, &data, &nbytes) != 0) return NULL;

    int64_t n = 1;
    for (int i = 0; i < info.ndim; i++) n *= info.shape[i];
    if (info.ndim == 0) n = 1;

    float *out = NULL;
    if (strcmp(info.dtype, "F32") == 0) {
        if (nbytes != (size_t)n * 4) goto fail;
        out = malloc((size_t)n * sizeof(float));
        if (!out) goto fail;
        for (int64_t i = 0; i < n; i++) out[i] = load_f32_le(data + (size_t)i * 4);
    } else if (strcmp(info.dtype, "F16") == 0) {
        if (nbytes != (size_t)n * 2) goto fail;
        out = malloc((size_t)n * sizeof(float));
        if (!out) goto fail;
        for (int64_t i = 0; i < n; i++) out[i] = f16_to_f32(load_u16_le(data + (size_t)i * 2));
    } else if (strcmp(info.dtype, "BF16") == 0) {
        if (nbytes != (size_t)n * 2) goto fail;
        out = malloc((size_t)n * sizeof(float));
        if (!out) goto fail;
        for (int64_t i = 0; i < n; i++) out[i] = bf16_to_f32(load_u16_le(data + (size_t)i * 2));
    } else if (strcmp(info.dtype, "F8_E4M3") == 0) {
        /* FP8 weight + 128x128 block scale (F32 in Base, F8_E8M0 in packed). */
        if (info.ndim != 2) goto fail;
        const int64_t out_dim = info.shape[0], in_dim = info.shape[1];
        if (nbytes != (size_t)out_dim * (size_t)in_dim) goto fail;
        if (out_dim % 128 || in_dim % 128) goto fail;

        char sname[512];
        if (make_scale_name(name, sname, sizeof(sname)) != 0) goto fail;
        st_info sinfo; uint8_t *sdata = NULL; size_t snb = 0;
        if (read_tensor_raw(db, sname, &sinfo, &sdata, &snb) != 0) goto fail;

        const int scale_f32 = (strcmp(sinfo.dtype, "F32") == 0);
        const int scale_e8  = (strcmp(sinfo.dtype, "F8_E8M0") == 0);
        const int64_t srows = out_dim / 128, scols = in_dim / 128;
        if ((!scale_f32 && !scale_e8) || sinfo.ndim != 2 ||
            sinfo.shape[0] != srows || sinfo.shape[1] != scols ||
            snb < (size_t)srows * (size_t)scols * (scale_f32 ? 4u : 1u)) {
            free(sdata); goto fail;
        }
        out = malloc((size_t)out_dim * (size_t)in_dim * sizeof(float));
        if (!out) { free(sdata); goto fail; }
        for (int64_t ob = 0; ob < srows; ob++) {
            for (int64_t ib = 0; ib < scols; ib++) {
                const size_t sidx = (size_t)ob * (size_t)scols + (size_t)ib;
                const float sc = scale_f32 ? load_f32_le(sdata + sidx * 4)
                                           : e8m0_to_f32(sdata[sidx]);
                for (int64_t r = 0; r < 128; r++) {
                    const int64_t row = ob * 128 + r;
                    const size_t base = (size_t)row * (size_t)in_dim + (size_t)ib * 128;
                    for (int64_t c = 0; c < 128; c++)
                        out[base + (size_t)c] = e4m3fn_to_f32(data[base + (size_t)c]) * sc;
                }
            }
        }
        free(sdata);
    } else {
        fprintf(stderr, "hf_read: unsupported dtype for f32 read: %s (%s)\n", info.dtype, name);
        goto fail;
    }

    free(data);
    if (n_out) *n_out = n;
    return out;

fail:
    free(data);
    free(out);
    return NULL;
}

void hf_close(hf_db *db) {
    if (!db) return;
    for (int i = 0; i < db->n_weights; i++) free(db->weights[i].name);
    free(db->weights);
    for (int i = 0; i < db->n_shards; i++) {
        st_shard *s = &db->shards[i];
        if (s->fp) fclose(s->fp);
        for (int j = 0; j < s->n_tensors; j++) free(s->tensors[j].name);
        free(s->tensors);
        free(s->file);
        free(s->path);
    }
    free(db->shards);
    free(db->hf_dir);
    free(db);
}

/* ================================ self-test ============================== */
#ifdef HFREAD_TEST
#include <unistd.h>

static int shard_num(const char *file) {
    const char *d = strstr(file, "model-");
    if (!d) return -1;
    d += 6;
    int v = 0, got = 0;
    while (*d >= '0' && *d <= '9') { v = v * 10 + (*d - '0'); d++; got = 1; }
    return got ? v : -1;
}

int main(void) {
    /* External-data test: probe repo-root hf/ then ../hf/ (make runs from
     * gguf-tools). SKIP (exit 0) when the checkpoint is absent — a self-test
     * must not fail on machines without the 87 GiB HF tree. */
    const char *dir = "hf/DeepSeek-V4-Flash-Base";
    hf_db *db = hf_open(dir);
    if (!db) {
        dir = "../hf/DeepSeek-V4-Flash-Base";
        db = hf_open(dir);
    }
    if (!db) { printf("SKIP: no HF checkpoint reachable — external-data test skipped\n"); return 0; }
    printf("opened %s: %d weights, %d shards\n", dir, db->n_weights, db->n_shards);

    /* --- pick a SMALL 1-D norm weight on a LOCAL shard (model-00001..00013) --- */
    const char *pick = NULL, *pickfile = NULL;
    const char *NS = "norm.weight";
    size_t NSL = strlen(NS);
    for (int i = 0; i < db->n_weights; i++) {
        const char *nm = db->weights[i].name;
        const char *file = db->shards[db->weights[i].shard_idx].file;
        int sn = shard_num(file);
        size_t L = strlen(nm);
        if (L >= NSL && strcmp(nm + L - NSL, NS) == 0 && sn >= 1 && sn <= 13) {
            int64_t shp[8]; int nd; char dt[16];
            if (hf_meta(db, nm, shp, &nd, dt) == 0 && nd == 1 && shp[0] <= (1 << 20)) {
                pick = nm; pickfile = file; break;
            }
        }
    }
    if (!pick) { fprintf(stderr, "no local 1-D norm tensor found\n"); hf_close(db); return 1; }

    int64_t shp[8]; int nd; char dt[16];
    hf_meta(db, pick, shp, &nd, dt);
    printf("\n[local 1-D tensor] %s  (shard %s)\n", pick, pickfile);
    printf("  hf_has=%d  meta: dtype=%s ndim=%d shape=[%lld]\n",
           hf_has(db, pick), dt, nd, (long long)shp[0]);

    int64_t n = 0;
    float *v = hf_read_f32(db, pick, &n);
    if (!v) { fprintf(stderr, "hf_read_f32 failed\n"); hf_close(db); return 1; }
    int64_t m = (n < 4096) ? n : 4096;
    double sum = 0; float mn = v[0], mx = v[0];
    for (int64_t i = 0; i < m; i++) { sum += v[i]; if (v[i] < mn) mn = v[i]; if (v[i] > mx) mx = v[i]; }
    printf("  read n=%lld  over first %lld: mean=%.6f min=%.6f max=%.6f\n",
           (long long)n, (long long)m, sum / (double)m, mn, mx);
    free(v);

    /* --- find a routed-expert weight; hf_meta only (NO data read) --- */
    const char *EXP = ".ffn.experts.";
    const char *first = NULL, *firstfile = NULL;
    const char *ex = NULL, *exfile = NULL;
    for (int i = 0; i < db->n_weights; i++) {
        const char *nm = db->weights[i].name;
        size_t L = strlen(nm);
        if (strstr(nm, EXP) && L >= 7 && strcmp(nm + L - 7, ".weight") == 0) {
            const char *file = db->shards[db->weights[i].shard_idx].file;
            if (!first) { first = nm; firstfile = file; }
            if (shard_num(file) >= 14) {  /* prefer a remote shard, if present */
                char pp[1024];
                snprintf(pp, sizeof(pp), "%s/%s", dir, file);
                if (access(pp, F_OK) == 0) { ex = nm; exfile = file; break; }
            }
        }
    }
    if (!ex) { ex = first; exfile = firstfile; }
    if (!ex) { fprintf(stderr, "no routed-expert weight found\n"); hf_close(db); return 1; }

    printf("\n[routed-expert weight] %s  (shard %s, num %d)\n", ex, exfile, shard_num(exfile));
    int64_t eshp[8]; int end_ = 0; char edt[16];
    if (hf_meta(db, ex, eshp, &end_, edt) == 0) {
        printf("  hf_meta (NO data read): dtype=%s ndim=%d shape=[", edt, end_);
        for (int i = 0; i < end_; i++) printf("%s%lld", i ? "," : "", (long long)eshp[i]);
        printf("]\n");
        char sn2[512];
        if (make_scale_name(ex, sn2, sizeof(sn2)) == 0 && hf_has(db, sn2)) {
            int64_t sshp[8]; int snd; char sdt[16];
            if (hf_meta(db, sn2, sshp, &snd, sdt) == 0) {
                printf("  companion scale: %s  dtype=%s ndim=%d shape=[", sn2, sdt, snd);
                for (int i = 0; i < snd; i++) printf("%s%lld", i ? "," : "", (long long)sshp[i]);
                printf("]\n");
            }
        }
    } else {
        printf("  hf_meta failed (shard likely not mounted): %s\n", exfile);
    }

    hf_close(db);
    printf("\nOK\n");
    return 0;
}
#endif /* HFREAD_TEST */
