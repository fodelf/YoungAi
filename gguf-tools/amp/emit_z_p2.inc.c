/* ============================================================ writer */

static void emit(const char *zdir, int layers, const char *out_path,
                 const char *in_path, size_t cli_align) {
    zlayer *zl = xmalloc((size_t)layers * sizeof(*zl));
    int d_model = 0, n_exp = 0;
    int n_present = scan_layers(zdir, layers, zl, &d_model, &n_exp);

    /* Flatten corr tensor descriptors (6 per present layer). */
    int n_corr = n_present * 6;
    corr_tensor *ct = xmalloc((size_t)n_corr * sizeof(*ct));
    for (int p = 0; p < n_present; p++) build_layer_tensors(&zl[p], d_model, n_exp, &ct[p * 6]);

    bool merge = (in_path != NULL);
    input_gguf in = {0};
    size_t align = cli_align;
    if (merge) {
        in = load_input(in_path);
        align = in.alignment;   /* keep base file's alignment */
    }
    if (align == 0) align = DEFAULT_ALIGN;

    /* Relative data-section offsets for corr tensors. In merge mode the base
     * tensors occupy [0, data_len); corr starts after that, aligned. */
    uint64_t off0 = merge ? pad_up(in.data_len, align) : 0;
    uint64_t roff = off0;
    for (int i = 0; i < n_corr; i++) {
        ct[i].offset = roff;
        roff += pad_up(ct[i].size, align);
    }

    FILE *f = fopen(out_path, "wb");
    if (!f) die_errno("open output", out_path);

    /* ---- header ---- */
    if (fwrite("GGUF", 1, 4, f) != 4) die("write magic");
    w_u32(f, merge ? in.version : GGUF_VERSION);
    uint64_t out_n_tensors = (merge ? in.n_tensors : 0) + (uint64_t)n_corr;
    /* corr KV count: present + d_model + n_expert + n_present + dl.{L}*n_present
     * (+ general.architecture only for standalone sidecar) (+ phi_yhat if set). */
    uint64_t corr_kv = 4 + (uint64_t)n_present + (merge ? 0 : 1) + (g_phi_yhat ? 1 : 0);
    uint64_t out_n_kv = (merge ? in.n_kv_kept : 0) + corr_kv;
    w_u64(f, out_n_tensors);
    w_u64(f, out_n_kv);

    /* ---- KVs ---- */
    if (merge) {
        if (in.kv_kept_len && fwrite(in.kv_kept, 1, in.kv_kept_len, f) != in.kv_kept_len)
            die("write kept KV");
    } else {
        w_kv_str(f, "general.architecture", "ds4-corr");
    }
    w_kv_bool(f, "ds4.corr.present", true);
    if (g_phi_yhat) w_kv_bool(f, "ds4.corr.phi_yhat", true); /* --feat yhat z: runtime binds routed_out as φ */
    w_kv_i32(f, "ds4.corr.d_model", d_model);
    w_kv_i32(f, "ds4.corr.n_expert", n_exp);
    w_kv_i32(f, "ds4.corr.n_present", n_present);
    for (int p = 0; p < n_present; p++) {
        char key[64];
        snprintf(key, sizeof(key), "ds4.corr.dl.%d", zl[p].L);
        w_kv_i32(f, key, zl[p].d_l);
    }

    /* ---- tensor infos ---- */
    if (merge) {
        if (in.tinfo_len && fwrite(in.tinfo_raw, 1, in.tinfo_len, f) != in.tinfo_len)
            die("write base tensor infos");
    }
    for (int i = 0; i < n_corr; i++) {
        w_str(f, ct[i].name);
        w_u32(f, (uint32_t)ct[i].n_dims);
        for (int j = 0; j < ct[i].n_dims; j++) w_u64(f, ct[i].ne[j]);
        w_u32(f, ct[i].type);
        w_u64(f, ct[i].offset);
    }

    /* ---- pad to data section ---- */
    off_t pos = ftello(f);
    if (pos < 0) die("ftell");
    size_t data_offset = pad_up((size_t)pos, align);
    write_padding(f, data_offset - (size_t)pos);

    /* ---- tensor data ---- */
    if (merge) {
        /* base data section verbatim, then pad to align before corr. */
        stream_copy(f, in.path, in.data_offset, in.data_len);
        write_padding(f, (size_t)(off0 - in.data_len));
    }
    for (int i = 0; i < n_corr; i++) {
        stream_copy(f, ct[i].src_path, ct[i].src_off, ct[i].size);
        write_padding(f, pad_up(ct[i].size, align) - (size_t)ct[i].size);
    }
    if (fclose(f) != 0) die_errno("close output", out_path);

    double corr_mib = 0;
    for (int i = 0; i < n_corr; i++) corr_mib += (double)ct[i].size;
    corr_mib /= 1048576.0;
    fprintf(stderr,
            "emit_z: wrote %s  mode=%s  layers_present=%d/%d  corr_tensors=%d  "
            "d_model=%d n_exp=%d  align=%zu  corr_data=%.2f MiB\n",
            out_path, merge ? "merge" : "sidecar", n_present, layers, n_corr,
            d_model, n_exp, align, corr_mib);
    if (merge) {
        free(in.kv_kept); free(in.tinfo_raw);
    }
    free(ct); free(zl);
}

/* ============================================================ verify (--check) */

typedef struct {
    char    *name;
    int      n_dims;
    uint64_t ne[4];
    uint32_t type;
    uint64_t offset;
    uint64_t size;   /* derived from offset deltas (type-agnostic) */
} tinfo;

typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    size_t   alignment;
    uint64_t data_offset;
    tinfo   *t;
    /* a couple corr KVs we look for */
    bool     corr_present;
    int      kv_dl[256];   /* indexed by layer L (up to 256), -1 = absent */
} parsed;

static parsed parse_full(const char *path) {
    parsed g = {0};
    for (int i = 0; i < 256; i++) g.kv_dl[i] = -1;
    FILE *f = fopen(path, "rb");
    if (!f) die_errno("open GGUF", path);
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) die("not a GGUF file");
    g.version = rd_u32(f, "version");
    g.n_tensors = rd_u64(f, "tensor count");
    g.n_kv = rd_u64(f, "kv count");
    g.alignment = DEFAULT_ALIGN;
    for (uint64_t i = 0; i < g.n_kv; i++) {
        char *key = rd_gguf_string(f);
        uint32_t type = rd_u32(f, "kv type");
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_TYPE_UINT32) {
            uint32_t a = rd_u32(f, "alignment"); if (a) g.alignment = a;
        } else if (strcmp(key, "ds4.corr.present") == 0 && type == GGUF_TYPE_BOOL) {
            uint8_t b; if (fread(&b, 1, 1, f) != 1) die("read bool"); g.corr_present = b != 0;
        } else if (strncmp(key, "ds4.corr.dl.", 12) == 0 && type == GGUF_TYPE_INT32) {
            int L = atoi(key + 12);
            int32_t v = (int32_t)rd_u32(f, "dl");
            if (L >= 0 && L < 256) g.kv_dl[L] = v;
        } else {
            skip_gguf_value(f, type, path);
        }
        free(key);
    }
    g.t = xmalloc((size_t)(g.n_tensors ? g.n_tensors : 1) * sizeof(*g.t));
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tinfo *t = &g.t[i];
        t->name = rd_gguf_string(f);
        t->n_dims = (int)rd_u32(f, "rank");
        if (t->n_dims < 1 || t->n_dims > 4) die("bad rank");
        for (int j = 0; j < t->n_dims; j++) t->ne[j] = rd_u64(f, "dim");
        t->type = rd_u32(f, "type");
        t->offset = rd_u64(f, "offset");
    }
    off_t meta_end = ftello(f);
    g.data_offset = pad_up((size_t)meta_end, g.alignment);
    fclose(f);
    /* derive each tensor size from offset deltas (sorted-by-offset assumption
     * holds for files this tool writes; fall back to file end for the last). */
    uint64_t fsz = file_size(path);
    uint64_t data_len = fsz - g.data_offset;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        uint64_t end = (i + 1 < g.n_tensors) ? g.t[i + 1].offset : data_len;
        g.t[i].size = end - g.t[i].offset;  /* padded size */
    }
    return g;
}

static tinfo *find_tensor(parsed *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (strcmp(g->t[i].name, name) == 0) return &g->t[i];
    return NULL;
}

/* read exactly len bytes at data_offset+rel into buf */
static void read_at(const char *path, uint64_t abs_off, void *buf, uint64_t len) {
    FILE *f = fopen(path, "rb");
    if (!f) die_errno("open", path);
    if (fseeko(f, (off_t)abs_off, SEEK_SET) != 0) die_errno("seek", path);
    if (len && fread(buf, 1, (size_t)len, f) != (size_t)len) die_errno("read", path);
    fclose(f);
}

static int check_cmd(const char *path, const char *zdir, int layers, const char *ref_path) {
    parsed g = parse_full(path);
    printf("== %s\n", path);
    printf("version=%u n_tensors=%" PRIu64 " n_kv=%" PRIu64 " alignment=%zu data_offset=%" PRIu64 "\n",
           g.version, g.n_tensors, g.n_kv, g.alignment, g.data_offset);
    printf("ds4.corr.present=%s\n", g.corr_present ? "true" : "false");
    int fail = 0;

    const char *parts[6] = { "corr_U", "corr_V", "corr_C", "corr_b", "corr_beta", "corr_delta" };
    int n_layers_found = 0;
    for (int L = 0; L < layers; L++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "blk.%d.corr_U", L);
        tinfo *u = find_tensor(&g, nm);
        if (!u) continue;
        n_layers_found++;
        int d_l_shape = (int)u->ne[0];
        int d_l_kv = (L < 256) ? g.kv_dl[L] : -1;
        printf("layer %d: d_l(shape U.ne[0])=%d  d_l(KV)=%d%s\n",
               L, d_l_shape, d_l_kv, (d_l_kv >= 0 && d_l_kv != d_l_shape) ? "  <<< MISMATCH" : "");
        if (d_l_kv >= 0 && d_l_kv != d_l_shape) fail = 1;
        for (int pi = 0; pi < 6; pi++) {
            snprintf(nm, sizeof(nm), "blk.%d.%s", L, parts[pi]);
            tinfo *t = find_tensor(&g, nm);
            if (!t) { printf("  MISSING %s\n", nm); fail = 1; continue; }
            printf("  %-22s nd=%d ne=[", nm, t->n_dims);
            for (int j = 0; j < t->n_dims; j++) printf("%s%" PRIu64, j ? "," : "", t->ne[j]);
            printf("] type=%u off=%" PRIu64 " size~%" PRIu64 "\n", t->type, t->offset, t->size);
            if (t->type != GGML_TYPE_F32) { printf("    <<< type != F32\n"); fail = 1; }
        }
    }
    printf("corr layers found: %d\n", n_layers_found);

    /* Round-trip corr DATA against z dumps, if available. */
    if (zdir) {
        zlayer *zl = xmalloc((size_t)layers * sizeof(*zl));
        int d_model = 0, n_exp = 0;
        int n_present = scan_layers(zdir, layers, zl, &d_model, &n_exp);
        for (int p = 0; p < n_present; p++) {
            zlayer *z = &zl[p];
            corr_tensor ct[6];
            build_layer_tensors(z, d_model, n_exp, ct);
            for (int i = 0; i < 6; i++) {
                tinfo *t = find_tensor(&g, ct[i].name);
                if (!t) { printf("roundtrip: %s absent in GGUF\n", ct[i].name); fail = 1; continue; }
                uint8_t *a = xmalloc(ct[i].size);
                uint8_t *b = xmalloc(ct[i].size);
                read_at(path, g.data_offset + t->offset, a, ct[i].size);
                read_at(ct[i].src_path, ct[i].src_off, b, ct[i].size);
                int ok = memcmp(a, b, (size_t)ct[i].size) == 0;
                if (!ok) { printf("roundtrip: %s DATA MISMATCH\n", ct[i].name); fail = 1; }
                free(a); free(b);
            }
            printf("roundtrip layer %d: %s\n", z->L, "data checked");
        }
        free(zl);
    }

    /* Compare original tensors against a reference GGUF (merge byte-identity). */
    if (ref_path) {
        parsed r = parse_full(ref_path);
        int compared = 0, mism = 0;
        for (uint64_t i = 0; i < r.n_tensors; i++) {
            tinfo *rt = &r.t[i];
            tinfo *ot = find_tensor(&g, rt->name);
            if (!ot) { printf("ref tensor '%s' MISSING in output\n", rt->name); fail = 1; continue; }
            /* compare unpadded-or-padded: use min of the two derived sizes */
            uint64_t n = rt->size < ot->size ? rt->size : ot->size;
            uint8_t *a = xmalloc(n ? n : 1);
            uint8_t *b = xmalloc(n ? n : 1);
            read_at(ref_path, r.data_offset + rt->offset, a, n);
            read_at(path, g.data_offset + ot->offset, b, n);
            if (memcmp(a, b, (size_t)n) != 0) { printf("ref tensor '%s' DATA MISMATCH\n", rt->name); mism++; fail = 1; }
            free(a); free(b);
            compared++;
        }
        printf("ref-compare: %d tensors compared, %d mismatched\n", compared, mism);
        for (uint64_t i = 0; i < r.n_tensors; i++) free(r.t[i].name);
        free(r.t);
    }

    for (uint64_t i = 0; i < g.n_tensors; i++) free(g.t[i].name);
    free(g.t);
    printf("CHECK: %s\n", fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

/* ============================================================ make-synth (test) */

/* Write a tiny valid GGUF with 3 small F32 tensors + a few KVs, for exercising
 * the merge path without the 45 GiB base model. Deterministic contents. */
static void make_synth(const char *path, size_t align) {
    if (align == 0) align = DEFAULT_ALIGN;
    struct { const char *name; int nd; uint64_t ne[2]; uint64_t n; } T[3] = {
        { "synth.a", 1, {4, 0}, 4 },
        { "synth.b", 2, {2, 3}, 6 },
        { "synth.c", 1, {8, 0}, 8 },
    };
    /* relative offsets */
    uint64_t off[3], roff = 0;
    for (int i = 0; i < 3; i++) { off[i] = roff; roff += pad_up(T[i].n * 4, align); }

    FILE *f = fopen(path, "wb");
    if (!f) die_errno("open", path);
    if (fwrite("GGUF", 1, 4, f) != 4) die("magic");
    w_u32(f, GGUF_VERSION);
    w_u64(f, 3);     /* n_tensors */
    w_u64(f, 2);     /* n_kv */
    w_kv_str(f, "general.architecture", "synthtest");
    { w_str(f, "general.alignment"); w_u32(f, GGUF_TYPE_UINT32); w_u32(f, (uint32_t)align); }
    for (int i = 0; i < 3; i++) {
        w_str(f, T[i].name);
        w_u32(f, (uint32_t)T[i].nd);
        for (int j = 0; j < T[i].nd; j++) w_u64(f, T[i].ne[j]);
        w_u32(f, GGML_TYPE_F32);
        w_u64(f, off[i]);
    }
    off_t pos = ftello(f);
    size_t data_off = pad_up((size_t)pos, align);
    write_padding(f, data_off - (size_t)pos);
    float seed = 1.0f;
    for (int i = 0; i < 3; i++) {
        for (uint64_t k = 0; k < T[i].n; k++) {
            float v = seed; seed += 1.0f;
            if (fwrite(&v, 4, 1, f) != 1) die("write synth data");
        }
        write_padding(f, pad_up(T[i].n * 4, align) - T[i].n * 4);
    }
    if (fclose(f) != 0) die_errno("close", path);
    fprintf(stderr, "emit_z: wrote synthetic GGUF %s (3 F32 tensors, align=%zu)\n", path, align);
}

/* ============================================================ CLI */

static void usage(void) {
    fprintf(stderr,
        "usage:\n"
        "  emit_z --out OUT.gguf [--zdir DIR] [--layers N] [--align N]\n"
        "         (sidecar: write ONLY corr tensors; no base model needed)\n"
        "  emit_z --in BASE.gguf --out OUT.gguf [--zdir DIR] [--layers N]\n"
        "         (merge: copy BASE verbatim, then append corr tensors)\n"
        "  emit_z --check FILE.gguf [--zdir DIR] [--layers N] [--ref REF.gguf]\n"
        "         (parse + verify corr tensors / z round-trip / ref byte-identity)\n"
        "  emit_z --make-synth OUT.gguf [--align N]   (tiny test GGUF)\n"
        "defaults: --zdir zdump  --layers 43  --align 32\n");
}

int main(int argc, char **argv) {
    const char *zdir = "zdump";
    const char *out_path = NULL;
    const char *in_path = NULL;
    const char *check_path = NULL;
    const char *ref_path = NULL;
    const char *synth_path = NULL;
    int layers = 43;
    size_t align = DEFAULT_ALIGN;
    bool zdir_set = false;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--zdir") && i + 1 < argc) { zdir = argv[++i]; zdir_set = true; }
        else if (!strcmp(a, "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(a, "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(a, "--check") && i + 1 < argc) check_path = argv[++i];
        else if (!strcmp(a, "--ref") && i + 1 < argc) ref_path = argv[++i];
        else if (!strcmp(a, "--make-synth") && i + 1 < argc) synth_path = argv[++i];
        else if (!strcmp(a, "--layers") && i + 1 < argc) layers = atoi(argv[++i]);
        else if (!strcmp(a, "--phi") && i + 1 < argc) g_phi_yhat = !strcmp(argv[++i], "yhat");
        else if (!strcmp(a, "--align") && i + 1 < argc) align = (size_t)atoi(argv[++i]);
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); return 0; }
        else { fprintf(stderr, "emit_z: unknown arg '%s'\n", a); usage(); return 2; }
    }
    if (layers < 1 || layers > 256) die("--layers out of range (1..256)");

    if (synth_path) { make_synth(synth_path, align); return 0; }
    if (check_path) return check_cmd(check_path, zdir_set ? zdir : zdir, layers, ref_path);
    if (!out_path) { usage(); return 2; }
    emit(zdir, layers, out_path, in_path, align);
    return 0;
}
