/* min_go_check.c — minimal Go-domain fast-validation primitive (pure C99).
 *
 * Part of the Go-domain 1-bit quantizer tooling (SPEC.md §12 "最小 Go 快验证").
 * This is the inner-loop, seconds-scale judge that runs after every knob change:
 * it compares a CANDIDATE `ds4 --dump-logprobs` JSON file against the Q8
 * REFERENCE `ds4 --dump-logprobs` JSON file (same prompt, greedy / --temp 0) and
 * prints:
 *   (a) greedy-match%  — fraction of generation steps where the chosen (argmax /
 *                        "selected") token id agrees between candidate and ref;
 *   (b) mean per-token logprob divergence — a SYMMETRIC KL over the reported
 *                        top-k, averaged over the compared steps.
 *
 * It loads NO model and links NO external libraries; the JSON scanner below is
 * hand-rolled and structure-aware (it respects string/array/object boundaries,
 * so a token whose *text* happens to be e.g. "logprob" or "id" cannot fool it).
 *
 * ---------------------------------------------------------------------------
 * ASSUMED INPUT SCHEMA (reconciled against ds4_cli.c:run_logprob_dump() and
 * json_write_token(), as of this writing — see TODO below):
 *
 *   {
 *     "source":"ds4",
 *     "prompt_tokens":<int>,
 *     "ctx":<int>,
 *     "top_k":<int>,
 *     "steps":[
 *       {
 *         "step":<int>,
 *         "selected":{"id":<int>,"text":"<str>","bytes":[<u8>,...]},
 *         "top_logprobs":[
 *           {"token":{"id":<int>,"text":"<str>","bytes":[<u8>,...]},
 *            "logit":<float>,"logprob":<float>},
 *           ...                                  // 0..top_k entries, id>=0 only
 *         ]
 *       },
 *       ...
 *     ]
 *   }
 *
 * Notes on the schema as actually emitted by ds4_cli.c:
 *   - "selected" is the greedy/argmax token for that step (ds4_session_argmax),
 *     so it is the right field for greedy-match regardless of --temp.
 *   - top_logprobs is written in descending order and may contain FEWER than
 *     top_k entries (ds4 stops at the first score with id < 0).
 *   - "logprob" is assumed to be a natural-log probability (nats). If ds4 ever
 *     switches to log2/log10 the exp() reconstruction below would need a base
 *     change — but KL is computed from the same field in both files, so a shared
 *     base only rescales the divergence, it never breaks greedy-match.
 *   - The final step can be the EOS token (ds4 writes it, then breaks), so both
 *     files may legitimately differ in step count once greedy diverges.
 *
 * // TODO(integration): reconcile with ds4_cli.c --dump-logprobs writer.
 *   The field names above are read directly from the current writer; if that
 *   writer changes (e.g. renames "selected"/"top_logprobs"/"logprob", or nests
 *   the chosen id differently) update FIELD_* macros below.
 * ---------------------------------------------------------------------------
 *
 * Divergence definition (documented so the number is reproducible):
 *   For each compared step we form the UNION of token ids present in either
 *   file's top_logprobs. Each side's probability mass for a union token is
 *   exp(logprob) if the token is in that side's top-k, else exp(floor) where
 *   floor = that side's minimum top-k logprob (a conservative stand-in for the
 *   truncated tail). Each side is then renormalised over the union support so it
 *   is a proper distribution, and we accumulate the symmetric KL
 *   0.5*(KL(p||q)+KL(q||p)) in nats. The reported number is the mean over all
 *   steps that had a non-empty top-k on both sides. 0 means identical top-k.
 *
 * Build:   cc -std=c99 -O2 min_go_check.c -o min_go_check
 * Usage:   min_go_check [--min-match PCT] [--max-kl NATS] CAND.json REF.json
 *   Exit 0: parsed OK and (if thresholds given) thresholds met.
 *   Exit 1: I/O or parse error.
 *   Exit 2: thresholds given and NOT met (for use as a CI / inner-loop gate).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <errno.h>

/* Field names emitted by ds4_cli.c — change here if the writer changes. */
#define FIELD_STEPS    "steps"
#define FIELD_SELECTED "selected"
#define FIELD_TOPK     "top_logprobs"
#define FIELD_TOKEN    "token"
#define FIELD_ID       "id"
#define FIELD_LOGPROB  "logprob"

/* ------------------------------------------------------------------ */
/* Minimal, structure-aware JSON scanner (no DOM, no allocations).     */
/* All functions take [p,e) and return a pointer or NULL on malformed. */
/* ------------------------------------------------------------------ */

static const char *js_ws(const char *p, const char *e) {
    while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

/* p must point at the opening quote; returns just past the closing quote. */
static const char *js_string(const char *p, const char *e) {
    if (p >= e || *p != '"') return NULL;
    p++;
    while (p < e) {
        char c = *p++;
        if (c == '\\') {            /* skip the escaped char (covers \" \\ \/ \n */
            if (p < e) p++;         /* and the 'u' of \uXXXX; the 4 hex digits   */
        } else if (c == '"') {      /* are plain chars consumed by the loop)     */
            return p;
        }
    }
    return NULL;
}

/* p points at the start of any JSON value; returns just past it. */
static const char *js_value(const char *p, const char *e) {
    p = js_ws(p, e);
    if (p >= e) return NULL;
    char c = *p;
    if (c == '"') return js_string(p, e);
    if (c == '{' || c == '[') {
        char open = c, close = (c == '{') ? '}' : ']';
        int depth = 1;
        p++;
        while (p < e) {
            p = js_ws(p, e);
            if (p >= e) return NULL;
            c = *p;
            if (c == '"') { p = js_string(p, e); if (!p) return NULL; continue; }
            /* In well-formed JSON each bracket type is independently balanced,
             * so counting only this value's own bracket type is sufficient. */
            if (c == open)  { depth++; p++; continue; }
            if (c == close) { depth--; p++; if (depth == 0) return p; continue; }
            p++;
        }
        return NULL;
    }
    /* number / true / false / null: run to the next structural delimiter. */
    while (p < e) {
        c = *p;
        if (c == ',' || c == '}' || c == ']' ||
            c == ' ' || c == '\t' || c == '\n' || c == '\r') break;
        p++;
    }
    return p;
}

/* p points at (ws then) '{'. Returns a pointer to the VALUE of `key`, or NULL. */
static const char *js_get(const char *p, const char *e, const char *key) {
    p = js_ws(p, e);
    if (p >= e || *p != '{') return NULL;
    p++;
    size_t klen = strlen(key);
    while (1) {
        p = js_ws(p, e);
        if (p >= e || *p == '}') return NULL;
        if (*p != '"') return NULL;                 /* malformed */
        const char *ks = p;
        const char *ke = js_string(p, e);
        if (!ke) return NULL;
        size_t this_len = (size_t)((ke - 1) - (ks + 1)); /* between the quotes */
        bool match = (this_len == klen) && (memcmp(ks + 1, key, klen) == 0);
        p = js_ws(ke, e);
        if (p >= e || *p != ':') return NULL;
        p = js_ws(p + 1, e);
        if (match) return p;                        /* points at the value */
        p = js_value(p, e);                         /* skip non-matching value */
        if (!p) return NULL;
        p = js_ws(p, e);
        if (p < e && *p == ',') { p++; continue; }
        return NULL;                                /* '}' or EOF */
    }
}

/* Array iteration: arr_begin returns first element start (NULL if empty/bad);
 * arr_next skips the current element and returns the next, or NULL at the end. */
static const char *js_arr_begin(const char *p, const char *e) {
    p = js_ws(p, e);
    if (p >= e || *p != '[') return NULL;
    p = js_ws(p + 1, e);
    if (p < e && *p == ']') return NULL;            /* empty array */
    return p;
}
static const char *js_arr_next(const char *p, const char *e) {
    p = js_value(p, e);
    if (!p) return NULL;
    p = js_ws(p, e);
    if (p < e && *p == ',') return js_ws(p + 1, e);
    return NULL;                                    /* ']' or EOF */
}

static bool js_int(const char *p, const char *e, long *out) {
    p = js_ws(p, e);
    char *end = NULL;
    errno = 0;
    long v = strtol(p, &end, 10);
    if (end == p) return false;
    (void)e;
    *out = v;
    return true;
}
static bool js_double(const char *p, const char *e, double *out) {
    p = js_ws(p, e);
    char *end = NULL;
    double v = strtod(p, &end);
    if (end == p) return false;
    (void)e;
    *out = v;
    return true;
}

/* ------------------------------------------------------------------ */
/* In-memory representation of a parsed dump.                          */
/* ------------------------------------------------------------------ */

typedef struct {
    long    sel_id;     /* greedy / "selected" token id (-1 if absent) */
    int     n_top;      /* number of top_logprobs entries              */
    long   *ids;        /* token ids (length n_top)                    */
    double *lp;         /* logprobs  (length n_top)                    */
} Step;

typedef struct {
    int   n_steps;
    Step *steps;
    long  top_k;        /* reported top_k (informational; -1 if absent) */
} Dump;

static char *read_whole_file(const char *path, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "min_go_check: cannot open %s: %s\n", path, strerror(errno)); return NULL; }
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); fprintf(stderr, "min_go_check: seek failed on %s\n", path); return NULL; }
    long sz = ftell(fp);
    if (sz < 0) { fclose(fp); fprintf(stderr, "min_go_check: tell failed on %s\n", path); return NULL; }
    rewind(fp);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); fprintf(stderr, "min_go_check: out of memory\n"); return NULL; }
    size_t got = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[got] = '\0';            /* NUL-terminate so strtol/strtod stop cleanly */
    *out_len = got;
    return buf;
}

static void free_dump(Dump *d) {
    if (!d->steps) return;
    for (int i = 0; i < d->n_steps; i++) {
        free(d->steps[i].ids);
        free(d->steps[i].lp);
    }
    free(d->steps);
    d->steps = NULL;
    d->n_steps = 0;
}

/* Parse one step object (pointer at '{') into *st. Returns false on malformed. */
static bool parse_step(const char *sp, const char *e, Step *st) {
    st->sel_id = -1;
    st->n_top = 0;
    st->ids = NULL;
    st->lp = NULL;

    const char *sel = js_get(sp, e, FIELD_SELECTED);
    if (sel) {
        const char *idv = js_get(sel, e, FIELD_ID);
        long id;
        if (idv && js_int(idv, e, &id)) st->sel_id = id;
    }

    const char *top = js_get(sp, e, FIELD_TOPK);
    if (top) {
        /* First pass: count entries so we can allocate exactly. */
        int cnt = 0;
        for (const char *it = js_arr_begin(top, e); it; it = js_arr_next(it, e)) cnt++;
        if (cnt > 0) {
            st->ids = (long *)malloc((size_t)cnt * sizeof(long));
            st->lp  = (double *)malloc((size_t)cnt * sizeof(double));
            if (!st->ids || !st->lp) { free(st->ids); free(st->lp); return false; }
            int n = 0;
            for (const char *it = js_arr_begin(top, e); it; it = js_arr_next(it, e)) {
                const char *tok = js_get(it, e, FIELD_TOKEN);
                const char *idv = tok ? js_get(tok, e, FIELD_ID) : NULL;
                const char *lpv = js_get(it, e, FIELD_LOGPROB);
                long id; double lp;
                if (idv && js_int(idv, e, &id) && lpv && js_double(lpv, e, &lp)) {
                    st->ids[n] = id;
                    st->lp[n]  = lp;
                    n++;
                }
            }
            st->n_top = n;
        }
    }
    /* If "selected" was absent but we have a top-1, fall back to it. */
    if (st->sel_id < 0 && st->n_top > 0) st->sel_id = st->ids[0];
    return true;
}

static bool parse_dump(const char *path, Dump *d) {
    d->n_steps = 0; d->steps = NULL; d->top_k = -1;
    size_t len = 0;
    char *buf = read_whole_file(path, &len);
    if (!buf) return false;
    const char *e = buf + len;

    const char *tkv = js_get(buf, e, "top_k");
    long tk;
    if (tkv && js_int(tkv, e, &tk)) d->top_k = tk;

    const char *steps = js_get(buf, e, FIELD_STEPS);
    if (!steps) {
        fprintf(stderr, "min_go_check: no \"%s\" array in %s\n", FIELD_STEPS, path);
        free(buf);
        return false;
    }

    int cap = 0, n = 0;
    for (const char *it = js_arr_begin(steps, e); it; it = js_arr_next(it, e)) {
        if (n == cap) {
            cap = cap ? cap * 2 : 16;
            Step *grown = (Step *)realloc(d->steps, (size_t)cap * sizeof(Step));
            if (!grown) { free(buf); free_dump(d); return false; }
            d->steps = grown;
        }
        if (!parse_step(it, e, &d->steps[n])) { free(buf); free_dump(d); return false; }
        n++;
    }
    d->n_steps = n;
    free(buf);
    return true;
}

/* ------------------------------------------------------------------ */
/* Metrics.                                                            */
/* ------------------------------------------------------------------ */

static double step_lp(const Step *s, long id, double floor_lp) {
    for (int i = 0; i < s->n_top; i++) if (s->ids[i] == id) return s->lp[i];
    return floor_lp;            /* truncated-tail stand-in */
}

static double min_lp(const Step *s) {
    double m = s->lp[0];
    for (int i = 1; i < s->n_top; i++) if (s->lp[i] < m) m = s->lp[i];
    return m;
}

/* Symmetric KL (nats) between the two top-k distributions for one step,
 * over the union support, each side renormalised. Returns -1 to signal "skip"
 * (a side had an empty top-k). */
static double sym_kl_step(const Step *a, const Step *b) {
    if (a->n_top == 0 || b->n_top == 0) return -1.0;

    /* Build the union of token ids. */
    int cap = a->n_top + b->n_top;
    long *u = (long *)malloc((size_t)cap * sizeof(long));
    if (!u) return -1.0;
    int nu = 0;
    for (int i = 0; i < a->n_top; i++) {
        bool seen = false;
        for (int j = 0; j < nu; j++) if (u[j] == a->ids[i]) { seen = true; break; }
        if (!seen) u[nu++] = a->ids[i];
    }
    for (int i = 0; i < b->n_top; i++) {
        bool seen = false;
        for (int j = 0; j < nu; j++) if (u[j] == b->ids[i]) { seen = true; break; }
        if (!seen) u[nu++] = b->ids[i];
    }

    double fa = min_lp(a), fb = min_lp(b);
    double *pa = (double *)malloc((size_t)nu * sizeof(double));
    double *pb = (double *)malloc((size_t)nu * sizeof(double));
    if (!pa || !pb) { free(u); free(pa); free(pb); return -1.0; }

    double sa = 0.0, sb = 0.0;
    const double TINY = 1e-300;
    for (int i = 0; i < nu; i++) {
        double xa = exp(step_lp(a, u[i], fa));
        double xb = exp(step_lp(b, u[i], fb));
        if (!(xa > 0.0)) xa = TINY;
        if (!(xb > 0.0)) xb = TINY;
        pa[i] = xa; sa += xa;
        pb[i] = xb; sb += xb;
    }
    double kl = 0.0;
    for (int i = 0; i < nu; i++) {
        double p = pa[i] / sa;          /* candidate */
        double q = pb[i] / sb;          /* reference */
        if (p > 0.0 && q > 0.0) kl += 0.5 * (p * log(p / q) + q * log(q / p));
    }
    free(u); free(pa); free(pb);
    return kl;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s [--min-match PCT] [--max-kl NATS] CANDIDATE.json REFERENCE.json\n"
        "  Compares two `ds4 --dump-logprobs` files (same prompt, greedy).\n"
        "  Prints greedy-match%% and mean symmetric-KL (nats) over the top-k.\n"
        "  With --min-match / --max-kl, exits 2 if a threshold is not met.\n",
        argv0);
}

int main(int argc, char **argv) {
    double min_match = -1.0;   /* disabled */
    double max_kl    = -1.0;   /* disabled */
    const char *cand = NULL, *ref = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--min-match") && i + 1 < argc) {
            min_match = strtod(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--max-kl") && i + 1 < argc) {
            max_kl = strtod(argv[++i], NULL);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]); return 0;
        } else if (!cand) {
            cand = argv[i];
        } else if (!ref) {
            ref = argv[i];
        } else {
            fprintf(stderr, "min_go_check: unexpected argument: %s\n", argv[i]);
            usage(argv[0]); return 1;
        }
    }
    if (!cand || !ref) { usage(argv[0]); return 1; }

    Dump dc, dr;
    if (!parse_dump(cand, &dc)) return 1;
    if (!parse_dump(ref, &dr)) { free_dump(&dc); return 1; }

    int n = dc.n_steps < dr.n_steps ? dc.n_steps : dr.n_steps;

    /* (a) greedy match over aligned positions. */
    int match = 0;
    for (int i = 0; i < n; i++)
        if (dc.steps[i].sel_id == dr.steps[i].sel_id) match++;
    double match_pct = (n > 0) ? 100.0 * (double)match / (double)n : 0.0;

    /* (b) mean symmetric KL over compared steps with non-empty top-k. */
    double kl_sum = 0.0;
    int kl_steps = 0;
    for (int i = 0; i < n; i++) {
        double k = sym_kl_step(&dc.steps[i], &dr.steps[i]);
        if (k >= 0.0) { kl_sum += k; kl_steps++; }
    }
    double kl_mean = (kl_steps > 0) ? kl_sum / (double)kl_steps : 0.0;

    printf("candidate      : %s (%d steps)\n", cand, dc.n_steps);
    printf("reference      : %s (%d steps)\n", ref, dr.n_steps);
    printf("positions_cmp  : %d\n", n);
    printf("greedy_match   : %d/%d (%.2f%%)\n", match, n, match_pct);
    printf("mean_sym_kl    : %.6f nats (over %d top-k steps)\n", kl_mean, kl_steps);

    int rc = 0;
    if (min_match >= 0.0 && match_pct < min_match) {
        printf("GATE: FAIL greedy_match %.2f%% < --min-match %.2f%%\n", match_pct, min_match);
        rc = 2;
    }
    if (max_kl >= 0.0 && kl_mean > max_kl) {
        printf("GATE: FAIL mean_sym_kl %.6f > --max-kl %.6f\n", kl_mean, max_kl);
        rc = 2;
    }
    if (rc == 0 && (min_match >= 0.0 || max_kl >= 0.0))
        printf("GATE: PASS\n");

    free_dump(&dc);
    free_dump(&dr);
    return rc;
}
