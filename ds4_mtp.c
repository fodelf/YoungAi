/* ds4_mtp.c -- knowledge-MTP: multi-token prediction from static language
 * knowledge. Self-contained: no engine, graph, tensor or backend includes;
 * the only bridge to the rest of ds4 is the tokenizer callback passed to
 * ds4_refcorpus_build(). See ds4_mtp.h for the module contract. */
#include "ds4_mtp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Reference corpus -- built-in language idioms.
 * =========================================================================
 *
 * Curation rule: each idiom is one VERBATIM run shaped as
 * [recognizable prefix][deterministic payload] -- the prefix is what the
 * suffix anchor matches, the payload is what gets drafted. Fragments whose
 * continuation is context-specific (identifiers, messages) stop where the
 * determinism stops. Indentation variants are separate entries because BPE
 * tokenizers fuse "\n\t+" runs into distinct tokens. */
static const char *const ref_idioms[] = {
    /* error handling -- the highest-frequency Go boilerplate.
     * Idioms sharing a prefix (the anchor ties at the shared prefix length)
     * are ordered by ASCENDING commonness: the matcher's tie-break is
     * "latest wins", so the most common continuation must come last. */
    "\t\tif err != nil {\n\t\t\treturn nil, err\n\t\t}\n",
    "\t\tif err != nil {\n\t\t\treturn err\n\t\t}\n",
    "\tif err != nil {\n\t\tt.Fatalf(\"unexpected error: %v\", err)\n\t}\n",
    "\tif err != nil {\n\t\tlog.Fatal(err)\n\t}\n",
    "\tif err != nil {\n\t\treturn fmt.Errorf(\"",
    "\tif err != nil {\n\t\treturn nil, err\n\t}\n",
    "\tif err != nil {\n\t\treturn err\n\t}\n",
    "\tif !ok {\n\t\treturn ",
    /* declarations / signatures */
    "package main\n\nimport (\n\t\"fmt\"\n",
    "func main() {\n\t",
    "(t *testing.T) {\n\t",
    ") error {\n\t",
    ") (string, error) {\n\t",
    ") ([]byte, error) {\n\t",
    " := make(map[string]",
    " := make([]",
    " := make(chan ",
    /* control flow */
    "\tfor i := 0; i < len(",
    "; i++ {\n\t\t",
    "\tfor _, v := range ",
    "\tfor k, v := range ",
    "\tfor i := range ",
    "\tswitch v := v.(type) {\n\tcase ",
    "\tdefault:\n\t\t",
    "\t\tcontinue\n\t}\n",
    "\t\tbreak\n\t}\n",
    "\treturn nil\n}\n\n",
    "\treturn true\n}\n\n",
    "\treturn false\n}\n\n",
    /* concurrency */
    "\tvar wg sync.WaitGroup\n\t",
    "\twg.Add(1)\n\tgo func() {\n\t\tdefer wg.Done()\n\t\t",
    "\twg.Wait()\n\t",
    "\tmu.Lock()\n\tdefer mu.Unlock()\n\t",
    "\ts.mu.Lock()\n\tdefer s.mu.Unlock()\n\t",
    "\tctx, cancel := context.WithTimeout(context.Background(), ",
    "\tdefer cancel()\n\t",
    "\tselect {\n\tcase <-ctx.Done():\n\t\treturn ctx.Err()\n\tcase ",
    /* common stdlib call shapes */
    "\tdefer f.Close()\n\t",
    "\tdefer resp.Body.Close()\n\t",
    "w http.ResponseWriter, r *http.Request) {\n\t",
    " := strconv.Itoa(",
    ", err := strconv.Atoi(",
    ", err := os.ReadFile(",
    ", err := json.Marshal(",
    "\terr := json.Unmarshal(data, &",
    " := fmt.Sprintf(\"",
    " := strings.Split(",
};

/* One token stream, idioms separated by -1. The separator can never equal a
 * real token, so anchors and copies stop at idiom boundaries naturally. */
struct ds4_refcorpus {
    int      *v;
    uint32_t  len;
};

static bool ref_append(ds4_mtp_tokenize_fn tokenize, void *tok_ctx,
                       const char *text, int **buf, uint32_t *len, uint32_t *cap) {
    int *toks = NULL;
    int n = 0;
    tokenize(tok_ctx, text, &toks, &n);
    if (!toks || n <= 0) { free(toks); return true; }
    while (*len + (uint32_t)n + 1u > *cap) {
        uint32_t ncap = *cap ? *cap * 2u : 1024u;
        int *nb = realloc(*buf, (size_t)ncap * sizeof(*nb));
        if (!nb) { free(toks); return false; }
        *buf = nb;
        *cap = ncap;
    }
    memcpy(*buf + *len, toks, (size_t)n * sizeof(**buf));
    *len += (uint32_t)n;
    (*buf)[(*len)++] = -1;
    free(toks);
    return true;
}

ds4_refcorpus *ds4_refcorpus_build(ds4_mtp_tokenize_fn tokenize, void *tok_ctx) {
    if (!tokenize) return NULL;
    int *buf = NULL;
    uint32_t len = 0, cap = 0;
    const size_t n_idioms = sizeof(ref_idioms) / sizeof(ref_idioms[0]);
    for (size_t i = 0; i < n_idioms; i++) {
        if (!ref_append(tokenize, tok_ctx, ref_idioms[i], &buf, &len, &cap)) {
            free(buf);
            return NULL;
        }
    }
    /* Extension files: plain-text snippets appended verbatim after the
     * built-ins (later entries win matcher ties => extensions override). */
    size_t n_files = 0;
    const char *ext = getenv("DS4_REF_CORPUS");
    if (ext && ext[0]) {
        char *paths = strdup(ext);
        for (char *p = paths ? strtok(paths, ":") : NULL; p; p = strtok(NULL, ":")) {
            FILE *fp = fopen(p, "rb");
            if (!fp) {
                fprintf(stderr, "ds4: ref-corpus: cannot open %s\n", p);
                continue;
            }
            fseek(fp, 0, SEEK_END);
            long sz = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            if (sz > 0 && sz <= 8 * 1024 * 1024) {   /* snippet files, not repos */
                char *text = malloc((size_t)sz + 1u);
                if (text && fread(text, 1, (size_t)sz, fp) == (size_t)sz) {
                    text[sz] = '\0';
                    if (ref_append(tokenize, tok_ctx, text, &buf, &len, &cap)) n_files++;
                } else {
                    fprintf(stderr, "ds4: ref-corpus: short read on %s\n", p);
                }
                free(text);
            } else {
                fprintf(stderr, "ds4: ref-corpus: %s empty or too large (>8MB)\n", p);
            }
            fclose(fp);
        }
        free(paths);
    }
    if (len == 0) { free(buf); return NULL; }
    ds4_refcorpus *rc = malloc(sizeof(*rc));
    if (!rc) { free(buf); return NULL; }
    rc->v = buf;
    rc->len = len;
    if (n_files || getenv("DS4_COPY_SPEC_LOG")) {
        fprintf(stderr, "ds4: ref-corpus: %u tokens (%zu built-in idioms, %zu extension files)\n",
                len, n_idioms, n_files);
    }
    return rc;
}

void ds4_refcorpus_free(ds4_refcorpus *rc) {
    if (!rc) return;
    free(rc->v);
    free(rc);
}

uint32_t ds4_refcorpus_match(const ds4_refcorpus *rc, const int *tail,
                             uint32_t len, uint32_t min_g, uint32_t cap,
                             int *out, uint32_t *out_n) {
    if (out_n) *out_n = 0;
    if (!rc || rc->len == 0 || !tail || !out || !out_n ||
        min_g == 0 || cap == 0 || len == 0) return 0;
    const uint32_t max_g = 32;           /* anchor extension cap (matches copy-spec) */
    const int *c = rc->v;
    const uint32_t clen = rc->len;
    const int last = tail[len - 1];
    uint32_t best_a = 0, best_src = 0;
    for (uint32_t s = 0; s < clen; s++) {
        if (c[s] != last) continue;      /* cheap last-token filter */
        uint32_t a = 1;
        /* -1 separators never equal a (non-negative) tail token, so the
         * anchor extension stops at idiom boundaries automatically. */
        while (a < max_g && a <= s && a < len && c[s - a] == tail[len - 1u - a]) a++;
        if (a >= best_a) { best_a = a; best_src = s + 1u; }   /* >=: latest wins */
    }
    if (best_a < min_g) return 0;
    uint32_t n = 0;
    while (n < cap && best_src + n < clen && c[best_src + n] >= 0) {
        out[n] = c[best_src + n];
        n++;
    }
    *out_n = n;
    return n > 0 ? best_a : 0;
}

/* =========================================================================
 * Statistical n-gram trie (GTRI file) -- corpus-statistics speculation.
 * =========================================================================
 *
 * Built offline from a real corpus (gguf-tools/go-onebit/mtp/
 * build_go_trie.py). It closes global boilerplate the reference corpus does
 * not curate, weighted by real-world frequency and gated by per-step
 * confidence so low-precision chains never reach the (expensive) verify
 * batch.
 *
 * File format v1 (little-endian): 32-byte header {magic "GTRI", version,
 * depth, min_count, n_nodes, root_off, root_n, n_vocab_hint} then n_nodes *
 * 16-byte nodes {i32 token; u32 count; u32 child_off; u32 child_n}. Child
 * blocks are contiguous and sorted by token id (binary search). count is the
 * corpus frequency of the path; sum(children) <= parent so the confidence
 * ratio child.count/parent.count is conservative. */

void ds4_gotrie_free(struct ds4_gotrie *t) {
    if (!t) return;
    free(t->nodes);
    free(t);
}

struct ds4_gotrie *ds4_gotrie_load(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: go-trie: cannot open %s\n", path);
        return NULL;
    }
    uint8_t hdr[32];
    struct ds4_gotrie *t = NULL;
    if (fread(hdr, 1, sizeof(hdr), fp) != sizeof(hdr) ||
        memcmp(hdr, "GTRI", 4) != 0) {
        fprintf(stderr, "ds4: go-trie: %s is not a GTRI file\n", path);
        fclose(fp);
        return NULL;
    }
    uint32_t u[7];
    memcpy(u, hdr + 4, sizeof(u));
    const uint32_t version = u[0], depth = u[1];
    const uint32_t n_nodes = u[3], root_off = u[4], root_n = u[5];
    if (version != 1 || depth < 2 || depth > 16 || n_nodes == 0 ||
        root_off > n_nodes || root_n > n_nodes - root_off) {
        fprintf(stderr, "ds4: go-trie: %s: bad header (version=%u depth=%u nodes=%u)\n",
                path, version, depth, n_nodes);
        fclose(fp);
        return NULL;
    }
    t = calloc(1, sizeof(*t));
    if (t) t->nodes = malloc((size_t)n_nodes * sizeof(t->nodes[0]));
    if (!t || !t->nodes ||
        fread(t->nodes, sizeof(t->nodes[0]), n_nodes, fp) != n_nodes) {
        fprintf(stderr, "ds4: go-trie: %s: truncated node array\n", path);
        fclose(fp);
        ds4_gotrie_free(t);
        return NULL;
    }
    fclose(fp);
    t->n_nodes = n_nodes;
    t->depth = depth;
    t->root_off = root_off;
    t->root_n = root_n;
    /* Validate every child block once so the walk can skip bounds checks:
     * in range, and sorted by token id for binary search. */
    for (uint32_t i = 0; i < n_nodes; i++) {
        const ds4_gotrie_node *nd = &t->nodes[i];
        if (nd->child_n == 0) continue;
        if (nd->child_off > n_nodes || nd->child_n > n_nodes - nd->child_off) {
            fprintf(stderr, "ds4: go-trie: %s: node %u child block out of range\n", path, i);
            ds4_gotrie_free(t);
            return NULL;
        }
        for (uint32_t c = 1; c < nd->child_n; c++) {
            if (t->nodes[nd->child_off + c - 1].token >= t->nodes[nd->child_off + c].token) {
                fprintf(stderr, "ds4: go-trie: %s: node %u children not sorted\n", path, i);
                ds4_gotrie_free(t);
                return NULL;
            }
        }
    }
    const char *env;
    /* Default confidence bar is deliberately high: on the A3 expert-streaming
     * path a K-token verify batch costs ~K expert gathers, so low-precision
     * chains (measured at 0.55: 1/5 and 2/3 accepted) are net losses. 0.80 per
     * step keeps only near-deterministic boilerplate. */
    t->conf = (env = getenv("DS4_GO_TRIE_CONF")) ? (float)atof(env) : 0.80f;
    t->min_cnt = (env = getenv("DS4_GO_TRIE_MINCNT")) ? (uint32_t)atoi(env) : 8;
    t->min_total = (env = getenv("DS4_GO_TRIE_MIN")) ? atoi(env) : 3;
    t->max_total = (env = getenv("DS4_GO_TRIE_MAX")) ? atoi(env) : 8;
    t->min_order = (env = getenv("DS4_GO_TRIE_MINORD")) ? atoi(env) : 2;
    if (t->min_total < 2) t->min_total = 2;
    if (t->max_total > 15) t->max_total = 15;   /* drafts[16] budget of the caller */
    if (t->max_total < t->min_total) t->max_total = t->min_total;
    if (t->min_order < 1) t->min_order = 1;
    t->log = getenv("DS4_GO_TRIE_LOG") != NULL;
    fprintf(stderr, "ds4: go-trie loaded: %s (%u nodes, depth %u, conf %.2f, propose %d..%d)\n",
            path, n_nodes, depth, (double)t->conf, t->min_total, t->max_total);
    return t;
}

/* Binary search `token` in the child block of `nd` (root when nd == NULL). */
static const ds4_gotrie_node *gotrie_child(const struct ds4_gotrie *t,
                                           const ds4_gotrie_node *nd, int token) {
    uint32_t lo = nd ? nd->child_off : t->root_off;
    uint32_t n  = nd ? nd->child_n   : t->root_n;
    uint32_t hi = lo + n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        int32_t mt = t->nodes[mid].token;
        if (mt == token) return &t->nodes[mid];
        if (mt < token) lo = mid + 1; else hi = mid;
    }
    return NULL;
}

/* One next-token prediction from the window w[0..wlen). Deepest-context-first;
 * backs off to a shorter context only when the longer one is absent/childless
 * (unseen), NOT when it is present but unconfident -- a confident shallow
 * prediction against a hesitant deep context is exactly the wasted-batch case
 * the confidence gate exists to prevent. Returns the token or -1. */
static int gotrie_predict(const struct ds4_gotrie *t, const int *w, int wlen) {
    int max_order = (int)t->depth - 1;
    if (max_order > wlen) max_order = wlen;
    for (int o = max_order; o >= t->min_order; o--) {
        const ds4_gotrie_node *nd = NULL;
        bool found = true;
        for (int i = 0; i < o; i++) {
            nd = gotrie_child(t, nd, w[wlen - o + i]);
            if (!nd) { found = false; break; }
        }
        if (!found || nd->child_n == 0) continue;   /* unseen -> back off */
        const ds4_gotrie_node *best = &t->nodes[nd->child_off];
        for (uint32_t c = 1; c < nd->child_n; c++) {
            if (t->nodes[nd->child_off + c].count > best->count)
                best = &t->nodes[nd->child_off + c];
        }
        if (best->count >= t->min_cnt &&
            (float)best->count >= t->conf * (float)nd->count)
            return best->token;
        return -1;   /* context known but genuinely uncertain -> stop the chain */
    }
    return -1;
}

uint32_t ds4_gotrie_propose(struct ds4_gotrie *t, const int *ctx,
                            uint32_t ctxlen, int must_first, int *out, int cap) {
    if (!t || cap < 1) return 0;
    int win[16];
    int wlen = 0;
    uint32_t wneed = t->depth - 1;
    uint32_t from = ctxlen > wneed ? ctxlen - wneed : 0;
    for (uint32_t i = from; i < ctxlen; i++) win[wlen++] = ctx[i];
    if (cap > t->max_total) cap = t->max_total;
    uint32_t n = 0;
    while ((int)n < cap) {
        int tok = gotrie_predict(t, win, wlen);
        if (tok < 0) break;
        if (n == 0 && tok != must_first) { t->gate_miss++; return 0; }
        out[n++] = tok;
        if (wlen == (int)wneed) {
            memmove(win, win + 1, (size_t)(wlen - 1) * sizeof(win[0]));
            wlen--;
        }
        win[wlen++] = tok;
    }
    return n >= (uint32_t)t->min_total ? n : 0;
}
