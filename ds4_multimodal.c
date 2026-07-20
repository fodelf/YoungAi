/* ds4_multimodal.c -- 多模态: modality-adapter registry. Self-contained;
 * see ds4_multimodal.h for the module contract. */
#include "ds4_multimodal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Text form of an encoder (see header: canonical for text-pipeline
 * consumers). NULL when the encoder is token-only. */
typedef int (*mm_encode_text_fn)(void *ctx, const uint8_t *data, size_t len,
                                 char **out_text);

typedef struct {
    char             *name;    /* modality key, e.g. "text", "image/png" */
    ds4_mm_encode_fn  encode;
    mm_encode_text_fn encode_text;
    void             *ctx;
} mm_entry;

typedef struct {
    char              *name;   /* modality family key, e.g. "image" */
    ds4_mm_enrich_fn   fn;
    void              *ctx;
} mm_enricher;

struct ds4_mm {
    mm_entry *v;
    size_t    n;
    size_t    cap;
    mm_enricher *er;
    size_t       er_n;
    size_t       er_cap;
    /* built-in text modality state */
    ds4_mm_tokenize_fn tokenize;
    void              *tok_ctx;
};

static int mm_register_full(ds4_mm *mm, const char *modality,
                            ds4_mm_encode_fn encode,
                            mm_encode_text_fn encode_text, void *ctx);

/* Built-in text modality: the text form is the identity; the token form is
 * derived from it by ds4_mm_encode (text form -> enrichers -> tokenizer). */
static int mm_text_encode_text(void *vmm, const uint8_t *data, size_t len,
                               char **out_text) {
    (void)vmm;
    char *text = malloc(len + 1u);
    if (!text) return -1;
    memcpy(text, data, len);
    text[len] = '\0';
    *out_text = text;
    return 0;
}

ds4_mm *ds4_mm_create(ds4_mm_tokenize_fn tokenize, void *tok_ctx) {
    ds4_mm *mm = calloc(1, sizeof(*mm));
    if (!mm) return NULL;
    mm->tokenize = tokenize;
    mm->tok_ctx = tok_ctx;
    if (mm_register_full(mm, "text", NULL, mm_text_encode_text, mm) != 0) {
        ds4_mm_free(mm);
        return NULL;
    }
    return mm;
}

void ds4_mm_free(ds4_mm *mm) {
    if (!mm) return;
    for (size_t i = 0; i < mm->n; i++) free(mm->v[i].name);
    free(mm->v);
    for (size_t i = 0; i < mm->er_n; i++) free(mm->er[i].name);
    free(mm->er);
    free(mm);
}

static int mm_register_full(ds4_mm *mm, const char *modality,
                            ds4_mm_encode_fn encode,
                            mm_encode_text_fn encode_text, void *ctx) {
    if (!mm || !modality || !modality[0] || (!encode && !encode_text)) return -1;
    for (size_t i = 0; i < mm->n; i++) {
        if (strcmp(mm->v[i].name, modality) == 0) {   /* replace */
            mm->v[i].encode = encode;
            mm->v[i].encode_text = encode_text;
            mm->v[i].ctx = ctx;
            return 0;
        }
    }
    if (mm->n == mm->cap) {
        size_t ncap = mm->cap ? mm->cap * 2u : 4u;
        mm_entry *nv = realloc(mm->v, ncap * sizeof(*nv));
        if (!nv) return -1;
        mm->v = nv;
        mm->cap = ncap;
    }
    char *name = strdup(modality);
    if (!name) return -1;
    mm->v[mm->n].name = name;
    mm->v[mm->n].encode = encode;
    mm->v[mm->n].encode_text = encode_text;
    mm->v[mm->n].ctx = ctx;
    mm->n++;
    return 0;
}

int ds4_mm_register(ds4_mm *mm, const char *modality,
                    ds4_mm_encode_fn encode, void *ctx) {
    if (!encode) return -1;
    return mm_register_full(mm, modality, encode, NULL, ctx);
}

/* Family-prefix match: exact name, or a prefix of `modality` ending at a '/'
 * boundary ("image" matches "image/png"). */
static int mm_name_matches(const char *name, const char *modality) {
    const size_t elen = strlen(name);
    const size_t mlen = strlen(modality);
    if (elen > mlen) return 0;
    if (strncmp(name, modality, elen) != 0) return 0;
    return elen == mlen || modality[elen] == '/';
}

/* Longest-prefix resolution: exact name first, then the longest registered
 * entry that prefixes `modality` at a '/' boundary ("image/png" -> "image").
 * This is what lets one encoder own a whole family without enumeration. */
static const mm_entry *mm_resolve(const ds4_mm *mm, const char *modality) {
    if (!mm || !modality || !modality[0]) return NULL;
    const mm_entry *best = NULL;
    size_t best_len = 0;
    for (size_t i = 0; i < mm->n; i++) {
        if (!mm_name_matches(mm->v[i].name, modality)) continue;
        const size_t elen = strlen(mm->v[i].name);
        if (elen >= best_len) { best = &mm->v[i]; best_len = elen; }
    }
    return best;
}

int ds4_mm_register_enricher(ds4_mm *mm, const char *modality,
                             ds4_mm_enrich_fn fn, void *ctx) {
    if (!mm || !modality || !modality[0] || !fn) return -1;
    if (mm->er_n == mm->er_cap) {
        size_t ncap = mm->er_cap ? mm->er_cap * 2u : 4u;
        mm_enricher *nv = realloc(mm->er, ncap * sizeof(*nv));
        if (!nv) return -1;
        mm->er = nv;
        mm->er_cap = ncap;
    }
    char *name = strdup(modality);
    if (!name) return -1;
    mm->er[mm->er_n].name = name;
    mm->er[mm->er_n].fn = fn;
    mm->er[mm->er_n].ctx = ctx;
    mm->er_n++;
    return 0;
}

/* Run every matching enricher over the ORIGINAL encoder text and append the
 * returned sections in registration order. Takes ownership of `base`;
 * returns the enriched text (== base when nothing matched/added). Enricher
 * failures (NULL) mean "no section", never an error: the encoder's own
 * facts already stand on their own. `base` stays alive until all enrichers
 * ran -- each one must see the untouched encoder text, not a peer's output. */
static char *mm_apply_enrichers(const ds4_mm *mm, const char *modality,
                                char *base) {
    if (!mm->er_n) return base;
    const size_t base_len = strlen(base);
    char  *out = NULL;      /* allocated on the first non-empty section */
    size_t len = 0, cap = 0;
    for (size_t i = 0; i < mm->er_n; i++) {
        if (!mm_name_matches(mm->er[i].name, modality)) continue;
        char *sec = mm->er[i].fn(mm->er[i].ctx, modality, base);
        if (!sec || !sec[0]) { free(sec); continue; }
        const size_t slen = strlen(sec);
        const size_t need = (out ? len : base_len) + 1u /* \n */ + slen + 1u;
        if (need > cap) {
            cap = need * 2u;
            char *nb = realloc(out, cap);
            if (!nb) { free(sec); break; }
            if (!out) { memcpy(nb, base, base_len + 1u); len = base_len; }
            out = nb;
        }
        if (len == 0 || out[len - 1u] != '\n') out[len++] = '\n';
        memcpy(out + len, sec, slen + 1u);
        len += slen;
        free(sec);
    }
    if (!out) return base;
    free(base);
    return out;
}

int ds4_mm_encode(const ds4_mm *mm, const char *modality,
                  const uint8_t *data, size_t len, int **toks, int *n_toks) {
    if (toks) *toks = NULL;
    if (n_toks) *n_toks = 0;
    if (!toks || !n_toks || !data) return -1;
    const mm_entry *e = mm_resolve(mm, modality);
    if (!e) return -1;                            /* unsupported: fail cleanly */
    /* Text-form encoders tokenize their ENRICHED text, so both output forms
     * carry the same facts. Token-only encoders pass through untouched. */
    if (e->encode_text && mm->tokenize) {
        char *text = NULL;
        if (e->encode_text(e->ctx, data, len, &text) != 0 || !text) return -1;
        text = mm_apply_enrichers(mm, modality, text);
        mm->tokenize(mm->tok_ctx, text, toks, n_toks);
        free(text);
        return (*toks && *n_toks > 0) ? 0 : -1;
    }
    if (!e->encode) return -1;
    return e->encode(e->ctx, data, len, toks, n_toks);
}

int ds4_mm_encode_as_text(const ds4_mm *mm, const char *modality,
                          const uint8_t *data, size_t len, char **out_text) {
    if (out_text) *out_text = NULL;
    if (!out_text || !data) return -1;
    const mm_entry *e = mm_resolve(mm, modality);
    if (!e || !e->encode_text) return -1;         /* unsupported or token-only */
    char *text = NULL;
    if (e->encode_text(e->ctx, data, len, &text) != 0 || !text) return -1;
    *out_text = mm_apply_enrichers(mm, modality, text);
    return 0;
}

int ds4_mm_supported(const ds4_mm *mm, const char *modality) {
    return mm_resolve(mm, modality) != NULL;
}

/* ---- External command encoder (popen bridge). ----
 * ctx owns the command string; the registry entry keeps it alive. The temp
 * file carries the raw content bytes; the command's stdout is the text that
 * feeds the built-in tokenizer. Failure modes (spawn/read/tokenize) all
 * return -1 with *toks NULL -- same honest-rejection contract as an
 * unregistered modality. */
typedef struct {
    char *argv0;
} mm_cmd_ctx;

/* Core of both command forms: bytes -> tmpfile -> `argv0 tmpfile` -> stdout
 * text (malloc'd, NUL-terminated). */
static int mm_cmd_run_text(mm_cmd_ctx *cc, const uint8_t *data, size_t len,
                           char **out_text) {
    *out_text = NULL;
    char tmp[] = "/tmp/ds4_mm_XXXXXX";
    int fd = mkstemp(tmp);
    if (fd < 0) return -1;
    const uint8_t *p = data;
    size_t left = len;
    while (left > 0) {
        ssize_t w = write(fd, p, left);
        if (w <= 0) { close(fd); unlink(tmp); return -1; }
        p += w;
        left -= (size_t)w;
    }
    close(fd);
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "%s %s", cc->argv0, tmp);
    FILE *fp = popen(cmd, "r");
    if (!fp) { unlink(tmp); return -1; }
    char  *text = NULL;
    size_t cap = 0, n = 0;
    char   chunk[4096];
    size_t got;
    while ((got = fread(chunk, 1, sizeof(chunk), fp)) > 0) {
        if (n + got + 1 > cap) {
            cap = (n + got + 1) * 2;
            char *nt = realloc(text, cap);
            if (!nt) { free(text); pclose(fp); unlink(tmp); return -1; }
            text = nt;
        }
        memcpy(text + n, chunk, got);
        n += got;
    }
    int rc = pclose(fp);
    unlink(tmp);
    if (rc != 0 || !text || n == 0) { free(text); return -1; }
    text[n] = '\0';
    *out_text = text;
    return 0;
}

static int mm_cmd_encode_text(void *vctx, const uint8_t *data, size_t len,
                              char **out_text) {
    return mm_cmd_run_text(vctx, data, len, out_text);
}

int ds4_mm_register_command(ds4_mm *mm, const char *modality, const char *argv0) {
    if (!mm || !modality || !argv0 || !argv0[0]) return -1;
    mm_cmd_ctx *cc = malloc(sizeof(*cc));
    if (!cc) return -1;
    cc->argv0 = strdup(argv0);
    /* Text-form only: ds4_mm_encode derives the token form from the enriched
     * text, so a separate token wrapper would just skip the enrichers. */
    if (!cc->argv0 ||
        mm_register_full(mm, modality, NULL, mm_cmd_encode_text, cc) != 0) {
        free(cc->argv0);
        free(cc);
        return -1;
    }
    return 0;   /* cc intentionally lives as long as the registry (process) */
}

/* ---- base64 (strict; the wire form of API image payloads). ---- */
int ds4_mm_b64_decode(const char *s, size_t slen, uint8_t **out, size_t *out_len) {
    if (out) *out = NULL;
    if (out_len) *out_len = 0;
    if (!s || !out || !out_len) return -1;
    uint8_t *buf = malloc(slen ? (slen / 4u + 1u) * 3u + 1u : 1u);
    if (!buf) return -1;
    uint32_t acc = 0;
    int      nacc = 0;      /* chars in acc */
    int      pad = 0;       /* '=' seen */
    size_t   n = 0;
    for (size_t i = 0; i < slen; i++) {
        const char c = s[i];
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        int v;
        if (c >= 'A' && c <= 'Z')      v = c - 'A';
        else if (c >= 'a' && c <= 'z') v = c - 'a' + 26;
        else if (c >= '0' && c <= '9') v = c - '0' + 52;
        else if (c == '+')             v = 62;
        else if (c == '/')             v = 63;
        else if (c == '=') { pad++; if (pad > 2) goto bad; nacc++; continue; }
        else goto bad;
        if (pad) goto bad;            /* data after padding */
        acc = (acc << 6) | (uint32_t)v;
        nacc++;
        if (nacc == 4) {
            buf[n++] = (uint8_t)(acc >> 16);
            buf[n++] = (uint8_t)(acc >> 8);
            buf[n++] = (uint8_t)acc;
            acc = 0;
            nacc = 0;
        }
    }
    if (nacc != 4 && nacc != 0) goto bad;   /* incomplete/unpadded group */
    if (nacc == 4) {                        /* final group carried padding */
        acc <<= 6 * pad;
        buf[n++] = (uint8_t)(acc >> 16);
        if (pad < 2) buf[n++] = (uint8_t)(acc >> 8);
        if (pad < 1) buf[n++] = (uint8_t)acc;
    }
    buf[n] = 0;
    *out = buf;
    *out_len = n;
    return 0;
bad:
    free(buf);
    return -1;
}
