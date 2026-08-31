/* ds4_spatial.c -- 多模态·物理方位插件. Self-contained; the module contract
 * and the sketch grammar it parses are in ds4_spatial.h. */
#include "ds4_spatial.h"

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- tiny string builder (module-local) ---- */
typedef struct { char *s; size_t len, cap; } sb;

static void sb_put(sb *b, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    const int need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need < 0) { va_end(ap2); return; }
    if (b->len + (size_t)need + 1u > b->cap) {
        const size_t ncap = (b->len + (size_t)need + 1u) * 2u;
        char *ns = realloc(b->s, ncap);
        if (!ns) { va_end(ap2); return; }
        b->s = ns;
        b->cap = ncap;
    }
    vsnprintf(b->s + b->len, b->cap - b->len, fmt, ap2);
    b->len += (size_t)need;
    va_end(ap2);
}

/* ---- sketch-line micro parser ---- */
static int p_lit(const char **p, const char *lit) {
    const size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return 0;
    *p += n;
    return 1;
}

static int p_int(const char **p, int *out) {
    char *e = NULL;
    const long v = strtol(*p, &e, 10);
    if (e == *p) return 0;
    *p = e;
    *out = (int)v;
    return 1;
}

static int p_hex6(const char **p, int *out) {
    int v = 0;
    for (int i = 0; i < 6; i++) {
        const char c = (*p)[i];
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else return 0;
        v = v * 16 + d;
    }
    *p += 6;
    *out = v;
    return 1;
}

/* UTF-8-safe excerpt: copy at most cap-1 bytes, backing off a split
 * codepoint, and mark truncation so the legend never lies about a full
 * string. */
static void excerpt_copy(char *dst, size_t cap, const char *src, size_t n) {
    size_t take = n < cap - 1u ? n : cap - 1u;
    const int truncated = take < n;
    if (truncated) {
        while (take > 0 && ((unsigned char)src[take] & 0xC0u) == 0x80u) take--;
        /* leave room for the ellipsis marker */
        while (take > 0 && take + 2u > cap - 1u) {
            take--;
            while (take > 0 && ((unsigned char)src[take] & 0xC0u) == 0x80u) take--;
        }
    }
    memcpy(dst, src, take);
    if (truncated) { memcpy(dst + take, "..", 2); take += 2u; }
    dst[take] = '\0';
}

static int scene_push(ds4_ui_scene *s, int *cap, ds4_ui_elem e) {
    if (s->n == *cap) {
        const int ncap = *cap ? *cap * 2 : 16;
        ds4_ui_elem *nv = realloc(s->v, (size_t)ncap * sizeof(*nv));
        if (!nv) return -1;
        s->v = nv;
        *cap = ncap;
    }
    s->v[s->n++] = e;
    return 0;
}

int ds4_ui_scene_parse(const char *sketch, ds4_ui_scene *out) {
    memset(out, 0, sizeof(*out));
    if (!sketch) return -1;
    int cap = 0;
    int n_rect = 0, n_text = 0;
    const char *line = sketch;
    while (*line) {
        const char *nl = strchr(line, '\n');
        const char *end = nl ? nl : line + strlen(line);
        const char *p = line;
        if (p_lit(&p, "[img ")) {
            int w, h;
            if (p_int(&p, &w) && p_lit(&p, "x") && p_int(&p, &h)) {
                out->img_w = w;
                out->img_h = h;
            }
        } else if (p_lit(&p, "[rect ")) {
            ds4_ui_elem e;
            memset(&e, 0, sizeof(e));
            e.kind = 'r';
            e.fg = -1;
            e.bg = -1;
            if (p_int(&p, &e.x) && p_lit(&p, ",") && p_int(&p, &e.y) &&
                p_lit(&p, " ") && p_int(&p, &e.w) && p_lit(&p, "x") &&
                p_int(&p, &e.h)) {
                const char *q = p;
                int c;
                if (p_lit(&q, " #") && p_hex6(&q, &c)) e.bg = c;
                e.ord = ++n_rect;
                if (scene_push(out, &cap, e) != 0) goto oom;
            }
        } else if (p_lit(&p, "[text ")) {
            ds4_ui_elem e;
            memset(&e, 0, sizeof(e));
            e.kind = 't';
            e.fg = -1;
            e.bg = -1;
            if (p_int(&p, &e.x) && p_lit(&p, ",") && p_int(&p, &e.y) &&
                p_lit(&p, " ") && p_int(&p, &e.w) && p_lit(&p, "x") &&
                p_int(&p, &e.h)) {
                /* optional " #fg on #bg" | " on #bg", then "…" payload */
                const char *q = p;
                int c;
                if (p_lit(&q, " #") && p_hex6(&q, &c)) { e.fg = c; p = q; }
                q = p;
                if (p_lit(&q, " on #") && p_hex6(&q, &c)) { e.bg = c; p = q; }
                const char *open = memchr(p, '"', (size_t)(end - p));
                if (open) {
                    /* the line closes as `"]`; take the last quote */
                    const char *close = NULL;
                    for (const char *t = end - 1; t > open; t--) {
                        if (*t == '"') { close = t; break; }
                    }
                    if (close && close > open + 0)
                        excerpt_copy(e.excerpt, sizeof(e.excerpt), open + 1,
                                     (size_t)(close - open - 1));
                }
                e.ord = ++n_text;
                if (scene_push(out, &cap, e) != 0) goto oom;
            }
        }
        /* unknown lines ([palette ...], future additions): skipped */
        if (!nl) break;
        line = nl + 1;
    }
    if (out->img_w <= 0 || out->img_h <= 0 || out->n == 0) {
        ds4_ui_scene_free(out);
        return -1;
    }
    /* containment: tightest strictly-larger rect wins; 3px slack forgives
     * anti-aliasing shrink at detected edges */
    for (int j = 0; j < out->n; j++) {
        const ds4_ui_elem *c = &out->v[j];
        int best = -1;
        long best_area = LONG_MAX;
        for (int i = 0; i < out->n; i++) {
            if (i == j || out->v[i].kind != 'r') continue;
            const ds4_ui_elem *r = &out->v[i];
            const long ra = (long)r->w * r->h;
            const long ca = (long)c->w * c->h;
            if (ra <= ca) continue;
            if (r->x - 3 > c->x || r->y - 3 > c->y ||
                r->x + r->w + 3 < c->x + c->w ||
                r->y + r->h + 3 < c->y + c->h) continue;
            if (ra < best_area) { best = i; best_area = ra; }
        }
        out->v[j].parent = best;
    }
    return 0;
oom:
    ds4_ui_scene_free(out);
    return -1;
}

void ds4_ui_scene_free(ds4_ui_scene *s) {
    if (!s) return;
    free(s->v);
    memset(s, 0, sizeof(*s));
}

int ds4_ui_children(const ds4_ui_scene *s, int parent, int *out, int cap) {
    int n = 0;
    for (int i = 0; i < s->n; i++) {
        if (s->v[i].parent != parent) continue;
        if (n < cap) {
            /* insertion sort by reading order (y, then x) */
            int k = n;
            while (k > 0 &&
                   (s->v[out[k - 1]].y > s->v[i].y ||
                    (s->v[out[k - 1]].y == s->v[i].y &&
                     s->v[out[k - 1]].x > s->v[i].x))) {
                out[k] = out[k - 1];
                k--;
            }
            out[k] = i;
        }
        n++;
    }
    return n;
}

int ds4_ui_stack(const ds4_ui_scene *s, const int *kids, int n,
                 char *dir, int *gaps_out) {
    if (n < 2) return 0;
    /* column first (dominant in UI): reading order is already y-sorted */
    int is_col = 1;
    for (int i = 1; i < n; i++) {
        const ds4_ui_elem *a = &s->v[kids[i - 1]];
        const ds4_ui_elem *b = &s->v[kids[i]];
        if (b->y < a->y + a->h - 2) { is_col = 0; break; }
    }
    if (is_col) {
        for (int i = 1; i < n; i++)
            gaps_out[i - 1] = s->v[kids[i]].y -
                              (s->v[kids[i - 1]].y + s->v[kids[i - 1]].h);
        *dir = 'c';
        return 1;
    }
    /* row: re-check in x order (reading order sorts x within equal y only) */
    int xo[64];
    if (n > 64) return 0;   /* 上限来自 xo[64] 栈数组: 超限=判定"无法确认横排", 非错误 */
    memcpy(xo, kids, (size_t)n * sizeof(*kids));
    for (int i = 1; i < n; i++) {   /* insertion sort by x */
        const int v = xo[i];
        int k = i;
        while (k > 0 && s->v[xo[k - 1]].x > s->v[v].x) { xo[k] = xo[k - 1]; k--; }
        xo[k] = v;
    }
    for (int i = 1; i < n; i++) {
        const ds4_ui_elem *a = &s->v[xo[i - 1]];
        const ds4_ui_elem *b = &s->v[xo[i]];
        if (b->x < a->x + a->w - 2) return 0;
    }
    for (int i = 1; i < n; i++)
        gaps_out[i - 1] = s->v[xo[i]].x - (s->v[xo[i - 1]].x + s->v[xo[i - 1]].w);
    *dir = 'r';
    return 1;
}

/* ---- annotate ---- */

static void sb_label(sb *b, const ds4_ui_elem *e) {
    sb_put(b, "%c%d", e->kind == 'r' ? 'r' : 't', e->ord);
}

/* page rect = spans (almost) the full image; its children are page level */
static int is_page_rect(const ds4_ui_scene *s, int i) {
    const ds4_ui_elem *e = &s->v[i];
    return e->kind == 'r' &&
           e->w * 10 >= s->img_w * 9 && e->h * 10 >= s->img_h * 9;
}

static const char *zone_name(const ds4_ui_scene *s, const ds4_ui_elem *e) {
    static const char *names[3][3] = {
        {"top-left", "top", "top-right"},
        {"left", "center", "right"},
        {"bottom-left", "bottom", "bottom-right"},
    };
    int cx = e->x + e->w / 2, cy = e->y + e->h / 2;
    int col = cx * 3 / (s->img_w > 0 ? s->img_w : 1);
    int row = cy * 3 / (s->img_h > 0 ? s->img_h : 1);
    if (col < 0) col = 0;
    if (col > 2) col = 2;
    if (row < 0) row = 0;
    if (row > 2) row = 2;
    return names[row][col];
}

#define ALIGN_TOL 3

static int centered_x_in(const ds4_ui_elem *c, const ds4_ui_elem *p) {
    return abs((c->x + c->w / 2) - (p->x + p->w / 2)) <= ALIGN_TOL;
}

static int centered_y_in(const ds4_ui_elem *c, const ds4_ui_elem *p) {
    return abs((c->y + c->h / 2) - (p->y + p->h / 2)) <= ALIGN_TOL;
}

char *ds4_spatial_annotate(const char *sketch) {
    ds4_ui_scene sc;
    if (ds4_ui_scene_parse(sketch, &sc) != 0) return NULL;
    sb b = {0};

    /* [ids]: the label system, so the model never has to count sketch lines */
    int n_rect = 0, n_text = 0;
    for (int i = 0; i < sc.n; i++)
        (sc.v[i].kind == 'r') ? n_rect++ : n_text++;
    sb_put(&b, "[ids");
    if (n_rect) sb_put(&b, " r1-r%d rects in sketch order;", n_rect);
    for (int i = 0; i < sc.n; i++) {
        if (sc.v[i].kind != 't') continue;
        sb_put(&b, " t%d\"%s\"", sc.v[i].ord, sc.v[i].excerpt);
    }
    sb_put(&b, "]");

    /* [where]: nine-grid zones for page-level elements */
    int printed = 0;
    for (int i = 0; i < sc.n; i++) {
        const ds4_ui_elem *e = &sc.v[i];
        const int page_level =
            e->parent < 0 || is_page_rect(&sc, e->parent);
        if (!page_level) continue;
        if (!printed) { sb_put(&b, "\n[where"); printed = 1; }
        else sb_put(&b, ";");
        sb_put(&b, " ");
        sb_label(&b, e);
        if (e->kind == 'r' && is_page_rect(&sc, i)) sb_put(&b, " page");
        else sb_put(&b, " %s", zone_name(&sc, e));
    }
    if (printed) sb_put(&b, "]");

    /* per-container facts: containment, stack flow, alignment */
    int kids[64], gaps[63];
    for (int i = 0; i < sc.n; i++) {
        if (sc.v[i].kind != 'r') continue;
        const int total = ds4_ui_children(&sc, i, kids, 64);
        const int n = total > 64 ? 64 : total;
        if (n == 0) continue;
        const ds4_ui_elem *pr = &sc.v[i];

        sb_put(&b, "\n[in ");
        sb_label(&b, pr);
        sb_put(&b, "(%d,%d):", pr->x, pr->y);
        for (int k = 0; k < n; k++) {
            sb_put(&b, " ");
            sb_label(&b, &sc.v[kids[k]]);
        }
        if (total > n) sb_put(&b, " +%d more", total - n);
        sb_put(&b, "]");

        char dir;
        if (ds4_ui_stack(&sc, kids, n, &dir, gaps)) {
            int uniform = 1;
            for (int k = 1; k < n - 1; k++)
                if (abs(gaps[k] - gaps[0]) > ALIGN_TOL) { uniform = 0; break; }
            sb_put(&b, "\n[stack ");
            sb_label(&b, pr);
            sb_put(&b, " %s", dir == 'c' ? "column" : "row");
            if (uniform) sb_put(&b, " gap=%dpx", gaps[0]);
            else {
                sb_put(&b, " gaps=");
                for (int k = 0; k < n - 1; k++)
                    sb_put(&b, "%s%d", k ? "," : "", gaps[k]);
                sb_put(&b, "px");
            }
            sb_put(&b, ":");
            for (int k = 0; k < n; k++) {
                sb_put(&b, " ");
                sb_label(&b, &sc.v[kids[k]]);
            }
            sb_put(&b, "]");
        }

        /* aligned groups among siblings: edge/center keys, ±3px */
        static const char *key_name[6] = {"left", "right", "center-x",
                                          "top", "bottom", "center-y"};
        for (int key = 0; key < 6; key++) {
            char used[64] = {0};
            for (int a = 0; a < n; a++) {
                if (used[a]) continue;
                int vals[64];
                for (int k = 0; k < n; k++) {
                    const ds4_ui_elem *e = &sc.v[kids[k]];
                    vals[k] = key == 0 ? e->x
                            : key == 1 ? e->x + e->w
                            : key == 2 ? e->x + e->w / 2
                            : key == 3 ? e->y
                            : key == 4 ? e->y + e->h
                                       : e->y + e->h / 2;
                }
                int group[64], gn = 0;
                for (int k = a; k < n; k++)
                    if (!used[k] && abs(vals[k] - vals[a]) <= ALIGN_TOL)
                        group[gn++] = k;
                if (gn >= 2) {
                    for (int g = 0; g < gn; g++) used[group[g]] = 1;
                    sb_put(&b, "\n[align %s:", key_name[key]);
                    for (int g = 0; g < gn; g++) {
                        sb_put(&b, " ");
                        sb_label(&b, &sc.v[kids[group[g]]]);
                    }
                    sb_put(&b, " (in ");
                    sb_label(&b, pr);
                    sb_put(&b, ")]");
                } else {
                    used[a] = 1;
                }
            }
        }

        /* centered-in-parent (the "is the button centered?" answer) */
        for (int axis = 0; axis < 2; axis++) {
            int printed_c = 0;
            for (int k = 0; k < n; k++) {
                const ds4_ui_elem *e = &sc.v[kids[k]];
                const int hit = axis == 0 ? centered_x_in(e, pr)
                                          : centered_y_in(e, pr);
                if (!hit) continue;
                if (!printed_c) {
                    sb_put(&b, "\n[align centered-%c in ", axis == 0 ? 'x' : 'y');
                    sb_label(&b, pr);
                    sb_put(&b, ":");
                    printed_c = 1;
                }
                sb_put(&b, " ");
                sb_label(&b, &sc.v[kids[k]]);
            }
            if (printed_c) sb_put(&b, "]");
        }
    }

    ds4_ui_scene_free(&sc);
    if (!b.s || !b.len) { free(b.s); return NULL; }
    return b.s;
}
