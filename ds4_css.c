/* ds4_css.c -- 多模态·CSS 理解插件. Builds on the ds4_spatial scene model
 * (parse + containment + flow detection); contract in ds4_css.h. */
#include "ds4_css.h"
#include "ds4_spatial.h"

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

#define TOL 3

char *ds4_css_annotate(const char *sketch) {
    ds4_ui_scene sc;
    if (ds4_ui_scene_parse(sketch, &sc) != 0) return NULL;
    sb b = {0};

    /* page background: the largest full-span rect's fill */
    int page = -1;
    long page_area = 0;
    for (int i = 0; i < sc.n; i++) {
        const ds4_ui_elem *e = &sc.v[i];
        if (e->kind != 'r' || e->bg < 0) continue;
        if (e->w * 10 < sc.img_w * 9 || e->h * 10 < sc.img_h * 9) continue;
        const long a = (long)e->w * e->h;
        if (a > page_area) { page = i; page_area = a; }
    }
    if (page >= 0)
        sb_put(&b, "[css page: background:#%06x]", sc.v[page].bg);

    int kids[64], gaps[63];
    for (int i = 0; i < sc.n; i++) {
        if (sc.v[i].kind != 'r' || i == page) continue;
        const int total = ds4_ui_children(&sc, i, kids, 64);
        const int n = total > 64 ? 64 : total;
        if (n == 0) continue;
        const ds4_ui_elem *pr = &sc.v[i];

        /* box model: padding = distance from container edges to the child
         * extremes (what a stylesheet would have said) */
        int minx = pr->x + pr->w, miny = pr->y + pr->h, maxx = pr->x, maxy = pr->y;
        for (int k = 0; k < n; k++) {
            const ds4_ui_elem *e = &sc.v[kids[k]];
            if (e->x < minx) minx = e->x;
            if (e->y < miny) miny = e->y;
            if (e->x + e->w > maxx) maxx = e->x + e->w;
            if (e->y + e->h > maxy) maxy = e->y + e->h;
        }
        int pt = miny - pr->y, pl = minx - pr->x;
        int pb = pr->y + pr->h - maxy, prr = pr->x + pr->w - maxx;
        if (pt < 0) pt = 0;
        if (pl < 0) pl = 0;
        if (pb < 0) pb = 0;
        if (prr < 0) prr = 0;

        sb_put(&b, "%s[css r%d:", b.len ? "\n" : "", pr->ord);
        if (pr->bg >= 0) sb_put(&b, " background:#%06x;", pr->bg);
        sb_put(&b, " padding:%dpx %dpx %dpx %dpx", pt, prr, pb, pl);

        char dir;
        if (ds4_ui_stack(&sc, kids, n, &dir, gaps)) {
            sb_put(&b, "; display:flex; flex-direction:%s",
                   dir == 'c' ? "column" : "row");
            if (n >= 2) {
                int uniform = 1;
                for (int k = 1; k < n - 1; k++)
                    if (abs(gaps[k] - gaps[0]) > TOL) { uniform = 0; break; }
                if (uniform) sb_put(&b, "; gap:%dpx", gaps[0]);
                else {
                    sb_put(&b, "; gaps:");
                    for (int k = 0; k < n - 1; k++)
                        sb_put(&b, "%s%d", k ? "," : "", gaps[k]);
                    sb_put(&b, "px");
                }
            }
            /* cross-axis alignment -> align-items, only when every child
             * agrees (a stylesheet-level fact, not a per-child accident) */
            int start = 1, center = 1, end = 1;
            for (int k = 0; k < n; k++) {
                const ds4_ui_elem *e = &sc.v[kids[k]];
                if (dir == 'c') {
                    if (abs(e->x - pr->x - pl) > TOL) start = 0;
                    if (abs((e->x + e->w / 2) - (pr->x + pr->w / 2)) > TOL) center = 0;
                    if (abs((e->x + e->w) - (pr->x + pr->w - prr)) > TOL) end = 0;
                } else {
                    if (abs(e->y - pr->y - pt) > TOL) start = 0;
                    if (abs((e->y + e->h / 2) - (pr->y + pr->h / 2)) > TOL) center = 0;
                    if (abs((e->y + e->h) - (pr->y + pr->h - pb)) > TOL) end = 0;
                }
            }
            if (center && n >= 2) sb_put(&b, "; align-items:center");
            else if (start) sb_put(&b, "; align-items:flex-start");
            else if (end) sb_put(&b, "; align-items:flex-end");
        }
        sb_put(&b, "]");
    }

    ds4_ui_scene_free(&sc);
    if (!b.s || !b.len) { free(b.s); return NULL; }
    return b.s;
}
