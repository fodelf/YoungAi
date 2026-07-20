/* =========================================================================
 * ds4_spatial -- 多模态·物理方位插件: UI 草图 -> 空间关系事实.
 * =========================================================================
 *
 * mm-ui (tools/mm_ui.swift) encodes a screenshot into geometry facts:
 * [rect X,Y WxH #hex] / [text X,Y WxH #fg on #bg "..."]. A text model --
 * especially a 1-bit quantized one -- is bad at re-deriving spatial
 * relations from raw coordinates ("is the button centered?", "右上角那个").
 * This module derives them CLOSED-FORM, deterministically, and appends them
 * as an enricher section (ds4_mm_register_enricher):
 *
 *   [ids ...]            element labels: rN/tN in sketch order, text excerpts
 *   [where ...]          nine-grid zone of page-level elements (top-left..)
 *   [in rP(x,y): a b]    containment tree, children in reading order
 *   [align key: ...]     same-parent aligned groups + centered-in-parent
 *   [stack rP column gap=N: ...]  vertical/horizontal flow with gaps
 *
 * Honest contract: pure geometry math over the encoder's facts -- no
 * semantic guessing. Unparseable input (a different encoder's output)
 * returns NULL = "no section", never a fake. The sketch grammar is the
 * contract with tools/mm_ui.swift; unknown lines are skipped so palette or
 * future additions do not break parsing. */
#ifndef DS4_SPATIAL_H
#define DS4_SPATIAL_H

#include <stddef.h>

typedef struct {
    char kind;         /* 'r' rect, 't' text */
    int  ord;          /* 1-based within kind, sketch order (r1.., t1..) */
    int  x, y, w, h;   /* original-image px, top-left origin (CSS-style) */
    int  fg, bg;       /* 0xRRGGBB, -1 = not reported by the encoder */
    char excerpt[32];  /* text items: leading bytes, UTF-8-safe truncation */
    int  parent;       /* index of tightest containing rect, -1 = page level */
} ds4_ui_elem;

typedef struct {
    int img_w, img_h;
    ds4_ui_elem *v;
    int n;
} ds4_ui_scene;

/* Parse an mm-ui sketch into a scene (containment resolved). Returns 0, or
 * -1 when the text is not a sketch ([img WxH] missing or no elements). */
int  ds4_ui_scene_parse(const char *sketch, ds4_ui_scene *out);
void ds4_ui_scene_free(ds4_ui_scene *s);

/* Children of rect `parent` (scene index, or -1 for page level), sorted in
 * reading order (y, then x). Fills up to `cap` indices, returns the total
 * child count (callers detect truncation by count > cap). */
int ds4_ui_children(const ds4_ui_scene *s, int parent, int *out, int cap);

/* Flow detection over reading-ordered children: 1 when they stack without
 * overlap along one axis; *dir = 'c' (column) or 'r' (row); gaps_out[n-1]
 * = successive gaps in px. 0 = no clean flow (free-form layout). */
int ds4_ui_stack(const ds4_ui_scene *s, const int *kids, int n,
                 char *dir, int *gaps_out);

/* The enricher: sketch text in, spatial-relations section out (malloc'd, no
 * trailing newline), NULL when the input is not a parseable sketch. */
char *ds4_spatial_annotate(const char *sketch);

#endif /* DS4_SPATIAL_H */
