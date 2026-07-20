/* =========================================================================
 * ds4_css -- 多模态·CSS 理解插件: UI 草图 -> CSS 布局事实.
 * =========================================================================
 *
 * The sketch gives raw geometry; the model still has to translate it into
 * the CSS mental model to answer "重写成 HTML/CSS" or "padding 是多少".
 * That translation is deterministic arithmetic, so this enricher does it
 * closed-form and appends CSS FACTS (ds4_mm_register_enricher):
 *
 *   [css page: background:#...]
 *   [css rP: background:#..; padding:T R B L px; display:flex;
 *        flex-direction:column|row; gap:Npx; align-items:...]
 *
 * Only DERIVED properties are emitted -- width/height/colors already stand
 * verbatim in the sketch ([rect]/[text] lines), repeating them would double
 * tokens for nothing. Deliberately NOT emitted: font-size (OCR box height
 * -> em conversion is noisy guesswork, and faking precision violates the
 * honesty contract), border-radius (the encoder cannot measure it yet).
 *
 * Element labels (rN) are the ds4_spatial label system; the two sections
 * are designed to be read together. Non-sketch input returns NULL. */
#ifndef DS4_CSS_H
#define DS4_CSS_H

/* The enricher: sketch text in, CSS-facts section out (malloc'd, no
 * trailing newline), NULL when the input is not a parseable sketch. */
char *ds4_css_annotate(const char *sketch);

#endif /* DS4_CSS_H */
