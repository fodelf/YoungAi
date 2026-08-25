/* server_primer.c — 机械拆分自 ds4_server.c (10528-10731 行): tool-primer 复制推测基元。 */

#include "server_internal.h"

void primer_copy_init(primer_copy *cm, const char *src, size_t len, bool anchored) {
    memset(cm, 0, sizeof(*cm));
    cm->src = src ? src : "";
    cm->len = src ? len : 0;
    cm->anchored = anchored;
    cm->dead = cm->len == 0;
}

static bool primer_copy_extends(const primer_copy *cm, const char *t, size_t tl,
                                size_t *np, int *nn) {
    int n = 0;
    *nn = 0;
    if (tl == 0) return false;
    if (!cm->started) {
        const char *hay = cm->src;
        size_t left = cm->len;
        while (left >= tl && n < PRIMER_COPY_MAX_POS) {
            const char *hit = memmem(hay, left, t, tl);
            if (!hit) break;
            size_t off = (size_t)(hit - cm->src);
            if (!cm->anchored || (off > 0 && cm->src[off - 1] == '\x01'))
                np[n++] = off + tl;
            size_t adv = (size_t)(hit - hay) + 1;
            hay += adv;
            left -= adv;
        }
    } else {
        for (int i = 0; i < cm->n_pos && n < PRIMER_COPY_MAX_POS; i++) {
            size_t o = cm->pos[i];
            if (o + tl <= cm->len && memcmp(cm->src + o, t, tl) == 0)
                np[n++] = o + tl;
        }
    }
    *nn = n;
    return n > 0;
}

static void primer_copy_commit(primer_copy *cm, const size_t *np, int nn) {
    memcpy(cm->pos, np, (size_t)nn * sizeof(*np));
    cm->n_pos = nn;
    cm->started = true;
}

/* logit-descending index sort. File-scope base pointer is fine: inference is
 * serialized through the single graph worker (same invariant the rest of the
 * server relies on). */
static const float *primer_sort_logits;

static int primer_logit_cmp(const void *a, const void *b) {
    const float la = primer_sort_logits[*(const int *)a];
    const float lb = primer_sort_logits[*(const int *)b];
    return la < lb ? 1 : la > lb ? -1 : 0;
}

/* One constrained greedy step. Returns the token to eval, or -1 for "stop
 * here". Stop intent is always honoured from the FREE argmax choice (EOS or
 * a stop char) — the constraint only governs what may be emitted when the
 * model does not want to stop. On infeasibility: soft mode drops the
 * constraint for the rest of the region (escape hatch for generative args
 * like new_string/content) and returns the free choice; hard mode (`hard`)
 * STOPS instead — used for verbatim-by-contract args (old_string), where a
 * fallen-back hallucination is strictly worse than a short verbatim span
 * (Gate v4.1 实测: old_string 逃生口逸出 panic 栈拼接幻觉)。 */
int primer_copy_step(server *s, primer_copy *cm, const char *stopchars,
                            int stop_tok, bool hard, bool *fell_back) {
    int free_tok = ds4_session_argmax(s->session);
    bool stop_here = free_tok == ds4_token_eos(s->engine) ||
                     (stop_tok >= 0 && free_tok == stop_tok);
    size_t gl = 0;
    char *gp = stop_here ? NULL : ds4_token_text(s->engine, free_tok, &gl);
    if (gp) {
        for (size_t i = 0; !stop_here && i < gl; i++)
            if (strchr(stopchars, gp[i])) stop_here = true;
    }
    if (stop_here) { free(gp); return -1; }

    size_t np[PRIMER_COPY_MAX_POS];
    int nn = 0;
    if (cm->dead) { free(gp); return free_tok; }
    if (gp && primer_copy_extends(cm, gp, gl, np, &nn)) {
        primer_copy_commit(cm, np, nn);
        free(gp);
        return free_tok;
    }
    /* ★置信门控的混合值区(2026-07-14, DS4_PRIMER_FREE_CONF=p, 0=关)★
     * 纯 copy 的抄写陷阱: 值需要"构造"(如 `go test ./...` 不在上下文里)时, 任何能延续
     * 某个 span 的 token 都被接受, 模型被拽去抄语义无关的 span(实测 ROUND2: command
     * 抄成 "clamp.go")。而占位符($PARAMETER_VALUE)恰恰出现在模型没把握的位置。
     * 门控: 自由 argmax 的 softmax 概率 ≥ p 时放行自由 token(模型有把握=让它构造),
     * 否则维持 copy 约束(没把握=抄上下文, 防占位符)。hard(verbatim)契约不受影响。
     * 视图统一(COPY_EMISSION lane): 本函数只在 PRIMER_GEN_COPY_C 内被调, 会话在
     * COPY lane 下 ds4_session_argmax 返回 raw 结果 —— p1 与放行的 free_tok 如今
     * 同为 raw argmax(旧状态: p1 按 raw logits 量、放行的却是惩罚后 argmax,
     * "量A放B")。下面的 max 扫描与 lane 无关地重算 raw argmax logit, 保留。 */
    if (gp && !hard) {
        const char *cs = getenv("DS4_PRIMER_FREE_CONF");
        const double thr = cs ? atof(cs) : 0.0;
        if (thr > 0.0) {
            const int nv0 = ds4_engine_vocab_size(s->engine);
            float *l0 = malloc((size_t)nv0 * sizeof(*l0));
            if (l0 && ds4_session_copy_logits(s->session, l0, nv0) == nv0) {
                float mx = l0[0];
                for (int i = 1; i < nv0; i++) if (l0[i] > mx) mx = l0[i];
                double sum = 0.0;
                for (int i = 0; i < nv0; i++) sum += exp((double)(l0[i] - mx));
                const double p1 = sum > 0.0 ? 1.0 / sum : 0.0;   /* free_tok(=raw argmax) 的概率 */
                if (p1 >= thr) {
                    free(l0); free(gp);
                    cm->dead = true;               /* 本参数值后续也自由(已进构造模式) */
                    if (fell_back) *fell_back = true;
                    return free_tok;
                }
            }
            free(l0);
        }
    }
    free(gp);

    /* free choice infeasible: take the best-logit feasible token instead */
    const int nv = ds4_engine_vocab_size(s->engine);
    float *lg = malloc((size_t)nv * sizeof(*lg));
    int   *idx = malloc((size_t)nv * sizeof(*idx));
    int    pick = -1;
    if (lg && idx && ds4_session_copy_logits(s->session, lg, nv) == nv) {
        for (int i = 0; i < nv; i++) idx[i] = i;
        primer_sort_logits = lg;
        qsort(idx, (size_t)nv, sizeof(*idx), primer_logit_cmp);
        const int tries = nv < 4096 ? nv : 4096;
        for (int i = 0; i < tries; i++) {
            const int cand = idx[i];
            if (cand == free_tok || cand == ds4_token_eos(s->engine)) continue;
            size_t cl = 0;
            char *cp = ds4_token_text(s->engine, cand, &cl);
            if (!cp) continue;
            bool has_stop = false;
            for (size_t k = 0; k < cl && !has_stop; k++)
                if (strchr(stopchars, cp[k])) has_stop = true;
            const bool ok = !has_stop && primer_copy_extends(cm, cp, cl, np, &nn);
            free(cp);
            if (ok) { primer_copy_commit(cm, np, nn); pick = cand; break; }
        }
    }
    free(lg); free(idx);
    if (pick >= 0) return pick;
    if (hard) return -1;                /* verbatim contract: stop, never invent */
    if (fell_back) *fell_back = true;   /* value is not a context span: free-run */
    cm->dead = true;
    return free_tok;
}

/* new!=old contract divergence point: best-logit token that is neither the
 * value closer nor EOS.  Used when the model tries to close a new_string that
 * is byte-identical to old_string -- an edit whose replacement equals its
 * target is never a valid call (API semantics), so decoding must diverge. */
int primer_divergence_token(server *s, int exclude_tok) {
    const int nv = ds4_engine_vocab_size(s->engine);
    float *lg = malloc((size_t)nv * sizeof(*lg));
    int best = -1;
    if (lg && ds4_session_copy_logits(s->session, lg, nv) == nv) {
        const int eos = ds4_token_eos(s->engine);
        for (int i = 0; i < nv; i++) {
            if (i == exclude_tok || i == eos) continue;
            if (best < 0 || lg[i] > lg[best]) best = i;
        }
    }
    free(lg);
    return best;
}
