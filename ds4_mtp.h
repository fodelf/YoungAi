/* =========================================================================
 * ds4_mtp -- knowledge-MTP: multi-token prediction from static language
 * knowledge, as a self-contained module.
 * =========================================================================
 *
 * This module owns every drafter whose knowledge is STATIC (known before the
 * session starts), decoupled from the engine/graph/tensor code: the engine
 * hands it a tokenizer callback once at init and a context tail at decode
 * time; it hands back candidate draft tokens. It never touches model state.
 *
 * Two knowledge sources, one contract:
 *   - reference corpus (ds4_refcorpus_*): built-in Go keyword / syntax-sugar
 *     idioms ("if err != nil { return err }", "go func() { defer wg.Done()"),
 *     tokenized once at init -- zero config, always available. Extension is
 *     DATA, not flags: DS4_REF_CORPUS=file1[:file2...] appends plain-text
 *     snippet files (e.g. gin handler boilerplate) to the built-in set.
 *   - statistical trie (ds4_gotrie_*): an n-gram trie with counts, built
 *     offline from a real corpus (gguf-tools/go-onebit/mtp/build_go_trie.py),
 *     loaded from a GTRI file (--go-trie / DS4_GO_TRIE). Confidence-gated.
 *
 * The contract that keeps this lossless: proposals are only ever DRAFTS. The
 * caller verifies every drafted token against the target model's penalized
 * argmax (the copy-spec VERIFY batch) before committing, and a miss (no
 * anchor / no confident chain) costs nothing -- decode just falls through to
 * the plain step. "不匹配就直接往下走."
 *
 * Distinct from the optional neural draft model (--mtp): that drafter is
 * bound to the inference graph and lives with it. */
#ifndef DS4_MTP_H
#define DS4_MTP_H

#include <stdbool.h>
#include <stdint.h>

/* Tokenizer callback: tokenize utf-8 `text`, return a plain-malloc'd token
 * array in *toks (module frees it) and its length in *n. */
typedef void (*ds4_mtp_tokenize_fn)(void *ctx, const char *text, int **toks, int *n);

/* ---- Reference corpus: built-in idioms + DS4_REF_CORPUS extensions. ---- */
typedef struct ds4_refcorpus ds4_refcorpus;

/* Build once at engine init (needs only the tokenizer). Never fails into a
 * broken state: returns NULL on OOM/empty, and every entry point accepts
 * NULL as "no corpus" (drafting is simply skipped). */
ds4_refcorpus *ds4_refcorpus_build(ds4_mtp_tokenize_fn tokenize, void *tok_ctx);
void ds4_refcorpus_free(ds4_refcorpus *rc);

/* Longest-suffix lookup: find the longest suffix of tail[0..len) (>= min_g
 * tokens, anchor capped at 32 like the transcript matcher) occurring in the
 * corpus and copy up to cap continuation tokens into out[]. Returns the
 * anchor length (0 = no anchor >= min_g: a free miss). Later corpus entries
 * win ties, so extension files override the built-ins. */
uint32_t ds4_refcorpus_match(const ds4_refcorpus *rc, const int *tail,
                             uint32_t len, uint32_t min_g, uint32_t cap,
                             int *out, uint32_t *out_n);

/* ---- Statistical n-gram trie (GTRI file). ----
 * The struct is public so the engine's close-time summary can read the
 * lifetime counters; everything else goes through the functions. */
typedef struct {
    int32_t  token;
    uint32_t count;
    uint32_t child_off;
    uint32_t child_n;
} ds4_gotrie_node;

struct ds4_gotrie {
    ds4_gotrie_node *nodes;
    uint32_t n_nodes;
    uint32_t depth;       /* longest stored n-gram; context depth-1 */
    uint32_t root_off;
    uint32_t root_n;
    /* Proposal policy (env, read once at load). */
    float    conf;        /* min child.count/node.count to extend the chain */
    uint32_t min_cnt;     /* min child count to extend */
    int      min_total;   /* min chain length worth a verify batch */
    int      max_total;   /* max chain length per proposal */
    int      min_order;   /* shortest context order the walk backs off to */
    bool     log;         /* DS4_GO_TRIE_LOG per-fire lines */
    /* Lifetime counters for the close-time summary (diagnostic only). */
    uint64_t fires;
    uint64_t gate_miss;   /* chain's first token != target argmax (free skip) */
    uint64_t sent;
    uint64_t committed;   /* drafted tokens the target accepted */
    uint64_t full;
};

struct ds4_gotrie *ds4_gotrie_load(const char *path);
void ds4_gotrie_free(struct ds4_gotrie *t);

/* Propose a chain of up to `cap` tokens continuing ctx[0..ctxlen). out[0]
 * must equal `must_first` (the target argmax already in hand -- the caller's
 * free gate) or the proposal is abandoned before costing anything. Returns
 * the chain length (0 = no confident proposal). */
uint32_t ds4_gotrie_propose(struct ds4_gotrie *t, const int *ctx,
                            uint32_t ctxlen, int must_first, int *out, int cap);

#endif /* DS4_MTP_H */
