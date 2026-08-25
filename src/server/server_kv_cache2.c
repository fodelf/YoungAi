/* server_kv_cache2.c — 机械拆分自 ds4_server.c (9202-9642 行): 磁盘 KV 缓存(存取/前缀查找)。 */

#include "server_internal.h"

#ifdef DS4_SERVER_TEST
double kv_entry_eviction_score(const kv_entry *e, const ds4_tokens *live,
                                      uint64_t now,
                                      const ds4_kvstore_eviction_context *incoming) {
    return ds4_kvstore_entry_eviction_score(e, live, now, incoming);
}
#endif

#ifdef DS4_SERVER_TEST
void kv_cache_evict(kv_disk_cache *kc, const ds4_tokens *live,
                           uint64_t extra_bytes,
                           const ds4_kvstore_eviction_context *incoming) {
    ds4_kvstore_evict(kc, live, extra_bytes, incoming);
}
#endif

static void kv_cache_log_cb(void *ud, ds4_kvstore_log_type type, const char *msg) {
    (void)ud;
    ds4_log_type stype = DS4_LOG_KVCACHE;
    if (type == DS4_KVSTORE_LOG_DEFAULT) stype = DS4_LOG_DEFAULT;
    else if (type == DS4_KVSTORE_LOG_WARNING) stype = DS4_LOG_WARNING;
    server_log(stype, "%s", msg);
}

bool kv_cache_open(kv_disk_cache *kc, const char *dir, uint64_t budget_mb,
                          bool reject_different_quant, kv_cache_options opt) {
    return ds4_kvstore_open(kc, dir, budget_mb, reject_different_quant, opt,
                            "ds4-server", kv_cache_log_cb, NULL);
}

void kv_cache_close(kv_disk_cache *kc) {
    ds4_kvstore_close(kc);
}

char *render_tokens_text(ds4_engine *engine, const ds4_tokens *tokens, size_t *out_len) {
    return ds4_kvstore_render_tokens_text(engine, tokens, out_len);
}

static bool byte_prefix_match(const char *text, size_t text_len,
                              const char *prefix, size_t prefix_len) {
    return ds4_kvstore_byte_prefix_match(text, text_len, prefix, prefix_len);
}

void tokens_copy_prefix(ds4_tokens *dst, const ds4_tokens *src, int n) {
    ds4_kvstore_tokens_copy_prefix(dst, src, n);
}

void build_prompt_from_exact_prefix_and_text_suffix(
        ds4_engine *engine,
        const ds4_tokens *exact_prefix,
        const char *suffix_text,
        ds4_tokens *out)
{
    ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        engine, exact_prefix, suffix_text, out);
}

int kv_cache_store_len(const kv_disk_cache *kc, int tokens) {
    return ds4_kvstore_store_len(kc, tokens);
}

int kv_cache_chat_anchor_pos(const kv_disk_cache *kc,
                                    const ds4_tokens *prompt,
                                    int user_token_id,
                                    int assistant_token_id) {
    return ds4_kvstore_chat_anchor_pos(kc, prompt, user_token_id, assistant_token_id);
}

int kv_cache_continued_store_target(const kv_disk_cache *kc, int live_tokens) {
    return ds4_kvstore_continued_store_target(kc, live_tokens);
}

/* A same-text-prefix file can be reused by a larger context, but not by a
 * smaller one: the payload was validated against the context capacity recorded
 * in the file.  If the existing file cannot be used by this server, replace it
 * so this context can still populate its own cache. */

#ifdef DS4_SERVER_TEST
bool kv_cache_file_size_fits(const kv_disk_cache *kc,
                                    uint64_t text_bytes,
                                    uint64_t payload_bytes,
                                    uint64_t tool_map_bytes,
                                    uint64_t *file_bytes_out,
                                    uint64_t *required_bytes_out) {
    return ds4_kvstore_file_size_fits(kc, text_bytes, payload_bytes,
                                      tool_map_bytes, file_bytes_out,
                                      required_bytes_out);
}
#endif

static bool kv_cache_tool_map_size_cb(void *ud, const char *text,
                                      uint64_t *bytes_out) {
    return kv_tool_map_serialized_size((server *)ud, text, bytes_out);
}

static bool kv_cache_tool_map_write_cb(void *ud, FILE *fp, const char *text,
                                       uint64_t *written_bytes) {
    return kv_tool_map_write((server *)ud, fp, text, written_bytes);
}

static int kv_cache_tool_map_load_cb(void *ud, FILE *fp, const void *wanted) {
    return kv_tool_map_load_from_pos((server *)ud, fp, (const stop_list *)wanted);
}

static ds4_kvstore_trailer_hooks kv_cache_tool_map_hooks(server *s,
                                                         const stop_list *wanted) {
    return (ds4_kvstore_trailer_hooks){
        .ud = s,
        .ext_flag = KV_EXT_TOOL_MAP,
        .serialized_size = kv_cache_tool_map_size_cb,
        .write = kv_cache_tool_map_write_cb,
        .load = kv_cache_tool_map_load_cb,
        .load_wanted = wanted,
    };
}

static bool kv_cache_store_live_prefix_text(server *s, const ds4_tokens *tokens,
                                            int store_len, const char *reason,
                                            const char *cache_text_override,
                                            uint8_t cache_text_ext,
                                            const char *cache_text_key) {
    char err[160] = {0};
    ds4_kvstore_trailer_hooks hooks = kv_cache_tool_map_hooks(s, NULL);
    return ds4_kvstore_store_live_prefix_text(&s->kv, s->engine, s->session,
                                              tokens, store_len, reason,
                                              cache_text_override,
                                              cache_text_ext,
                                              cache_text_key,
                                              &hooks, err, sizeof(err));
}

bool kv_cache_store_live_prefix(server *s, const ds4_tokens *tokens,
                                       int store_len, const char *reason) {
    return kv_cache_store_live_prefix_text(s, tokens, store_len, reason,
                                           NULL, 0, NULL);
}

void kv_cache_store_current(server *s, const char *reason) {
    const ds4_tokens *tokens = ds4_session_tokens(s->session);
    if (!tokens) return;

    char *visible_text = NULL;
    uint8_t visible_ext = 0;
    const char *visible_key = NULL;
    pthread_mutex_lock(&s->tool_mu);
    if (s->responses_live.valid &&
        s->responses_live.live_tokens == tokens->len &&
        s->responses_live.visible_text &&
        s->responses_live.visible_text[0])
    {
        visible_text = xstrdup(s->responses_live.visible_text);
        visible_ext = KV_EXT_RESPONSES_VISIBLE;
        visible_key = "responses-visible";
    } else if (s->thinking_live.valid &&
               s->thinking_live.live_tokens == tokens->len &&
               s->thinking_live.visible_text &&
               s->thinking_live.visible_text[0])
    {
        visible_text = xstrdup(s->thinking_live.visible_text);
        visible_ext = KV_EXT_THINKING_VISIBLE;
        visible_key = "thinking-visible";
    }
    pthread_mutex_unlock(&s->tool_mu);

    /* A visible live checkpoint can contain hidden reasoning that the client
     * intentionally does not replay.  For disk recovery after a session switch,
     * key that payload by the visible protocol transcript, not by rendering the
     * hidden sampled tokens.  On load, DS4 restores the hidden KV payload and
     * tokenizes only the visible suffix that follows this key. */
    if (visible_text) {
        kv_cache_store_live_prefix_text(s, tokens, tokens->len, reason,
                                        visible_text, visible_ext, visible_key);
        free(visible_text);
    } else {
        kv_cache_store_live_prefix(s, tokens, tokens->len, reason);
    }
}

void kv_cache_note_store(kv_disk_cache *kc, int tokens) {
    ds4_kvstore_note_store(kc, tokens);
}

int kv_cache_suppress_continued_store(kv_disk_cache *kc, int tokens) {
    return ds4_kvstore_suppress_continued_store(kc, tokens);
}

void kv_cache_restore_suppressed_continued(kv_disk_cache *kc,
                                                  int old_tokens,
                                                  int suppressed_tokens) {
    ds4_kvstore_restore_suppressed_continued(kc, old_tokens, suppressed_tokens);
}

void kv_cache_discard_failed_disk_entry(server *s, const char *path) {
    if (!s || !path) return;
    if (unlink(path) == 0) {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: kv cache discarded reason=prefill-failed file=%s",
                   path);
    } else if (errno != ENOENT) {
        server_log(DS4_LOG_WARNING,
                   "ds4-server: kv cache failed to discard prefill-failed file=%s: %s",
                   path, strerror(errno));
    }
    s->kv.continued_last_store_tokens = 0;
    ds4_session_invalidate(s->session);
}

void kv_cache_maybe_store_continued(server *s) {
    kv_disk_cache *kc = &s->kv;
    const ds4_tokens *tokens = ds4_session_tokens(s->session);
    if (!tokens) return;
    const int target = kv_cache_continued_store_target(kc, tokens->len);
    if (target == 0) return;
    if (kv_cache_store_live_prefix(s, tokens, target, "continued")) {
        kv_cache_note_store(kc, target);
    }
}

#ifdef DS4_SERVER_TEST
int kv_cache_find_text_prefix(kv_disk_cache *kc, const char *prompt_text,
                                     int quant_bits, int ctx_size) {
    return ds4_kvstore_find_text_prefix(kc, prompt_text, 0, quant_bits, ctx_size);
}
#endif

int kv_cache_try_load_text(server *s, const char *prompt_text,
                                  ds4_tokens *effective_prompt,
                                  char **loaded_path_out,
                                  uint8_t *loaded_ext_flags_out,
                                  bool responses_protocol) {
    if (loaded_path_out) *loaded_path_out = NULL;
    if (loaded_ext_flags_out) *loaded_ext_flags_out = 0;
    ds4_kvstore_load_result lr = {0};
    ds4_kvstore_trailer_hooks hooks = kv_cache_tool_map_hooks(s, NULL);
    int loaded = ds4_kvstore_try_load_text(&s->kv, s->engine, s->session,
                                           prompt_text, effective_prompt, &lr,
                                           &hooks, responses_protocol);
    if (loaded > 0) {
        if (loaded_path_out && lr.path) *loaded_path_out = xstrdup(lr.path);
        if (loaded_ext_flags_out) *loaded_ext_flags_out = lr.ext_flags;
    }
    ds4_kvstore_load_result_free(&lr);
    return loaded;
}

int kv_cache_try_load(server *s, const request *req,
                             ds4_tokens *effective_prompt,
                             char **loaded_path_out,
                             uint8_t *loaded_ext_flags_out) {
    return kv_cache_try_load_text(s, req ? req->prompt_text : NULL,
                                  effective_prompt,
                                  loaded_path_out,
                                  loaded_ext_flags_out,
                                  req && req->api == API_RESPONSES);
}

int live_text_prefix_prompt(server *s, const request *req,
                                   ds4_tokens *effective_prompt) {
    if (!s || !req || !req->prompt_text || !effective_prompt) return 0;
    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len <= 0) return 0;

    size_t live_text_len = 0;
    char *live_text = render_tokens_text(s->engine, live_tokens, &live_text_len);
    const size_t prompt_text_len = strlen(req->prompt_text);
    if (!byte_prefix_match(req->prompt_text, prompt_text_len,
                           live_text, live_text_len))
    {
        free(live_text);
        return 0;
    }

    /* This is the core text-prefix case.  The live graph is authoritative, so
     * keep its sampled tokenization and tokenize only the request bytes that
     * come after it.  Reusing req->prompt's token suffix would be wrong: full
     * prompt BPE may have merged across this byte boundary. */
    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->prompt_text + live_text_len,
        effective_prompt);
    free(live_text);
    return live_tokens->len;
}

/* Tool-output-only Responses continuation.
 *
 * Some clients send just the new tool outputs after a tool call.  There is no
 * long visible prefix to match in that shape; the call_id itself is the
 * protocol binding to the previous live assistant output.  Use it only when the
 * remembered live frontier and call-id set match exactly. */
int responses_live_continuation_prompt(server *s, const request *req,
                                              int live_pos,
                                              ds4_tokens *effective_prompt,
                                              int *matched_ids) {
    if (!s || !req || !effective_prompt) return 0;
    if (req->api != API_RESPONSES || !req->responses_live_suffix_text) return 0;
    if (req->responses_live_call_ids.len == 0) return 0;
    if (!responses_live_matches_request(s, &req->responses_live_call_ids,
                                        live_pos)) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->responses_live_suffix_text,
        effective_prompt);
    if (matched_ids) *matched_ids = req->responses_live_call_ids.len;
    return live_tokens->len;
}

/* Tool-result Anthropic continuation.
 *
 * /v1/messages has no server-side response object like the OpenAI Responses
 * API, but its tool_use_id is still a precise continuation handle inside a live
 * local agent loop.  When the IDs and live token frontier match, continue from
 * the sampled DSML state and append only the user tool_result suffix. */
int anthropic_live_continuation_prompt(server *s, const request *req,
                                              int live_pos,
                                              ds4_tokens *effective_prompt,
                                              int *matched_ids) {
    if (!s || !req || !effective_prompt) return 0;
    if (req->api != API_ANTHROPIC || !req->anthropic_live_suffix_text) return 0;
    if (req->anthropic_live_call_ids.len == 0) return 0;
    if (!anthropic_live_matches_request(s, &req->anthropic_live_call_ids,
                                        live_pos)) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->anthropic_live_suffix_text,
        effective_prompt);
    if (matched_ids) *matched_ids = req->anthropic_live_call_ids.len;
    return live_tokens->len;
}

/* Visible-replay Responses continuation.
 *
 * Other clients send the full visible transcript on every turn even though the
 * API semantics still make the request a continuation.  For Responses, exact
 * token-prefix matching is the wrong first question: hidden reasoning may be
 * live in KV but absent from the replay by design.  Instead, verify that the
 * request's rendered text begins with the visible transcript remembered at the
 * live frontier.  If it does, continue from the live token prefix and tokenize
 * only the bytes after that visible boundary.
 *
 * If this check fails, DS4 has no special Responses state to trust.  The caller
 * then uses normal token/text/disk matching, which is the correct fallback for
 * cold starts, edits, restarts, or cross-client replays. */
int responses_live_visible_prefix_prompt(server *s, const request *req,
                                                int live_pos,
                                                ds4_tokens *effective_prompt) {
    if (!s || !req || !req->prompt_text || !effective_prompt) return 0;
    if (req->api != API_RESPONSES) return 0;

    const size_t prompt_len = strlen(req->prompt_text);
    size_t visible_len = 0;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = s->responses_live.valid &&
              s->responses_live.live_tokens == live_pos &&
              s->responses_live.visible_text &&
              s->responses_live.visible_len < prompt_len &&
              byte_prefix_match(req->prompt_text, prompt_len,
                                s->responses_live.visible_text,
                                s->responses_live.visible_len);
    if (ok) visible_len = s->responses_live.visible_len;
    pthread_mutex_unlock(&s->tool_mu);
    if (!ok) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->prompt_text + visible_len,
        effective_prompt);
    return live_tokens->len;
}

/* Tool-less thinking continuation.
 *
 * Chat/completions and Anthropic do not have a previous_response_id object that
 * binds a later request to the last sampled turn.  Still, after a normal
 * tool-less thinking answer, the next prompt renderer intentionally omits that
 * hidden reasoning.  The live KV state is richer than the visible transcript.
 *
 * Remembering the visible transcript as a key lets us keep the sampled hidden
 * KV when the next request clearly extends that same visible history.  This is
 * the same byte-prefix idea used by the disk cache: the client-visible text
 * selects the checkpoint, while the payload stays the exact sampled token
 * frontier.  If the visible key does not match, callers fall back to ordinary
 * token/text/disk matching. */
int thinking_live_visible_prefix_prompt(server *s, const request *req,
                                               int live_pos,
                                               ds4_tokens *effective_prompt) {
    if (!s || !req || !req->prompt_text || !effective_prompt) return 0;
    if (req->kind != REQ_CHAT || req->api == API_RESPONSES) return 0;

    const size_t prompt_len = strlen(req->prompt_text);
    size_t visible_len = 0;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = s->thinking_live.valid &&
              s->thinking_live.live_tokens == live_pos &&
              s->thinking_live.visible_text &&
              s->thinking_live.visible_len < prompt_len &&
              byte_prefix_match(req->prompt_text, prompt_len,
                                s->thinking_live.visible_text,
                                s->thinking_live.visible_len);
    if (ok) visible_len = s->thinking_live.visible_len;
    pthread_mutex_unlock(&s->tool_mu);
    if (!ok) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->prompt_text + visible_len,
        effective_prompt);
    return live_tokens->len;
}

/* =========================================================================
 * Trace Diagnostics.
 * =========================================================================
 *
 * The human transcript is not enough to debug prompt-cache misses.  The model
 * may generate text that is semantically accepted as a tool call, while the
 * next OpenAI request re-renders a slightly different canonical DSML block.
 * That creates a token mismatch even if the conversation "looks" continuous.
 *
 * When --trace is enabled we therefore record the exact cache decision and a
 * small token window around the first mismatch between the live KV checkpoint
 * and the incoming prompt.  Normal server logs stay compact; trace files get
 * enough data to diagnose tokenizer-boundary and canonicalization problems.
 */
