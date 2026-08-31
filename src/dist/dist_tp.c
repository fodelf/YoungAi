/* dist_tp.c — 机械拆自 ds4_distributed.c: TP all-reduce 传输 + TP run loops(Tensor-parallel)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Tensor-parallel all-reduce transport (Stage 2 skeleton)
 *
 * Distinct from the layer-pipeline path below. A single persistent TCP socket
 * connects the two TP peers. Per TP sync point each peer holds a partial
 * [count] float vector (its ff-slice contribution to a layer's down_proj);
 * ds4_dist_tp_allreduce_f32 exchanges and sums them so both peers end with the
 * full vector. Two-node sum-all-reduce = a single ordered full exchange.
 *
 * Deadlock-safety: the listener side sends-then-receives, the connector side
 * receives-then-sends, so a writer always has a reader draining the socket.
 * Floats travel in host byte order — both peers are little-endian arm64 Macs,
 * matching the activation-payload convention (control headers still use htonl).
 * Buffers are KB-to-MB scratch, grown lazily and capped (no per-token alloc on
 * the hot path, no model load — memory-safe by construction).
 * ========================================================================= */

#define DS4_DIST_TP_MAX_FLOATS (16u * 1024u * 1024u) /* 64 MiB scratch cap */

struct ds4_dist_tp {
    int fd;              /* peer socket (owned) */
    bool send_first;     /* listener true, connector false */
    float *recv_scratch; /* peer partial buffer, grown lazily */
    uint32_t scratch_floats;
};

static int dist_tp_grow_scratch(ds4_dist_tp *tp, uint32_t count) {
    if (count <= tp->scratch_floats) return 0;
    if (count > DS4_DIST_TP_MAX_FLOATS) return -1;
    float *p = realloc(tp->recv_scratch, (size_t)count * sizeof(float));
    if (!p) return -1;
    tp->recv_scratch = p;
    tp->scratch_floats = count;
    return 0;
}

static ds4_dist_tp *dist_tp_alloc(int fd, bool send_first) {
    ds4_dist_tp *tp = calloc(1, sizeof(*tp));
    if (!tp) { close(fd); return NULL; }
    tp->fd = fd;
    tp->send_first = send_first;
    /* The TP socket is private to the all-reduce path, which drives it as a
     * full-duplex non-blocking pump (dist_tp_exchange). Flip it non-blocking
     * once here; the legacy ordered send/recv helpers are no longer used on it. */
    int fl = fcntl(fd, F_GETFL, 0);
    if (fl >= 0) (void)fcntl(fd, F_SETFL, fl | O_NONBLOCK);
    return tp;
}

ds4_dist_tp *ds4_dist_tp_listen(const char *host, int port, char *err, size_t errlen) {
    int ls = dist_open_listener(host, port, err, errlen);
    if (ls < 0) return NULL;
    int fd = accept(ls, NULL, NULL);
    close(ls);
    if (fd < 0) {
        if (errlen) snprintf(err, errlen, "tp accept failed: %s", strerror(errno));
        return NULL;
    }
    dist_set_socket_low_latency(fd);
    return dist_tp_alloc(fd, /*send_first=*/true);
}

ds4_dist_tp *ds4_dist_tp_connect(const char *host, int port, char *err, size_t errlen) {
    int fd = dist_connect_endpoint(host, port, err, errlen);
    if (fd < 0) return NULL;
    return dist_tp_alloc(fd, /*send_first=*/false);
}

void ds4_dist_tp_free(ds4_dist_tp *tp) {
    if (!tp) return;
    if (tp->fd >= 0) close(tp->fd);
    free(tp->recv_scratch);
    free(tp);
}

/* Full-duplex frame exchange over the dedicated (non-blocking) TP socket.
 *
 * Sends one ALLREDUCE frame (12-byte header + `count` payload floats from `buf`)
 * while concurrently receiving the peer's frame (header into a local record,
 * payload into recv_scratch). TCP is full-duplex, so overlapping the two
 * transfers costs ~one one-way latency instead of the two serialized transfers
 * an ordered send-then-recv pays — on the per-token TP hot path the exchange is
 * on the critical path of every tp-layer, so halving it directly cuts decode
 * latency. A single poll() loop pumps whichever direction is ready, so the read
 * side is always drained: deadlock-free at any payload size, send_first no
 * longer matters (both peers run this identical path). */
static int dist_tp_exchange(ds4_dist_tp *tp, const float *buf, uint32_t count) {
    const size_t payload = (size_t)count * sizeof(float);
    const ds4_dist_frame_header sh = {
        htonl(DS4_DIST_MAGIC), htonl(DS4_DIST_MSG_ALLREDUCE), htonl((uint32_t)payload)
    };
    ds4_dist_frame_header rh;
    const size_t total = sizeof(sh) + payload;
    size_t sent = 0, got = 0;

    const int timeout_sec = DIST_SOCKET_TIMEOUT_SEC;

    while (sent < total || got < total) {
        struct pollfd pfd = { .fd = tp->fd, .events = 0, .revents = 0 };
        if (sent < total) pfd.events |= POLLOUT;
        if (got  < total) pfd.events |= POLLIN;
        int pr = poll(&pfd, 1, timeout_sec * 1000);
        if (pr < 0) { if (errno == EINTR) continue; return -1; }
        if (pr == 0) return -1; /* timed out */
        if (pfd.revents & (POLLERR | POLLNVAL)) return -1;

        if ((pfd.revents & POLLOUT) && sent < total) {
            const void *p; size_t len;
            if (sent < sizeof(sh)) { p = (const unsigned char *)&sh + sent; len = sizeof(sh) - sent; }
            else { size_t po = sent - sizeof(sh); p = (const unsigned char *)buf + po; len = payload - po; }
            ssize_t n = send(tp->fd, p, len, 0);
            if (n < 0) { if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {} else return -1; }
            else if (n == 0) return -1;
            else sent += (size_t)n;
        }
        if ((pfd.revents & (POLLIN | POLLHUP)) && got < total) {
            void *p; size_t len;
            if (got < sizeof(rh)) { p = (unsigned char *)&rh + got; len = sizeof(rh) - got; }
            else { size_t po = got - sizeof(rh); p = (unsigned char *)tp->recv_scratch + po; len = payload - po; }
            ssize_t n = recv(tp->fd, p, len, 0);
            if (n < 0) { if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {} else return -1; }
            else if (n == 0) return -1; /* peer closed mid-frame */
            else got += (size_t)n;
        }
    }

    if (ntohl(rh.magic) != DS4_DIST_MAGIC ||
        ntohl(rh.type) != DS4_DIST_MSG_ALLREDUCE ||
        ntohl(rh.bytes) != (uint32_t)payload) return -1;
    return 0;
}

int ds4_dist_tp_allreduce_f32(ds4_dist_tp *tp, float *buf, uint32_t count) {
    if (!tp || tp->fd < 0 || !buf || count == 0) return -1;
    if (dist_tp_grow_scratch(tp, count) != 0) return -1;
    if (dist_tp_exchange(tp, buf, count) != 0) return -1;
    for (uint32_t i = 0; i < count; i++) buf[i] += tp->recv_scratch[i];
    return 0;
}

/* Loopback self-test: two in-process peers over socketpair() verify the
 * sum-all-reduce result and frame format. No model, no network, KB buffers. */
typedef struct {
    int fd;
    bool send_first;
    float *vec;
    uint32_t count;
    int rc;
} dist_tp_selftest_arg;

static void *dist_tp_selftest_thread(void *ud) {
    dist_tp_selftest_arg *a = ud;
    ds4_dist_tp *tp = dist_tp_alloc(a->fd, a->send_first);
    a->fd = -1; /* ownership moved into tp */
    if (!tp) { a->rc = -1; return NULL; }
    a->rc = ds4_dist_tp_allreduce_f32(tp, a->vec, a->count);
    ds4_dist_tp_free(tp);
    return NULL;
}

int ds4_dist_tp_selftest(void) {
    const uint32_t count = 4096; /* one n_embd vector */
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return -1;

    float *a = malloc(count * sizeof(float));
    float *b = malloc(count * sizeof(float));
    if (!a || !b) { free(a); free(b); close(sv[0]); close(sv[1]); return -1; }
    for (uint32_t i = 0; i < count; i++) { a[i] = (float)i; b[i] = (float)(2 * i) + 1.0f; }

    dist_tp_selftest_arg ta = { sv[0], true,  a, count, 0 };
    dist_tp_selftest_arg tb = { sv[1], false, b, count, 0 };
    pthread_t th;
    pthread_create(&th, NULL, dist_tp_selftest_thread, &tb);
    dist_tp_selftest_thread(&ta);
    pthread_join(th, NULL);

    int rc = (ta.rc == 0 && tb.rc == 0) ? 0 : -1;
    for (uint32_t i = 0; rc == 0 && i < count; i++) {
        float expect = (float)i + (float)(2 * i) + 1.0f; /* a[i]+b[i] */
        if (a[i] != expect || b[i] != expect) { rc = -1; break; }
    }
    free(a);
    free(b);
    return rc;
}

/* =========================================================================
 * Tensor-parallel run loops (Stage 2). Both peers load the full model and run
 * the same tokens in lockstep. The leader (coordinator) owns tokenization and
 * greedy sampling and broadcasts each accepted token to the follower (worker)
 * over the TP control socket; the MoE down_proj all-reduce inside the decode
 * graph keeps them synchronized layer by layer. Prefill is replicated (batch
 * path, no all-reduce) so both build identical KV before decode. Eval goes
 * through the local slice primitive (full layer range) to bypass the pipeline.
 * ========================================================================= */

#define DS4_DIST_MSG_TP_PROMPT 11u
#define DS4_DIST_MSG_TP_STEP   12u
#define DS4_DIST_MSG_TP_BATCH  13u  /* copy-spec: [n_tokens][tokens...] (n=0 stop, n=1 bare, n>1 verify) */
#define DS4_DIST_MSG_TP_ACCEPT 14u  /* copy-spec: [accept_len] after a verify batch */

static int dist_tp_send_prompt(ds4_dist_tp *tp, const ds4_tokens *toks) {
    int fd = tp->fd;
    uint32_t n = (uint32_t)(toks->len > 0 ? toks->len : 0);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_TP_PROMPT,
                                (uint32_t)((1u + n) * sizeof(uint32_t))) != 0) return -1;
    uint32_t hdr = htonl(n);
    if (dist_write_full(fd, &hdr, sizeof(hdr)) != 0) return -1;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t t = htonl((uint32_t)toks->v[i]);
        if (dist_write_full(fd, &t, sizeof(t)) != 0) return -1;
    }
    return 0;
}

static int dist_tp_recv_prompt(ds4_dist_tp *tp, ds4_tokens *out) {
    int fd = tp->fd;
    uint32_t type = 0, bytes = 0;
    char err[64];
    if (dist_read_frame_header(fd, &type, &bytes, err, sizeof(err)) <= 0) return -1;
    if (type != DS4_DIST_MSG_TP_PROMPT || bytes < sizeof(uint32_t)) return -1;
    uint32_t n_net = 0;
    if (dist_read_full(fd, &n_net, sizeof(n_net)) <= 0) return -1;
    uint32_t n = ntohl(n_net);
    if (bytes != (uint32_t)((1u + n) * sizeof(uint32_t))) return -1;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t t = 0;
        if (dist_read_full(fd, &t, sizeof(t)) <= 0) return -1;
        ds4_tokens_push(out, (int)(int32_t)ntohl(t));
    }
    return 0;
}

/* TP prefill feeds the whole prompt through all layers. With expert-offload the
 * set of routed experts touched by a wide token batch can blow past a 16 GB
 * GPU's wired limit, so feed the prompt in small chunks to bound the live
 * expert working set. 4 keeps both peers (incl. M1 Pro) under the wired
 * ceiling. */
static uint32_t dist_tp_prefill_chunk(void) {
    return 4u;
}

static int dist_tp_argmax(const float *logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

/* copy-spec TP protocol: the leader proposes a token run (n=1 bare round, n>1 a
 * verify batch [first_token, copied...]); the follower mirrors the same eval so
 * its KV stays in lockstep for the next bare round's Phase-3 all-reduce. n=0 is
 * the stop sentinel. After a verify batch the leader sends the accepted length so
 * both peers roll their layer-slice KV back to the same boundary. */
static int dist_tp_send_batch(ds4_dist_tp *tp, const int *tokens, uint32_t n) {
    int fd = tp->fd;
    if (dist_write_frame_header(fd, DS4_DIST_MSG_TP_BATCH,
                                (uint32_t)((1u + n) * sizeof(uint32_t))) != 0) return -1;
    uint32_t hn = htonl(n);
    if (dist_write_full(fd, &hn, sizeof(hn)) != 0) return -1;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t t = htonl((uint32_t)tokens[i]);
        if (dist_write_full(fd, &t, sizeof(t)) != 0) return -1;
    }
    return 0;
}

static int dist_tp_recv_batch(ds4_dist_tp *tp, int *tokens, uint32_t cap, uint32_t *n_out) {
    int fd = tp->fd;
    uint32_t type = 0, bytes = 0; char err[64];
    if (dist_read_frame_header(fd, &type, &bytes, err, sizeof(err)) <= 0) return -1;
    if (type != DS4_DIST_MSG_TP_BATCH || bytes < sizeof(uint32_t)) return -1;
    uint32_t hn = 0;
    if (dist_read_full(fd, &hn, sizeof(hn)) <= 0) return -1;
    uint32_t n = ntohl(hn);
    if (bytes != (uint32_t)((1u + n) * sizeof(uint32_t)) || n > cap) return -1;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t t = 0;
        if (dist_read_full(fd, &t, sizeof(t)) <= 0) return -1;
        tokens[i] = (int)(int32_t)ntohl(t);
    }
    *n_out = n;
    return 0;
}

static int dist_tp_recv_accept(ds4_dist_tp *tp, uint32_t *accept_len) {
    int fd = tp->fd;
    uint32_t type = 0, bytes = 0; char err[64];
    if (dist_read_frame_header(fd, &type, &bytes, err, sizeof(err)) <= 0) return -1;
    if (type != DS4_DIST_MSG_TP_ACCEPT || bytes != sizeof(uint32_t)) return -1;
    uint32_t v = 0;
    if (dist_read_full(fd, &v, sizeof(v)) <= 0) return -1;
    *accept_len = ntohl(v);
    return 0;
}

static int dist_run_tp_leader(ds4_engine *engine, const ds4_dist_options *opt,
                              const ds4_dist_generation_options *gen) {
    char err[256];
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, gen->ctx_size > 0 ? gen->ctx_size : 4096) != 0) {
        fprintf(stderr, "ds4: TP leader: failed to create session\n");
        return 1;
    }
    ds4_dist_tp *tp = ds4_engine_tp(engine);
    if (!tp) { fprintf(stderr, "ds4: TP leader: no peer link\n"); ds4_session_free(session); return 1; }

    ds4_tokens prompt = {0};
    if (gen->prompt && dist_prompt_is_rendered_chat(gen->prompt))
        ds4_tokenize_rendered_chat(engine, gen->prompt, &prompt);
    else
        ds4_encode_chat_prompt(engine, gen->system, gen->prompt, gen->think_mode, &prompt);
    if (prompt.len <= 0) { fprintf(stderr, "ds4: TP leader: empty prompt\n"); ds4_session_free(session); return 1; }

    if (dist_tp_send_prompt(tp, &prompt) != 0) {
        fprintf(stderr, "ds4: TP leader: failed to send prompt\n");
        ds4_tokens_free(&prompt); ds4_session_free(session); return 1;
    }

    const int vocab = ds4_engine_vocab_size(engine);
    const uint32_t last_layer = (uint32_t)ds4_engine_layer_count(engine) - 1u;
    float *logits = malloc((size_t)vocab * sizeof(float));
    int rc = 1;
    if (!logits) { ds4_tokens_free(&prompt); ds4_session_free(session); return 1; }

    if (ds4_session_layer_slice_reset(session, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4: TP leader: slice reset: %s\n", err);
        goto leader_done;
    }
    double t_prefill0 = dist_now_sec();
    uint32_t pchunk = dist_tp_prefill_chunk();
    for (uint32_t off = 0; off < (uint32_t)prompt.len; off += pchunk) {
        uint32_t n = (uint32_t)prompt.len - off;
        if (n > pchunk) n = pchunk;
        bool last = (off + n >= (uint32_t)prompt.len);
        if (ds4_session_eval_layer_slice(session, prompt.v + off, n, off,
                                         0, last_layer, NULL, NULL, last,
                                         last ? logits : NULL, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: TP leader: prefill failed: %s\n", err);
            goto leader_done;
        }
    }
    double t_prefill = dist_now_sec() - t_prefill0;

    int max_tokens = gen->n_predict > 0 ? gen->n_predict : 64;
    uint32_t pos = (uint32_t)prompt.len;
    int eos = ds4_token_eos(engine);
    fprintf(stderr, "ds4: TP leader: prompt=%d tokens, generating up to %d (tp_layers=%u)\n",
            prompt.len, max_tokens, opt->tp_layers ? opt->tp_layers : last_layer + 1u);
    rc = 0;
    double t_decode_sum = 0.0;   /* wall-clock spent inside eval during decode */
    int decoded = 0;
    /* copy-spec 整族已删除(2026-08-05 用户裁决): TP decode 纯 BARE 单 token 轮。 */
    for (int n = 0; n < max_tokens; ) {
        int token = dist_tp_argmax(logits, vocab);
        if (token == eos) break;
        double t_dec0 = dist_now_sec();
            if (dist_tp_send_batch(tp, &token, 1u) != 0) { fprintf(stderr, "ds4: TP leader: send step failed\n"); rc = 1; break; }
            size_t tl = 0; char *txt = ds4_token_text(engine, token, &tl);
            if (txt) { fwrite(txt, 1, tl, stdout); fflush(stdout); free(txt); }
            if (ds4_session_eval_layer_slice(session, &token, 1, pos, 0, last_layer,
                                             NULL, NULL, true, logits, err, sizeof(err)) != 0) {
                fprintf(stderr, "\nds4: TP leader: decode failed: %s\n", err); rc = 1; break;
            }
            t_decode_sum += dist_now_sec() - t_dec0;
                        pos++; decoded++; n++;
    }

    dist_tp_send_batch(tp, NULL, 0u); /* n=0 signals follower to stop */
    fputc('\n', stdout);
    fprintf(stderr,
            "ds4: TP leader: prefill %d tok in %.3fs (%.1f tok/s) | decode %d tok in %.3fs (%.2f tok/s)\n",
            prompt.len, t_prefill, t_prefill > 0 ? prompt.len / t_prefill : 0.0,
            decoded, t_decode_sum, t_decode_sum > 0 ? decoded / t_decode_sum : 0.0);
leader_done:
    free(logits);
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    return rc;
}

static int dist_run_tp_follower(ds4_engine *engine, const ds4_dist_options *opt, int ctx_size) {
    (void)opt;
    char err[256];
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size > 0 ? ctx_size : 4096) != 0) {
        fprintf(stderr, "ds4: TP follower: failed to create session\n");
        return 1;
    }
    ds4_dist_tp *tp = ds4_engine_tp(engine);
    if (!tp) { fprintf(stderr, "ds4: TP follower: no peer link\n"); ds4_session_free(session); return 1; }

    ds4_tokens prompt = {0};
    if (dist_tp_recv_prompt(tp, &prompt) != 0 || prompt.len <= 0) {
        fprintf(stderr, "ds4: TP follower: failed to receive prompt\n");
        ds4_tokens_free(&prompt); ds4_session_free(session); return 1;
    }
    const uint32_t last_layer = (uint32_t)ds4_engine_layer_count(engine) - 1u;
    if (ds4_session_layer_slice_reset(session, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4: TP follower: slice reset: %s\n", err);
        ds4_tokens_free(&prompt); ds4_session_free(session); return 1;
    }
    uint32_t pchunk = dist_tp_prefill_chunk();
    for (uint32_t off = 0; off < (uint32_t)prompt.len; off += pchunk) {
        uint32_t n = (uint32_t)prompt.len - off;
        if (n > pchunk) n = pchunk;
        if (ds4_session_eval_layer_slice(session, prompt.v + off, n, off,
                                         0, last_layer, NULL, NULL, false, NULL, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: TP follower: prefill failed: %s\n", err);
            ds4_tokens_free(&prompt); ds4_session_free(session); return 1;
        }
    }
    fprintf(stderr, "ds4: TP follower: prefilled %d prompt tokens, following leader\n", prompt.len);
    uint32_t pos = (uint32_t)prompt.len;
    /* copy-spec: mirror the leader's batch protocol. n=1 -> bare round (Phase-3
     * expert-split single-token, lockstep AR); n>1 -> verify batch (full, no AR)
     * then recv accept_len and roll the slice KV back to the accepted boundary so
     * both peers stay in sync; n=0 -> stop. row_logits is required by
     * verify_batch_argmax even though the follower discards it. */
    const int vocab = ds4_engine_vocab_size(engine);
    float *row_logits = malloc((size_t)64 * (size_t)vocab * sizeof(float));
    int batch[64];
    while (row_logits) {
        uint32_t n = 0;
        if (dist_tp_recv_batch(tp, batch, 64u, &n) != 0) { fprintf(stderr, "ds4: TP follower: recv batch failed\n"); break; }
        if (n == 0) break;   /* leader signaled stop */
        if (n == 1) {
            if (ds4_session_eval_layer_slice(session, batch, 1, pos, 0, last_layer,
                                             NULL, NULL, false, NULL, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4: TP follower: decode failed: %s\n", err); break;
            }
            pos++;
        } else {
            if (ds4_session_verify_batch_argmax(session, batch, n, pos, 0, last_layer,
                                                NULL, row_logits, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4: TP follower: verify batch failed: %s\n", err); break;
            }
            uint32_t accept_len = 0;
            if (dist_tp_recv_accept(tp, &accept_len) != 0) { fprintf(stderr, "ds4: TP follower: recv accept failed\n"); break; }
            (void)ds4_session_layer_slice_rollback(session, pos + accept_len, err, sizeof(err));
            pos += accept_len;
        }
    }
    free(row_logits);
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    return 0;
}

int ds4_dist_run(ds4_engine *engine, const ds4_dist_options *opt, const ds4_dist_generation_options *gen) {
    if (!engine || !opt) {
        fprintf(stderr, "ds4: distributed runtime requires an open engine and options\n");
        return 1;
    }
    char err[256];
    if (dist_validate_options(opt, err, sizeof(err)) != 0 ||
        dist_validate_layers_for_model(opt, (uint32_t)ds4_engine_layer_count(engine), err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4: %s\n", err);
        return 2;
    }

    signal(SIGPIPE, SIG_IGN);

    if (opt->tp_enabled) {
        if (opt->role == DS4_DISTRIBUTED_COORDINATOR) return dist_run_tp_leader(engine, opt, gen);
        if (opt->role == DS4_DISTRIBUTED_WORKER) return dist_run_tp_follower(engine, opt, gen ? gen->ctx_size : 0);
        return 1;
    }
    if (opt->role == DS4_DISTRIBUTED_COORDINATOR) {
        return dist_run_coordinator(engine, opt, gen);
    }
    if (opt->role == DS4_DISTRIBUTED_WORKER) {
        return dist_run_worker(engine, opt, gen ? gen->ctx_size : 0);
    }

    fprintf(stderr, "ds4: distributed runtime requested without a distributed role\n");
    return 1;
}

