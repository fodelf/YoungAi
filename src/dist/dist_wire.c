/* dist_wire.c — 机械拆自 ds4_distributed.c: 线上编码助手(Wire Encoding Helpers)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Wire Encoding Helpers
 * ========================================================================= */

void dist_hello_to_wire(ds4_dist_hello_fixed *h) {
    h->model_id = htonl(h->model_id);
    h->quant_bits = htonl(h->quant_bits);
    h->layer_start = htonl(h->layer_start);
    h->layer_end = htonl(h->layer_end);
    h->has_output = htonl(h->has_output);
    h->has_hidden = htonl(h->has_hidden);
    h->ctx_size = htonl(h->ctx_size);
    h->n_layers = htonl(h->n_layers);
    h->listen_port = htonl(h->listen_port);
    h->model_name_len = htonl(h->model_name_len);
}

void dist_hello_from_wire(ds4_dist_hello_fixed *h) {
    h->model_id = ntohl(h->model_id);
    h->quant_bits = ntohl(h->quant_bits);
    h->layer_start = ntohl(h->layer_start);
    h->layer_end = ntohl(h->layer_end);
    h->has_output = ntohl(h->has_output);
    h->has_hidden = ntohl(h->has_hidden);
    h->ctx_size = ntohl(h->ctx_size);
    h->n_layers = ntohl(h->n_layers);
    h->listen_port = ntohl(h->listen_port);
    h->model_name_len = ntohl(h->model_name_len);
}

uint64_t dist_u64_from_halves(uint32_t hi, uint32_t lo) {
    return ((uint64_t)hi << 32) | lo;
}

/* FNV-1a over little-endian token IDs.  This is not a security primitive; it is
 * a compact session invariant so distributed workers can reject same-position
 * but different-prefix KV state before doing layer work. */

static uint64_t dist_token_hash_update(uint64_t h, int token) {
    uint32_t t = (uint32_t)token;
    for (int i = 0; i < 4; i++) {
        h ^= (uint64_t)((t >> (i * 8)) & 0xffu);
        h *= DS4_DIST_TOKEN_HASH_PRIME;
    }
    return h;
}

uint64_t dist_token_hash_update_span(uint64_t h, const int *tokens, uint32_t n_tokens) {
    for (uint32_t i = 0; i < n_tokens; i++) h = dist_token_hash_update(h, tokens[i]);
    return h;
}

uint64_t dist_token_hash_prefix(const int *tokens, uint32_t n_tokens) {
    return dist_token_hash_update_span(DS4_DIST_TOKEN_HASH_INIT, tokens, n_tokens);
}

int dist_session_token_hash_prefix(
        ds4_session *session,
        uint32_t n_tokens,
        uint64_t *hash,
        char *err,
        size_t errlen) {
    const ds4_tokens *checkpoint = ds4_session_tokens(session);
    if (!hash || !checkpoint || checkpoint->len < 0 || (uint32_t)checkpoint->len < n_tokens) {
        if (errlen) snprintf(err, errlen, "distributed session has no %u-token prefix", n_tokens);
        return 1;
    }
    *hash = dist_token_hash_prefix(checkpoint->v, n_tokens);
    return 0;
}

bool dist_bytes_have_nul(const void *p, uint32_t len) {
    return len != 0 && memchr(p, '\0', len) != NULL;
}

void dist_u64_to_halves(uint64_t v, uint32_t *hi, uint32_t *lo) {
    *hi = (uint32_t)(v >> 32);
    *lo = (uint32_t)v;
}

void dist_work_from_wire(ds4_dist_work_fixed *w) {
    w->model_id = ntohl(w->model_id);
    w->session_hi = ntohl(w->session_hi);
    w->session_lo = ntohl(w->session_lo);
    w->request_hi = ntohl(w->request_hi);
    w->request_lo = ntohl(w->request_lo);
    w->prefix_hash_hi = ntohl(w->prefix_hash_hi);
    w->prefix_hash_lo = ntohl(w->prefix_hash_lo);
    w->result_hash_hi = ntohl(w->result_hash_hi);
    w->result_hash_lo = ntohl(w->result_hash_lo);
    w->pos0 = ntohl(w->pos0);
    w->n_tokens = ntohl(w->n_tokens);
    w->layer_start = ntohl(w->layer_start);
    w->layer_end = ntohl(w->layer_end);
    w->flags = ntohl(w->flags);
    w->token_bytes = ntohl(w->token_bytes);
    w->input_hc_bytes = ntohl(w->input_hc_bytes);
    w->input_hc_bits = ntohl(w->input_hc_bits);
    w->route_count = ntohl(w->route_count);
    w->route_index = ntohl(w->route_index);
    w->route_bytes = ntohl(w->route_bytes);
    w->draft_cap = ntohl(w->draft_cap);
    w->accept_len = ntohl(w->accept_len);
}

void dist_work_to_wire(ds4_dist_work_fixed *w) {
    w->model_id = htonl(w->model_id);
    w->session_hi = htonl(w->session_hi);
    w->session_lo = htonl(w->session_lo);
    w->request_hi = htonl(w->request_hi);
    w->request_lo = htonl(w->request_lo);
    w->prefix_hash_hi = htonl(w->prefix_hash_hi);
    w->prefix_hash_lo = htonl(w->prefix_hash_lo);
    w->result_hash_hi = htonl(w->result_hash_hi);
    w->result_hash_lo = htonl(w->result_hash_lo);
    w->pos0 = htonl(w->pos0);
    w->n_tokens = htonl(w->n_tokens);
    w->layer_start = htonl(w->layer_start);
    w->layer_end = htonl(w->layer_end);
    w->flags = htonl(w->flags);
    w->token_bytes = htonl(w->token_bytes);
    w->input_hc_bytes = htonl(w->input_hc_bytes);
    w->input_hc_bits = htonl(w->input_hc_bits);
    w->route_count = htonl(w->route_count);
    w->route_index = htonl(w->route_index);
    w->route_bytes = htonl(w->route_bytes);
    w->draft_cap = htonl(w->draft_cap);
    w->accept_len = htonl(w->accept_len);
}

void dist_route_from_wire(ds4_dist_route_fixed *r) {
    r->host_len = ntohl(r->host_len);
    r->port = ntohl(r->port);
    r->layer_start = ntohl(r->layer_start);
    r->layer_end = ntohl(r->layer_end);
    r->flags = ntohl(r->flags);
}

void dist_route_to_wire(ds4_dist_route_fixed *r) {
    r->host_len = htonl(r->host_len);
    r->port = htonl(r->port);
    r->layer_start = htonl(r->layer_start);
    r->layer_end = htonl(r->layer_end);
    r->flags = htonl(r->flags);
}

void dist_route_return_from_wire(ds4_dist_route_return_fixed *r) {
    r->kind = ntohl(r->kind);
    r->host_len = ntohl(r->host_len);
    r->port = ntohl(r->port);
}

void dist_route_return_to_wire(ds4_dist_route_return_fixed *r) {
    r->kind = htonl(r->kind);
    r->host_len = htonl(r->host_len);
    r->port = htonl(r->port);
}

void dist_result_to_wire(ds4_dist_result_fixed *r) {
    r->request_hi = htonl(r->request_hi);
    r->request_lo = htonl(r->request_lo);
    r->result_hash_hi = htonl(r->result_hash_hi);
    r->result_hash_lo = htonl(r->result_hash_lo);
    r->status = htonl(r->status);
    r->result_kind = htonl(r->result_kind);
    r->telemetry_count = htonl(r->telemetry_count);
    r->telemetry_bytes = htonl(r->telemetry_bytes);
    r->payload_bytes = htonl(r->payload_bytes);
    r->payload_bits = htonl(r->payload_bits);
    r->draft_count = htonl(r->draft_count);
}

void dist_result_from_wire(ds4_dist_result_fixed *r) {
    r->request_hi = ntohl(r->request_hi);
    r->request_lo = ntohl(r->request_lo);
    r->result_hash_hi = ntohl(r->result_hash_hi);
    r->result_hash_lo = ntohl(r->result_hash_lo);
    r->status = ntohl(r->status);
    r->result_kind = ntohl(r->result_kind);
    r->telemetry_count = ntohl(r->telemetry_count);
    r->telemetry_bytes = ntohl(r->telemetry_bytes);
    r->payload_bytes = ntohl(r->payload_bytes);
    r->payload_bits = ntohl(r->payload_bits);
    r->draft_count = ntohl(r->draft_count);
}

void dist_snapshot_req_to_wire(ds4_dist_snapshot_req_fixed *s) {
    s->model_id = htonl(s->model_id);
    s->session_hi = htonl(s->session_hi);
    s->session_lo = htonl(s->session_lo);
    s->request_hi = htonl(s->request_hi);
    s->request_lo = htonl(s->request_lo);
    s->token_hash_hi = htonl(s->token_hash_hi);
    s->token_hash_lo = htonl(s->token_hash_lo);
    s->token_count = htonl(s->token_count);
    s->layer_start = htonl(s->layer_start);
    s->layer_end = htonl(s->layer_end);
}

void dist_snapshot_req_from_wire(ds4_dist_snapshot_req_fixed *s) {
    s->model_id = ntohl(s->model_id);
    s->session_hi = ntohl(s->session_hi);
    s->session_lo = ntohl(s->session_lo);
    s->request_hi = ntohl(s->request_hi);
    s->request_lo = ntohl(s->request_lo);
    s->token_hash_hi = ntohl(s->token_hash_hi);
    s->token_hash_lo = ntohl(s->token_hash_lo);
    s->token_count = ntohl(s->token_count);
    s->layer_start = ntohl(s->layer_start);
    s->layer_end = ntohl(s->layer_end);
}

void dist_snapshot_begin_to_wire(ds4_dist_snapshot_begin_fixed *s) {
    s->model_id = htonl(s->model_id);
    s->session_hi = htonl(s->session_hi);
    s->session_lo = htonl(s->session_lo);
    s->request_hi = htonl(s->request_hi);
    s->request_lo = htonl(s->request_lo);
    s->token_hash_hi = htonl(s->token_hash_hi);
    s->token_hash_lo = htonl(s->token_hash_lo);
    s->token_count = htonl(s->token_count);
    s->layer_start = htonl(s->layer_start);
    s->layer_end = htonl(s->layer_end);
    s->payload_hi = htonl(s->payload_hi);
    s->payload_lo = htonl(s->payload_lo);
    s->status = htonl(s->status);
    s->token_bytes = htonl(s->token_bytes);
    s->message_bytes = htonl(s->message_bytes);
}

void dist_snapshot_begin_from_wire(ds4_dist_snapshot_begin_fixed *s) {
    s->model_id = ntohl(s->model_id);
    s->session_hi = ntohl(s->session_hi);
    s->session_lo = ntohl(s->session_lo);
    s->request_hi = ntohl(s->request_hi);
    s->request_lo = ntohl(s->request_lo);
    s->token_hash_hi = ntohl(s->token_hash_hi);
    s->token_hash_lo = ntohl(s->token_hash_lo);
    s->token_count = ntohl(s->token_count);
    s->layer_start = ntohl(s->layer_start);
    s->layer_end = ntohl(s->layer_end);
    s->payload_hi = ntohl(s->payload_hi);
    s->payload_lo = ntohl(s->payload_lo);
    s->status = ntohl(s->status);
    s->token_bytes = ntohl(s->token_bytes);
    s->message_bytes = ntohl(s->message_bytes);
}

void dist_snapshot_chunk_to_wire(ds4_dist_snapshot_chunk_fixed *s) {
    s->request_hi = htonl(s->request_hi);
    s->request_lo = htonl(s->request_lo);
    s->chunk_bytes = htonl(s->chunk_bytes);
}

void dist_snapshot_chunk_from_wire(ds4_dist_snapshot_chunk_fixed *s) {
    s->request_hi = ntohl(s->request_hi);
    s->request_lo = ntohl(s->request_lo);
    s->chunk_bytes = ntohl(s->chunk_bytes);
}

void dist_snapshot_done_to_wire(ds4_dist_snapshot_done_fixed *s) {
    s->request_hi = htonl(s->request_hi);
    s->request_lo = htonl(s->request_lo);
    s->status = htonl(s->status);
    s->message_bytes = htonl(s->message_bytes);
}

void dist_snapshot_done_from_wire(ds4_dist_snapshot_done_fixed *s) {
    s->request_hi = ntohl(s->request_hi);
    s->request_lo = ntohl(s->request_lo);
    s->status = ntohl(s->status);
    s->message_bytes = ntohl(s->message_bytes);
}

void dist_telemetry_to_wire(ds4_dist_telemetry_fixed *t) {
    t->layer_start = htonl(t->layer_start);
    t->layer_end = htonl(t->layer_end);
    t->route_index = htonl(t->route_index);
    t->pos0 = htonl(t->pos0);
    t->n_tokens = htonl(t->n_tokens);
    t->eval_usec = htonl(t->eval_usec);
    t->downstream_wait_usec = htonl(t->downstream_wait_usec);
    t->forward_send_usec = htonl(t->forward_send_usec);
    t->input_bytes = htonl(t->input_bytes);
    t->output_bytes = htonl(t->output_bytes);
}

void dist_telemetry_from_wire(ds4_dist_telemetry_fixed *t) {
    t->layer_start = ntohl(t->layer_start);
    t->layer_end = ntohl(t->layer_end);
    t->route_index = ntohl(t->route_index);
    t->pos0 = ntohl(t->pos0);
    t->n_tokens = ntohl(t->n_tokens);
    t->eval_usec = ntohl(t->eval_usec);
    t->downstream_wait_usec = ntohl(t->downstream_wait_usec);
    t->forward_send_usec = ntohl(t->forward_send_usec);
    t->input_bytes = ntohl(t->input_bytes);
    t->output_bytes = ntohl(t->output_bytes);
}

uint32_t dist_usec_since(double t0, double t1) {
    if (t1 <= t0) return 0;
    const double usec = (t1 - t0) * 1000000.0;
    if (usec >= (double)UINT32_MAX) return UINT32_MAX;
    return (uint32_t)(usec + 0.5);
}

