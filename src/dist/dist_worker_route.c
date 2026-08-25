/* dist_worker_route.c — 机械拆自 ds4_distributed.c: worker 路由解析(Worker Route Parsing)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Worker Route Parsing And Forwarding
 * ========================================================================= */

bool dist_route_get_entry(
        const void *route_blob,
        uint32_t route_bytes,
        uint32_t route_count,
        uint32_t target_index,
        ds4_dist_route_entry *out,
        char *err,
        size_t errlen) {
    if (!route_blob || !out || target_index >= route_count) {
        if (errlen) snprintf(err, errlen, "invalid route entry index");
        return false;
    }
    const uint8_t *p = route_blob;
    uint32_t remaining = route_bytes;
    for (uint32_t i = 0; i < route_count; i++) {
        if (remaining < sizeof(ds4_dist_route_fixed)) {
            if (errlen) snprintf(err, errlen, "truncated route entry");
            return false;
        }
        ds4_dist_route_fixed fixed;
        memcpy(&fixed, p, sizeof(fixed));
        dist_route_from_wire(&fixed);
        p += sizeof(fixed);
        remaining -= (uint32_t)sizeof(fixed);
        if (fixed.host_len == 0 || fixed.host_len >= NI_MAXHOST || fixed.host_len > remaining) {
            if (errlen) snprintf(err, errlen, "invalid route host length");
            return false;
        }
        if (dist_bytes_have_nul(p, fixed.host_len)) {
            if (errlen) snprintf(err, errlen, "route host contains NUL bytes");
            return false;
        }
        if (i == target_index) {
            memcpy(out->host, p, fixed.host_len);
            out->host[fixed.host_len] = '\0';
            out->port = fixed.port;
            out->layer_start = fixed.layer_start;
            out->layer_end = fixed.layer_end;
            out->flags = fixed.flags;
            out->fd = -1;
            return true;
        }
        p += fixed.host_len;
        remaining -= fixed.host_len;
    }
    if (remaining != 0) {
        if (errlen) snprintf(err, errlen, "route payload has trailing bytes");
        return false;
    }
    if (errlen) snprintf(err, errlen, "route entry not found");
    return false;
}

bool dist_route_get_return_target(
        const void *route_blob,
        uint32_t route_bytes,
        uint32_t route_count,
        ds4_dist_route_return *out,
        char *err,
        size_t errlen) {
    if (!route_blob || !out || route_count == 0) {
        if (errlen) snprintf(err, errlen, "invalid route final destination");
        return false;
    }
    const uint8_t *p = route_blob;
    uint32_t remaining = route_bytes;
    for (uint32_t i = 0; i < route_count; i++) {
        if (remaining < sizeof(ds4_dist_route_fixed)) {
            if (errlen) snprintf(err, errlen, "truncated route entry");
            return false;
        }
        ds4_dist_route_fixed fixed;
        memcpy(&fixed, p, sizeof(fixed));
        dist_route_from_wire(&fixed);
        p += sizeof(fixed);
        remaining -= (uint32_t)sizeof(fixed);
        if (fixed.host_len > remaining) {
            if (errlen) snprintf(err, errlen, "invalid route host length");
            return false;
        }
        if (dist_bytes_have_nul(p, fixed.host_len)) {
            if (errlen) snprintf(err, errlen, "route host contains NUL bytes");
            return false;
        }
        p += fixed.host_len;
        remaining -= fixed.host_len;
    }
    if (remaining < sizeof(ds4_dist_route_return_fixed)) {
        if (errlen) snprintf(err, errlen, "route payload missing final destination");
        return false;
    }
    ds4_dist_route_return_fixed fixed;
    memcpy(&fixed, p, sizeof(fixed));
    dist_route_return_from_wire(&fixed);
    p += sizeof(fixed);
    remaining -= (uint32_t)sizeof(fixed);
    if (fixed.host_len >= NI_MAXHOST || fixed.host_len > remaining) {
        if (errlen) snprintf(err, errlen, "invalid route final destination host length");
        return false;
    }
    if (dist_bytes_have_nul(p, fixed.host_len)) {
        if (errlen) snprintf(err, errlen, "route final destination host contains NUL bytes");
        return false;
    }
    memset(out, 0, sizeof(*out));
    out->kind = fixed.kind;
    out->port = fixed.port;
    if (fixed.host_len) {
        memcpy(out->host, p, fixed.host_len);
        out->host[fixed.host_len] = '\0';
    }
    p += fixed.host_len;
    remaining -= fixed.host_len;
    if (remaining != 0) {
        if (errlen) snprintf(err, errlen, "route payload has trailing bytes");
        return false;
    }
    return true;
}

bool dist_route_validate_blob(
        const void *route_blob,
        uint32_t route_bytes,
        uint32_t route_count,
        uint32_t n_layers,
        char *err,
        size_t errlen) {
    if (route_count == 0) {
        if (route_bytes == 0) return true;
        if (errlen) snprintf(err, errlen, "route payload has entries without a route count");
        return false;
    }
    if (!route_blob) {
        if (errlen) snprintf(err, errlen, "route payload is missing");
        return false;
    }

    const uint8_t *p = route_blob;
    uint32_t remaining = route_bytes;
    uint32_t prev_end = UINT32_MAX;
    for (uint32_t i = 0; i < route_count; i++) {
        if (remaining < sizeof(ds4_dist_route_fixed)) {
            if (errlen) snprintf(err, errlen, "truncated route entry");
            return false;
        }
        ds4_dist_route_fixed fixed;
        memcpy(&fixed, p, sizeof(fixed));
        dist_route_from_wire(&fixed);
        p += sizeof(fixed);
        remaining -= (uint32_t)sizeof(fixed);

        if (fixed.host_len == 0 || fixed.host_len >= NI_MAXHOST || fixed.host_len > remaining) {
            if (errlen) snprintf(err, errlen, "invalid route host length");
            return false;
        }
        if (dist_bytes_have_nul(p, fixed.host_len)) {
            if (errlen) snprintf(err, errlen, "route host contains NUL bytes");
            return false;
        }
        if (fixed.port == 0 || fixed.port > 65535u) {
            if (errlen) snprintf(err, errlen, "invalid route port");
            return false;
        }
        if (fixed.layer_start >= n_layers || fixed.layer_end >= n_layers ||
            fixed.layer_end < fixed.layer_start) {
            if (errlen) snprintf(err, errlen, "invalid route layer range");
            return false;
        }
        if ((fixed.flags & ~DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0) {
            if (errlen) snprintf(err, errlen, "invalid route flags");
            return false;
        }
        if ((fixed.flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0 &&
            fixed.layer_end + 1u != n_layers) {
            if (errlen) snprintf(err, errlen, "route logits require final layer");
            return false;
        }
        if (i != 0 && fixed.layer_start != prev_end + 1u) {
            if (errlen) snprintf(err, errlen, "route layer ranges are not contiguous");
            return false;
        }

        p += fixed.host_len;
        remaining -= fixed.host_len;
        prev_end = fixed.layer_end;
    }
    if (remaining < sizeof(ds4_dist_route_return_fixed)) {
        if (errlen) snprintf(err, errlen, "route payload missing final destination");
        return false;
    }
    ds4_dist_route_return_fixed ret;
    memcpy(&ret, p, sizeof(ret));
    dist_route_return_from_wire(&ret);
    p += sizeof(ret);
    remaining -= (uint32_t)sizeof(ret);
    if (ret.host_len >= NI_MAXHOST || ret.host_len > remaining) {
        if (errlen) snprintf(err, errlen, "invalid route final destination host length");
        return false;
    }
    if (dist_bytes_have_nul(p, ret.host_len)) {
        if (errlen) snprintf(err, errlen, "route final destination host contains NUL bytes");
        return false;
    }
    if (ret.kind != DS4_DIST_ROUTE_RETURN_UPSTREAM) {
        if (errlen) snprintf(err, errlen, "unsupported route final destination");
        return false;
    }
    if (ret.host_len != 0 || ret.port != 0) {
        if (errlen) snprintf(err, errlen, "invalid upstream route final destination");
        return false;
    }
    p += ret.host_len;
    remaining -= ret.host_len;
    if (remaining != 0) {
        if (errlen) snprintf(err, errlen, "route payload has trailing bytes");
        return false;
    }
    return true;
}

int dist_send_work_frame(
        int fd,
        const ds4_dist_work_fixed *work,
        const int *tokens,
        const float *input_hc,
        const void *route_blob) {
    if (!work || !tokens || work->n_tokens == 0) return -1;
    const uint64_t token_bytes = (uint64_t)work->n_tokens * sizeof(uint32_t);
    if (token_bytes > UINT32_MAX || work->token_bytes != (uint32_t)token_bytes) return -1;
    if (work->input_hc_bytes != 0 && !input_hc) return -1;
    if (work->route_bytes != 0 && !route_blob) return -1;
    uint64_t input_hc_values = 0;
    if (work->input_hc_bytes != 0 &&
        !dist_activation_values_from_wire_bytes(work->input_hc_bits,
                                                work->input_hc_bytes,
                                                &input_hc_values))
        return -1;
    const uint64_t frame_bytes = sizeof(ds4_dist_work_fixed) +
                                 (uint64_t)work->token_bytes +
                                 work->input_hc_bytes +
                                 work->route_bytes;
    if (frame_bytes > UINT32_MAX) return -1;

    ds4_dist_work_fixed wire = *work;
    dist_work_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_WORK, (uint32_t)frame_bytes) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (uint32_t i = 0; i < work->n_tokens; i++) {
        uint32_t t = htonl((uint32_t)tokens[i]);
        if (dist_write_full(fd, &t, sizeof(t)) != 0) return -1;
    }
    if (work->input_hc_bytes &&
        dist_write_activation_payload(fd,
                                      input_hc,
                                      input_hc_values,
                                      work->input_hc_bits) != 0)
        return -1;
    if (work->route_bytes && dist_write_full(fd, route_blob, work->route_bytes) != 0) return -1;
    return 0;
}

int dist_worker_upstream_send_work_result(
        ds4_dist_worker_upstream *upstream,
        uint64_t request_id,
        uint64_t result_hash,
        uint32_t status,
        uint32_t result_kind,
        uint32_t payload_bits,
        const ds4_dist_telemetry_fixed *telemetry,
        uint32_t telemetry_count,
        const void *payload,
        uint32_t payload_bytes,
        const uint32_t *draft_tokens,
        uint32_t draft_count) {
    pthread_mutex_lock(&upstream->write_mu);
    int rc = dist_send_work_result(upstream->fd,
                                   request_id,
                                   result_hash,
                                   status,
                                   result_kind,
                                   payload_bits,
                                   telemetry,
                                   telemetry_count,
                                   payload,
                                   payload_bytes,
                                   draft_tokens,
                                   draft_count);
    pthread_mutex_unlock(&upstream->write_mu);
    return rc;
}

int dist_worker_upstream_send_work_error(
        ds4_dist_worker_upstream *upstream,
        uint64_t request_id,
        const char *msg) {
    pthread_mutex_lock(&upstream->write_mu);
    int rc = dist_send_work_error(upstream->fd, request_id, msg);
    pthread_mutex_unlock(&upstream->write_mu);
    return rc;
}

