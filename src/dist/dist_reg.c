/* dist_reg.c — 机械拆自 ds4_distributed.c: worker 注册(Worker Registration)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Worker Registration
 * ========================================================================= */

int dist_send_hello(ds4_engine *engine, const ds4_dist_options *opt, int ctx_size, uint32_t listen_port, int fd) {
    uint32_t n_layers = (uint32_t)ds4_engine_layer_count(engine);
    const char *model_name = ds4_engine_model_name(engine);
    if (!model_name) model_name = "unknown";
    size_t model_name_len = strlen(model_name);
    if (model_name_len > DS4_DIST_MAX_MODEL_NAME) model_name_len = DS4_DIST_MAX_MODEL_NAME;

    ds4_dist_hello_fixed h = {
        (uint32_t)ds4_engine_model_id(engine),
        (uint32_t)ds4_engine_routed_quant_bits(engine),
        opt->layers.start,
        dist_resolved_layer_end(opt, n_layers),
        opt->layers.has_output ? 1u : 0u,
        1u,
        ctx_size > 0 ? (uint32_t)ctx_size : 0u,
        n_layers,
        listen_port,
        (uint32_t)model_name_len
    };
    ds4_dist_hello_fixed wire = h;
    dist_hello_to_wire(&wire);

    uint32_t bytes = (uint32_t)sizeof(wire) + (uint32_t)model_name_len;
    if (dist_write_frame_header(fd, DS4_DIST_MSG_HELLO, bytes) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    if (model_name_len && dist_write_full(fd, model_name, model_name_len) != 0) return -1;
    return 0;
}

int dist_recv_hello(int fd, ds4_dist_hello_fixed *hello, char *model_name, size_t model_name_cap, char *err, size_t errlen) {
    uint32_t type = 0, bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) return rc;
    if (type != DS4_DIST_MSG_HELLO) {
        if (errlen) snprintf(err, errlen, "expected HELLO frame, got type %u", type);
        dist_discard_bytes(fd, bytes);
        return -1;
    }
    if (bytes < sizeof(*hello) || bytes > sizeof(*hello) + DS4_DIST_MAX_MODEL_NAME) {
        if (errlen) snprintf(err, errlen, "invalid HELLO payload length %u", bytes);
        dist_discard_bytes(fd, bytes);
        return -1;
    }

    ds4_dist_hello_fixed wire;
    rc = dist_read_full(fd, &wire, sizeof(wire));
    if (rc <= 0) return rc == 0 ? 0 : -1;
    dist_hello_from_wire(&wire);

    uint32_t remaining = bytes - (uint32_t)sizeof(wire);
    if (wire.model_name_len != remaining || wire.model_name_len > DS4_DIST_MAX_MODEL_NAME) {
        if (errlen) snprintf(err, errlen, "invalid HELLO model name length %u", wire.model_name_len);
        dist_discard_bytes(fd, remaining);
        return -1;
    }

    if (model_name_cap) model_name[0] = '\0';
    if (wire.model_name_len) {
        char tmp[DS4_DIST_MAX_MODEL_NAME + 1u];
        rc = dist_read_full(fd, tmp, wire.model_name_len);
        if (rc <= 0) return rc == 0 ? 0 : -1;
        if (dist_bytes_have_nul(tmp, wire.model_name_len)) {
            if (errlen) snprintf(err, errlen, "HELLO model family contains NUL bytes");
            return -1;
        }
        tmp[wire.model_name_len] = '\0';
        if (model_name_cap) {
            snprintf(model_name, model_name_cap, "%s", tmp);
        }
    }

    *hello = wire;
    return 1;
}


bool dist_coordinator_debug_enabled(const ds4_dist_coordinator_state *state) {
    return state && state->debug;
}

