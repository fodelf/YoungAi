/* dist_cli_check.c — dist CLI 选项校验与引擎装配(从 dist_cli.c 拆出, 500 行守卫)。
 * 解析在 dist_cli.c; 这里只做 role/地址/reverse-connect 约束检查与 engine options 落位。 */
#include "dist_internal.h"

int dist_validate_options(const ds4_dist_options *opt, char *err, size_t errlen) {
    if (!opt) {
        if (errlen) snprintf(err, errlen, "missing distributed options");
        return 1;
    }

    if (opt->role == DS4_DISTRIBUTED_NONE) {
        if (opt->layers.set || opt->listen_host || opt->listen_port ||
            opt->coordinator_host || opt->coordinator_port ||
            opt->prefill_chunk != 0 || opt->prefill_window != 0 ||
            opt->activation_bits != 0) {
            if (errlen) snprintf(err, errlen, "distributed options require --role coordinator or --role worker");
            return 1;
        }
        return 0;
    }

    /* Tensor-parallel mode loads the whole model on every peer (it recombines
     * routed_out element-wise, not by layer slice), so it does not take a
     * --layers range. Pipeline (layer-slice) mode still requires it. */
    if (!opt->layers.set && !opt->tp_enabled) {
        if (errlen) snprintf(err, errlen, "--role %s requires --layers", dist_role_name(opt->role));
        return 1;
    }
    if (opt->prefill_window > 64u) {
        if (errlen) snprintf(err, errlen, "--dist-prefill-window must be <= 64");
        return 1;
    }
    if (opt->activation_bits != 0 && !dist_activation_bits_valid(opt->activation_bits)) {
        if (errlen) snprintf(err, errlen, "--dist-activation-bits must be 32, 16, or 8");
        return 1;
    }

    /* --reverse-connect flips network roles so the coordinator connects and the
     * worker listens — used to dodge a host where one connect direction fails.
     * The address flags swap accordingly (pipeline and TP modes alike). */
    const bool rev = opt->reverse_connect;

    if (opt->role == DS4_DISTRIBUTED_COORDINATOR) {
        if (rev) {
            if (!opt->coordinator_host || opt->coordinator_port <= 0) {
                if (errlen) snprintf(err, errlen, "--role coordinator (reverse-connect) requires --coordinator HOST PORT (the worker's listen address)");
                return 1;
            }
            return 0;
        }
        if (!opt->listen_host || opt->listen_port <= 0) {
            if (errlen) snprintf(err, errlen, "--role coordinator requires --listen HOST PORT");
            return 1;
        }
        if (opt->coordinator_host || opt->coordinator_port) {
            if (errlen) snprintf(err, errlen, "--role coordinator must not use --coordinator");
            return 1;
        }
        return 0;
    }

    if (opt->role == DS4_DISTRIBUTED_WORKER) {
        if (rev) {
            if (!opt->listen_host || opt->listen_port <= 0) {
                if (errlen) snprintf(err, errlen, "--role worker (reverse-connect) requires --listen HOST PORT");
                return 1;
            }
            return 0;
        }
        if (!opt->coordinator_host || opt->coordinator_port <= 0) {
            if (errlen) snprintf(err, errlen, "--role worker requires --coordinator HOST PORT");
            return 1;
        }
        if (opt->prefill_chunk != 0) {
            if (errlen) snprintf(err, errlen, "--dist-prefill-chunk requires --role coordinator");
            return 1;
        }
        if (opt->prefill_window != 0) {
            if (errlen) snprintf(err, errlen, "--dist-prefill-window requires --role coordinator");
            return 1;
        }
        if (opt->activation_bits != 0) {
            if (errlen) snprintf(err, errlen, "--dist-activation-bits requires --role coordinator");
            return 1;
        }
        return 0;
    }

    if (errlen) snprintf(err, errlen, "invalid distributed role");
    return 1;
}

int ds4_dist_prepare_engine_options(
        const ds4_dist_options *opt,
        ds4_engine_options *engine,
        char *err,
        size_t errlen) {
    if (dist_validate_options(opt, err, errlen) != 0) return 1;
    if (opt && opt->replay_check && opt->role != DS4_DISTRIBUTED_COORDINATOR) {
        if (errlen) snprintf(err, errlen, "--dist-replay-check requires --role coordinator");
        return 1;
    }
    if (engine && opt) {
        engine->distributed = *opt;
        if (opt->tp_enabled) {
            /* Tensor parallelism replicates the whole layer stack on both peers
             * and splits only the down_proj compute, so each machine must load
             * the full model (no pipeline slice). */
            engine->load_slice = false;
            engine->load_output = true;
        } else if (ds4_dist_enabled(opt)) {
            engine->load_slice = true;
            engine->load_layer_start = opt->layers.start;
            engine->load_layer_end = opt->layers.has_output ? UINT32_MAX : opt->layers.end;
            engine->load_output = opt->layers.has_output || opt->role == DS4_DISTRIBUTED_COORDINATOR;
        }
    }
    return 0;
}

int dist_validate_layers_for_model(const ds4_dist_options *opt, uint32_t n_layers, char *err, size_t errlen) {
    if (!opt || opt->role == DS4_DISTRIBUTED_NONE || !opt->layers.set) return 0;
    if (n_layers == 0) {
        if (errlen) snprintf(err, errlen, "model reports no layers");
        return 1;
    }

    const uint32_t last = n_layers - 1u;
    if (opt->layers.start > last) {
        if (errlen) snprintf(err, errlen, "layer range starts past final model layer %u", last);
        return 1;
    }
    if (!opt->layers.has_output && opt->layers.end > last) {
        if (errlen) snprintf(err, errlen, "layer range ends past final model layer %u", last);
        return 1;
    }
    if (opt->role == DS4_DISTRIBUTED_COORDINATOR && opt->layers.start != 0) {
        if (errlen) snprintf(err, errlen, "coordinator layer range must start at layer 0");
        return 1;
    }
    return 0;
}

