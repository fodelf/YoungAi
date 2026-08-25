/* dist_cli.c — 机械拆自 ds4_distributed.c: CLI 解析与公共入口(CLI Option Parsing And Public Entrypoint)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * CLI Option Parsing And Public Entrypoint
 * ========================================================================= */

static bool dist_parse_role(const char *s, ds4_distributed_role *out) {
    if (!s || !out) return false;
    if (!strcmp(s, "none")) {
        *out = DS4_DISTRIBUTED_NONE;
        return true;
    }
    if (!strcmp(s, "coordinator")) {
        *out = DS4_DISTRIBUTED_COORDINATOR;
        return true;
    }
    if (!strcmp(s, "worker")) {
        *out = DS4_DISTRIBUTED_WORKER;
        return true;
    }
    return false;
}

static bool dist_parse_u32_component(const char *p, size_t len, uint32_t *out) {
    if (!p || len == 0 || !out) return false;
    char buf[32];
    if (len >= sizeof(buf)) return false;
    memcpy(buf, p, len);
    buf[len] = '\0';

    errno = 0;
    char *end = NULL;
    unsigned long v = strtoul(buf, &end, 10);
    if (errno != 0 || end == buf || *end != '\0' || v > UINT32_MAX) return false;
    *out = (uint32_t)v;
    return true;
}

static bool dist_parse_layers(const char *s, ds4_distributed_layers *out, char *err, size_t errlen) {
    if (!s || !out) {
        if (errlen) snprintf(err, errlen, "missing layer range");
        return false;
    }

    const char *colon = strchr(s, ':');
    if (!colon || colon == s || colon[1] == '\0') {
        if (errlen) snprintf(err, errlen, "expected A:B or A:output");
        return false;
    }
    if (strchr(colon + 1, ':')) {
        if (errlen) snprintf(err, errlen, "layer range has too many ':' separators");
        return false;
    }

    ds4_distributed_layers parsed = {0};
    if (!dist_parse_u32_component(s, (size_t)(colon - s), &parsed.start)) {
        if (errlen) snprintf(err, errlen, "invalid start layer in %s", s);
        return false;
    }

    const char *end = colon + 1;
    if (!strcmp(end, "output")) {
        parsed.end = UINT32_MAX;
        parsed.has_output = true;
    } else {
        if (!dist_parse_u32_component(end, strlen(end), &parsed.end)) {
            if (errlen) snprintf(err, errlen, "invalid end layer in %s", s);
            return false;
        }
        if (parsed.end < parsed.start) {
            if (errlen) snprintf(err, errlen, "layer range end precedes start in %s", s);
            return false;
        }
    }

    parsed.set = true;
    *out = parsed;
    return true;
}

static const char *dist_cli_need_arg(
        int *index,
        int argc,
        char **argv,
        const char *arg,
        char *err,
        size_t errlen) {
    if (!index || !argv || *index + 1 >= argc) {
        if (errlen) snprintf(err, errlen, "%s requires an argument", arg);
        return NULL;
    }
    return argv[++*index];
}

static bool dist_cli_parse_port(const char *s, const char *arg, int *out, char *err, size_t errlen) {
    if (!s || !out) {
        if (errlen) snprintf(err, errlen, "%s requires a TCP port", arg);
        return false;
    }
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno != 0 || s[0] == '\0' || *end != '\0' || v <= 0 || v > 65535) {
        if (errlen) snprintf(err, errlen, "invalid value for %s: %s", arg, s);
        return false;
    }
    *out = (int)v;
    return true;
}

bool ds4_dist_enabled(const ds4_dist_options *opt) {
    return opt && opt->role != DS4_DISTRIBUTED_NONE;
}

ds4_dist_options *ds4_dist_options_create(void) {
    return calloc(1, sizeof(ds4_dist_options));
}

void ds4_dist_options_free(ds4_dist_options *opt) {
    free(opt);
}

void ds4_dist_usage(FILE *fp) {
    fprintf(fp,
        "  --role ROLE\n"
        "      Distributed role: coordinator or worker.\n"
        "  --layers A:B\n"
        "      Inclusive distributed layer slice, e.g. 10:20 or 21:output.\n"
        "  --listen HOST PORT\n"
        "      Coordinator TCP listen address. Workers may later use it to force their data listener.\n"
        "  --coordinator HOST PORT\n"
        "      Coordinator TCP address for --role worker.\n"
        "  --dist-prefill-chunk N\n"
        "      Coordinator prefill pipeline chunk size. Default: session cap, normally 4096.\n"
        "      Non-default values are experimental and can change logits unless validated.\n"
        "  --dist-prefill-window N\n"
        "      Coordinator max end-to-end prefill chunks in flight. Default: workers+2, capped at 8.\n"
        "  --dist-activation-bits N\n"
        "      Coordinator hidden-state transport width: 32, 16, or 8. Default: 32.\n"
        "  --dist-replay-check\n"
        "      Coordinator diagnostic: reset and replay the prompt, then compare logits.\n"
        "  --tp\n"
        "      Tensor-parallel mode (Stage 2 skeleton): both peers load the full\n"
        "      model and split the MoE down_proj, summing partial outputs via an\n"
        "      all-reduce. Use with --role coordinator/--listen and --role worker/--coordinator.\n"
        "  --tp-layers N\n"
        "      Apply the TP down_proj split only to the first N layers (implies --tp).\n"
        "      0 (default with --tp) means all layers; small N brings the path up on 2-3 layers.\n"
        "  --mtp-role worker|coordinator\n"
        "      Layer-pipeline only. 'worker': run the MTP drafter on the worker that\n"
        "      holds the final layers + output head (drafts from its own final hidden;\n"
        "      requires --mtp on that worker). 'coordinator': run the drafter on the\n"
        "      coordinator (holds output head + token_embd + MTP; worker runs only the\n"
        "      backbone slice and returns hidden state). Requires --mtp on that side.\n"
        "  --debug\n"
        "      Print coordinator route/debug logs. Workers keep their normal logs without this.\n"
    );
}

ds4_dist_cli_parse_result ds4_dist_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_dist_options *opt,
        char *err,
        size_t errlen) {
    if (!arg) return DS4_DIST_CLI_NOT_MATCHED;
    if (!strcmp(arg, "--role")) {
        const char *role = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!role) return DS4_DIST_CLI_ERROR;
        if (!opt || !dist_parse_role(role, &opt->role)) {
            if (errlen) snprintf(err, errlen,
                                 "invalid distributed role: %s (valid roles: none, coordinator, worker)",
                                 role);
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--layers")) {
        const char *layers = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!layers) return DS4_DIST_CLI_ERROR;
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        if (!dist_parse_layers(layers, &opt->layers, err, errlen)) {
            char detail[160];
            if (errlen && err[0] != '\0') {
                snprintf(detail, sizeof(detail), "%s", err);
                snprintf(err, errlen, "invalid --layers %s: %s", layers, detail);
            }
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--listen")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        if (opt->listen_host || opt->listen_port) {
            if (errlen) snprintf(err, errlen, "specify --listen only once");
            return DS4_DIST_CLI_ERROR;
        }
        const char *host = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!host) return DS4_DIST_CLI_ERROR;
        const char *port = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!port) return DS4_DIST_CLI_ERROR;
        if (!dist_cli_parse_port(port, arg, &opt->listen_port, err, errlen)) return DS4_DIST_CLI_ERROR;
        opt->listen_host = host;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--coordinator")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        if (opt->coordinator_host || opt->coordinator_port) {
            if (errlen) snprintf(err, errlen, "specify --coordinator only once");
            return DS4_DIST_CLI_ERROR;
        }
        const char *host = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!host) return DS4_DIST_CLI_ERROR;
        const char *port = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!port) return DS4_DIST_CLI_ERROR;
        if (!dist_cli_parse_port(port, arg, &opt->coordinator_port, err, errlen)) return DS4_DIST_CLI_ERROR;
        opt->coordinator_host = host;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-prefill-chunk")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        if (!dist_parse_positive_u32(value, arg, &opt->prefill_chunk, err, errlen)) {
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-prefill-window")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        if (!dist_parse_positive_u32(value, arg, &opt->prefill_window, err, errlen)) {
            return DS4_DIST_CLI_ERROR;
        }
        if (opt->prefill_window > 64u) {
            if (errlen) snprintf(err, errlen, "%s must be <= 64", arg);
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-activation-bits")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        uint32_t bits = 0;
        if (!dist_parse_positive_u32(value, arg, &bits, err, errlen)) {
            return DS4_DIST_CLI_ERROR;
        }
        if (!dist_activation_bits_valid(bits)) {
            if (errlen) snprintf(err, errlen, "%s must be 32, 16, or 8", arg);
            return DS4_DIST_CLI_ERROR;
        }
        opt->activation_bits = bits;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-replay-check")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        opt->replay_check = true;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--tp")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        opt->tp_enabled = true;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--tp-layers")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        if (!dist_parse_positive_u32(value, arg, &opt->tp_layers, err, errlen)) {
            return DS4_DIST_CLI_ERROR;
        }
        opt->tp_enabled = true; /* --tp-layers implies TP mode */
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--debug")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        opt->debug = true;
        return DS4_DIST_CLI_MATCHED;
    }
    return DS4_DIST_CLI_NOT_MATCHED;
}

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

    /* TP reverse-connect (DS4_TP_REVERSE_CONNECT=1) flips network roles so the
     * coordinator connects and the worker listens — used to dodge a host where
     * one connect direction fails. The address flags swap accordingly. */
    bool tp_rev = false;
    if (opt->tp_enabled) {
        const char *rev = getenv("DS4_TP_REVERSE_CONNECT");
        tp_rev = (rev && *rev && rev[0] != '0');
    }
    /* Layer-pipeline reverse-connect (DS4_DIST_REVERSE_CONNECT, legacy
     * DS4_TP_REVERSE_CONNECT): coordinator dials a listening worker. Only for the
     * non-TP layer-pipeline path; same address-flag swap as TP reverse. */
    bool dist_rev = false;
    if (!opt->tp_enabled) {
        const char *rev = getenv("DS4_DIST_REVERSE_CONNECT");
        if (!rev || !*rev) rev = getenv("DS4_TP_REVERSE_CONNECT");
        dist_rev = (rev && *rev && rev[0] != '0');
    }

    if (opt->role == DS4_DISTRIBUTED_COORDINATOR) {
        if (tp_rev || dist_rev) {
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
        if (tp_rev || dist_rev) {
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

