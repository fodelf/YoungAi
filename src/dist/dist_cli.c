/* dist_cli.c — 机械拆自 ds4_distributed.c: CLI 解析与公共入口(CLI Option Parsing And Public Entrypoint)。行为零变化。 */
#include "dist_internal.h"
#ifndef DS4_NO_GPU
#include "ds4_gpu.h"
#endif

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
        "  --dist-prefill-cap N\n"
        "      Coordinator session prefill batch cap (0 = auto); bounds the routed-expert\n"
        "      working set of wide prefill batches on offload hosts.\n"
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
        "  --expert-fetch-serve\n"
        "      Serve routed-expert byte ranges from this host's model file to the peer\n"
        "      (dual-host expert streaming; pairs with the peer's expert-fetch client).\n"
        "  --expert-fetch-port N\n"
        "      Listen port for --expert-fetch-serve. Default: 5606.\n"
        "  --expert-fetch-dial HOST PORT\n"
        "      Serve-dial mode: this host dials the peer's expert-fetch accept listener\n"
        "      instead of listening (for peers whose outbound connect is broken).\n"
        "  --expert-fetch-host HOST\n"
        "      Client half: pull routed-expert bytes from the peer serving at HOST\n"
        "      (port from --expert-fetch-port, default 5606). GPU builds only.\n"
        "  --expert-fetch-accept-port N\n"
        "      Client half, accept mode: LISTEN on N for the peer's serve-dial. GPU builds only.\n"
        "  --expert-stage\n"
        "      Stage predicted next-layer experts from the peer into RAM one layer\n"
        "      ahead (needs a configured expert-fetch client). GPU builds only.\n"
        "  --reverse-connect\n"
        "      Flip who dials whom: the coordinator connects to a LISTENING worker\n"
        "      (pipeline and TP modes). For hosts where one connect direction fails\n"
        "      (macOS Local Network privacy: in-process connect over the bridge gets\n"
        "      EHOSTUNREACH). Pass the same flag on both peers; address flags swap:\n"
        "      coordinator takes --coordinator HOST PORT (the worker's listen address),\n"
        "      worker takes --listen HOST PORT.\n"
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
    if (!strcmp(arg, "--dist-prefill-cap")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        if (!dist_parse_positive_u32(value, arg, &opt->prefill_cap, err, errlen)) {
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
    if (!strcmp(arg, "--expert-fetch-serve")) {
        g_efetch_serve = 1;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--expert-fetch-port")) {
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        uint32_t port = 0;
        if (!dist_parse_positive_u32(value, arg, &port, err, errlen) || port > 65535u) {
            if (errlen) snprintf(err, errlen, "--expert-fetch-port must be 1..65535");
            return DS4_DIST_CLI_ERROR;
        }
        g_efetch_port = (int)port;
#ifndef DS4_NO_GPU
        /* 同一 rendezvous 端口: serve 半边在这听, client 半边往这拨。 */
        ds4_gpu_set_expert_fetch_client(NULL, (int)port, 0);
#endif
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--expert-fetch-host")) {
        const char *host = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!host) return DS4_DIST_CLI_ERROR;
#ifndef DS4_NO_GPU
        ds4_gpu_set_expert_fetch_client(host, 0, 0);
        return DS4_DIST_CLI_MATCHED;
#else
        if (errlen) snprintf(err, errlen, "%s is not supported in CPU builds", arg);
        return DS4_DIST_CLI_ERROR;
#endif
    }
    if (!strcmp(arg, "--expert-fetch-accept-port")) {
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        uint32_t port = 0;
        if (!dist_parse_positive_u32(value, arg, &port, err, errlen) || port > 65535u) {
            if (errlen) snprintf(err, errlen, "--expert-fetch-accept-port must be 1..65535");
            return DS4_DIST_CLI_ERROR;
        }
#ifndef DS4_NO_GPU
        ds4_gpu_set_expert_fetch_client(NULL, 0, (int)port);
        return DS4_DIST_CLI_MATCHED;
#else
        if (errlen) snprintf(err, errlen, "%s is not supported in CPU builds", arg);
        return DS4_DIST_CLI_ERROR;
#endif
    }
    if (!strcmp(arg, "--expert-stage")) {
#ifndef DS4_NO_GPU
        ds4_gpu_set_expert_stage(1);
        return DS4_DIST_CLI_MATCHED;
#else
        if (errlen) snprintf(err, errlen, "%s is not supported in CPU builds", arg);
        return DS4_DIST_CLI_ERROR;
#endif
    }
    if (!strcmp(arg, "--expert-fetch-dial")) {
        const char *host = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!host) return DS4_DIST_CLI_ERROR;
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        uint32_t port = 0;
        if (!dist_parse_positive_u32(value, arg, &port, err, errlen) || port > 65535u) {
            if (errlen) snprintf(err, errlen, "--expert-fetch-dial PORT must be 1..65535");
            return DS4_DIST_CLI_ERROR;
        }
        g_efetch_dial_host = host;
        g_efetch_dial_port = (int)port;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--reverse-connect")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        opt->reverse_connect = true;
        g_dist_reverse_connect = 1;
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

