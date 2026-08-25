/* dist_util.c — 机械拆自 ds4_distributed.c: 小工具·本地文件助手·可调上限(Small Utilities / Local File Helpers / Tunable Limits)。行为零变化。 */
#include "dist_internal.h"


void dist_mtp_print_summary(ds4_dist_session *d, const char *tag) {
    if (!d || d->mtp_calls == 0) return;
    const double first_hit_pct = d->mtp_round1 ?
        100.0 * (double)d->mtp_first_hit / (double)d->mtp_round1 : 0.0;
    const double accept_per_call = d->mtp_calls ?
        (double)d->mtp_accept_tokens / (double)d->mtp_calls : 0.0;
    const uint64_t draft_accepts = d->mtp_accept_tokens > d->mtp_calls ?
        d->mtp_accept_tokens - d->mtp_calls : 0;
    const double draft_accept_pct = d->mtp_draft_tokens ?
        100.0 * (double)draft_accepts / (double)d->mtp_draft_tokens : 0.0;
    const double forwards_per_call = d->mtp_calls ?
        (double)d->mtp_forwards / (double)d->mtp_calls : 0.0;
    const double tok_per_forward = d->mtp_forwards ?
        (double)d->mtp_accept_tokens / (double)d->mtp_forwards : 0.0;
    fprintf(stderr,
            "ds4: dist-mtp %s: calls=%llu round1=%llu verify=%llu first_hit=%.2f%% "
            "accepted=%llu drafts=%llu draft_accept=%.2f%% tok/call=%.2f "
            "forwards=%llu fwd/call=%.2f tok/fwd=%.2f disabled=%llu\n",
            tag ? tag : "summary",
            (unsigned long long)d->mtp_calls,
            (unsigned long long)d->mtp_round1,
            (unsigned long long)d->mtp_verify,
            first_hit_pct,
            (unsigned long long)d->mtp_accept_tokens,
            (unsigned long long)d->mtp_draft_tokens,
            draft_accept_pct,
            accept_per_call,
            (unsigned long long)d->mtp_forwards,
            forwards_per_call,
            tok_per_forward,
            (unsigned long long)d->mtp_disabled_cycles);
}

/* =========================================================================
 * Small Utilities And Forward Declarations
 * ========================================================================= */

uint32_t dist_prefill_send_depth(uint32_t chunk_count) {
    uint32_t depth = 2;
    const char *env = getenv("DS4_DIST_PREFILL_SEND_DEPTH");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 1 && v <= 8) {
            depth = (uint32_t)v;
        }
    }
    if (chunk_count != 0 && depth > chunk_count) depth = chunk_count;
    return depth ? depth : 1;
}



uint32_t dist_resolved_layer_end(const ds4_dist_options *opt, uint32_t n_layers) {
    if (opt->layers.has_output) return n_layers - 1u;
    return opt->layers.end;
}

const char *dist_role_name(ds4_distributed_role role) {
    switch (role) {
    case DS4_DISTRIBUTED_NONE:        return "none";
    case DS4_DISTRIBUTED_COORDINATOR: return "coordinator";
    case DS4_DISTRIBUTED_WORKER:      return "worker";
    }
    return "unknown";
}

void dist_sleep_reconnect(void) {
    sleep(1);
}

double dist_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

/* =========================================================================
 * Local File And Size Helpers
 * ========================================================================= */

int dist_payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            if (errlen) snprintf(err, errlen, "failed to write distributed payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}

int dist_payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        if (errlen) snprintf(err, errlen, "truncated distributed payload");
        return 1;
    }
    uint8_t *p = ptr;
    uint64_t original = bytes;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            if (errlen) snprintf(err, errlen, "failed to read distributed payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

int dist_payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4] = {
        (uint8_t)v,
        (uint8_t)(v >> 8),
        (uint8_t)(v >> 16),
        (uint8_t)(v >> 24),
    };
    return dist_payload_write_bytes(fp, b, sizeof(b), err, errlen);
}

int dist_payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (dist_payload_read_bytes(fp, b, sizeof(b), remaining, err, errlen) != 0) return 1;
    *v = (uint32_t)b[0] |
         ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) |
         ((uint32_t)b[3] << 24);
    return 0;
}

int dist_payload_copy_bytes(
        FILE *src,
        FILE *dst,
        uint64_t bytes,
        uint64_t *remaining,
        char *err,
        size_t errlen) {
    if (remaining && *remaining < bytes) {
        if (errlen) snprintf(err, errlen, "truncated distributed payload");
        return 1;
    }
    uint8_t *buf = malloc(DS4_DIST_SNAPSHOT_CHUNK_BYTES);
    if (!buf) {
        if (errlen) snprintf(err, errlen, "out of memory copying distributed payload");
        return 1;
    }
    int rc = 0;
    uint64_t left = bytes;
    while (left != 0) {
        const size_t n = left > DS4_DIST_SNAPSHOT_CHUNK_BYTES ?
            DS4_DIST_SNAPSHOT_CHUNK_BYTES : (size_t)left;
        if (fread(buf, 1, n, src) != n) {
            if (errlen) snprintf(err, errlen, "failed to read distributed payload");
            rc = 1;
            break;
        }
        if (fwrite(buf, 1, n, dst) != n) {
            if (errlen) snprintf(err, errlen, "failed to write distributed payload");
            rc = 1;
            break;
        }
        left -= n;
    }
    free(buf);
    if (rc == 0 && remaining) *remaining -= bytes;
    return rc;
}

int dist_copy_file_range(
        FILE *src,
        uint64_t offset,
        uint64_t bytes,
        FILE *dst,
        char *err,
        size_t errlen) {
    if (offset > (uint64_t)LLONG_MAX || fseeko(src, (off_t)offset, SEEK_SET) != 0) {
        if (errlen) snprintf(err, errlen, "failed to seek distributed KV shard");
        return 1;
    }
    return dist_payload_copy_bytes(src, dst, bytes, NULL, err, errlen);
}

int dist_rewind_file(FILE *fp, const char *what, char *err, size_t errlen) {
    if (fflush(fp) != 0 || fseeko(fp, 0, SEEK_SET) != 0) {
        if (errlen) snprintf(err, errlen, "failed to rewind %s", what);
        return 1;
    }
    return 0;
}

int dist_measure_file(FILE *fp, uint64_t *bytes, const char *what, char *err, size_t errlen) {
    if (!bytes) return 1;
    if (fflush(fp) != 0) {
        if (errlen) snprintf(err, errlen, "failed to flush %s", what);
        return 1;
    }
    off_t pos = ftello(fp);
    if (pos < 0) {
        if (errlen) snprintf(err, errlen, "failed to measure %s", what);
        return 1;
    }
    *bytes = (uint64_t)pos;
    return 0;
}

FILE *dist_tmpfile_or_err(const char *what, char *err, size_t errlen) {
    FILE *fp = tmpfile();
    if (!fp && errlen) snprintf(err, errlen, "failed to create %s temp file: %s",
                                what, strerror(errno));
    return fp;
}

bool dist_u64_add(uint64_t *acc, uint64_t add) {
    if (!acc || *acc > UINT64_MAX - add) return false;
    *acc += add;
    return true;
}

bool dist_u64_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out) return false;
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

/* =========================================================================
 * Tunable Limits
 * ========================================================================= */

int dist_socket_buffer_bytes(void) {
    int mb = 128;
    const char *env = getenv("DS4_DIST_SOCKET_BUFFER_MB");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 0 && v <= 512) {
            mb = (int)v;
        }
    }
    return mb > 0 ? mb * 1024 * 1024 : 0;
}

uint32_t dist_worker_prefetch_depth(void) {
    uint32_t depth = 2;
    const char *env = getenv("DS4_DIST_WORKER_PREFETCH_DEPTH");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 1 && v <= 8) {
            depth = (uint32_t)v;
        }
    }
    return depth;
}

uint32_t dist_worker_forward_window(void) {
    uint32_t depth = 4;
    const char *env = getenv("DS4_DIST_WORKER_FORWARD_WINDOW");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 1 && v <= 64) {
            depth = (uint32_t)v;
        }
    }
    return depth;
}

bool dist_parse_positive_u32(
        const char *s,
        const char *name,
        uint32_t *out,
        char *err,
        size_t errlen) {
    if (!s || !out) {
        if (errlen) snprintf(err, errlen, "%s requires a positive integer", name);
        return false;
    }
    errno = 0;
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || s[0] == '\0' || *end != '\0' || v == 0 || v > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "invalid value for %s: %s", name, s);
        return false;
    }
    *out = (uint32_t)v;
    return true;
}

uint32_t dist_env_u32_clamped(const char *name, uint32_t defv, uint32_t minv, uint32_t maxv) {
    const char *s = getenv(name);
    if (!s || !s[0]) return defv;
    errno = 0;
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || s[0] == '\0' || *end != '\0') return defv;
    if (v < minv) v = minv;
    if (v > maxv) v = maxv;
    return (uint32_t)v;
}

bool dist_env_enabled(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && !(v[0] == '0' && v[1] == '\0');
}

