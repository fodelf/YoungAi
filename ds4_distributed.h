#ifndef DS4_DISTRIBUTED_H
#define DS4_DISTRIBUTED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4.h"

/* Distributed inference is an engine backend, not a separate frontend API.
 * Programs parse distributed options here, then keep using the normal
 * ds4_session_* calls. Only worker/coordinator serving modes call
 * ds4_dist_run() directly.
 */
typedef ds4_distributed_options ds4_dist_options;
typedef struct ds4_dist_session ds4_dist_session;

/* Tensor-parallel all-reduce transport (Stage 2 skeleton). A single persistent
 * socket between the two TP peers; ds4_dist_tp_allreduce_f32 sums each peer's
 * partial [count] float vector so both end with the full result. The listener
 * side calls ds4_dist_tp_listen, the connector ds4_dist_tp_connect. */
typedef struct ds4_dist_tp ds4_dist_tp;
ds4_dist_tp *ds4_dist_tp_listen(const char *host, int port, char *err, size_t errlen);
ds4_dist_tp *ds4_dist_tp_connect(const char *host, int port, char *err, size_t errlen);
int ds4_dist_tp_allreduce_f32(ds4_dist_tp *tp, float *buf, uint32_t count);
void ds4_dist_tp_free(ds4_dist_tp *tp);
ds4_dist_tp *ds4_engine_tp(ds4_engine *e); /* implemented in ds4.c */
int ds4_dist_tp_selftest(void); /* in-process loopback correctness check */


/* Options used by standalone `./ds4 --role coordinator -p ...` generation.
 * Interactive tools and the server go through the normal ds4_session API.
 */
typedef struct {
    const char *prompt;
    const char *system;
    const char *dump_logits_path;
    const char *dump_logprobs_path;
    int dump_logprobs_top_k;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    ds4_think_mode think_mode;
} ds4_dist_generation_options;

typedef enum {
    DS4_DIST_CLI_ERROR = -1,
    DS4_DIST_CLI_NOT_MATCHED = 0,
    DS4_DIST_CLI_MATCHED = 1,
} ds4_dist_cli_parse_result;

/* Shared option parsing. */
bool ds4_dist_enabled(const ds4_dist_options *opt);
ds4_dist_options *ds4_dist_options_create(void);
void ds4_dist_options_free(ds4_dist_options *opt);
void ds4_dist_usage(FILE *fp);
ds4_dist_cli_parse_result ds4_dist_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_dist_options *opt,
        char *err,
        size_t errlen);

/* Applies distributed layer-loading choices to the engine options before the
 * model is loaded.
 */
int ds4_dist_prepare_engine_options(
        const ds4_dist_options *opt,
        ds4_engine_options *engine,
        char *err,
        size_t errlen);

/* Coordinator session backend used by ds4.c. These mirror the normal session
 * operations; callers outside the engine should not need to call them directly.
 */
int ds4_dist_session_create(
        ds4_dist_session **out,
        ds4_engine *engine,
        const ds4_dist_options *opt,
        ds4_session *owner,
        int ctx_size,
        char *err,
        size_t errlen);
void ds4_dist_session_free(ds4_dist_session *d);

/* Returns 1 when the coordinator has full layer coverage, 0 when workers are
 * still missing, and -1 for configuration or internal errors.
 */
int ds4_dist_session_route_ready(ds4_dist_session *d, char *err, size_t errlen);

/* Synchronize the distributed KV state to the requested prompt timeline. */
int ds4_dist_session_sync(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        const ds4_tokens *prompt,
        float *logits,
        char *err,
        size_t errlen);

/* Evaluate one additional token across the current distributed route. */
int ds4_dist_session_eval(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        int token,
        float *logits,
        char *err,
        size_t errlen);

/* mtp.md Phase 1 (Scheme A): speculative decode across the layer-pipeline route.
 * Commits first_token plus any MTP draft tokens the target model verifies, into
 * accepted[0..ret-1]. Returns the committed count (>=1) or -1 on hard failure.
 * Falls back to a single-token eval when no drafter/worker is available. */
int ds4_dist_session_eval_speculative(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        int first_token,
        int eos_token,
        int *accepted,
        int accepted_cap,
        float *logits,
        char *err,
        size_t errlen);

/* Save/load use the normal DSV4 payload format. The coordinator gathers or
 * pushes remote layer shards internally so saved files are topology-neutral.
 */
int ds4_dist_session_save_payload(
        ds4_dist_session *d,
        ds4_session *owner,
        FILE *fp,
        char *err,
        size_t errlen);
int ds4_dist_session_load_payload(
        ds4_dist_session *d,
        ds4_session *owner,
        FILE *fp,
        uint64_t payload_bytes,
        char *err,
        size_t errlen);

/* Standalone distributed mode. Workers stay in this loop; coordinator one-shot
 * mode uses it for `./ds4 --role coordinator -p ...`.
 */
int ds4_dist_run(ds4_engine *engine, const ds4_dist_options *opt, const ds4_dist_generation_options *gen);

/* project.md P2.2 low-cost variant: remote expert pread service.  Both
 * machines load the byte-identical GGUF; the worker's faster (and during the
 * coordinator's pipeline half, idle) SSD serves raw file-range reads over
 * Thunderbolt so the coordinator's expert gather draws from both disks at
 * once.  Server side is enabled by DS4_DIST_EXPERT_FETCH_SERVE=1 (+ optional
 * DS4_DIST_EXPERT_FETCH_PORT, default 5606) and is called by the engine right
 * after the model is opened; it duplicates the model fd.  The client opens
 * n_conns sockets (one per fetcher thread; ds4_dist_expert_fetch is
 * slot-bound, lock-free) and validates that the remote file size matches.
 * Purely a byte transport: wrong/unavailable service degrades to local reads. */
int ds4_dist_expert_fetch_maybe_serve(int model_fd, uint64_t model_size);
/* Wave 29 (implemented in ds4_metal.m): eagerly dial the peer-SSD fetch
 * connections from the post-accept quiet window on the worker. No-op unless
 * DS4_DIST_EXPERT_FETCH_HOST is set. */
void ds4_gpu_expert_remote_fetch_kick(void);
int ds4_dist_expert_fetch_client_init(const char *host, int port, int n_conns, uint64_t model_size);
/* Wave 30 reverse-established transport: the worker's in-process outbound
 * dials to the coordinator fail (EHOSTUNREACH) while the coordinator's dials
 * to the worker always succeed, so only the TCP establishment direction is
 * reversed -- the wire protocol is unchanged.  accept_init: fetch CLIENT
 * listens on `port` and waits (<=120s) for the pread server to dial in,
 * then performs the normal client handshake on each accepted socket.
 * serve_dial: pread SERVER side; gated on DS4_DIST_EXPERT_FETCH_SERVE_DIAL_HOST
 * (+_PORT default 5607, +_CONNS), dials the peer in a background thread with
 * a 2s backoff and runs the standard serving loop per connection. */
int ds4_dist_expert_fetch_accept_init(int port, int n_conns, uint64_t model_size);
int ds4_dist_expert_fetch_serve_dial(int model_fd, uint64_t model_size);
int ds4_dist_expert_fetch(int slot, uint64_t off, void *dst, uint32_t len);
/* Pipelined variants: per slot, recv order must match send order; up to a few
 * requests may be in flight (bounded by the socket buffers). */
int ds4_dist_expert_fetch_send(int slot, uint64_t off, uint32_t len);
int ds4_dist_expert_fetch_recv(int slot, void *dst, uint32_t len);

#endif
