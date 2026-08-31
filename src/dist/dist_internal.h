/* =========================================================================
 * ds4_distributed.c - Distributed inference runtime.
 * =========================================================================
 *
 * This module owns the DS4 distributed transport and orchestration layer. The
 * rest of the engine still sees a normal ds4_session: when distributed mode is
 * active, ds4.c delegates sync/eval/save/load to the coordinator session API in
 * this file.
 *
 * Workers execute contiguous model slices with the same graph-slice entry
 * points used by the local engine. KV snapshots remain topology-independent:
 * save gathers worker-owned layer tensors into the normal DSV4 payload, and
 * load splits a normal DSV4 payload across the currently registered route.
 */


/* dist_internal.h — 上面是原 ds4_distributed.c(9836 行) 的文件头注释, 原样保留。
 * 该文件已机械拆分为 src/dist 各 .c(行为零变化); 本头是拆分文件间的内部接缝:
 * 原先文件内 static 且被多个拆分文件引用的函数在此声明(命名不变, 定义处去 static),
 * 协议记录/状态类型见 dist_proto.h / dist_state.h。只供 src/dist 各 .c 包含,
 * 对外接口不变, 仍是仓根 ds4_distributed.h。 */
#ifndef DS4_DIST_INTERNAL_H
#define DS4_DIST_INTERNAL_H

#include "ds4_distributed.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "dist_proto.h"
#include "dist_state.h"

/* 跨文件宏(原样搬移: DIST_DEBUG@framing, token-hash 常量@wire, DIST_COORD_DEBUG@reg) */
#ifdef DS4_DIST_TRACE
#define DIST_DEBUG(...) do { \
    fprintf(stderr, "ds4: distributed debug: " __VA_ARGS__); \
    fputc('\n', stderr); \
} while (0)
#else
#define DIST_DEBUG(...) ((void)0)
#endif
#define DS4_DIST_TOKEN_HASH_INIT 1469598103934665603ull
#define DS4_DIST_TOKEN_HASH_PRIME 1099511628211ull

#define DIST_COORD_DEBUG(state, ...) do { \
    if (dist_coordinator_debug_enabled(state)) fprintf(stderr, __VA_ARGS__); \
} while (0)

/* — 定义于 dist_util.c — */
void dist_mtp_print_summary(ds4_dist_session *d, const char *tag);
uint32_t dist_prefill_send_depth(uint32_t chunk_count);
uint32_t dist_resolved_layer_end(const ds4_dist_options *opt, uint32_t n_layers);
const char *dist_role_name(ds4_distributed_role role);
void dist_sleep_reconnect(void);
double dist_now_sec(void);
int dist_payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen);
int dist_payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen);
int dist_payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen);
int dist_payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen);
int dist_payload_copy_bytes( FILE *src, FILE *dst, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen);
int dist_copy_file_range( FILE *src, uint64_t offset, uint64_t bytes, FILE *dst, char *err, size_t errlen);
int dist_rewind_file(FILE *fp, const char *what, char *err, size_t errlen);
int dist_measure_file(FILE *fp, uint64_t *bytes, const char *what, char *err, size_t errlen);
FILE *dist_tmpfile_or_err(const char *what, char *err, size_t errlen);
bool dist_u64_add(uint64_t *acc, uint64_t add);
bool dist_u64_mul(uint64_t a, uint64_t b, uint64_t *out);
int dist_socket_buffer_bytes(void);
uint32_t dist_worker_prefetch_depth(void);
uint32_t dist_worker_forward_window(void);
bool dist_parse_positive_u32( const char *s, const char *name, uint32_t *out, char *err, size_t errlen);
uint32_t dist_env_u32_clamped(const char *name, uint32_t defv, uint32_t minv, uint32_t maxv);
bool dist_env_enabled(const char *name);

/* — 定义于 dist_transport.c — */
uint32_t dist_activation_bits_or_default(uint32_t bits);
bool dist_activation_bits_valid(uint32_t bits);
bool dist_activation_wire_bytes(uint32_t bits, uint64_t values, uint32_t *out);
bool dist_activation_values_from_wire_bytes(uint32_t bits, uint32_t bytes, uint64_t *out);
bool dist_activation_wire_bytes_from_f32_bytes(uint32_t bits, uint32_t f32_bytes, uint32_t *out);
int dist_write_activation_payload( int fd, const float *src, uint64_t values, uint32_t bits);
int dist_decode_activation_payload( const void *wire, uint32_t bits, uint32_t wire_bytes, float **out, uint32_t *out_f32_bytes, bool *out_uses_wire, char *err, size_t errlen);

/* — 定义于 dist_framing.c — */
int dist_set_socket_low_latency(int fd);
int dist_write_full(int fd, const void *buf, size_t len);
int dist_read_full(int fd, void *buf, size_t len);
int dist_write_frame_header(int fd, uint32_t type, uint32_t bytes);
int dist_read_frame_header(int fd, uint32_t *type, uint32_t *bytes, char *err, size_t errlen);
int dist_discard_bytes(int fd, uint32_t bytes);
int dist_send_error(int fd, const char *msg);
void dist_peer_name(int fd, char *host, size_t hostlen, char *port, size_t portlen);
int dist_open_listener(const char *host, int port, char *err, size_t errlen);
int dist_listener_port(int fd);
int dist_connect_endpoint_once(const char *host, int port, int *last_errno, char *err, size_t errlen);
int dist_connect_endpoint(const char *host, int port, char *err, size_t errlen);

/* — 定义于 dist_wire.c — */
void dist_hello_to_wire(ds4_dist_hello_fixed *h);
void dist_hello_from_wire(ds4_dist_hello_fixed *h);
uint64_t dist_u64_from_halves(uint32_t hi, uint32_t lo);
uint64_t dist_token_hash_update_span(uint64_t h, const int *tokens, uint32_t n_tokens);
uint64_t dist_token_hash_prefix(const int *tokens, uint32_t n_tokens);
int dist_session_token_hash_prefix( ds4_session *session, uint32_t n_tokens, uint64_t *hash, char *err, size_t errlen);
bool dist_bytes_have_nul(const void *p, uint32_t len);
void dist_u64_to_halves(uint64_t v, uint32_t *hi, uint32_t *lo);
void dist_work_from_wire(ds4_dist_work_fixed *w);
void dist_work_to_wire(ds4_dist_work_fixed *w);
void dist_route_from_wire(ds4_dist_route_fixed *r);
void dist_route_to_wire(ds4_dist_route_fixed *r);
void dist_route_return_from_wire(ds4_dist_route_return_fixed *r);
void dist_route_return_to_wire(ds4_dist_route_return_fixed *r);
void dist_result_to_wire(ds4_dist_result_fixed *r);
void dist_result_from_wire(ds4_dist_result_fixed *r);
void dist_snapshot_req_to_wire(ds4_dist_snapshot_req_fixed *s);
void dist_snapshot_req_from_wire(ds4_dist_snapshot_req_fixed *s);
void dist_snapshot_begin_to_wire(ds4_dist_snapshot_begin_fixed *s);
void dist_snapshot_begin_from_wire(ds4_dist_snapshot_begin_fixed *s);
void dist_snapshot_chunk_to_wire(ds4_dist_snapshot_chunk_fixed *s);
void dist_snapshot_chunk_from_wire(ds4_dist_snapshot_chunk_fixed *s);
void dist_snapshot_done_to_wire(ds4_dist_snapshot_done_fixed *s);
void dist_snapshot_done_from_wire(ds4_dist_snapshot_done_fixed *s);
void dist_telemetry_to_wire(ds4_dist_telemetry_fixed *t);
void dist_telemetry_from_wire(ds4_dist_telemetry_fixed *t);
uint32_t dist_usec_since(double t0, double t1);

/* — 定义于 dist_reg.c — */
int dist_send_hello(ds4_engine *engine, const ds4_dist_options *opt, int ctx_size, uint32_t listen_port, int fd);
int dist_recv_hello(int fd, ds4_dist_hello_fixed *hello, char *model_name, size_t model_name_cap, char *err, size_t errlen);
bool dist_coordinator_debug_enabled(const ds4_dist_coordinator_state *state);

/* — 定义于 dist_coord_route.c — */
void dist_coordinator_add_worker( ds4_dist_coordinator_state *state, int fd, const char *peer_host, const char *peer_port, const ds4_dist_hello_fixed *hello, const char *model_name);
void dist_coordinator_report_plan(ds4_dist_coordinator_state *state);
void dist_route_plan_free(ds4_dist_route_plan *plan);
void dist_coordinator_forget_route_workers( ds4_dist_coordinator_state *state, const ds4_dist_route_plan *plan);
bool dist_coordinator_build_route_plan( ds4_dist_coordinator_state *state, ds4_dist_route_plan *plan, uint64_t *generation, char *err, size_t errlen);
int dist_logits_argmax(const float *logits, int n_vocab);
bool dist_coordinator_ensure_route( ds4_dist_coordinator_state *state, ds4_dist_route_plan *plan, uint64_t *generation, char *err, size_t errlen);
uint64_t dist_coordinator_generation(ds4_dist_coordinator_state *state);

/* — 定义于 dist_coord_dispatch.c — */
int dist_recv_result_alloc( int fd, const ds4_dist_coordinator_state *state, uint64_t request_id, uint32_t *kind, uint64_t *result_hash, void **payload, uint32_t *payload_bytes, uint32_t *draft_out, /* optional caller buffer of >=16 ids */ uint32_t *draft_n_out, /* optional out: number of draft ids read */ char *err, size_t errlen);
int dist_coordinator_send_remote_work_on_fd( ds4_dist_coordinator_state *state, const ds4_dist_route_plan *plan, int fd, const int *tokens, uint32_t n_tokens, uint32_t pos0, uint64_t session_id, uint64_t request_id, uint64_t prefix_hash, uint64_t result_hash, bool reset_session, bool ack_only, const float *hidden_hc, uint32_t hidden_hc_bytes, uint32_t draft_cap, /* docs/archive/mtp.md Phase 1: ask last-layer worker to draft */ uint32_t accept_len, /* docs/archive/mtp.md Phase 1: roll back prev spec batch first */ uint32_t extra_flags, /* extra DS4_DIST_WORK_F_* bits (e.g. DRAFT/VERIFY) */ char *err, size_t errlen);

/* — 定义于 dist_coord_eval.c — */
int dist_coordinator_eval_span( ds4_dist_coordinator_state *state, ds4_session *session, const ds4_dist_route_plan *plan, const int *tokens, uint32_t n_tokens, uint32_t pos0, uint64_t session_id, uint64_t request_id, bool reset_session, float *logits, ds4_dist_spec_io *spec, char *err, size_t errlen);

/* — 定义于 dist_coord_gen.c — */
bool dist_prompt_is_rendered_chat(const char *prompt);
int dist_write_logits_dump( ds4_dist_coordinator_state *state, const ds4_dist_generation_options *gen, const ds4_tokens *prompt, const ds4_dist_route_plan *plan, const float *logits);
int dist_coordinator_rebuild_from_transcript( ds4_dist_coordinator_state *state, ds4_session *session, ds4_dist_route_plan *plan, const ds4_tokens *transcript, uint64_t session_id, uint64_t *request_id, float *logits, uint64_t *plan_generation, bool forget_route, char *err, size_t errlen);
int dist_write_logprobs_dump( ds4_dist_coordinator_state *state, const ds4_dist_generation_options *gen, const ds4_tokens *prompt, ds4_dist_route_plan *plan, ds4_session *session, uint64_t session_id, uint64_t *request_id, float *logits);

/* — 定义于 dist_prefill_pipe.c — */
int dist_prefill_sender_init( ds4_dist_prefill_sender *sender, ds4_dist_coordinator_state *state, const ds4_dist_route_plan *plan, const ds4_tokens *prompt, uint64_t session_id, int fd, uint32_t chunk_count, uint32_t max_hidden_bytes, char *err, size_t errlen);
void dist_prefill_sender_destroy(ds4_dist_prefill_sender *sender);
ds4_dist_prefill_send_slot *dist_prefill_sender_acquire_slot( ds4_dist_prefill_sender *sender, char *err, size_t errlen);
int dist_prefill_sender_enqueue_slot( ds4_dist_prefill_sender *sender, char *err, size_t errlen);
void dist_prefill_sender_finish(ds4_dist_prefill_sender *sender);
void dist_prefill_sender_cancel(ds4_dist_prefill_sender *sender);
void *dist_prefill_sender_main(void *arg);
void dist_prefill_reader_emit_progress( ds4_dist_prefill_result_reader *reader, uint32_t *reported);
bool dist_prefill_reader_wait_emit_progress( ds4_dist_prefill_result_reader *reader, uint32_t *reported);
bool dist_prefill_reader_wait_flow_window( ds4_dist_prefill_result_reader *reader, uint32_t submitted, uint32_t window, uint32_t *reported);
void *dist_prefill_result_reader_main(void *arg);

/* — 定义于 dist_prefill.c — */
bool dist_coordinator_can_pipeline_prefill( const ds4_dist_coordinator_state *state, const ds4_dist_route_plan *plan, ds4_session *session, uint32_t n_tokens, uint32_t chunk_cap);
int dist_coordinator_prefill_chunk_cap( const ds4_dist_coordinator_state *state, ds4_session *session, uint32_t *chunk_cap, char *err, size_t errlen);
void dist_report_prefill_progress(ds4_session *session, uint32_t current, uint32_t total);
int dist_coordinator_prefill_prompt_pipelined( ds4_dist_coordinator_state *state, ds4_session *session, const ds4_dist_route_plan *plan, const ds4_tokens *prompt, uint32_t span_start, uint32_t n_tokens, bool reset_first_chunk, uint32_t chunk_cap, uint64_t session_id, uint64_t *request_id, float *logits, char *err, size_t errlen);
int dist_coordinator_prefill_prompt( ds4_dist_coordinator_state *state, ds4_session *session, const ds4_dist_route_plan *plan, const ds4_tokens *prompt, uint64_t session_id, uint64_t *request_id, float *logits, char *err, size_t errlen);

/* — 定义于 dist_coord_recover.c — */
int dist_run_coordinator_generation( ds4_dist_coordinator_state *state, const ds4_dist_generation_options *gen);

/* — 定义于 dist_coord_ctl.c — */
void *dist_coordinator_client_main(void *arg);
void *dist_coordinator_accept_main(void *arg);
uint64_t dist_make_session_id(const void *ptr);
int dist_session_ensure_route(ds4_dist_session *d, char *err, size_t errlen);

/* — 定义于 dist_kv_snapshot.c — */
int dist_write_snapshot_load_begin( int fd, const ds4_dist_snapshot_begin_fixed *begin, const int *tokens);
int dist_read_snapshot_begin_frame( int fd, ds4_dist_snapshot_begin_fixed *begin, char *msg, size_t msg_cap, char *err, size_t errlen);
int dist_read_snapshot_done_frame( int fd, uint64_t request_id, char *err, size_t errlen);
int dist_receive_snapshot_chunks_to_file( int fd, uint64_t request_id, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);

/* — 定义于 dist_payload.c — */
bool dist_kv_raw_live_valid(const ds4_dist_kv_layout *layout);
int dist_kv_parse_layer_payload( ds4_engine *engine, FILE *fp, uint64_t bytes, uint32_t expected_start, uint32_t expected_end, ds4_dist_kv_layout *layout, bool *layout_set, uint32_t *n_comp, uint32_t *n_index_comp, ds4_dist_kv_shard_file *shard, char *err, size_t errlen);
int dist_kv_write_session_header( FILE *fp, const ds4_dist_kv_layout *layout, char *err, size_t errlen);
uint32_t dist_kv_route_shard_count(const ds4_dist_session *d);
void dist_kv_route_shard( const ds4_dist_session *d, uint32_t shard, uint32_t *layer_start, uint32_t *layer_end, const ds4_dist_route_entry **entry);
int dist_kv_route_validate( const ds4_dist_session *d, char *err, size_t errlen);
void dist_kv_shards_close(ds4_dist_kv_shard_file *shards, uint32_t count);
int dist_save_remote_shard_to_file( ds4_dist_session *d, const ds4_dist_route_entry *entry, const ds4_tokens *tokens, uint64_t token_hash, FILE *fp, uint64_t *payload_bytes_out, char *err, size_t errlen);
int dist_prepare_shard_from_session_payload( ds4_dist_session *d, FILE *src, uint64_t *remaining, const ds4_dist_kv_layout *layout, const uint32_t *n_comp, const uint32_t *n_index_comp, uint32_t layer_start, uint32_t layer_end, FILE **tmp_out, uint64_t *payload_bytes_out, char *err, size_t errlen);
int dist_load_remote_shard_from_payload( ds4_dist_session *d, const ds4_dist_route_entry *entry, const int *tokens, uint32_t token_count, uint64_t token_hash, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);

/* — 定义于 dist_coord_main.c — */
extern int g_dist_reverse_connect;   /* --reverse-connect (dist_cli.c 置位) */

/* expert-fetch 服务端配置 (--expert-fetch-serve / --expert-fetch-port /
 * --expert-fetch-dial, dist_cli.c 置位; 定义在 dist_rfetch.c)。 */
extern int g_efetch_serve;
extern int g_efetch_port;
extern const char *g_efetch_dial_host;
extern int g_efetch_dial_port;
bool dist_reverse_connect_enabled(void);

/* 所有 dist socket 的收发超时(秒)。60 秒 = 远大于任何一次跨机 forward,
 * 又能让真死链在一分钟内暴露而不是永久挂住 graph worker。 */
#define DIST_SOCKET_TIMEOUT_SEC 60
void *dist_coordinator_reverse_connect_main(void *arg);
int dist_run_coordinator(ds4_engine *engine, const ds4_dist_options *opt, const ds4_dist_generation_options *gen);

/* — 定义于 dist_worker_loop.c — */
int dist_send_work_result( int fd, uint64_t request_id, uint64_t result_hash, uint32_t status, uint32_t result_kind, uint32_t payload_bits, const ds4_dist_telemetry_fixed *telemetry, uint32_t telemetry_count, const void *payload, uint32_t payload_bytes, const uint32_t *draft_tokens, uint32_t draft_count);
int dist_send_work_error(int fd, uint64_t request_id, const char *msg);
int dist_send_snapshot_begin( int fd, const ds4_dist_snapshot_begin_fixed *begin, const int *tokens, const char *msg);
int dist_send_snapshot_error( int fd, uint64_t request_id, uint64_t session_id, uint32_t model_id, uint32_t layer_start, uint32_t layer_end, const char *msg);
int dist_send_snapshot_done(int fd, uint64_t request_id, uint32_t status, const char *msg);
int dist_send_snapshot_file_chunks(int fd, uint64_t request_id, FILE *fp, uint64_t bytes);

/* — 定义于 dist_worker_route.c — */
bool dist_route_get_entry( const void *route_blob, uint32_t route_bytes, uint32_t route_count, uint32_t target_index, ds4_dist_route_entry *out, char *err, size_t errlen);
bool dist_route_get_return_target( const void *route_blob, uint32_t route_bytes, uint32_t route_count, ds4_dist_route_return *out, char *err, size_t errlen);
bool dist_route_validate_blob( const void *route_blob, uint32_t route_bytes, uint32_t route_count, uint32_t n_layers, char *err, size_t errlen);
int dist_send_work_frame( int fd, const ds4_dist_work_fixed *work, const int *tokens, const float *input_hc, const void *route_blob);
int dist_worker_upstream_send_work_result( ds4_dist_worker_upstream *upstream, uint64_t request_id, uint64_t result_hash, uint32_t status, uint32_t result_kind, uint32_t payload_bits, const ds4_dist_telemetry_fixed *telemetry, uint32_t telemetry_count, const void *payload, uint32_t payload_bytes, const uint32_t *draft_tokens, uint32_t draft_count);
int dist_worker_upstream_send_work_error( ds4_dist_worker_upstream *upstream, uint64_t request_id, const char *msg);

/* — 定义于 dist_worker_fwd.c — */
void dist_worker_upstream_init( ds4_dist_worker_upstream *upstream, ds4_dist_worker_state *state, int fd);
void dist_worker_upstream_destroy(ds4_dist_worker_upstream *upstream);
int dist_forward_work_to_next( ds4_dist_worker_upstream *upstream, const ds4_dist_route_entry *next, const ds4_dist_work_fixed *work, const int *tokens, const float *hidden_hc, uint32_t hidden_hc_bytes, const ds4_dist_telemetry_fixed *telemetry, const void *route_blob);

/* — 定义于 dist_worker_kv.c — */
ds4_dist_worker_session *dist_worker_get_session_locked( ds4_dist_worker_state *state, uint64_t session_id, char *err, size_t errlen);
uint32_t dist_worker_clear_sessions(ds4_dist_worker_state *state);
int dist_mem_read(ds4_dist_mem_reader *r, void *dst, uint32_t len);
int dist_worker_handle_snapshot_save( ds4_dist_worker_state *state, ds4_dist_worker_upstream *upstream, uint32_t bytes);
int dist_worker_handle_snapshot_load( ds4_dist_worker_state *state, ds4_dist_worker_upstream *upstream, uint32_t bytes);

/* — 定义于 dist_worker_exec.c — */
int dist_worker_process_work_payload( ds4_dist_worker_state *state, ds4_dist_worker_upstream *upstream, const void *payload, uint32_t bytes);

/* — 定义于 dist_worker_prefetch.c — */
int dist_worker_read_loop_prefetch(ds4_dist_worker_state *state, int fd);
void *dist_worker_data_listener_main(void *arg);

/* — 定义于 dist_worker_main.c — */
int dist_run_worker(ds4_engine *engine, const ds4_dist_options *opt, int ctx_size);

/* — 定义于 dist_cli.c — */
int dist_validate_options(const ds4_dist_options *opt, char *err, size_t errlen);
int dist_validate_layers_for_model(const ds4_dist_options *opt, uint32_t n_layers, char *err, size_t errlen);

#endif /* DS4_DIST_INTERNAL_H */
