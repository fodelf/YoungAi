/* dist_proto.h — 机械拆自 ds4_distributed.c: 协议常量与线上记录(Protocol Constants And Wire Records)。只供 dist_internal.h 包含。 */
#ifndef DS4_DIST_PROTO_H
#define DS4_DIST_PROTO_H

/* =========================================================================
 * Protocol Constants And Wire Records
 * ========================================================================= */

#define DS4_DIST_MAGIC 0x44533444u /* DS4D */
#define DS4_DIST_MSG_HELLO 1u
#define DS4_DIST_MSG_ERROR 2u
#define DS4_DIST_MSG_WORK 3u
#define DS4_DIST_MSG_RESULT 4u
#define DS4_DIST_MSG_SNAPSHOT_SAVE_REQ 5u
#define DS4_DIST_MSG_SNAPSHOT_BEGIN 6u
#define DS4_DIST_MSG_SNAPSHOT_CHUNK 7u
#define DS4_DIST_MSG_SNAPSHOT_DONE 8u
#define DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN 9u
#define DS4_DIST_MSG_ALLREDUCE 10u /* tensor-parallel partial-sum exchange */
#define DS4_DIST_MAX_MODEL_NAME 127u
#define DS4_DIST_WORK_F_INPUT_HC 0x00000001u
#define DS4_DIST_WORK_F_OUTPUT_LOGITS 0x00000002u
#define DS4_DIST_WORK_F_RESET_SESSION 0x00000004u
#define DS4_DIST_WORK_F_ACK_ONLY 0x00000008u
/* docs/archive/mtp.md Phase 1: after producing the final hidden state + logits, the worker
 * holding the last layers should run the MTP drafter and append up to
 * work.draft_cap candidate token ids to its RESULT payload. */
#define DS4_DIST_WORK_F_DRAFT 0x00000010u
/* docs/archive/mtp.md Phase 1: the K-token candidate batch verification pass. The last-layer
 * worker runs the output head on every row and returns per-row argmax token ids
 * in the RESULT draft channel instead of a single logits row. */
#define DS4_DIST_WORK_F_VERIFY 0x00000020u
/* Wave 69: continuation chunk of a row-chunked (pipelined) VERIFY batch. The
 * coordinator splits one verify batch into N chunks so its layer-0:k compute of
 * chunk c+1 overlaps the worker's layer-k:out compute of chunk c (the ~50%
 * serial coordinator-idle the layer-pipeline otherwise pays). A CONT chunk tells
 * the worker: do NOT roll back (you are mid-batch, prior chunks' KV must stay)
 * and do NOT reset spec_base_len (keep the batch-start position recorded by the
 * first, non-CONT chunk, so the next batch's accept_len rolls back to the right
 * place). KV is appended incrementally chunk-by-chunk on both hosts, so the
 * result is byte-identical to the single-frame batch. */
#define DS4_DIST_WORK_F_VERIFY_CONT 0x00000040u
#define DS4_DIST_WORK_F_VALID_MASK \
    (DS4_DIST_WORK_F_INPUT_HC | DS4_DIST_WORK_F_OUTPUT_LOGITS | \
     DS4_DIST_WORK_F_RESET_SESSION | DS4_DIST_WORK_F_ACK_ONLY | \
     DS4_DIST_WORK_F_DRAFT | DS4_DIST_WORK_F_VERIFY | DS4_DIST_WORK_F_VERIFY_CONT)
#define DS4_DIST_RESULT_ACK 0u
#define DS4_DIST_RESULT_HIDDEN_STATE 1u
#define DS4_DIST_RESULT_LOGITS 2u
#define DS4_DIST_ACTIVATION_BITS_DEFAULT 32u
#define DS4_DIST_ROUTE_F_OUTPUT_LOGITS 0x00000001u
#define DS4_DIST_ROUTE_RETURN_UPSTREAM 1u
#define DS4_DIST_RECV_TRANSPORT_ERROR 1
#define DS4_DIST_RECV_REMOTE_ERROR 2
#define DS4_DIST_SNAPSHOT_CHUNK_BYTES (8u * 1024u * 1024u)

typedef struct {
    uint32_t magic;
    uint32_t type;
    uint32_t bytes;
} ds4_dist_frame_header;

typedef struct {
    uint32_t model_id;
    uint32_t quant_bits;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t has_output;
    uint32_t has_hidden;
    uint32_t ctx_size;
    uint32_t n_layers;
    uint32_t listen_port;
    uint32_t model_name_len;
} ds4_dist_hello_fixed;

typedef struct {
    uint32_t model_id;
    uint32_t session_hi;
    uint32_t session_lo;
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t prefix_hash_hi;
    uint32_t prefix_hash_lo;
    uint32_t result_hash_hi;
    uint32_t result_hash_lo;
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t flags;
    uint32_t token_bytes;
    uint32_t input_hc_bytes;
    uint32_t input_hc_bits;
    uint32_t route_count;
    uint32_t route_index;
    uint32_t route_bytes;
    /* docs/archive/mtp.md Phase 1 speculative fields (0 on every non-MTP frame, so the wire
     * layout is behavior-identical to the pre-MTP protocol once both ends are
     * rebuilt). draft_cap: how many MTP candidates the last-layer worker may
     * draft for this step. accept_len: number of tokens the coordinator
     * accepted from the previous speculative batch; the worker truncates its
     * layer-slice KV to this length before applying the new span. */
    uint32_t draft_cap;
    uint32_t accept_len;
} ds4_dist_work_fixed;

typedef struct {
    uint32_t host_len;
    uint32_t port;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t flags;
} ds4_dist_route_fixed;

typedef struct {
    uint32_t kind;
    uint32_t host_len;
    uint32_t port;
} ds4_dist_route_return_fixed;

typedef struct {
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t result_hash_hi;
    uint32_t result_hash_lo;
    uint32_t status;
    uint32_t result_kind;
    uint32_t telemetry_count;
    uint32_t telemetry_bytes;
    uint32_t payload_bytes;
    uint32_t payload_bits;
    /* docs/archive/mtp.md Phase 1: count of MTP draft token ids (uint32 each) appended after
     * the logits/telemetry payload. 0 unless the WORK frame set DS4_DIST_WORK_F_DRAFT
     * and the worker successfully drafted. */
    uint32_t draft_count;
} ds4_dist_result_fixed;

typedef struct {
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t route_index;
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t eval_usec;
    uint32_t downstream_wait_usec;
    uint32_t forward_send_usec;
    uint32_t input_bytes;
    uint32_t output_bytes;
} ds4_dist_telemetry_fixed;

typedef struct {
    uint32_t model_id;
    uint32_t session_hi;
    uint32_t session_lo;
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t token_hash_hi;
    uint32_t token_hash_lo;
    uint32_t token_count;
    uint32_t layer_start;
    uint32_t layer_end;
} ds4_dist_snapshot_req_fixed;

typedef struct {
    uint32_t model_id;
    uint32_t session_hi;
    uint32_t session_lo;
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t token_hash_hi;
    uint32_t token_hash_lo;
    uint32_t token_count;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t payload_hi;
    uint32_t payload_lo;
    uint32_t status;
    uint32_t token_bytes;
    uint32_t message_bytes;
} ds4_dist_snapshot_begin_fixed;

typedef struct {
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t chunk_bytes;
} ds4_dist_snapshot_chunk_fixed;

typedef struct {
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t status;
    uint32_t message_bytes;
} ds4_dist_snapshot_done_fixed;


#endif /* DS4_DIST_PROTO_H */
