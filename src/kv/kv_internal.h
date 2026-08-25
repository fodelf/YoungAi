/* kv_internal.h — 磁盘 KV checkpoint 模块内部头(重构阶段3, 自 ds4_kvstore.c
 * 机械三分: kv_util=基元/sha1, kv_cache=条目缓存与淘汰策略, kv_file=文件读写
 * 与 trailer)。只声明跨文件符号, 命名与拆分前一致。 */
#ifndef DS4_KV_INTERNAL_H
#define DS4_KV_INTERNAL_H

#include <stdbool.h>
#include <stdint.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "ds4_kvstore.h"

#define KV_CACHE_MAGIC0 'K'
#define KV_CACHE_MAGIC1 'V'
#define KV_CACHE_MAGIC2 'C'
#define KV_CACHE_VERSION 1u
/* Header byte 20 carries the graph-payload ABI.  It is separate from the outer
 * file version because the KVC envelope can remain stable while the serialized
 * ds4_session internals become unsafe to restore across runtime changes. */
#define KV_CACHE_PAYLOAD_ABI 2u
#define KV_CACHE_DEFAULT_MIN_TOKENS 512
#define KV_CACHE_DEFAULT_COLD_MAX_TOKENS 30000
/* Tokenizers may merge text across the prompt boundary. Trimming a small tail
 * still improves the cheap token-prefix path, while text-prefix lookup handles
 * cases where canonical prompt tokenization spells the same bytes differently.
 * The 2048 alignment also matches the backend prefill chunk schedule, which
 * keeps compressor row finalization identical to a cold full prompt. */
#define KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS 32
#define KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS 2048
#define KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS 10000
/* Disk-hit counts are evidence that a checkpoint was useful, but only while
 * the workload still resembles the one that produced those hits. */
#define KV_CACHE_MIN_EFFECTIVE_HITS 0.01
/* A continued checkpoint that is a strict prefix of the incoming store is a
 * routine waypoint on the same path. Keep recent hits meaningful, but make
 * never-hit or stale waypoints cheap victims while pre-evicting for the new
 * store. */
#define KV_CACHE_CONTINUED_PREFIX_MIN_FACTOR 0.05
#define KV_CACHE_CONTINUED_PREFIX_HIT_FACTOR 0.45
/* Cold/evict/shutdown checkpoints are intentional anchors, not just automatic
 * waypoints in a single growing conversation. Give them a soft prior so they
 * survive comparable continued entries, while still allowing pressure and poor
 * density to evict them. */
#define KV_CACHE_ANCHOR_REASON_SCORE_FACTOR 2.0


typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} kv_buf;

/* kv_util.c */
void kv_die(const char *msg);
void *kv_xmalloc(size_t n);
void *kv_xrealloc(void *p, size_t n);
char *kv_xstrdup(const char *s);
void kv_buf_reserve(kv_buf *b, size_t add);
void kv_buf_append(kv_buf *b, const void *p, size_t n);
void kv_buf_putc(kv_buf *b, char c);
void kv_buf_puts(kv_buf *b, const char *s);
void kv_buf_printf(kv_buf *b, const char *fmt, ...);
char *kv_buf_take(kv_buf *b);
double kv_now_sec(void);
const char *kv_log_name(const ds4_kvstore *kc);
void kv_logf(ds4_kvstore *kc, ds4_kvstore_log_type type, const char *fmt, ...);
void kv_le_put64(uint8_t *p, uint64_t v);
uint64_t kv_le_get64(const uint8_t *p);
bool kv_mkdir_p(const char *path);
void kv_cache_push(ds4_kvstore *kc, ds4_kvstore_entry e);

/* kv_cache.c */
void kv_cache_refresh(ds4_kvstore *kc);
bool kv_cache_file_text_matches(const char *path, const char sha[41], const char *text, size_t text_len);
bool kv_cache_existing_compatible(ds4_kvstore *kc, const char *path,
                                  const char sha[41],
                                  const char *text, size_t text_len,
                                  int model_id, int quant_bits, int ctx_size);


#endif /* DS4_KV_INTERNAL_H */
