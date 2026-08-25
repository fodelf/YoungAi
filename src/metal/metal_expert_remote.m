/* metal_expert_remote.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

/* One gather work unit = one tensor (gate/up/down) of one active expert.
 * Per-tensor units triple the available IO queue depth versus per-expert
 * units: a 6-expert decode layer exposes 18 concurrent preads instead of 6,
 * which matters because random NVMe bandwidth scales with QD. */
/* Resolve one gather unit to (source file offset, destination, length) plus
 * an optional in-RAM hard-copy source.  Shared by the local copy path and the
 * remote-fetch workers.  Returns 0 on a bad expert id. */
static int ds4_gpu_expert_gather_unit_resolve(
        ds4_metal_expert_gather_ctx *ctx,
        uint32_t unit,
        uint64_t *src_off,
        uint8_t **dst,
        uint64_t *len,
        const uint8_t **hard_src) {
    const uint32_t slot = unit / 3u;
    const uint32_t part = unit % 3u;   /* 0=gate 1=up 2=down */
    const uint32_t id = ctx->active_ids[slot];
    if (id >= ctx->n_expert_total) return 0;
    const ds4_metal_expert_source_entry *src_entry =
        ds4_gpu_expert_source_hard_find_ex(ctx->model_map, ctx->layer_index, id, false);
    *hard_src = NULL;
    switch (part) {
    case 0:
        *len = ctx->gate_expert_bytes;
        *src_off = ctx->gate_offset + (uint64_t)id * ctx->gate_expert_bytes;
        *dst = ctx->gate_dst + (uint64_t)slot * ctx->gate_expert_bytes;
        if (src_entry) *hard_src = (const uint8_t *)src_entry->gate_copy;
        break;
    case 1:
        *len = ctx->gate_expert_bytes;
        *src_off = ctx->up_offset + (uint64_t)id * ctx->gate_expert_bytes;
        *dst = ctx->up_dst + (uint64_t)slot * ctx->gate_expert_bytes;
        if (src_entry) *hard_src = (const uint8_t *)src_entry->up_copy;
        break;
    default:
        *len = ctx->down_expert_bytes;
        *src_off = ctx->down_offset + (uint64_t)id * ctx->down_expert_bytes;
        *dst = ctx->down_dst + (uint64_t)slot * ctx->down_expert_bytes;
        if (src_entry) *hard_src = (const uint8_t *)src_entry->down_copy;
        break;
    }
    return 1;
}

void ds4_gpu_expert_gather_copy_unit(ds4_metal_expert_gather_ctx *ctx, uint32_t unit) {
    uint64_t len = 0;
    uint64_t src_off = 0;
    uint8_t *dst = NULL;
    const uint8_t *hard_src = NULL;
    if (!ds4_gpu_expert_gather_unit_resolve(ctx, unit, &src_off, &dst, &len, &hard_src)) {
        ctx->ok = 0;
        return;
    }
    /* Staged one layer ahead from the peer's SSD?  RAM copy beats any disk. */
    if (!hard_src) {
        const uint32_t expert_id = ctx->active_ids[unit / 3u];
        /* Frequency-pinned expert (DS4_EXPERT_PIN_FILE): serve from the mlocked
         * mmap instead of pread — residency alone never pays on the pread path,
         * which reads the SSD regardless (measured 2026-07-17: pins armed,
         * hit_mib=0.0, decode still ~1.2GB/token cold on both hosts). Same file
         * offsets as the pread, so bytes are bit-exact; a budget-skipped row
         * simply faults through the mmap like the legacy gather. */
        if (ds4_gpu_expert_pin_hot(ctx->layer_index, expert_id)) {
            const double pt0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
            memcpy(dst, ctx->map + src_off, (size_t)len);
            if (ctx->io_profile) {
                __sync_fetch_and_add(&g_io_prof_copy_ns,
                                     (uint64_t)((ds4_gpu_now_ms() - pt0) * 1e6));
                __sync_fetch_and_add(&g_io_prof_hit_bytes, len);
            }
            return;
        }
        uint64_t slen = 0;
        const uint8_t *staged = ds4_gpu_expert_stage_find(ctx->layer_index,
                                                          expert_id,
                                                          unit % 3u,
                                                          &slen);
        if (!staged) {
            /* Predicted but still in flight: a bounded wait for RAM bytes
             * beats an immediate cold read of the slow local disk. */
            const uint32_t wait_us = ds4_gpu_expert_stage_wait_us();
            if (wait_us > 0 &&
                ds4_gpu_expert_stage_pending(ctx->layer_index, expert_id)) {
                const double deadline = ds4_gpu_now_ms() + (double)wait_us / 1000.0;
                do {
                    usleep(100);
                    staged = ds4_gpu_expert_stage_find(ctx->layer_index, expert_id,
                                                       unit % 3u, &slen);
                } while (!staged && ds4_gpu_now_ms() < deadline);
            }
        }
        if (staged && slen == len) {
            const double t0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
            memcpy(dst, staged, (size_t)len);
            if (ctx->io_profile) {
                __sync_fetch_and_add(&g_io_prof_copy_ns,
                                     (uint64_t)((ds4_gpu_now_ms() - t0) * 1e6));
                __sync_fetch_and_add(&g_io_prof_hit_bytes, len);
            }
            __sync_fetch_and_add(&g_stage_hit_bytes_total, len);
            return;
        }
    }
    if (hard_src) {
        const double t0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
        memcpy(dst, hard_src, (size_t)len);
        if (ctx->io_profile) {
            __sync_fetch_and_add(&g_io_prof_copy_ns,
                                 (uint64_t)((ds4_gpu_now_ms() - t0) * 1e6));
            __sync_fetch_and_add(&g_io_prof_hit_bytes, len);
        }
        return;
    }
    /* P1.1 single-copy path: SSD -> Shared scratch in one pread, no page
     * fault, no second memcpy.  Bytes are identical to the mmap copy (same
     * file offsets), so logits are bit-exact by construction. */
    if (ctx->use_pread) {
        const double t0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
        if (ds4_gpu_pread_full(ctx->pread_fd, dst, src_off, (size_t)len)) {
            if (ctx->io_profile) {
                __sync_fetch_and_add(&g_io_prof_pread_ns,
                                     (uint64_t)((ds4_gpu_now_ms() - t0) * 1e6));
                __sync_fetch_and_add(&g_io_prof_cold_bytes, len);
            }
            return;
        }
        /* pread failed (EIO/short read): fall back to the proven mmap copy. */
        __sync_fetch_and_add(&g_io_prof_pread_fallbacks, 1u);
    }
    if (ctx->io_profile) {
        const double tf0 = ds4_gpu_now_ms();
        g_io_prof_touch_sink += ds4_gpu_expert_touch_pages(ctx->map + src_off, (size_t)len);
        const double tc0 = ds4_gpu_now_ms();
        __sync_fetch_and_add(&g_io_prof_fault_ns, (uint64_t)((tc0 - tf0) * 1e6));
        memcpy(dst, ctx->map + src_off, (size_t)len);
        __sync_fetch_and_add(&g_io_prof_copy_ns,
                             (uint64_t)((ds4_gpu_now_ms() - tc0) * 1e6));
        __sync_fetch_and_add(&g_io_prof_cold_bytes, len);
        return;
    }
    memcpy(dst, ctx->map + src_off, (size_t)len);
}

/* Complete a claimed unit locally (used for page-cache hits, tail units and
 * link-failure fallbacks so correctness never depends on the peer).  Prefers
 * the single-copy pread path like the local gather workers; cached pages make
 * pread a RAM copy anyway. */
static void ds4_gpu_expert_remote_local_finish(ds4_metal_expert_gather_ctx *ctx,
                                               uint64_t src_off,
                                               uint8_t *dst,
                                               uint64_t len) {
    const double t0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
    if (!(ctx->use_pread &&
          ds4_gpu_pread_full(ctx->pread_fd, dst, src_off, (size_t)len))) {
        memcpy(dst, ctx->map + src_off, (size_t)len);
    }
    if (ctx->io_profile) {
        __sync_fetch_and_add(&g_io_prof_copy_ns,
                             (uint64_t)((ds4_gpu_now_ms() - t0) * 1e6));
        __sync_fetch_and_add(&g_io_prof_cold_bytes, len);
    }
}

/* Last few gather units must stay local: a ~2ms remote round-trip claimed at
 * the end of a layer stalls every finished local thread (measured: the tail
 * ate the mid-layer remote gains).  Reserve about one unit per local gather
 * thread; locals retire those in a single parallel round. */
static uint32_t ds4_gpu_expert_remote_tail_reserve(void) {
    static uint32_t cached;
    static int initialized;
    if (!initialized) {
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_EXPERT_REMOTE_TAIL_RESERVE",
                                     ds4_gpu_expert_gather_threads());
        if (v > 64u) v = 64u;
        cached = (uint32_t)v;
        initialized = 1;
    }
    return cached;
}

/* The Thunderbolt link idles for the whole worker pipeline half (~250ms per
 * decoded token); TCP slow-start-after-idle then collapses the congestion
 * window, so the first staged fetches of the next token (the hash layers 0/1,
 * tightest deadlines!) crawl through the ramp-up.  While parked, each fetch
 * thread keeps its connection warm with a small periodic read.  0 disables. */
static uint32_t ds4_gpu_expert_link_keepalive_ms(void) {
    static uint32_t cached = UINT32_MAX;
    if (cached == UINT32_MAX) {
        /* Default off: measured a net regression (2.03 -> 1.79 t/s) with 40ms
         * keepalives; kept as an A/B knob only. */
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_EXPERT_LINK_KEEPALIVE_MS", 0u);
        if (v > 1000u) v = 1000u;
        cached = (uint32_t)v;
    }
    return cached;
}

static void *ds4_gpu_expert_remote_fetch_worker(void *arg) {
    const int slot = (int)(intptr_t)arg;
    uint8_t *ka_buf = NULL;
    const uint32_t ka_bytes = 262144u;
    for (;;) {
        int do_keepalive = 0;
        pthread_mutex_lock(&g_rf_mu);
        while (!g_rf_ctx && !ds4_gpu_expert_stage_has_work()) {
            const uint32_t ka_ms = ds4_gpu_expert_link_keepalive_ms();
            if (ka_ms == 0 || g_rf_link_down) {
                pthread_cond_wait(&g_rf_cv, &g_rf_mu);
                continue;
            }
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += (long)ka_ms * 1000000L;
            while (ts.tv_nsec >= 1000000000L) { ts.tv_nsec -= 1000000000L; ts.tv_sec++; }
            if (pthread_cond_timedwait(&g_rf_cv, &g_rf_mu, &ts) == ETIMEDOUT &&
                !g_rf_ctx && !ds4_gpu_expert_stage_has_work()) {
                do_keepalive = 1;
                break;
            }
        }
        ds4_metal_expert_gather_ctx *ctx = (ds4_metal_expert_gather_ctx *)g_rf_ctx;
        pthread_mutex_unlock(&g_rf_mu);

        if (do_keepalive) {
            if (!ka_buf) ka_buf = (uint8_t *)malloc(ka_bytes);
            if (ka_buf && !g_rf_link_down) {
                (void)ds4_dist_expert_fetch(slot, 0, ka_buf, ka_bytes);
            }
            continue;
        }

        if (!ctx) {
            /* Staging work: pull the predicted next-layer experts from the
             * peer into the RAM slots (runs during the current layer's local
             * gather + GPU window; no layer-tail constraint).  Most recently
             * armed slot first: its deadline is closest and link time spent
             * on the stale parity is wasted. */
            const int first = g_stage_recent & 1;
            for (int pass = 0; pass < 2; pass++) {
                ds4_metal_stage_slot *s = &g_stage[(first + pass) & 1];
                while (s->gen != 0 && ds4_gpu_expert_stage_fetch_one(s, slot)) {}
            }
            continue;
        }

        /* Pipelined fetch: keep up to 2 requests in flight on this slot's
         * connection so the server-side pread and the wire transfer overlap
         * and the request RTT leaves the per-unit critical path.  Units whose
         * bytes are already in the LOCAL page cache (prefetch read-ahead or
         * natural reuse) are finished with a RAM copy instead of spending
         * Thunderbolt bandwidth on them. */
        ds4_metal_rf_pending pend[2];
        uint32_t npend = 0;
        int exhausted = 0;
        const uint32_t total_units = ctx->n_active * 3u;
        const uint32_t tail_reserve = ds4_gpu_expert_remote_tail_reserve();
        const uint32_t remote_cutoff =
            total_units > tail_reserve ? total_units - tail_reserve : 0u;
        for (;;) {
            while (!exhausted && npend < 2u) {
                if (g_rf_link_down) { exhausted = 1; break; }
                /* Tail guard: never claim into the reserved zone (peek, then
                 * re-check after the claim in case of a race). */
                if (ctx->next_slot >= remote_cutoff) { exhausted = 1; break; }
                const uint32_t unit = __sync_fetch_and_add(&ctx->next_slot, 1u);
                if (unit >= total_units) { exhausted = 1; break; }
                uint64_t len = 0;
                uint64_t src_off = 0;
                uint8_t *dst = NULL;
                const uint8_t *hard_src = NULL;
                if (!ds4_gpu_expert_gather_unit_resolve(ctx, unit, &src_off, &dst, &len, &hard_src)) {
                    ctx->ok = 0;
                    __sync_fetch_and_add(&ctx->done, 1u);
                    continue;
                }
                if (hard_src) {
                    const double t0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
                    memcpy(dst, hard_src, (size_t)len);
                    if (ctx->io_profile) {
                        __sync_fetch_and_add(&g_io_prof_copy_ns,
                                             (uint64_t)((ds4_gpu_now_ms() - t0) * 1e6));
                        __sync_fetch_and_add(&g_io_prof_hit_bytes, len);
                    }
                    __sync_fetch_and_add(&ctx->done, 1u);
                    continue;
                }
                if (unit >= remote_cutoff) {
                    /* Raced into the tail zone: keep it local. */
                    ds4_gpu_expert_remote_local_finish(ctx, src_off, dst, len);
                    __sync_fetch_and_add(&ctx->done, 1u);
                    exhausted = 1;
                    break;
                }
                if (ds4_gpu_expert_range_mostly_cached(ctx->map, src_off, len)) {
                    ds4_gpu_expert_remote_local_finish(ctx, src_off, dst, len);
                    __sync_fetch_and_add(&ctx->done, 1u);
                    continue;
                }
                if (!ds4_dist_expert_fetch_send(slot, src_off, (uint32_t)len)) {
                    if (!g_rf_link_down) {
                        g_rf_link_down = 1;
                        fprintf(stderr,
                                "ds4: expert remote fetch link down; gather continues local-only\n");
                    }
                    ds4_gpu_expert_remote_local_finish(ctx, src_off, dst, len);
                    __sync_fetch_and_add(&ctx->done, 1u);
                    exhausted = 1;
                    break;
                }
                pend[npend].src_off = src_off;
                pend[npend].dst = dst;
                pend[npend].len = len;
                pend[npend].t0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
                npend++;
            }
            if (npend == 0) break;
            /* Drain the oldest in-flight response. */
            if (ds4_dist_expert_fetch_recv(slot, pend[0].dst, (uint32_t)pend[0].len)) {
                if (ctx->io_profile) {
                    __sync_fetch_and_add(&g_io_prof_remote_ns,
                                         (uint64_t)((ds4_gpu_now_ms() - pend[0].t0) * 1e6));
                    __sync_fetch_and_add(&g_io_prof_remote_bytes, pend[0].len);
                }
            } else {
                if (!g_rf_link_down) {
                    g_rf_link_down = 1;
                    fprintf(stderr,
                            "ds4: expert remote fetch link down; gather continues local-only\n");
                }
                ds4_gpu_expert_remote_local_finish(ctx, pend[0].src_off, pend[0].dst, pend[0].len);
            }
            __sync_fetch_and_add(&ctx->done, 1u);
            pend[0] = pend[1];
            npend--;
        }

        /* Wait for the entry to retire this ctx so we don't spin on an
         * exhausted cursor (pointer compare only; never dereferenced here). */
        pthread_mutex_lock(&g_rf_mu);
        while (g_rf_ctx == ctx) pthread_cond_wait(&g_rf_cv, &g_rf_mu);
        pthread_mutex_unlock(&g_rf_mu);
    }
    return NULL;
}

/* Lazy init on the serial gather path: by the first gather both engines are
 * up (the distributed session is already established), so the connect is
 * race-free with the peer's listener.
 * Wave 27/28: the first connect can hit a transient EHOSTUNREACH (Thunderbolt
 * bridge ARP starves under prefill rfetch saturation -- observed on the
 * worker->coordinator reverse-efetch dial while ping/route were fine once the
 * link went quiet).  A one-shot failure used to disable remote fetch for the
 * whole run; wave-27's 8x2s retries all landed inside the ~150s code-edit
 * prefill storm and still died.  Retry every 2s for up to 150 attempts
 * (~5min) so attempts reach the decode phase where the link is idle. */
static pthread_mutex_t g_rf_init_mu = PTHREAD_MUTEX_INITIALIZER;

static int g_rf_live;

/* live fetch worker threads */

int ds4_gpu_expert_remote_fetch_slots(void) {
    static int attempts;
    static double retry_after_ms;
    if (g_rf_live > 0) return g_rf_live;
    /* Wave 30 accept mode (reverse-established transport): the kick thread
     * owns the whole init -- the gather path must not dial anything. */
    if (getenv("DS4_DIST_EXPERT_FETCH_ACCEPT_PORT")) return g_rf_live;
    const char *host = getenv("DS4_DIST_EXPERT_FETCH_HOST");
    if (!host || !host[0]) return 0;
    /* Wave 29: also dialed from the quiet-window kick thread; serialize with
     * the gather path (a contended caller just reports "not yet"). */
    if (pthread_mutex_trylock(&g_rf_init_mu) != 0) return 0;
    if (g_rf_live > 0 || attempts >= 150) {
        const int n = g_rf_live;
        pthread_mutex_unlock(&g_rf_init_mu);
        return n;
    }
    const double now = ds4_gpu_now_ms();
    if (attempts > 0 && now < retry_after_ms) {
        pthread_mutex_unlock(&g_rf_init_mu);
        return 0;
    }
    attempts++;
    retry_after_ms = now + 2000.0;
    const int port = (int)ds4_gpu_env_u64("DS4_DIST_EXPERT_FETCH_PORT", 5606u);
    int conns = (int)ds4_gpu_env_u64("DS4_DIST_EXPERT_FETCH_CONNS", 3u);
    if (conns > 8) conns = 8;
    /* Handshake against the BASE model size (efetch serves the ~81 GiB GGUF),
     * not g_model_map_size which the MTP draft load overwrites with 2.14 GiB. */
    const uint64_t efetch_size = g_efetch_model_size ? g_efetch_model_size : g_model_map_size;
    const int n = ds4_dist_expert_fetch_client_init(host, port, conns, efetch_size);
    for (int i = 0; i < n; i++) {
        pthread_t th;
        if (pthread_create(&th, NULL, ds4_gpu_expert_remote_fetch_worker,
                           (void *)(intptr_t)i) == 0) {
            pthread_detach(th);
            g_rf_live++;
        }
    }
    if (g_rf_live) {
        fprintf(stderr,
                "ds4: expert gather pulls from peer SSD via %d remote fetch thread(s)%s "
                "(attempt %d)\n",
                g_rf_live, attempts > 1 ? " after retry" : "", attempts);
    } else if (attempts >= 150) {
        fprintf(stderr,
                "ds4: expert-fetch disabled after 150 failed connect attempts\n");
    } else if (attempts == 1 || attempts % 16 == 0) {
        fprintf(stderr,
                "ds4: expert-fetch connect attempt %d/150 failed; retrying every 2s\n",
                attempts);
    }
    const int live = g_rf_live;
    pthread_mutex_unlock(&g_rf_init_mu);
    return live;
}

/* Wave 29: the in-run retries almost never see a quiet bridge -- coordinator
 * staging keeps 6 efetch connections saturating the Thunderbolt link for the
 * entire run (prefill AND decode), so the worker's reverse-efetch ARP probes
 * starve from the first gather to the last (observed: 31 straight
 * EHOSTUNREACH).  The only reliably quiet window is right after the worker
 * accepts the coordinator's control connection, before the first prefill
 * frame.  Dial from a short-lived background thread in that window. */
static void *ds4_gpu_expert_remote_fetch_kick_main(void *arg) {
    (void)arg;
    /* Wave 30 accept mode: the worker cannot dial out at all (in-process
     * EHOSTUNREACH every attempt while shell tools succeed), so it LISTENS
     * and the coordinator's serve-dial thread connects in.  This kick fires
     * right after the worker accepts the control connection -- the same
     * moment the coordinator's engine (already up, it just dialed us) starts
     * its serve-dial backoff loop, so the rendezvous is race-free. */
    const char *aport = getenv("DS4_DIST_EXPERT_FETCH_ACCEPT_PORT");
    if (aport && aport[0]) {
        pthread_mutex_lock(&g_rf_init_mu);
        if (g_rf_live == 0) {
            const int port = atoi(aport);
            int conns = (int)ds4_gpu_env_u64("DS4_DIST_EXPERT_FETCH_CONNS", 3u);
            if (conns > 8) conns = 8;
            const int n = ds4_dist_expert_fetch_accept_init(port, conns,
                              g_efetch_model_size ? g_efetch_model_size : g_model_map_size);
            for (int i = 0; i < n; i++) {
                pthread_t th;
                if (pthread_create(&th, NULL, ds4_gpu_expert_remote_fetch_worker,
                                   (void *)(intptr_t)i) == 0) {
                    pthread_detach(th);
                    g_rf_live++;
                }
            }
            if (g_rf_live) {
                fprintf(stderr,
                        "ds4: expert gather pulls from peer SSD via %d remote fetch "
                        "thread(s) (accept mode)\n",
                        g_rf_live);
            }
        }
        pthread_mutex_unlock(&g_rf_init_mu);
        return NULL;
    }
    for (int i = 0; i < 10 && ds4_gpu_expert_remote_fetch_slots() == 0; i++) {
        usleep(700 * 1000);
    }
    return NULL;
}

void ds4_gpu_expert_remote_fetch_kick(void) {
    const char *host = getenv("DS4_DIST_EXPERT_FETCH_HOST");
    const char *aport = getenv("DS4_DIST_EXPERT_FETCH_ACCEPT_PORT");
    if ((!host || !host[0]) && (!aport || !aport[0])) return;
    pthread_t th;
    if (pthread_create(&th, NULL, ds4_gpu_expert_remote_fetch_kick_main, NULL) == 0) {
        pthread_detach(th);
    }
}

static ds4_metal_expert_gather_worker_arg g_expert_gather_worker_args[16];
