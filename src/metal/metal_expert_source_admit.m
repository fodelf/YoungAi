/* metal_expert_source_admit.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

uint32_t ds4_gpu_expert_gather_threads(void) {
    static int initialized;
    if (!initialized) {
        initialized = 1;
        if (ds4_gpu_expert_offload_enabled()) {
            fprintf(stderr,
                    "ds4: A3 expert CPU gather parallel copy enabled: %u threads\n",
                    (unsigned)DS4_METAL_EXPERT_GATHER_THREADS);
        }
    }
    return DS4_METAL_EXPERT_GATHER_THREADS;
}

/* ---- project.md P0.1/P1.1: single-copy pread ----
 *
 * Measured bottleneck (execution log 2026-06-10): the A3 gather moves cold
 * expert bytes at ~1.5GB/s because every byte pays a 16KiB mmap page fault
 * (SSD -> page cache) plus a memcpy (page cache -> Shared scratch), serialized
 * behind a per-layer command drain.  The pread path replaces the fault+memcpy
 * double copy with one pread() per expert tensor straight into the scratch
 * MTLBuffer (token byte-exact, ~2.2x measured). */

int ds4_gpu_pread_full(int fd, void *dst, uint64_t src_off, size_t len) {
    uint8_t *p = (uint8_t *)dst;
    while (len > 0) {
        ssize_t r = pread(fd, p, len, (off_t)src_off);
        if (r < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (r == 0) return 0;
        p += (size_t)r;
        src_off += (uint64_t)r;
        len -= (size_t)r;
    }
    return 1;
}

/* Resolve once from the serial gather entry (before worker threads spawn) so
 * the cached statics never race.  Auto-adapts: streaming/offload models get
 * the single-copy pread win, resident models don't need it; a missing or
 * ambiguous model fd falls back to the proven mmap gather. */
int ds4_gpu_expert_pread_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_expert_offload_enabled() ? 1 : 0;
        if (cached && (g_model_fd < 0 || g_model_fd_conflict)) {
            fprintf(stderr,
                    "ds4: expert gather has no unambiguous model fd; "
                    "falling back to mmap gather\n");
            cached = 0;
        }
        if (cached) {
            fprintf(stderr, "ds4: expert gather single-copy pread enabled (fd=%d)\n",
                    g_model_fd);
        }
    }
    return cached;
}

/* Wave 24: verify/prefill batch gathers read hundreds of MiB of cold expert
 * bytes per layer with ~zero reuse (batch hit_mib==0 in every profile run),
 * yet a kc=12 verify round streams ~5.8GiB through the coordinator page
 * cache and evicts the mmap-resident backbone (Q8 attention/shared) pages.
 * The next single-token frame then re-faults the backbone inside the GPU
 * command execution: decode drain_ms ballooned 14.5 -> 46ms/layer (spikes to
 * 855ms on the first layers after a verify), inflating round-1 from ~550ms to
 * ~2.3s.  Route batch-site cold preads through a separate F_NOCACHE
 * descriptor so one-shot batch bytes stop evicting hot pages.  Always on; a
 * host without F_NOCACHE support falls back to the shared cached fd (there is
 * no way to emulate the no-cache read, so the fallback is the honest path). */
int g_expert_gather_nocache_call;

/* set on the serial encode path per gather call */

int ds4_gpu_expert_batch_nocache_enabled(void) {
    return 1;
}

/* Wave 25 A/B verdict: NOCACHE won on first-touch prefill chunks (smoke
 * 2.02 -> 2.09) but lost on kc<=16 verify rounds (code-edit 1.68 -> 1.48):
 * verify unions overlap decode-hot experts and neighbouring rounds, so
 * bypassing the cache re-reads warm bytes from the slow mini SSD.  Keep
 * NOCACHE for big (prefill-sized) batches only (threshold history: 24 was
 * calibrated for K=16 verify caps; the K=64 ladder sends kc=25/33 verify
 * batches, so 64 keeps every verify batch on the cached fd while prefill
 * frames of 128 tokens stay NOCACHE). */
uint32_t ds4_gpu_expert_batch_nocache_min_tokens(void) {
    return DS4_METAL_BATCH_NOCACHE_MIN_TOKENS;
}
