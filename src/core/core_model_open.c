/* core_model_open.c — expert keep-map + model_open/parse (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"

/* Parse the optional ds4.expert_keep_map.* metadata written by gguf-tools/
 * shrink_gguf.py. When present, the routed-expert tensors only carry the kept
 * rows; this builds the original-id -> compact-slot table so routing still works
 * with the full 256-wide router logits. No-op for a full model. */
static void load_expert_keep_map(ds4_model *m) {
    ds4_array_ref counts_arr;
    ds4_array_ref ids_arr;
    const bool has_counts = model_get_array(m, "ds4.expert_keep_map.kept_counts", &counts_arr);
    const bool has_ids    = model_get_array(m, "ds4.expert_keep_map.original_ids", &ids_arr);
    if (!has_counts && !has_ids) return;
    if (has_counts != has_ids) {
        ds4_die("ds4.expert_keep_map: kept_counts and original_ids must appear together");
    }
    if (counts_arr.type != GGUF_VALUE_INT32 && counts_arr.type != GGUF_VALUE_UINT32) {
        ds4_die("ds4.expert_keep_map.kept_counts must be an int32 array");
    }
    if (ids_arr.type != GGUF_VALUE_INT32 && ids_arr.type != GGUF_VALUE_UINT32) {
        ds4_die("ds4.expert_keep_map.original_ids must be an int32 array");
    }
    if (counts_arr.len == 0 || counts_arr.len > 1024) {
        ds4_die("ds4.expert_keep_map.kept_counts has an unreasonable length");
    }

    m->expert_layer_count = (uint32_t)counts_arr.len;
    m->expert_kept_count = calloc(m->expert_layer_count, sizeof(m->expert_kept_count[0]));
    m->expert_orig_to_compact = calloc((size_t)m->expert_layer_count * DS4_N_EXPERT,
                                       sizeof(m->expert_orig_to_compact[0]));
    if (!m->expert_kept_count || !m->expert_orig_to_compact) {
        ds4_die("out of memory while allocating expert keep-map");
    }
    for (size_t i = 0; i < (size_t)m->expert_layer_count * DS4_N_EXPERT; i++) {
        m->expert_orig_to_compact[i] = -1;          /* -1 == dropped */
    }

    uint64_t expected_ids_len = 0;
    ds4_cursor cc = cursor_at(m, counts_arr.data_pos);
    for (uint32_t il = 0; il < m->expert_layer_count; il++) {
        int32_t v = 0;
        if (!cursor_read(&cc, &v, sizeof(v))) ds4_die(cc.error);
        if (v < 1 || v > (int32_t)DS4_N_EXPERT) {
            ds4_die("ds4.expert_keep_map.kept_counts has an out-of-range value");
        }
        m->expert_kept_count[il] = (uint16_t)v;
        expected_ids_len += (uint64_t)v;
    }
    if (ids_arr.len != expected_ids_len) {
        ds4_die("ds4.expert_keep_map.original_ids length does not match sum(kept_counts)");
    }

    ds4_cursor ic = cursor_at(m, ids_arr.data_pos);
    for (uint32_t il = 0; il < m->expert_layer_count; il++) {
        const uint16_t k = m->expert_kept_count[il];
        int16_t *row = m->expert_orig_to_compact + (size_t)il * DS4_N_EXPERT;
        for (uint16_t slot = 0; slot < k; slot++) {
            int32_t orig = 0;
            if (!cursor_read(&ic, &orig, sizeof(orig))) ds4_die(ic.error);
            if (orig < 0 || orig >= (int32_t)DS4_N_EXPERT) {
                ds4_die("ds4.expert_keep_map.original_ids contains an out-of-range expert id");
            }
            if (row[orig] != -1) {
                ds4_die("ds4.expert_keep_map.original_ids has a duplicate expert id in one layer");
            }
            row[orig] = (int16_t)slot;
        }
    }
    m->expert_shrunken = true;
    uint32_t min_kept = DS4_N_EXPERT;
    for (uint32_t il = 0; il < m->expert_layer_count; il++) {
        if (m->expert_kept_count[il] < min_kept) min_kept = m->expert_kept_count[il];
    }
    fprintf(stderr,
            "ds4: reduced-expert model: keep-map over %u layers (min kept %u of %u)\n",
            m->expert_layer_count, min_kept, (uint32_t)DS4_N_EXPERT);
}


void model_close(ds4_model *m) {
    if (!m) return;
    if (m->corr) { corr_free(m->corr); m->corr = NULL; }
    if (m->residual) { residual_free(m->residual); m->residual = NULL; }
    if (m->zchain) { ds4_zchain_free(m->zchain); m->zchain = NULL; }
    free(m->kv);
    free(m->tensors);
    free(m->expert_kept_count);
    free(m->expert_orig_to_compact);
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

static void model_prefetch_cpu_mapping(const ds4_model *m) {
    if (!m || !m->map || m->size == 0) return;

    /*
     * CPU generation touches expert weights according to router decisions, so a
     * long decode can fault in model pages that the prompt never touched. On
     * current Darwin kernels we have seen those late file-backed faults trigger
     * an OS-level VM panic in map-count accounting. This hint does not copy or
     * pin the GGUF; it just asks the kernel to start bringing the read-only
     * mapping into the page cache before token generation reaches it.
     */
#if defined(POSIX_MADV_WILLNEED)
    const int rc = posix_madvise((void *)m->map, (size_t)m->size, POSIX_MADV_WILLNEED);
    if (rc != 0) {
        ds4_log(stderr,
                DS4_LOG_WARNING,
                "ds4: warning: POSIX_MADV_WILLNEED failed for CPU model mapping: %s\n",
                strerror(rc));
    }
#else
    (void)m;
#endif
}

/* Read the GGUF metadata table.  Values stay in the mmap; we store offsets so
 * later validation can decode only the keys it needs. */
static void parse_metadata(ds4_model *m, ds4_cursor *c) {
    m->kv = calloc((size_t)m->n_kv, sizeof(m->kv[0]));
    if (!m->kv) ds4_die("out of memory while allocating metadata table");

    m->alignment = 32;

    for (uint64_t i = 0; i < m->n_kv; i++) {
        ds4_kv *kv = &m->kv[i];

        if (!cursor_string(c, &kv->key)) ds4_die(c->error);
        if (!cursor_u32(c, &kv->type)) ds4_die(c->error);

        kv->value_pos = c->pos;

        if (ds4_streq(kv->key, "general.alignment") &&
            kv->type == GGUF_VALUE_UINT32)
        {
            ds4_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t alignment;
            if (cursor_u32(&tmp, &alignment) && alignment != 0) {
                m->alignment = alignment;
            }
        }

        if (!skip_value(c, kv->type, 0)) ds4_die(c->error);
    }
}

/* Read the tensor directory and convert relative GGUF offsets to absolute
 * mmap offsets.  Tensor bytes are still never copied here. */
static void parse_tensors(ds4_model *m, ds4_cursor *c) {
    m->tensors = calloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    if (!m->tensors) ds4_die("out of memory while allocating tensor table");

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];

        if (!cursor_string(c, &t->name)) ds4_die(c->error);
        if (!cursor_u32(c, &t->ndim)) ds4_die(c->error);
        if (t->ndim == 0 || t->ndim > DS4_MAX_DIMS) {
            ds4_die("tensor has an unsupported number of dimensions");
        }

        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) ds4_die(c->error);
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) {
                ds4_die("tensor element count overflow");
            }
            t->elements *= t->dim[d];
        }

        if (!cursor_u32(c, &t->type)) ds4_die(c->error);
        if (!cursor_u64(c, &t->rel_offset)) ds4_die(c->error);

        if (!tensor_nbytes(t->type, t->elements, &t->bytes)) {
            ds4_log(stderr,
                DS4_LOG_WARNING,
                "ds4: warning: tensor %.*s has unsupported GGUF type %u\n",
                (int)t->name.len, t->name.ptr, t->type);
        }
    }

    m->tensor_data_pos = align_up(c->pos, m->alignment);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_pos) {
            ds4_die("tensor offset overflow");
        }
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->bytes != 0 &&
            (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset))
        {
            ds4_die("tensor points outside GGUF file");
        }
        if (t->bytes > m->max_tensor_bytes) {
            m->max_tensor_bytes = t->bytes;
        }
    }
}

/* Only the engine BASE model may arm process-wide env defaults from its
 * tensor types (the go1b/go2b numeric-safety block inside model_open below).
 * Support-model opens run in the base model's process -- the residual/corr
 * sidecars and tokenizer-only opens (a go-family sidecar would otherwise arm
 * go defaults onto a non-go BASE sampler). model_open's signature is frozen
 * (ds4_internal.h; external caller ds4_corr.c), so the base loader opts in
 * through this file-scope flag instead of a new parameter -- default false
 * keeps every sidecar/tokenizer open arming-free. */
bool g_model_open_arm_env_defaults = false;

/* Open and map the GGUF once.  Metal needs a shared mapping for no-copy
 * MTLBuffers; CPU uses a private read-only mapping to avoid Darwin VM stress.
 * Tokenizer-only callers pass prefetch_cpu=false so inspecting tokens never
 * walks the huge tensor payload. */
void model_open(ds4_model *m, const char *path, bool metal_mapping,
                       bool prefetch_cpu) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd == -1) ds4_die_errno("cannot open model", path);

    struct stat st;
    if (fstat(fd, &st) == -1) ds4_die_errno("cannot stat model", path);
    if (st.st_size < 32) ds4_die("model file is too small to be GGUF");

    /*
     * Metal wraps slices of this mapping as no-copy MTLBuffers, so the Metal
     * path keeps the file-backed shared mapping. The CPU path only reads the
     * weights through normal pointers and should not inherit Metal's VM policy:
     * use a private read-only mapping there.
     *
     * This is deliberately defensive against an OS-level Darwin VM bug observed
     * while the CPU backend streams the very large GGUF through a shared mmap:
     * the kernel can panic in VM map-count accounting instead of returning a
     * normal user-space failure. Keeping CPU inference off the shared mapping
     * avoids that VM accounting path while preserving normal file-backed reads.
     */
    const int mmap_flags = metal_mapping ? MAP_SHARED : MAP_PRIVATE;
    void *map = MAP_FAILED;
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    /* 先丢掉本文件已有的 page cache 页, 再带 MADV_HUGEPAGE 映射 —— 否则内核会拿
     * cache 里现成的 4KiB 页直接用, 大页永远只覆盖新读入的那部分(实测卡在 12-16%)。
     * POSIX_FADV_DONTNEED 只作用于这一个 fd 对应的文件, 是进程级操作, 不需要特权。 */
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    /* ★2MiB 对齐映射★: 文件页 THP 要求虚拟地址按大页对齐(文件偏移 0 本就满足),
     * 否则整段退回 4KiB。内核给的地址通常不对齐, 实测大页覆盖率只有 8.5%。
     * 做法: 先占一段带余量的匿名区探出对齐地址, 再 MAP_FIXED 把文件映上去 ——
     * MAP_FIXED 直接覆盖占位区, 中间没有别的线程能插进来抢地址。 */
    {
        const size_t HP = 2u * 1024u * 1024u;
        const size_t want = (size_t)st.st_size;
        void *probe = mmap(NULL, want + HP, PROT_NONE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (probe != MAP_FAILED) {
            uintptr_t base = (uintptr_t)probe;
            uintptr_t aligned = (base + HP - 1u) & ~(uintptr_t)(HP - 1u);
            void *fixed = mmap((void *)aligned, want, PROT_READ,
                               mmap_flags | MAP_FIXED, fd, 0);
            if (fixed != MAP_FAILED) {
                if (aligned > base) (void)munmap(probe, (size_t)(aligned - base));
                const uintptr_t tail = aligned + want;
                const uintptr_t probe_end = base + want + HP;
                if (probe_end > tail) (void)munmap((void *)tail, (size_t)(probe_end - tail));
                map = fixed;
            } else {
                (void)munmap(probe, want + HP);
            }
        }
    }
#endif
    if (map == MAP_FAILED)
        map = mmap(NULL, (size_t)st.st_size, PROT_READ, mmap_flags, fd, 0);
    if (map == MAP_FAILED) ds4_die_errno("cannot mmap model", path);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    /* ★大页★: GB10 这类统一内存机器上 GPU 直接走 host 页表访问权重
     * (PageableMemoryAccess=1 且 UsesHostPageTables=1)。90 GiB 模型按 4KiB 页就是
     * 2200 万个页表项, TLB 完全装不下 —— 每次专家读都在做地址翻译, 表现为 GPU 利用率
     * 只有 6% 而 CPU/GPU/磁盘全都不忙。本机 THP 是 madvise 模式(不显式要就永远 4KiB),
     * 故此处显式请求; 内核不支持文件页 THP 时该调用无害地失败。 */
    {
        (void)madvise(map, (size_t)st.st_size, MADV_HUGEPAGE);
        /* MADV_HUGEPAGE 只影响"之后新读入"的页; 已在 page cache 里的 4KiB 页不会自动
         * 合并(khugepaged 每 10s 才扫 16MiB, 对 90GiB 等于没有)。MADV_COLLAPSE(6.1+)
         * 主动就地合并, 是把覆盖率从个位数推上去的唯一手段。它是同步的, 分段做并允许
         * 单段失败(碎片导致某段凑不出连续 2MiB 很正常, 跳过即可)。 */
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
        {
            const size_t CH = 512u * 1024u * 1024u;   /* 每次 512MiB, 避免一次长阻塞 */
            size_t done = 0;
            while (done < (size_t)st.st_size) {
                size_t n = (size_t)st.st_size - done;
                if (n > CH) n = CH;
                (void)madvise((char *)map + done, n, MADV_COLLAPSE);
                done += n;
            }
        }
    }
#endif

    m->fd = fd;
    m->map = map;
    m->size = (uint64_t)st.st_size;

    ds4_cursor c = cursor_at(m, 0);
    uint32_t magic;
    if (!cursor_u32(&c, &magic)) ds4_die(c.error);
    if (magic != DS4_GGUF_MAGIC) ds4_die("model is not a GGUF file");
    if (!cursor_u32(&c, &m->version)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_tensors)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_kv)) ds4_die(c.error);

    if (m->version != 3) ds4_die("only GGUF v3 is supported");

    parse_metadata(m, &c);
    parse_tensors(m, &c);
    /* ★模型嗅探自动武装已删除(2026-08-05 用户铁律)★: "带环境变量控制模型本身的代码
     * 不应该存在, 定型硬编码优化在模型变化的时候肯定是 bug"。旧块按 go1b/go2b 张量类型
     * setenv(MATH_SAFE/KV_RAW_F32/ROPE_EXP2_LOG2, GO2B 另加 REPEAT_FREQ=1) — 那是
     * 2026-07 超低比特 mono 时代的定型配方, 对高保真新模型(r64 top1 71.4 vs FP 71.7)
     * 是错配毒药(实测: REPEAT_FREQ=1 令生成空)。行为只由显式 env/配置决定; 需要数值
     * 安全的 launcher 自己显式设, 引擎不替模型做主。 */
    load_expert_keep_map(m);
#ifndef DS4_NO_GPU
    /* Upload the reduced-expert routing LUT to the GPU so routed-MoE matvecs can
     * translate full-256 router ids to compact expert-tensor slots. No-op for a
     * full model (keep-map absent). GPU-only: the keep-map table itself
     * (expert_orig_to_compact, built in load_expert_keep_map) is the source of
     * truth; the CPU reference path is not wired for shrunken models. */
    if (m->expert_shrunken && m->expert_orig_to_compact) {
        if (!ds4_gpu_set_expert_keep_lut(m->expert_orig_to_compact, m->expert_layer_count)) {
            ds4_die("failed to upload reduced-expert routing LUT to the GPU");
        }
    }
#endif

    if (!metal_mapping && prefetch_cpu) model_prefetch_cpu_mapping(m);
}

