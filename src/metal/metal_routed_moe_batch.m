/* metal_routed_moe_batch.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
/* EXCEPTION(>500行): 单函数 ds4_gpu_routed_moe_batch_tensor, 函数内拆分是后续工序 */
#import "metal_internal.h"
#include "vq_fmt.h"

int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const ds4_gpu_residual_set *residual,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        uint32_t                slot_start,   /* TP Phase-3 batch split: owned slot range */
        uint32_t                slot_count,   /* 0 or n_expert => no split (full) */
        bool                   *mid_is_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const ds4_gpu_residual_set *go1b_res = (residual && residual->gate_ptr) ? residual : NULL;
    if (!out || !gate || !up || !mid || !x || !model_map || !selected || !weights ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 || n_expert > 6) {
        return 0;
    }
    if ((expert_in_dim % 256u) != 0 || (expert_mid_dim % 256u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> expertsbuf = ds4_gpu_tensor_buffer(experts);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        NSUInteger selected_off = ds4_gpu_tensor_offset(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        const uint64_t x_bytes = (uint64_t)n_tokens * expert_in_dim * sizeof(float);
        const uint64_t mid_bytes = (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);
        const uint64_t selected_bytes = (uint64_t)n_tokens * n_expert * sizeof(int);
        const uint64_t weights_bytes = (uint64_t)n_tokens * n_expert * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < mid_bytes ||
            ds4_gpu_tensor_bytes(up) < mid_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes) {
            fprintf(stderr, "ds4: Metal routed batch MoE received undersized activation buffers\n");
            return 0;
        }
        if (n_expert > 1 &&
            (!expertsbuf ||
             ds4_gpu_tensor_bytes(experts) < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float))) {
            fprintf(stderr, "ds4: Metal routed batch MoE received undersized expert output buffer\n");
            return 0;
        }

        const uint64_t gate_tensor_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
        const uint64_t down_tensor_bytes = (uint64_t)n_total_expert * down_expert_bytes;
        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        /* 合一 VQ GGUF: base gate/up 张量不在文件里(offset=0/bytes=0), do_vq 路
         * 从 blob+down 取数, gate/up view 不建也不引用 — 建 0 区间 view 会硬失败。 */
        const int vq_no_base_gate =
            (go1b_res && go1b_res->vq && gate_expert_bytes == 0);
        g_wrap_mlock_suppress = 1;   /* routed expert tensors: gathered, never wired */
        id<MTLBuffer> gate_buf = nil, up_buf = nil, down_buf = nil;
        if (!vq_no_base_gate) {
            gate_buf = ds4_gpu_wrap_model_range(model_map, model_size, gate_offset, gate_tensor_bytes, &gate_inner);
            up_buf = ds4_gpu_wrap_model_range(model_map, model_size, up_offset, gate_tensor_bytes, &up_inner);
        }
        /* ★冷 w2 也在 blob 时 down 是 bytes=0 影子张量(2026-08-01): --no-down 合并的文件里
         * base ffn_down_exps 不存在(省 11.42 GiB), 引擎侧由 routed_down_shadow() 顶住维度,
         * 真实字节从 blob 的 which=2 槽取 ⇒ 这里不该再要 down_buf, 也不该判它 nil 为失败。 */
        /* 判据用 offset 而非 bytes: 影子张量的维度是真的(routed_down_shadow 填了 DS4 常量),
         * 所以 down_tensor_bytes 按维度算出来非零; 真正的标志是 abs_offset==0(文件里没有它)。*/
        const int vq_no_base_down = (down_offset == 0);
        if (!vq_no_base_down)
            down_buf = ds4_gpu_wrap_model_range(model_map, model_size, down_offset, down_tensor_bytes, &down_inner);
        g_wrap_mlock_suppress = 0;
        if ((!vq_no_base_gate && (!gate_buf || !up_buf)) || (!vq_no_base_down && !down_buf)) {
            fprintf(stderr, "ds4: [moe-buf-nil] L%u gate=%d up=%d down=%d vq_no_base_gate=%d\n",
                    layer_index, gate_buf != nil, up_buf != nil, down_buf != nil, (int)vq_no_base_gate);
            return 0;
        }
        uint32_t source_n_total_expert = n_total_expert;
        /* 1-bit residual (go1b_res): decode-path (n_tokens=1) go1b mm_id matmul over a
         * CPU-gathered compacted copy of the active residual experts, summed into
         * gate/up. go1b has no mv_id kernel (force_mm), so decode runs the mm_id path;
         * the residual reuses the base's work map by matching its compacted layout. */
        const void *res_gate_ptr = NULL, *res_up_ptr = NULL, *res_down_ptr = NULL;
        const float *res_lut = NULL;   /* sparse residual: expert id -> slot or -1 */
        int do_residual = 0;
        int do_hot = 0;                /* go2b merged sidecar: hot/cold split path */
        int do_vq = 0;                 /* v2.2 VQ blob: 全专家 dequant→f16 scratch 路 */
        if (go1b_res && go1b_res->vq) { do_vq = 1; }
        if (getenv("DS4_VQ_DEBUG") && do_vq) {
            const float *xf = (const float*)((const uint8_t*)ds4_gpu_tensor_buffer(x).contents + ds4_gpu_tensor_offset(x));
            double n2=0; int nnan=0; for(uint32_t k=0;k<expert_in_dim;k++){ float v=xf[k]; if(v!=v||v>1e30f||v<-1e30f)nnan++; n2+=(double)v*v; }
            fprintf(stderr, "[VQ_XIN] L%u xnorm=%.3f nan/inf=%d/%u\n", layer_index, sqrt(n2), nnan, expert_in_dim);
        }
        if (go1b_res) {
            /* residual mm_id output scratch: gate/up passes need n_tokens*n_expert*mid
             * floats, the down pass n_tokens*n_expert*out — size for the larger and
             * grow across prefill chunks (decode fits the old 128 KiB floor). The
             * n_tokens==1 gate this replaces silently dropped the residual on every
             * prefill/batch forward, so scored NLL and the KV built during prefill
             * came from the bare base while decode ran corrected weights. */
            const uint64_t rneed = (uint64_t)n_tokens * n_expert *
                (uint64_t)(expert_mid_dim > out_dim ? expert_mid_dim : out_dim) * sizeof(float);
            if (!g_moe_go1b_res_scratch || (uint64_t)g_moe_go1b_res_scratch.length < rneed)
                g_moe_go1b_res_scratch = [g_device newBufferWithLength:(NSUInteger)rneed
                                                               options:MTLResourceStorageModePrivate];
            res_gate_ptr = go1b_res->gate_ptr;
            res_up_ptr   = go1b_res->up_ptr;
            res_down_ptr = go1b_res->down_ptr;
            res_lut      = go1b_res->lut;
            do_residual = (g_moe_go1b_res_scratch != nil && res_gate_ptr && res_up_ptr && res_down_ptr);
            if (go1b_res->merged2b) {
                /* go2b sidecar: never run the legacy add pass; the split below
                 * needs the LUT (sparse hot set) to route picks. */
                do_hot = (res_lut && res_gate_ptr && res_up_ptr && res_down_ptr);
                do_residual = 0;
            }
            /* Residual hot-pin (DS4_RESID_PIN_MLOCK_MB, default off): one-time
             * per layer, wires the pin-file experts' residual slots so the
             * per-token gather memcpy below stops paging the sidecar off SSD.
             * go1b residual only: its per-expert strides equal the base's
             * (identical dims/type); the go2b merged sidecar (do_hot) uses
             * g2_* strides and is not covered here. */
            if (do_residual)
                ds4_gpu_resid_pin_mlock_layer(layer_index, res_gate_ptr, res_up_ptr,
                                              res_down_ptr, res_lut,
                                              gate_expert_bytes, down_expert_bytes);
            if (getenv("DS4_RESIDUAL_DEBUG"))
                fprintf(stderr, "ds4: [residual-batch] L%u do=%d ntok=%u\n", layer_index, do_residual, n_tokens);
        }
        /* VQ 层自包含: gather 已把全专家 dequant 成完整 f16 权重(gate/up VQ blob + w2 hot-blob
         * 或冷 base 展开), MoE 走单 mm_id pass。residual_set_for 对 VQ 层把 gate_ptr/up_ptr/
         * down_ptr 都设成 vq_raw(≠go1b residual 格式), 若让 do_residual/do_hot 生效会把 VQ blob
         * 当成 go1b sign 字节误读并叠加 → logits 塌 BOS。故 VQ 路强制关闭这两条叠加 pass。 */
        if (do_vq) { do_residual = 0; do_hot = 0; }
        /* P-OVL state hoisted to function scope: the encode block must read the
         * active expert set when the two-pass overlap defers the byte gather. */
        uint32_t active_ids[1024];
        uint32_t n_active = 0;
        bool moe_overlap_active = false;
        /* R5-C go2b split state (hoisted: encode blocks run after the gather scope) */
        uint32_t hot_slot_ids[64];
        uint32_t n_hot_slots = 0;
        uint64_t n_hot_picks = 0;
        int hot_snapshotted = 0;

        /* DS4_METAL_EXPERT_OFFLOAD is meant for over-budget target models whose
         * routed experts are deliberately non-resident and must be gathered into
         * compact scratch/pool before the MoE kernels index them -- a non-resident
         * wrapped mmap view reads as ZEROS on the GPU, so the selected experts must
         * be CPU-copied into resident scratch first.  The MTP support model is mapped
         * as a fully resident model range; forcing its Q4_K routed experts through the
         * A3 scratch path just copies resident bytes again and made distributed MTP a
         * net slowdown.  Leave non-offload (resident) routed tensors on the direct path.
         *
         * go1b (strict 1-bit, mv-less) ships as a 42 GiB over-budget offload model and
         * must take this gather path too: the byte gather + slot-remap below are quant-
         * type agnostic, and the q2-specific pool/stream/source-cache sub-paths stay
         * gated to IQ2_XXS+Q2_K so go1b falls through to the basic resident-scratch
         * gather and then the mm_id kernel reads its experts from that resident copy. */
        const bool routed_pair_is_q2 =
            gate_type == DS4_METAL_TENSOR_IQ2_XXS && down_type == DS4_METAL_TENSOR_Q2_K;
        const bool routed_is_go1b = (gate_type == DS4_METAL_TENSOR_GO1B);
        /* go2b (type 41) as a BASE routed-expert type (monolithic mixed model, not
         * the overlay sidecar): the byte gather + slot-remap below are quant-type
         * agnostic (routed_expert_block_bytes handles go2b=68B), and the go2b mm_id
         * kernel is dispatched by gate_type downstream, so go2b falls through the
         * same basic resident-scratch gather as go1b.  Without this it stays resident
         * and OOMs a 16 GiB host on a 59 GiB monolithic model. */
        const bool routed_is_go2b = (gate_type == DS4_METAL_TENSOR_GO2B);
        const bool a3_expert_offload =
            ds4_gpu_expert_offload_enabled() &&
            !ds4_gpu_expert_offload_direct_enabled() &&
            (routed_pair_is_q2 || routed_is_go1b || routed_is_go2b);
        if (a3_expert_offload) {
            const int io_profile = ds4_gpu_expert_io_profile_enabled();
            /* Prefill-sized batch bytes are one-shot first-touch reads: keep
             * them out of the page cache.  Verify rounds (kc<=16) stay cached
             * -- their unions overlap decode-hot experts (wave 25 A/B). */
            g_expert_gather_nocache_call =
                (n_tokens >= ds4_gpu_expert_batch_nocache_min_tokens()) &&
                ds4_gpu_expert_batch_nocache_enabled();
            const int was_batched = (g_batch_cb != nil);
            double drain_ms = 0.0;
            if (was_batched) {
                const double drain_t0 = io_profile ? ds4_gpu_now_ms() : 0.0;
                if (ds4_gpu_expert_drain_commands("routed MoE drain") == 0) return 0;
                if (io_profile) drain_ms = ds4_gpu_now_ms() - drain_t0;
            }
            /* TP Phase-3 batch split: compact this peer's owned slot range to the
             * front in place (selectedbuf/weightsbuf are CPU-visible after the drain),
             * then run the rest with n_expert=slot_count so the gather/GEMM touch only
             * half of each token's experts -> halves the cold batch (prefill) IO. The
             * partial out is summed across peers by the caller's batch all-reduce.
             * dst=t*cnt+j < src=t*n_expert+slot_start+j, so the forward copy is safe. */
            if (slot_count > 0u && slot_count < n_expert) {
                int32_t *sel = (int32_t *)((uint8_t *)selectedbuf.contents + selected_off);
                float   *wt  = (float *)((uint8_t *)weightsbuf.contents + ds4_gpu_tensor_offset(weights));
                for (uint32_t t = 0; t < n_tokens; t++) {
                    for (uint32_t j = 0; j < slot_count; j++) {
                        sel[(size_t)t * slot_count + j] = sel[(size_t)t * n_expert + slot_start + j];
                        wt[(size_t)t * slot_count + j]  = wt[(size_t)t * n_expert + slot_start + j];
                    }
                }
                n_expert = slot_count;
            }
            /* Wave 36: batch union prediction.  The batch path never enqueued
             * prefetch jobs at all (only decode did, and with one row): verify
             * gathers ran unpredicted/cold.  Snapshot the drained batch hidden
             * and predict the NEXT layer's expert union; advisories land in
             * the next drain's idle-disk window.  Cached-fd batches only --
             * NOCACHE preads bypass the page cache, warming is useless there. */
            if (ds4_gpu_expert_prefetch_enabled() &&
                !g_expert_gather_nocache_call &&
                n_tokens > 1u &&
                xbuf.storageMode == MTLStorageModeShared) {
                const float *x_cpu = (const float *)((const uint8_t *)xbuf.contents +
                                                     (size_t)ds4_gpu_tensor_offset(x));
                ds4_gpu_expert_prefetch_enqueue_batch(layer_index + 1u, x_cpu,
                                                      expert_in_dim, n_tokens);
            }
            /* §2.4/§3.5 降激活: thin per-token picks before the union is built so the
             * gather (and the dominant per-token expert IO) shrinks.  selectedbuf is
             * CPU-valid here (drained above when batched; the non-batched path already
             * reads it via collect below).  Default OFF => byte-identical no-op. */
            ds4_gpu_moe_thin_picks(selectedbuf, selected_off, weightsbuf,
                                   ds4_gpu_tensor_offset(weights), n_tokens, n_expert,
                                   gate_type);
            ds4_gpu_moe_sim_keepmap(selectedbuf, selected_off, weightsbuf,
                                    ds4_gpu_tensor_offset(weights), n_tokens, n_expert, layer_index);
            const uint64_t total_picks_u64 = (uint64_t)n_tokens * n_expert;
            if (total_picks_u64 > UINT32_MAX) {
                if (was_batched) (void)ds4_gpu_begin_commands();
                return 0;
            }
            /* Pass 1 only: count the active set first so we can choose between
             * per-expert gather and P1.2 full-layer streaming before the slot
             * remap rewrites the selected-id buffer. */
            int compact_ok = ds4_gpu_collect_active_experts(selectedbuf,
                                                            selected_off,
                                                            (uint32_t)total_picks_u64,
                                                            n_total_expert,
                                                            active_ids,
                                                            1024,
                                                            &n_active);
            if (getenv("DS4_DUMP_ACTIVE")) {   /* raw picks (with repetition = frequency) per layer */
                const int32_t *sel = (const int32_t *)((uint8_t *)selectedbuf.contents + selected_off);
                fprintf(stderr, "PICKS L%u", layer_index);
                for (uint32_t i = 0; i < (uint32_t)total_picks_u64; i++) fprintf(stderr, " %d", sel[i]);
                fprintf(stderr, "\n");
            }
            /* Snapshot the ORIGINAL expert ids (selectedbuf is CPU-valid here: drained
             * above when batched, else already host-computed).  The remap below rewrites
             * selectedbuf to compact slots IN PLACE, so the go1b corr must read C[e]/beta[e]
             * by true id from this copy, not the corrupted live selected tensor. Covers
             * decode (router_selected) and prefill (batch_router_selected); both flow
             * through this one function, and each layer's corr consumes the snapshot
             * before the next layer's MoE overwrites it. */
            if (routed_is_go1b && total_picks_u64 > 0) {
                const uint64_t sbytes = total_picks_u64 * sizeof(int32_t);
                if (!g_corr_saved_selected || ds4_gpu_tensor_bytes(g_corr_saved_selected) < sbytes) {
                    ds4_gpu_tensor_free(g_corr_saved_selected);
                    g_corr_saved_selected = ds4_gpu_tensor_alloc(sbytes);
                }
                id<MTLBuffer> savedbuf = g_corr_saved_selected ? ds4_gpu_tensor_buffer(g_corr_saved_selected) : nil;
                if (savedbuf)
                    memcpy(savedbuf.contents,
                           (const uint8_t *)selectedbuf.contents + selected_off, (size_t)sbytes);
            }
            if (io_profile) ds4_gpu_expert_io_prof_reset();
            const double gather_t0 = io_profile ? ds4_gpu_now_ms() : 0.0;
            /* P-OVL eligibility: the expert-half overlap only applies to the
             * mm_id verify shape (batched, n_expert>1, n_tokens at/above the
             * grouped-GEMM threshold) with enough active experts to split.  When
             * it fires the byte gather is deferred to the per-pass encode below
             * and the stream/pool/source-cache paths are skipped. */
            moe_overlap_active =
                compact_ok &&
                !do_hot && !do_vq &&
                ds4_gpu_moe_overlap_enabled() &&
                was_batched &&
                !g_quality_mode &&
                n_expert > 1u &&
                n_tokens >= ds4_gpu_moe_mm_id_min() &&
                ds4_gpu_mul_mm_id_map0_name(n_expert) != NULL &&
                n_active >= ds4_gpu_moe_overlap_min_experts();
            /* P1.2: a big prefill chunk activates nearly every expert of the
             * layer, so ~256 scattered reads degenerate to random IO.  Stream
             * the whole gate/up/down tensors sequentially instead and keep the
             * original expert ids (selected buffer untouched, kernels index
             * the full-layer scratch exactly like the resident direct path). */
            int stream_used = 0;
            if (compact_ok && !moe_overlap_active && n_tokens > 1u && !g_expert_pool_warm_batch &&
                gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                down_type == DS4_METAL_TENSOR_Q2_K &&
                ds4_gpu_expert_stream_enabled() &&
                !ds4_gpu_expert_stream_rfetch_bypass() &&
                (uint64_t)n_active * 100ull >=
                    (uint64_t)n_total_expert * ds4_gpu_expert_stream_threshold_pct()) {
                stream_used = ds4_gpu_stream_layer_experts_to_scratch(model_map,
                                                                      gate_offset,
                                                                      up_offset,
                                                                      down_offset,
                                                                      gate_expert_bytes,
                                                                      down_expert_bytes,
                                                                      n_total_expert);
            }
            /* 1-bit residual (mm_id path) reuses the base's compacted work map, so it
             * only applies on the compacted gather path, not the full-layer stream. */
            if (do_residual && stream_used) do_residual = 0;
            int vq_remapped = 0;
            if (compact_ok && !stream_used) {
                compact_ok = ds4_gpu_remap_selected_to_slots(selectedbuf,
                                                             selected_off,
                                                             (uint32_t)total_picks_u64,
                                                             n_total_expert,
                                                             active_ids,
                                                             n_active);
                vq_remapped = 1;
            }
            if (getenv("DS4_VQ_DEBUG") && do_vq && layer_index < 1) {
                const int32_t *sc = (const int32_t*)((const uint8_t*)selectedbuf.contents + selected_off);
                fprintf(stderr, "[VQ_DBG3] L%u stream_used=%d remapped=%d compact_ok=%d n_active=%u sel[0..5]=%d,%d,%d,%d,%d,%d\n",
                        layer_index, stream_used, vq_remapped, compact_ok, n_active,
                        sc[0],sc[1],sc[2],sc[3],sc[4],sc[5]);
            }
            int pool_used = 0;
            if (!stream_used && !moe_overlap_active && compact_ok && g_expert_pool_warm_batch &&
                gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                down_type == DS4_METAL_TENSOR_Q2_K) {
                pool_used = ds4_gpu_try_load_layer_experts_to_pool(model_map,
                                                                   layer_index,
                                                                   selectedbuf,
                                                                   selected_off,
                                                                   (uint32_t)total_picks_u64,
                                                                   n_active,
                                                                   active_ids,
                                                                   gate_offset,
                                                                   up_offset,
                                                                   down_offset,
                                                                   gate_expert_bytes,
                                                                   down_expert_bytes,
                                                                   n_total_expert,
                                                                   &gate_buf,
                                                                   &up_buf,
                                                                   &down_buf,
                                                                   &source_n_total_expert);
            }
            /* source-cache note is just bookkeeping + (hard_copy) admit, not a
             * gather: run it for the overlap path too so the per-pass gather
             * below hits the RAM hard copies via hard_find. */
            if (!stream_used && compact_ok && !pool_used &&
                gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                down_type == DS4_METAL_TENSOR_Q2_K) {
                ds4_gpu_expert_source_cache_note(model_map,
                                                 layer_index,
                                                 n_active,
                                                 active_ids,
                                                 gate_offset,
                                                 up_offset,
                                                 down_offset,
                                                 gate_expert_bytes,
                                                 down_expert_bytes);
            }
            const double copy_t0 = (!pool_used && !stream_used && ds4_gpu_expert_profile_is_enabled()) ?
                ds4_gpu_now_ms() : 0.0;
            int load_ok;
            if (do_vq) {
                load_ok = compact_ok && ds4_gpu_vq_unified_gather(model_map,
                              (const uint8_t *)go1b_res->gate_ptr,
                              n_active, active_ids,
                              down_offset, down_expert_bytes,
                              expert_in_dim, expert_mid_dim, out_dim);
            } else if (do_hot) {
                /* R5-C unified go2b: every active expert lands in the go2b scratch
                 * (hot from the merged sidecar, cold fabricated with d2=0), so the
                 * MoE below keeps bare's single-map three-tile shape. */
                load_ok = compact_ok && ds4_gpu_hot_unified_gather(model_map,
                              n_active, active_ids, res_lut,
                              res_gate_ptr, res_up_ptr, res_down_ptr,
                              gate_offset, up_offset, down_offset,
                              gate_expert_bytes, down_expert_bytes,
                              expert_in_dim, expert_mid_dim, out_dim);
            } else {
                load_ok = compact_ok && (stream_used || pool_used || moe_overlap_active ||
                          ds4_gpu_load_layer_experts_to_scratch(model_map,
                                                                 layer_index,
                                                                 n_active,
                                                                 active_ids,
                                                                 gate_offset,
                                                                 up_offset,
                                                                 down_offset,
                                                                 gate_expert_bytes,
                                                                 down_expert_bytes,
                                                                 n_total_expert));
            }
            /* P-OVL defers the byte gather to the per-pass encode: size the
             * shared scratch now so the per-pass range gathers (slots [lo,hi))
             * have valid destination addresses. */
            if (load_ok && moe_overlap_active &&
                !ds4_gpu_ensure_moe_scratch(n_active, gate_expert_bytes, down_expert_bytes)) {
                load_ok = 0;
            }
            if (io_profile) {
                ds4_gpu_expert_io_prof_report("batch",
                                              stream_used ? "stream" :
                                              (pool_used ? "pool" : "gather"),
                                              layer_index,
                                              n_active,
                                              n_tokens,
                                              ds4_gpu_now_ms() - gather_t0,
                                              drain_ms);
            }
            if (!pool_used && !stream_used && compact_ok && ds4_gpu_expert_profile_is_enabled()) {
                ds4_gpu_expert_profile_record(layer_index,
                                              active_ids,
                                              n_active,
                                              n_total_expert,
                                              gate_expert_bytes,
                                              down_expert_bytes,
                                              (uint32_t)total_picks_u64,
                                              ds4_gpu_now_ms() - copy_t0);
            }
            if (!load_ok) {
                if (was_batched) (void)ds4_gpu_begin_commands();
                return 0;
            }
            if (do_vq) {
                /* CPU MoE(匹配量化脚本 dq_expert_fp)。DS4_VQ_DIAG=1: 写 ref 并继续跑 GPU F16W 末尾对比。 */
                int vqdiag = getenv("DS4_VQ_DIAG") != NULL;
                const char *vg = getenv("DS4_VQ_GPU");
                int vqgpu  = vqdiag || !(vg && vg[0] == '0');   /* 默认 GPU F16W(2026-08-09 A/B: decode 3.2×/prefill 6.4×, 贪心输出与 CPU 参考路逐字一致); DS4_VQ_GPU=0 回 CPU 参考路 */
                if (!vqgpu || vqdiag) {   /* 纯GPU 跳过 CPU; 生产/DIAG 都要算 CPU(生产=输出, DIAG=参考) */
                if (was_batched) (void)ds4_gpu_end_commands();
                const float *xin = (const float*)((const uint8_t*)xbuf.contents + ds4_gpu_tensor_offset(x));
                const int32_t *sel = (const int32_t*)((const uint8_t*)selectedbuf.contents + selected_off);
                const float *rw = (const float*)((const uint8_t*)weightsbuf.contents + ds4_gpu_tensor_offset(weights));
                float *outp = vqdiag ? (float*)malloc((size_t)n_tokens*out_dim*sizeof(float))
                                     : (float*)((uint8_t*)outbuf.contents + ds4_gpu_tensor_offset(out));
                /* f16 scratch(内存安全) → 每专家转 f32 temp + cblas_sgemv(向量化); temp reuse */
                const uint16_t *gsc=(const uint16_t*)g_moe_vq_gate_scratch.contents;
                const uint16_t *usc=(const uint16_t*)g_moe_vq_up_scratch.contents;
                const uint16_t *dnc=(const uint16_t*)g_moe_vq_down_scratch.contents;
                const int IN=(int)expert_in_dim, MID=(int)expert_mid_dim, OUT=(int)out_dim;
                const float L=clamp;
                /* per-slot f32 权重缓存(该层各专家转一次, reuse 跨 token): n_active×(2·MID·IN+OUT·MID) f32
                 * 但内存受限 → 仅缓存"是否已转"标记 + 单专家 temp(32MB×3 reuse)。按专家外层循环批处理 token。 */
                float *w1f=(float*)malloc((size_t)MID*IN*sizeof(float));
                float *w3f=(float*)malloc((size_t)MID*IN*sizeof(float));
                float *w2f=(float*)malloc((size_t)OUT*MID*sizeof(float));
                float *gg=(float*)malloc((size_t)MID*sizeof(float));
                float *hb=(float*)malloc((size_t)MID*sizeof(float));
                for (uint32_t t=0;t<n_tokens;t++){ float *o=outp+(uint64_t)t*OUT; for(int d=0;d<OUT;d++)o[d]=0.0f; }
                for (uint32_t sl=0; sl<n_active; sl++){
                    /* 该 slot 是否被任何 token 选中 */
                    int used=0; for(uint32_t t=0;t<n_tokens&&!used;t++) for(uint32_t pk=0;pk<n_expert;pk++) if(sel[(uint64_t)t*n_expert+pk]==(int)sl){used=1;break;}
                    if(!used) continue;
                    const uint16_t *g=gsc+(uint64_t)sl*MID*IN,*u=usc+(uint64_t)sl*MID*IN,*dn=dnc+(uint64_t)sl*OUT*MID;
                    for(size_t j=0;j<(size_t)MID*IN;j++){ w1f[j]=ds4vq_f16(g[j]); w3f[j]=ds4vq_f16(u[j]); }
                    for(size_t j=0;j<(size_t)OUT*MID;j++) w2f[j]=ds4vq_f16(dn[j]);
                    for (uint32_t t=0;t<n_tokens;t++){
                        float w=0; for(uint32_t pk=0;pk<n_expert;pk++) if(sel[(uint64_t)t*n_expert+pk]==(int)sl){ w=rw[(uint64_t)t*n_expert+pk]; break; }
                        if(w==0.0f) continue;
                        const float *xt=xin+(uint64_t)t*IN; float *o=outp+(uint64_t)t*OUT;
                        cblas_sgemv(CblasRowMajor,CblasNoTrans,MID,IN,1.0f,w1f,IN,xt,1,0.0f,gg,1);
                        cblas_sgemv(CblasRowMajor,CblasNoTrans,MID,IN,1.0f,w3f,IN,xt,1,0.0f,hb,1);
                        for(int m=0;m<MID;m++){ float ga=gg[m],ua=hb[m];
                            if(L>0){ if(ga>L)ga=L; if(ua>L)ua=L; if(ua<-L)ua=-L; }
                            hb[m]=(ga/(1.0f+expf(-ga)))*ua; }
                        cblas_sgemv(CblasRowMajor,CblasNoTrans,OUT,MID,w,w2f,MID,hb,1,1.0f,o,1);
                    }
                }
                free(w1f);free(w3f);free(w2f);free(gg);free(hb);
                if (was_batched) (void)ds4_gpu_begin_commands();
                if (!vqgpu) return 1;   /* 生产: CPU 输出已写 outbuf */
                /* DIAG: 存 CPU 参考, GPU 跑完在末尾对比 */
                g_vq_diag_ref = outp; g_vq_diag_layer = layer_index; g_vq_diag_ntok = n_tokens; g_vq_diag_out = out_dim;
                }   /* 结束 (!vqgpu || vqdiag) 的 CPU 块 */
                /* GPU F16W: 设 VQ f16 scratch, fall through 到下方 plain mm_id passes(map→gate→up→swiglu→down→sum) */
                gate_buf = g_moe_vq_gate_scratch; up_buf = g_moe_vq_up_scratch;
                down_buf = g_moe_vq_down_scratch; source_n_total_expert = n_active;
            } else if (do_hot) {
                gate_buf = g_moe_hot_gate_scratch;
                up_buf = g_moe_hot_up_scratch;
                down_buf = g_moe_hot_down_scratch;
                source_n_total_expert = n_active;
            } else if (!pool_used) {
                gate_buf = g_moe_scratch_gate;
                up_buf = g_moe_scratch_up;
                down_buf = g_moe_scratch_down;
                source_n_total_expert = stream_used ? n_total_expert : n_active;
            }
            gate_inner = 0;
            up_inner = 0;
            down_inner = 0;
            if (was_batched && ds4_gpu_begin_commands() == 0) return 0;
        }

        const uint32_t pair_rows = n_tokens * n_expert;
        const uint64_t down_scratch_bytes = (uint64_t)pair_rows * out_dim * sizeof(float);
        if ((n_expert > 1 && !expertsbuf &&
             !ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                              &g_moe_down_scratch_bytes,
                                              (NSUInteger)down_scratch_bytes,
                                              "ds4_moe_down_scratch"))) {
            return 0;
        }

        const uint32_t gate_nr0 = ds4_gpu_routed_mv_nr0(gate_type);
        const uint32_t down_nr0 = ds4_gpu_routed_mv_nr0(down_type);
        id<MTLComputePipelineState> gate_mv_pipeline = ds4_gpu_routed_mv_pipeline(gate_type);
        id<MTLComputePipelineState> down_mv_pipeline = ds4_gpu_routed_mv_pipeline(down_type);
        id<MTLComputePipelineState> gate_mm_pipeline = nil;
        id<MTLComputePipelineState> up_mm_pipeline = nil;
        id<MTLComputePipelineState> down_mm_pipeline = nil;
        /* Strict-1-bit (go1b) routed experts ship only the grouped mm_id matmul kernel
         * -- there is no hand-written mul_mv_id / pair / sum6 variant.  Detect "mm-only"
         * quants by the absent mv pipeline and force the mm_id path for every batch size
         * (the grouped GEMM is a correct general matmul at n_tokens=1, so decode -- routed
         * here with n_tokens=1 -- works too). */
        const bool force_mm = (gate_mv_pipeline == nil) || (down_mv_pipeline == nil);
        if (force_mm) {
            if (ds4_gpu_mul_mm_id_map0_name(n_expert) == NULL) {
                fprintf(stderr, "ds4: mm-only routed MoE (gate=%u down=%u) has no mm_id map for n_expert=%u\n",
                        gate_type, down_type, n_expert);
                return 0;
            }
        } else if (gate_nr0 == 0 || down_nr0 == 0 || !gate_mv_pipeline || !down_mv_pipeline) {
            fprintf(stderr, "ds4: unsupported Metal routed batch MoE quant types gate=%u down=%u\n",
                    gate_type, down_type);
            return 0;
        }

        ds4_gpu_mul_mv_id_args gate_args =
            ds4_gpu_make_mul_mv_id_args(expert_in_dim, expert_mid_dim, source_n_total_expert,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, n_expert, n_tokens, gate_nr0);
        ds4_gpu_mul_mv_id_args down_args =
            ds4_gpu_make_mul_mv_id_args(expert_mid_dim, out_dim, source_n_total_expert,
                                          down_row_bytes, down_expert_bytes,
                                          n_expert, n_expert, n_tokens, down_nr0);
        /*
         * Grouped-GEMM threshold. The mm_id path maps token-rows per expert
         * and reads each active expert's weights exactly ONCE (expert-major),
         * while the mv/pair paths re-read the expert weights once per
         * (expert,token) pair: a 12-token verify batch re-reads ~486MiB of
         * scratch instead of the ~270MiB union, through matvec-shaped kernels
         * with much lower effective bandwidth. The old >=32 threshold was
         * tuned when the only small batches were MTP's 2-token suffixes; PC.1
         * copy-speculation batches are 5..12 tokens and sit right in the slow
         * gap. Default lowered to 8 (A/B: DS4_METAL_MOE_MM_ID_MIN=32 restores
         * the old split, =0 forces never).
         */
        const bool use_mm_id = (force_mm || n_tokens >= ds4_gpu_moe_mm_id_min()) &&
                               ds4_gpu_mul_mm_id_map0_name(n_expert) != NULL;
        /*
         * Speculative verification is neither normal decode nor large prefill:
         * the target model must verify a small suffix in one layer-major pass.
         * For that shape the prefill expert-major GEMM path is too large, but
         * the decode pair kernels are exactly the right primitive: they read
         * the same activation once and compute routed gate/up together for
         * every selected expert row, and their dispatch grid is
         * (n_expert x n_tokens) pairs - shape generic.
         *
         * PC.1 copy speculation widened the verify batches from MTP's 2 to
         * 5..12 tokens, which used to fall through to the generic split-mv
         * path: measured (2026-06-10 code-edit run) drain 100-256 ms/layer for
         * a 12-token batch = ~5-10x worse per token than the >=32-token
         * grouped GEMM (1.3 ms/token/layer). Route everything below the GEMM
         * threshold through the pair kernels instead; cap adjustable for A/B
         * (DS4_METAL_MOE_TINY_PAIR_MAX=4 restores the old behavior).
         */
        static uint32_t tiny_pair_max_cached;
        static int tiny_pair_max_init;
        if (!tiny_pair_max_init) {
            uint64_t v = ds4_gpu_env_u64("DS4_METAL_MOE_TINY_PAIR_MAX", 16u);
            if (v > 31u) v = 31u;
            tiny_pair_max_cached = (uint32_t)v;
            tiny_pair_max_init = 1;
        }
        const bool use_tiny_pair_mv =
            !g_quality_mode &&
            n_tokens <= tiny_pair_max_cached &&
            !use_mm_id &&
            ((gate_type == DS4_METAL_TENSOR_IQ2_XXS && g_moe_mul_mv_id_iq2_xxs_pair_pipeline) ||
             (gate_type == DS4_METAL_TENSOR_Q4_K && g_moe_mul_mv_id_q4_k_pair_pipeline));
        ds4_gpu_mul_mm_id_map_args gate_map_args = { 0 };
        ds4_gpu_mul_mm_id_args gate_mm_args = { 0 };
        ds4_gpu_mul_mm_id_args down_mm_args = { 0 };
        id<MTLComputePipelineState> map_pipeline = nil;
        /*
         * The grouped routed-MoE matmul loads activation tiles as half before
         * using SIMD-group MMA.  Store the SwiGLU/route-weight intermediate in
         * that same precision so the down projection avoids a large F32 mid
         * write/read. --quality keeps the older F32 intermediate.
         */
        const bool request_mid_f16 = !g_quality_mode;
        if (use_mm_id) {
            gate_map_args =
                ds4_gpu_make_mul_mm_id_map_args(expert_in_dim, source_n_total_expert, 1, n_expert, n_tokens);
            /* R5-C unified go2b: the scratch holds 68-byte blocks for every
             * active expert, so strides and kernels switch to go2b wholesale. */
            uint64_t eff_gate_row = do_hot ? (uint64_t)expert_in_dim / 256u * 68u : gate_row_bytes;
            uint64_t eff_gate_exp = do_hot ? (uint64_t)expert_mid_dim * eff_gate_row : gate_expert_bytes;
            uint64_t eff_down_row = do_hot ? (uint64_t)expert_mid_dim / 256u * 68u : down_row_bytes;
            uint64_t eff_down_exp = do_hot ? (uint64_t)out_dim * eff_down_row : down_expert_bytes;
            uint32_t eff_gate_type = do_hot ? DS4_METAL_TENSOR_GO2B : gate_type;
            uint32_t eff_down_type = do_hot ? DS4_METAL_TENSOR_GO2B : down_type;
            if (do_vq) {   /* f16 scratch: 行=in*2B, 专家=mid*行 */
                eff_gate_row = (uint64_t)expert_in_dim * 2u;
                eff_gate_exp = (uint64_t)expert_mid_dim * eff_gate_row;
                eff_down_row = (uint64_t)expert_mid_dim * 2u;
                eff_down_exp = (uint64_t)out_dim * eff_down_row;
                eff_gate_type = DS4_METAL_TENSOR_F16W;
                eff_down_type = DS4_METAL_TENSOR_F16W;
            }
            gate_mm_args =
                ds4_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, source_n_total_expert,
                                              eff_gate_row, eff_gate_exp,
                                              1, n_expert, n_tokens);
            down_mm_args =
                ds4_gpu_make_mul_mm_id_args_src1_size(expert_mid_dim, out_dim, source_n_total_expert,
                                                        eff_down_row, eff_down_exp,
                                                        n_expert, n_expert, n_tokens,
                                                        request_mid_f16 ? sizeof(uint16_t) : sizeof(float));

            map_pipeline = ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_expert));
            gate_mm_pipeline = ds4_gpu_routed_mm_pipeline(eff_gate_type);
            up_mm_pipeline = ds4_gpu_routed_mm_pipeline(eff_gate_type);
            down_mm_pipeline = request_mid_f16 ?
                ds4_gpu_routed_mm_f16_rhs_pipeline(eff_down_type) :
                ds4_gpu_routed_mm_pipeline(eff_down_type);
            if (do_vq && getenv("DS4_VQ_DEBUG"))
                fprintf(stderr, "[VQ_DBG4] map=%d gate=%d up=%d down=%d eff_gt=%u eff_dt=%u mid_f16=%d\n",
                        map_pipeline!=nil, gate_mm_pipeline!=nil, up_mm_pipeline!=nil,
                        down_mm_pipeline!=nil, eff_gate_type, eff_down_type, request_mid_f16);
            if (!map_pipeline || !gate_mm_pipeline || !up_mm_pipeline || !down_mm_pipeline) {
                if (do_vq && getenv("DS4_VQ_DEBUG")) fprintf(stderr, "[VQ_DBG4] ★pipeline nil → return 0★\n");
                return 0;
            }
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        const bool moe_stage_profile =
            getenv("DS4_METAL_MOE_STAGE_PROFILE") != NULL && g_batch_cb != nil;
        const char *moe_stage_filter = getenv("DS4_METAL_MOE_STAGE_PROFILE_FILTER");
        const char *moe_path =
            use_mm_id ? "mm_id" :
            (use_tiny_pair_mv ? "tiny_pair_mv" : "mv");
        double moe_stage_t0 = moe_stage_profile ? ds4_gpu_now_ms() : 0.0;
        if (moe_stage_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            moe_stage_t0 = ds4_gpu_now_ms();
        }
#define DS4_METAL_PROFILE_MOE_STAGE(name) do { \
            if (ok && moe_stage_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = ds4_gpu_now_ms(); \
                    const int print_stage = \
                        !moe_stage_filter || !moe_stage_filter[0] || \
                        strstr(stage_name, moe_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "ds4: Metal routed MoE stage layer=%u tokens=%u pairs=%u experts=%u " \
                                "gate=%s down=%s path=%s mid=%s %s=%.3f ms\n", \
                                layer_index, n_tokens, pair_rows, n_expert, \
                                ds4_gpu_metal_tensor_type_name(gate_type), \
                                ds4_gpu_metal_tensor_type_name(down_type), \
                                moe_path, \
                                request_mid_f16 ? "f16" : "f32", \
                                stage_name, now_ms - moe_stage_t0); \
                    } \
                    moe_stage_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        const NSUInteger gate_smem = ds4_gpu_routed_mv_smem(gate_type);
        const NSUInteger down_smem = ds4_gpu_routed_mv_smem(down_type);
        id<MTLComputePipelineState> down_sum6_pipeline = nil;
        if (down_type == DS4_METAL_TENSOR_Q2_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q2_k_sum6_pipeline;
        } else if (down_type == DS4_METAL_TENSOR_Q4_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q4_k_sum6_pipeline;
        }
        const bool direct_down_sum =
            !g_quality_mode &&
            !use_mm_id &&
            n_expert == 6 &&
            n_tokens <= 4u &&
            down_sum6_pipeline != nil;
        int ok = 0;
        if (moe_overlap_active) {
            /*
             * P-OVL (bit-exact expert N-way overlap).  Partition the active
             * expert set into N disjoint scratch slot-ranges.  Gather + encode
             * pass p, COMMIT it so the GPU starts, then gather pass p+1 on the
             * CPU while the GPU runs pass p -- the routed-expert disk gather
             * (the decode bottleneck) overlaps the GEMM (total ~= T_gather +
             * T_gpu/N).  The mm_id slot-range gate guarantees pass p's GEMM
             * never reads outside [lo,hi), so the disjoint CPU writes during the
             * next pass's gather do not alias the in-flight reads.  Each pass
             * writes only its own picks' rows in the per-pair expert output
             * buffer; the single sum_experts (fixed 6-pick order) makes the
             * result bit-identical to the single-pass path.  swiglu runs over
             * all pair_rows each pass: the other passes' rows are garbage but
             * unread by this pass's slot-gated down GEMM, and are recomputed
             * identically before being consumed.
             */
            const bool ov_mid_f16 = use_mm_id && !g_quality_mode && request_mid_f16;
            if (mid_is_f16) *mid_is_f16 = ov_mid_f16;
            id<MTLBuffer> ov_down_dst = expertsbuf;
            const NSUInteger ov_down_dst_off = ds4_gpu_tensor_offset(experts);
            uint32_t ov_passes = ds4_gpu_moe_overlap_passes();
            if (ov_passes > n_active) ov_passes = n_active;   /* >= 1 expert per pass */
            if (ov_passes < 1u) ov_passes = 1u;
            ok = 1;
            for (uint32_t pass = 0; pass < ov_passes && ok; pass++) {
                const uint32_t lo = (uint32_t)((uint64_t)pass * n_active / ov_passes);
                const uint32_t hi = (uint32_t)((uint64_t)(pass + 1u) * n_active / ov_passes);
                if (lo >= hi) continue;
                /* CPU/disk gather of this pass's experts into scratch slots
                 * [lo,hi).  For passes after the first this runs while the GPU
                 * executes the committed previous pass; the slot-range gate
                 * keeps the regions disjoint so there is no read/write alias. */
                if (!ds4_gpu_gather_experts_run(model_map, layer_index, hi - lo,
                                                active_ids + lo,
                                                (uint8_t *)g_moe_scratch_gate.contents + (uint64_t)lo * gate_expert_bytes,
                                                (uint8_t *)g_moe_scratch_up.contents + (uint64_t)lo * gate_expert_bytes,
                                                (uint8_t *)g_moe_scratch_down.contents + (uint64_t)lo * down_expert_bytes,
                                                gate_offset, up_offset, down_offset,
                                                gate_expert_bytes, down_expert_bytes,
                                                n_total_expert)) {
                    ok = 0;
                    break;
                }
                gate_map_args.slot_lo = (int32_t)lo;
                gate_map_args.slot_hi = (int32_t)hi;
                ok = ds4_gpu_encode_mul_mm_id_map(cb, map_pipeline, &gate_map_args,
                                                  &gate_mm_args, selectedbuf,
                                                  ds4_gpu_tensor_offset(selected));
                if (ok) ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb, gate_mm_pipeline,
                                &gate_mm_args, gate_buf, (NSUInteger)gate_inner,
                                xbuf, ds4_gpu_tensor_offset(x), gatebuf,
                                ds4_gpu_tensor_offset(gate));
                if (ok) ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb, up_mm_pipeline,
                                &gate_mm_args, up_buf, (NSUInteger)up_inner,
                                xbuf, ds4_gpu_tensor_offset(x), upbuf,
                                ds4_gpu_tensor_offset(up));
                if (ok) ok = ds4_gpu_encode_moe_swiglu_weight(cb, gatebuf,
                                ds4_gpu_tensor_offset(gate), upbuf,
                                ds4_gpu_tensor_offset(up), midbuf,
                                ds4_gpu_tensor_offset(mid), weightsbuf,
                                ds4_gpu_tensor_offset(weights), expert_mid_dim,
                                pair_rows, clamp, ov_mid_f16);
                if (ok) ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb, down_mm_pipeline,
                                &down_mm_args, down_buf, (NSUInteger)down_inner,
                                midbuf, ds4_gpu_tensor_offset(mid), ov_down_dst,
                                ov_down_dst_off);
                if (ok && pass + 1u < ov_passes) {
                    /* Commit this pass (async; GPU starts) and reopen the batch
                     * so the next pass's CPU gather overlaps the running GPU. */
                    if (!ds4_gpu_flush_commands()) { ok = 0; break; }
                    cb = ds4_gpu_command_buffer(&owned);
                    if (!cb) { ok = 0; break; }
                }
            }
            if (ok) {
                ok = ds4_gpu_encode_moe_sum_experts(cb, ov_down_dst, ov_down_dst_off,
                                                    outbuf, ds4_gpu_tensor_offset(out),
                                                    out_dim, n_expert, n_tokens);
            }
            if (!ok) return 0;
            if (!ds4_gpu_finish_command_buffer(cb, owned, "routed batch MoE")) return 0;
            return 1;
        }
        if (use_mm_id) {
            /*
             * The routed pair ids are the same for gate, up, and down. Build
             * the expert-major work map once, then reuse it for all three
             * batched expert matmuls.
             */
            ok = ds4_gpu_encode_mul_mm_id_map(cb,
                                                map_pipeline,
                                                &gate_map_args,
                                                &gate_mm_args,
                                                selectedbuf,
                                                ds4_gpu_tensor_offset(selected));
            DS4_METAL_PROFILE_MOE_STAGE("map");
            if (ok) {
                ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                           gate_mm_pipeline,
                                                           &gate_mm_args,
                                                           gate_buf,
                                                           (NSUInteger)gate_inner,
                                                           xbuf,
                                                           ds4_gpu_tensor_offset(x),
                                                           gatebuf,
                                                           ds4_gpu_tensor_offset(gate));
                DS4_METAL_PROFILE_MOE_STAGE("gate");
            }
            if (ok) {
                ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                   up_mm_pipeline,
                                                   &gate_mm_args,
                                                   up_buf,
                                                   (NSUInteger)up_inner,
                                                   xbuf,
                                                   ds4_gpu_tensor_offset(x),
                                                   upbuf,
                                                   ds4_gpu_tensor_offset(up));
                DS4_METAL_PROFILE_MOE_STAGE("up");
            }
            if (ok && do_hot && n_hot_picks > 0) {
                /* R5-C hot pass (gate/up): gather go2b hot experts, build the hot
                 * work map over the SAME shared map buffer (base tiles already
                 * encoded; same-CB dispatches execute in order), run one go2b
                 * matmul per matrix writing the rows the base pass left unclaimed. */
                const uint64_t g2_gate_row = (uint64_t)expert_in_dim / 256u * 68u;
                const uint64_t g2_gate_exp = (uint64_t)expert_mid_dim * g2_gate_row;
                const uint64_t g2_down_row = (uint64_t)expert_mid_dim / 256u * 68u;
                const uint64_t g2_down_exp = (uint64_t)out_dim * g2_down_row;
                const uint64_t need_g = (uint64_t)n_hot_slots * g2_gate_exp;
                const uint64_t need_d = (uint64_t)n_hot_slots * g2_down_exp;
                if (!g_moe_hot_gate_scratch || (uint64_t)g_moe_hot_gate_scratch.length < need_g) {
                    g_moe_hot_gate_scratch = [g_device newBufferWithLength:(NSUInteger)need_g options:MTLResourceStorageModeShared];
                    g_moe_hot_up_scratch   = [g_device newBufferWithLength:(NSUInteger)need_g options:MTLResourceStorageModeShared];
                }
                if (!g_moe_hot_down_scratch || (uint64_t)g_moe_hot_down_scratch.length < need_d)
                    g_moe_hot_down_scratch = [g_device newBufferWithLength:(NSUInteger)need_d options:MTLResourceStorageModeShared];
                const uint64_t hot_picks_total = (uint64_t)n_tokens * n_expert;
                const uint64_t selbytes = hot_picks_total * sizeof(int32_t);
                if (!g_moe_hot_sel || (uint64_t)g_moe_hot_sel.length < selbytes)
                    g_moe_hot_sel = [g_device newBufferWithLength:(NSUInteger)selbytes options:MTLResourceStorageModeShared];
                int hot_ok = (g_moe_hot_gate_scratch && g_moe_hot_up_scratch &&
                              g_moe_hot_down_scratch && g_moe_hot_sel) ? 1 : 0;
                if (hot_ok) {
                    int32_t *hs = (int32_t *)g_moe_hot_sel.contents;
                    for (uint32_t i = 0; i < (uint32_t)hot_picks_total; i++)
                        hs[i] = (g_hot_pick_slot[i] >= 0) ? g_hot_pick_slot[i] : 0xFFFF;
                    for (uint32_t j = 0; j < n_hot_slots; j++) {
                        const uint64_t sslot = hot_slot_ids[j];
                        memcpy((uint8_t *)g_moe_hot_gate_scratch.contents + (uint64_t)j * g2_gate_exp,
                               (const uint8_t *)res_gate_ptr + sslot * g2_gate_exp, g2_gate_exp);
                        memcpy((uint8_t *)g_moe_hot_up_scratch.contents + (uint64_t)j * g2_gate_exp,
                               (const uint8_t *)res_up_ptr + sslot * g2_gate_exp, g2_gate_exp);
                        memcpy((uint8_t *)g_moe_hot_down_scratch.contents + (uint64_t)j * g2_down_exp,
                               (const uint8_t *)res_down_ptr + sslot * g2_down_exp, g2_down_exp);
                    }
                }
                id<MTLComputePipelineState> hot_gate_pipe = ds4_gpu_routed_mm_pipeline(DS4_METAL_TENSOR_GO2B);
                if (hot_ok && hot_gate_pipe) {
                    ds4_gpu_mul_mm_id_map_args hot_map_args =
                        ds4_gpu_make_mul_mm_id_map_args(expert_in_dim, n_hot_slots, 1, n_expert, n_tokens);
                    ds4_gpu_mul_mm_id_args hot_gate_args =
                        ds4_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, n_hot_slots,
                                                      g2_gate_row, g2_gate_exp,
                                                      1, n_expert, n_tokens);
                    ok = ds4_gpu_encode_mul_mm_id_map(cb, map_pipeline, &hot_map_args,
                             &hot_gate_args, g_moe_hot_sel, 0)
                      && ds4_gpu_encode_mul_mm_id_mapped_tile(cb, hot_gate_pipe, &hot_gate_args,
                             g_moe_hot_gate_scratch, 0, xbuf, ds4_gpu_tensor_offset(x),
                             gatebuf, ds4_gpu_tensor_offset(gate))
                      && ds4_gpu_encode_mul_mm_id_mapped_tile(cb, hot_gate_pipe, &hot_gate_args,
                             g_moe_hot_up_scratch, 0, xbuf, ds4_gpu_tensor_offset(x),
                             upbuf, ds4_gpu_tensor_offset(up));
                    DS4_METAL_PROFILE_MOE_STAGE("hot-gate-up");
                } else if (do_hot) {
                    ok = 0;   /* split active but resources missing: fail loud, no silent quality fork */
                }
            }
            if (ok && do_residual && ds4_gpu_ensure_res_scratch(n_active, gate_expert_bytes)) {
                /* CPU-gather the active residual experts into the compacted scratch
                 * (matching the base offload layout), then run the go1b mm_id matmul
                 * with the base's work map and sum into gate/up before swiglu. */
                for (uint32_t i = 0; i < n_active; i++) {
                    uint32_t e = active_ids[i];
                    int slot = res_lut ? (int)res_lut[e] : (int)e;   /* sparse: -1 => no residual */
                    uint8_t *gd = (uint8_t *)g_moe_res_gate_scratch.contents + (uint64_t)i * gate_expert_bytes;
                    uint8_t *ud = (uint8_t *)g_moe_res_up_scratch.contents + (uint64_t)i * gate_expert_bytes;
                    if (slot >= 0) {
                        memcpy(gd, (const uint8_t *)res_gate_ptr + (uint64_t)slot * gate_expert_bytes, gate_expert_bytes);
                        memcpy(ud, (const uint8_t *)res_up_ptr + (uint64_t)slot * gate_expert_bytes, gate_expert_bytes);
                    } else {   /* zero go1b block (d=0) dequantizes to 0 => no correction */
                        memset(gd, 0, gate_expert_bytes);
                        memset(ud, 0, gate_expert_bytes);
                    }
                }
                const uint32_t go_n = (uint32_t)n_tokens * n_expert * expert_mid_dim;
                ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb, gate_mm_pipeline, &gate_mm_args,
                         g_moe_res_gate_scratch, 0, xbuf, ds4_gpu_tensor_offset(x),
                         g_moe_go1b_res_scratch, 0)
                  && ds4_gpu_encode_add_f32_1d(cb, gatebuf, ds4_gpu_tensor_offset(gate),
                         g_moe_go1b_res_scratch, 0, gatebuf, ds4_gpu_tensor_offset(gate), go_n)
                  && ds4_gpu_encode_mul_mm_id_mapped_tile(cb, up_mm_pipeline, &gate_mm_args,
                         g_moe_res_up_scratch, 0, xbuf, ds4_gpu_tensor_offset(x),
                         g_moe_go1b_res_scratch, 0)
                  && ds4_gpu_encode_add_f32_1d(cb, upbuf, ds4_gpu_tensor_offset(up),
                         g_moe_go1b_res_scratch, 0, upbuf, ds4_gpu_tensor_offset(up), go_n);
                if (getenv("DS4_RESIDUAL_DEBUG")) {
                    const uint8_t *g0 = (const uint8_t*)g_moe_res_gate_scratch.contents;
                    const uint8_t *rp = (const uint8_t*)res_gate_ptr;
                    unsigned long gsum = 0, rsum = 0;
                    for (int b = 0; b < 512; b++) { gsum += g0[b]; rsum += rp[b]; }
                    fprintf(stderr, "ds4: [res-mm-apply] L%u n_active=%u go_n=%u ok=%d gathered512=%lu src512=%lu eb=%llu\n",
                            layer_index, n_active, go_n, ok, gsum, rsum,
                            (unsigned long long)gate_expert_bytes);
                }
            }
        } else if (use_tiny_pair_mv) {
            id<MTLComputePipelineState> pair_pipeline =
                gate_type == DS4_METAL_TENSOR_IQ2_XXS ?
                    g_moe_mul_mv_id_iq2_xxs_pair_pipeline :
                    g_moe_mul_mv_id_q4_k_pair_pipeline;
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 (NSUInteger)gate_inner,
                                                 up_buf,
                                                 (NSUInteger)up_inner,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                                  gate_mv_pipeline,
                                                  &gate_args,
                                                  gate_buf,
                                                  (NSUInteger)gate_inner,
                                                  xbuf,
                                                  ds4_gpu_tensor_offset(x),
                                                  gatebuf,
                                                  ds4_gpu_tensor_offset(gate),
                                                  selectedbuf,
                                                  ds4_gpu_tensor_offset(selected),
                                                  gate_smem,
                                                  2,
                                                  false) &&
                 ds4_gpu_encode_mul_mv_id(cb,
                                                  gate_mv_pipeline,
                                                  &gate_args,
                                                  up_buf,
                                                  (NSUInteger)up_inner,
                                                  xbuf,
                                                  ds4_gpu_tensor_offset(x),
                                                  upbuf,
                                                  ds4_gpu_tensor_offset(up),
                                                  selectedbuf,
                                                  ds4_gpu_tensor_offset(selected),
                                                  gate_smem,
                                                  2,
                                                  false);
        }
        /* 1-bit residual for decode is applied inside the use_mm_id branch above
         * (go1b has no mv_id kernel, so decode always takes the mm_id path). */
        DS4_METAL_PROFILE_MOE_STAGE("gate_up");
        const bool use_fused_activation = !g_quality_mode;
        const bool use_mid_f16 =
            use_mm_id &&
            use_fused_activation &&
            request_mid_f16;
        if (mid_is_f16) *mid_is_f16 = use_mid_f16;
        if (ok && use_fused_activation) {
            ok = ds4_gpu_encode_moe_swiglu_weight(cb,
                                                    gatebuf,
                                                    ds4_gpu_tensor_offset(gate),
                                                    upbuf,
                                                    ds4_gpu_tensor_offset(up),
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    weightsbuf,
                                                    ds4_gpu_tensor_offset(weights),
                                                    expert_mid_dim,
                                                    pair_rows,
                                                    clamp,
                                                    use_mid_f16);
        } else if (ok && clamp > 1.0e-6f) {
            ok = ds4_gpu_encode_unary_f32_rows(cb,
                                                 g_unary_clamp_pipeline,
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 expert_mid_dim,
                                                 pair_rows,
                                                 0,
                                                 -FLT_MAX,
                                                 clamp);
            if (ok) {
                ok = ds4_gpu_encode_unary_f32_rows(cb,
                                                     g_unary_silu_pipeline,
                                                     gatebuf,
                                                     ds4_gpu_tensor_offset(gate),
                                                     midbuf,
                                                     ds4_gpu_tensor_offset(mid),
                                                     expert_mid_dim,
                                                     pair_rows,
                                                     1,
                                                     0.0f,
                                                     0.0f);
            }
            if (ok) {
                ok = ds4_gpu_encode_unary_f32_rows(cb,
                                                 g_unary_clamp_pipeline,
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 expert_mid_dim,
                                                 pair_rows,
                                                 0,
                                                 -clamp,
                                                 clamp);
            }
            if (ok) {
                ds4_gpu_bin_args mul_args =
                    ds4_gpu_make_bin_same_rows_args(expert_mid_dim, pair_rows);
                ok = ds4_gpu_encode_bin_f32_rows(cb,
                                                   g_mul_pipeline,
                                                   &mul_args,
                                                   midbuf,
                                                   ds4_gpu_tensor_offset(mid),
                                                   upbuf,
                                                   ds4_gpu_tensor_offset(up),
                                                   midbuf,
                                                   ds4_gpu_tensor_offset(mid));
            }
        } else if (ok) {
            ok = ds4_gpu_encode_swiglu_flat(cb,
                                              gatebuf,
                                              ds4_gpu_tensor_offset(gate),
                                              upbuf,
                                              ds4_gpu_tensor_offset(up),
                                              midbuf,
                                              ds4_gpu_tensor_offset(mid),
                                              (uint32_t)((uint64_t)pair_rows * expert_mid_dim));
        }
        if (ok && !use_fused_activation) {
            ds4_gpu_bin_args weight_args =
                ds4_gpu_make_bin_rowwise_scalar_args(expert_mid_dim, pair_rows);
            ok = ds4_gpu_encode_bin_f32_rows(cb,
                                               g_bin_mul_scalar_pipeline,
                                               &weight_args,
                                               midbuf,
                                               ds4_gpu_tensor_offset(mid),
                                               weightsbuf,
                                               ds4_gpu_tensor_offset(weights),
                                               midbuf,
                                               ds4_gpu_tensor_offset(mid));
        }
        DS4_METAL_PROFILE_MOE_STAGE("activation_weight");

        id<MTLBuffer> down_dst = n_expert == 1 ? outbuf : (expertsbuf ? expertsbuf : g_moe_down_scratch_buffer);
        NSUInteger down_dst_off = n_expert == 1 ? ds4_gpu_tensor_offset(out) :
            (expertsbuf ? ds4_gpu_tensor_offset(experts) : 0);
        if (ok) {
            if (direct_down_sum) {
                ok = ds4_gpu_encode_mul_mv_id_sum6(cb,
                                                     down_sum6_pipeline,
                                                     &down_args,
                                                     down_buf,
                                                     (NSUInteger)down_inner,
                                                     midbuf,
                                                     ds4_gpu_tensor_offset(mid),
                                                     outbuf,
                                                     ds4_gpu_tensor_offset(out),
                                                     selectedbuf,
                                                     ds4_gpu_tensor_offset(selected),
                                                     down_smem,
                                                     2);
            } else if (use_mm_id) {
                /* R5-C: the hot gate/up pass rebuilt the shared work map from the
                 * hot sel; restore the base map before the base down tile. */
                if (do_hot && n_hot_picks > 0)
                    ok = ok && ds4_gpu_encode_mul_mm_id_map(cb, map_pipeline, &gate_map_args,
                                   &gate_mm_args, selectedbuf, ds4_gpu_tensor_offset(selected));
                if (ok) ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                       down_mm_pipeline,
                                                       &down_mm_args,
                                                       down_buf,
                                                       (NSUInteger)down_inner,
                                                       midbuf,
                                                       ds4_gpu_tensor_offset(mid),
                                                       down_dst,
                                                       down_dst_off);
            } else {
                ok = ds4_gpu_encode_mul_mv_id(cb,
                                                     down_mv_pipeline,
                                                     &down_args,
                                                     down_buf,
                                                     (NSUInteger)down_inner,
                                                     midbuf,
                                                     ds4_gpu_tensor_offset(mid),
                                                     down_dst,
                                                     down_dst_off,
                                                     selectedbuf,
                                                     ds4_gpu_tensor_offset(selected),
                                                     down_smem,
                                                     2,
                                                     false);
            }
        }
        if (ok && do_residual && use_mm_id && !direct_down_sum) {
            /* residual down: gather active residual down experts into a compacted
             * scratch, run the go1b mm_id down matmul with the base's down map, and
             * sum into the per-expert down output (before sum_experts). */
            uint64_t dneed = (uint64_t)n_active * down_expert_bytes;
            if (!g_moe_res_down_scratch || (uint64_t)g_moe_res_down_scratch.length < dneed)
                g_moe_res_down_scratch = [g_device newBufferWithLength:dneed options:MTLResourceStorageModeShared];
            if (g_moe_res_down_scratch) {
                for (uint32_t i = 0; i < n_active; i++) {
                    uint32_t e = active_ids[i];
                    int slot = res_lut ? (int)res_lut[e] : (int)e;
                    uint8_t *dd = (uint8_t *)g_moe_res_down_scratch.contents + (uint64_t)i * down_expert_bytes;
                    if (slot >= 0)
                        memcpy(dd, (const uint8_t *)res_down_ptr + (uint64_t)slot * down_expert_bytes, down_expert_bytes);
                    else
                        memset(dd, 0, down_expert_bytes);
                }
                const uint32_t don = (uint32_t)n_tokens * n_expert * out_dim;
                ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb, down_mm_pipeline, &down_mm_args,
                         g_moe_res_down_scratch, 0, midbuf, ds4_gpu_tensor_offset(mid),
                         g_moe_go1b_res_scratch, 0)
                  && ds4_gpu_encode_add_f32_1d(cb, down_dst, down_dst_off,
                         g_moe_go1b_res_scratch, 0, down_dst, down_dst_off, don);
                if (getenv("DS4_RESIDUAL_DEBUG"))
                    fprintf(stderr, "ds4: [res-down] L%u don=%u ok=%d\n", layer_index, don, ok);
            }
        }
        if (ok && do_hot && n_hot_picks > 0 && use_mm_id) {
            /* R5-C hot down: hot map + one go2b down matmul writing the hot rows
             * of down_dst (disjoint from the base rows; sum_experts then folds
             * every row exactly once). */
            const uint64_t g2_down_row = (uint64_t)expert_mid_dim / 256u * 68u;
            const uint64_t g2_down_exp = (uint64_t)out_dim * g2_down_row;
            id<MTLComputePipelineState> hot_down_pipe = request_mid_f16 ?
                ds4_gpu_routed_mm_f16_rhs_pipeline(DS4_METAL_TENSOR_GO2B) :
                ds4_gpu_routed_mm_pipeline(DS4_METAL_TENSOR_GO2B);
            ds4_gpu_mul_mm_id_map_args hot_map_args =
                ds4_gpu_make_mul_mm_id_map_args(expert_mid_dim, n_hot_slots, 1, n_expert, n_tokens);
            ds4_gpu_mul_mm_id_args hot_down_args =
                ds4_gpu_make_mul_mm_id_args_src1_size(expert_mid_dim, out_dim, n_hot_slots,
                                                        g2_down_row, g2_down_exp,
                                                        n_expert, n_expert, n_tokens,
                                                        request_mid_f16 ? sizeof(uint16_t) : sizeof(float));
            ok = hot_down_pipe != nil
              && ds4_gpu_encode_mul_mm_id_map(cb, map_pipeline, &hot_map_args,
                     &hot_down_args, g_moe_hot_sel, 0)
              && ds4_gpu_encode_mul_mm_id_mapped_tile(cb, hot_down_pipe, &hot_down_args,
                     g_moe_hot_down_scratch, 0, midbuf, ds4_gpu_tensor_offset(mid),
                     down_dst, down_dst_off);
            DS4_METAL_PROFILE_MOE_STAGE("hot-down");
        }
        DS4_METAL_PROFILE_MOE_STAGE("down");
        if (ok && n_expert > 1 && !direct_down_sum) {
            ok = ds4_gpu_encode_moe_sum_experts(cb,
                                                       down_dst,
                                                       down_dst_off,
                                                       outbuf,
                                                       ds4_gpu_tensor_offset(out),
                                                       out_dim,
                                                       n_expert,
                                                       n_tokens);
        }
        DS4_METAL_PROFILE_MOE_STAGE("sum");
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "routed batch MoE")) return 0;
        if (g_vq_diag_ref) {
            const float *gpu = (const float*)((const uint8_t*)outbuf.contents + ds4_gpu_tensor_offset(out));
            uint32_t N = g_vq_diag_ntok * g_vq_diag_out; int nn=0, nbig=0; double mxd=0; int firstdiv=-1;
            for (uint32_t k=0;k<N;k++){ float gv=gpu[k], cv=g_vq_diag_ref[k]; double d=fabs((double)gv-cv);
                if(gv!=gv)nn++; else if(d>mxd)mxd=d; if(d>0.05){ nbig++; if(firstdiv<0)firstdiv=(int)k; } }
            fprintf(stderr,"[VQ_DIFF] L%u N=%u gpu_nan=%d big_diff(>0.05)=%d maxd=%.4f first@%d(tok%d,d%d ref=%.3f)\n",
                    g_vq_diag_layer, N, nn, nbig, mxd, firstdiv, firstdiv>=0?firstdiv/g_vq_diag_out:-1,
                    firstdiv>=0?firstdiv%g_vq_diag_out:-1, firstdiv>=0?g_vq_diag_ref[firstdiv]:0.0);
            free(g_vq_diag_ref); g_vq_diag_ref=NULL;
        }
#undef DS4_METAL_PROFILE_MOE_STAGE
    }

    return 1;
}
