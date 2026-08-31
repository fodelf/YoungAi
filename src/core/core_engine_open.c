/* core_engine_open.c — 引擎打开(装配全部子系统) (机械拆分自 ds4.c, 重构阶段4)。 */
/* EXCEPTION(>500行): 单函数 ds4_engine_open, 函数内拆分是后续工序(需真模型逐位闸) */
#include "core_internal.h"
int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    ds4_engine *e = xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->backend = opt->backend;
    e->quality = opt->quality;
    e->distributed = opt->distributed;
    g_ds4_spec_enabled = opt->spec ? 1 : 0;
    g_ds4_draft_gguf_path = opt->draft_gguf_path;
    e->power_percent = opt->power_percent > 0 ? opt->power_percent : 100;
    if (e->power_percent > 100) e->power_percent = 100;
    if ((opt->directional_steering_attn != 0.0f || opt->directional_steering_ffn != 0.0f) &&
        (!opt->directional_steering_file || !opt->directional_steering_file[0]))
    {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        free(e);
        *out = NULL;
        return 1;
    }
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        e->directional_steering_file = ds4_strdup(opt->directional_steering_file);
        e->directional_steering_attn_scale = opt->directional_steering_attn;
        e->directional_steering_ffn_scale = opt->directional_steering_ffn;
    }
    if (opt->n_threads > 0) g_requested_threads = (uint32_t)opt->n_threads;
    ds4_acquire_instance_lock();

    bool load_slice = opt->load_slice;
    uint32_t load_layer_start = opt->load_layer_start;
    uint32_t load_layer_end = opt->load_layer_end;
    bool load_output = opt->load_output;
    if (opt->distributed.role != DS4_DISTRIBUTED_NONE &&
        opt->distributed.layers.set)
    {
        load_slice = true;
        load_layer_start = opt->distributed.layers.start;
        load_layer_end = opt->distributed.layers.has_output ?
                         UINT32_MAX : opt->distributed.layers.end;
        load_output = opt->distributed.layers.has_output;
    }
    /* MTP drafter 拓扑整族已删除(2026-08-05)。 */
    const bool include_output_head = load_output;
    const bool mtp_keep_token_embd = false;
    const bool graph_backend = ds4_backend_uses_graph(opt->backend);
    /* BASE model: the only open allowed to arm go1b/go2b env defaults (all
     * sidecar opens leave the flag false). */
    g_model_open_arm_env_defaults = true;
    model_open(&e->model, opt->model_path, graph_backend, !opt->inspect_only);
    g_model_open_arm_env_defaults = false;
    if (opt->warm_weights) model_warm_weights(&e->model);
    if (!opt->inspect_only) vocab_load(&e->vocab, &e->model);
    config_validate_model(&e->model);
    weights_bind(&e->weights, &e->model);
    dspark_bind_with_draft(&e->dspark, &e->model, graph_backend);
    if (opt->inspect_only) {
        *out = e;
        return 0;
    }

    /* go1b "hidden variable z^L" four-loss correction sidecar. An explicit --corr
     * PATH is always honoured; otherwise, when this model uses strict-1-bit (go1b)
     * routed experts, auto-detect ds4-go1b-corr.gguf next to the -m model. Absent
     * or unreadable => exactly the pure 1-bit path (fully backward compatible). */
    {
        const char *corr_path = opt->corr_path;
        char corr_auto[1024];
        if (!corr_path || !corr_path[0]) {
            bool is_go1b = false;
            for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                if (e->weights.layer[il].ffn_gate_exps &&
                    e->weights.layer[il].ffn_gate_exps->type == DS4_TENSOR_GO1B) {
                    is_go1b = true;
                    break;
                }
            }
            if (is_go1b && opt->model_path) {
                const char *slash = strrchr(opt->model_path, '/');
                if (slash) {
                    size_t dlen = (size_t)(slash - opt->model_path) + 1;   /* keep '/' */
                    if (dlen < sizeof(corr_auto) - sizeof("ds4-go1b-corr.gguf")) {
                        memcpy(corr_auto, opt->model_path, dlen);
                        memcpy(corr_auto + dlen, "ds4-go1b-corr.gguf", sizeof("ds4-go1b-corr.gguf"));
                        if (access(corr_auto, R_OK) == 0) corr_path = corr_auto;
                    }
                } else if (access("ds4-go1b-corr.gguf", R_OK) == 0) {
                    snprintf(corr_auto, sizeof(corr_auto), "ds4-go1b-corr.gguf");
                    corr_path = corr_auto;
                }
            }
        }
        if (corr_path && corr_path[0]) {
            e->model.corr = corr_load(corr_path, graph_backend);
        }
        /* 1-bit residual sidecar (--residual): a second go1b layer per hot expert,
         * summed into the base expert output. Absent => single 1-bit (today). */
        {   const char *res_path = (opt->residual_path && opt->residual_path[0])
                                 ? opt->residual_path : NULL;
            const char *vq_dir = (opt->vq_dir_path && opt->vq_dir_path[0])
                               ? opt->vq_dir_path : NULL;   /* --vq-dir */
            if (vq_dir && !e->model.residual) {
                e->model.residual = vq_dir_load(vq_dir);
                if (e->model.residual) res_path = NULL;
            }
            if (res_path && res_path[0])
                e->model.residual = residual_load(res_path, graph_backend);
            /* 合一 VQ GGUF: 未显式指定时, 文件自带 blob 张量即自动装载(文件即权威)。 */
            if (!e->model.residual)
                e->model.residual = vq_model_load(&e->model);
        }
        /* go-onebit 优化链: 外部 --zchain 文件优先(实验覆盖); 否则合一
         * GGUF 内嵌 blk.L.opt_* 张量(ds4.zchain.present)自动装载。GE 增益乘进
         * 路由权重 + 逐 token routed 缩放 λ(x)。都缺 => 素颜 1bit+signref 基座。 */
        {
            const char *zchain_path = (opt->zchain_path && opt->zchain_path[0])
                                    ? opt->zchain_path : NULL;
            if (zchain_path && zchain_path[0]) {
                e->model.zchain = ds4_zchain_load(zchain_path, DS4_N_LAYER,
                                                  DS4_N_EXPERT, DS4_N_EMBD);
                /* 显式请求的侧车打不开/空链 => 硬失败。静默裸跑过一次假对照
                 * (+z==裸, 2026-08-20), 判决容不得兜底。 */
                if (!e->model.zchain) {
                    fprintf(stderr, "ds4: zchain %s requested but unusable -- aborting (no silent bare-model fallback)\n",
                            zchain_path);
                    exit(1);
                }
            } else {
                e->model.zchain = zchain_from_model(&e->model);
            }
        /* 第4文件(2026-08-20 用户四文件设计): drafter 反修放大器侧车 --draft-zchain。
         * 3 层链(mtp.0/1/2)合并进主链尾部槽 43..45 ⇒ 单 GPU 表一次上传;
         * 合并链同挂主/draft 两个 model, ffn_batch 的 zch=model->zchain 两侧都取到,
         * drafter FFN 按 il=43+b 索引。显式请求打不开 => 硬失败(无静默兜底)。 */
        {
            const char *draft_zc = opt->draft_zchain_path;
            if (draft_zc && draft_zc[0]) {
                if (!g_draft_model || !e->dspark.ready) {
                    fprintf(stderr, "ds4: --draft-zchain requires a mounted drafter (--draft-gguf)\n");
                    exit(1);
                }
                struct ds4_zchain *dz = ds4_zchain_load(draft_zc, 3, DS4_N_EXPERT, DS4_N_EMBD);
                if (!dz) {
                    fprintf(stderr, "ds4: draft zchain %s unusable -- aborting\n", draft_zc);
                    exit(1);
                }
                struct ds4_zchain *base = e->model.zchain;
                struct ds4_zchain *mg = xmalloc(sizeof(*mg));
                memset(mg, 0, sizeof(*mg));
                mg->n_layer = (uint32_t)DS4_N_LAYER + 3u;
                mg->n_expert = DS4_N_EXPERT;
                mg->d_model = DS4_N_EMBD;
                mg->layer = xcalloc(mg->n_layer, sizeof(mg->layer[0]));
                if (base) {
                    memcpy(mg->layer, base->layer, (size_t)DS4_N_LAYER * sizeof(mg->layer[0]));
                    mg->n_ops_total = base->n_ops_total;
                    mg->n_ge_layers = base->n_ge_layers;
                    mg->map = base->map; mg->map_size = base->map_size;
                }
                memcpy(mg->layer + DS4_N_LAYER, dz->layer, 3u * sizeof(mg->layer[0]));
                mg->n_ops_total += dz->n_ops_total;
                mg->n_ge_layers += dz->n_ge_layers;
                /* base/dz 壳被 merged alias(单例, 进程生命周期), 不 free */
                e->model.zchain = mg;
                g_draft_model->zchain = mg;
                fprintf(stderr, "ds4: draft zchain merged: %s (3 layers @ slots 43..45)\n", draft_zc);
            } else if (g_draft_model && e->model.zchain) {
                /* 无 draft 侧车但有主侧车: drafter FFN 的 zch 取 dmodel->zchain,
                 * 保持 NULL 即旁路(主链槽 0..42 与 drafter il 43+b 互不相扰) */
                g_draft_model->zchain = NULL;
            }
        }
#ifndef DS4_NO_GPU
            if (e->model.zchain && graph_backend &&
                !zchain_gpu_upload(e->model.zchain)) {
                fprintf(stderr, "ds4: zchain GPU upload failed -- aborting (no silent quality downgrade)\n");
                exit(1);
            }
#endif
        }
        /* go-trie/ref-corpus drafter 设施已随 copy-spec/MTP 整族删除(2026-08-05 用户裁决)。 */
        /* 多模态 registry, same tokenizer bridge. The image family binds an
         * external encoder command when one is present -- resolution:
         * --mm-image-cmd, else ./mm-ui (the frontend-domain UI-sketch
         * tool, `make mm-ui`; cwd-relative like the metal shader dir, so
         * --chdir applies). Absent => image content is honestly rejected
         * upstream (server 400s image blocks instead of dropping them). */
        e->mm = ds4_mm_create(engine_tokenize_cb, e);
        if (e->mm) {
            const char *mm_cmd = opt->mm_image_cmd;
            if ((!mm_cmd || !mm_cmd[0]) && access("mm-ui", X_OK) == 0)
                mm_cmd = "./mm-ui";
            if (mm_cmd && mm_cmd[0] &&
                ds4_mm_register_command(e->mm, "image", mm_cmd) == 0)
                fprintf(stderr, "ds4: multimodal image encoder: %s\n", mm_cmd);
            /* 前端域两插件, 顺序=节顺序: 物理方位先(关系), CSS 后(换算)。
             * 与编码器解耦: 换编码器(--mm-image-cmd)后输出若非草图格式,
             * 两节自然缺席, 不伪造。 */
            ds4_mm_register_enricher(e->mm, "image", engine_mm_spatial_enrich, NULL);
            ds4_mm_register_enricher(e->mm, "image", engine_mm_css_enrich, NULL);
        }
    }
    if (e->backend == DS4_BACKEND_CPU && !cpu_load_directional_steering(e)) {
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    /* MTP 支持模型加载已整族删除(2026-08-05 用户裁决: Go 定型优化)。 */
#ifndef DS4_NO_GPU
    if (e->backend == DS4_BACKEND_CUDA) {
#ifdef __APPLE__
        fprintf(stderr, "ds4: CUDA backend requested but this build is linked with Metal, not CUDA\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
#endif
    }
    if (e->backend == DS4_BACKEND_METAL) {
#ifndef __APPLE__
        fprintf(stderr, "ds4: Metal backend requested but this build is linked with CUDA, not Metal\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
#endif
    }
    if (graph_backend) {
        e->metal_ready = ds4_gpu_init() != 0;
        if (!e->metal_ready) {
            fprintf(stderr, "ds4: %s backend unavailable; aborting startup\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        ds4_gpu_set_quality(e->quality);
        (void)ds4_gpu_set_model_fd(e->model.fd);
        /* project.md P2.2 low-cost variant: when DS4_DIST_EXPERT_FETCH_SERVE=1
         * (worker side), serve raw model-file range reads so the peer's expert
         * gather can draw from this machine's faster idle SSD over Thunderbolt. */
        (void)ds4_dist_expert_fetch_maybe_serve(e->model.fd, e->model.size);
        /* Wave 30 reverse-established variant: when the worker cannot dial out
         * (asymmetric bridge), the coordinator dials the worker's accept-mode
         * listener instead and serves preads on the dialed sockets. */
        (void)ds4_dist_expert_fetch_serve_dial(e->model.fd, e->model.size);
        int model_map_ok = 0;
        uint64_t base_l1_resident_bytes = 0;
        /* Dynamic resident/offload route: keep routed experts resident -- direct
         * GPU read, no per-layer CPU gather, full decode speed -- whenever the
         * fully-resident model fits the memory budget; only stream when it would
         * bust it. AUTO 是唯一路径(env 覆盖口已删): 预算来自 --mem-budget-mb,
         * 缺省退 GPU recommended working set。 */
        bool expert_offload_requested;
        {
            uint64_t full_resident_bytes = 0;
            if (load_slice) {
                ds4_model_map_span_vec bb_probe, exp_probe;
                if (weights_model_map_spans_split_slice(&e->weights, load_layer_start,
                        load_layer_end, include_output_head, mtp_keep_token_embd,
                        &bb_probe, &exp_probe)) {
                    for (uint32_t i = 0; i < bb_probe.len; i++)
                        full_resident_bytes += bb_probe.v[i].end - bb_probe.v[i].off;
                    for (uint32_t i = 0; i < exp_probe.len; i++)
                        full_resident_bytes += exp_probe.v[i].end - exp_probe.v[i].off;
                    free(bb_probe.v);
                    free(exp_probe.v);
                }
            } else {
                full_resident_bytes = e->model.size - e->model.tensor_data_pos;
            }
            uint64_t auto_budget = ds4_runtime_mem_budget_bytes();
            if (auto_budget == 0) auto_budget = ds4_gpu_recommended_max_working_set_bytes();
            /* 判定线=预算 85%(与 L1 闸同值是巧合, 语义独立): 全驻留贴线时 KV/scratch
             * 一涨就顶穿看门狗, 提前转 stream(offload)保命换速。 */
            #define DS4_OFFLOAD_AUTO_FRAC 0.85
            expert_offload_requested = (auto_budget > 0) && (full_resident_bytes > 0) &&
                (full_resident_bytes > (uint64_t)((double)auto_budget * DS4_OFFLOAD_AUTO_FRAC));
            fprintf(stderr,
                    "ds4: expert-offload AUTO: full-resident %.2f GiB vs %.2f GiB budget -> %s\n",
                    (double)full_resident_bytes / DS4_GIB, (double)auto_budget / DS4_GIB,
                    expert_offload_requested ? "stream (offload)" : "resident (fast)");
        }
        /* VQ blob 专家不进 Metal span(ds4.c:2074): 前向唯一路径=CPU gather→f16 scratch
         * (ds4.c:2277 从 mmap vq_raw 读)。resident(offload=0)会让 MoE kernel 去 residency
         * set 直读不存在的专家 span → 崩。双机切层后 planned<budget 时 AUTO 会误选 resident,
         * 故 VQ blob 恒强制 offload。 */
        if (g_vq_experts_blob && !expert_offload_requested) {
            fprintf(stderr, "ds4: VQ blob 专家: 强制 offload(CPU gather 是唯一前向路径, 覆盖 AUTO resident)\n");
            expert_offload_requested = true;
        }
        ds4_gpu_set_expert_offload(expert_offload_requested ? 1 : 0);
        if (load_slice) {
            char load_end[32];
            if (load_output && load_layer_end == UINT32_MAX) {
                snprintf(load_end, sizeof(load_end), "output");
            } else if (load_output) {
                snprintf(load_end, sizeof(load_end), "%u+output", load_layer_end);
            } else {
                snprintf(load_end, sizeof(load_end), "%u", load_layer_end);
            }

            if (expert_offload_requested) {
                ds4_model_map_span_vec bb, exp;
                if (!weights_model_map_spans_split_slice(&e->weights,
                                                         load_layer_start,
                                                         load_layer_end,
                                                         include_output_head,
                                                         mtp_keep_token_embd,
                                                         &bb,
                                                         &exp))
                {
                    fprintf(stderr, "ds4: invalid expert-offload model load layer slice %u:%s\n",
                            load_layer_start,
                            load_end);
                    ds4_engine_close(e);
                    *out = NULL;
                    return 1;
                }
                const uint32_t total = bb.len + exp.len;
                uint64_t *offsets = xmalloc((size_t)total * sizeof(offsets[0]));
                uint64_t *sizes = xmalloc((size_t)total * sizeof(sizes[0]));
                bool *resident = xmalloc((size_t)total * sizeof(resident[0]));
                uint64_t resident_bytes = 0, reclaimable_bytes = 0;
                uint32_t n = 0;
                for (uint32_t i = 0; i < bb.len; i++) {
                    offsets[n] = bb.v[i].off;
                    sizes[n] = bb.v[i].end - bb.v[i].off;
                    resident[n] = true;
                    resident_bytes += sizes[n];
                    n++;
                }
                for (uint32_t i = 0; i < exp.len; i++) {
                    offsets[n] = exp.v[i].off;
                    sizes[n] = exp.v[i].end - exp.v[i].off;
                    resident[n] = false;
                    reclaimable_bytes += sizes[n];
                    n++;
                }
                uint64_t split_max_tensor = bb.max_tensor_bytes;
                if (exp.max_tensor_bytes > split_max_tensor) split_max_tensor = exp.max_tensor_bytes;
                base_l1_resident_bytes = resident_bytes;
                fprintf(stderr,
                        "ds4: restricting %s model map to layers %u:%s with expert offload "
                        "(%.2f GiB backbone resident, %.2f GiB routed experts reclaimable; "
                        "%u backbone + %u expert spans)\n",
                        ds4_backend_name(e->backend),
                        load_layer_start,
                        load_end,
                        (double)resident_bytes / 1073741824.0,
                        (double)reclaimable_bytes / 1073741824.0,
                        bb.len,
                        exp.len);
                ds4_l1_budget_gate(base_l1_resident_bytes, 0);
                model_map_ok = ds4_gpu_set_model_map_spans_split(e->model.map,
                                                                 e->model.size,
                                                                 offsets,
                                                                 sizes,
                                                                 resident,
                                                                 total,
                                                                 split_max_tensor);
                if (model_map_ok) {
                    engine_register_layer_routers(e, load_layer_start, load_layer_end);
                }
                free(offsets);
                free(sizes);
                free(resident);
                free(bb.v);
                free(exp.v);
            } else {
                ds4_model_map_span_vec spans;
                if (!weights_model_map_spans(&e->weights,
                                             load_layer_start,
                                             load_layer_end,
                                             include_output_head,
                                             mtp_keep_token_embd,
                                             &spans))
                {
                    fprintf(stderr, "ds4: invalid model load layer slice %u:%s\n",
                            load_layer_start,
                            load_end);
                    ds4_engine_close(e);
                    *out = NULL;
                    return 1;
                }
                uint64_t *offsets = xmalloc((size_t)spans.len * sizeof(offsets[0]));
                uint64_t *sizes = xmalloc((size_t)spans.len * sizeof(sizes[0]));
                uint64_t span_bytes = 0;
                for (uint32_t i = 0; i < spans.len; i++) {
                    offsets[i] = spans.v[i].off;
                    sizes[i] = spans.v[i].end - spans.v[i].off;
                    span_bytes += sizes[i];
                }
                base_l1_resident_bytes = span_bytes;
                fprintf(stderr,
                        "ds4: restricting %s model map to layers %u:%s (%u spans, %.2f GiB tensor span)\n",
                        ds4_backend_name(e->backend),
                        load_layer_start,
                        load_end,
                        spans.len,
                        (double)span_bytes / 1073741824.0);
                ds4_l1_budget_gate(base_l1_resident_bytes, 0);
                model_map_ok = ds4_gpu_set_model_map_spans(e->model.map,
                                                            e->model.size,
                                                            offsets,
                                                            sizes,
                                                            spans.len,
                                                            spans.max_tensor_bytes);
                free(offsets);
                free(sizes);
                free(spans.v);
            }
        } else if (expert_offload_requested) {
            /* Reduced-memory load: wire only the backbone (attn / shared FFN /
             * embedding / output) into the GPU residency set and keep the routed
             * experts reclaimable. The hot path still resolves every tensor's
             * buffer; cold experts just are not pinned resident. */
            ds4_model_map_span_vec bb, exp;
            if (!weights_model_map_spans_split(&e->weights, &bb, &exp)) {
                fprintf(stderr,
                        "ds4: DS4_METAL_EXPERT_OFFLOAD requested but span split failed; "
                        "falling back to the full-residency loader\n");
                base_l1_resident_bytes = e->model.size - e->model.tensor_data_pos;
                ds4_l1_budget_gate(base_l1_resident_bytes, 0);
                model_map_ok = ds4_gpu_set_model_map_range(e->model.map,
                                                           e->model.size,
                                                           e->model.tensor_data_pos,
                                                           e->model.size - e->model.tensor_data_pos,
                                                           e->model.max_tensor_bytes);
            } else {
                const uint32_t total = bb.len + exp.len;
                uint64_t *offsets = xmalloc((size_t)total * sizeof(offsets[0]));
                uint64_t *sizes = xmalloc((size_t)total * sizeof(sizes[0]));
                bool *resident = xmalloc((size_t)total * sizeof(resident[0]));
                uint64_t resident_bytes = 0, reclaimable_bytes = 0;
                uint32_t n = 0;
                for (uint32_t i = 0; i < bb.len; i++) {
                    offsets[n] = bb.v[i].off;
                    sizes[n] = bb.v[i].end - bb.v[i].off;
                    resident[n] = true;
                    resident_bytes += sizes[n];
                    n++;
                }
                for (uint32_t i = 0; i < exp.len; i++) {
                    offsets[n] = exp.v[i].off;
                    sizes[n] = exp.v[i].end - exp.v[i].off;
                    resident[n] = false;
                    reclaimable_bytes += sizes[n];
                    n++;
                }
                uint64_t split_max_tensor = bb.max_tensor_bytes;
                if (exp.max_tensor_bytes > split_max_tensor) split_max_tensor = exp.max_tensor_bytes;
                fprintf(stderr,
                        "ds4: expert-offload model map: %.2f GiB backbone resident, "
                        "%.2f GiB routed experts reclaimable (%u backbone + %u expert spans)\n",
                        (double)resident_bytes / 1073741824.0,
                        (double)reclaimable_bytes / 1073741824.0,
                        bb.len, exp.len);
                base_l1_resident_bytes = resident_bytes;
                ds4_l1_budget_gate(base_l1_resident_bytes, 0);
                model_map_ok = ds4_gpu_set_model_map_spans_split(e->model.map,
                                                                 e->model.size,
                                                                 offsets,
                                                                 sizes,
                                                                 resident,
                                                                 total,
                                                                 split_max_tensor);
                if (model_map_ok) {
                    engine_register_layer_routers(e, 0, DS4_MAX_LAYER - 1);
                }
                free(offsets);
                free(sizes);
                free(resident);
                free(bb.v);
                free(exp.v);
            }
        } else {
            base_l1_resident_bytes = e->model.size - e->model.tensor_data_pos;
            ds4_l1_budget_gate(base_l1_resident_bytes, 0);
            model_map_ok = ds4_gpu_set_model_map_range(e->model.map,
                                                       e->model.size,
                                                       e->model.tensor_data_pos,
                                                       e->model.size - e->model.tensor_data_pos,
                                                       e->model.max_tensor_bytes);
        }
        if (!model_map_ok) {
            fprintf(stderr,
                    "ds4: %s failed to map model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        /* 副 drafter map 注册必须在主模型 map 之后: ds4_gpu_set_model_map 换 base 时
         * release_all 会连带释放先注册的 range(2026-08-21 实测顺序坑)。 */
        /* 副 map 注册失败 => drafter 读到的是不可用的裸指针(实测 logits NaN, 草稿全废、
         * acc 塌到 1.00 而速度看似"正常")。明确停用 drafter, 不接受静默劣化。 */
        if (g_draft_model &&
            !ds4_gpu_register_aux_model_map(g_draft_model->map, g_draft_model->size)) {
            fprintf(stderr, "ds4: draft gguf aux map registration failed -- disabling drafter "
                            "(speculation off; plain decode continues)\n");
            e->dspark.ready = false;
            g_dspark_ready_global = 0;
            g_dspark_bound_for_prefill = NULL;
        }
        if (!accelerator_cache_model_tensors(e->backend, &e->model)) {
            fprintf(stderr, "ds4: %s failed to prepare startup model cache\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        fprintf(stderr, "ds4: %s backend initialized for graph diagnostics\n",
                ds4_backend_name(e->backend));
    }
#else
    if (graph_backend) {
        fprintf(stderr, "ds4: %s backend requested but this build has no graph backend support; aborting startup\n",
                ds4_backend_name(e->backend));
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
#endif

    ds4_multi_bench_run(e);   /* DS4_MULTI_BENCH=N: 并发批实测(跑完退出) */
    ds4_eval_ids_run(e);   /* DS4_EVAL_IDS 在场则跑完仪器直接退出, 不返回 */
    *out = e;
    return 0;
}
