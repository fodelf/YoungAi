/* core_eval_ids.c — 路由预取注册 + DS4_EVAL_IDS 终审 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU
/* project.md P2.1: register the locally-loaded routed layers' router metadata
 * (F16 gate_inp + optional exp_probs_b + expert tensor offsets) so the GPU
 * backend can re-evaluate the next layer's router on the CPU during decode and
 * read predicted experts ahead.  Only the local slice is registered: issuing
 * read-ahead for layers another machine executes would waste SSD bandwidth.
 * Purely advisory metadata; never affects inference results. */
void engine_register_layer_routers(ds4_engine *e, uint32_t start, uint32_t end) {
    if (e->model.expert_shrunken) return;   /* compact-slot models: ids differ */
    if (end >= DS4_MAX_LAYER) end = DS4_MAX_LAYER - 1;
    uint32_t registered = 0;
    for (uint32_t il = start; il <= end && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &e->weights.layer[il];
        if (!l->ffn_gate_exps) continue;   /* no routed MoE on this layer */
        const char *why = NULL;
        uint64_t hash_off = UINT64_MAX;
        uint32_t hash_k = 0, hash_rows = 0;
        if (!l->ffn_up_exps || !l->ffn_down_exps) why = "missing expert tensors";
        else if (!l->ffn_gate_inp) why = "no gate_inp";
        else if (l->ffn_gate_inp->type != DS4_TENSOR_F16 &&
                 l->ffn_gate_inp->type != DS4_TENSOR_F32) why = "gate_inp quantized";
        else if (l->ffn_gate_tid2eid) {
            /* Early layers hash-route by token id (tid2eid I32 [k][n_vocab]):
             * exact prediction, register the table. */
            const ds4_tensor *t = l->ffn_gate_tid2eid;
            if (t->type == DS4_TENSOR_I32 && t->ndim == 2 &&
                t->dim[0] == DS4_N_EXPERT_USED && t->dim[1] > 0) {
                hash_off = t->abs_offset;
                hash_k = (uint32_t)t->dim[0];
                hash_rows = (uint32_t)t->dim[1];
            } else {
                why = "unrecognized tid2eid layout";
            }
        }
        if (!l->ffn_gate_exps || !l->ffn_up_exps) continue;   /* 内嵌 VQ: blob 为源, 不注册 base 专家预取 */
        const uint64_t n_exp = l->ffn_gate_exps->dim[2];
        if (!why && (n_exp == 0 || n_exp != DS4_N_EXPERT)) why = "expert count";
        if (why) {
            fprintf(stderr,
                    "ds4: layer %u router not registered for prefetch (%s; gate_inp type=%u, "
                    "n_exp=%llu)\n",
                    il, why,
                    l->ffn_gate_inp ? l->ffn_gate_inp->type : 9999u,
                    (unsigned long long)n_exp);
            continue;
        }
        if (ds4_gpu_register_layer_router(e->model.map,
                                          il,
                                          l->ffn_gate_inp->abs_offset,
                                          l->ffn_gate_inp->type == DS4_TENSOR_F32,
                                          l->ffn_exp_probs_b ? l->ffn_exp_probs_b->abs_offset
                                                             : UINT64_MAX,
                                          l->ffn_gate_exps->abs_offset,
                                          l->ffn_up_exps->abs_offset,
                                          l->ffn_down_exps->abs_offset,
                                          l->ffn_gate_exps->bytes / n_exp,
                                          l->ffn_down_exps->bytes / n_exp,
                                          (uint32_t)l->ffn_gate_inp->dim[0],
                                          (uint32_t)n_exp,
                                          hash_off,
                                          hash_k,
                                          hash_rows)) {
            registered++;
        }
    }
    if (registered) {
        fprintf(stderr,
                "ds4: registered %u local routed layers for cross-layer expert prefetch\n",
                registered);
    }
}
#endif

/* ---- DS4_EVAL_IDS 终审仪器 -------------------------------------------------
 * 用途: 与量化器锚文件里同序列的 FP logits 直接对账(held 区 Σmin), 判"引擎口径
 * 还原率 vs 量化器口径"。差距大时再用 HDUMP 的逐层 hidden 找第一分歧层。
 *
 *   DS4_EVAL_IDS=<ids文件>     原始 token id 流, 空白分隔的十进制整数(每行一个亦可)
 *   DS4_EVAL_LOGITS=<out.bin>  每位置最终 logits, f32 [S][DS4_N_VOCAB], 顺序流式写盘
 *   DS4_EVAL_HDUMP=<dir>       每层出口 hidden → h_L%02d.bin(见 eval_hdump_batch_layer)
 *   DS4_EVAL_NO_BOS=1          不在流首插 BOS(默认插, 即"裸 BOS 起")
 *
 * 走的是分布式推理那条已验证的裸 token 通道(ds4_session_eval_layer_slice), 天然不过
 * chat 模板/DSML, 与量化器 embed(ids) 直喂同口径。跑完 exit(0), 不采样。
 * 内存: logits 每次只留一个位置(VOCAB f32 ≈ 0.5 MB), 写一行冲一行, 不驻留 S×VOCAB。 */
static int *eval_ids_load(const char *path, uint32_t *out_n, uint32_t vocab) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ds4: [EVAL_IDS] 打不开 %s -- aborting\n", path);
        exit(1);
    }
    uint32_t cap = 4096, n = 0;
    int *ids = xmalloc((size_t)cap * sizeof(int));
    long v;
    while (fscanf(f, "%ld", &v) == 1) {
        if (v < 0 || (uint64_t)v >= (uint64_t)vocab) {
            fprintf(stderr, "ds4: [EVAL_IDS] %s 第 %u 个 id=%ld 越界 [0,%u) -- aborting\n",
                    path, n, v, vocab);
            exit(1);
        }
        if (n == cap) {
            cap *= 2;
            int *nb = realloc(ids, (size_t)cap * sizeof(int));
            if (!nb) { fprintf(stderr, "ds4: [EVAL_IDS] ids 扩容失败\n"); exit(1); }
            ids = nb;
        }
        ids[n++] = (int)v;
    }
    fclose(f);
    if (!n) {
        fprintf(stderr, "ds4: [EVAL_IDS] %s 里没有解析出任何 id -- aborting\n", path);
        exit(1);
    }
    *out_n = n;
    return ids;
}

/* 并发批实测(DS4_MULTI_BENCH=N, 2026-08-21): 建 N 个会话各喂不同 prompt, 先各自
 * 单独解码若干步做基准, 再用 ds4_session_eval_multi 批量推进同样的步数, 对比
 *   ①聚合 token/s(红利有多大)  ②每个会话选出的 token 序列是否与单独解码逐字相同(无损)。
 * 用 DS4_MULTI_BENCH_STEPS 调步数(默认 48)。 */
void ds4_multi_bench_run(ds4_engine *e) {
    const char *env = getenv("DS4_MULTI_BENCH");
    if (!env || !env[0]) return;
    uint32_t n = (uint32_t)atoi(env);
    if (n < 2u && !getenv("DS4_MULTI_FORCE")) n = 2u;
    if (n < 1u) n = 1u;
    if (n > 8u) n = 8u;
    uint32_t steps = 48u;
    { const char *sv = getenv("DS4_MULTI_BENCH_STEPS"); if (sv && atoi(sv) > 0) steps = (uint32_t)atoi(sv); }
    if (steps > 500u) steps = 500u;
    /* 只跑批模式(跳过单路基准): 演示 8 路并发时不必等基准 */
    const int batch_only = getenv("DS4_MULTI_BENCH_BATCH_ONLY") != NULL;

    /* 8 道真代码题(并发批处理演示用): 各自独立、长度相近, 便于横向比较 */
    static const char *prompts[8] = {
        "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals. Return only the code.",
        "Write a Go function `LRUCache` with Get and Put in O(1). Return only the code.",
        "Write a C function that reverses a singly linked list in place. Return only the code.",
        "Write a SQL query that returns each department's second-highest salary. Return only the query.",
        "Write a Python function `binary_search(arr, target)` returning the index or -1. Return only the code.",
        "Write a bash script that finds the 10 largest files under a directory. Return only the script.",
        "Write a JavaScript function `debounce(fn, wait)` and explain nothing. Return only the code.",
        "Write a Rust function that counts word frequencies in a string. Return only the code.",
    };
    char err[256];
    ds4_session *ss[8] = {0};
    ds4_tokens toks[8];
    memset(toks, 0, sizeof(toks));
    for (uint32_t i = 0; i < n; i++) {
        if (ds4_session_create(&ss[i], e, 4096) != 0 || !ss[i]) {
            fprintf(stderr, "ds4: [multi-bench] 会话 %u 创建失败\n", i); exit(1);
        }
        { uint32_t pi = i;
          const char *sv = getenv("DS4_MULTI_BENCH_START");
          if (sv) pi = ((uint32_t)atoi(sv) + i) & 7u;
          const char *ptext = getenv("DS4_MULTI_BENCH_SAMEPROMPT") ? prompts[0] : prompts[pi];
          /* 与 CLI 同一条渲染路: 走 chat 模板才是正经问答, 裸文本只会续写题面 */
          if (getenv("DS4_MULTI_BENCH_RAW")) ds4_tokenize_text(e, ptext, &toks[i]);
          else ds4_encode_chat_prompt(e, NULL, ptext, DS4_THINK_NONE, &toks[i]); }
        if (ds4_session_sync(ss[i], &toks[i], err, sizeof err) != 0) {
            fprintf(stderr, "ds4: [multi-bench] 会话 %u prefill 失败: %s\n", i, err); exit(1);
        }
    }
    /* 基准: 各会话单独逐 token 解码 steps 步, 记下选出的 token */
    int (*ref)[512] = xmalloc((size_t)n * sizeof(*ref));
    float *ref_l1 = xmalloc((size_t)n * DS4_N_VOCAB * sizeof(float));   /* 第 1 步后的 logits */
    int (*bat)[512] = xmalloc((size_t)n * sizeof(*bat));
    double t0 = now_sec();
    for (uint32_t i = 0; !batch_only && i < n; i++) {
        for (uint32_t k = 0; k < steps; k++) {
            int best = 0; float bv = -1e30f;
            for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                if (ss[i]->logits[v] > bv) { bv = ss[i]->logits[v]; best = (int)v; }
            ref[i][k] = best;
            if (ds4_session_eval(ss[i], best, err, sizeof err) != 0) {
                fprintf(stderr, "ds4: [multi-bench] 单路解码失败: %s\n", err); exit(1);
            }
            if (k == 0) memcpy(ref_l1 + (uint64_t)i * DS4_N_VOCAB, ss[i]->logits,
                               (size_t)DS4_N_VOCAB * sizeof(float));
        }
    }
    const double seq_s = now_sec() - t0;
    if (!batch_only)
        fprintf(stderr, "ds4: [multi-bench] 单路逐个跑: %u 会话 × %u token = %u, 用时 %.2fs ⇒ 聚合 %.2f t/s\n",
                n, steps, n * steps, seq_s, (double)(n * steps) / seq_s);

    /* 批: 重建会话到同一起点, 用 eval_multi 同步推进 */
    for (uint32_t i = 0; i < n; i++) {
        ds4_session_free(ss[i]);
        ss[i] = NULL;
        if (ds4_session_create(&ss[i], e, 4096) != 0 ||
            ds4_session_sync(ss[i], &toks[i], err, sizeof err) != 0) {
            fprintf(stderr, "ds4: [multi-bench] 会话重建失败\n"); exit(1);
        }
    }
    uint32_t mismatch = 0, first_bad = 0;
    int cur[8];
    t0 = now_sec();
    for (uint32_t k = 0; k < steps; k++) {
        for (uint32_t i = 0; i < n; i++) {
            int best = 0; float bv = -1e30f;
            for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                if (ss[i]->logits[v] > bv) { bv = ss[i]->logits[v]; best = (int)v; }
            cur[i] = best;
            bat[i][k] = best;
            if (best != ref[i][k]) { if (!mismatch) first_bad = k; mismatch++; }
        }
        if (ds4_session_eval_multi(ss, cur, n, err, sizeof err) != 0) {
            fprintf(stderr, "ds4: [multi-bench] 批解码失败: %s\n", err); exit(1);
        }
        if (k == 0 && !batch_only) {   /* 第 1 步 logits 与单路对账 */
            for (uint32_t i = 0; i < n; i++) {
                const float *r = ref_l1 + (uint64_t)i * DS4_N_VOCAB;
                double dmax = 0.0; int ar = 0, ab = 0; float br = -1e30f, bb = -1e30f;
                for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++) {
                    const double d = fabs((double)r[v] - (double)ss[i]->logits[v]);
                    if (d > dmax) dmax = d;
                    if (r[v] > br) { br = r[v]; ar = (int)v; }
                    if (ss[i]->logits[v] > bb) { bb = ss[i]->logits[v]; ab = (int)v; }
                }
                fprintf(stderr, "ds4: [multi-bench] 会话%u(位置%d) 第1步: max|Δlogit|=%.4g 单路top1=%d 批top1=%d %s\n",
                        i, ss[i]->checkpoint.len, dmax, ar, ab, ar == ab ? "(一致)" : "(不同)");
            }
        }
    }
    const double bat_s = now_sec() - t0;
    fprintf(stderr, "ds4: [multi-bench] %u 路批处理: %u token, 用时 %.2fs ⇒ 聚合 %.2f t/s%s\n",
            n, n * steps, bat_s, (double)(n * steps) / bat_s,
            batch_only ? "" : "");
    if (!batch_only)
        fprintf(stderr, "ds4: [multi-bench] 相对单路加速 %.2fx\n", seq_s / bat_s);
    /* 质量目视: 两种模式各自的文本(逐位不同是数值等价的正常结果, 关键看是否连贯) */
    if (getenv("DS4_MULTI_BENCH_TEXT")) {
        for (uint32_t i = 0; i < n; i++) {
            static char buf[16384]; size_t off = 0;
            for (uint32_t k = 0; k < steps && off < sizeof(buf) - 64; k++) {
                size_t l = 0;
                const char *p = ds4_token_text(e, ref[i][k], &l);
                if (p && off + l < sizeof(buf) - 1) { memcpy(buf + off, p, l); off += l; }
            }
            buf[off] = 0;
            if (!batch_only) fprintf(stderr, "ds4: [multi-bench] 会话%u 单路: %s\n", i, buf);
            off = 0;
            for (uint32_t k = 0; k < steps && off < sizeof(buf) - 64; k++) {
                size_t l = 0;
                const char *p = ds4_token_text(e, bat[i][k], &l);
                if (p && off + l < sizeof(buf) - 1) { memcpy(buf + off, p, l); off += l; }
            }
            buf[off] = 0;
            fprintf(stderr, "\n===== 会话 %u =====\n%s\n", i, buf);
        }
    }
    if (!batch_only) fprintf(stderr, "ds4: [multi-bench] 与单路逐位对照: %s (不同 %u/%u%s)\n",
            mismatch ? "有差异" : "完全一致", mismatch, n * steps,
            mismatch ? "" : ", 无损");
    if (mismatch && !batch_only) fprintf(stderr, "ds4: [multi-bench] 首个不同在第 %u 步\n", first_bad);
    for (uint32_t i = 0; i < n; i++) ds4_session_free(ss[i]);
    free(ref);
    exit(0);
}

void ds4_eval_ids_run(ds4_engine *e) {
    const char *idp = getenv("DS4_EVAL_IDS");
    if (!idp || !idp[0]) return;
#ifdef DS4_NO_GPU
    (void)e;
    fprintf(stderr, "ds4: [EVAL_IDS] 需要 graph 后端 -- aborting\n");
    exit(1);
#else
    const uint32_t vocab = (uint32_t)DS4_N_VOCAB;
    uint32_t nfile = 0;
    int *file_ids = eval_ids_load(idp, &nfile, vocab);

    /* 默认在流首插 BOS(裸 BOS 起); DS4_EVAL_NO_BOS=1 则原样喂。 */
    const int no_bos = getenv("DS4_EVAL_NO_BOS") != NULL;
    const uint32_t n = no_bos ? nfile : nfile + 1u;
    int *ids = xmalloc((size_t)n * sizeof(int));
    if (no_bos) {
        memcpy(ids, file_ids, (size_t)nfile * sizeof(int));
    } else {
        ids[0] = e->vocab.bos_id;
        memcpy(ids + 1, file_ids, (size_t)nfile * sizeof(int));
    }
    free(file_ids);
    /* 首 8 个生效 id 打出来: 与锚文件对齐与否一眼可验(错位一格 Σmin 就全废)。 */
    fprintf(stderr, "ds4: [EVAL_IDS] S=%u (文件 %u%s) vocab=%u 首8: ",
            n, nfile, no_bos ? ", 无 BOS" : ", 首插 BOS", vocab);
    for (uint32_t i = 0; i < n && i < 8u; i++) fprintf(stderr, "%d ", ids[i]);
    fprintf(stderr, "\n");

    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, (int)n + 64) != 0 || !s) {
        fprintf(stderr, "ds4: [EVAL_IDS] 会话创建失败 -- aborting\n");
        exit(1);
    }
    const uint32_t pcap = (uint32_t)ds4_session_prefill_cap(s);
    if (!pcap) { fprintf(stderr, "ds4: [EVAL_IDS] prefill_cap=0 -- aborting\n"); exit(1); }
    /* 不设 DS4_METAL_PREFILL_CHUNK 时 prefill_cap 等于整个 ctx(一次灌完), 那会让 hc
     * 暂存和单批显存都按 S 放大。仪器自己按 512 分块(DS4_EVAL_CHUNK 可调), 与 A3
     * 的 PREFILL_CHUNK=512 口径一致; 分块只影响批大小, 不影响数值。 */
    uint32_t cap = 512u;
    { const char *cv = getenv("DS4_EVAL_CHUNK"); if (cv && atoi(cv) > 0) cap = (uint32_t)atoi(cv); }
    if (cap > pcap) cap = pcap;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    float *hc = xmalloc((size_t)cap * hc_dim * sizeof(float));

    FILE *lf = NULL; float *lg = NULL;
    const char *lp = getenv("DS4_EVAL_LOGITS");
    if (lp && lp[0]) {
        lf = fopen(lp, "wb");
        if (!lf) {
            fprintf(stderr, "ds4: [EVAL_IDS] 打不开 %s -- aborting\n", lp);
            exit(1);
        }
        lg = xmalloc((size_t)vocab * sizeof(float));
    }

    char err[256];
    for (uint32_t p0 = 0; p0 < n; p0 += cap) {
        const uint32_t nt = (n - p0 < cap) ? (n - p0) : cap;
        /* layer_start=0 ⇒ 由 token 直接 embed(input_hc=NULL); 走到最后一层拿整批出口 HC。
         * 逐层 hidden 由 eval_hdump_batch_layer 在同一趟前向里顺带写出, 不重复前向。 */
        if (ds4_session_eval_layer_slice(s, ids + p0, nt, p0, 0, (uint32_t)DS4_N_LAYER - 1u,
                                         NULL, hc, false, NULL, err, sizeof err) != 0) {
            fprintf(stderr, "ds4: [EVAL_IDS] prefill 失败 @pos %u: %s -- aborting\n", p0, err);
            exit(1);
        }
        if (lf) {
            /* 逐位置过输出头: eval_output_head_from_hc 只算传入批的最后一行, 所以按
             * n_tokens=1 逐位置喂该位置的 HC 隐状态。 */
            for (uint32_t t = 0; t < nt; t++) {
                if (ds4_session_eval_output_head_from_hc(s, hc + (uint64_t)t * hc_dim, 1u,
                                                         lg, err, sizeof err) != 0) {
                    fprintf(stderr, "ds4: [EVAL_IDS] 输出头失败 @pos %u: %s -- aborting\n",
                            p0 + t, err);
                    exit(1);
                }
                if (fwrite(lg, sizeof(float), vocab, lf) != vocab) {
                    fprintf(stderr, "ds4: [EVAL_IDS] logits 短写 @pos %u -- aborting\n", p0 + t);
                    exit(1);
                }
            }
            fflush(lf);
        }
        fprintf(stderr, "ds4: [EVAL_IDS] %u/%u\n", p0 + nt, n);
    }
    if (lf && fclose(lf) != 0) {
        fprintf(stderr, "ds4: [EVAL_IDS] 关闭 %s 失败 -- aborting\n", lp);
        exit(1);
    }
    fprintf(stderr, "ds4: [EVAL_IDS] 完成 S=%u vocab=%u%s%s\n", n, vocab,
            lp && lp[0] ? " logits已写" : "",
            getenv("DS4_EVAL_HDUMP") ? " hidden已写" : "");
    ds4_session_free(s);
    free(hc); free(lg); free(ids);
    exit(0);
#endif
}

