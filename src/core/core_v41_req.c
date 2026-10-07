/* core_v41_req.c — 并发请求态的公共出口(2026-09-30, batch.md §3.2; 服务端调度器 server_sched_v41.c 与 --multi-probe 用)。
 *
 * 一条请求 = 一个请求态(KV + 采样面 + 取 token 用的落点/惩罚缓冲 + 草稿器)。生命周期:
 *   open(整状态, 预填块 2048 行) → prefill_step 一块一块(调度器在块间去发心跳 / 探客户端 / 让解码先跑)
 *   → 最后一块出首个 token, 状态收缩(只剩 KV + 几十 MB) → 之后每轮由 ds4_v41_multi_round 与其他请求合批 → close。
 * 采样面按请求(psamp), 不再碰进程全局 g_decode_sampling —— 单 worker 时代"按请求覆写全局"在合批下会串台。
 * 预填 / 取 token / 快照回滚与单请求路 generate_argmax 同一份代码, 所以温 0 下逐字节同是构造上的, 门只是复核。
 * ★投机进合批(batch.md 第二期)★: 每路各出草稿(v41_draft_step, 各自的三塔窗口/图), 验证行 1+k 拼进同一次前向(Σ行 ≤ 批态 cap, 超了从 k 最大的
 * 路往下削), 各路按自己的行取 token、接受最长前缀、各自回滚; 一轮吐出的 token 排进 outq, 调用方按序 emit。 */
#include "core_internal.h"
#include <time.h>
#ifndef DS4_NO_GPU

struct ds4_v41_req {
    ds4_engine *e;
    ds4_v41_state st;
    ds4_gpu_tensor *am;        /* 设备槽: 每行 16 B(采样核 4 个 int; argmax 只用第 0 个) */
    float *rowbuf;             /* 惩罚路: 读回整行主机罚完采; NULL = 设备路 */
    v41_hist hist;             /* 生成段 token 史(惩罚路才记) */
    uint64_t rng;
    ds4_decode_sampling sp;    /* 本请求的采样面(st.psamp 指向它) */
    const int32_t *prompt;
    uint32_t np, c0, cap;
    int32_t next;              /* 还没进模型的下一个 token: 预填完 = 首个生成 token; 每轮合批后 = 新 token */
    int prefilled;
    /* 投机: 草稿器 + 调度账(与 generate_argmax 同一套); spec = 0 时每轮就是纯解码 1 行 */
    int spec;
    ds4_v41_draft dr;
    v41_sched cal;
    int32_t outq[DS4_MTP_MAX_BLOCK + 2u]; int nout;   /* 这一轮吐出的 token(接受的草稿 + 模型自己的那个), 调用方 take 走 */
    uint32_t rounds, acc_sum, off_sum;   /* 投机账: 轮数 / 接受的草稿位 / 出过的草稿位(服务监控页算接受率用) */
};

struct ds4_v41_batch *ds4_v41_batch_open(ds4_engine *e, int cap) {
    if (!e || cap < 1 || !ds4_engine_is_v41(e) || !e->metal_ready) return NULL;
    ds4_v41_batch *b = xmalloc(sizeof *b);
    if (!v41_batch_alloc(b, (uint32_t)cap)) { free(b); return NULL; }
    b->e = e;
    return b;
}

void ds4_v41_batch_close(struct ds4_v41_batch *b) {
    if (!b) return;
    v41_batch_free(b);
    free(b);
}

struct ds4_v41_req *ds4_v41_req_open(ds4_engine *e, const int *prompt, int n_prompt, int n_predict, const ds4_decode_sampling *sp) {
    if (!e || !prompt || n_prompt < 1 || !ds4_engine_is_v41(e) || !e->metal_ready) return NULL;
    const uint32_t np = (uint32_t)n_prompt;
    uint32_t ctx = g_ds4_v41.ctx;   /* 上下文只从模型元数据来(用户 2026-09-22) */
    if (ctx == 0 || np + 1u > ctx) { fprintf(stderr, "ds4: 提示 %u token 超过上下文 %u\n", np, ctx); return NULL; }
    /* 只按这一趟真正用得到的位置分配(与 generate_argmax 同一条账): np + 上限 + 投机余量; ctx 仍是硬边界 */
    const uint64_t need = (uint64_t)np + (uint64_t)(n_predict > 0 ? n_predict : 0) + DS4_MTP_MAX_BLOCK + 2u;
    if ((uint64_t)ctx > need) ctx = (uint32_t)need;
    const uint32_t ck = g_ds4_v41_chunk > 0 ? (uint32_t)g_ds4_v41_chunk : DS4_V41_CHUNK;
    struct ds4_v41_req *r = xmalloc(sizeof *r);
    memset(r, 0, sizeof *r);
    r->e = e; r->prompt = (const int32_t *)prompt; r->np = np; r->cap = ck < np ? ck : np;
    r->sp = sp ? *sp : g_decode_sampling;
    if (!v41_state_alloc(&r->st, r->cap, ctx, DS4_MTP_MAX_BLOCK + 2u)) { free(r); return NULL; }
    r->st.head_last_only = 1;   /* 预填块只算末位 logits(core_v41.h) */
    r->st.psamp = &r->sp;
    r->am = ds4_gpu_tensor_alloc((uint64_t)(DS4_MTP_MAX_BLOCK + 2u) * 16u);
    /* 三条取 token 的路(core_v41_sample.c): 温度 > 0 且无惩罚 = 设备采样核; 任一惩罚非零 = 主机惩罚路; 否则设备 argmax */
    const bool penal = r->sp.dry_multiplier > 0.f || r->sp.freq_penalty != 0.f || r->sp.presence_penalty != 0.f;
    const bool dev_sample = r->sp.temperature > 0.f && !penal;
    if (penal) {
        r->rowbuf = xmalloc((size_t)DS4_N_VOCAB * 4u);
        r->hist.cap = (uint32_t)(n_predict > 0 ? n_predict : 0) + 2u;
        r->hist.tok = xmalloc((size_t)r->hist.cap * sizeof(int32_t));
        if (r->sp.dry_multiplier > 0.f) r->hist.brk = ds4_decode_breakers(e, DS4_N_VOCAB);
    }
    r->rng = r->sp.seed ? r->sp.seed : ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());   /* 与 CLI 同一条规则 */
    r->st.dev_sample = dev_sample ? 1 : 0;
    r->st.samp = (ds4_gpu_sample_params){ .temperature = r->sp.temperature, .top_p = r->sp.top_p, .min_p = r->sp.min_p,
                                          .top_k = r->sp.top_k, .seed = r->rng };
    /* 投机: 惩罚路不接(与 generate_argmax 同: 惩罚要按 token 史改 logits); 没三塔的 GGUF dr.ready=0 就是纯解码 */
    r->spec = (g_ds4_v41_dspark && !penal && v41_draft_alloc(e, &r->dr)) ? 1 : 0;
    if (r->spec && dev_sample) { r->dr.dev_sample = 1; r->dr.samp = r->st.samp; r->dr.samp.stream = 1u; r->st.spec_q = r->dr.st.logits; }
    if (!r->am) { ds4_v41_req_close(r); return NULL; }
    return r;
}

void ds4_v41_req_close(struct ds4_v41_req *r) {
    if (!r) return;
    if (r->spec) v41_draft_free(&r->dr);
    if (r->am) ds4_gpu_tensor_free(r->am);
    v41_state_free(&r->st);
    free(r->rowbuf); free(r->hist.tok); free((void *)r->hist.brk);
    free(r);
}

int ds4_v41_req_prefill_step(struct ds4_v41_req *r) {
    if (!r) return -1;
    if (r->prefilled) return 1;
    if (!v41_prefill_chunk(r->e, &r->st, r->prompt, r->np, &r->c0, r->cap)) return -1;
    if (r->c0 < r->np) return 0;
    /* 最后一块: 末位 logits → 首个 token(设备 argmax / 采样核 / 主机惩罚路), 然后收缩成解码态(投机的验证批要 1+块长 行) */
    if (!v41_next_token(&r->st, r->am, r->st.last_logit_row, r->rowbuf, &r->rng, &r->hist, &r->next)) return -1;
    if (!v41_state_shrink(&r->st, r->spec ? DS4_MTP_MAX_BLOCK + 1u : 1u)) return -1;
    r->prefilled = 1;
    return 1;
}

void ds4_v41_req_progress(const struct ds4_v41_req *r, int *c0, int *np) { if (c0) *c0 = (int)r->c0; if (np) *np = (int)r->np; }
int ds4_v41_req_next(const struct ds4_v41_req *r) { return r->next; }
int ds4_v41_req_pos(const struct ds4_v41_req *r) { return (int)r->st.n_past; }
int ds4_v41_req_room(const struct ds4_v41_req *r) { return r->st.ctx > r->st.n_past ? (int)(r->st.ctx - r->st.n_past) : 0; }
int ds4_v41_req_take(struct ds4_v41_req *r, int *out, int max) {
    int n = r->nout < max ? r->nout : max;
    for (int i = 0; i < n; i++) out[i] = r->outq[i];
    r->nout = 0;
    return n;
}

int ds4_v41_multi_step(struct ds4_v41_batch *b, struct ds4_v41_req **r, int n) {
    ds4_v41_state *m[DS4_V41_GEMV_MAX_TOK];
    int32_t tok[DS4_V41_GEMV_MAX_TOK];
    if (!b || !r || n < 1 || n > (int)b->cap) return 1;
    for (int i = 0; i < n; i++) {
        if (!r[i] || !r[i]->prefilled) { fprintf(stderr, "ds4: V4.1 合批第 %d 路还没预填完\n", i); return 1; }
        m[i] = &r[i]->st; tok[i] = r[i]->next;
    }
    uint32_t ones[DS4_V41_GEMV_MAX_TOK]; int32_t wants[DS4_V41_GEMV_MAX_TOK];
    for (int i = 0; i < n; i++) ones[i] = 1u;
    const bool walked = v41_multi_graph_ready(b, m, ones, (uint32_t)n) && v41_multi_graph_round(b->e, b, m, tok, ones, (uint32_t)n, wants);
    if (!walked && !v41_multi_step(b->e, b, m, tok, NULL, (uint32_t)n)) return 1;
    for (int i = 0; i < n; i++) {
        if (walked) r[i]->next = wants[i];
        else if (!v41_multi_pick(b, m[i], (uint32_t)i, r[i]->rowbuf, &r[i]->rng, &r[i]->hist, &r[i]->next)) return 1;
        r[i]->outq[0] = r[i]->next; r[i]->nout = 1;
    }
    return 0;
}

/* 一轮(投机): 各路出草稿定 k → 验证行拼进一次前向 → 各路取 want、接受最长前缀、回滚 → outq = 接受的草稿 + 模型自己的那个(= 新 next)。
 * 温 0 下与单请求路同一串输出(接受条件就是"主模型自己也会选这个 token"); 采样下按设备核的拒绝采样(分布同, 硬币不同)。 */
int ds4_v41_multi_round(struct ds4_v41_batch *b, struct ds4_v41_req **r, int n) {
    ds4_v41_state *m[DS4_V41_GEMV_MAX_TOK];
    int32_t tok[DS4_V41_GEMV_MAX_TOK], batch[DS4_V41_GEMV_MAX_TOK][DS4_MTP_MAX_BLOCK + 1u];
    uint32_t rows[DS4_V41_GEMV_MAX_TOK], R = 0;
    int drafted[DS4_V41_GEMV_MAX_TOK];
    if (!b || !r || n < 1 || n > (int)b->cap) return 1;
    const double t0 = now_sec();
    /* ★投机只在同批 ≤ 2 路时开★(09-30 金融提示实测): 合批里骨架已被各路摊掉, 每多一行只剩专家字节 + 注意力(~7 ms), 而一行验证按接受直方图只值 0.43 个 token
     * (P≥1 0.75 / ≥2 0.40 / ≥3 0.26 / ≥4 0.11 / ≥5 0.05), 一行别的请求值 1 个 —— N=3: 投机总 41.8 t/s 对纯合批 65.7; N=1: 投机 46.1 对纯 30.4。
     * 2 路没量(估投机 ~55 对纯 ~49), 先按 ≤ 2 开; 量了再改。 */
    const int allow_spec = n <= 2;
    for (int i = 0; i < n; i++) {
        struct ds4_v41_req *q = r[i];
        if (!q || !q->prefilled) { fprintf(stderr, "ds4: V4.1 合批第 %d 路还没预填完\n", i); return 1; }
        m[i] = &q->st; batch[i][0] = q->next; rows[i] = 1u; drafted[i] = 0;
        if (!allow_spec || !q->spec || q->st.n_past + DS4_MTP_MAX_BLOCK + 1u > q->st.ctx) continue;
        const double td = now_sec();
        if (!v41_draft_step(q->e, &q->st, &q->dr, q->next, q->st.n_past - 1u)) continue;   /* 出不了草稿(料不齐)就纯解码这一轮 */
        drafted[i] = 1;
        if (!q->dr.last_warm) v41_sched_draft_cost(&q->cal, (now_sec() - td) * 1e3);
        float val = 0.f;
        uint32_t k = g_ds4_v41_verify_k ? g_ds4_v41_verify_k : v41_draft_pick_k(q->dr.host_conf, q->dr.block, &val, &q->cal);
        if (k > q->dr.block) k = q->dr.block;
        for (uint32_t j = 0; j < k; j++) batch[i][j + 1u] = q->dr.host_ids[j + 1u];
        rows[i] = 1u + k;
    }
    for (int i = 0; i < n; i++) R += rows[i];
    while (R > b->cap) {   /* 装不下: 从行最多的那一路往下削一位(它的第 k 位本来就最难兑现) */
        int big = 0; for (int i = 1; i < n; i++) if (rows[i] > rows[big]) big = i;
        rows[big]--; R--;
    }
    uint32_t off = 0;
    for (int i = 0; i < n; i++) { for (uint32_t j = 0; j < rows[i]; j++) tok[off + j] = batch[i][j]; off += rows[i]; }
    /* 整步走图(core_v41_mgraph.c): 键中或能捕就发图(快照字节在图里, 主机只记账); 否则直发(验证批先备份环格)。两条路输出逐字节同。 */
    int32_t wants[DS4_V41_GEMV_MAX_TOK];
    bool walked = v41_multi_graph_ready(b, m, rows, (uint32_t)n) && v41_multi_graph_round(b->e, b, m, tok, rows, (uint32_t)n, wants);
    if (!walked) {
        for (int i = 0; i < n; i++) if (rows[i] > 1u && !v41_spec_snapshot(m[i], rows[i])) return 1;   /* 验证批会盖环里的格, 先备份 */
        if (!v41_multi_step(b->e, b, m, tok, rows, (uint32_t)n)) return 1;
    }
    const double t1 = now_sec();
    off = 0;
    for (int i = 0; i < n; i++) {
        struct ds4_v41_req *q = r[i];
        int32_t want[DS4_MTP_MAX_BLOCK + 1u];
        if (walked) { for (uint32_t j = 0; j < rows[i]; j++) want[j] = wants[off + j]; }
        else if (rows[i] == 1u) {
            if (!v41_multi_pick(b, m[i], off, q->rowbuf, &q->rng, &q->hist, &want[0])) return 1;
        } else if (!v41_multi_pick_rows(b, m[i], off, rows[i], batch[i], want)) return 1;
        uint32_t a = 0;
        while (a + 1u < rows[i] && want[a] == batch[i][a + 1u]) a++;   /* 接受最长前缀(采样下 want 已按"接受 ⇒ 草稿 / 拒绝 ⇒ 残差"拼好) */
        /* ★每个出过草稿的轮都要观测, 包括 k=0 的轮★(core_v41.h 的校准死锁: 只在 k≥1 时观测, 首轮被拒 ⇒ ρ=0 ⇒ 永远 k=0; 09-30 spec2 N=1 实撞: 2 轮 0 接受后 62 轮全是白付的草稿) */
        if (drafted[i] && !g_ds4_v41_verify_k) v41_sched_observe(&q->cal, q->dr.host_conf, rows[i] > 1u ? (a >= 1u) : (want[0] == q->dr.host_ids[1]));
        if (rows[i] > 1u) {
            if (!v41_spec_rollback(m[i], 1u + a)) return 1;
            v41_sched_cost(&q->cal, rows[i], (t1 - t0) * 1e3);   /* 合批一步的墙钟是共享的: 每路记同一个数(偏保守) */
            q->rounds++; q->acc_sum += a; q->off_sum += rows[i] - 1u;
        } else if (q->spec) v41_sched_cost(&q->cal, 1u, (t1 - t0) * 1e3);
        q->nout = 0;
        for (uint32_t j = 0; j < a; j++) q->outq[q->nout++] = batch[i][j + 1u];   /* 白赚的那几位 */
        q->outq[q->nout++] = want[a];
        q->next = want[a];
        off += rows[i];
    }
    return 0;
}
void ds4_v41_req_spec_stats(const struct ds4_v41_req *r, int *rounds, int *offered, int *accepted) {
    if (rounds) *rounds = (int)r->rounds;
    if (offered) *offered = (int)r->off_sum;
    if (accepted) *accepted = (int)r->acc_sum;
}

/* ---- 内存账(服务端准入用): 与 core_v41_state.c 的分配式同源, 改那边的形状要改这里 ----
 * 预填期峰值 = 稠密 + 注意力侧行缓冲(块行) + KV(按上下文) + 窗口/余行 + engram 三块 + 索引草稿(2048 行 × 组数向上取 2 的幂 × 5 B);
 * 收缩后常驻 = KV + 几行的注意力侧 + 窗口环 + mainh。后端自己的暂存(GEMM bf16 等)不在账上, 准入留倍数余量(server_sched_v41.c)。 */
static uint64_t v41_bytes_rows(uint32_t cap, uint32_t logits_rows, int with_attn) {
    const uint64_t E = DS4_N_EMBD, HC = DS4_N_HC, FF = DS4_N_FF_EXP, NE = DS4_N_EXPERT, K = DS4_N_EXPERT_USED, mix = 2 * HC + HC * HC;
    const uint64_t HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, IH = DS4_N_INDEXER_HEAD, IK = DS4_N_INDEXER_HEAD_DIM;
    const uint64_t low = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    uint64_t per = 8 + 2 * HC * E * 4 + mix * 4 + 3 * HC * 4 + HC * HC * 4 + 3 * E * 4 + NE * 4 + K * 8 + E * 4 + 3 * FF * 4 + 2 * E * 4;
    if (with_attn) per += 4 + 2 * Q * 4 + 2 * NH * HD * 4 + 6 * HD * 4 + IK * 4 + IH * IK * 4 + IH * 4 + (uint64_t)DS4_N_INDEXER_TOP_K * 4 + low * 4;
    return per * cap + (uint64_t)logits_rows * DS4_N_VOCAB * 4 + E * 4;
}
static uint64_t v41_bytes_kv(uint32_t ctx, uint32_t cap) {
    const uint64_t HD = DS4_N_HEAD_DIM, SWA = DS4_N_SWA, E = DS4_N_EMBD;
    uint64_t b = (uint64_t)DS4_N_LAYER * (SWA + cap) * HD * 4 + (uint64_t)ctx * 4;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g_ds4_v41.is_kv_source[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ratio) continue;
        b += ((uint64_t)ctx / ratio + 2) * (DS4_V41_CKV_BYTES + DS4_V41_IDXK_BYTES);
        if (ratio > 1) b += 2 * (ratio + (uint64_t)cap) * HD * 4;
    }
    if (g_ds4_v41.mtp_block && g_ds4_v41.n_mtp_target) b += (uint64_t)DS4_N_SWA * g_ds4_v41.n_mtp_target * E * 4;
    return b;
}
uint64_t ds4_v41_req_prefill_bytes(int n_prompt) {
    const uint32_t np = n_prompt > 0 ? (uint32_t)n_prompt : 1u;
    const uint32_t ck = g_ds4_v41_chunk > 0 ? (uint32_t)g_ds4_v41_chunk : DS4_V41_CHUNK, cap = ck < np ? ck : np;
    uint64_t b = v41_bytes_rows(cap, DS4_MTP_MAX_BLOCK + 2u, 1) + v41_bytes_kv(g_ds4_v41.ctx, cap);
    if (g_ds4_v41.n_engram) {
        const ds4_v41_cfg *v = &g_ds4_v41;
        const uint64_t cols = (uint64_t)(v->engram_max_ngram - 1) * v->engram_heads, HD = v->engram_head_dim;
        b += (uint64_t)cap * (v->n_engram * cols * (HD + HD / 32) + cols * HD * 4 + (DS4_N_HC + 1) * DS4_N_EMBD * 4);
    }
    uint64_t ng = 1; while (ng < np) ng <<= 1;   /* 索引草稿按 2 的幂翻倍长(v41_index_scratch_prepare); L20 ratio 1 ⇒ 组数 = 位置数 */
    const uint64_t bs = g_ds4_v41.candidate_block_size > 0 ? (uint64_t)g_ds4_v41.candidate_block_size : 1u;
    b += (uint64_t)(cap / 32u + DS4_MTP_MAX_BLOCK + 2u) * ng * 4u + (uint64_t)cap * ((ng + bs - 1) / bs);   /* C1: 打分 Rb=n/32 行 f32 + 掩码整块行按块 */
    return b;
}
uint64_t ds4_v41_req_resident_bytes(void) {
    const uint32_t rc = DS4_MTP_MAX_BLOCK + 1u;
    return v41_bytes_rows(rc, 0u, 1) - v41_bytes_rows(rc, 0u, 0) + v41_bytes_kv(g_ds4_v41.ctx, rc);
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_req_nonempty_tu;
