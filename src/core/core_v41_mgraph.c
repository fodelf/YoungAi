/* core_v41_mgraph.c — 合批整步的 CUDA graph(2026-09-30 傍晚, batch.md 第二期; nsys 定罪: 缓存段的核 11~13 µs 一发, 主机 ~5 µs 发一发,
 * 三路分流也喂不饱 GPU; 主流 835 发/步 ≈ 4 ms 间隙)。
 *
 * 与单请求整步图(core_decode_graph.c)同一套原语与口径: 各路"设备位置"(st->graph=1, 位置从批态 pos 的视图读, grid 按位置桶上限开), 输入走 pinned 槽
 * + 零拷贝小核, 末尾各路 argmax/采样核落 am, 零拷贝读回; engram 各路自己的取行任务(自旋等 flag)。不同的是键: 一张图 = (成员集, 各路行数, 各路位置桶,
 * 暂存代号, 各路索引草稿代号, 采样面); 缓存 MG_SLOTS 张按最久没用淘汰 —— 投机每轮各路行数会变, 组合多, 钉 --dspark-verify K 时只有 2^N 种。
 * 能捕的前提: 各路这个行数直发暖过(懒分配全建好; 验证批的快照缓冲由直发那一轮建)。捕获失败 = 这一轮走直发, 不关进程级开关。
 * 输出与直发逐字节同(同一批核换个发法; 门在 fable5 09-30)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU

#define MG_SLOTS 8
#define MG_BUCKET 1024u   /* 位置桶宽(与 core_decode_graph.c 同): 桶内各核 grid 上限不变, 任一路跨桶就换图 */

typedef struct {
    void *exec;
    uint32_t nm, R;
    ds4_v41_state *mem[DS4_V41_GEMV_MAX_TOK];
    uint64_t uid[DS4_V41_GEMV_MAX_TOK];   /* 成员状态的序号(core_v41.h uid): 指针会被关掉再开的状态复用, 序号不会 */
    uint32_t nr[DS4_V41_GEMV_MAX_TOK], lo[DS4_V41_GEMV_MAX_TOK], cap[DS4_V41_GEMV_MAX_TOK], idx_gen[DS4_V41_GEMV_MAX_TOK];
    int dev_sample[DS4_V41_GEMV_MAX_TOK]; const ds4_gpu_tensor *spec_q[DS4_V41_GEMV_MAX_TOK];
    uint64_t gen, used;
} mg_inst;
typedef struct {
    mg_inst g[MG_SLOTS];
    int32_t *tokv, *posv; float *onehot; int32_t *next;   /* pinned: 输入槽 / 各行 argmax 落点(每行 16 B) */
    ds4_gpu_tensor *amv[DS4_V41_GEMV_MAX_TOK];             /* am 的逐行视图(argmax 核只写自己那块) */
    uint64_t tick; uint32_t launches, captures, misses;
} v41_mgraph;

void v41_mgraph_free(ds4_v41_batch *b) {
    v41_mgraph *g = (v41_mgraph *)b->mg;
    if (!g) return;
    for (int i = 0; i < MG_SLOTS; i++) if (g->g[i].exec) ds4_gpu_decode_graph_free(g->g[i].exec);
    for (uint32_t i = 0; i < DS4_V41_GEMV_MAX_TOK; i++) if (g->amv[i]) ds4_gpu_tensor_free(g->amv[i]);
    ds4_gpu_host_free(g->tokv); ds4_gpu_host_free(g->posv); ds4_gpu_host_free(g->onehot); ds4_gpu_host_free(g->next);
    if (g->launches) fprintf(stderr, "ds4: [mgraph] 合批走图 %u 步, 捕获 %u 次, 键不中 %u 次\n", g->launches, g->captures, g->misses);
    free(g); b->mg = NULL;
}

static bool mg_alloc(ds4_v41_batch *b) {
    if (b->mg) return true;
    v41_mgraph *g = xmalloc(sizeof *g); memset(g, 0, sizeof *g);
    const uint32_t cap = b->cap;
    g->tokv = ds4_gpu_host_alloc((uint64_t)cap * 4); g->posv = ds4_gpu_host_alloc((uint64_t)cap * 4);
    g->onehot = ds4_gpu_host_alloc((uint64_t)cap * DS4_N_HC * 4); g->next = ds4_gpu_host_alloc((uint64_t)cap * 16u);
    b->mg = g;
    if (!g->tokv || !g->posv || !g->onehot || !g->next) { v41_mgraph_free(b); return false; }
    for (uint32_t i = 0; i < cap * DS4_N_HC; i++) g->onehot[i] = (i % DS4_N_HC) == 0 ? 1.0f : 0.0f;
    for (uint32_t i = 0; i < cap; i++) { g->amv[i] = ds4_gpu_tensor_view(b->am, (uint64_t)i * 16u, 16u); if (!g->amv[i]) { v41_mgraph_free(b); return false; } }
    return true;
}

static uint32_t mg_bucket_cap(const ds4_v41_state *st, uint32_t n) {
    uint32_t cap = (st->n_past / MG_BUCKET + 1u) * MG_BUCKET - 1u;   /* 本批首行位置的上限, 末行到 cap+n−1 */
    if (cap > st->ctx - n) cap = st->ctx - n;
    return cap;
}

static mg_inst *mg_find(v41_mgraph *g, ds4_v41_state **m, const uint32_t *nr, uint32_t nm) {
    const uint64_t gen = ds4_gpu_v41_scratch_generation();
    for (int s = 0; s < MG_SLOTS; s++) {
        mg_inst *in = &g->g[s];
        if (!in->exec || in->nm != nm || in->gen != gen) continue;
        bool hit = true;
        for (uint32_t i = 0; hit && i < nm; i++)
            if (in->mem[i] != m[i] || in->uid[i] != m[i]->uid || in->nr[i] != nr[i] || m[i]->n_past < in->lo[i] || m[i]->n_past + nr[i] - 1u > in->cap[i] ||
                in->idx_gen[i] != m[i]->iscap_gen || in->dev_sample[i] != m[i]->dev_sample || in->spec_q[i] != m[i]->spec_q) hit = false;
        if (hit) return in;
    }
    return NULL;
}

bool v41_multi_graph_ready(ds4_v41_batch *b, ds4_v41_state **m, const uint32_t *nr, uint32_t nm) {
    if (!g_ds4_v41_graph || g_ds4_v41_prof || g_ds4_v41_hook || !nm || nm > b->cap) return false;
    for (uint32_t i = 0; i < nm; i++) {
        const ds4_v41_state *st = m[i];
        if (st->draft || st->dump_prefix || nr[i] == 0 || nr[i] >= DS4_MTP_MAX_BLOCK + 2u) return false;
        if (st->n_direct_n[nr[i]] == 0) return false;                  /* 这个行数没直发暖过: 懒分配/核属性还没建 */
        if (nr[i] > 1u && !st->snap_win[0]) return false;              /* 验证批的快照缓冲由直发那一轮 v41_spec_snapshot 建 */
        if (st->no_engram && g_ds4_v41.n_engram) return false;
    }
    return true;
}

/* 捕获: 各路先按桶上限把三种暂存长够(捕获态不许分配), 挂视图、置 graph=1, 录一整步 */
static mg_inst *mg_capture(ds4_engine *e, ds4_v41_batch *b, ds4_v41_state **m, const uint32_t *r0, const uint32_t *nr, uint32_t nm, uint32_t R) {
    v41_mgraph *g = (v41_mgraph *)b->mg;
    ds4_v41_state *B = &b->rows;
    int slot = 0;
    for (int s = 0; s < MG_SLOTS; s++) { if (!g->g[s].exec) { slot = s; break; } if (g->g[s].used < g->g[slot].used) slot = s; }
    mg_inst *in = &g->g[slot];
    if (in->exec) { ds4_gpu_decode_graph_free(in->exec); in->exec = NULL; }
    /* ①各路的暂存先长够(索引草稿按桶上限的组数; 候选块 / 注意力分段暂存按道 —— 在对应的道上 prepare 才长到那一道的那块) */
    const int lanes = (g_ds4_v41_lanes && nm > 1u) ? ds4_gpu_lanes_fork((int)nm) : 0;
    bool ok = true;
    for (uint32_t i = 0; ok && i < nm; i++) {
        ds4_v41_state *st = m[i];
        in->lo[i] = st->n_past; in->cap[i] = mg_bucket_cap(st, nr[i]);
        uint32_t ng_max = 0;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (!g_ds4_v41.is_index_source[il]) continue;
            const uint32_t r = ds4_layer_compress_ratio(il);
            if (r && (in->cap[i] + nr[i]) / r > ng_max) ng_max = (in->cap[i] + nr[i]) / r;
        }
        if (ng_max && !v41_index_scratch_prepare(st, ng_max, nr[i], nr[i])) ok = false;
        if (lanes && !ds4_gpu_lane_begin((int)i)) ok = false;
        if (ok && !ds4_gpu_v41_attn_scratch_prepare(nr[i], DS4_N_HEAD, DS4_N_HEAD_DIM)) ok = false;
        if (ok && g_ds4_v41.candidate_source_layer >= 0 && g_ds4_v41.candidate_block_size > 0) {
            const uint32_t cr = ds4_layer_compress_ratio((uint32_t)g_ds4_v41.candidate_source_layer), cbs = (uint32_t)g_ds4_v41.candidate_block_size;
            const uint32_t cng = cr ? (in->cap[i] + nr[i]) / cr : 0u;
            if (cng && !ds4_gpu_v41_candidate_scratch_prepare(nr[i], (cng + cbs - 1u) / cbs)) ok = false;
        }
        if (ok && !ds4_gpu_v41_indexer_scratch_prepare(nr[i], DS4_N_INDEXER_HEAD)) ok = false;   /* --idx-mma 的 q 整数尾数暂存, 按道分(开关关着 = no-op) */
        if (lanes) ds4_gpu_lane_end();
    }
    if (lanes) (void)ds4_gpu_lanes_join();
    if (!ok) return NULL;
    /* ②挂视图、各路进设备位置口径, 录图 */
    v41_rowview rv[DS4_V41_GEMV_MAX_TOK]; memset(rv, 0, sizeof rv);
    for (uint32_t i = 0; ok && i < nm; i++) {
        if (!v41_attach(m[i], B, r0[i], nr[i], &rv[i])) ok = false;
        else { m[i]->graph = 1; m[i]->graph_pos_lo = in->lo[i]; m[i]->graph_pos_cap = in->cap[i]; m[i]->egraph_uploaded = 0; }
    }
    void *exec = NULL;
    if (ok && ds4_gpu_decode_graph_capture_begin()) {
        ok = ds4_gpu_tensor_write_zerocopy(B->tok, 0, g->tokv, (uint64_t)R * 4) && ds4_gpu_tensor_write_zerocopy(B->pos, 0, g->posv, (uint64_t)R * 4) &&
             ds4_gpu_tensor_write_zerocopy(B->pre_mix, 0, g->onehot, (uint64_t)R * DS4_N_HC * 4);
        if (ok) ok = v41_multi_body(e, b, m, r0, nr, nm, R);
        /* 末尾各路出 token: 采样开 = 一发采样核出 nr 行(拒绝采样用各自的塔 logits); 否则逐行 argmax。都落 am 的对应行, 零拷贝整块读回 */
        for (uint32_t i = 0; ok && i < nm; i++) {
            ds4_v41_state *st = m[i];
            if (st->dev_sample) {
                ds4_gpu_tensor *amr = ds4_gpu_tensor_view(b->am, (uint64_t)r0[i] * 16u, (uint64_t)nr[i] * 16u);
                if (!amr) { ok = false; break; }
                ok = ds4_gpu_v41_sample_tensor(amr, B->logits, r0[i], nr[i], DS4_N_VOCAB, B->pos, B->tok, &st->samp, nr[i] > 1u ? st->spec_q : NULL) != 0;
                ds4_gpu_tensor_free(amr);
            } else for (uint32_t j = 0; ok && j < nr[i]; j++) ok = ds4_gpu_v41_argmax_tensor(g->amv[r0[i] + j], B->logits, r0[i] + j, DS4_N_VOCAB) != 0;
        }
        if (ok) ok = ds4_gpu_tensor_read_zerocopy(g->next, b->am, 0, (uint64_t)R * 16u) != 0;
        exec = ds4_gpu_decode_graph_capture_end();   /* 不管 ok 与否都要收捕获, 否则流一直停在捕获态 */
    } else ok = false;
    for (uint32_t i = 0; i < nm; i++) { m[i]->graph = 0; v41_detach(m[i], &rv[i]); }
    if (!ok || !exec) { if (exec) ds4_gpu_decode_graph_free(exec); return NULL; }
    in->exec = exec; in->nm = nm; in->R = R; in->gen = ds4_gpu_v41_scratch_generation();
    for (uint32_t i = 0; i < nm; i++) { in->mem[i] = m[i]; in->uid[i] = m[i]->uid; in->nr[i] = nr[i]; in->idx_gen[i] = m[i]->iscap_gen; in->dev_sample[i] = m[i]->dev_sample; in->spec_q[i] = m[i]->spec_q; }
    g->captures++;
    fprintf(stderr, "ds4: [mgraph] %u 路 %u 行的图: 槽 %d, 位置桶", nm, R, slot);
    for (uint32_t i = 0; i < nm; i++) fprintf(stderr, " [%u,%u]", in->lo[i], in->cap[i]);
    fputc('\n', stderr);
    return in;
}

bool v41_multi_graph_round(ds4_engine *e, ds4_v41_batch *b, ds4_v41_state **m, const int32_t *tok, const uint32_t *nr, uint32_t nm, int32_t *want) {
    if (!mg_alloc(b)) return false;
    v41_mgraph *g = (v41_mgraph *)b->mg;
    uint32_t r0[DS4_V41_GEMV_MAX_TOK], R = 0;
    for (uint32_t i = 0; i < nm; i++) { r0[i] = R; R += nr[i]; }
    if (!R || R > b->cap) return false;
    for (uint32_t i = 0; i < nm; i++) if (m[i]->n_past + nr[i] > m[i]->ctx || m[i]->hc) return false;
    mg_inst *in = mg_find(g, m, nr, nm);
    if (!in) {
        g->misses++;
        if (!(in = mg_capture(e, b, m, r0, nr, nm, R))) { fprintf(stderr, "ds4: ★[mgraph] 捕获失败, 这一轮走直发★\n"); return false; }
    }
    in->used = ++g->tick;
    /* 起手(与 v41_multi_step / dg_begin_step 同一套): 各路的 n/pos0/计数、hist 先落(engram 哈希要回看它), 槽先落内存再发图 */
    for (uint32_t i = 0; i < nm; i++) {
        ds4_v41_state *st = m[i];
        st->n = nr[i]; st->pos0 = st->n_past; st->idx_owner = -1; st->cand_owner = -1; st->idx_topk = 0; st->idx_ratio = 0;
        st->stop_early = 0; st->ced_skip = 0; st->egraph_err = 0; st->mainh_wrote = st->mainh ? (nr[i] < st->mainh_cap ? nr[i] : st->mainh_cap) : 0;
        for (uint32_t j = 0; j < nr[i]; j++) { st->hist[st->pos0 + j] = tok[r0[i] + j]; g->tokv[r0[i] + j] = tok[r0[i] + j]; g->posv[r0[i] + j] = (int32_t)(st->pos0 + j); }
        if (nr[i] > 1u) {   /* 验证批的快照记账(字节在图里: 窗口环按层 / 压缩器余行在追加核里), 与 v41_graph_batch_launch 同式 */
            st->snap_n = nr[i]; st->snap_past = st->pos0; st->snap_on = 1u;
            for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                if (!g_ds4_v41.is_kv_source[il]) continue;
                const uint32_t ratio = ds4_layer_compress_ratio(il);
                if (ratio > 1u) st->snap_cpend[il] = st->pos0 % ratio;
            }
        }
    }
    __sync_synchronize();
    for (uint32_t i = 0; i < nm; i++) {
        if (m[i]->no_engram) continue;
        if (!v41_engram_prefetch(e, m[i]) || !v41_engram_graph_arm(m[i])) return false;   /* 取行先提交; 本步序号进 want 槽 */
    }
    if (!ds4_gpu_decode_graph_launch(in->exec)) return false;
    g->launches++;
    bool eg_ok = true;
    for (uint32_t i = 0; i < nm; i++) if (!m[i]->no_engram && !v41_engram_graph_serve(m[i])) eg_ok = false;   /* 图跑着时主机收各路的 pread、逐层置位 */
    if (!ds4_gpu_synchronize()) return false;
    __sync_synchronize();
    for (uint32_t i = 0; i < nm; i++) if (!m[i]->no_engram && v41_engram_graph_err(m[i])) eg_ok = false;
    if (!eg_ok) { fprintf(stderr, "ds4: [mgraph] engram 取行失败\n"); return false; }
    for (uint32_t i = 0; i < nm; i++) {
        v41_sample_pick(g->next + 4u * r0[i], tok + r0[i], nr[i], m[i]->dev_sample, want + r0[i]);
        /* 位置按闭式推进(图里的核按设备位置写缓存, 主机只记同一个数; 与 core_decode_graph.c dg_advance 同式) */
        ds4_v41_state *st = m[i];
        v41_multi_advance(st, nr[i]);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (!g_ds4_v41.is_kv_source[il]) continue;
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (!ratio) continue;
            st->ng_src[il] = st->n_past / ratio;
            st->cpend[il] = ratio > 1u ? st->n_past % ratio : 0u;
        }
    }
    return true;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_mgraph_nonempty_tu;
