/* core_v41_gen.c — 合批生成驱动(2026-10-09): 一串作业(提示 + 采样面 + 上限)走并发请求态出口(core_v41_req.c), 至多 bcap 路一起解码。
 *
 * 两个调用方, 一份代码:
 *   训练器探针 / 奖励回路采样(core_ptrain_probe.c): refill = 0 —— 整组跑完才开下一组。组的划分与 10-06 合批改造后一字不差
 *     (合批的行数影响稠密段舍入, 组一变采样文件就不再逐字节同, 门 = ptgate 的 probe/sample 文件 diff);
 *   出题(--gen-jobs, cli_gen_jobs.c, kdgen 用): refill = 1 —— 哪一路写完就立刻补下一个作业。0009 的出题生成长度 p10/p90 = 463/2500 token,
 *     整组等最长那一路, 尾巴上大半行空转。
 * 同提示的相邻作业只预填第一份, 其余挂分身(ds4_v41_req_fork, 首个 token 按各自种子取, KV 收缩后深拷)。
 * 一次只有一个整状态(预填块行缓冲)活着: 开一路 → 挂它的分身 → 预填完收缩 → 再开下一路。
 * 每个作业写完(EOS / 到上限 / 上下文满)回调一次 done; 失败返回 false, 已回调的作业照旧有效。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU

typedef struct {
    struct ds4_v41_req *r;
    uint32_t job, made;
    int live, eos;
    char *buf; size_t n, cap;
} v41_gen_lane;

/* 一个 token 进 lane 的文本; 返回 1 = 这一路收口(EOS 或到上限) */
static int v41_gen_emit(ds4_engine *e, v41_gen_lane *L, int tok, uint32_t max_tok) {
    L->made++;
    if (tok == ds4_token_eos(e)) { L->eos = 1; return 1; }
    size_t len = 0;
    char *t = ds4_token_text(e, tok, &len);
    if (t) {
        if (L->n + len + 1 > L->cap) { L->cap = (L->n + len + 1) * 2; L->buf = realloc(L->buf, L->cap); }
        memcpy(L->buf + L->n, t, len); L->n += len; L->buf[L->n] = 0;
        free(t);
    }
    return L->made >= max_tok;
}

static void v41_gen_finish(v41_gen_lane *L, ds4_v41_gen_done_fn done, void *ud) {
    if (done) done(ud, L->job, L->buf ? L->buf : "", L->n, L->eos, L->made);
    free(L->buf);
    ds4_v41_req_close(L->r);
    memset(L, 0, sizeof *L);
}

/* 收口(不活 / 上下文满)的 lane 交卷; 返回交了几份 */
static uint32_t v41_gen_reap(v41_gen_lane *lane, uint32_t cap, ds4_v41_gen_done_fn done, void *ud) {
    uint32_t k = 0;
    for (uint32_t i = 0; i < cap; i++)
        if (lane[i].r && (!lane[i].live || ds4_v41_req_room(lane[i].r) <= 0)) { v41_gen_finish(&lane[i], done, ud); k++; }
    return k;
}

static bool v41_gen_same_prompt(const ds4_v41_gen_job *a, const ds4_v41_gen_job *b) {
    return a->len == b->len && a->max_tok == b->max_tok && !memcmp(a->ids, b->ids, (size_t)a->len * sizeof(int));
}

/* 把空着的 lane 填上: 从 *next 起按作业序取, 每个新提示开一个请求态, 紧跟着的同提示作业挂分身, 然后预填 */
static bool v41_gen_admit(ds4_engine *e, const ds4_v41_gen_job *jobs, uint32_t n, uint32_t *next, v41_gen_lane *lane, uint32_t cap) {
    for (;;) {
        uint32_t slot = cap;
        for (uint32_t i = 0; i < cap; i++) if (!lane[i].r) { slot = i; break; }
        if (slot == cap || *next >= n) return true;
        const uint32_t j0 = (*next)++;
        lane[slot] = (v41_gen_lane){ .job = j0 };
        lane[slot].r = ds4_v41_req_open(e, jobs[j0].ids, (int)jobs[j0].len, (int)jobs[j0].max_tok, &jobs[j0].sp);
        if (!lane[slot].r) { fprintf(stderr, "ds4: [合批生成] 第 %u 个作业请求态开不出来\n", j0); return false; }
        uint32_t members[DS4_V41_MULTI_MAX], nm = 0;
        members[nm++] = slot;
        for (uint32_t i = 0; i < cap && *next < n; i++) {
            if (lane[i].r || !v41_gen_same_prompt(&jobs[*next], &jobs[j0])) continue;
            const uint32_t j = (*next)++;
            lane[i] = (v41_gen_lane){ .job = j };
            lane[i].r = ds4_v41_req_fork(lane[slot].r, &jobs[j].sp);
            if (!lane[i].r) { fprintf(stderr, "ds4: [合批生成] 第 %u 个作业分身挂不上\n", j); return false; }
            members[nm++] = i;
        }
        int prc;
        while ((prc = ds4_v41_req_prefill_step(lane[slot].r)) == 0) {}
        if (prc < 0) { fprintf(stderr, "ds4: [合批生成] 第 %u 个作业预填失败\n", j0); return false; }
        for (uint32_t k = 0; k < nm; k++) {   /* 首个 token 预填就出了 */
            v41_gen_lane *L = &lane[members[k]];
            L->live = !v41_gen_emit(e, L, ds4_v41_req_next(L->r), jobs[L->job].max_tok);
        }
    }
}

bool ds4_v41_gen_run(ds4_engine *e, const ds4_v41_gen_job *jobs, uint32_t n, uint32_t bcap, int refill, ds4_v41_gen_done_fn done, void *ud) {
    if (bcap < 1u) bcap = 1u;
    if (bcap > DS4_V41_MULTI_MAX) bcap = DS4_V41_MULTI_MAX;
    const uint32_t cap = n < bcap ? n : bcap;
    if (!cap) return true;
    struct ds4_v41_batch *b = ds4_v41_batch_open(e, (int)cap);
    if (!b) { fprintf(stderr, "ds4: [合批生成] 批态开不出来(%u 行)\n", cap); return false; }
    v41_gen_lane lane[DS4_V41_MULTI_MAX];
    memset(lane, 0, sizeof lane);
    uint32_t next = 0;
    bool ok = true;
    for (;;) {
        if (refill) {   /* 收口的先交卷腾位, 再补; 补进来的首 token 就收口的再交卷再补, 直到稳定 */
            while (ok) { v41_gen_reap(lane, cap, done, ud); ok = v41_gen_admit(e, jobs, n, &next, lane, cap); if (!ok || !v41_gen_reap(lane, cap, done, ud)) break; }
        } else {
            uint32_t busy = 0;
            for (uint32_t i = 0; i < cap; i++) if (lane[i].r) busy++;
            if (!busy) ok = v41_gen_admit(e, jobs, n, &next, lane, cap);
        }
        if (!ok) break;
        struct ds4_v41_req *act[DS4_V41_MULTI_MAX]; uint32_t who[DS4_V41_MULTI_MAX], na = 0;
        for (uint32_t i = 0; i < cap; i++)
            if (lane[i].r && lane[i].live && ds4_v41_req_room(lane[i].r) > 0) { act[na] = lane[i].r; who[na++] = i; }
        if (!na) {   /* 整组模式: 这一组都收口了, 一起交卷再开下一组 */
            for (uint32_t i = 0; i < cap; i++) if (lane[i].r) v41_gen_finish(&lane[i], done, ud);
            if (next >= n) break;
            continue;
        }
        if (ds4_v41_multi_step(b, act, (int)na) != 0) { fprintf(stderr, "ds4: [合批生成] 合批一步失败(%u 路)\n", na); ok = false; break; }
        for (uint32_t k = 0; k < na; k++) {
            v41_gen_lane *L = &lane[who[k]];
            int o[DS4_MTP_MAX_BLOCK + 2u];
            const int m = ds4_v41_req_take(act[k], o, (int)(DS4_MTP_MAX_BLOCK + 2u));
            for (int j = 0; j < m && L->live; j++) if (v41_gen_emit(e, L, o[j], jobs[L->job].max_tok)) L->live = 0;
        }
    }
    for (uint32_t i = 0; i < cap; i++) if (lane[i].r) { free(lane[i].buf); ds4_v41_req_close(lane[i].r); }   /* 失败时没写完的: 不回调 */
    ds4_v41_batch_close(b);
    return ok;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_gen_nonempty_tu;
