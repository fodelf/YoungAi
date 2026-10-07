/* server_sched_v41.c — V4.1 并发调度器(2026-09-30, batch.md §3.2; 用户令"支持 CUDA 架构的并发")。
 *
 * --batch N(N ≥ 2)时替掉"单 worker 一次一个 job": 准入 → 一次一块预填(最老的那条) → 一步合批解码(所有预填完的)
 * → 时间片 → 收尾。协议 / 流式 / 工具标记 / 停止串 / 挂断探测全用 server_generate_v41.c 那套每请求回调(v41_gen),
 * 客户端看到的字节与单请求路一样; token 从引擎的请求态出口来(ds4_v41_api.h 的 ds4_v41_req_* / ds4_v41_multi_step)。
 * 时间片: 预填一块用时 T, 之后解码步累计 ≥ T 才轮到下一块 —— 预填与解码各半(batch.md 唯一的策略常量; 全让预填 ⇒ 解码路
 * 每 2 s 才出 1 个 token, 全让解码 ⇒ 后来的请求首 token 无限期)。没预填时解码连跑, 没解码时预填连跑。
 * 准入按内存: 开一条请求的预填要 MemAvailable ≥ 2 × 预填峰值字节(引擎按分配式算, ds4_v41_req_prefill_bytes; 倍数余量给后端暂存与
 * 索引草稿翻倍的瞬态); 不够就让它在队列里等(不 503), 已在跑的请求继续; 一个都没在跑还不够 = 这条请求本身装不下, 回错。
 * 没有的(如实): 投机(每路每步 1 个 token; 第二期); 跨请求前缀缓存; 上下文满按 length 收(与单请求路同)。 */
#include "server_internal.h"

typedef struct {
    job *j;
    v41_gen g;
    struct ds4_v41_req *r;
    int phase;                 /* 1 等预填(还没开请求态) / 2 预填中 / 3 解码中 */
    uint64_t order;            /* 到达序: 预填按它排队 */
} v41_lane;

/* Linux 的 MemAvailable(MB); 读不到 = -1 = 不拦(这条路只有 CUDA 引擎能到, Mac 上只是编译) */
static long mem_available_mb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256]; long kb = -1;
    while (fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    fclose(f);
    return kb < 0 ? -1 : kb / 1024;
}

static void lane_finish(v41_lane *L, int rc) {
    if (L->r) ds4_v41_req_spec_stats(L->r, &L->g.spec_rounds, &L->g.spec_offered, &L->g.spec_accepted);   /* 监控的草稿账 */
    v41_gen_end(&L->g, rc);
    ds4_v41_req_close(L->r);
    job_finish(L->j);
    memset(L, 0, sizeof *L);
}

/* 队列里取一个 job 进空道; begin 失败(请求不合法, 错误响应已发)就直接收尾。返回 false = 没取到(不阻塞时队列空 / 阻塞时服务在停) */
static bool lane_admit(server *s, v41_lane *lanes, int cap, int *nl, uint64_t *order, bool block) {
    job *j = block ? dequeue(s) : dequeue_try(s);
    if (!j) return false;
    v41_lane *L = NULL;
    for (int i = 0; i < cap && !L; i++) if (!lanes[i].j) L = &lanes[i];
    L->j = j;
    if (!v41_gen_begin(s, j, &L->g)) { job_finish(j); memset(L, 0, sizeof *L); return true; }
    L->phase = 1; L->order = ++*order; (*nl)++;
    return true;
}

/* 最老的还没预填完的道跑一块(或先开它的请求态); 返回 false = 这一轮没动预填(内存不够在等) */
static bool sched_prefill(server *s, v41_lane *P, int *nl, int n_dec, double *t_chunk, double *t_dec) {
    v41_gen *g = &P->g; job *j = P->j;
    if (P->phase == 1) {
        const long avail = mem_available_mb();
        const uint64_t peak_mb = ds4_v41_req_prefill_bytes(g->prompt_tokens) >> 20;
        if (avail >= 0 && (uint64_t)avail < 2u * peak_mb) {
            if (n_dec > 0) return false;   /* 等解码道退出腾内存 */
            server_log(DS4_LOG_DEFAULT, "ds4-server: %s ctx=%s 预填要 %llu MB 峰值(准入 2 倍), 余量只有 %ld MB, 装不下",
                       j->req.kind == REQ_CHAT ? "chat" : "completion", g->ctx_span, (unsigned long long)peak_mb, avail);
            lane_finish(P, 1); (*nl)--;   /* rc≠0 且没出过 token = 预填失败响应 */
            return true;
        }
        P->r = ds4_v41_req_open(s->engine, j->req.prompt.v, j->req.prompt.len, g->max_tokens, &g->sp);
        if (!P->r) { lane_finish(P, 1); (*nl)--; return true; }
        P->phase = 2;
        mon_prefill(s, j->mon, g->prompt_tokens, 0, g->max_tokens);   /* 监控: 这条道开始读提示(准入等待不算读) */
        server_log(DS4_LOG_PREFILL, "ds4-server: %s ctx=%s%s%s prompt start (V4.1 并发预填, decode_max=%d, 余量 %ld MB, 峰值 %llu MB, 同批解码 %d 路)",
                   j->req.kind == REQ_CHAT ? "chat" : "completion", g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags,
                   g->max_tokens, avail, (unsigned long long)peak_mb, n_dec);
    }
    const double t0 = now_sec();
    const int rc = ds4_v41_req_prefill_step(P->r);
    *t_chunk = now_sec() - t0; *t_dec = 0;
    int c0 = 0, np = 0;
    ds4_v41_req_progress(P->r, &c0, &np);
    if (rc < 0) { lane_finish(P, 1); (*nl)--; return true; }
    if (v41_progress_cb(g, "prefill_chunk", c0, np) != 0) { lane_finish(P, 1); (*nl)--; return true; }   /* 客户端走了: end 按 client_gone 收 */
    if (rc == 1) {
        P->phase = 3;
        if (v41_emit(ds4_v41_req_next(P->r), g) != 0) { lane_finish(P, 0); (*nl)--; }   /* 首个 token(或 max_tokens=0 直接收) */
    }
    return true;
}

void v41_sched_run(server *s) {
    const int cap = s->batch_max;   /* 道数(同时活着的请求数) */
    /* 批态行数按解码小批核路上限(8)开, 不按道数: 投机的验证批每路 1+k 行, 行数只由引擎在 multi_round 里按 8 分配(装不下从 k 最大的路削) */
    struct ds4_v41_batch *b = ds4_v41_batch_open(s->engine, 8);
    if (!b) die("ds4-server: V4.1 批态分配失败(--batch)");
    v41_lane lanes[DS4_SERVER_BATCH_LANES];
    memset(lanes, 0, sizeof lanes);
    int nl = 0;
    uint64_t order = 0;
    double t_chunk = 0, t_dec = 0;   /* 时间片账: 上一块预填用时 / 之后解码累计 */
    server_log(DS4_LOG_DEFAULT, "ds4-server: V4.1 并发调度器: 最多 %d 路合批解码, 预填一次一路, 时间片预填/解码各半, 准入 = 余量 ≥ 2 × 预填峰值(常驻 %llu MB/路)",
               cap, (unsigned long long)(ds4_v41_req_resident_bytes() >> 20));
    for (;;) {
        while (nl < cap && lane_admit(s, lanes, cap, &nl, &order, nl == 0)) {}
        if (nl == 0) break;   /* 阻塞取到 NULL = 服务在停 */
        v41_lane *P = NULL;
        v41_lane *dl[DS4_SERVER_BATCH_LANES]; struct ds4_v41_req *dreq[DS4_SERVER_BATCH_LANES]; int nd = 0;
        for (int i = 0; i < cap; i++) {
            if (!lanes[i].j) continue;
            if (lanes[i].phase == 3) nd++;
            else if (!P || lanes[i].order < P->order) P = &lanes[i];
        }
        if (P && (nd == 0 || t_dec >= t_chunk)) (void)sched_prefill(s, P, &nl, nd, &t_chunk, &t_dec);
        nd = 0;
        for (int i = 0; i < cap; i++) {
            if (!lanes[i].j || lanes[i].phase != 3) continue;
            if (ds4_v41_req_room(lanes[i].r) < 1) { lane_finish(&lanes[i], 0); nl--; continue; }   /* 上下文满: finish 留 length */
            dl[nd] = &lanes[i]; dreq[nd] = lanes[i].r; nd++;
        }
        if (nd == 0) continue;
        const double t0 = now_sec();
        if (ds4_v41_multi_round(b, dreq, nd) != 0) {   /* 投机一轮(引擎投机关 / 没三塔时就是纯解码一步) */
            for (int i = 0; i < nd; i++) { lane_finish(dl[i], 1); nl--; }
            continue;
        }
        for (int i = 0; i < nd; i++) {
            int toks[16];
            const int nt = ds4_v41_req_take(dl[i]->r, toks, 16);
            for (int t = 0; t < nt; t++)
                if (v41_emit(toks[t], &dl[i]->g) != 0) { lane_finish(dl[i], 0); nl--; break; }   /* EOS / 上限 / 停止串 / 挂断: 后面接受的位不再吐 */
        }
        t_dec += now_sec() - t0;
    }
    for (int i = 0; i < cap; i++) if (lanes[i].j) lane_finish(&lanes[i], lanes[i].phase == 3 ? 0 : 1);   /* 服务在停: 收尾按 shutdown 报 */
    ds4_v41_batch_close(b);
}
