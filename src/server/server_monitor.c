/* server_monitor.c — 监控: 每条请求从入队到收尾的记录、总计、延迟直方图, 与 GET /metrics 的 JSON(2026-10-07)。
 *
 * 为什么要它: 服务端以前只有 stderr 日志, 想知道"现在在干什么、刚才那条多快、GPU 吃满了没"得盯终端翻滚。参考 Strata
 * (github.com/Niko1221/Strata)的 Monitor 页: 一条 JSON(/metrics)+ 一页浏览器(web/monitor.html), 指标名与它逐个对齐。
 * 钩子只有六个(mon_begin / mon_prefill / mon_prefill_progress / mon_first_token / mon_token / mon_end), 全挂在生成路已有的事件点上,
 * 取锁都是微秒级, 对解码一步 ~25 ms 没有可见影响。读不到的数一律 null, 不编数。
 * mon_end 幂等: 记录不在了就什么都不做 —— client_main 在 job 结束后补一次 "error" 收尾, 兜住 worker 没走到收尾的失败路
 * (请求不合法 / 预填准入拒绝), 正常路 worker 先收, 补的那次是空操作。 */
#include "server_monitor.h"

static double wall_now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

/* ---- 活请求表 ---------------------------------------------------------------- */
static mon_live *live_find(struct server_monitor *m, uint64_t id) {
    if (!id) return NULL;
    for (int i = 0; i < m->live_len; i++) if (m->live[i].id == id) return &m->live[i];
    return NULL;
}

static mon_live *live_add(struct server_monitor *m) {
    if (m->live_len == m->live_cap) {
        m->live_cap = m->live_cap ? m->live_cap * 2 : 8;
        m->live = xrealloc(m->live, (size_t)m->live_cap * sizeof(*m->live));
    }
    mon_live *L = &m->live[m->live_len++];
    memset(L, 0, sizeof *L);
    return L;
}

static void live_remove(struct server_monitor *m, mon_live *L) {
    const int i = (int)(L - m->live);
    if (i < m->live_len - 1) memmove(L, L + 1, (size_t)(m->live_len - 1 - i) * sizeof(*L));
    m->live_len--;
}

/* 2 s 窗速率(Strata _rate_of): 窗内首尾差分; 窗内跨度不足 0.25 s 就用首 token 以来的均值(分母下限 0.25 s) */
static double live_rate(const mon_live *L, double now) {
    if (L->state != MON_GENERATING || L->t_first <= 0.0) return 0.0;
    if (L->rlen >= 2) {
        const int newest = (L->rhead + MON_RATE_RING - 1) % MON_RATE_RING;
        int oldest = -1;
        for (int k = L->rlen; k >= 1; k--) {   /* 环里从老到新找第一个落在窗内的样本 */
            const int idx = (L->rhead + MON_RATE_RING - k) % MON_RATE_RING;
            if (now - L->rt[idx] <= MON_RATE_WINDOW_S) { oldest = idx; break; }
        }
        if (oldest >= 0 && L->rt[newest] - L->rt[oldest] >= MON_RATE_MIN_SPAN_S) {
            const double r = (double)(L->rn[newest] - L->rn[oldest]) / (L->rt[newest] - L->rt[oldest]);
            return r > 0.0 ? r : 0.0;
        }
    }
    double span = now - L->t_first;
    if (span < MON_RATE_MIN_SPAN_S) span = MON_RATE_MIN_SPAN_S;
    return (double)L->generated / span;
}

static void rates_locked(const struct server_monitor *m, double now, double *tok_s, double *tok_s_mean, double *prefill) {
    double a = 0, b = 0, c = 0;
    for (int i = 0; i < m->live_len; i++) {
        const mon_live *L = &m->live[i];
        if (L->state == MON_GENERATING) {
            a += live_rate(L, now);
            if (L->t_first > 0.0) b += (double)L->generated / (now - L->t_first > 1e-6 ? now - L->t_first : 1e-6);
            /* 生成期也报本请求的预填均速(Strata 的 prefill_tok_s_mean 在整条请求期间都是这条请求的读提示速率): 页面"预填 · 本请求"那格 */
            if (L->t_read > 0.0 && L->t_first - L->t_read > 1e-6) c += (double)(L->prompt_tokens - L->cached) / (L->t_first - L->t_read);
        } else if (L->state == MON_READING && L->t_read > 0.0 && now - L->t_read > 1e-6) {
            c += (double)(L->prompt_read - L->cached) / (now - L->t_read);   /* 新读的 token, 不含缓存前缀 */
        }
    }
    *tok_s = a; *tok_s_mean = b; *prefill = c;
}

void mon_live_rates(struct server_monitor *m, double *tok_s, double *tok_s_mean, double *prefill_tok_s) {
    pthread_mutex_lock(&m->mu);
    rates_locked(m, now_sec(), tok_s, tok_s_mean, prefill_tok_s);
    pthread_mutex_unlock(&m->mu);
}

void mon_series_push(mon_series *s, double v) {
    s->v[s->head] = v;
    s->head = (s->head + 1) % MON_HW_HISTORY;
    if (s->len < MON_HW_HISTORY) s->len++;
}

/* ---- 延迟直方图(vLLM 桶, 秒) ---------------------------------------------------- */
const double mon_buckets[MON_BUCKETS] = {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0};
static void hist_observe(mon_hist *h, double v, uint32_t times) {
    if (!(v >= 0.0) || !times) return;
    for (int i = 0; i < MON_BUCKETS; i++) if (v <= mon_buckets[i]) h->count[i] += times;
    h->sum += v * times; h->n += times;
}

/* ---- 生命周期 ---------------------------------------------------------------- */
struct server_monitor *mon_open(server *s) {
    struct server_monitor *m = xmalloc(sizeof *m);
    memset(m, 0, sizeof *m);
    m->s = s;
    m->since = wall_now();
    m->next_id = 1;
    pthread_mutex_init(&m->mu, NULL);
    pthread_cond_init(&m->stop_cv, NULL);
    mon_hw_static_read(&m->hw_static);
    mon_hw_start(m);
    return m;
}

void mon_close(struct server_monitor *m) {
    if (!m) return;
    mon_hw_stop(m);
    pthread_cond_destroy(&m->stop_cv);
    pthread_mutex_destroy(&m->mu);
    free(m->live);
    free(m);
}

uint64_t mon_begin(server *s, const request *r, const char *path) {
    struct server_monitor *m = s->mon;
    if (!m) return 0;
    pthread_mutex_lock(&m->mu);
    mon_live *L = live_add(m);
    L->id = m->next_id++;
    L->state = MON_QUEUED;
    L->api = r->api; L->kind = r->kind;
    L->stream = r->stream; L->tools = r->has_tools; L->thinking = ds4_think_mode_enabled(r->think_mode) != 0;
    snprintf(L->path, sizeof L->path, "%s", path ? path : "");
    L->started_at = wall_now();
    L->t_queued = now_sec();
    L->prompt_tokens = r->prompt.len;
    L->max_tokens = r->max_tokens;
    const uint64_t id = L->id;
    pthread_mutex_unlock(&m->mu);
    return id;
}

void mon_prefill(server *s, uint64_t id, int prompt_tokens, int cached, int max_tokens) {
    struct server_monitor *m = s->mon;
    if (!m || !id) return;
    pthread_mutex_lock(&m->mu);
    mon_live *L = live_find(m, id);
    if (L) {
        L->state = MON_READING;
        L->t_read = now_sec();
        L->prompt_tokens = prompt_tokens;
        L->cached = cached > 0 ? cached : 0;
        L->prompt_read = L->cached;
        L->max_tokens = max_tokens;
    }
    pthread_mutex_unlock(&m->mu);
}

void mon_prefill_progress(server *s, uint64_t id, int current, int total) {
    struct server_monitor *m = s->mon;
    if (!m || !id) return;
    pthread_mutex_lock(&m->mu);
    mon_live *L = live_find(m, id);
    if (L && L->state == MON_READING) {
        if (total > L->prompt_tokens) L->prompt_tokens = total;
        L->prompt_read = current < 0 ? 0 : current > L->prompt_tokens ? L->prompt_tokens : current;
    }
    pthread_mutex_unlock(&m->mu);
}

void mon_first_token(server *s, uint64_t id) {
    struct server_monitor *m = s->mon;
    if (!m || !id) return;
    pthread_mutex_lock(&m->mu);
    mon_live *L = live_find(m, id);
    if (L && L->state != MON_GENERATING) {
        L->state = MON_GENERATING;
        L->t_first = now_sec();
        if (L->t_read <= 0.0) L->t_read = L->t_first;
        L->prompt_read = L->prompt_tokens;
        L->rhead = L->rlen = 0;
        L->rt[0] = L->t_first; L->rn[0] = 0; L->rhead = 1; L->rlen = 1;
    }
    pthread_mutex_unlock(&m->mu);
}

void mon_token(server *s, uint64_t id, int generated) {
    struct server_monitor *m = s->mon;
    if (!m || !id) return;
    pthread_mutex_lock(&m->mu);
    mon_live *L = live_find(m, id);
    if (L) {
        if (L->state != MON_GENERATING) { L->state = MON_GENERATING; L->t_first = now_sec(); L->prompt_read = L->prompt_tokens; }
        L->generated = generated;
        L->rt[L->rhead] = now_sec(); L->rn[L->rhead] = generated;
        L->rhead = (L->rhead + 1) % MON_RATE_RING;
        if (L->rlen < MON_RATE_RING) L->rlen++;
    }
    pthread_mutex_unlock(&m->mu);
}

void mon_end(server *s, uint64_t id, const char *finish, int generated, int drafts_offered, int drafts_accepted) {
    struct server_monitor *m = s->mon;
    if (!m || !id) return;
    pthread_mutex_lock(&m->mu);
    mon_live *L = live_find(m, id);
    if (L) {
        const double now = now_sec();
        mon_done d;
        memset(&d, 0, sizeof d);
        d.time = L->started_at;
        d.duration_s = now - L->t_queued;
        d.first_token_s = L->t_first > 0.0 ? L->t_first - L->t_queued : MON_NA;
        d.prompt_ms = L->t_read > 0.0 ? ((L->t_first > 0.0 ? L->t_first : now) - L->t_read) * 1e3 : MON_NA;
        d.decode_ms = L->t_first > 0.0 ? (now - L->t_first) * 1e3 : MON_NA;
        snprintf(d.finish, sizeof d.finish, "%s", finish ? finish : "error");
        d.api = L->api; d.stream = L->stream;
        d.prompt_tokens = L->prompt_tokens; d.reused = L->cached;
        d.output_tokens = generated > 0 ? generated : L->generated;
        d.prompt_read = L->t_first > 0.0 ? L->prompt_tokens : L->prompt_read;
        d.drafts_offered = drafts_offered; d.drafts_accepted = drafts_accepted;
        m->hist[m->hist_head] = d;
        m->hist_head = (m->hist_head + 1) % MON_HISTORY;
        if (m->hist_len < MON_HISTORY) m->hist_len++;
        m->totals.requests++;
        m->totals.prompt_tokens += (uint64_t)d.prompt_tokens;
        m->totals.reused += (uint64_t)d.reused;
        m->totals.output_tokens += (uint64_t)d.output_tokens;
        if (mon_known(d.prompt_ms)) m->totals.prompt_ms += d.prompt_ms;
        if (mon_known(d.decode_ms)) m->totals.decode_ms += d.decode_ms;
        if (drafts_offered > 0) m->totals.drafts_offered += (uint64_t)drafts_offered;
        if (drafts_accepted > 0) m->totals.drafts_accepted += (uint64_t)drafts_accepted;
        hist_observe(&m->ttft, d.first_token_s, 1);
        /* token 间隔: 引擎只报解码总时长, 按平均间隔记 n−1 次(Strata 同) */
        if (d.output_tokens > 1 && d.decode_ms > 0.0)
            hist_observe(&m->itl, d.decode_ms / 1e3 / (d.output_tokens - 1), (uint32_t)(d.output_tokens - 1));
        hist_observe(&m->e2e, d.duration_s, 1);
        live_remove(m, L);
    }
    pthread_mutex_unlock(&m->mu);
}

void mon_kv_event(server *s, const char *event, int tokens, int parked, uint64_t bytes, int evicted) {
    struct server_monitor *m = s->mon;
    if (!m) return;
    pthread_mutex_lock(&m->mu);
    if (!strcmp(event, "parked")) m->kv.parks++;
    else if (!strcmp(event, "restored")) m->kv.restores++;
    if (evicted > 0) m->kv.evictions += (uint64_t)evicted;
    m->kv.parked = parked; m->kv.bytes = bytes;
    snprintf(m->kv.last_event, sizeof m->kv.last_event, "%s", event);
    m->kv.last_tokens = tokens; m->kv.last_at = wall_now();
    pthread_mutex_unlock(&m->mu);
}

/* ---- /metrics JSON ----------------------------------------------------------- */
static void jnum(buf *b, double v, int digits) {   /* MON_NA(负) → null; digits 0 = 不带小数 */
    if (!mon_known(v)) { ds4_buf_puts(b, "null"); return; }
    if (digits <= 0) ds4_buf_printf(b, "%.0f", v);
    else ds4_buf_printf(b, "%.*f", digits, v);
}
static void jkey(buf *b, const char *k) { ds4_buf_putc(b, '"'); ds4_buf_puts(b, k); ds4_buf_puts(b, "\":"); }
static void jint_or_null(buf *b, long long v, bool valid) { if (valid) ds4_buf_printf(b, "%lld", v); else ds4_buf_puts(b, "null"); }
static void jstr_or_null(buf *b, const char *v) { if (v && v[0]) json_escape(b, v); else ds4_buf_puts(b, "null"); }

static const char *api_name(api_style a) { return a == API_ANTHROPIC ? "anthropic" : a == API_RESPONSES ? "responses" : "openai"; }
static const char *state_name(mon_state st) { return st == MON_GENERATING ? "generating" : st == MON_READING ? "reading" : "queued"; }

static void series_json(buf *b, const mon_series *s) {
    ds4_buf_putc(b, '[');
    for (int k = s->len; k >= 1; k--) {
        const int idx = (s->head + MON_HW_HISTORY - k) % MON_HW_HISTORY;
        if (k != s->len) ds4_buf_putc(b, ',');
        jnum(b, s->v[idx], 2);
    }
    ds4_buf_putc(b, ']');
}

static void done_json(buf *b, const mon_done *d) {
    const double decode_tok_s = d->output_tokens > 0 && d->decode_ms > 0.0 ? d->output_tokens / (d->decode_ms / 1e3) : MON_NA;
    ds4_buf_putc(b, '{');
    jkey(b, "time"); jnum(b, d->time, 3);
    ds4_buf_putc(b, ','); jkey(b, "duration_s"); jnum(b, d->duration_s, 1);
    ds4_buf_putc(b, ','); jkey(b, "finish"); json_escape(b, d->finish);
    ds4_buf_putc(b, ','); jkey(b, "api"); json_escape(b, api_name(d->api));
    ds4_buf_putc(b, ','); jkey(b, "stream"); ds4_buf_puts(b, d->stream ? "true" : "false");
    ds4_buf_putc(b, ','); jkey(b, "prompt_tokens"); ds4_buf_printf(b, "%d", d->prompt_tokens);
    ds4_buf_putc(b, ','); jkey(b, "reused"); ds4_buf_printf(b, "%d", d->reused);
    ds4_buf_putc(b, ','); jkey(b, "output_tokens"); ds4_buf_printf(b, "%d", d->output_tokens);
    ds4_buf_putc(b, ','); jkey(b, "prompt_total"); ds4_buf_printf(b, "%d", d->prompt_tokens);
    ds4_buf_putc(b, ','); jkey(b, "prompt_read"); ds4_buf_printf(b, "%d", d->prompt_read);
    ds4_buf_putc(b, ','); jkey(b, "first_token_s"); jnum(b, d->first_token_s, 3);
    ds4_buf_putc(b, ','); jkey(b, "prompt_ms"); jnum(b, d->prompt_ms, 1);
    ds4_buf_putc(b, ','); jkey(b, "decode_ms"); jnum(b, d->decode_ms, 1);
    ds4_buf_putc(b, ','); jkey(b, "decode_tok_s"); jnum(b, decode_tok_s, 1);
    ds4_buf_putc(b, ','); jkey(b, "hit_rate"); ds4_buf_puts(b, "null");   /* 专家全常驻, 没有专家缓存命中率这个量; 留键给页面/看板 */
    ds4_buf_putc(b, ','); jkey(b, "drafts_offered"); jint_or_null(b, d->drafts_offered, d->drafts_offered >= 0);
    ds4_buf_putc(b, ','); jkey(b, "drafts_accepted"); jint_or_null(b, d->drafts_accepted, d->drafts_accepted >= 0);
    ds4_buf_putc(b, '}');
}

static void live_json(buf *b, const struct server_monitor *m, double now) {
    const mon_live *newest = NULL; int queued = 0, reading = 0, generating = 0;
    for (int i = 0; i < m->live_len; i++) {
        const mon_live *L = &m->live[i];
        if (L->state == MON_QUEUED) { queued++; continue; }
        if (L->state == MON_READING) reading++; else generating++;
        if (!newest || L->t_read >= newest->t_read) newest = L;
    }
    const char *state = generating ? "generating" : reading ? "reading" : "idle";
    double tok_s, tok_s_mean, prefill;
    rates_locked(m, now, &tok_s, &tok_s_mean, &prefill);
    const bool busy = newest != NULL;
    ds4_buf_putc(b, '{');
    jkey(b, "state"); json_escape(b, state);
    ds4_buf_putc(b, ','); jkey(b, "queued"); ds4_buf_printf(b, "%d", queued);
    ds4_buf_putc(b, ','); jkey(b, "phase"); ds4_buf_puts(b, "null");
    ds4_buf_putc(b, ','); jkey(b, "prompt_tokens"); jint_or_null(b, busy ? newest->prompt_tokens : 0, busy);
    ds4_buf_putc(b, ','); jkey(b, "prompt_read"); jint_or_null(b, busy ? newest->prompt_read : 0, busy && newest->state == MON_READING);
    ds4_buf_putc(b, ','); jkey(b, "prompt_total"); jint_or_null(b, busy ? newest->prompt_tokens : 0, busy && newest->state == MON_READING);
    ds4_buf_putc(b, ','); jkey(b, "generated"); jint_or_null(b, busy ? newest->generated : 0, busy);
    ds4_buf_putc(b, ','); jkey(b, "max_tokens"); jint_or_null(b, busy ? newest->max_tokens : 0, busy);
    ds4_buf_putc(b, ','); jkey(b, "elapsed_s"); jnum(b, busy ? now - newest->t_read : MON_NA, 1);
    ds4_buf_putc(b, ','); jkey(b, "tok_s"); jnum(b, generating ? tok_s : MON_NA, 1);
    ds4_buf_putc(b, ','); jkey(b, "tok_s_mean"); jnum(b, generating ? tok_s_mean : MON_NA, 1);
    ds4_buf_putc(b, ','); jkey(b, "prefill_tok_s_mean"); jnum(b, busy ? prefill : MON_NA, 0);
    ds4_buf_putc(b, ','); jkey(b, "tok_s_window_s"); jnum(b, generating ? MON_RATE_WINDOW_S : MON_NA, 1);
    if (m->s->batch_max >= 2) {   /* 并发调度器: 每条道一行(Strata 的 slots 视图) */
        ds4_buf_putc(b, ','); jkey(b, "parallel"); ds4_buf_printf(b, "%d", m->s->batch_max);
        ds4_buf_putc(b, ','); jkey(b, "running"); ds4_buf_printf(b, "%d", reading + generating);
        ds4_buf_putc(b, ','); jkey(b, "waiting"); ds4_buf_printf(b, "%d", queued);
        ds4_buf_putc(b, ','); jkey(b, "outside_slots"); ds4_buf_puts(b, "0");
        ds4_buf_putc(b, ','); jkey(b, "slots"); ds4_buf_putc(b, '[');
        int slot = 0;
        for (int i = 0; i < m->live_len && slot < m->s->batch_max; i++) {
            const mon_live *L = &m->live[i];
            if (L->state == MON_QUEUED) continue;
            if (slot) ds4_buf_putc(b, ',');
            ds4_buf_printf(b, "{\"slot\":%d,\"state\":\"%s\",\"prompt_tokens\":%d,\"generated\":%d,\"elapsed_s\":", slot, state_name(L->state), L->prompt_tokens, L->generated);
            jnum(b, now - L->t_read, 1);
            ds4_buf_puts(b, ",\"tok_s\":"); jnum(b, L->state == MON_GENERATING ? live_rate(L, now) : MON_NA, 1);
            ds4_buf_putc(b, '}');
            slot++;
        }
        for (; slot < m->s->batch_max; slot++) ds4_buf_printf(b, "%s{\"slot\":%d,\"state\":\"idle\",\"held_tokens\":0}", slot ? "," : "", slot);
        ds4_buf_putc(b, ']');
    }
    ds4_buf_putc(b, '}');
}

static void hardware_json(buf *b, const struct server_monitor *m, double now) {
    const mon_hw_now *h = &m->hw_now;
    double tok_s, tok_s_mean, prefill;
    rates_locked(m, now, &tok_s, &tok_s_mean, &prefill);
    ds4_buf_putc(b, '{');
    jkey(b, "gpu_util"); jnum(b, h->gpu_util, 1);
    ds4_buf_putc(b, ','); jkey(b, "gpu_mem_used"); jnum(b, h->gpu_mem_used, 0);
    ds4_buf_putc(b, ','); jkey(b, "gpu_mem_total"); jnum(b, h->gpu_mem_total, 0);
    ds4_buf_putc(b, ','); jkey(b, "gpu_temp"); jnum(b, h->gpu_temp, 1);
    ds4_buf_putc(b, ','); jkey(b, "gpu_power"); jnum(b, h->gpu_power, 1);
    ds4_buf_putc(b, ','); jkey(b, "gpu_power_limit"); jnum(b, h->gpu_power_limit, 1);
    ds4_buf_putc(b, ','); jkey(b, "gpu_pcie_gen"); jnum(b, h->gpu_pcie_gen, 0);
    ds4_buf_putc(b, ','); jkey(b, "gpu_pcie_gen_max"); jnum(b, h->gpu_pcie_gen_max, 0);
    ds4_buf_putc(b, ','); jkey(b, "gpu_pcie_width"); jnum(b, h->gpu_pcie_width, 0);
    ds4_buf_putc(b, ','); jkey(b, "gpu_pcie_rx_mb"); jnum(b, h->gpu_pcie_rx_mb, 2);
    ds4_buf_putc(b, ','); jkey(b, "gpu_pcie_tx_mb"); jnum(b, h->gpu_pcie_tx_mb, 2);
    ds4_buf_putc(b, ','); jkey(b, "cpu"); jnum(b, h->cpu, 1);
    ds4_buf_putc(b, ','); jkey(b, "ram_used"); jnum(b, h->ram_used, 0);
    ds4_buf_putc(b, ','); jkey(b, "ram_total"); jnum(b, h->ram_total, 0);
    ds4_buf_putc(b, ','); jkey(b, "ram_kernel_total"); jnum(b, h->ram_kernel_total, 0);
    ds4_buf_putc(b, ','); jkey(b, "disk_read_mb"); jnum(b, h->disk_read_mb, 2);
    ds4_buf_putc(b, ','); jkey(b, "disk_write_mb"); jnum(b, h->disk_write_mb, 2);
    ds4_buf_putc(b, ','); jkey(b, "tok_s"); jnum(b, tok_s, 1);
    ds4_buf_putc(b, ','); jkey(b, "tok_s_mean"); jnum(b, tok_s_mean, 1);
    ds4_buf_putc(b, ','); jkey(b, "prefill_tok_s_mean"); jnum(b, prefill, 0);
    ds4_buf_putc(b, '}');
}

static void conversation_cache_json(buf *b, const struct server_monitor *m) {
    const server *s = m->s;
    int requests_reused = 0; const mon_done *last = m->hist_len ? &m->hist[(m->hist_head + MON_HISTORY - 1) % MON_HISTORY] : NULL;
    for (int k = 1; k <= m->hist_len; k++) if (m->hist[(m->hist_head + MON_HISTORY - k) % MON_HISTORY].reused > 0) requests_reused++;
    ds4_buf_putc(b, '{');
    jkey(b, "enabled"); ds4_buf_puts(b, s->kv.enabled ? "true" : "false");
    ds4_buf_putc(b, ','); jkey(b, "budget_mib"); jint_or_null(b, (long long)(s->kv.budget_bytes >> 20), s->kv.enabled);
    ds4_buf_putc(b, ','); jkey(b, "slots"); ds4_buf_puts(b, "null");   /* 磁盘 KV 按字节预算, 不按条数 */
    ds4_buf_putc(b, ','); jkey(b, "parked"); ds4_buf_printf(b, "%d", m->kv.parked);
    ds4_buf_putc(b, ','); jkey(b, "bytes"); ds4_buf_printf(b, "%llu", (unsigned long long)m->kv.bytes);
    ds4_buf_putc(b, ','); jkey(b, "parks"); ds4_buf_printf(b, "%llu", (unsigned long long)m->kv.parks);
    ds4_buf_putc(b, ','); jkey(b, "restores"); ds4_buf_printf(b, "%llu", (unsigned long long)m->kv.restores);
    ds4_buf_putc(b, ','); jkey(b, "evictions"); ds4_buf_printf(b, "%llu", (unsigned long long)m->kv.evictions);
    ds4_buf_putc(b, ','); jkey(b, "requests"); ds4_buf_printf(b, "%llu", (unsigned long long)m->totals.requests);
    ds4_buf_putc(b, ','); jkey(b, "requests_reused"); ds4_buf_printf(b, "%d", requests_reused);
    ds4_buf_putc(b, ','); jkey(b, "reused_tokens"); ds4_buf_printf(b, "%llu", (unsigned long long)m->totals.reused);
    ds4_buf_putc(b, ','); jkey(b, "prompt_tokens"); ds4_buf_printf(b, "%llu", (unsigned long long)m->totals.prompt_tokens);
    ds4_buf_putc(b, ','); jkey(b, "last_prompt"); jint_or_null(b, last ? last->prompt_tokens : 0, last != NULL);
    ds4_buf_putc(b, ','); jkey(b, "last_reused"); jint_or_null(b, last ? last->reused : 0, last != NULL);
    ds4_buf_putc(b, ','); jkey(b, "last_event"); jstr_or_null(b, m->kv.last_event);
    ds4_buf_putc(b, ','); jkey(b, "last_tokens"); jint_or_null(b, m->kv.last_tokens, m->kv.last_event[0] != 0);
    ds4_buf_putc(b, ','); jkey(b, "last_at"); jnum(b, m->kv.last_event[0] ? m->kv.last_at : MON_NA, 3);
    ds4_buf_putc(b, '}');
}

void mon_metrics_json(server *s, buf *b, bool all_requests) {
    struct server_monitor *m = s->mon;
    const double now = now_sec();
    pthread_mutex_lock(&m->mu);
    ds4_buf_puts(b, "{\"engine\":{");
    jkey(b, "model"); json_escape(b, s->engine ? ds4_engine_model_name(s->engine) : "");   /* 离线单测没有引擎 */
    ds4_buf_putc(b, ','); jkey(b, "max_context"); ds4_buf_printf(b, "%d", server_ctx_size(s));
    ds4_buf_putc(b, ','); jkey(b, "images"); ds4_buf_puts(b, s->engine && ds4_engine_mm(s->engine) ? "true" : "false");
    ds4_buf_putc(b, ','); jkey(b, "backend"); json_escape(b, s->backend_name ? s->backend_name : "");
    ds4_buf_putc(b, ','); jkey(b, "variant"); json_escape(b, ds4_engine_is_v41(s->engine) ? "v4.1" : "v4");
    ds4_buf_putc(b, ','); jkey(b, "spec"); ds4_buf_puts(b, ds4_engine_is_v41(s->engine) && ds4_engine_v41_dspark() ? "true" : "false");
    ds4_buf_putc(b, ','); jkey(b, "batch"); ds4_buf_printf(b, "%d", s->batch_max);
    ds4_buf_putc(b, ','); jkey(b, "kv_disk"); ds4_buf_puts(b, s->kv.enabled ? "true" : "false");
    ds4_buf_puts(b, "},\"live\":");
    live_json(b, m, now);
    ds4_buf_puts(b, ",\"requests\":[");
    const int show = all_requests || m->hist_len < MON_SHOW ? m->hist_len : MON_SHOW;
    for (int k = 1; k <= show; k++) {   /* 最新的在前 */
        if (k > 1) ds4_buf_putc(b, ',');
        done_json(b, &m->hist[(m->hist_head + MON_HISTORY - k) % MON_HISTORY]);
    }
    ds4_buf_printf(b, "],\"requests_kept\":%d,\"totals\":{", m->hist_len);
    jkey(b, "since"); jnum(b, m->since, 3);
    ds4_buf_printf(b, ",\"requests\":%llu,\"prompt_tokens\":%llu,\"reused\":%llu,\"output_tokens\":%llu,\"prompt_ms\":%.1f,\"decode_ms\":%.1f,\"drafts_offered\":%llu,\"drafts_accepted\":%llu}",
               (unsigned long long)m->totals.requests, (unsigned long long)m->totals.prompt_tokens, (unsigned long long)m->totals.reused,
               (unsigned long long)m->totals.output_tokens, m->totals.prompt_ms, m->totals.decode_ms,
               (unsigned long long)m->totals.drafts_offered, (unsigned long long)m->totals.drafts_accepted);
    ds4_buf_puts(b, ",\"hardware\":");
    hardware_json(b, m, now);
    ds4_buf_puts(b, ",\"hardware_static\":{");
    jkey(b, "gpu_name"); jstr_or_null(b, m->hw_static.gpu_name);
    ds4_buf_putc(b, ','); jkey(b, "gpu_count"); ds4_buf_printf(b, "%d", m->hw_static.gpu_count);
    ds4_buf_putc(b, ','); jkey(b, "gpu_mem_source"); jstr_or_null(b, m->hw_static.gpu_mem_source);
    ds4_buf_putc(b, ','); jkey(b, "cpu_name"); jstr_or_null(b, m->hw_static.cpu_name);
    ds4_buf_putc(b, ','); jkey(b, "cores"); jint_or_null(b, m->hw_static.cores, m->hw_static.cores > 0);
    ds4_buf_putc(b, ','); jkey(b, "threads"); jint_or_null(b, m->hw_static.threads, m->hw_static.threads > 0);
    ds4_buf_puts(b, "},\"history\":{");
    static const char *series_keys[MON_S_COUNT] = {"gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used", "disk_read_mb", "tok_s", "prefill_tok_s_mean"};
    for (int i = 0; i < MON_S_COUNT; i++) {
        if (i) ds4_buf_putc(b, ',');
        jkey(b, series_keys[i]); series_json(b, &m->series[i]);
    }
    ds4_buf_puts(b, "},\"conversation_cache\":");
    conversation_cache_json(b, m);
    ds4_buf_puts(b, ",\"time\":");
    jnum(b, wall_now(), 3);
    ds4_buf_puts(b, "}\n");
    pthread_mutex_unlock(&m->mu);
}
