/* server_monitor_prom.c — GET /metrics 的 Prometheus 文本(2026-10-07)。
 *
 * 什么时候给文本: 请求头 Accept 带 text/plain 或 application/openmetrics-text(Prometheus 抓取就是这样问的), 或 URL 带 ?format=prometheus;
 * 其它一律 JSON(监控页、裸 curl)。
 * 指标名用 vLLM 的(vllm:num_requests_running / _waiting / kv_cache_usage_perc / prompt_tokens_total / generation_tokens_total /
 * request_success_total / prefix_cache_queries_total / prefix_cache_hits_total / spec_decode_num_draft_tokens_total /
 * spec_decode_num_accepted_tokens_total / num_preemptions_total + 三个延迟直方图), 给 vLLM 写的看板和告警不改一行就能读这台服务;
 * vLLM 没名字的事实按 JSON 的键名挂在 ds4: 下(Strata 用 strata: 前缀挂同一组键; 换前缀是因为这是 ds4 不是 Strata, 键的后缀一字不差):
 * live_state{state} / live_tok_s / live_prefill_tok_s_mean / live_prompt_read / live_slots{state} / engine_max_context /
 * totals_prompt_seconds_total / totals_decode_seconds_total / last_decode_tok_s / gpu_util{gpu} / gpu_mem_used_bytes / gpu_mem_total_bytes /
 * gpu_temp_celsius / gpu_power_watts / cpu / ram_used_bytes / ram_total_bytes。读不到的值不出样本(不是 0)。 */
#include "server_monitor.h"

extern const double mon_buckets[MON_BUCKETS];

static void metric(buf *b, const char *name, const char *kind, const char *help, const char *lab, double v) {
    ds4_buf_printf(b, "# HELP %s %s\n# TYPE %s %s\n%s{%s} %.10g\n", name, help, name, kind, name, lab, v);
}

static void histogram(buf *b, const char *name, const char *help, const char *lab, const mon_hist *h) {
    ds4_buf_printf(b, "# HELP %s %s\n# TYPE %s histogram\n", name, help, name);
    for (int i = 0; i < MON_BUCKETS; i++) ds4_buf_printf(b, "%s_bucket{%s,le=\"%g\"} %u\n", name, lab, mon_buckets[i], h->count[i]);
    ds4_buf_printf(b, "%s_bucket{%s,le=\"+Inf\"} %u\n%s_sum{%s} %.6f\n%s_count{%s} %u\n", name, lab, h->n, name, lab, h->sum, name, lab, h->n);
}

/* ds4: 族: 每族只出一次 HELP/TYPE, 值 NAN 的样本整条不出 */
typedef struct { buf *b; const char *lab; const char *typed[32]; int ntyped; } ds4_fam;
static void ds4_metric(ds4_fam *f, const char *name, const char *kind, const char *help, const char *labels, double v) {
    if (!mon_known(v)) return;
    bool seen = false;
    for (int i = 0; i < f->ntyped && !seen; i++) seen = !strcmp(f->typed[i], name);
    if (!seen) {
        if (f->ntyped < 32) f->typed[f->ntyped++] = name;
        ds4_buf_printf(f->b, "# HELP ds4:%s %s\n# TYPE ds4:%s %s\n", name, help, name, kind);
    }
    ds4_buf_printf(f->b, "ds4:%s{%s%s} %.10g\n", name, f->lab, labels ? labels : "", v);
}

void mon_prometheus_text(server *s, buf *b) {
    struct server_monitor *m = s->mon;
    char lab[160];
    {
        buf esc = {0};   /* model_name 标签: 反斜杠与双引号按 Prometheus 规矩转义 */
        const char *model = s->engine ? ds4_engine_model_name(s->engine) : "";
        for (const char *p = model ? model : "ds4"; *p; p++) { if (*p == '\\' || *p == '"') ds4_buf_putc(&esc, '\\'); ds4_buf_putc(&esc, *p); }
        snprintf(lab, sizeof lab, "model_name=\"%s\"", esc.ptr ? esc.ptr : "ds4");
        ds4_buf_free(&esc);
    }
    pthread_mutex_lock(&m->mu);
    int queued = 0, reading = 0, generating = 0; const mon_live *newest = NULL;
    for (int i = 0; i < m->live_len; i++) {
        const mon_live *L = &m->live[i];
        if (L->state == MON_QUEUED) { queued++; continue; }
        if (L->state == MON_READING) reading++; else generating++;
        if (!newest || L->t_read >= newest->t_read) newest = L;
    }
    double tok_s, tok_s_mean, prefill;
    /* 速率在放锁之后取(mon_live_rates 自己取锁): 两次读之间最多差一个 token, Prometheus 15 s 抓一次不在乎 */
    const int ctx = server_ctx_size(s);
    const int used = newest ? newest->prompt_tokens + newest->generated : 0;
    const bool busy = newest != NULL;
    const char *state = generating ? "generating" : reading ? "reading" : "idle";
    const int prompt_read = newest && newest->state == MON_READING ? newest->prompt_read : 0;
    struct { uint64_t requests, prompt_tokens, reused, output_tokens, drafts_offered, drafts_accepted; double prompt_ms, decode_ms; } t;
    t.requests = m->totals.requests; t.prompt_tokens = m->totals.prompt_tokens; t.reused = m->totals.reused; t.output_tokens = m->totals.output_tokens;
    t.drafts_offered = m->totals.drafts_offered; t.drafts_accepted = m->totals.drafts_accepted; t.prompt_ms = m->totals.prompt_ms; t.decode_ms = m->totals.decode_ms;
    const mon_hist ttft = m->ttft, itl = m->itl, e2e = m->e2e;
    const mon_hw_now hw = m->hw_now;
    const mon_done *lastp = m->hist_len ? &m->hist[(m->hist_head + MON_HISTORY - 1) % MON_HISTORY] : NULL;
    const double last_tok_s = lastp && lastp->output_tokens > 0 && lastp->decode_ms > 0 ? lastp->output_tokens / (lastp->decode_ms / 1e3) : MON_NA;
    const int batch = s->batch_max;
    int slot_state[3] = {0, 0, 0};   /* idle / reading / generating */
    if (batch >= 2) { slot_state[1] = reading; slot_state[2] = generating; slot_state[0] = batch - reading - generating; if (slot_state[0] < 0) slot_state[0] = 0; }
    pthread_mutex_unlock(&m->mu);
    mon_live_rates(m, &tok_s, &tok_s_mean, &prefill);

    metric(b, "vllm:num_requests_running", "gauge", "Requests reading their prompt or generating.", lab, (double)(reading + generating));
    metric(b, "vllm:num_requests_waiting", "gauge", "Requests waiting for their turn.", lab, (double)queued);
    metric(b, "vllm:kv_cache_usage_perc", "gauge", "The running request's share of the context (1 = full).", lab, ctx && busy ? (used > ctx ? 1.0 : (double)used / ctx) : 0.0);
    metric(b, "vllm:prompt_tokens_total", "counter", "Prompt tokens of the finished requests.", lab, (double)t.prompt_tokens);
    metric(b, "vllm:generation_tokens_total", "counter", "Generated tokens.", lab, (double)t.output_tokens);
    metric(b, "vllm:request_success_total", "counter", "Finished requests.", lab, (double)t.requests);
    metric(b, "vllm:prefix_cache_queries_total", "counter", "Prompt tokens looked up in the prompt cache.", lab, (double)t.prompt_tokens);
    metric(b, "vllm:prefix_cache_hits_total", "counter", "Prompt tokens the prompt cache already held.", lab, (double)t.reused);
    metric(b, "vllm:spec_decode_num_draft_tokens_total", "counter", "DSpark draft tokens offered.", lab, (double)t.drafts_offered);
    metric(b, "vllm:spec_decode_num_accepted_tokens_total", "counter", "DSpark draft tokens accepted.", lab, (double)t.drafts_accepted);
    metric(b, "vllm:num_preemptions_total", "counter", "Preempted requests (ds4 does not preempt).", lab, 0.0);
    histogram(b, "vllm:time_to_first_token_seconds", "Time to the first token.", lab, &ttft);
    histogram(b, "vllm:inter_token_latency_seconds", "Time between two tokens.", lab, &itl);
    histogram(b, "vllm:e2e_request_latency_seconds", "A request from start to end.", lab, &e2e);

    ds4_fam f = {b, lab, {0}, 0};
    static const char *states[] = {"idle", "reading", "generating"};
    for (int i = 0; i < 3; i++) {
        char l[40]; snprintf(l, sizeof l, ",state=\"%s\"", states[i]);
        ds4_metric(&f, "live_state", "gauge", "1 for what the engine is doing (live.state).", l, strcmp(state, states[i]) == 0 ? 1.0 : 0.0);
    }
    ds4_metric(&f, "live_tok_s", "gauge", "Decode rate over the last seconds (live.tok_s).", NULL, generating ? tok_s : 0.0);
    ds4_metric(&f, "live_prefill_tok_s_mean", "gauge", "Prompt reading rate of the running request (live.prefill_tok_s_mean).", NULL, busy ? prefill : 0.0);
    ds4_metric(&f, "live_prompt_read", "gauge", "Prompt tokens read so far by the running request (live.prompt_read).", NULL, (double)prompt_read);
    if (batch >= 2) for (int i = 0; i < 3; i++) {
        char l[40]; snprintf(l, sizeof l, ",state=\"%s\"", states[i]);
        ds4_metric(&f, "live_slots", "gauge", "The batch slots in each state (live.slots).", l, (double)slot_state[i]);
    }
    ds4_metric(&f, "engine_max_context", "gauge", "The engine's context (engine.max_context).", NULL, (double)ctx);
    ds4_metric(&f, "totals_prompt_seconds_total", "counter", "Time spent reading prompts (totals.prompt_ms).", NULL, t.prompt_ms / 1000.0);
    ds4_metric(&f, "totals_decode_seconds_total", "counter", "Time spent generating (totals.decode_ms).", NULL, t.decode_ms / 1000.0);
    ds4_metric(&f, "last_decode_tok_s", "gauge", "The last request's decode rate (requests[0].decode_tok_s).", NULL, last_tok_s);
    ds4_metric(&f, "gpu_util", "gauge", "GPU busy, percent (hardware.gpu_util).", ",gpu=\"0\"", hw.gpu_util);
    ds4_metric(&f, "gpu_mem_used_bytes", "gauge", "GPU memory in use (hardware.gpu_mem_used).", ",gpu=\"0\"", hw.gpu_mem_used);
    ds4_metric(&f, "gpu_mem_total_bytes", "gauge", "GPU memory (hardware.gpu_mem_total).", ",gpu=\"0\"", hw.gpu_mem_total);
    ds4_metric(&f, "gpu_temp_celsius", "gauge", "GPU temperature (hardware.gpu_temp).", ",gpu=\"0\"", hw.gpu_temp);
    ds4_metric(&f, "gpu_power_watts", "gauge", "GPU power draw (hardware.gpu_power).", ",gpu=\"0\"", hw.gpu_power);
    ds4_metric(&f, "cpu", "gauge", "CPU busy, percent (hardware.cpu).", NULL, hw.cpu);
    ds4_metric(&f, "ram_used_bytes", "gauge", "RAM in use (hardware.ram_used).", NULL, hw.ram_used);
    ds4_metric(&f, "ram_total_bytes", "gauge", "RAM (hardware.ram_total).", NULL, hw.ram_total);
    ds4_metric(&f, "ram_kernel_total_bytes", "gauge", "RAM the kernel can allocate, physical minus reserved (hardware.ram_kernel_total).", NULL, hw.ram_kernel_total);
}
