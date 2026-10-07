/* server_batch.c — 机械拆分自 ds4_server.c (12247-12448 行): 任务队列与并发批处理。 */

#include "server_internal.h"

bool enqueue(server *s, job *j) {
    pthread_mutex_lock(&s->mu);
    if (s->stopping) {
        pthread_mutex_unlock(&s->mu);
        return false;
    }
    if (s->tail) s->tail->next = j; else s->head = j;
    s->tail = j;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mu);
    return true;
}

static job *dequeue_locked(server *s) {   /* 调用方持锁 */
    job *j = s->head;
    if (!j) return NULL;
    s->head = j->next;
    if (!s->head) s->tail = NULL;
    j->next = NULL;
    return j;
}

job *dequeue(server *s) {
    pthread_mutex_lock(&s->mu);
    while (!s->head && !s->stopping) pthread_cond_wait(&s->cv, &s->mu);
    job *j = dequeue_locked(s);
    pthread_mutex_unlock(&s->mu);
    return j;
}

job *dequeue_try(server *s) {   /* 不阻塞: V4.1 并发调度器在道没满时看一眼队列(server_sched_v41.c) */
    pthread_mutex_lock(&s->mu);
    job *j = dequeue_locked(s);
    pthread_mutex_unlock(&s->mu);
    return j;
}

/* ===== 并发批处理快路 (2026-08-21) ==================================
 * 引擎侧 ds4_session_eval_multi 能把 N 路解码拼进同一次前向 —— 骨干 dense/FFN/MoE 与
 * 输出头的权重只读一遍(实测 8 路聚合 1.43×)。但 server 一直是"单 worker 逐个 job",
 * 8 路并发只是排队(实测 32.8 t/s = 单流速度)。
 *
 * 这条快路只接管"简单请求"(非流式 / 无工具 / chat / Anthropic|OpenAI): 每路建自己的
 * 临时会话, 各自 prefill, 然后联合解码。复杂请求(流式、工具、responses 续接、跨请求
 * 前缀缓存复用)一律走原来的 generate_job —— 那条路依赖单会话 KV 与工具活状态, 不动。
 * 代价: 批内请求放弃跨请求前缀缓存复用(各自新会话), 所以默认关, DS4_SERVER_BATCH=N 开。 */
static bool job_batchable(const job *j) {
    return j && !j->req.stream && !j->req.has_tools &&
           (j->req.kind == REQ_CHAT || j->req.kind == REQ_COMPLETION) &&
           (j->req.api == API_ANTHROPIC || j->req.api == API_OPENAI) &&
           j->req.prompt.len > 0;
}

/* 队列里再摘最多 max 个可合批的 job(不阻塞; 保持队列顺序) */
static uint32_t dequeue_batchable(server *s, job **out, uint32_t max) {
    uint32_t n = 0;
    pthread_mutex_lock(&s->mu);
    job **prev = &s->head;
    for (job *cur = s->head; cur && n < max; ) {
        job *next = cur->next;
        if (job_batchable(cur)) {
            *prev = next;
            if (s->tail == cur) s->tail = (*prev) ? s->tail : NULL;
            cur->next = NULL;
            out[n++] = cur;
        } else {
            prev = &cur->next;
        }
        cur = next;
    }
    /* tail 重算(上面的摘除可能动到尾) */
    s->tail = NULL;
    for (job *cur = s->head; cur; cur = cur->next) s->tail = cur;
    pthread_mutex_unlock(&s->mu);
    return n;
}

void job_finish(job *j) {
    pthread_mutex_lock(&j->mu);
    j->done = true;
    pthread_cond_signal(&j->cv);
    pthread_mutex_unlock(&j->mu);
}

static void generate_jobs_batched(server *s, job **jobs, uint32_t n) {
    char err[160];
    ds4_session *sess[DS4_SERVER_BATCH_LANES] = {0};
    buf text[DS4_SERVER_BATCH_LANES];
    int completion[DS4_SERVER_BATCH_LANES] = {0}, maxtok[DS4_SERVER_BATCH_LANES] = {0},
        prompt_tokens[DS4_SERVER_BATCH_LANES] = {0};
    bool done[DS4_SERVER_BATCH_LANES] = {false};
    const char *finish[DS4_SERVER_BATCH_LANES];
    uint64_t rng[DS4_SERVER_BATCH_LANES];
    char id[DS4_SERVER_BATCH_LANES][96];
    memset(text, 0, sizeof(text));
    const double t0 = now_sec();

    for (uint32_t i = 0; i < n; i++) {
        finish[i] = "length";
        snprintf(id[i], sizeof(id[i]), "chatcmpl-%llu", (unsigned long long)++s->seq);
        rng[i] = jobs[i]->req.seed ? jobs[i]->req.seed
                                   : (((uint64_t)time(NULL) << 32) ^ ((uint64_t)s->seq << 1) ^ (uint64_t)i);
        err[0] = 0;
        mon_prefill(s, jobs[i]->mon, jobs[i]->req.prompt.len, 0, jobs[i]->req.max_tokens);   /* 监控: 临时会话 sync = 整段预填 */
        if (ds4_session_create(&sess[i], s->engine, s->ctx_size) != 0 || !sess[i] ||
            ds4_session_sync(sess[i], &jobs[i]->req.prompt, err, sizeof err) != 0) {
            server_log(DS4_LOG_WARNING, "ds4-server: 批处理会话 %u 建立失败: %s", i, err);
            done[i] = true; finish[i] = "error";
            continue;
        }
        prompt_tokens[i] = jobs[i]->req.prompt.len;
        mon_first_token(s, jobs[i]->mon);
        /* 请求级惩罚(frequency/presence): 原路 generate_job 有, 批路一开始漏了 ⇒ 客户端
         * 传的 frequency_penalty 被静默丢弃(实测: 开 0.25 与不开 18/20 逐字相同)。
         * 生成区边界 = 此刻的 checkpoint(prompt 到此为止), 与原路口径一致。 */
        ds4_session_set_request_penalties(sess[i], jobs[i]->req.frequency_penalty,
                                          jobs[i]->req.presence_penalty);
        int mt = jobs[i]->req.max_tokens;
        const int room = ds4_session_ctx(sess[i]) - ds4_session_pos(sess[i]);
        if (mt <= 0) mt = s->default_tokens;
        if (mt > room) mt = room;
        if (s->max_output_tokens > 0 && mt > s->max_output_tokens) mt = s->max_output_tokens;
        maxtok[i] = mt;
    }

    /* 联合解码: 每步各自采样, 活跃行拼成一次前向 */
    for (;;) {
        ds4_session *act[DS4_SERVER_BATCH_LANES]; int tok[DS4_SERVER_BATCH_LANES];
        uint32_t idx[DS4_SERVER_BATCH_LANES], na = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (done[i]) continue;
            if (completion[i] >= maxtok[i]) { done[i] = true; finish[i] = "length"; continue; }
            const int t = ds4_session_sample(sess[i], jobs[i]->req.temperature,
                                             jobs[i]->req.top_k, jobs[i]->req.top_p,
                                             jobs[i]->req.min_p, &rng[i]);
            if (t == ds4_token_eos(s->engine)) { done[i] = true; finish[i] = "stop"; continue; }
            size_t plen = 0;
            char *piece = ds4_token_text(s->engine, t, &plen);
            if (piece && plen) buf_append(&text[i], piece, plen);
            completion[i]++;
            mon_token(s, jobs[i]->mon, completion[i]);
            act[na] = sess[i]; tok[na] = t; idx[na] = i; na++;
        }
        if (!na) break;
        err[0] = 0;
        if (na == 1) {
            if (ds4_session_eval(act[0], tok[0], err, sizeof err) != 0) {
                done[idx[0]] = true; finish[idx[0]] = "error";
            }
        } else if (ds4_session_eval_multi(act, tok, na, err, sizeof err) != 0) {
            server_log(DS4_LOG_WARNING, "ds4-server: 批解码失败: %s", err);
            for (uint32_t k = 0; k < na; k++) { done[idx[k]] = true; finish[idx[k]] = "error"; }
        }
    }

    int total = 0;
    for (uint32_t i = 0; i < n; i++) total += completion[i];
    server_log(DS4_LOG_GENERATION,
               "ds4-server: 批处理 %u 路 gen=%d 用时 %.2fs ⇒ 聚合 %.2f t/s",
               n, total, now_sec() - t0, total / (now_sec() - t0));

    for (uint32_t i = 0; i < n; i++) {
        char *content = NULL, *reasoning = NULL;
        tool_calls calls = {0};
        bool recovered = false;
        const char *fin = finish[i];
        char perr[160]; perr[0] = 0;
        /* completions 是裸续写口径: 生成的就是答案本身, 不做 chat 消息解析 */
        if (jobs[i]->req.kind == REQ_CHAT)
            (void)parse_generated_message_for_response(text[i].ptr ? text[i].ptr : "",
                                                       false, false,
                                                       ds4_think_mode_enabled(jobs[i]->req.think_mode),
                                                       &fin, perr, sizeof perr,
                                                       &content, &reasoning, &calls, &recovered);
        const char *body = content ? content : (text[i].ptr ? text[i].ptr : "");
        if (jobs[i]->req.api == API_ANTHROPIC)
            anthropic_final_response(jobs[i]->fd, s->enable_cors, &jobs[i]->req, id[i],
                                     body, reasoning, &calls, fin,
                                     prompt_tokens[i], completion[i]);
        else
            final_response(jobs[i]->fd, s->enable_cors, &jobs[i]->req, id[i],
                           body, reasoning, &calls, fin,
                           prompt_tokens[i], completion[i]);
        free(content); free(reasoning); tool_calls_free(&calls);
        buf_free(&text[i]);
        if (sess[i]) ds4_session_free(sess[i]);
        mon_end(s, jobs[i]->mon, fin, completion[i], -1, -1);
        job_finish(jobs[i]);
    }
}

void *worker_main(void *arg) {
    server *s = arg;
    /* V4.1 + --batch N(≥2): 整个 worker 交给并发调度器(server_sched_v41.c); 下面的 V4 合批快路与单 job 圈不再走 */
    if (ds4_engine_is_v41(s->engine) && s->batch_max >= 2) { v41_sched_run(s); return NULL; }
    for (;;) {
        job *j = dequeue(s);
        if (!j) break;
        if (s->batch_max >= 2 && job_batchable(j)) {
            job *batch[DS4_SERVER_BATCH_LANES];
            batch[0] = j;
            /* 聚集窗口: 首个可合批的 job 到达后等一小会儿, 让同时发出的其余请求也排进来
             * (实测不等的话 8 路里常有 1 路晚到, 只能合 7 路)。60ms 对单请求延迟
             * 的影响远小于一次前向(29ms/token × N)。 */
            {
                struct timespec ts = { .tv_sec = DS4_SERVER_BATCH_WAIT_MS / 1000,
                                       .tv_nsec = (long)(DS4_SERVER_BATCH_WAIT_MS % 1000) * 1000000L };
                nanosleep(&ts, NULL);
            }
            const uint32_t extra = dequeue_batchable(s, batch + 1, (uint32_t)s->batch_max - 1u);
            if (extra > 0) { generate_jobs_batched(s, batch, extra + 1u); continue; }
        }
        generate_job(s, j);
        job_finish(j);
    }
    return NULL;
}
