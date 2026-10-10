/* server_tests_monitor.c — 监控(server_monitor*.c)离线单测(2026-10-07): 请求生命周期 → /metrics JSON 字段、收尾幂等、
 * 最近请求条数(默认 12 / ?requests=all)、Prometheus 文本的 vLLM 指标名、Accept 头判法、/monitor 页能被服务。
 * 不要模型: server 结构体只填监控用到的几项(engine = NULL ⇒ 模型名空串)。 */
#include "server_tests_internal.h"

static server *monitor_test_server(void) {
    server *s = xmalloc(sizeof *s);
    memset(s, 0, sizeof *s);
    s->ctx_size = 4096;
    s->backend_name = "test";
    pthread_mutex_init(&s->mu, NULL);
    s->mon = mon_open(s);
    return s;
}

static void monitor_test_server_free(server *s) {
    mon_close(s->mon);
    pthread_mutex_destroy(&s->mu);
    free(s);
}

static char *metrics_json(server *s, bool all) {
    buf b = {0};
    mon_metrics_json(s, &b, all);
    return ds4_buf_take(&b);
}

static int count_substr(const char *s, const char *needle) {
    int n = 0;
    for (const char *p = strstr(s, needle); p; p = strstr(p + 1, needle)) n++;
    return n;
}

/* 取 "key": 后面的数字(JSON 里键唯一时用) */
static double json_number_after(const char *json, const char *key) {
    const char *p = strstr(json, key);
    if (!p) return -1;
    return strtod(p + strlen(key), NULL);
}

void test_monitor_request_lifecycle_reaches_metrics(void) {
    server *s = monitor_test_server();
    request r;
    request_init(&r, REQ_CHAT, 256);
    r.api = API_OPENAI;
    r.prompt.len = 100;
    r.stream = true;

    char *j0 = metrics_json(s, false);
    TEST_ASSERT(strstr(j0, "\"state\":\"idle\"") != NULL);
    TEST_ASSERT(strstr(j0, "\"requests_kept\":0") != NULL);
    TEST_ASSERT(strstr(j0, "\"max_context\":4096") != NULL);
    TEST_ASSERT(strstr(j0, "\"history\":{\"gpu_util\":[") != NULL);
    TEST_ASSERT(strstr(j0, "\"conversation_cache\":{\"enabled\":false") != NULL);
    free(j0);

    const uint64_t id = mon_begin(s, &r, "/v1/chat/completions");
    TEST_ASSERT(id != 0);
    char *j1 = metrics_json(s, false);
    TEST_ASSERT(strstr(j1, "\"state\":\"idle\"") != NULL);   /* 排队中的请求不算忙 */
    TEST_ASSERT(strstr(j1, "\"queued\":1") != NULL);
    free(j1);

    mon_prefill(s, id, 100, 20, 256);
    mon_prefill_progress(s, id, 50, 100);
    char *j2 = metrics_json(s, false);
    TEST_ASSERT(strstr(j2, "\"state\":\"reading\"") != NULL);
    TEST_ASSERT(strstr(j2, "\"queued\":0") != NULL);
    TEST_ASSERT(strstr(j2, "\"prompt_read\":50") != NULL);
    TEST_ASSERT(strstr(j2, "\"prompt_total\":100") != NULL);
    free(j2);

    mon_first_token(s, id);
    for (int t = 1; t <= 12; t++) {   /* 分摊在 ~0.3 s 里: 2 s 窗口的差分速率要有跨度才成立 */
        struct timespec ts = {0, 25 * 1000 * 1000};
        nanosleep(&ts, NULL);
        mon_token(s, id, t);
    }
    char *j3 = metrics_json(s, false);
    TEST_ASSERT(strstr(j3, "\"state\":\"generating\"") != NULL);
    TEST_ASSERT(strstr(j3, "\"generated\":12") != NULL);
    TEST_ASSERT(json_number_after(j3, "\"live\":{\"state\":\"generating\",\"queued\":0,\"phase\":null,\"prompt_tokens\":100,\"prompt_read\":null,\"prompt_total\":null,\"generated\":12,\"max_tokens\":256,\"elapsed_s\":") > 0.0);
    const char *tok = strstr(j3, "\"tok_s\":");
    TEST_ASSERT(tok != NULL && strtod(tok + 8, NULL) > 10.0);   /* 12 个 token / 0.3 s ≈ 40 t/s, 怎么也在 10 以上 */
    free(j3);

    mon_end(s, id, "stop", 12, 15, 9);
    mon_end(s, id, "error", 0, -1, -1);   /* 幂等: 第二次收尾不再记一条 */
    char *j4 = metrics_json(s, false);
    TEST_ASSERT(strstr(j4, "\"state\":\"idle\"") != NULL);
    TEST_ASSERT(strstr(j4, "\"requests_kept\":1") != NULL);
    TEST_ASSERT(strstr(j4, "\"finish\":\"stop\"") != NULL);
    TEST_ASSERT(strstr(j4, "\"output_tokens\":12") != NULL);
    TEST_ASSERT(strstr(j4, "\"reused\":20") != NULL);
    TEST_ASSERT(strstr(j4, "\"drafts_offered\":15") != NULL);
    TEST_ASSERT(strstr(j4, "\"drafts_accepted\":9") != NULL);
    TEST_ASSERT(strstr(j4, "\"totals\":{\"since\":") != NULL);
    TEST_ASSERT(strstr(j4, "\"requests\":1,\"prompt_tokens\":100,\"reused\":20,\"output_tokens\":12,") != NULL);
    TEST_ASSERT(strstr(j4, "\"drafts_offered\":15,\"drafts_accepted\":9}") != NULL);
    TEST_ASSERT(strstr(j4, "\"requests_reused\":1") != NULL);
    free(j4);

    request_free(&r);
    monitor_test_server_free(s);
}

void test_monitor_recent_requests_default_twelve_or_all(void) {
    server *s = monitor_test_server();
    request r;
    request_init(&r, REQ_COMPLETION, 8);
    r.api = API_OPENAI;
    r.prompt.len = 4;
    for (int i = 0; i < 15; i++) {
        const uint64_t id = mon_begin(s, &r, "/v1/completions");
        mon_prefill(s, id, 4, 0, 8);
        mon_first_token(s, id);
        mon_token(s, id, 1);
        mon_end(s, id, i % 2 ? "length" : "stop", 1, -1, -1);
    }
    char *dflt = metrics_json(s, false);
    TEST_ASSERT(strstr(dflt, "\"requests_kept\":15") != NULL);
    TEST_ASSERT(count_substr(dflt, "\"finish\":") == 12);
    TEST_ASSERT(strstr(dflt, "\"drafts_offered\":null") != NULL);   /* 没投机 = null, 不是 0 */
    free(dflt);
    char *all = metrics_json(s, true);
    TEST_ASSERT(count_substr(all, "\"finish\":") == 15);
    free(all);
    request_free(&r);
    monitor_test_server_free(s);
}

void test_monitor_prometheus_text_uses_vllm_names(void) {
    server *s = monitor_test_server();
    request r;
    request_init(&r, REQ_CHAT, 8);
    r.api = API_ANTHROPIC;
    r.prompt.len = 7;
    const uint64_t id = mon_begin(s, &r, "/v1/messages");
    mon_prefill(s, id, 7, 0, 8);
    mon_first_token(s, id);
    /* token 之间隔 2 ms: macOS 的 CLOCK_MONOTONIC 只有微秒粒度, 三次调用挤在同一微秒里解码时长就是 0, 间隔直方图记不上(首跑实撞) */
    for (int t = 1; t <= 3; t++) { struct timespec ts = {0, 2 * 1000 * 1000}; nanosleep(&ts, NULL); mon_token(s, id, t); }
    mon_end(s, id, "stop", 3, 4, 2);
    buf b = {0};
    mon_prometheus_text(s, &b);
    /* 没有引擎 ⇒ 模型名空 ⇒ 标签落到 "ds4"(Strata 同样落到 "strata"); 真服务里是 GGUF 的模型名 */
    TEST_ASSERT(strstr(b.ptr, "# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{model_name=\"ds4\"} 0\n") != NULL);
    TEST_ASSERT(strstr(b.ptr, "vllm:request_success_total{model_name=\"ds4\"} 1\n") != NULL);
    TEST_ASSERT(strstr(b.ptr, "vllm:prompt_tokens_total{model_name=\"ds4\"} 7\n") != NULL);
    TEST_ASSERT(strstr(b.ptr, "vllm:generation_tokens_total{model_name=\"ds4\"} 3\n") != NULL);
    TEST_ASSERT(strstr(b.ptr, "vllm:spec_decode_num_draft_tokens_total{model_name=\"ds4\"} 4\n") != NULL);
    TEST_ASSERT(strstr(b.ptr, "vllm:spec_decode_num_accepted_tokens_total{model_name=\"ds4\"} 2\n") != NULL);
    TEST_ASSERT(strstr(b.ptr, "vllm:time_to_first_token_seconds_bucket{model_name=\"ds4\",le=\"+Inf\"} 1\n") != NULL);
    TEST_ASSERT(strstr(b.ptr, "vllm:inter_token_latency_seconds_count{model_name=\"ds4\"} 2\n") != NULL);   /* 3 个 token = 2 个间隔 */
    TEST_ASSERT(strstr(b.ptr, "vllm:e2e_request_latency_seconds_count{model_name=\"ds4\"} 1\n") != NULL);
    TEST_ASSERT(strstr(b.ptr, "# TYPE ds4:live_state gauge\n") != NULL);
    TEST_ASSERT(strstr(b.ptr, "ds4:live_state{model_name=\"ds4\",state=\"idle\"} 1\n") != NULL);
    TEST_ASSERT(strstr(b.ptr, "ds4:engine_max_context{model_name=\"ds4\"} 4096\n") != NULL);
    TEST_ASSERT(count_substr(b.ptr, "# TYPE ds4:live_state") == 1);   /* 一族只出一次 TYPE */
    TEST_ASSERT(strstr(b.ptr, "ds4:gpu_temp_celsius") == NULL || strstr(b.ptr, "ds4:gpu_temp_celsius{model_name=\"ds4\",gpu=\"0\"} ") != NULL);   /* 读不到就整条不出, 不出 0 */
    ds4_buf_free(&b);
    request_free(&r);
    monitor_test_server_free(s);
}

void test_monitor_accept_header_selects_prometheus(void) {
    const char *h1 = "GET /metrics HTTP/1.1\r\nHost: x\r\nAccept: text/plain; version=0.0.4\r\n\r\n";
    const char *h2 = "GET /metrics HTTP/1.1\r\nAccept: application/openmetrics-text,*/*\r\n\r\n";
    const char *h3 = "GET /metrics HTTP/1.1\r\nAccept: application/json\r\n\r\n";
    const char *h4 = "GET /metrics HTTP/1.1\r\nUser-Agent: curl\r\n\r\n";
    TEST_ASSERT(http_accepts_text(h1, strlen(h1)));
    TEST_ASSERT(http_accepts_text(h2, strlen(h2)));
    TEST_ASSERT(!http_accepts_text(h3, strlen(h3)));
    TEST_ASSERT(!http_accepts_text(h4, strlen(h4)));
}

/* 走真实的 HTTP 入口(client_main): 查询串与 Accept 头决定 /metrics 给 JSON 还是 Prometheus 文本 */
static char *http_roundtrip(server *s, const char *request_text) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return xstrdup("");
    int sockbuf = 512 * 1024;   /* 应答比 AF_UNIX 默认缓冲大, 且读在写完之后(同线程): 两端都放宽 */
    (void)setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &sockbuf, sizeof(sockbuf));
    (void)setsockopt(sv[1], SOL_SOCKET, SO_RCVBUF, &sockbuf, sizeof(sockbuf));
    TEST_ASSERT(send_all(sv[1], request_text, strlen(request_text)));
    client_arg *ca = xmalloc(sizeof *ca);
    ca->srv = s; ca->fd = sv[0];
    pthread_mutex_lock(&s->mu); s->clients++; pthread_mutex_unlock(&s->mu);
    (void)client_main(ca);   /* 同步跑: 它自己关 sv[0] */
    char *out = read_socket_text(sv[1]);
    close(sv[1]);
    return out;
}

void test_monitor_http_route_selects_json_or_prometheus(void) {
    server *s = monitor_test_server();
    pthread_cond_init(&s->clients_cv, NULL);
    char *json = http_roundtrip(s, "GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n");
    TEST_ASSERT(strstr(json, "HTTP/1.1 200 OK") != NULL);
    TEST_ASSERT(strstr(json, "Content-Type: application/json") != NULL);
    TEST_ASSERT(strstr(json, "{\"engine\":{\"model\":\"\"") != NULL);
    free(json);
    char *prom_q = http_roundtrip(s, "GET /metrics?format=prometheus HTTP/1.1\r\nHost: x\r\n\r\n");
    TEST_ASSERT(strstr(prom_q, "Content-Type: text/plain; version=0.0.4; charset=utf-8") != NULL);
    TEST_ASSERT(strstr(prom_q, "\n# HELP vllm:num_requests_running") != NULL);
    free(prom_q);
    char *prom_a = http_roundtrip(s, "GET /metrics HTTP/1.1\r\nAccept: application/openmetrics-text\r\n\r\n");
    TEST_ASSERT(strstr(prom_a, "Content-Type: text/plain; version=0.0.4") != NULL);
    free(prom_a);
    char *page = http_roundtrip(s, "GET /monitor?x=1 HTTP/1.1\r\nHost: x\r\n\r\n");
    TEST_ASSERT(strstr(page, "HTTP/1.1 200 OK") != NULL);
    TEST_ASSERT(strstr(page, "DwarfStar 监控") != NULL);
    free(page);
    char *missing = http_roundtrip(s, "GET /nope HTTP/1.1\r\n\r\n");
    TEST_ASSERT(strstr(missing, "HTTP/1.1 404") != NULL);
    free(missing);
    pthread_cond_destroy(&s->clients_cv);
    monitor_test_server_free(s);
}

void test_monitor_page_route_serves_html(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;
    int sockbuf = 512 * 1024;   /* 页面比 AF_UNIX 默认缓冲大, 同 chat 页测试 */
    TEST_ASSERT(setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &sockbuf, sizeof(sockbuf)) == 0);
    TEST_ASSERT(setsockopt(sv[1], SOL_SOCKET, SO_RCVBUF, &sockbuf, sizeof(sockbuf)) == 0);
    TEST_ASSERT(serve_page_file(sv[0], false, DS4_MONITOR_PAGE_FILE));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
    TEST_ASSERT(strstr(out, "Content-Type: text/html; charset=utf-8") != NULL);
    TEST_ASSERT(strstr(out, "/metrics") != NULL);
    TEST_ASSERT(strstr(out, "innerHTML") == NULL);   /* 页面只用 textContent/DOM 填数据, 服务端串不进 HTML */
    free(out);
    close(sv[0]);
    close(sv[1]);

    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;
    serve_page_file(sv[0], false, "web/__no_such_page__.html");
    shutdown(sv[0], SHUT_WR);
    out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "HTTP/1.1 404") != NULL);
    TEST_ASSERT(strstr(out, "web/__no_such_page__.html") != NULL);
    free(out);
    close(sv[0]);
    close(sv[1]);
}
