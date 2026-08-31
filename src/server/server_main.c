/* server_main.c — ds4_server.c 拆分后的生产入口: main() 与信号/服务装配。
 * 其余全部域见同目录 server_*.c; 内部接口在 server_internal.h。 */
#include "server_internal.h"

#ifndef DS4_SERVER_TEST
int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = stop_signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    server_config cfg = parse_options(argc, argv);
    if (cfg.chdir_path && chdir(cfg.chdir_path) != 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: failed to chdir to %s: %s",
                   cfg.chdir_path, strerror(errno));
        return 1;
    }

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &cfg.engine) != 0) return 1;

    log_context_memory(cfg.engine.backend, cfg.ctx_size);
    if (cfg.engine.distributed.role == DS4_DISTRIBUTED_COORDINATOR &&
        cfg.kv_cache.continued_interval_tokens > 0) {
        /* Mid-prefill continued checkpoints need a QUIESCENT frontier to stage
         * the worker KV snapshot; dual-host pipelined prefill has none
         * (measured 2026-07-07: mid-flight staging desyncs the worker snapshot
         * -> "distributed result metadata mismatch" -> route collapse). Cold /
         * evict / shutdown saves happen at quiescent points and stay enabled. */
        server_log(DS4_LOG_DEFAULT,
                   "ds4-server: continued KV checkpoints disabled for the distributed coordinator "
                   "(mid-prefill staging needs a quiescent frontier)");
        cfg.kv_cache.continued_interval_tokens = 0;
    }
    if (cfg.engine.distributed.role == DS4_DISTRIBUTED_WORKER) {
        ds4_dist_generation_options gen = {
            .ctx_size = cfg.ctx_size,
        };
        int rc = ds4_dist_run(engine, &cfg.engine.distributed, &gen);
        ds4_engine_close(engine);
        return rc;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_size) != 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: failed to create %s session",
                   ds4_backend_name(cfg.engine.backend));
        ds4_engine_close(engine);
        return 1;
    }

    server s;
    memset(&s, 0, sizeof(s));
    s.engine = engine;
    s.session = session;
    s.default_tokens = cfg.default_tokens;
    s.max_output_tokens = cfg.max_output_tokens;
    s.force_nothink = cfg.force_nothink;
    s.tool_primer = cfg.tool_primer;
    g_force_nothink = cfg.force_nothink;
    s.disable_exact_dsml_tool_replay = cfg.disable_exact_dsml_tool_replay;
    s.tool_mem.max_entries = cfg.tool_memory_max_ids;
    s.enable_cors = cfg.enable_cors;
    if (cfg.kv_disk_dir) {
        kv_cache_open(&s.kv, cfg.kv_disk_dir, cfg.kv_disk_space_mb,
                      cfg.kv_cache_reject_different_quant, cfg.kv_cache);
    }
    if (s.disable_exact_dsml_tool_replay) {
        server_log(DS4_LOG_DEFAULT,
                   "ds4-server: exact DSML tool replay disabled; tool history uses canonical JSON rendering");
    }
    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cv, NULL);
    pthread_cond_init(&s.clients_cv, NULL);
    pthread_mutex_init(&s.tool_mu, NULL);
    pthread_mutex_init(&s.trace_mu, NULL);
    if (cfg.trace_path) {
        s.trace = fopen(cfg.trace_path, "w");
        if (!s.trace) {
            server_log(DS4_LOG_DEFAULT, "ds4-server: failed to open trace file %s: %s",
                       cfg.trace_path, strerror(errno));
            server_close_resources(&s);
            return 1;
        }
        setvbuf(s.trace, NULL, _IONBF, 0);
        server_log(DS4_LOG_DEFAULT, "ds4-server: tracing session to %s", cfg.trace_path);
    }

    pthread_t worker;
    s.ctx_size = cfg.ctx_size;
    s.batch_max = cfg.batch_max;
    if (s.batch_max >= 2)
        server_log(DS4_LOG_GENERATION,
                   "ds4-server: 并发批处理已开 (最多 %d 路合批; 仅非流式/无工具的 chat 请求)",
                   s.batch_max);
    if (pthread_create(&worker, NULL, worker_main, &s) != 0) die("failed to start worker");

    int lfd = listen_on(cfg.host, cfg.port);
    if (lfd < 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: failed to listen on %s:%d: %s", cfg.host, cfg.port, strerror(errno));
        pthread_mutex_lock(&s.mu);
        s.stopping = true;
        pthread_cond_broadcast(&s.cv);
        pthread_mutex_unlock(&s.mu);
        pthread_join(worker, NULL);
        server_close_resources(&s);
        return 1;
    }
    g_listen_fd = lfd;
    server_log(DS4_LOG_DEFAULT, "ds4-server: listening on http://%s:%d", cfg.host, cfg.port);
    {
        struct stat page_st;
        if (stat(DS4_CHAT_PAGE_FILE, &page_st) == 0) {
            server_log(DS4_LOG_DEFAULT, "ds4-server: browser chat page on http://%s:%d/",
                       cfg.host, cfg.port);
        } else {
            server_log(DS4_LOG_DEFAULT,
                       "ds4-server: %s not found from this working directory; "
                       "GET / will 404 (--chdir to the repo root enables the chat page)",
                       DS4_CHAT_PAGE_FILE);
        }
    }

    while (!g_stop_requested) {
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0) {
            if (g_stop_requested) break;
            if (errno == EINTR) continue;
            server_log(DS4_LOG_DEFAULT, "ds4-server: accept failed: %s", strerror(errno));
            continue;
        }
        if (g_stop_requested) {
            close(fd);
            break;
        }

        configure_client_socket(fd);
        client_arg *ca = xmalloc(sizeof(*ca));
        ca->srv = &s;
        ca->fd = fd;
        pthread_mutex_lock(&s.mu);
        s.clients++;
        pthread_mutex_unlock(&s.mu);
        pthread_t th;
        if (pthread_create(&th, NULL, client_main, ca) != 0) {
            pthread_mutex_lock(&s.mu);
            s.clients--;
            pthread_cond_broadcast(&s.clients_cv);
            pthread_mutex_unlock(&s.mu);
            free(ca);
            close(fd);
            continue;
        }
        pthread_detach(th);
    }
    if (g_listen_fd >= 0) {
        close(lfd);
        g_listen_fd = -1;
    }

    server_log(DS4_LOG_DEFAULT, "ds4-server: shutdown requested, draining requests");
    pthread_mutex_lock(&s.mu);
    s.stopping = true;
    pthread_cond_broadcast(&s.cv);
    pthread_mutex_unlock(&s.mu);
    pthread_join(worker, NULL);
    pthread_mutex_lock(&s.mu);
    while (s.clients > 0) pthread_cond_wait(&s.clients_cv, &s.mu);
    pthread_mutex_unlock(&s.mu);

    const ds4_tokens *tokens = ds4_session_tokens(s.session);
    if (s.kv.enabled && tokens && tokens->len >= s.kv.opt.min_tokens) {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: persisting current KV cache before shutdown tokens=%d",
                   tokens->len);
        kv_cache_store_current(&s, "shutdown");
    }
    server_close_resources(&s);
    return 0;
}
#endif /* DS4_SERVER_TEST */
