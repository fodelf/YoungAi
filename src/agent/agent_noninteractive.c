/* agent_noninteractive.c — 机械拆分自 ds4_agent.c: Interactive Runtime Loop (1/2: 非交互模式)。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

/* ============================================================================
 * Interactive Runtime Loop
 * ============================================================================
 */

static void agent_noninteractive_marker(const char *msg) {
    write_all(STDERR_FILENO, msg, strlen(msg));
    write_all(STDERR_FILENO, "\n", 1);
}

static int agent_read_stdin_available(agent_input_buf *in, bool *eof) {
    char buf[4096];
    for (;;) {
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n > 0) {
            agent_input_buf_append(in, buf, (size_t)n);
            continue;
        }
        if (n == 0) {
            *eof = true;
            return 0;
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
        perror("ds4-agent: read stdin");
        return -1;
    }
}

/* Headless mode is intentionally just another front-end for the same worker.
 * With -p/--prompt it is a one-shot execution.  Without -p it becomes a small
 * stdin protocol: announce readiness on stderr, collect bytes until stdin has
 * been quiet for 200 ms, submit that buffer as one prompt, and keep reading so
 * later input can be queued while the model is still working. */
int run_agent_non_interactive(ds4_engine *engine, agent_config *cfg) {
    agent_worker worker;
    if (agent_worker_init(&worker, engine, cfg) != 0) return 1;

    const bool one_shot = cfg->gen.prompt != NULL;
    bool one_shot_submitted = false;
    bool stdin_eof = false;
    bool waiting_announced = false;
    bool stdin_nonblock = false;
    int old_stdin_flags = 0;
    agent_input_buf input = {0};
    agent_prompt_queue queue = {0};
    double quiet_deadline = 0.0;
    int rc = 0;

    if (!one_shot) {
        if (set_nonblock(STDIN_FILENO, true, &old_stdin_flags) != 0) {
            perror("ds4-agent: nonblocking stdin");
            agent_worker_free(&worker);
            return 1;
        }
        stdin_nonblock = true;
    }

    while (true) {
        bool initialized = worker_is_initialized(&worker, NULL);
        bool idle = worker_is_idle(&worker);

        if (one_shot && !one_shot_submitted && initialized) {
            if (worker_submit(&worker, cfg->gen.prompt))
                one_shot_submitted = true;
            idle = false;
        }

        if (!one_shot && queue.len && idle) {
            char *queued = agent_prompt_queue_take_all(&queue);
            if (worker_submit(&worker, queued)) {
                idle = false;
            } else {
                agent_prompt_queue_push_front(&queue, queued);
                queued = NULL;
            }
            free(queued);
        }

        if (!one_shot && initialized && idle && !queue.len &&
            input.len == 0 && !stdin_eof && !waiting_announced)
        {
            agent_noninteractive_marker("+DWARFSTAR_WAITING");
            waiting_announced = true;
        }

        int timeout_ms = -1;
        if (!one_shot && input.len > 0) {
            double rem = quiet_deadline - now_sec();
            timeout_ms = rem <= 0.0 ? 0 : (int)(rem * 1000.0) + 1;
        }

        struct pollfd pfd[2];
        int nfds = 0;
        int wake_idx = nfds;
        pfd[nfds++] = (struct pollfd){.fd = worker.wake_fd[0], .events = POLLIN};
        int stdin_idx = -1;
        if (!one_shot && initialized && !stdin_eof) {
            stdin_idx = nfds;
            pfd[nfds++] = (struct pollfd){.fd = STDIN_FILENO, .events = POLLIN};
        }

        int prc = poll(pfd, (nfds_t)nfds, timeout_ms);
        if (prc < 0) {
            if (errno == EINTR) continue;
            perror("ds4-agent: poll");
            rc = 1;
            break;
        }
        if (pfd[wake_idx].revents & POLLIN) drain_wake_fd(worker.wake_fd[0]);
        if (stdin_idx >= 0 && (pfd[stdin_idx].revents & (POLLIN | POLLHUP))) {
            size_t old_len = input.len;
            if (agent_read_stdin_available(&input, &stdin_eof) != 0) {
                rc = 1;
                break;
            }
            if (input.len != old_len) {
                quiet_deadline = now_sec() + 0.200;
                waiting_announced = false;
            }
        }

        char *out = NULL;
        size_t out_len = 0;
        agent_status st = {0};
        worker_consume(&worker, &out, &out_len, &st);
        if (out && out_len) {
            write_all(STDOUT_FILENO, out, out_len);
            fflush(stdout);
        }
        free(out);

        if (worker_take_queued_user_drain_request(&worker)) {
            char *queued = agent_prompt_queue_take_all(&queue);
            worker_answer_queued_user_drain(&worker, queued);
        }

        if (st.state == AGENT_WORKER_ERROR) {
            fprintf(stderr, "ds4-agent: %s\n",
                    st.error[0] ? st.error : "worker error");
            rc = 1;
            break;
        }

        if (!one_shot && input.len > 0 &&
            (stdin_eof || now_sec() >= quiet_deadline))
        {
            char *prompt = agent_input_buf_take(&input);
            if (worker_is_idle(&worker) && queue.len == 0) {
                if (!worker_submit(&worker, prompt)) {
                    agent_prompt_queue_push(&queue, prompt);
                    agent_noninteractive_marker("+DWARFSTAR_QUEUED");
                }
            } else {
                agent_prompt_queue_push(&queue, prompt);
                agent_noninteractive_marker("+DWARFSTAR_QUEUED");
            }
            free(prompt);
            waiting_announced = false;
        }

        if (one_shot && one_shot_submitted && worker_is_idle(&worker)) break;
        if (!one_shot && stdin_eof && input.len == 0 &&
            queue.len == 0 && worker_is_idle(&worker))
            break;
    }

    /* Drain anything published between the final status transition and the
     * loop exit.  This keeps stdout complete without adding another protocol. */
    char *out = NULL;
    size_t out_len = 0;
    worker_consume(&worker, &out, &out_len, NULL);
    if (out && out_len) {
        write_all(STDOUT_FILENO, out, out_len);
        fflush(stdout);
    }
    free(out);

    if (stdin_nonblock) fcntl(STDIN_FILENO, F_SETFL, old_stdin_flags);
    agent_input_buf_free(&input);
    agent_prompt_queue_free(&queue);
    agent_worker_free(&worker);
    return rc;
}
