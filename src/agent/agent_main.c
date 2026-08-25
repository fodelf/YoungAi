/* agent_main.c — 机械拆分自 ds4_agent.c: Interactive Runtime Loop (2/2: 交互主循环 + main)。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

/* Main UI loop.  poll() multiplexes stdin with the worker wake pipe; all
 * terminal writes go through editor_write_async() so linenoise, status footer,
 * model output, and tool output never race each other. */
static int run_agent(ds4_engine *engine, agent_config *cfg) {
    agent_worker worker;
    if (agent_worker_init(&worker, engine, cfg) != 0) return 1;

    char hist[PATH_MAX];
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    snprintf(hist, sizeof(hist), "%s/.ds4_agent_history", home);
    /* The agent uses ANSI scroll regions when possible: model/tool output
     * scrolls above the live linenoise prompt and status footer, so streaming
     * tokens do not require repainting the bottom rows.  Terminals without
     * scroll-region support fall back to the older prompt-below-output path. */
    linenoiseSetMultiLine(1);
    linenoiseHistorySetMaxLen(512);
    linenoiseHistoryLoad(hist);
    agent_completion_worker = &worker;
    linenoiseSetCompletionCallback(agent_switch_completion_callback);

    agent_status st;
    worker_get_status(&worker, &st);
    char prompt[160];
    char statusline[4096];
    build_prompt_text(&st, prompt, sizeof(prompt));
    build_footer_text(&st, NULL, 80, statusline, sizeof(statusline));

    agent_editor editor = {0};
    agent_prompt_queue queue = {0};
    if (editor_start(&editor, prompt, statusline, NULL) != 0) {
        fprintf(stderr, "ds4-agent: failed to start line editor\n");
        agent_worker_free(&worker);
        return 1;
    }
    editor_write_welcome_banner(&editor, cfg, prompt, statusline);

    char *initial_pending = cfg->gen.prompt && cfg->gen.prompt[0] ?
                            xstrdup(cfg->gen.prompt) : NULL;

    bool running = true;
    bool exit_save_handled = false;
    bool show_welcome_after_restart = false;
    bool force_status_redraw_after_restart = false;
    char *restore_line = NULL;
    while (running) {
        struct pollfd pfd[2] = {
            {.fd = STDIN_FILENO, .events = POLLIN},
            {.fd = worker.wake_fd[0], .events = POLLIN},
        };
        int timeout = (!editor.paste_open && !editor.paste_start_pending &&
                       linenoiseEditQueuedInput(&editor.edit) > 0) ? 0 : 100;
        int rc = poll(pfd, 2, timeout);
        if (rc < 0 && errno != EINTR) break;

        if (agent_sigint) {
            agent_sigint = 0;
            if (worker_is_idle(&worker)) {
                editor_cancel_input_with_hint(&editor, prompt, statusline);
            } else {
                worker_interrupt(&worker);
            }
        }

        if (rc > 0 && (pfd[1].revents & POLLIN)) drain_wake_fd(worker.wake_fd[0]);

        char *out = NULL;
        size_t out_len = 0;
        worker_consume(&worker, &out, &out_len, &st);
        build_prompt_text(&st, prompt, sizeof(prompt));
        int footer_cols = editor.edit.cols > 0 ? (int)editor.edit.cols : 80;
        build_footer_text(&st, &queue, footer_cols, statusline, sizeof(statusline));
        if (out && out_len) {
            bool force_show = st.state == AGENT_WORKER_IDLE ||
                              st.state == AGENT_WORKER_ERROR ||
                              st.state == AGENT_WORKER_STOPPED;
            editor_write_async(&editor, out, out_len, prompt, statusline, force_show);
        } else {
            editor_set_prompt_status(&editor, prompt, statusline);
            if (editor.hidden && (st.state == AGENT_WORKER_IDLE ||
                                  st.state == AGENT_WORKER_ERROR ||
                                  st.state == AGENT_WORKER_STOPPED))
                editor_show(&editor);
        }
        if (st.state == AGENT_WORKER_ERROR && st.error[0]) {
            char msg[320];
            int n = snprintf(msg, sizeof(msg), "\nds4-agent: %s\n", st.error);
            editor_write_async(&editor, msg, n > 0 ? (size_t)n : 0,
                               prompt, statusline, true);
            pthread_mutex_lock(&worker.mu);
            worker.status.state = AGENT_WORKER_IDLE;
            worker.status.error[0] = '\0';
            pthread_mutex_unlock(&worker.mu);
        }
        free(out);

        if (worker_take_queued_user_drain_request(&worker)) {
            char *echo = agent_prompt_queue_take_all_echo(&queue);
            char *queued = agent_prompt_queue_take_all(&queue);
            if (echo) {
                build_footer_text(&st, &queue, footer_cols, statusline, sizeof(statusline));
                editor_write_async(&editor, echo, strlen(echo), prompt, statusline, true);
                free(echo);
            }
            worker_answer_queued_user_drain(&worker, queued);
            continue;
        }

        char web_approval_msg[256];
        if (worker_take_web_approval_request(&worker, web_approval_msg,
                                             sizeof(web_approval_msg)))
        {
            char *saved_input = NULL;
            if (editor.active && editor.edit.buf && editor.edit.len)
                saved_input = xstrndup(editor.edit.buf, editor.edit.len);
            editor_stop(&editor);
            editor_restore_terminal_layout(&editor);
            agent_yes_no_options approval_opts = {
                .timeout_sec = 30,
                .timeout_answer = AGENT_YES_NO_AUTO_NO,
            };
            bool approval_timed_out = false;
            bool allow = agent_prompt_yes_no_ex(web_approval_msg,
                                                &approval_opts,
                                                &approval_timed_out);
            worker_answer_web_approval(&worker, allow,
                approval_timed_out ? "Chrome browser start approval timed out" : NULL);
            worker_get_status(&worker, &st);
            build_prompt_text(&st, prompt, sizeof(prompt));
            int restart_cols = editor.edit.cols > 0 ? (int)editor.edit.cols : 80;
            build_footer_text(&st, &queue, restart_cols, statusline, sizeof(statusline));
            editor_start(&editor, prompt, statusline, saved_input);
            free(saved_input);
            continue;
        }

        if (initial_pending && worker_is_idle(&worker)) {
            if (worker_submit(&worker, initial_pending)) {
                free(initial_pending);
                initial_pending = NULL;
            }
        }

        if (!initial_pending && queue.len && worker_is_idle(&worker)) {
            char *echo = agent_prompt_queue_take_all_echo(&queue);
            char *queued = agent_prompt_queue_take_all(&queue);
            if (worker_submit(&worker, queued)) {
                linenoiseHistoryAdd(queued);
                linenoiseHistorySave(hist);
                build_footer_text(&st, &queue, footer_cols, statusline, sizeof(statusline));
                if (echo)
                    editor_write_async(&editor, echo, strlen(echo), prompt, statusline, true);
            } else {
                agent_prompt_queue_push_front(&queue, queued);
                queued = NULL;
            }
            free(echo);
            free(queued);
        }

        if (rc > 0 && (pfd[0].revents & POLLIN)) editor_read_stdin(&editor);

        if (queue.len && editor_take_queued_byte(&editor, 24)) { /* Ctrl+X */
            char *queued = agent_prompt_queue_pop(&queue);
            editor_replace_input(&editor, queued);
            worker_get_status(&worker, &st);
            build_prompt_text(&st, prompt, sizeof(prompt));
            footer_cols = editor.edit.cols > 0 ? (int)editor.edit.cols : 80;
            build_footer_text(&st, &queue, footer_cols, statusline, sizeof(statusline));
            editor_set_prompt_status(&editor, prompt, statusline);
            free(queued);
        }
        if (queue.len && !worker_is_idle(&worker) && editor_take_bare_escape(&editor)) {
            worker_interrupt(&worker);
        }

        if (!editor.paste_open && !editor.paste_start_pending &&
            linenoiseEditQueuedInput(&editor.edit) > 0)
        {
            if (editor.hidden) {
                /* A user key while the model is in the middle of a partial
                 * output line means the prompt must become visible again. End
                 * the model line explicitly; otherwise linenoise would redraw
                 * on top of generated text. */
                editor_show(&editor);
            }
            errno = 0;
            char *line = linenoiseEditFeed(&editor.edit);
            if (line == linenoiseEditMore) {
                /* Still editing. */
            } else if (!line) {
                if (errno == EAGAIN) {
                    if (!worker_is_idle(&worker)) {
                        worker_interrupt(&worker);
                    } else {
                        editor_cancel_input_with_hint(&editor, prompt, statusline);
                    }
                } else {
                    running = false;
                }
            } else {
                char *cmd = line;
                while (*cmd == ' ' || *cmd == '\t' || *cmd == '\r' || *cmd == '\n') cmd++;
                char *end = cmd + strlen(cmd);
                while (end > cmd && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) end--;
                *end = '\0';

                bool was_below_output = editor.prompt_below_output;
                bool had_output_line_open = editor.output_line_open;
                int saved_output_col = editor.output_col;
                editor_stop(&editor);
                bool busy = !worker_is_idle(&worker);
                if (!cmd[0]) {
                    /* Empty input: just reopen the editor. */
                } else if (!strcmp(cmd, "/help")) {
                    runtime_help();
                } else if (!strcmp(cmd, "/save")) {
                    if (busy) {
                        worker_request_save(&worker);
                        printf("save scheduled at next safe point\n");
                    } else {
                        char err[160] = {0};
                        if (!agent_worker_save_session(&worker, err, sizeof(err)))
                            printf("save failed: %s\n", err);
                    }
                } else if (!strcmp(cmd, "/compact")) {
                    worker_request_compact(&worker);
                    if (busy)
                        printf("compaction scheduled at next safe point\n");
                } else if (!strcmp(cmd, "/list")) {
                    agent_worker_list_sessions(&worker);
                } else if (!strncmp(cmd, "/corr", 5) &&
                           (cmd[5] == '\0' || cmd[5] == ' ' || cmd[5] == '\t')) {
                    /* R3-h multi-domain sidecar plugin: /corr lists the
                     * gguf/sidecars/ registry; /corr <name|path|off> swaps
                     * the z-domain between generations (~90MB, seconds). */
                    char *arg = cmd + 5;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    if (!arg[0]) {
                        printf("sidecar registry (gguf/sidecars/):\n");
                        DIR *dp = opendir("gguf/sidecars");
                        if (dp) {
                            struct dirent *de;
                            while ((de = readdir(dp)) != NULL) {
                                size_t n = strlen(de->d_name);
                                if (n > 5 && !strcmp(de->d_name + n - 5, ".gguf"))
                                    printf("  %.*s\n", (int)(n - 5), de->d_name);
                            }
                            closedir(dp);
                        } else {
                            printf("  (no gguf/sidecars/ directory)\n");
                        }
                        printf("usage: /corr <name|path|off>\n");
                    } else {
                        char path[512];
                        if (!strcmp(arg, "off")) {
                            path[0] = '\0';
                        } else if (strchr(arg, '/')) {
                            snprintf(path, sizeof(path), "%s", arg);
                        } else {
                            snprintf(path, sizeof(path), "gguf/sidecars/%s.gguf", arg);
                        }
                        if (ds4_engine_corr_switch(engine, path[0] ? path : NULL) == 0)
                            printf("corr domain -> %s\n", path[0] ? path : "(off, pure 1-bit)");
                        else
                            printf("corr switch failed (kept current): %s\n", path);
                    }
                } else if (!strncmp(cmd, "/power", 6) &&
                           (cmd[6] == '\0' || cmd[6] == ' ' || cmd[6] == '\t')) {
                    char *arg = cmd + 6;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    if (!arg[0]) {
                        printf("usage: /power <1..100>\n");
                    } else {
                        int power = 0;
                        if (!parse_power_percent(arg, &power)) {
                            printf("usage: /power <1..100>\n");
                        } else {
                            worker_request_power(&worker, power);
                        }
                    }
                } else if (cmd[0] == '/' && !agent_slash_command_known(cmd)) {
                    ssize_t ignored = write(STDOUT_FILENO, "\a", 1);
                    (void)ignored;
                    restore_line = xstrdup(cmd);
                } else if (cmd[0] == '/' && busy) {
                    printf("command requires the model to be idle: %s\n", cmd);
                } else if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) {
                    editor_restore_terminal_layout(&editor);
                    agent_exit_save_result exit_save =
                        agent_maybe_save_before_exiting(&worker);
                    if (exit_save == AGENT_EXIT_NOW) {
                        exit(0);
                    } else if (exit_save == AGENT_EXIT_CLEAN) {
                        exit_save_handled = true;
                        running = false;
                    }
                } else if (!strcmp(cmd, "/new")) {
                    editor_restore_terminal_layout(&editor);
                    if (agent_maybe_save_before_leaving_session(&worker)) {
                        char err[160] = {0};
                        if (!agent_worker_reset_to_sysprompt(&worker, err, sizeof(err))) {
                            printf("new session failed: %s\n", err);
                        } else {
                            show_welcome_after_restart = true;
                        }
                    }
                } else if (!strncmp(cmd, "/switch", 7) &&
                           (cmd[7] == '\0' || cmd[7] == ' ' || cmd[7] == '\t')) {
                    char *arg = cmd + 7;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    if (!arg[0]) {
                        printf("usage: /switch <sha-prefix>\n");
                    } else {
                        editor_restore_terminal_layout(&editor);
                        if (agent_maybe_save_before_leaving_session(&worker)) {
                            char *sha = arg;
                            while (*arg && *arg != ' ' && *arg != '\t') arg++;
                            if (*arg) *arg = '\0';
                            char err[160] = {0};
                            if (!agent_worker_switch_session(&worker, sha,
                                                             AGENT_HISTORY_DEFAULT_TURNS,
                                                             err, sizeof(err)))
                                printf("switch failed: %s\n", err);
                            else
                                force_status_redraw_after_restart = true;
                        }
                    }
                } else if (!strncmp(cmd, "/del", 4) &&
                           (cmd[4] == '\0' || cmd[4] == ' ' || cmd[4] == '\t')) {
                    char *arg = cmd + 4;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    if (!arg[0]) {
                        printf("usage: /del <sha-prefix>\n");
                    } else {
                        char *sha_arg = arg;
                        while (*arg && *arg != ' ' && *arg != '\t') arg++;
                        if (*arg) *arg = '\0';
                        char sha[41] = {0};
                        char err[160] = {0};
                        if (agent_worker_delete_session(&worker, sha_arg,
                                                        sha, err, sizeof(err)))
                            printf("deleted session %.8s\n", sha);
                        else
                            printf("delete failed: %s\n", err);
                    }
                } else if (!strncmp(cmd, "/strip", 6) &&
                           (cmd[6] == '\0' || cmd[6] == ' ' || cmd[6] == '\t')) {
                    char *arg = cmd + 6;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    if (!arg[0]) {
                        printf("usage: /strip <sha-prefix>\n");
                    } else {
                        char *sha_arg = arg;
                        while (*arg && *arg != ' ' && *arg != '\t') arg++;
                        if (*arg) *arg = '\0';
                        char sha[41] = {0};
                        uint32_t tokens = 0;
                        char err[160] = {0};
                        if (agent_worker_strip_session(&worker, sha_arg,
                                                       sha, &tokens,
                                                       err, sizeof(err)))
                            printf("stripped session %.8s (%u tokens)\n",
                                   sha, tokens);
                        else
                            printf("strip failed: %s\n", err);
                    }
                } else if (!strncmp(cmd, "/history", 8) &&
                           (cmd[8] == '\0' || cmd[8] == ' ' || cmd[8] == '\t')) {
                    char *arg = cmd + 8;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    int history_turns = arg[0] ?
                        agent_parse_int_default(arg, AGENT_HISTORY_DEFAULT_TURNS,
                                                1, AGENT_HISTORY_MAX_TURNS) :
                        AGENT_HISTORY_DEFAULT_TURNS;
                    char err[160] = {0};
                    if (!agent_worker_show_history(&worker, history_turns,
                                                   err, sizeof(err)))
                        printf("history failed: %s\n", err);
                } else if (busy) {
                    agent_prompt_queue_push(&queue, cmd);
                } else {
                    linenoiseHistoryAdd(cmd);
                    linenoiseHistorySave(hist);
                    if (worker_submit(&worker, cmd)) {
                        agent_echo_user_prompt(cmd);
                    } else {
                        restore_line = xstrdup(cmd);
                    }
                }
                linenoiseFree(line);

                if (running) {
                    worker_get_status(&worker, &st);
                    build_prompt_text(&st, prompt, sizeof(prompt));
                    int restart_cols = editor.edit.cols > 0 ? (int)editor.edit.cols : 80;
                    build_footer_text(&st, &queue, restart_cols, statusline, sizeof(statusline));
                    editor_start(&editor, prompt, statusline, restore_line);
                    if (!editor.scroll_region && was_below_output) {
                        editor.output_line_open = had_output_line_open;
                        editor.prompt_below_output = was_below_output;
                        editor.output_col = saved_output_col;
                    }
                    if (show_welcome_after_restart) {
                        editor_write_welcome_banner(&editor, cfg, prompt, statusline);
                        show_welcome_after_restart = false;
                    }
                    if (force_status_redraw_after_restart) {
                        editor_write_async(&editor, "", 0, prompt, statusline, true);
                        force_status_redraw_after_restart = false;
                    }
                    free(restore_line);
                    restore_line = NULL;
                }
            }
        }
    }

    free(initial_pending);
    free(restore_line);
    agent_prompt_queue_free(&queue);
    editor_stop(&editor);
    editor_restore_terminal_layout(&editor);
    linenoiseSetCompletionCallback(NULL);
    agent_completion_worker = NULL;
    if (!exit_save_handled) {
        agent_exit_save_result exit_save =
            agent_maybe_save_before_exiting(&worker);
        if (exit_save == AGENT_EXIT_NOW) exit(0);
    }
    agent_worker_free(&worker);
    return 0;
}

int main(int argc, char **argv) {
    agent_config cfg = parse_options(argc, argv);
    if (cfg.chdir_path && chdir(cfg.chdir_path) != 0) {
        fprintf(stderr, "ds4-agent: failed to chdir to %s: %s\n",
                cfg.chdir_path, strerror(errno));
        return 1;
    }
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &cfg.engine) != 0) return 1;
    log_context_memory(cfg.engine.backend, cfg.gen.ctx_size);

    struct sigaction old_int;
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = agent_sigint_handler;
    bool sigint_installed = !cfg.non_interactive &&
        sigaction(SIGINT, &sa, &old_int) == 0;

    int rc = cfg.non_interactive ?
        run_agent_non_interactive(engine, &cfg) :
        run_agent(engine, &cfg);

    if (sigint_installed) sigaction(SIGINT, &old_int, NULL);
    ds4_engine_close(engine);
    return rc;
}
