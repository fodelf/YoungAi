/* agent_sessions_list.c — 机械拆分自 ds4_agent.c: Session Listing, History Rendering, And Completion (2/3: 历史渲染 + 列表/补全)。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

/* Re-render saved transcript text for /history and /switch.  It intentionally
 * uses the same assistant/token renderer as live output, so restored history
 * looks like the original terminal stream instead of raw rendered-chat text. */
static void agent_history_render_text(agent_worker *w, const char *text,
                                      size_t len, int user_turns) {
    if (user_turns <= 0) return;
    if (user_turns > AGENT_HISTORY_MAX_TURNS)
        user_turns = AGENT_HISTORY_MAX_TURNS;

    const char *end = text + len;
    agent_history_render_compaction_summary(w, text, len);

    bool tool_only = false;
    const char *p = agent_history_start_for_turns(text, len, user_turns,
                                                  &tool_only);
    if (p >= end) {
        agent_publish(w, "\n(no user history)\n", strlen("\n(no user history)\n"));
        return;
    }

    bool color = isatty(STDOUT_FILENO) != 0;
    if (color) agent_publish(w, "\n\x1b[90m", strlen("\n\x1b[90m"));
    else agent_publish(w, "\n", 1);
    if (tool_only) {
        agent_publishf(w, "--- session history: recent tool/assistant events ---\n");
    } else {
        agent_publishf(w, "--- session history: last %d user turn%s ---\n",
                       user_turns, user_turns == 1 ? "" : "s");
    }
    if (color) agent_publish(w, "\x1b[0m", 4);

    while (p < end) {
        agent_history_mark mark = AGENT_HISTORY_MARK_NONE;
        size_t mark_len = 0;
        const char *m = agent_history_next_marker(p, end, &mark, &mark_len);
        if (!m) break;
        const char *content = m + mark_len;
        agent_history_mark next_mark = AGENT_HISTORY_MARK_NONE;
        size_t next_len = 0;
        const char *next = agent_history_next_marker(content, end,
                                                     &next_mark, &next_len);
        const char *content_end = next ? next : end;
        const char *tp = content, *te = content_end;
        agent_history_trim(&tp, &te);

        if (mark == AGENT_HISTORY_MARK_USER) {
            if (agent_history_is_tool_user(tp, te)) {
                if (color) {
                    const char *s = "\x1b[90mTool result:\n";
                    agent_publish(w, s, strlen(s));
                } else {
                    agent_publish(w, "Tool result:\n", strlen("Tool result:\n"));
                }
                agent_history_publish_limited(w, tp, te, 12, 3000);
                if (color) agent_publish(w, "\x1b[0m", 4);
            } else {
                if (color) {
                    const char *s = "\x1b[1;32mUser:\x1b[0m\n";
                    agent_publish(w, s, strlen(s));
                } else {
                    agent_publish(w, "User:\n", strlen("User:\n"));
                }
                agent_history_publish_limited(w, tp, te, 24, 6000);
            }
        } else if (mark == AGENT_HISTORY_MARK_ASSISTANT) {
            if (color) {
                const char *s = "\x1b[1;37mAssistant:\x1b[0m\n";
                agent_publish(w, s, strlen(s));
            } else {
                agent_publish(w, "Assistant:\n", strlen("Assistant:\n"));
            }
            agent_history_render_assistant(w, tp, te);
        }
        p = content_end;
    }

    if (color) {
        const char *s = "\x1b[90m--- end history ---\x1b[0m\n";
        agent_publish(w, s, strlen(s));
    } else {
        agent_publish(w, "--- end history ---\n", strlen("--- end history ---\n"));
    }
}

/* Render recent saved transcript text without mutating the live session. */
bool agent_worker_show_history(agent_worker *w, int user_turns,
                                      char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    size_t text_len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, &w->transcript,
                                                &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render session text");
        return false;
    }
    agent_history_render_text(w, text, text_len, user_turns);
    free(text);
    return true;
}

typedef struct {
    ds4_kvstore_entry entry;
    char *title;
} agent_session_list_item;

static int agent_session_list_cmp_recent(const void *a, const void *b) {
    const agent_session_list_item *sa = a, *sb = b;
    uint64_t ta = sa->entry.last_used ? sa->entry.last_used : sa->entry.created_at;
    uint64_t tb = sb->entry.last_used ? sb->entry.last_used : sb->entry.created_at;
    if (ta < tb) return 1;
    if (ta > tb) return -1;
    return strcmp(sa->entry.sha, sb->entry.sha);
}

static void agent_session_list_free(agent_session_list_item *v, int n) {
    for (int i = 0; i < n; i++) {
        ds4_kvstore_entry_free(&v[i].entry);
        free(v[i].title);
    }
    free(v);
}

static void agent_session_list_push(agent_session_list_item **v, int *len,
                                    int *cap, ds4_kvstore_entry entry,
                                    char *title) {
    if (*len == *cap) {
        *cap = *cap ? *cap * 2 : 16;
        *v = xrealloc(*v, (size_t)*cap * sizeof((*v)[0]));
    }
    (*v)[(*len)++] = (agent_session_list_item){
        .entry = entry,
        .title = title,
    };
}

/* Print resumable sessions from ~/.ds4/kvcache.  sysprompt.kv is intentionally
 * ignored because it is an implementation cache, not a user session. */
void agent_worker_list_sessions(agent_worker *w) {
    DIR *d = opendir(w->cache_dir);
    if (!d) {
        printf("no sessions: %s\n", strerror(errno));
        return;
    }

    int cols = renderer_terminal_cols();
    size_t title_budget = cols > 16 ? (size_t)(cols - 12) : 20;
    if (title_budget > 160) title_budget = 160;

    agent_session_list_item *sessions = NULL;
    int sessions_len = 0, sessions_cap = 0;
    const uint8_t model_id = (uint8_t)ds4_engine_model_kv_id(w->engine);
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        char *path = ds4_kvstore_path_join(w->cache_dir, de->d_name);
        ds4_kvstore_entry e = {0};
        if (ds4_kvstore_read_entry_file(path, sha, &e)) {
            if (e.model_id == model_id) {
                char *title = agent_session_title_from_file(path, title_budget);
                agent_session_list_push(&sessions, &sessions_len, &sessions_cap,
                                        e, title);
            } else {
                ds4_kvstore_entry_free(&e);
            }
        }
        free(path);
    }
    closedir(d);
    if (!sessions_len) {
        printf("no saved sessions\n");
        return;
    }

    qsort(sessions, (size_t)sessions_len, sizeof(sessions[0]),
          agent_session_list_cmp_recent);

    bool color = isatty(STDOUT_FILENO) != 0;
    const char *sha_on = color ? "\x1b[1;96m" : "";
    const char *title_on = color ? "\x1b[1;97m" : "";
    const char *help_on = color ? "\x1b[97m" : "";
    const char *dim = color ? "\x1b[90m" : "";
    const char *reset = color ? "\x1b[0m" : "";

    for (int i = 0; i < sessions_len; i++) {
        ds4_kvstore_entry *e = &sessions[i].entry;
        char age[32];
        agent_format_age(e->last_used ? e->last_used : e->created_at,
                         age, sizeof(age));
        printf("%s%.8s%s %s>%s %s%s%s\n",
               sha_on, e->sha, reset, dim, reset,
               title_on, sessions[i].title, reset);
        printf("         %s> %s, %u tokens, %.2f MB%s%s\n\n",
               dim, age, e->tokens,
               (double)e->file_size / (1024.0 * 1024.0),
               e->payload_bytes == 0 ? ", stripped" : "",
               reset);
    }
    printf("%sUse /switch <id> to select a session, /del <id> to remove, "
           "/strip <id> to strip KV cache.%s\n",
           help_on, reset);
    agent_session_list_free(sessions, sessions_len);
}

typedef struct {
    char sha[41];
    uint64_t last_used;
} agent_completion_session;

typedef struct {
    agent_completion_session *v;
    int len;
    int cap;
} agent_completion_sessions;

static void agent_completion_sessions_push(agent_completion_sessions *s,
                                           const char sha[41],
                                           uint64_t last_used) {
    if (s->len == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 16;
        s->v = xrealloc(s->v, (size_t)s->cap * sizeof(s->v[0]));
    }
    memcpy(s->v[s->len].sha, sha, 41);
    s->v[s->len].last_used = last_used;
    s->len++;
}

static int agent_completion_session_cmp(const void *a, const void *b) {
    const agent_completion_session *sa = a, *sb = b;
    if (sa->last_used < sb->last_used) return 1;
    if (sa->last_used > sb->last_used) return -1;
    return strcmp(sa->sha, sb->sha);
}

/* Tab completion for /switch.  Suggestions are sorted by recent use and accept
 * either an empty prefix or any unambiguous hex prefix. */
void agent_switch_completion_callback(const char *buf,
                                             linenoiseCompletions *lc) {
    agent_worker *w = agent_completion_worker;
    static const char cmd[] = "/switch";
    const size_t cmd_len = sizeof(cmd) - 1;
    if (!w || !buf || strncmp(buf, cmd, cmd_len) != 0) return;

    const char *p = buf + cmd_len;
    if (*p && *p != ' ' && *p != '\t') return;
    while (*p == ' ' || *p == '\t') p++;

    const char *prefix = p;
    size_t prefix_len = strlen(prefix);
    for (size_t i = 0; i < prefix_len; i++) {
        if (!isxdigit((unsigned char)prefix[i])) return;
    }
    if (prefix_len > 40) return;

    DIR *d = opendir(w->cache_dir);
    if (!d) return;

    agent_completion_sessions sessions = {0};
    const uint8_t model_id = (uint8_t)ds4_engine_model_kv_id(w->engine);
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        if (prefix_len && strncasecmp(sha, prefix, prefix_len) != 0) continue;

        uint64_t last_used = 0;
        char *path = ds4_kvstore_path_join(w->cache_dir, de->d_name);
        ds4_kvstore_entry e = {0};
        if (ds4_kvstore_read_entry_file(path, sha, &e)) {
            if (e.model_id == model_id) last_used = e.last_used;
            else last_used = UINT64_MAX;
            ds4_kvstore_entry_free(&e);
        } else {
            last_used = UINT64_MAX;
        }
        free(path);
        if (last_used == UINT64_MAX) continue;
        agent_completion_sessions_push(&sessions, sha, last_used);
    }
    closedir(d);

    qsort(sessions.v, (size_t)sessions.len, sizeof(sessions.v[0]),
          agent_completion_session_cmp);
    for (int i = 0; i < sessions.len; i++) {
        char line[64];
        int sha_chars = prefix_len > 8 ? 40 : 8;
        snprintf(line, sizeof(line), "/switch %.*s",
                 sha_chars, sessions.v[i].sha);
        linenoiseAddCompletion(lc, line);
    }
    free(sessions.v);
}

/* Resolve a user-provided SHA prefix to exactly one saved session file. */
bool agent_worker_find_session(agent_worker *w, const char *prefix,
                                      char sha_out[41], char **path_out,
                                      char *err, size_t err_len) {
    size_t plen = strlen(prefix);
    if (plen == 0 || plen > 40) {
        snprintf(err, err_len, "invalid session SHA prefix");
        return false;
    }
    for (size_t i = 0; i < plen; i++) {
        if (!isxdigit((unsigned char)prefix[i])) {
            snprintf(err, err_len, "invalid session SHA prefix");
            return false;
        }
    }

    DIR *d = opendir(w->cache_dir);
    if (!d) {
        snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }
    int matches = 0;
    char match_sha[41] = {0};
    char *match_path = NULL;
    const uint8_t model_id = (uint8_t)ds4_engine_model_kv_id(w->engine);
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        if (strncasecmp(sha, prefix, plen) != 0) continue;
        char *path = ds4_kvstore_path_join(w->cache_dir, de->d_name);
        ds4_kvstore_entry e = {0};
        bool same_model = ds4_kvstore_read_entry_file(path, sha, &e) &&
                          e.model_id == model_id;
        ds4_kvstore_entry_free(&e);
        if (!same_model) {
            free(path);
            continue;
        }
        matches++;
        if (matches == 1) {
            memcpy(match_sha, sha, sizeof(match_sha));
            match_path = path;
        } else {
            free(path);
        }
    }
    closedir(d);
    if (matches == 0) {
        snprintf(err, err_len, "no saved session matches %.40s", prefix);
        return false;
    }
    if (matches > 1) {
        snprintf(err, err_len, "session prefix %.40s is ambiguous", prefix);
        free(match_path);
        return false;
    }
    memcpy(sha_out, match_sha, 41);
    *path_out = match_path;
    return true;
}
