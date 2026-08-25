/* agent_tools_search.c — 机械拆分自 ds4_agent.c: Edit And Search Tools (2/2: search 工具)。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

typedef struct {
    const char *query;
    const char *glob;
    regex_t regex;
    bool use_regex;
    bool regex_ready;
    bool case_sensitive;
    int context;
    int max_results;
    int results;
    agent_buf out;
} agent_search_ctx;

static bool agent_literal_match(const char *s, size_t n, const char *q,
                                bool case_sensitive) {
    size_t qn = strlen(q);
    if (!qn) return true;
    if (qn > n) return false;
    for (size_t i = 0; i + qn <= n; i++) {
        bool ok = true;
        for (size_t j = 0; j < qn; j++) {
            unsigned char a = (unsigned char)s[i + j];
            unsigned char b = (unsigned char)q[j];
            if (!case_sensitive) {
                a = (unsigned char)tolower(a);
                b = (unsigned char)tolower(b);
            }
            if (a != b) {
                ok = false;
                break;
            }
        }
        if (ok) return true;
    }
    return false;
}

static bool agent_search_line_matches(agent_search_ctx *ctx, const char *s, size_t n) {
    if (ctx->use_regex) {
        char *line = xstrndup(s, n);
        int rc = regexec(&ctx->regex, line, 0, NULL, 0);
        free(line);
        return rc == 0;
    }
    return agent_literal_match(s, n, ctx->query, ctx->case_sensitive);
}

static void agent_search_emit_line(agent_search_ctx *ctx, const char *data,
                                   agent_line_span sp, int line_no) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "  %d ", line_no);
    agent_buf_puts(&ctx->out, prefix);
    agent_buf_append(&ctx->out, data + sp.start, sp.content_end - sp.start);
    agent_buf_puts(&ctx->out, "\n");
}

/* Search one text file and emit matching lines with plain line numbers. */
static void agent_search_file(agent_search_ctx *ctx, const char *path) {
    if (ctx->results >= ctx->max_results) return;
    if (ctx->glob && ctx->glob[0]) {
        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;
        if (fnmatch(ctx->glob, base, 0) != 0 && fnmatch(ctx->glob, path, 0) != 0)
            return;
    }
    char err[256];
    char *data = NULL;
    size_t len = 0;
    if (agent_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0) return;
    if (memchr(data, '\0', len)) {
        free(data);
        return;
    }
    agent_line_spans spans = {0};
    agent_split_lines(data, len, &spans);
    bool printed_file = false;
    int last_context_line = -1;
    for (int i = 0; i < spans.len && ctx->results < ctx->max_results; i++) {
        agent_line_span sp = spans.v[i];
        if (!agent_search_line_matches(ctx, data + sp.start, sp.content_end - sp.start))
            continue;
        if (!printed_file) {
            agent_buf_puts(&ctx->out, path);
            agent_buf_puts(&ctx->out, "\n");
            printed_file = true;
        }
        int from = i - ctx->context;
        int to = i + ctx->context;
        if (from < 0) from = 0;
        if (to >= spans.len) to = spans.len - 1;
        if (from <= last_context_line) from = last_context_line + 1;
        for (int j = from; j <= to; j++) {
            agent_search_emit_line(ctx, data, spans.v[j], j + 1);
            last_context_line = j;
        }
        ctx->results++;
    }
    if (printed_file) agent_buf_puts(&ctx->out, "\n");
    agent_line_spans_free(&spans);
    free(data);
}

/* Recursively search a file or directory, avoiding .git and stopping once the
 * result cap is reached. */
static void agent_search_path(agent_search_ctx *ctx, const char *path, int depth) {
    if (ctx->results >= ctx->max_results || depth > 24) return;
    struct stat st;
    if (lstat(path, &st) != 0) return;
    if (S_ISREG(st.st_mode)) {
        agent_search_file(ctx, path);
        return;
    }
    if (!S_ISDIR(st.st_mode)) return;
    DIR *dir = opendir(path);
    if (!dir) return;
    struct dirent *de;
    while ((de = readdir(dir)) != NULL && ctx->results < ctx->max_results) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        if (!strcmp(de->d_name, ".git")) continue;
        char child[PATH_MAX];
        snprintf(child, sizeof(child), "%s/%s", path, de->d_name);
        agent_search_path(ctx, child, depth + 1);
    }
    closedir(dir);
}

/* Implement the search tool using either literal matching or POSIX regex. */
char *agent_tool_search(agent_worker *w, const agent_tool_call *call) {
    (void)w;
    const char *query = agent_tool_arg_value(call, "query");
    if (!query || !query[0]) return xstrdup("Tool error: search requires query\n");
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) path = ".";
    const char *mode = agent_tool_arg_value(call, "mode");
    agent_search_ctx ctx = {
        .query = query,
        .glob = agent_tool_arg_value(call, "glob"),
        .use_regex = mode && !strcmp(mode, "regex"),
        .case_sensitive = agent_parse_bool_default(agent_tool_arg_value(call, "case_sensitive"), true),
        .context = agent_parse_int_default(agent_tool_arg_value(call, "context"), 0, 0, 5),
        .max_results = agent_parse_int_default(agent_tool_arg_value(call, "max_results"), 50, 1, 500),
    };
    if (ctx.use_regex) {
        int flags = REG_EXTENDED | REG_NOSUB;
        if (!ctx.case_sensitive) flags |= REG_ICASE;
        int rc = regcomp(&ctx.regex, query, flags);
        if (rc != 0) {
            char msg[256];
            regerror(rc, &ctx.regex, msg, sizeof(msg));
            agent_buf b = {0};
            agent_buf_puts(&b, "Tool error: invalid regex: ");
            agent_buf_puts(&b, msg);
            agent_buf_puts(&b, "\n");
            return agent_buf_take(&b);
        }
        ctx.regex_ready = true;
    }
    agent_search_path(&ctx, path, 0);
    if (ctx.regex_ready) regfree(&ctx.regex);
    if (!ctx.out.ptr) agent_buf_puts(&ctx.out, "No matches\n");
    else {
        char hdr[96];
        snprintf(hdr, sizeof(hdr), "%d match%s shown\n\n",
                 ctx.results, ctx.results == 1 ? "" : "es");
        size_t hdr_len = strlen(hdr);
        if (ctx.out.len + hdr_len + 1 > ctx.out.cap) {
            ctx.out.cap = ctx.out.len + hdr_len + 1;
            ctx.out.ptr = xrealloc(ctx.out.ptr, ctx.out.cap);
        }
        memmove(ctx.out.ptr + hdr_len, ctx.out.ptr, ctx.out.len + 1);
        memcpy(ctx.out.ptr, hdr, hdr_len);
        ctx.out.len += hdr_len;
    }
    return agent_buf_take(&ctx.out);
}
