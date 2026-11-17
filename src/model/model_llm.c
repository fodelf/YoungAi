/* model_llm.c — 助理调模型: 127.0.0.1:<模型端口> POST /v1/chat/completions(stream), 解析 SSE 增量(正文/思考/tool_calls/用量)。
 * 走 HTTP 而不是进程内调引擎: 模型在 ds4-server 子进程里(主进程永不退, 模型会被训练停掉、换模型重起), 这正是两者分开的理由。
 * 不带 temperature(10-10 用户: "不要温度"): 服务端按模型卡配方采样; 工具语法位服务端自己强制贪心(10-11 修的 v41_syntax_greedy)。
 * 带 max_tokens = 官方思考模式的默认上限(10-18 用户"先看 max-token 试一试", 收回 10-10 的"不要长度限制"): 10-18 鹈鹕一条思考 43k 不收口
 * 到用户按停, 官方线上同样的请求到 64K 就 finish=length 断掉。顶到上限时 content 为空, asst_run.c 把它当"没产出"报给用户。 */
enum { ASST_MAX_TOKENS = 65536 };   /* = DeepSeek API 文档思考模式 max_tokens 默认(非思考 8K, effort=max 128K); 改它 = 偏离官方口径 */
/*
 * ds4-server 的流式响应是 Connection: close 分隔(不是 chunked), 读到对端关就是完了。
 * 出错会怎样: 连不上 → err="model is not running"(训练中/没装模型), 页面照实显示; 中途 stop → 关 socket, ds4-server 探到断开自己停生成。 */
#include "../assistant/asst_internal.h"
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

void asst_llm_out_free(asst_llm_out *o) {
    ds4_buf_free(&o->content);
    ds4_buf_free(&o->reasoning);
    for (int i = 0; i < ASST_MAX_CALLS; i++) ds4_buf_free(&o->call_args[i]);
}

static void take_call(const char *tc, asst_llm_out *o) {
    long idx = 0; asst_jint(tc, "index", &idx);
    if (idx < 0 || idx >= ASST_MAX_CALLS) return;
    if (idx + 1 > o->ncalls) o->ncalls = (int)idx + 1;
    char *v = NULL, *fn = NULL;
    if (asst_jstr(tc, "id", &v) && v[0]) snprintf(o->call_id[idx], sizeof o->call_id[idx], "%s", v);
    free(v); v = NULL;
    if (asst_jraw(tc, "function", &fn)) {
        if (asst_jstr(fn, "name", &v) && v[0]) snprintf(o->call_name[idx], sizeof o->call_name[idx], "%s", v);
        free(v); v = NULL;
        if (asst_jstr(fn, "arguments", &v)) ds4_buf_puts(&o->call_args[idx], v);
        free(v);
    }
    free(fn);
}

static void take_chunk(const char *js, const asst_llm_cb *cb, asst_llm_out *o) {
    char *e = NULL, *choices = NULL, *c0 = NULL, *delta = NULL, *v = NULL, *usage = NULL;
    if (asst_jraw(js, "error", &e)) { char *m = NULL; if (!asst_jstr(e, "message", &m)) m = strdup(e); snprintf(o->err, sizeof o->err, "%s", m); free(m); free(e); return; }
    if (asst_jraw(js, "usage", &usage) && usage[0] == '{') { asst_jint(usage, "prompt_tokens", &o->prompt_tokens); asst_jint(usage, "completion_tokens", &o->completion_tokens); }
    free(usage);
    if (!asst_jraw(js, "choices", &choices) || !asst_jarr_at(choices, 0, &c0)) { free(choices); return; }
    if (asst_jstr(c0, "finish_reason", &v)) snprintf(o->finish, sizeof o->finish, "%s", v);
    free(v); v = NULL;
    if (asst_jraw(c0, "delta", &delta)) {
        if (asst_jstr(delta, "reasoning_content", &v) && v[0]) { ds4_buf_puts(&o->reasoning, v); if (cb && cb->on_think) cb->on_think(cb->ud, v, strlen(v)); }
        free(v); v = NULL;
        if (asst_jstr(delta, "content", &v) && v[0]) { ds4_buf_puts(&o->content, v); if (cb && cb->on_text) cb->on_text(cb->ud, v, strlen(v)); }
        free(v); v = NULL;
        char *tcs = NULL, *tc = NULL;
        if (asst_jraw(delta, "tool_calls", &tcs)) for (int i = 0; asst_jarr_at(tcs, i, &tc); i++, free(tc), tc = NULL) take_call(tc, o);
        free(tcs);
    }
    free(delta); free(c0); free(choices);
}

bool asst_sse_feed(ds4_buf *pending, const char *data, size_t n, const asst_llm_cb *cb, asst_llm_out *o) {
    ds4_buf_append(pending, data, n);
    ds4_buf_putc(pending, 0); pending->len--;
    size_t consumed = 0;
    for (;;) {
        char *blk = pending->ptr + consumed;
        char *end = strstr(blk, "\n\n");
        if (!end) break;
        *end = 0;
        ds4_buf js = {0};
        for (char *line = blk, *nl; line && *line; line = nl ? nl + 1 : NULL) {
            nl = strchr(line, '\n'); if (nl) *nl = 0;
            if (line[strlen(line) - 1] == '\r') line[strlen(line) - 1] = 0;
            if (!strncmp(line, "data:", 5)) ds4_buf_puts(&js, line[5] == ' ' ? line + 6 : line + 5);
        }
        if (js.len && strcmp(js.ptr, "[DONE]")) take_chunk(js.ptr, cb, o);
        ds4_buf_free(&js);
        consumed = (size_t)(end - pending->ptr) + 2;
    }
    if (consumed) { memmove(pending->ptr, pending->ptr + consumed, pending->len - consumed); pending->len -= consumed; }
    return !o->err[0];
}

static bool send_all(int fd, const char *p, size_t n) {
    while (n) { ssize_t w = send(fd, p, n, 0); if (w < 0) { if (errno == EINTR) continue; return false; } p += w; n -= (size_t)w; }
    return true;
}

/* 请求体里的图片块(asst_image.c 拼的 image_url data URL, base64 里没有引号, 所以 "}} 就是块尾)→ 一句说明文字。
 * 为什么: 模型子进程不带视觉塔时服务端对 image_url 回 400 "invalid JSON request", 以前整轮就断了(10-18 spark 实撞: render_image 画完图回传就死);
 * 图片块还存在会话历史里, 之后每一轮都带着, 一个会话永远 400。去图后模型至少拿到路径和文字, 轮次继续。 */
bool asst_llm_strip_images(ds4_buf *body) {
    static const char HEAD[] = ",{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:", NOTE[] = ",{\"type\":\"text\",\"text\":\"[image omitted: this model has no vision tower and cannot see pictures; the file path is in the text]\"}";
    bool any = false; ds4_buf out = {0}; const char *p = body->ptr, *end = body->ptr + body->len;
    ds4_buf_putc(body, 0); body->len--;
    for (;;) {
        const char *h = strstr(p, HEAD); if (!h) break;
        const char *t = strstr(h + sizeof HEAD - 1, "\"}}"); if (!t) break;
        ds4_buf_append(&out, p, (size_t)(h - p)); ds4_buf_puts(&out, NOTE); p = t + 3; any = true;
    }
    if (any) { ds4_buf_append(&out, p, (size_t)(end - p)); ds4_buf_free(body); *body = out; } else ds4_buf_free(&out);
    return any;
}

/* 一次 POST: 返回 HTTP 状态(传输层失败 0, o->err 已填); 200 的 SSE 边收边喂回调 */
static int llm_once(int port, const char *conv, const ds4_buf *body, const asst_llm_cb *cb, asst_llm_out *o) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { snprintf(o->err, sizeof o->err, "socket: %s", strerror(errno)); return 0; }
    struct sockaddr_in a; memset(&a, 0, sizeof a); a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&a, sizeof a)) { close(fd); snprintf(o->err, sizeof o->err, "model is not running (nothing on 127.0.0.1:%d; it may be training or not loaded)", port); return 0; }
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    ds4_buf req = {0};
    ds4_buf_printf(&req, "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n", port);
    if (conv && conv[0]) ds4_buf_printf(&req, "X-DS4-Conversation: %.63s\r\n", conv);   /* 会话归属: 服务端据此只删这个助理的 KB, 别的助理留着 */
    ds4_buf_printf(&req, "Content-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", body->len);
    ds4_buf_append(&req, body->ptr, body->len);
    const bool sent = send_all(fd, req.ptr, req.len);
    ds4_buf_free(&req);
    if (!sent) { close(fd); snprintf(o->err, sizeof o->err, "send to model failed"); return 0; }
    ds4_buf head = {0}, pending = {0};
    bool in_body = false; int status = 0;
    char buf[16384];
    for (;;) {
        if (cb && cb->stop && cb->stop(cb->ud)) { o->stopped = true; break; }
        struct pollfd p = { fd, POLLIN, 0 };
        const int pr = poll(&p, 1, 300);   /* 300 ms 一醒: 看停止键; 生成本身不受影响 */
        if (pr < 0) { if (errno == EINTR) continue; break; }
        if (pr == 0) continue;
        const ssize_t n = recv(fd, buf, sizeof buf, 0);
        if (n < 0) { if (errno == EINTR) continue; snprintf(o->err, sizeof o->err, "recv: %s", strerror(errno)); break; }
        if (n == 0) break;
        if (in_body) { if (status == 200) asst_sse_feed(&pending, buf, (size_t)n, cb, o); else ds4_buf_append(&pending, buf, (size_t)n); continue; }
        ds4_buf_append(&head, buf, (size_t)n); ds4_buf_putc(&head, 0); head.len--;
        char *e = strstr(head.ptr, "\r\n\r\n");
        if (!e) continue;
        sscanf(head.ptr, "HTTP/%*s %d", &status);
        in_body = true;
        const size_t off = (size_t)(e - head.ptr) + 4;
        if (status == 200) asst_sse_feed(&pending, head.ptr + off, head.len - off, cb, o); else ds4_buf_append(&pending, head.ptr + off, head.len - off);
    }
    close(fd);
    if (status != 200 && !o->stopped && !o->err[0]) {
        ds4_buf_putc(&pending, 0);
        char *m = NULL, *e = NULL;
        if (asst_jraw(pending.ptr, "error", &e) && asst_jstr(e, "message", &m)) snprintf(o->err, sizeof o->err, "model HTTP %d: %s", status, m);
        else snprintf(o->err, sizeof o->err, "model HTTP %d: %.300s", status, pending.ptr ? pending.ptr : "");
        free(m); free(e);
    }
    ds4_buf_free(&head); ds4_buf_free(&pending);
    return status;
}

bool asst_llm_chat(int port, const char *conv, const char *messages, const char *tools, const asst_llm_cb *cb, asst_llm_out *o) {
    memset(o, 0, sizeof *o);
    ds4_buf body = {0};
    /* ★助理默认开思考★(2026-10-12 用户定"Agent 默认开启"): 不思考模式下模型把过渡句当思考段, 收口位 24 次采样 8 次吐游离 </think>、1 次 EOS
     * 直接收工没终稿; 开思考后同任务 12 步 43 次工具调用每步都正常收口并交出带链接的终稿(fable5 10-12)。代价是每步多 1.5k~3.5k 思考 token(约 1 分钟)。
     * 思考段原文随 reasoning_content 回传(asst_run.c), 服务端按官方模板把同一轮里的 <think>…</think> 原样渲染回去, KV 前缀才接得上。 */
    /* 不带 reasoning_effort = 官方默认 high(75)。10-18 试过 low(50): 鹈鹕照样进环, 用户令还原; 服务端的 low=50 档留着(以前 low 被折成 75 是错的)。 */
    ds4_buf_printf(&body, "{\"model\":\"deepseek-v4-flash\",\"stream\":true,\"think\":true,\"max_tokens\":%d,\"stream_options\":{\"include_usage\":true},\"messages\":%s",
                   ASST_MAX_TOKENS, messages);
    if (tools && tools[0]) ds4_buf_printf(&body, ",\"tools\":%s", tools);
    ds4_buf_putc(&body, '}');
    int status = llm_once(port, conv, &body, cb, o);
    if (status == 400 && !o->stopped && asst_llm_strip_images(&body)) {   /* 没视觉塔的模型拒带图请求: 去图重发一次(服务端在解析阶段就拒, 第一发不花推理) */
        asst_llm_out_free(o); memset(o, 0, sizeof *o);
        status = llm_once(port, conv, &body, cb, o);
    }
    ds4_buf_free(&body);
    (void)status;
    return !o->err[0];
}
