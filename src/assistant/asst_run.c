/* asst_run.c — 助理的运行: 一条工人线程按 FIFO 跑"你发的一条消息 → 模型 ⇄ 工具若干轮 → 模型不再调工具"。
 * 为什么一次只跑一个: 机器上只有一份模型; ds4-server 的活 KV 检查点只有一个, 两个会话交替跑 = 来回从盘恢复, 排队反而快。
 * 运行线程不知道有没有人在看: 它只往 events.jsonl 追加, 页面(任何设备、刷新、关了再开)从任意偏移接着读(asst_sse.c)。Muse 的"关了 app 继续跑"靠这个。
 * 不设轮数上限(与"不设上限"口径一致); 停止键随时停。同名同参的工具连调第三次, 结果前加一句提醒 —— 提示, 不拦。
 * 校验相(10-23, asst_verify.c): 这一趟动过手的运行, 模型第一次不调工具(想收口)时不结束, 先把这趟的账(调了什么 / 产出改动了什么 / 哪些没看过 / 点名却不存在的)
 *   以 user 角色摆回去再给它一步, 最多 ASST_VERIFY_ROUNDS 次; 页面上是一条灰色系统气泡(verify_note 事件), 不是用户说的话。做 → 核 → 答是范式, 不分任务类型。
 * 要审批时线程睡在 asst_cv 上等答复(或停止), 会话状态 waiting_approval; 每个 tool_call id 都必须有一条 tool 结果, 不然下一轮历史就不成对, 停止时没跑的也补"cancelled"。 */
#include "asst_verify.h"
#include <time.h>
#include <unistd.h>

typedef struct job { char sid[ASST_ID_LEN + 1]; char *text, *device, *caps, *apps, *files; char client[40]; struct job *next; } job;
static job *q_head, *q_tail;
static char cur[ASST_ID_LEN + 1], cur_client[40], cur_caps[256], cur_apps[256];
static bool cur_stop, appr_wait, appr_once_only;
static char cur_call[64];   /* 正在执行的工具调用 id: 长工具发进度事件时挂到这张卡上 */
static char appr_id[64], appr_dec[16];
static int g_port;

bool asst_run_stopping(void) { return cur_stop; }

const char *asst_run_state(const char *sid) {
    if (cur[0] && !strcmp(cur, sid)) return appr_wait ? "waiting_approval" : asst_device_waiting() ? "waiting_device" : "running";
    for (job *j = q_head; j; j = j->next) if (!strcmp(j->sid, sid)) return "queued";
    return "idle";
}

static void ev_str(const char *sid, const char *ev, const char *key, const char *val) {
    ds4_buf b = {0}; ds4_buf_printf(&b, "{\"ev\":\"%s\",\"%s\":", ev, key); ds4_json_escape(&b, val); ds4_buf_putc(&b, '}');
    asst_ev(sid, b.ptr); ds4_buf_free(&b);
}

static void on_text(void *ud, const char *s, size_t n) { (void)n; ev_str(ud, "text", "d", s); }
static void on_think(void *ud, const char *s, size_t n) { (void)n; ev_str(ud, "think", "d", s); }
static bool want_stop(void *ud) { (void)ud; pthread_mutex_lock(&asst_mu); const bool s = cur_stop; pthread_mutex_unlock(&asst_mu); return s; }

/* 没有"调了 N 次还没答就催收口"的提醒(10-23 删掉原来的第 10 次起每条结果尾巴催一句): 它掐的是正在变好的活 —— 鹈鹕那趟原话
 * "I got a nudge to answer. I should quickly verify the final render once and then answer", 改到一半就交了; DeepSeek Harness 没有这条,
 * 官方原件同一句请求调 41~47 次, 每次都有用。当初加它是为 10-12"附近有什么吃的"30 次猜坐标, 那个病根后来由 locate 工具 + nearby 坐标参数治了。
 * 打转只剩两道机械护栏: 同名同参连调第三次起提醒换路(下面 repeat), 和校验相(asst_verify.c)收口前摆事实。 */

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + t.tv_nsec * 1e-9; }

/* 等你答复: 返回 once/session/always/deny/stop */
static void wait_answer(const char *sid, const char *call_id, const asst_gres *g, const char *tool, const char *args, char *dec, size_t dn) {
    pthread_mutex_lock(&asst_mu);
    appr_wait = true; appr_once_only = g->once_only; appr_dec[0] = 0;
    snprintf(appr_id, sizeof appr_id, "%s", call_id);
    pthread_mutex_unlock(&asst_mu);
    ds4_buf b = {0};
    ds4_buf_puts(&b, "{\"ev\":\"approval\",\"id\":"); ds4_json_escape(&b, call_id);
    ds4_buf_puts(&b, ",\"name\":"); ds4_json_escape(&b, tool);
    ds4_buf_puts(&b, ",\"args\":"); ds4_json_escape(&b, args);
    ds4_buf_puts(&b, ",\"reason\":"); ds4_json_escape(&b, g->reason);
    ds4_buf_puts(&b, ",\"target\":"); ds4_json_escape(&b, g->target);
    ds4_buf_puts(&b, ",\"path\":"); ds4_json_escape(&b, g->path);
    ds4_buf_puts(&b, ",\"cls\":"); ds4_json_escape(&b, g->cls);
    if (g->extra[0]) { ds4_buf_puts(&b, ",\"pay\":"); ds4_buf_puts(&b, g->extra); }   /* 交易卡: 现场截图/落点/金额/命中原因(sbx_pay.c), 页面画成红卡 */
    ds4_buf_printf(&b, ",\"once_only\":%s,\"warn\":%s}", g->once_only ? "true" : "false", g->warn ? "true" : "false");
    asst_ev(sid, b.ptr); ds4_buf_free(&b);
    asst_notify_approval(sid, tool, g);   /* 卡挂着整条队列都停: 人多半不在页面前, 推一条叫回来(在看的那台设备照在场规则压着) */
    pthread_mutex_lock(&asst_mu);
    while (!appr_dec[0] && !cur_stop) pthread_cond_wait(&asst_cv, &asst_mu);
    snprintf(dec, dn, "%s", cur_stop && !appr_dec[0] ? "stop" : appr_dec);
    appr_wait = false;
    pthread_mutex_unlock(&asst_mu);
}

static char *run_call(const char *sid, const asst_tctx *tc, const char *call_id, const char *tool, const char *args, char *decision, size_t dn, int *ok, asst_gres *g) {   /* g = 守门结果, 交回调用方记审计 */
    *ok = 0;
    char *bad = asst_tool_args_error(tc, tool, args);   /* 参数不合 schema 不进守门: 否则缺的参数当空串, 守门按工作区放行、执行读出个"0 行"当成功(asst_tools.c) */
    if (bad) { snprintf(decision, dn, "invalid"); return bad; }
    asst_guard(sid, tool, args, g);
    if (g->dec == ASST_DENY) { snprintf(decision, dn, "blocked"); ds4_buf b = {0}; ds4_buf_printf(&b, "blocked by the assistant's safety rules: %s\n", g->reason); return ds4_buf_take(&b); }
    if (g->dec == ASST_ASK) {
        char dec[16]; wait_answer(sid, call_id, g, tool, args, dec, sizeof dec);
        if (!strcmp(dec, "stop")) { snprintf(decision, dn, "cancelled"); return strdup("cancelled: the user stopped the run\n"); }
        if (!strcmp(dec, "deny")) { snprintf(decision, dn, "denied"); ds4_buf b = {0}; ds4_buf_printf(&b, "the user denied this (%s). Do not retry the same thing; ask the user or find another way.\n", g->reason); return ds4_buf_take(&b); }
        snprintf(decision, dn, "%s", dec);
        if (!strcmp(g->cls, "pay") && !strcmp(dec, "once")) asst_pay_approved(sid, g->extra);   /* 安全窗口批一次管这一页(sbx_pay.c) */
        if (!g->once_only && (!strcmp(dec, "session") || !strcmp(dec, "always"))) asst_guard_grant(sid, g->cls, g->target, dec);
    } else snprintf(decision, dn, "auto");
    if (asst_device_tool(tool)) {   /* 设备工具: 交给发消息的那台设备; 要结果的等它答复(页面关着就等着, 和等审批一样), 交出去的动作立刻回 */
        char *r = NULL; *ok = asst_device_request(sid, cur_client, call_id, tool, args, &r);
        snprintf(decision, dn, "%s", *ok ? "device" : strstr(r, "cancelled") == r ? "cancelled" : "device");
        if (*ok) { char *post = asst_tool_post(sid, tool, r); if (post) { free(r); r = post; } }   /* locate: 反查地址 + 记进会话, 之后的事实行带着它 */
        return r;
    }
    snprintf(cur_call, sizeof cur_call, "%s", call_id);
    char *r = asst_tool_exec(sid, tool, args, g, ok);
    cur_call[0] = 0;
    if (*ok) asst_guard_mark(sid, g->cls, g->path);
    return r;
}

/* 长工具(make_video 逐帧渲染几十秒到几分钟)的进度: 页面工具卡显示这一句, 不让人对着"进行中"干等。只在工人线程里调 */
void asst_run_progress(const char *text) {
    if (!cur[0] || !cur_call[0]) return;
    ds4_buf b = {0};
    ds4_buf_puts(&b, "{\"ev\":\"tool_progress\",\"id\":"); ds4_json_escape(&b, cur_call);
    ds4_buf_puts(&b, ",\"text\":"); ds4_json_escape(&b, text); ds4_buf_putc(&b, '}');
    asst_ev(cur, b.ptr); ds4_buf_free(&b);
}

static void audit_call(const char *sid, const char *tool, const char *args, const char *decision, int ok, double ms, const char *result, const asst_gres *g) {
    ds4_buf b = {0};
    ds4_buf_printf(&b, "{\"session\":\"%s\",\"tool\":", sid); ds4_json_escape(&b, tool);
    ds4_buf_puts(&b, ",\"cls\":"); ds4_json_escape(&b, g->cls);   /* 活动页按类筛: "所有付款审批" = cls pay */
    if (g->extra[0]) { ds4_buf_puts(&b, ",\"pay\":"); ds4_buf_puts(&b, g->extra); }
    char cut[401]; snprintf(cut, sizeof cut, "%s", args); ds4_buf_puts(&b, ",\"args\":"); ds4_json_escape(&b, cut);
    ds4_buf_printf(&b, ",\"decision\":\"%s\",\"ok\":%s,\"ms\":%.0f,\"sandbox\":\"%s\"", decision, ok ? "true" : "false", ms, asst_sbx_kind());
    snprintf(cut, sizeof cut, "%s", result); ds4_buf_puts(&b, ",\"result\":"); ds4_json_escape(&b, cut);
    ds4_buf_putc(&b, '}');
    asst_audit(b.ptr); ds4_buf_free(&b);
}

static void run_one(const char *sid, const char *text, const char *device, const char *files) {
    ds4_buf b = {0}, full = {0};
    const bool zh = asst_lang_is_zh(text);
    char tid[ASST_ID_LEN + 1]; bool task = asst_task_of(sid, tid);   /* 任务会话(asst_task.c): 多一行任务事实、工具表多 task_*、收尾记进展; 普通对话里模型调了 plan_propose 也会变成任务(下面每轮重看) */
    char *ctx = asst_device_line(device, cur_caps, zh, sid), *att = asst_attach_line(files, zh);
    ds4_buf_puts(&full, ctx); ds4_buf_putc(&full, '\n'); free(ctx);   /* 事实行在用户文本前面: 模型看到, 页面不显示(事件里只有 text) */
    if (task) { char *tl = asst_task_fact_line(tid, zh); if (tl) { ds4_buf_puts(&full, tl); ds4_buf_putc(&full, '\n'); free(tl); } }
    if (att) { ds4_buf_puts(&full, att); ds4_buf_putc(&full, '\n'); free(att); }
    ds4_buf_puts(&full, text);
    ds4_buf_puts(&b, "{\"role\":\"user\",\"content\":"); asst_content_with_images(&b, full.ptr, files); ds4_buf_putc(&b, '}');   /* 图片附件随消息进模型(asst_image.c) */
    asst_msg(sid, b.ptr); ds4_buf_free(&b); ds4_buf_free(&full);
    ev_str(sid, "user", "text", text);
    char upd[32]; snprintf(upd, sizeof upd, "%ld", (long)time(NULL)); asst_meta_set(sid, "updated", upd);
    char last_sig[512] = ""; int repeat = 0;
    asst_ledger led; asst_ledger_init(&led);   /* 这一趟的账: 校验相摆事实用 */
    const char *end = "done";
    char *last_text = NULL;   /* 最后一轮的正文: 任务列表里的"上次进展"、通知的正文 */
    const double t_run = now_s();
    for (;;) {
        char *msgs = asst_messages(sid);
        asst_llm_cb cb = { on_text, on_think, want_stop, (void *)sid };
        asst_llm_out o;
        if (!task) task = asst_task_of(sid, tid);   /* 上一轮刚调了 plan_propose: 这一轮起工具表就带 task_* 与 plan_step */
        const asst_tctx tc = { cur_caps, cur_apps, task ? tid : NULL, task && asst_plan_has(tid) };
        char *tools = asst_tools_json(&tc);
        const bool good = asst_llm_chat(g_port, sid, msgs, tools, &cb, &o);
        free(msgs); free(tools);
        ds4_buf m = {0};
        ds4_buf_puts(&m, "{\"role\":\"assistant\",\"content\":"); ds4_json_escape(&m, o.content.ptr ? o.content.ptr : "");
        if (o.reasoning.len) { ds4_buf_puts(&m, ",\"reasoning_content\":"); ds4_json_escape(&m, o.reasoning.ptr); }   /* 原样回传: 下一条提示才是上一条 + 模型吐的字节, KV 前缀才接得上 */
        const int nc = good && !o.stopped ? o.ncalls : 0;
        if (nc) {
            ds4_buf_puts(&m, ",\"tool_calls\":[");
            for (int i = 0; i < nc; i++) {
                if (!o.call_id[i][0]) snprintf(o.call_id[i], sizeof o.call_id[i], "call_%ld_%d", (long)time(NULL), i);
                ds4_buf_puts(&m, i ? ",{\"id\":" : "{\"id\":"); ds4_json_escape(&m, o.call_id[i]);
                ds4_buf_puts(&m, ",\"type\":\"function\",\"function\":{\"name\":"); ds4_json_escape(&m, o.call_name[i]);
                ds4_buf_puts(&m, ",\"arguments\":"); ds4_json_escape(&m, o.call_args[i].ptr ? o.call_args[i].ptr : "{}"); ds4_buf_puts(&m, "}}");
            }
            ds4_buf_putc(&m, ']');
        }
        ds4_buf_putc(&m, '}');
        if (o.content.len || nc) asst_msg(sid, m.ptr);   /* 停在半截的正文也存: 下一轮历史与页面看到的一致 */
        ds4_buf_free(&m);
        if (o.content.len) { free(last_text); last_text = strdup(o.content.ptr); }
        ds4_buf u = {0}; ds4_buf_printf(&u, "{\"ev\":\"usage\",\"prompt\":%ld,\"completion\":%ld}", o.prompt_tokens, o.completion_tokens); asst_ev(sid, u.ptr); ds4_buf_free(&u);
        if (!good) { ev_str(sid, "error", "msg", o.err); asst_llm_out_free(&o); end = "error"; break; }
        if (o.stopped) { asst_llm_out_free(&o); end = "stopped"; break; }
        if (!nc && !o.content.len && !strcmp(o.finish, "length")) {   /* 顶到 max_tokens(model_llm.c)时思考还没收口: 一个字的回答都没有, 照实告诉用户而不是静默结束 */
            char msg[160]; snprintf(msg, sizeof msg, "model hit the output limit after %ld tokens of thinking without producing an answer", o.completion_tokens);
            ev_str(sid, "error", "msg", msg); asst_llm_out_free(&o); end = "error"; break;
        }
        if (!nc) {   /* 想收口: 动过手就先核一遍(事实消息), 没得核 / 轮数到了才真结束 */
            char *note = asst_verify_note(&led, o.content.ptr ? o.content.ptr : "");
            asst_llm_out_free(&o);
            if (!note) break;
            ds4_buf vm = {0}; ds4_buf_puts(&vm, "{\"role\":\"user\",\"content\":"); ds4_json_escape(&vm, note); ds4_buf_putc(&vm, '}'); asst_msg(sid, vm.ptr); ds4_buf_free(&vm);
            ds4_buf_printf(&vm, "{\"ev\":\"verify_note\",\"round\":%d,\"text\":", led.rounds); ds4_json_escape(&vm, note); ds4_buf_putc(&vm, '}'); asst_ev(sid, vm.ptr); ds4_buf_free(&vm);
            ds4_buf_printf(&vm, "{\"session\":\"%s\",\"tool\":\"verify\",\"cls\":\"verify\",\"args\":\"check %d of %d\",\"decision\":\"note\",\"ok\":true,\"ms\":0,\"sandbox\":\"%s\",\"result\":", sid, led.rounds, ASST_VERIFY_ROUNDS, asst_sbx_kind());
            char cut[401]; snprintf(cut, sizeof cut, "%s", note); ds4_json_escape(&vm, cut); ds4_buf_putc(&vm, '}'); asst_audit(vm.ptr); ds4_buf_free(&vm);
            free(note);
            continue;
        }
        bool stopped = false;
        for (int i = 0; i < nc; i++) {
            const char *id = o.call_id[i], *name = o.call_name[i], *args = o.call_args[i].ptr ? o.call_args[i].ptr : "{}";
            ds4_buf e = {0};
            ds4_buf_puts(&e, "{\"ev\":\"tool\",\"id\":"); ds4_json_escape(&e, id); ds4_buf_puts(&e, ",\"name\":"); ds4_json_escape(&e, name);
            ds4_buf_puts(&e, ",\"args\":"); ds4_json_escape(&e, args); ds4_buf_putc(&e, '}');
            asst_ev(sid, e.ptr); ds4_buf_free(&e);
            char decision[16] = "cancelled"; int ok = 0; char *res; asst_gres g; memset(&g, 0, sizeof g);
            const double t0 = now_s();
            void *snap = !strcmp(name, "run") ? asst_ledger_snap() : NULL;   /* run 能写任何工作区文件: 前后扫一遍比出改了什么 */
            if (stopped || want_stop(NULL)) { stopped = true; res = strdup("cancelled: the user stopped the run\n"); }
            else res = run_call(sid, &tc, id, name, args, decision, sizeof decision, &ok, &g);
            if (!strcmp(decision, "cancelled")) stopped = true;
            asst_ledger_note(&led, name, &g, res, ok && !stopped);
            if (snap) asst_ledger_diff(&led, snap, name);
            char sig[512]; snprintf(sig, sizeof sig, "%s|%.400s", name, args);
            repeat = !strcmp(sig, last_sig) ? repeat + 1 : 1; snprintf(last_sig, sizeof last_sig, "%s", sig);
            if (repeat >= 3) { ds4_buf r = {0}; ds4_buf_printf(&r, "[note: this is the %d-th identical %s call in a row; if it is not working, change approach or ask the user]\n", repeat, name); ds4_buf_puts(&r, res); free(res); res = ds4_buf_take(&r); }
            const double ms = (now_s() - t0) * 1000;
            if (strcmp(decision, "cancelled")) audit_call(sid, name, args, decision, ok, ms, res, &g);
            ds4_buf t = {0};
            ds4_buf_puts(&t, "{\"ev\":\"tool_end\",\"id\":"); ds4_json_escape(&t, id);
            ds4_buf_printf(&t, ",\"ok\":%s,\"decision\":\"%s\",\"ms\":%.0f,\"result\":", ok ? "true" : "false", decision, ms);
            char head[4097]; snprintf(head, sizeof head, "%s", res); ds4_json_escape(&t, head);
            ds4_buf_printf(&t, ",\"bytes\":%zu}", strlen(res));
            asst_ev(sid, t.ptr); ds4_buf_free(&t);
            ds4_buf tm = {0};
            ds4_buf_puts(&tm, "{\"role\":\"tool\",\"tool_call_id\":"); ds4_json_escape(&tm, id);
            /* pick_file 拍/选的是图、phone_* 带回的截图 ⇒ 工具结果里顺手带上 image_url 块, 模型直接看, 不用再去读路径(asst_image.c) */
            char picked[ASST_PATH] = ""; const char *sv = !strcmp(name, "pick_file") ? strstr(res, "saved: ") : NULL;
            if (sv) { const char *sp = strchr(sv + 7, ' '); if (sp && (size_t)(sp - sv - 7) < sizeof picked) snprintf(picked, sizeof picked, "%.*s", (int)(sp - sv - 7), sv + 7); }
            if (asst_phone_tool(name) && !strncmp(res, "screenshot: .phone/", 19)) { const char *nl = strchr(res, '\n'); if (nl && (size_t)(nl - res - 12) < sizeof picked) snprintf(picked, sizeof picked, "%.*s", (int)(nl - res - 12), res + 12); }
            /* edit_image 等产图工具: 结果开头若干行 "image: 路径"(成品、叠红掩膜) 都带上, 换行分隔(asst_image_parts 按行收) */
            if (!picked[0]) for (const char *ln = res; ln && !strncmp(ln, "image: ", 7); ln = strchr(ln, '\n') ? strchr(ln, '\n') + 1 : NULL) {
                const char *nl = strchr(ln, '\n'); const size_t L = nl ? (size_t)(nl - ln - 7) : strlen(ln + 7);
                if (strlen(picked) + L + 2 < sizeof picked) { if (picked[0]) strcat(picked, "\n"); strncat(picked, ln + 7, L); }
            }
            ds4_buf_puts(&tm, ",\"content\":"); asst_content_with_images(&tm, res, picked); ds4_buf_putc(&tm, '}');
            asst_msg(sid, tm.ptr); ds4_buf_free(&tm);
            free(res);
        }
        asst_llm_out_free(&o);
        if (stopped) { end = "stopped"; break; }
    }
    snprintf(upd, sizeof upd, "%ld", (long)time(NULL)); asst_meta_set(sid, "updated", upd);
    /* 终稿里写出的工作区文件 → 交付卡(asst_file.c; 照 ChatGPT 代码解释器的"终稿里写路径就出下载链接"); 在 end 之前发, 页面收到 end 时卡已经在了 */
    char *fl = asst_file_mentioned(last_text);
    if (fl) { ds4_buf f = {0}; ds4_buf_printf(&f, "{\"ev\":\"files\",\"paths\":%s}", fl); asst_ev(sid, f.ptr); ds4_buf_free(&f); free(fl); }
    ev_str(sid, "end", "reason", end);
    if (task) asst_task_run_end(tid, end, last_text);
    asst_notify_run_end(sid, end, last_text, now_s() - t_run);   /* 任务做完 / 跑了很久 → 推一条到用户设备(asst_notify.c) */
    free(last_text);
}

static void *worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&asst_mu);
        while (!q_head) pthread_cond_wait(&asst_cv, &asst_mu);
        job *j = q_head; q_head = j->next; if (!q_head) q_tail = NULL;
        snprintf(cur, sizeof cur, "%s", j->sid); snprintf(cur_client, sizeof cur_client, "%s", j->client);
        snprintf(cur_caps, sizeof cur_caps, "%s", j->caps ? j->caps : ""); snprintf(cur_apps, sizeof cur_apps, "%s", j->apps ? j->apps : ""); cur_stop = false;
        pthread_mutex_unlock(&asst_mu);
        run_one(j->sid, j->text, j->device, j->files);
        pthread_mutex_lock(&asst_mu); cur[0] = 0; pthread_cond_broadcast(&asst_cv); pthread_mutex_unlock(&asst_mu);
        free(j->text); free(j->device); free(j->caps); free(j->apps); free(j->files); free(j);
    }
    return NULL;
}

void asst_run_start(int model_port) {
    g_port = model_port;
    pthread_t th; pthread_create(&th, NULL, worker, NULL); pthread_detach(th);
}

bool asst_run_submit(const char *sid, const char *text, const char *device, const char *client, const char *caps, const char *apps, const char *files, bool *ahead, char *err, size_t errn) {
    *ahead = false;
    if (!asst_id_ok(sid)) { snprintf(err, errn, "bad session id"); return false; }
    if (!text || !text[0]) { snprintf(err, errn, "empty message"); return false; }
    pthread_mutex_lock(&asst_mu);
    const bool busy = strcmp(asst_run_state(sid), "idle") != 0;
    if (!busy) {
        /* 真排队只有一种: 别的会话正占着工人线程(或已在队里)。本会话忙在上面就拒了, 所以这里不会是自己。
         * 页面靠这个决定显示"排队中"还是"思考中" —— 以前页面发完一律写"排队中", 而 user 事件常比 send 的回包先到, 被它盖回去后就一直挂着(10-13 用户实撞)。 */
        *ahead = cur[0] || q_head;
        job *j = calloc(1, sizeof *j); snprintf(j->sid, sizeof j->sid, "%s", sid); j->text = strdup(text); j->device = device ? strdup(device) : NULL; j->caps = caps ? strdup(caps) : NULL;
        j->apps = apps ? strdup(apps) : NULL; j->files = files ? strdup(files) : NULL;
        snprintf(j->client, sizeof j->client, "%s", client ? client : "");
        if (q_tail) q_tail->next = j; else q_head = j;
        q_tail = j;
        pthread_cond_broadcast(&asst_cv);
    }
    pthread_mutex_unlock(&asst_mu);
    if (busy) snprintf(err, errn, "this conversation is still running; stop it or wait");
    return !busy;
}

void asst_run_stop(const char *sid) {
    pthread_mutex_lock(&asst_mu);
    if (cur[0] && !strcmp(cur, sid)) cur_stop = true;
    for (job **pp = &q_head; *pp; ) {
        if (!strcmp((*pp)->sid, sid)) { job *d = *pp; *pp = d->next; free(d->text); free(d->device); free(d->caps); free(d->apps); free(d->files); free(d); } else pp = &(*pp)->next;
    }
    q_tail = NULL; for (job *j = q_head; j; j = j->next) q_tail = j;
    pthread_cond_broadcast(&asst_cv);
    pthread_mutex_unlock(&asst_mu);
}

bool asst_run_answer(const char *sid, const char *call_id, const char *decision, char *err, size_t errn) {
    const bool valid = !strcmp(decision, "once") || !strcmp(decision, "session") || !strcmp(decision, "always") || !strcmp(decision, "deny");
    if (!valid) { snprintf(err, errn, "decision must be once, session, always or deny"); return false; }
    pthread_mutex_lock(&asst_mu);
    bool ok = cur[0] && !strcmp(cur, sid) && appr_wait && !strcmp(appr_id, call_id);
    if (!ok) snprintf(err, errn, "no pending approval with that id");
    else if (appr_once_only && strcmp(decision, "once") && strcmp(decision, "deny")) { ok = false; snprintf(err, errn, "this action can only be approved once"); }
    else { snprintf(appr_dec, sizeof appr_dec, "%s", decision); pthread_cond_broadcast(&asst_cv); }
    pthread_mutex_unlock(&asst_mu);
    return ok;
}
