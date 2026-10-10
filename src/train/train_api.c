/* train_api.c — 工作台的路由(主进程 ds4-train 与模型子进程 ds4-server 共用)。总述见 train_internal.h。
 * 状态里的 mode: training(有训练链在跑) / serving(模型子进程装着) / loading(在装) / lobby(都没有)。发车不按 mode 拦: 模型装着时发车 = cycle 脚本先停模型子进程(训完自动回来)。
 * 管子进程的操作(发车/换模型/起服)只在主进程做: serving=1(调用方是模型子进程自己)一律拒, 免得两个进程各管各的。 */
#include "train_internal.h"
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>

bool tr_query_get(const char *query, const char *key, char *val, size_t n) {
    const size_t kl = strlen(key);
    for (const char *p = query; p && *p; ) {
        const char *amp = strchr(p, '&');
        const size_t seg = amp ? (size_t)(amp - p) : strlen(p);
        if (seg > kl && !strncmp(p, key, kl) && p[kl] == '=') {
            size_t o = 0;
            for (size_t i = kl + 1; i < seg && o + 1 < n; i++) {
                if (p[i] == '%' && i + 2 < seg) { unsigned v = 0; if (sscanf(p + i + 1, "%2x", &v) == 1) { val[o++] = (char)v; i += 2; continue; } }
                val[o++] = p[i] == '+' ? ' ' : p[i];
            }
            val[o] = 0;
            return true;
        }
        p = amp ? amp + 1 : NULL;
    }
    val[0] = 0;
    return false;
}

/* POST 体: {"k":"v",...} 扁平对象取字符串/数值(数值照原文给) */
bool tr_body_get(const tr_req *rq, const char *key, char *out, size_t n) {
    out[0] = 0;
    if (!rq->body) return false;
    const char *p = rq->body;
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    for (;;) {
        json_ws(&p);
        if (*p == '}' || !*p) return false;
        char *k = NULL;
        if (!json_string(&p, &k)) return false;
        json_ws(&p);
        if (*p != ':') { free(k); return false; }
        p++;
        if (!strcmp(k, key)) {
            free(k);
            json_ws(&p);
            if (*p == '"') { char *v = NULL; if (!json_string(&p, &v)) return false; snprintf(out, n, "%s", v); free(v); return true; }
            char *raw = NULL; if (!json_raw_value(&p, &raw)) return false; snprintf(out, n, "%s", raw); free(raw); return true;
        }
        free(k);
        if (!json_skip_value(&p)) return false;
        json_ws(&p);
        if (*p == ',') p++;
    }
}
/* 参数白名单: 只许字母数字 + extra 里的字符。它们会原样进 execvp 的 argv(不经 shell), 但脚本里再拼命令就说不准, 所以空格/分号/引号一律拒 */
bool tr_clean(const char *s, const char *extra) {
    for (; *s; s++) if (!((*s >= 'a' && *s <= 'z') || (*s >= 'A' && *s <= 'Z') || (*s >= '0' && *s <= '9') || strchr(extra, *s))) return false;
    return true;
}
static int reply(ds4_buf *out, bool ok, const char *err, int code_fail) {
    ds4_buf_printf(out, "{\"ok\":%s,\"error\":", ok ? "true" : "false"); ds4_json_escape(out, err ? err : ""); ds4_buf_putc(out, '}');
    return ok ? 200 : code_fail;
}

/* 工作台页 web/studio.html(聊天 / 训练 / 语料 / 记录 四个 tab, 左右布局; 10-10 用户 "和主流工具一样左右布局 … 再多一个聊天 tab") */
static int serve_page(ds4_buf *out, const char **ctype) {
    char p[TR_PATH + 32]; snprintf(p, sizeof p, "%.900s/web/studio.html", tr_root);
    size_t n = 0; char *t = tr_slurp(p, &n);
    if (!t) { ds4_buf_puts(out, "{\"error\":\"没有 web/studio.html(要在仓库根起)\"}"); return 404; }
    ds4_buf_append(out, t, n); free(t);
    *ctype = "text/html; charset=utf-8";
    return 200;
}

/* mode: training(作业线程在: 停模型/训/门/出题/装回来) / loading(模型子进程在装或在停) / serving(模型子进程装着、冒烟过) / lobby(都没有)。
 * proc 里的 script/judge/cycle 沿用老名字(页面按它们显示 "训练中 · wt2 门" / "交接中"): script = 训练或出题在跑, judge = 门在跑, cycle = 作业线程在, ds4 = ./ds4 子进程在 */
static int api_status(ds4_buf *out, int serving, int port) {
    tr_procs p; tr_proc_scan(&p);
    tr_job_info j; tr_job_status(&j);
    const tr_model_phase mp = tr_model_state();
    const bool busy = j.active;
    const char *mode = busy ? "training" : (serving || tr_model_live()) ? "serving" : (mp == TR_MODEL_STARTING || mp == TR_MODEL_STOPPING) ? "loading" : "lobby";
    static const char *const phase_name[] = { "", "stop_model", "train", "gate", "gen", "restart" };
    ds4_buf_printf(out, "{\"time\":%ld,\"mem_avail_mb\":%ld,\"port\":%d,\"mode\":\"%s\",\"busy\":%s,\"downloading\":%s,\"phase\":\"%s\",\"run\":",
                   (long)time(NULL), tr_mem_avail_mb(), port, mode, busy ? "true" : "false", p.dl ? "true" : "false", phase_name[j.phase]);
    ds4_json_escape(out, j.run);
    ds4_buf_printf(out, ",\"proc\":{\"scanned\":%s,\"script\":%s,\"ds4\":%s,\"judge\":%s,\"cycle\":%s,\"server\":%s},\"root\":",
                   p.scanned ? "true" : "false", (j.phase == TR_JOB_TRAIN || j.phase == TR_JOB_GEN) ? "true" : "false", (p.ds4 || j.child) ? "true" : "false",
                   j.phase == TR_JOB_GATE ? "true" : "false", busy ? "true" : "false", (p.server || serving || mp == TR_MODEL_UP) ? "true" : "false");
    ds4_json_escape(out, tr_root); ds4_buf_putc(out, '}');
    return 200;
}

/* 发车(train / gen): 参数白名单 → 作业线程(train_job.c: 先停模型子进程, 训完再装回来) */
static int api_launch(const tr_req *rq, ds4_buf *out, int serving, int port, bool gen) {
    char err[256] = "", a1[TR_PATH], a2[64], a3[64], a4[64], a5[512];
    struct stat st; char abs[TR_PATH + 64];
    bool ok = false;
    if (serving) return reply(out, false, "这是模型子进程; 发车要从主进程 ds4-train 的页面操作", 409);
    if (!gen) {
        tr_body_get(rq, "data", a1, sizeof a1); tr_body_get(rq, "epochs", a2, sizeof a2); tr_body_get(rq, "layers", a3, sizeof a3);
        tr_body_get(rq, "lr", a4, sizeof a4); tr_body_get(rq, "extra", a5, sizeof a5);
        snprintf(abs, sizeof abs, "%.900s/%.100s", tr_root, a1);
        if (!a1[0] || !tr_clean(a1, "._/-") || strstr(a1, "..") || stat(abs, &st)) snprintf(err, sizeof err, "料路径不合法或不存在: %s", a1);
        else if (!tr_clean(a2, "") || !tr_clean(a3, "-") || !tr_clean(a4, ".-") || !tr_clean(a5, "=,._-/")) snprintf(err, sizeof err, "参数只许字母数字与 . - / = ,");
        else ok = tr_job_train(a1, a2[0] ? a2 : "3", a3[0] ? a3 : "0-39", a4[0] ? a4 : "2e-4", a5, TR_MODEL_PORT(port), err, sizeof err);
    } else {
        tr_body_get(rq, "dir", a1, sizeof a1); tr_body_get(rq, "rounds", a2, sizeof a2);
        snprintf(abs, sizeof abs, "%.900s/%.100s/chunks", tr_root, a1);
        if (!a1[0] || !tr_clean(a1, "._/-") || strstr(a1, "..") || stat(abs, &st)) snprintf(err, sizeof err, "料目录不合法或没有 chunks/: %s", a1);
        else if (!tr_clean(a2, "")) snprintf(err, sizeof err, "轮数只许数字");
        else ok = tr_job_gen(a1, a2[0] ? a2 : "1", TR_MODEL_PORT(port), err, sizeof err);
    }
    ds4_buf_printf(out, "{\"ok\":%s,\"error\":", ok ? "true" : "false"); ds4_json_escape(out, err); ds4_buf_putc(out, '}');
    return ok ? 200 : strstr(err, "在跑") || strstr(err, "正在") ? 409 : 400;
}

/* 文档目录 → 原文料 <dir>/<dir>.text.jsonl: chunks/ 下每个 .txt 一行 {"text": 全文}(unsloth 的 continued-pretraining 口径, 训练器按 token 自己切段)。
 * 每次上传文档后整份重写, 文件就是目录的镜像; 想出成问答再训的, 另点"出题"(gen) —— 两条路都通, 不强迫先出题。 */
static unsigned rebuild_text_jsonl(const char *dir) {
    char ch[TR_PATH + 320], outp[TR_PATH + 600];
    snprintf(ch, sizeof ch, "%s/%s/chunks", tr_datad, dir);
    snprintf(outp, sizeof outp, "%s/%s/%s.text.jsonl", tr_datad, dir, dir);
    DIR *d = opendir(ch); if (!d) return 0;
    ds4_buf b = {0}; unsigned n = 0; struct dirent *de;
    while ((de = readdir(d))) {
        const size_t l = strlen(de->d_name);
        if (l <= 4 || strcmp(de->d_name + l - 4, ".txt")) continue;
        char p[TR_PATH + 600]; snprintf(p, sizeof p, "%s/%s", ch, de->d_name);
        size_t len = 0; char *t = tr_slurp(p, &len);
        if (!t) continue;
        size_t k = len; while (k && (t[k - 1] == '\n' || t[k - 1] == '\r' || t[k - 1] == ' ')) t[--k] = 0;
        if (k) { ds4_buf_puts(&b, "{\"text\":"); ds4_json_escape(&b, t); ds4_buf_puts(&b, "}\n"); n++; }
        free(t);
    }
    closedir(d);
    FILE *f = fopen(outp, "wb");
    if (f) { if (b.len) fwrite(b.ptr, 1, b.len, f); fclose(f); }
    ds4_buf_free(&b);
    return n;
}

/* 上传(原始体): kind=jsonl → $datad/<name>(逐行校验是带 messages 或 text 的 JSON 对象); kind=doc → $datad/<dir>/chunks/<name>.txt(文档: 同时重写原文料 <dir>.text.jsonl, 也可再 gen 出题) */
static int api_upload(const tr_req *rq, ds4_buf *out) {
    char kind[16], name[256], dir[256], err[300] = "";
    tr_query_get(rq->query, "kind", kind, sizeof kind); tr_query_get(rq->query, "name", name, sizeof name); tr_query_get(rq->query, "dir", dir, sizeof dir);
    if (!rq->body || !rq->body_len) return reply(out, false, "空文件", 400);
    if (rq->body_len > TR_UPLOAD_MAX) return reply(out, false, "超过 64 MB", 400);
    if (!name[0] || !tr_clean(name, "._-") || strstr(name, "..")) return reply(out, false, "文件名只许字母数字与 . _ -", 400);
    char path[TR_PATH + 600];
    if (!strcmp(kind, "jsonl")) {
        const size_t L = strlen(name);
        if (L < 7 || strcmp(name + L - 6, ".jsonl")) return reply(out, false, "料文件要以 .jsonl 结尾", 400);
        unsigned rows = 0, lineno = 0;   /* 校验: 每个非空行是一个 JSON 对象且带 messages 或 text; 坏一行整份拒(训练器同一条规矩) */
        for (const char *p = rq->body, *end = rq->body + rq->body_len; p < end; ) {
            const char *nl = memchr(p, '\n', (size_t)(end - p)); const size_t len = nl ? (size_t)(nl - p) : (size_t)(end - p);
            lineno++;
            size_t k = 0; while (k < len && (p[k] == ' ' || p[k] == '\t' || p[k] == '\r')) k++;
            if (k < len) {
                char *line = malloc(len + 1); memcpy(line, p, len); line[len] = 0;
                const char *q = line; const bool okj = json_skip_value(&q);
                const bool obj = line[k] == '{' && (strstr(line, "\"messages\"") || strstr(line, "\"text\""));
                free(line);
                if (!okj || !obj) { snprintf(err, sizeof err, "第 %u 行不是带 messages 或 text 的 JSON 对象", lineno); return reply(out, false, err, 400); }
                rows++;
            }
            p = nl ? nl + 1 : end;
        }
        snprintf(path, sizeof path, "%s/%s", tr_datad, name);
        FILE *f = fopen(path, "wb");
        if (!f || fwrite(rq->body, 1, rq->body_len, f) != rq->body_len) { if (f) fclose(f); return reply(out, false, "写不进料目录", 500); }
        fclose(f);
        ds4_buf_printf(out, "{\"ok\":true,\"path\":\"gguf-tools/data/posttrain/%s\",\"rows\":%u}", name, rows);
        return 200;
    }
    if (!strcmp(kind, "doc")) {
        if (!dir[0] || !tr_clean(dir, "._-")) return reply(out, false, "料目录名只许字母数字与 . _ -", 400);
        snprintf(path, sizeof path, "%s/%s", tr_datad, dir); mkdir(path, 0755);
        snprintf(path, sizeof path, "%s/%s/chunks", tr_datad, dir); mkdir(path, 0755);
        char base[256]; snprintf(base, sizeof base, "%s", name);
        char *dot = strrchr(base, '.'); if (dot && dot != base) *dot = 0;   /* 块名 = 去扩展名 + .txt(gen 只认 chunks/ 下的 .txt) */
        snprintf(path, sizeof path, "%s/%s/chunks/%s.txt", tr_datad, dir, base);
        FILE *f = fopen(path, "wb");
        if (!f || fwrite(rq->body, 1, rq->body_len, f) != rq->body_len) { if (f) fclose(f); return reply(out, false, "写不进料目录", 500); }
        fclose(f);
        const unsigned rows = rebuild_text_jsonl(dir);
        ds4_buf_printf(out, "{\"ok\":true,\"path\":\"gguf-tools/data/posttrain/%s/chunks/%s.txt\",\"bytes\":%zu,\"text_jsonl\":\"gguf-tools/data/posttrain/%s/%s.text.jsonl\",\"rows\":%u}",
                       dir, base, rq->body_len, dir, dir, rows);
        return 200;
    }
    return reply(out, false, "kind 只认 jsonl / doc", 400);
}

/* 项目 logo(web/logo.jpg = site/img/youngai-avatar.jpg 缩到 192 px): 侧栏、聊天头像、favicon 共用 */
static int serve_logo(ds4_buf *out, const char **ctype) {
    char p[TR_PATH + 32]; snprintf(p, sizeof p, "%.900s/web/logo.jpg", tr_root);
    size_t n = 0; char *t = tr_slurp(p, &n);
    if (!t) { ds4_buf_puts(out, "{\"error\":\"没有 web/logo.jpg\"}"); return 404; }
    ds4_buf_append(out, t, n); free(t);
    *ctype = "image/jpeg";
    return 200;
}

int tr_api(const tr_req *rq, ds4_buf *out, const char **ctype, int serving, int port) {
    const bool post = !strcmp(rq->method, "POST");
    *ctype = "application/json; charset=utf-8";
    if (!strcmp(rq->path, "/") || !strcmp(rq->path, "/index.html") || !strcmp(rq->path, "/studio") || !strcmp(rq->path, "/train")) return serve_page(out, ctype);
    if (!strcmp(rq->path, "/logo.jpg") || !strcmp(rq->path, "/favicon.ico")) return serve_logo(out, ctype);
    if (!strcmp(rq->path, "/api/train/status")) return api_status(out, serving, port);
    { const int mc = tr_models_api(rq, out, serving, port); if (mc >= 0) return mc; }
    if (!strcmp(rq->path, "/api/train/runs")) { tr_json_runs(out); return 200; }
    if (!strcmp(rq->path, "/api/train/run")) {
        char name[256]; tr_query_get(rq->query, "name", name, sizeof name);
        if (!tr_json_run(out, name)) { ds4_buf_puts(out, "{\"error\":\"没有这趟\"}"); return 404; }
        return 200;
    }
    if (!strcmp(rq->path, "/api/train/data")) { tr_json_data(out); return 200; }
    if (!strcmp(rq->path, "/api/train/preview")) {   /* 语料 tab 的预览: jsonl 前 n 行原样(每行截 4 KB), 页面自己解析着显示 */
        char rel[512], ns[16]; tr_query_get(rq->query, "path", rel, sizeof rel); tr_query_get(rq->query, "n", ns, sizeof ns);
        const int want = ns[0] ? atoi(ns) : 5;
        if (!rel[0] || !tr_clean(rel, "._/-") || strstr(rel, "..")) { ds4_buf_puts(out, "{\"error\":\"路径不合法\"}"); return 400; }
        char abs[TR_PATH + 600]; snprintf(abs, sizeof abs, "%.900s/%s", tr_root, rel);
        FILE *f = fopen(abs, "rb");
        if (!f) { ds4_buf_puts(out, "{\"error\":\"没有这个文件\"}"); return 404; }
        char *ln = malloc(1u << 20); int k = 0;
        ds4_buf_puts(out, "{\"rows\":[");
        while (k < want && k < 50 && fgets(ln, 1 << 20, f)) {
            size_t L = strlen(ln); while (L && (ln[L - 1] == '\n' || ln[L - 1] == '\r')) ln[--L] = 0;
            if (!L) continue;
            if (L > 4096) ln[4096] = 0;
            if (k++) ds4_buf_putc(out, ',');
            ds4_json_escape(out, ln);
        }
        free(ln); fclose(f);
        ds4_buf_puts(out, "]}");
        return 200;
    }
    if (!post) { ds4_buf_puts(out, "{\"error\":\"没有这个路径(或要 POST)\"}"); return 404; }
    char err[256] = "";
    if (!strcmp(rq->path, "/api/train/start")) return api_launch(rq, out, serving, port, false);
    if (!strcmp(rq->path, "/api/train/gen")) return api_launch(rq, out, serving, port, true);
    if (!strcmp(rq->path, "/api/train/stop")) { if (serving) return reply(out, false, "要从主进程 ds4-train 的页面操作", 409); return reply(out, tr_job_stop(err, sizeof err), err, 409); }
    if (!strcmp(rq->path, "/api/train/upload")) return api_upload(rq, out);
    if (!strcmp(rq->path, "/api/train/serve")) {   /* 大厅 → 起模型子进程(按 gguf/serve_pick.txt) */
        if (serving) return reply(out, false, "服务已经在跑", 409);
        return reply(out, tr_model_start(TR_MODEL_PORT(port), false, err, sizeof err), err, 409);
    }
    ds4_buf_puts(out, "{\"error\":\"没有这个路径\"}");
    return 404;
}
