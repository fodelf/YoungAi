/* ============================ extract_completion ============================ */

/* 逐分支照抄 .py 的 extract_completion(text, prompt, stops, strip_prompt, close_brace)。
 * 返回带长度的 str_t —— "完整重写"形态的前缀标记 "\x00FULL\x00" 含 NUL 字节。*/
static str_t extract_completion(const char *text, const char *prompt, const char *const *stops,
                                int strip_prompt, int close_brace) {
    /* 服务端 mode:code 常给围栏; 取第一个围栏体, 否则用原文 */
    slist fences = find_fences(text);
    char *body = xstrdup(fences.n ? fences.v[0] : text);

    /* chat 完整重写形态(2026-08-18): 围栏体自带入口函数完整定义 —— 直接原样返回,
     * eval 侧当独立完整程序拼 test(续写式剥重叠对它必然错位)。*/
    if (strip_prompt && fences.n) {
        slist dm = find_decl_names(prompt, 1);
        if (dm.n) {
            const char *ent = dm.v[dm.n - 1];
            const char *best = NULL; size_t bl = 0;
            for (size_t i = 0; i < fences.n; i++) {
                const char *f = fences.v[i];
                if (!has_decl(f, ent) || !has_full_decl(f, ent)) continue;
                size_t L = strlen(f);
                if (!best || L > bl) { best = f; bl = L; }   /* max(key=len): 并列取先出现者 */
            }
            if (best) {
                size_t rl = rstrip_len(best, bl);
                sb_t o = {0};
                sb_add(&o, FULL_MARK, 6);
                sb_add(&o, best, rl);
                sb_ch(&o, '\n');
                free(body); sl_free(&dm); sl_free(&fences);
                str_t r = { o.p, o.n };
                return r;
            }
        }
        sl_free(&dm);
    }

    /* 若模型把题面(签名)复述了, 剥掉与 prompt 重叠的头部。
     * completions 裸续写(strip_prompt=0)禁用: response 本就是纯续写, 而"最后一行锚"
     * 对 HumanEval 恒为 \"\"\" — 会命中续写里任意 docstring 把正确答案整段扔掉(t4 实证)。*/
    char *p_tail = py_rstrip(prompt);
    if (!strip_prompt) {
        /* pass */
    } else if (*p_tail && strstr(body, p_tail)) {
        const char *hit = strstr(body, p_tail);
        char *nb = xstrdup(hit + strlen(p_tail));
        free(body); body = nb;
    } else {
        char *sig = NULL;
        if (*p_tail) { char *ll = py_last_line(p_tail); sig = py_strip(ll, strlen(ll)); free(ll); }
        else sig = xstrdup("");
        if (*sig && strstr(body, sig)) {
            const char *hit = strstr(body, sig);
            char *nb = xstrdup(hit + strlen(sig));
            free(body); body = nb;
            if (body[0] == ':') { char *t = xstrdup(body + 1); free(body); body = t; }
        }
        free(sig);
    }
    free(p_tail);

    /* 客户端截断(不依赖服务端 stop 参数) */
    size_t cut = strlen(body);
    for (int i = 0; stops[i]; i++) {
        const char *h = strstr(body, stops[i]);
        if (h) { size_t k = (size_t)(h - body); if (k < cut) cut = k; }
    }
    body[cut] = 0;

    /* Go 漂移截断(2026-07-27 v2): ①入口函数被重复声明处=确定的漂移起点, 截掉;
     * ②截到最后一个第 0 列 "}"(gofmt 顶层函数闭合必在列 0) —— 不用花括号深度配平,
     * Go 类型语法的内联括号(interface{}/struct{})会骗计数器。*/
    if (close_brace) {
        slist ms = find_decl_names(prompt, 0);
        if (ms.n) {
            char pat[256];
            snprintf(pat, sizeof pat, "func %s(", ms.v[ms.n - 1]);
            const char *h = strstr(body, pat);
            if (h) body[(size_t)(h - body)] = 0;
        }
        sl_free(&ms);
        /* rfind("\n}") */
        size_t bl = strlen(body);
        if (bl >= 2) for (size_t i = bl - 1; i > 0; i--) {
            if (body[i - 1] == '\n' && body[i] == '}') { body[i + 1] = 0; break; }
        }
    }
    size_t rl = rstrip_len(body, strlen(body));
    sb_t o = {0};
    sb_add(&o, body, rl);
    sb_ch(&o, '\n');
    free(body); sl_free(&fences);
    str_t r = { o.p, o.n };
    return r;
}

/* ============================ 子进程执行 ============================ */

typedef struct { int rc, timed_out, spawn_errno; char *out; size_t out_n; char *err; size_t err_n; } proc_t;

static pthread_mutex_t g_fork_mu = PTHREAD_MUTEX_INITIALIZER;
static double now_s(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec * 1e-6; }

static void sink_append(char **buf, size_t *n, size_t *cap, const char *d, size_t dn) {
    if (*n + dn + 1 > *cap) {
        size_t c = *cap ? *cap : 65536;
        while (c < *n + dn + 1) c *= 2;
        *buf = xrealloc(*buf, c); *cap = c;
    }
    memcpy(*buf + *n, d, dn); *n += dn; (*buf)[*n] = 0;
    /* 只有末尾 400-800 字符会被读; 超过 2 MiB 时丢头保尾, 防生成代码狂打日志把内存吃干
     * (.py 无上限, 这是转录清单里显式记录的偏差, 不影响任何判分读数)。*/
    if (*n > (2u << 20)) { size_t keep = 1u << 20; memmove(*buf, *buf + (*n - keep), keep); *n = keep; (*buf)[*n] = 0; }
}

/* subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=T) */
static void run_proc(char *const argv[], const char *cwd, double timeout_s, proc_t *r) {
    memset(r, 0, sizeof *r);
    int po[2], pe[2], px[2];
    pthread_mutex_lock(&g_fork_mu);
    if (pipe(po) || pipe(pe) || pipe(px)) { pthread_mutex_unlock(&g_fork_mu); r->spawn_errno = errno; return; }
    for (int i = 0; i < 2; i++) { fcntl(po[i], F_SETFD, FD_CLOEXEC); fcntl(pe[i], F_SETFD, FD_CLOEXEC); fcntl(px[i], F_SETFD, FD_CLOEXEC); }
    pid_t pid = fork();
    if (pid == 0) {
        dup2(po[1], 1); dup2(pe[1], 2);
        int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) { dup2(devnull, 0); close(devnull); }
        if (cwd && chdir(cwd) != 0) { int e = errno; ssize_t w = write(px[1], &e, sizeof e); (void)w; _exit(127); }
        execvp(argv[0], argv);
        int e = errno; ssize_t w = write(px[1], &e, sizeof e); (void)w;
        _exit(127);
    }
    close(po[1]); close(pe[1]); close(px[1]);
    pthread_mutex_unlock(&g_fork_mu);
    if (pid < 0) { close(po[0]); close(pe[0]); close(px[0]); r->spawn_errno = errno; return; }

    int se = 0;
    if (read(px[0], &se, sizeof se) == (ssize_t)sizeof se) r->spawn_errno = se;
    close(px[0]);

    size_t ocap = 0, ecap = 0;
    double deadline = now_s() + timeout_s;
    int of = po[0], ef = pe[0];
    fcntl(of, F_SETFL, O_NONBLOCK); fcntl(ef, F_SETFL, O_NONBLOCK);
    char tmp[65536];
    while (of >= 0 || ef >= 0) {
        double left = deadline - now_s();
        if (left <= 0) { r->timed_out = 1; break; }
        struct pollfd pf[2]; int np = 0;
        int io = -1, ie = -1;
        if (of >= 0) { pf[np].fd = of; pf[np].events = POLLIN; io = np++; }
        if (ef >= 0) { pf[np].fd = ef; pf[np].events = POLLIN; ie = np++; }
        int pr = poll(pf, (nfds_t)np, (int)(left * 1000) + 1);
        if (pr < 0) { if (errno == EINTR) continue; break; }
        if (pr == 0) continue;
        if (io >= 0 && (pf[io].revents & (POLLIN | POLLHUP | POLLERR))) {
            ssize_t k = read(of, tmp, sizeof tmp);
            if (k > 0) sink_append(&r->out, &r->out_n, &ocap, tmp, (size_t)k);
            else if (k == 0 || (k < 0 && errno != EAGAIN && errno != EINTR)) { close(of); of = -1; }
        }
        if (ie >= 0 && (pf[ie].revents & (POLLIN | POLLHUP | POLLERR))) {
            ssize_t k = read(ef, tmp, sizeof tmp);
            if (k > 0) sink_append(&r->err, &r->err_n, &ecap, tmp, (size_t)k);
            else if (k == 0 || (k < 0 && errno != EAGAIN && errno != EINTR)) { close(ef); ef = -1; }
        }
    }
    if (r->timed_out) {
        kill(pid, SIGKILL);
        if (of >= 0) close(of);
        if (ef >= 0) close(ef);
    } else {
        if (of >= 0) close(of);
        if (ef >= 0) close(ef);
    }
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    r->rc = WIFEXITED(st) ? WEXITSTATUS(st) : -(WTERMSIG(st));
    if (!r->out) r->out = xstrdup("");
    if (!r->err) r->err = xstrdup("");
}
static void proc_free(proc_t *r) { free(r->out); free(r->err); r->out = r->err = NULL; }

/* text=True 的 universal newlines: \r\n / 孤立 \r → \n */
static char *unix_newlines(const char *s, size_t n, size_t *out_n) {
    char *o = xmalloc(n + 1); size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\r') { if (i + 1 < n && s[i + 1] == '\n') i++; o[k++] = '\n'; }
        else o[k++] = s[i];
    }
    o[k] = 0; if (out_n) *out_n = k;
    return o;
}
/* r.stderr[-N:] / r.stdout[-N:] —— 按字符切 */
static char *tail_of(const char *s, size_t n, int chars) {
    size_t nn; char *u = unix_newlines(s, n, &nn);
    size_t off = tail_chars(u, nn, chars);
    char *r = xstrndup(u + off, nn - off);
    free(u);
    return r;
}

/* ============================ 判定器 ============================ */

typedef struct { const char *prompt, *test, *entry_point, *test_setup, *import_; } row_t;
typedef struct { int ok; char *err; } judge_t;

static int is_full(const str_t *c) { return c->n >= 6 && !memcmp(c->p, FULL_MARK, 6); }

static judge_t eval_python(const row_t *row, const str_t *completion, double timeout_s) {
    judge_t J = { 0, NULL };
    sb_t prog = {0};
    if (is_full(completion)) {
        sb_add(&prog, completion->p + 6, completion->n - 6);
    } else {
        sb_puts(&prog, row->prompt);
        sb_add(&prog, completion->p, completion->n);
    }
    sb_ch(&prog, '\n');
    sb_puts(&prog, row->test);
    sb_fmt(&prog, "\ncheck(%s)\n", row->entry_point);
    /* sys.executable → C 侧走 PATH 上的 python3(被判代码的目标运行时) */
    char *argv[] = { (char *)"python3", (char *)"-c", prog.p, NULL };
    proc_t r;
    run_proc(argv, NULL, timeout_s, &r);
    if (r.spawn_errno) {
        char b[256]; snprintf(b, sizeof b, "harness: [Errno %d] %s: 'python3'", r.spawn_errno, strerror(r.spawn_errno));
        J.ok = 0; J.err = xstrdup(b);
    } else if (r.timed_out) {
        J.ok = 0; J.err = xstrdup("timeout");
    } else {
        J.ok = (r.rc == 0);
        J.err = J.ok ? xstrdup("") : tail_of(r.err, r.err_n, 800);
    }
    proc_free(&r); sb_free(&prog);
    return J;
}

static judge_t eval_go(const row_t *row, const str_t *completion, double timeout_s) {
    judge_t J = { 0, NULL };
    /* 单文件拼装(2026-07-27 修): humaneval-x go 的 test 与解答共享同文件 import。
     * 合并全部 import 路径去重 → 一个 _test.go; goimports -w 删未用 import。*/
    char tmpl[4096];
    const char *td = getenv("TMPDIR");
    if (!td || !*td) td = "/tmp";
    size_t tl = strlen(td);
    snprintf(tmpl, sizeof tmpl, "%s%spubbench_go_XXXXXX", td, (tl && td[tl - 1] == '/') ? "" : "/");
    char *d = mkdtemp(tmpl);
    if (!d) { J.ok = 0; char b[256]; snprintf(b, sizeof b, "harness: [Errno %d] %s", errno, strerror(errno)); J.err = xstrdup(b); return J; }

    const char *setup = row->test_setup ? row->test_setup : "";
    sb_t sol = {0};
    if (is_full(completion)) sb_add(&sol, completion->p + 6, completion->n - 6);
    else { sb_puts(&sol, row->prompt); sb_add(&sol, completion->p, completion->n); }

    sb_t setup_test = {0};
    sb_puts(&setup_test, setup); sb_ch(&setup_test, '\n'); sb_puts(&setup_test, row->test);

    slist paths = {0};
    const char *srcs[4] = { row->import_ ? row->import_ : "", sol.p ? sol.p : "", setup, row->test };
    for (int i = 0; i < 4; i++) {
        slist got = go_import_paths(srcs[i]);
        for (size_t t = 0; t < got.n; t++) {
            int dup = 0;
            for (size_t u = 0; u < paths.n; u++) if (!strcmp(paths.v[u], got.v[t])) { dup = 1; break; }
            if (!dup) sl_push(&paths, xstrdup(got.v[t]));
        }
        sl_free(&got);
    }
    sb_t src = {0};
    sb_puts(&src, "package main\n\n");
    if (paths.n) {
        sb_puts(&src, "import (\n");
        for (size_t t = 0; t < paths.n; t++) sb_fmt(&src, "    \"%s\"\n", paths.v[t]);
        sb_puts(&src, ")\n");
    }
    sb_ch(&src, '\n');
    char *h1 = go_strip_headers(sol.p ? sol.p : "");
    char *h2 = go_strip_headers(setup_test.p ? setup_test.p : "");
    sb_puts(&src, h1); sb_ch(&src, '\n'); sb_puts(&src, h2);
    free(h1); free(h2);
    sl_free(&paths); sb_free(&sol); sb_free(&setup_test);

    char p[4200];
    snprintf(p, sizeof p, "%s/sol_test.go", d);
    FILE *f = fopen(p, "w");
    if (!f) { char b[256]; snprintf(b, sizeof b, "harness: [Errno %d] %s: '%s'", errno, strerror(errno), p); J.ok = 0; J.err = xstrdup(b); sb_free(&src); return J; }
    fwrite(src.p, 1, src.n, f); fclose(f); sb_free(&src);

    proc_t r;
    char gi[4096];
    const char *home = getenv("HOME");
    snprintf(gi, sizeof gi, "%s/go/bin/goimports", home ? home : "");
    struct stat st;
    if (stat(gi, &st) == 0) {
        char *av[] = { gi, (char *)"-w", p, NULL };
        run_proc(av, NULL, 60, &r);
        if (r.timed_out) { proc_free(&r); J.ok = 0; J.err = xstrdup("timeout"); return J; }
        proc_free(&r);
    }
    { char *av[] = { (char *)"go", (char *)"mod", (char *)"init", (char *)"pubbench", NULL };
      run_proc(av, d, 60, &r);
      if (r.spawn_errno) { char b[256]; snprintf(b, sizeof b, "harness: [Errno %d] %s: 'go'", r.spawn_errno, strerror(r.spawn_errno)); proc_free(&r); J.ok = 0; J.err = xstrdup(b); return J; }
      if (r.timed_out) { proc_free(&r); J.ok = 0; J.err = xstrdup("timeout"); return J; }
      proc_free(&r); }
    { char *av[] = { (char *)"go", (char *)"mod", (char *)"tidy", NULL };
      run_proc(av, d, 120, &r);
      if (r.timed_out) { proc_free(&r); J.ok = 0; J.err = xstrdup("timeout"); return J; }
      proc_free(&r); }
    { char *av[] = { (char *)"go", (char *)"test", (char *)"./...", NULL };
      run_proc(av, d, timeout_s, &r);
      if (r.timed_out) { proc_free(&r); J.ok = 0; J.err = xstrdup("timeout"); return J; }
      J.ok = (r.rc == 0);
      if (J.ok) J.err = xstrdup("");
      else {
          char *a = tail_of(r.out, r.out_n, 400), *b = tail_of(r.err, r.err_n, 400);
          sb_t e = {0}; sb_puts(&e, a); sb_puts(&e, b);
          free(a); free(b);
          J.err = e.p ? e.p : xstrdup("");
      }
      proc_free(&r); }
    return J;
}

/* ============================ call_server ============================ */

typedef struct { char *txt; size_t txt_n; char *raw; size_t raw_n; double dt; int failed; char err[512]; } srv_t;

/* api=completions: BASE 模型口径 = /v1/completions raw 裸续写(BOS+题面), 返回纯续写文本。
 * api=chat: 原 /v1/chat/completions mode:code(instruct/chat 部署用)。*/
static srv_t call_server(const char *url, const char *prompt, int max_tokens, int timeout, const char *mode, const char *api) {
    srv_t R; memset(&R, 0, sizeof R);
    int is_comp = !strcmp(api, "completions");
    sb_t body = {0};
    if (is_comp) {
        sb_puts(&body, "{\"model\": \"ds4\", \"prompt\": ");
        sb_t pp = {0}; sb_puts(&pp, BOS); sb_puts(&pp, prompt);
        json_esc(&body, pp.p, pp.n, 1);
        sb_free(&pp);
        sb_fmt(&body, ", \"temperature\": 0, \"max_tokens\": %d, \"raw\": true}", max_tokens);
    } else {
        sb_puts(&body, "{\"model\": \"ds4\", \"messages\": [{\"role\": \"user\", \"content\": ");
        json_esc(&body, prompt, strlen(prompt), 1);
        sb_fmt(&body, "}], \"temperature\": 0, \"max_tokens\": %d, \"stream\": false, \"mode\": ", max_tokens);
        json_esc(&body, mode, strlen(mode), 1);
        sb_ch(&body, '}');
    }
    char full[2048];
    size_t ul = strlen(url);
    while (ul && url[ul - 1] == '/') ul--;              /* url.rstrip("/") */
    snprintf(full, sizeof full, "%.*s%s", (int)ul, url, is_comp ? "/v1/completions" : "/v1/chat/completions");

    double t0 = now_s();
    char *raw = NULL; size_t raw_n = 0;
    if (http_post_json(full, body.p, timeout, &raw, &raw_n, R.err, sizeof R.err) != 0) {
        R.failed = 1; sb_free(&body);
        return R;
    }
    R.dt = now_s() - t0;
    sb_free(&body);
    size_t dec_n;
    char *dec = utf8_replace(raw, raw_n, &dec_n);       /* .decode("utf-8","replace") */
    free(raw);
    R.raw = dec; R.raw_n = dec_n;
    R.txt = xstrdup(""); R.txt_n = 0;                   /* txt = "" 起手, 解析失败即回落 raw */
    jv *j = json_loads(dec, dec_n);
    if (j) {
        jv *ch = jget(j, "choices");
        if (ch && ch->type == JARR && ch->n > 0) {
            jv *c = ch->kid[0];
            jv *t = NULL;
            if (is_comp) t = jget(c, "text");
            else { jv *m = jget(c, "message"); t = m ? jget(m, "content") : NULL; }
            if (t && t->type == JSTR) { free(R.txt); R.txt = xstrndup(t->s, t->slen); R.txt_n = t->slen; }
            else { free(R.txt); R.txt = xstrndup(dec, dec_n); R.txt_n = dec_n; }
        } else { free(R.txt); R.txt = xstrndup(dec, dec_n); R.txt_n = dec_n; }
        jfree(j);
    } else { free(R.txt); R.txt = xstrndup(dec, dec_n); R.txt_n = dec_n; }
    return R;
}

/* ============================ 参数 ============================ */

typedef struct {
    const char *suite, *url, *tag, *mode, *api, *out_dir;
    int limit, offset, max_tokens, http_timeout, jobs, eval_timeout;
    const char *cmp_a, *cmp_b, *rejudge_path;
    int selftest;
} args_t;

/* ============================ run_suite ============================ */

typedef struct {
    const args_t *a;
    rows_t rows;
    const char *const *stops;
    int is_go;
    const char *lang;
    jv **results;
    int done_n;
    size_t next;
    pthread_mutex_t mu;
} job_t;

static void one_task(job_t *J, size_t k) {
    const args_t *a = J->a;
    jv *row = J->rows.row[k];
    char tidbuf[128];
    const char *tid = jstr(row, "task_id", NULL);
    if (!tid) { snprintf(tidbuf, sizeof tidbuf, "%s/%d", a->suite, a->offset + (int)k); tid = tidbuf; }

    /* ★Go 生成端 prompt 补全(2026-08-05)★: 判定端一直补 package+import 编译, 生成端却发
     * 裸函数 — BASE 模型的 Go 语料函数永远在文件头之后, 裸函数=分布外 ⇒ TODO 弃权/风格漂移。*/
    const char *prompt = jstr_req(row, "prompt");
    sb_t gp = {0};
    if (J->is_go) {
        const char *imp0 = jstr(row, "import", "");
        char *imp = py_strip(imp0, strlen(imp0));
        sb_puts(&gp, "package main\n\n");
        if (*imp) { sb_puts(&gp, imp); sb_puts(&gp, "\n\n"); }
        const char *pl = prompt;
        while (*pl == '\n') pl++;                        /* .lstrip("\n") */
        sb_puts(&gp, pl);
        free(imp);
    } else {
        sb_puts(&gp, prompt);
    }
    const char *gen_prompt = gp.p ? gp.p : "";

    srv_t S = call_server(a->url, gen_prompt, a->max_tokens, a->http_timeout, a->mode, a->api);
    str_t comp; char *err; int ok; double dt;
    char *text, *raw; size_t text_n, raw_n;
    if (S.failed) {
        text = xstrdup(""); text_n = 0; raw = xstrdup(""); raw_n = 0; dt = 0.0;
        comp = str_cstr(""); ok = 0;
        char b[600]; snprintf(b, sizeof b, "request: %s", S.err);
        err = xstrdup(b);
    } else {
        text = S.txt; text_n = S.txt_n; raw = S.raw; raw_n = S.raw_n; dt = S.dt;
        comp = extract_completion(text, gen_prompt, J->stops,
                                  strcmp(a->api, "completions") != 0, J->is_go);
        row_t r = { prompt, jstr(row, "test", NULL), jstr(row, "entry_point", NULL),
                    jstr(row, "test_setup", ""), jstr(row, "import", "") };
        /* .py 里 row["test"]/row["entry_point"] 在 eval_* 的 try 之外, 缺字段会被
         * one_task 的 except 兜成 err="request: 'test'"; 两份入库数据集恒有这些字段,
         * C 侧直接判死(转录清单 B 组) */
        if (!r.test || (!J->is_go && !r.entry_point)) die("KeyError: 'test'");
        judge_t j = J->is_go ? eval_go(&r, &comp, a->eval_timeout) : eval_python(&r, &comp, a->eval_timeout);
        ok = j.ok; err = j.err;
    }

    jv *rec = jnew(JOBJ);
    jpush(rec, xstrdup("task_id"), jstrv(tid, strlen(tid)));
    jpush(rec, xstrdup("lang"), jstrv(J->lang, strlen(J->lang)));
    jpush(rec, xstrdup("tag"), jstrv(a->tag, strlen(a->tag)));
    jpush(rec, xstrdup("pass"), jboolv(ok));
    jpush(rec, xstrdup("err"), jstrv(err, strlen(err)));
    jpush(rec, xstrdup("gen_seconds"), jfloatv(py_round1(dt)));
    jpush(rec, xstrdup("prompt"), jstrv(prompt, strlen(prompt)));
    jpush(rec, xstrdup("completion"), jstrv(comp.p, comp.n));
    jpush(rec, xstrdup("response_text"), jstrv(text, text_n));
    jpush(rec, xstrdup("response_raw"), jstrv(raw, raw_n));

    pthread_mutex_lock(&J->mu);
    J->results[k] = rec;
    int dn = ++J->done_n;
    pthread_mutex_unlock(&J->mu);

    char suffix[512] = "";
    if (*err) {
        char *ll = py_last_line(err);
        size_t hb = head_chars(ll, strlen(ll), 80);
        snprintf(suffix, sizeof suffix, " (%.*s)", (int)hb, ll);
        free(ll);
    }
    logln("[%d/%zu] %s gen=%.0fs %s%s", dn, J->rows.n, tid, dt, ok ? "PASS" : "FAIL", suffix);

    free(text); free(raw); free(err); free(comp.p); sb_free(&gp);
}

static void *worker_main(void *p) {
    job_t *J = p;
    while (1) {
        pthread_mutex_lock(&J->mu);
        size_t k = J->next++;
        pthread_mutex_unlock(&J->mu);
        if (k >= J->rows.n) break;
        one_task(J, k);
    }
    return NULL;
}

/* Python 切片 rows[off : off+limit] 语义(含负下标折返与钳位) */
