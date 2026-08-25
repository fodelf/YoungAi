static rows_t py_slice(rows_t r, int off, int lim) {
    long n = (long)r.n;
    long a = off, b = (long)off + (long)lim;
    if (a < 0) { a += n; if (a < 0) a = 0; }
    if (a > n) a = n;
    if (b < 0) { b += n; if (b < 0) b = 0; }
    if (b > n) b = n;
    if (b < a) b = a;
    rows_t o; o.row = r.row + a; o.n = (size_t)(b - a);
    return o;
}

static void run_suite(const args_t *a) {
    job_t J; memset(&J, 0, sizeof J);
    pthread_mutex_init(&J.mu, NULL);
    J.a = a;
    int is_go = strcmp(a->suite, "humaneval") != 0;
    J.is_go = is_go;
    if (!is_go) { J.rows = dataset(0); J.stops = PY_STOP; J.lang = "python"; }
    else { J.rows = dataset(1); J.stops = GO_STOP; J.lang = "go"; }
    J.rows = py_slice(J.rows, a->offset, a->limit);
    makedirs(a->out_dir);
    char out_path[4096];
    snprintf(out_path, sizeof out_path, "%s/pubbench_%s_%s.jsonl", a->out_dir, a->suite, a->tag);
    J.results = xmalloc((J.rows.n ? J.rows.n : 1) * sizeof *J.results);
    memset(J.results, 0, (J.rows.n ? J.rows.n : 1) * sizeof *J.results);

    /* ★并发(2026-08-18 用户令"测试并发不要串行")★: server 推理单 worker 串行, 但并发请求
     * 消掉 client 间隙+判题(go test 1-5s/题)与生成流水; 结果按题序落盘。
     * ThreadPoolExecutor.map 的语义 = 任务按序入队 FIFO, 结果按序取回。*/
    if (a->jobs <= 0) die("ValueError: max_workers must be greater than 0");
    int nth = a->jobs;
    if ((size_t)nth > J.rows.n) nth = (int)J.rows.n;
    if (nth < 1) nth = 1;
    pthread_t *th = xmalloc((size_t)nth * sizeof *th);
    for (int t = 0; t < nth; t++) pthread_create(&th[t], NULL, worker_main, &J);
    for (int t = 0; t < nth; t++) pthread_join(th[t], NULL);

    FILE *out = fopen(out_path, "w");
    if (!out) die("OSError: [Errno %d] %s: '%s'", errno, strerror(errno), out_path);
    int n_pass = 0;
    for (size_t k = 0; k < J.rows.n; k++) {
        jv *rec = J.results[k];
        jv *pv = jget(rec, "pass");
        n_pass += (pv && pv->b) ? 1 : 0;
        sb_t line = {0};
        json_dump(&line, rec, 0);                       /* ensure_ascii=False */
        fwrite(line.p, 1, line.n, out); fputc('\n', out);
        sb_free(&line);
    }
    fclose(out);
    const char *note = is_go ? " (go judge=experimental, 人工抽查后作数)" : "";
    logln("[done] %s tag=%s pass@1 = %d/%zu%s", a->suite, a->tag, n_pass, J.rows.n, note);
    logln("[raw] %s", out_path);
    jv *sum = jnew(JOBJ);
    jpush(sum, xstrdup("suite"), jstrv(a->suite, strlen(a->suite)));
    jpush(sum, xstrdup("tag"), jstrv(a->tag, strlen(a->tag)));
    jpush(sum, xstrdup("pass"), jintv(n_pass));
    jpush(sum, xstrdup("total"), jintv((long long)J.rows.n));
    jpush(sum, xstrdup("raw"), jstrv(out_path, strlen(out_path)));
    sb_t line = {0};
    json_dump(&line, sum, 1);                           /* print(json.dumps(...)) = ensure_ascii 默认 True */
    fwrite(line.p, 1, line.n, stdout); fputc('\n', stdout);
    sb_free(&line); jfree(sum);
    free(th); free(J.results);
}

/* ============================ rejudge ============================ */

typedef struct {
    rows_t rows;
    const char *api;
    size_t next;
    int n_pass;
    pthread_mutex_t mu;
} rej_t;

static void rejudge_one(rej_t *R, size_t idx) {
    jv *r = R->rows.row[idx];
    const char *lang = jstr_req(r, "lang");
    int is_go = strcmp(lang, "python") != 0;
    const char *const *stops = is_go ? GO_STOP : PY_STOP;

    const char *prompt = jstr_req(r, "prompt");
    const char *test = jstr(r, "test", "");
    const char *entry = jstr(r, "entry_point", "");
    const char *setup = "", *imp = "";
    /* 数据集字段(test/entry_point)不在 jsonl 里 → 从数据集按 task_id 回填 */
    if (!*test) {
        rows_t ds = dataset(is_go);
        const char *tid = jstr_req(r, "task_id");
        jv *full = NULL;
        for (size_t i = 0; i < ds.n; i++) {
            const char *t = jstr(ds.row[i], "task_id", NULL);
            if (t && !strcmp(t, tid)) full = ds.row[i];   /* dict 语义: 后者覆盖 */
        }
        if (!full) die("KeyError: '%s'", tid);
        test = jstr_req(full, "test");
        entry = jstr(full, "entry_point", "");
        setup = jstr(full, "test_setup", "");
        imp = jstr(full, "import", "");
    }
    const char *rtext = jstr_req(r, "response_text");
    str_t comp = extract_completion(rtext, prompt, stops, strcmp(R->api, "completions") != 0, is_go);
    row_t row = { prompt, test, entry, setup, imp };
    judge_t j = is_go ? eval_go(&row, &comp, 15) : eval_python(&row, &comp, 15);

    jv *pv = jget(r, "pass");
    int was = (pv && pv->type == JBOOL) ? pv->b : 0;
    char changed[64] = "";
    if (j.ok != was) snprintf(changed, sizeof changed, "  [改判 %s→%s]", was ? "True" : "False", j.ok ? "True" : "False");
    char suffix[512] = "";
    if (*j.err) {
        char *ll = py_last_line(j.err);
        size_t hb = head_chars(ll, strlen(ll), 70);
        snprintf(suffix, sizeof suffix, " (%.*s)", (int)hb, ll);
        free(ll);
    }
    logln("%s: %s%s%s", jstr_req(r, "task_id"), j.ok ? "PASS" : "FAIL", changed, suffix);

    jset(r, "pass", jboolv(j.ok));
    jset(r, "err", jstrv(j.err, strlen(j.err)));
    jset(r, "completion", jstrv(comp.p, comp.n));

    pthread_mutex_lock(&R->mu);
    R->n_pass += j.ok ? 1 : 0;
    pthread_mutex_unlock(&R->mu);
    free(j.err); free(comp.p);
}
static void *rejudge_worker(void *p) {
    rej_t *R = p;
    while (1) {
        pthread_mutex_lock(&R->mu);
        size_t k = R->next++;
        pthread_mutex_unlock(&R->mu);
        if (k >= R->rows.n) break;
        rejudge_one(R, k);
    }
    return NULL;
}

/* 离线重判: 用已存 response_text 重新抽取+评测(修抽取器后免重生成), 原地重写 pass/completion */
static void rejudge(const char *path, const char *api) {
    rej_t R; memset(&R, 0, sizeof R);
    pthread_mutex_init(&R.mu, NULL);
    R.rows = load_jsonl(path);
    R.api = api;
    int nth = 8;                                        /* 判题并发(用户令 08-18) */
    if ((size_t)nth > R.rows.n) nth = (int)R.rows.n;
    if (nth < 1) nth = 1;
    pthread_t th[8];
    for (int t = 0; t < nth; t++) pthread_create(&th[t], NULL, rejudge_worker, &R);
    for (int t = 0; t < nth; t++) pthread_join(th[t], NULL);

    FILE *f = fopen(path, "w");
    if (!f) die("OSError: [Errno %d] %s: '%s'", errno, strerror(errno), path);
    for (size_t i = 0; i < R.rows.n; i++) {
        sb_t line = {0};
        json_dump(&line, R.rows.row[i], 0);
        fwrite(line.p, 1, line.n, f); fputc('\n', f);
        sb_free(&line);
    }
    fclose(f);
    logln("[rejudge] %s: pass@1 = %d/%zu", path, R.n_pass, R.rows.n);
}

/* ============================ verdict / compare ============================ */

/* 两个互相独立的决策闸。阈值是"决策带"不是测量值: 粗到经得起 n=20 的 ±2 题噪声,
 * 编码的是 2026-07-25 对话已定的逻辑(fable5.md): 参赛的标的=可运行作品+诚实方法论;
 * 买机的标的=产品阶段(驻留/MTP/KV/日用速度)是否真实存在。不是大赛评审标准。*/
static void verdict(int B, int Q, int N, const char *base_tag, const char *quant_tag) {
    printf("\n");
    printf("== 判决闸门: 基线=%s B=%d/%d | 量化=%s Q=%d/%d | n=%d 噪声底≈±2题 ==\n",
           base_tag, B, N, quant_tag, Q, N, N);
    if ((double)B < 0.6 * N) {
        printf("[INVALID] 基线 %d/%d < 60%% — q2 基线编程域不该这么低; 先查 harness/服务层"
               "(历史教训: 服务层bug曾把好引擎打成零代码), 本轮分数不作任何决策依据\n", B, N);
        return;
    }
    if (B == 0) die("ZeroDivisionError: division by zero");
    double ret = (double)Q / (double)B;
    /* 故事口径(用户裁决 07-25): 消费级单机+量化模型+直接可用编程。双机组网/慢速运行不构成故事。
     * 阈值=Claude 的专业判断(07-25 晚定稿, 责任署名, 错误可检验形态在 fable5.md 当日条目):
     * 锚=竞争替代 — 64G Mac 上人人五分钟可跑 Qwen3-coder 30B 级免费模型, 故事必须明显打赢它;
     * 50% 保留(≈CodeLlama-13B 档)打不赢, 保留七成的 frontier 级 MoE + 1M ctx + 自研引擎才成立。*/
    const char *g1, *g2;
    if ((double)Q >= 0.6 * N && ret >= 0.7)
        g1 = "GO — 故事成立: 单机可用速度实录为 Demo; 现役双机只作开发素材, 购机为截稿(08-16)前关键路径";
    else if ((double)Q >= 0.5 * N && ret >= 0.5)
        g1 = "BORDER — 打不赢'懒人替代'(64G 免费 30B coder), quant 故事不达标; fallback=引擎+官方q2单机(96G)故事存在但评级偏弱, 默认弃本届";
    else
        g1 = "NO — 质量主张撑不住, 什么机器都救不了故事; 回算法迭代";
    if ((double)Q >= 0.6 * N && ret >= 0.7)
        g2 = "GO — 买 96G M4 Max(非 64G: 多的几千块买'量化 vs 官方q2 同机对照演示'位+日常 parity 参照; 按 9.5t/s@120GB/s 线性外推 546GB/s≈30-40t/s, 到手 24h ds4-bench 实测替换)";
    else
        g2 = "不买本周期 — 竞争锚下故事不成立, 机器无标的; 下个质量里程碑再判";
    printf("[GOAI 参赛闸] %s\n", g1);
    printf("[买机闸]      %s\n", g2);
    printf("[口径] 质量闸先行, 两闸都由质量分决定; GO 态下购机档位服务'消费级单机直接编程'故事; "
           "报名前读 goaihz.com 章程 (三赛道截止 2026-08-16, 具身 08-20)\n");
    if (!strncmp(g1, "GO", 2) || !strncmp(g2, "GO", 2))
        printf("[下单前加固] GO≠下单, 动钱前三点全过: "
               "① LIMIT=164 跑满全套收窄置信区间(n=20 delta CI≈±31%% → n=164 ≈±11%%) "
               "② 第二条腿 gen_coding_probe 真实任务可用性过(HumanEval 是短函数补全, ≠真实编辑工作流) "
               "③ 新机到手 24h 内 ds4-bench 实测替换所有速度外推, 不达可用速度按残值退/售\n");
}

/* dict{task_id: rec}(保序, 同键后者覆盖) */
typedef struct { char **k; jv **v; size_t n, cap; } dict_t;
static void dict_put(dict_t *d, const char *k, jv *v) {
    for (size_t i = 0; i < d->n; i++) if (!strcmp(d->k[i], k)) { d->v[i] = v; return; }
    if (d->n == d->cap) { d->cap = d->cap ? d->cap * 2 : 32; d->k = xrealloc(d->k, d->cap * sizeof *d->k); d->v = xrealloc(d->v, d->cap * sizeof *d->v); }
    d->k[d->n] = xstrdup(k); d->v[d->n] = v; d->n++;
}
static jv *dict_get(const dict_t *d, const char *k) {
    for (size_t i = 0; i < d->n; i++) if (!strcmp(d->k[i], k)) return d->v[i];
    return NULL;
}
static int cmp_str(const void *a, const void *b) { return strcmp(*(const char *const *)a, *(const char *const *)b); }

static void load_dict(const char *path, dict_t *d) {
    rows_t r = load_jsonl(path);
    for (size_t i = 0; i < r.n; i++) dict_put(d, jstr_req(r.row[i], "task_id"), r.row[i]);
}

static void compare(const char *fa, const char *fb) {
    dict_t A = {0}, B = {0};
    load_dict(fa, &A); load_dict(fb, &B);
    if (!A.n || !B.n) die("StopIteration");
    const char *ka = jstr_req(A.v[0], "tag"), *kb = jstr_req(B.v[0], "tag");
    /* both = sorted(set(A) & set(B)) —— Python 字符串序 = UTF-8 字节序 */
    char **both = xmalloc((A.n ? A.n : 1) * sizeof *both);
    size_t nb = 0;
    for (size_t i = 0; i < A.n; i++) if (dict_get(&B, A.k[i])) both[nb++] = A.k[i];
    qsort(both, nb, sizeof *both, cmp_str);
    int pa = 0, pb = 0;
    for (size_t i = 0; i < nb; i++) {
        jv *x = jget(dict_get(&A, both[i]), "pass"), *y = jget(dict_get(&B, both[i]), "pass");
        pa += (x && x->b) ? 1 : 0;
        pb += (y && y->b) ? 1 : 0;
    }
    printf("(顺序约定: A=基线=%s, B=量化=%s)\n", ka, kb);
    printf("%-18s %8s %8s\n", "task_id", ka, kb);
    for (size_t i = 0; i < nb; i++) {
        jv *x = jget(dict_get(&A, both[i]), "pass"), *y = jget(dict_get(&B, both[i]), "pass");
        printf("%-18s %8s %8s\n", both[i], (x && x->b) ? "PASS" : ".", (y && y->b) ? "PASS" : ".");
    }
    printf("%-18s %5d/%zu %5d/%zu   delta(%s-%s) = %+d\n", "TOTAL", pa, nb, pb, nb, kb, ka, pb - pa);
    verdict(pa, pb, (int)nb, ka, kb);
    free(both);
}

/* ============================ selftest ============================ */

/* harness 自检: 拿数据集自带的 canonical_solution 冒充模型输出, 走与真跑完全同一条
 * 抽取+判定链路。**期望 164/164** —— 标准答案判不过 = harness 坏了(缺 go 工具链/
 * 抽取器误剥/拼装规则跑偏), 与被测模型的质量无关。换机器/换 Go 版本后先跑这个,
 * 免得把环境问题算到模型头上。*/
static int selftest(const char *suite, int limit, int offset, int eval_timeout) {
    int is_go = strcmp(suite, "humaneval") != 0;
    rows_t rows = py_slice(dataset(is_go), offset, limit);
    const char *lang = is_go ? "go" : "python";
    (void)lang;
    int n_pass = 0;
    slist failed = {0};
    double t0 = now_s();
    for (size_t k = 0; k < rows.n; k++) {
        jv *row = rows.row[k];
        char tidbuf[128];
        const char *tid = jstr(row, "task_id", NULL);
        if (!tid) { snprintf(tidbuf, sizeof tidbuf, "%s/%d", suite, offset + (int)k); tid = tidbuf; }
        /* canonical_solution 就是"续写部分", 与 completions 裸续写口径同形, 直接当 completion */
        const char *cs = jstr_req(row, "canonical_solution");
        str_t comp = str_cstr(cs);
        row_t r = { jstr_req(row, "prompt"), jstr(row, "test", NULL), jstr(row, "entry_point", NULL),
                    jstr(row, "test_setup", ""), jstr(row, "import", "") };
        if (!r.test) die("KeyError: 'test'");
        if (!is_go && !r.entry_point) die("KeyError: 'entry_point'");
        judge_t j = is_go ? eval_go(&r, &comp, eval_timeout) : eval_python(&r, &comp, eval_timeout);
        n_pass += j.ok ? 1 : 0;
        if (!j.ok) {
            sl_push(&failed, xstrdup(tid));
            char *ll = py_last_line(j.err);
            size_t hb = head_chars(ll, strlen(ll), 100);
            logln("%s: FAIL  (%.*s)", tid, (int)hb, ll);
            free(ll);
        } else if ((k + 1) % 20 == 0) {
            logln("[selftest] %zu/%zu ... %d pass", k + 1, rows.n, n_pass);
        }
        free(j.err); free(comp.p);
    }
    logln("[selftest] %s: %d/%zu 标准答案通过 (%.0fs)", suite, n_pass, rows.n, now_s() - t0);
    if (failed.n) {
        sb_t s = {0};
        for (size_t i = 0; i < failed.n; i++) { if (i) sb_ch(&s, ' '); sb_puts(&s, failed.v[i]); }
        logln("[selftest] 未通过: %s", s.p ? s.p : "");
        logln("[selftest] ★harness 有问题★ — 标准答案本应全过, 先修环境/判定器再跑模型");
        sb_free(&s);
    }
    sl_free(&failed);
    return n_pass == (int)rows.n;
}

/* ============================ main / argparse ============================ */

static const char *USAGE =
    "usage: pubbench [-h] [--suite {humaneval,humaneval-x-go}] [--url URL] [--tag TAG]\n"
    "                [--limit LIMIT] [--offset OFFSET] [--max-tokens MAX_TOKENS]\n"
    "                [--mode MODE] [--api {chat,completions}]\n"
    "                [--http-timeout HTTP_TIMEOUT] [--jobs JOBS]\n"
    "                [--eval-timeout EVAL_TIMEOUT] [--out-dir OUT_DIR]\n"
    "                [--compare A.jsonl B.jsonl] [--rejudge FILE.jsonl] [--selftest]\n";

static void ap_err(const char *fmt, ...) {
    char b[512];
    va_list ap; va_start(ap, fmt); vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
    fputs(USAGE, stderr);
    fprintf(stderr, "pubbench: error: %s\n", b);
    exit(2);
}
static int ap_int(const char *opt, const char *s) {
    char *end;
    long v = strtol(s, &end, 10);
    if (end == s || *end) ap_err("argument %s: invalid int value: '%s'", opt, s);
    return (int)v;
}
static int env_int(const char *name, const char *def) {
    const char *s = getenv(name);
    if (!s) s = def;
    char *end;
    long v = strtol(s, &end, 10);
    if (end == s || *end) die("ValueError: invalid literal for int() with base 10: '%s'", s);
    return (int)v;
}
static const char *env_str(const char *name, const char *def) { const char *s = getenv(name); return s ? s : def; }

/* .py 的 out-dir 默认 = os.path.dirname(os.path.abspath(__file__)) + "/../reports/pubbench"
 * (未规范化, 打印时带 "..")。C 版同形, 但 dirname 是可执行文件所在目录(calib/ 而非
 * scripts/) —— 落地目录同一个, 打印出来的路径少一处目录名差异, 见转录清单 B 组。*/
static const char *default_out_dir(void) {
    static char buf[4096], dir[4096];
    buf[0] = 0;
#ifdef __APPLE__
    uint32_t sz = sizeof buf;
    if (_NSGetExecutablePath(buf, &sz) != 0) buf[0] = 0;
#else
    ssize_t k = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (k > 0) buf[k] = 0; else buf[0] = 0;
#endif
    if (!buf[0]) { snprintf(dir, sizeof dir, "../reports/pubbench"); return dir; }
    char *slash = strrchr(buf, '/');
    if (slash) *slash = 0;
    snprintf(dir, sizeof dir, "%s/../reports/pubbench", buf);
    return dir;
}

int main(int argc, char **argv) {
    setvbuf(stderr, NULL, _IONBF, 0);
    signal(SIGPIPE, SIG_IGN);
    curl_global_init(CURL_GLOBAL_DEFAULT);
    CACHE = env_str("PUBBENCH_CACHE", "/tmp/pubbench_cache");

    args_t a;
    a.suite = "humaneval";
    a.url = env_str("DS4_URL", "http://127.0.0.1:8080");
    a.tag = "run";                                      /* 例: base_q2 / vq14 */
    a.limit = 20;
    a.offset = 0;
    a.max_tokens = 320;
    a.mode = "code";                                    /* ds4 server mode:code */
    a.api = env_str("PUBBENCH_API", "chat");            /* completions=BASE 裸续写口径 */
    a.http_timeout = 900;                               /* 慢机 2-5 t/s 留足 */
    a.jobs = env_int("PUBBENCH_JOBS", "4");             /* 并发(生成+判题流水) */
    a.eval_timeout = 15;
    a.out_dir = env_str("PUBBENCH_OUT", default_out_dir());
    a.cmp_a = a.cmp_b = a.rejudge_path = NULL;
    a.selftest = 0;

    for (int i = 1; i < argc; i++) {
        char opt[64], *eq;
        const char *arg = argv[i];
        snprintf(opt, sizeof opt, "%s", arg);
        const char *inline_val = NULL;
        if ((eq = strchr(opt, '=')) && !strncmp(opt, "--", 2)) { *eq = 0; inline_val = arg + (eq - opt) + 1; }
        #define NEXTV(name) (inline_val ? inline_val : (i + 1 < argc ? argv[++i] : (ap_err("argument %s: expected one argument", name), (char *)NULL)))
        if (!strcmp(opt, "-h") || !strcmp(opt, "--help")) { fputs(USAGE, stdout); return 0; }
        else if (!strcmp(opt, "--suite")) {
            a.suite = NEXTV("--suite");
            if (strcmp(a.suite, "humaneval") && strcmp(a.suite, "humaneval-x-go"))
                ap_err("argument --suite: invalid choice: '%s' (choose from 'humaneval', 'humaneval-x-go')", a.suite);
        }
        else if (!strcmp(opt, "--url")) a.url = NEXTV("--url");
        else if (!strcmp(opt, "--tag")) a.tag = NEXTV("--tag");
        else if (!strcmp(opt, "--limit")) a.limit = ap_int("--limit", NEXTV("--limit"));
        else if (!strcmp(opt, "--offset")) a.offset = ap_int("--offset", NEXTV("--offset"));
        else if (!strcmp(opt, "--max-tokens")) a.max_tokens = ap_int("--max-tokens", NEXTV("--max-tokens"));
        else if (!strcmp(opt, "--mode")) a.mode = NEXTV("--mode");
        else if (!strcmp(opt, "--api")) {
            a.api = NEXTV("--api");
            if (strcmp(a.api, "chat") && strcmp(a.api, "completions"))
                ap_err("argument --api: invalid choice: '%s' (choose from 'chat', 'completions')", a.api);
        }
        else if (!strcmp(opt, "--http-timeout")) a.http_timeout = ap_int("--http-timeout", NEXTV("--http-timeout"));
        else if (!strcmp(opt, "--jobs")) a.jobs = ap_int("--jobs", NEXTV("--jobs"));
        else if (!strcmp(opt, "--eval-timeout")) a.eval_timeout = ap_int("--eval-timeout", NEXTV("--eval-timeout"));
        else if (!strcmp(opt, "--out-dir")) a.out_dir = NEXTV("--out-dir");
        else if (!strcmp(opt, "--compare")) {
            if (inline_val) ap_err("argument --compare: expected 2 arguments");
            if (i + 2 >= argc) ap_err("argument --compare: expected 2 arguments");
            a.cmp_a = argv[++i]; a.cmp_b = argv[++i];
        }
        else if (!strcmp(opt, "--rejudge")) a.rejudge_path = NEXTV("--rejudge");
        else if (!strcmp(opt, "--selftest")) a.selftest = 1;
        else ap_err("unrecognized arguments: %s", arg);
        #undef NEXTV
    }

    if (a.selftest) return selftest(a.suite, a.limit, a.offset, a.eval_timeout) ? 0 : 1;
    if (a.rejudge_path) { rejudge(a.rejudge_path, a.api); return 0; }
    if (a.cmp_a) compare(a.cmp_a, a.cmp_b);
    else run_suite(&a);
    return 0;
}
