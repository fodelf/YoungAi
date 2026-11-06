/* studio_rl_scan.c — 直播页(/rl)的数据: 只读盘, 不碰模型、不起子进程。总述见 studio_rl.c。
 * 读哪些(都是自学习循环 gguf-tools/scripts/vstep_loop_spark.sh 自己写的, 这里不另存一份):
 *   gguf/go-onebit/evo/vloop/ledger.tsv     一轮一行: 判(接力/起点/丢/留出升/无ckpt/料失败)+ 四把尺 + 时间; 第 0 行 = 循环起点(基线)
 *   gguf/go-onebit/evo/vloop/loop.log       脚本的 M() 行; 只取 "== 轮 N" 的时间当停模型窗口的起点(账本只有结束时间)
 *   gguf/go-onebit/evo/vloop/rNNN/          本轮的料: rows.log / raw.log 的题数, 各子目录 raw.jsonl 里官方 API 返回的 usage, *.log 里的 API_USAGE 行(出题)
 *   gguf/v41/posttrain/pt-<标签>/train.log  训练器逐步落的: step(损失/梯度范数/秒)、step 0 与 epoch 行的留出、decision 行的决策位
 * 花费 = 官方 API 用量 × 官方价(下面 PRICE_*), 高峰/空闲按文件最后写入的北京时间分(一轮的调用在一两分钟里打完); 法定节假日不认(按工作日算, 只会多算)。
 *   出题调用(api_round.py)11-04 之前没记用量, 那部分不在账里; 本机算力(训练、FP 锚、判决尺)不是钱, 页面报的是机时。
 * 出错会怎样: 哪个文件没有就少那几个数(JSON 里是 null), 不报错 —— 页面是给人看的, 一轮料失败不该让整页打不开。 */
#include "studio_internal.h"
#include <dirent.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>

/* 官方价(元 / 百万 token, api-docs.deepseek.com/zh-cn/quick_start/pricing, 2026-11-04 抄): deepseek-flash 高峰 命中 0.04 / 未命中 2 / 输出 8, 空闲减半。
 * 高峰 = 北京时间周一至周五 9:00-12:00、14:00-18:00。官方调价就改这里。 */
#define PRICE_HIT 0.04
#define PRICE_MISS 2.0
#define PRICE_OUT 8.0
#define BJ_OFFSET (8 * 3600)

static bool peak_bj(long t) {
    const time_t b = (time_t)(t + BJ_OFFSET); struct tm tm; gmtime_r(&b, &tm);
    return tm.tm_wday >= 1 && tm.tm_wday <= 5 && ((tm.tm_hour >= 9 && tm.tm_hour < 12) || (tm.tm_hour >= 14 && tm.tm_hour < 18));
}

/* "11-04 15:53" / "11-04 15:53:13"(北京时间, 不带年)→ 秒。年取现在的; 算出来比现在晚一天以上就是去年(跨年那几天) */
static long bj_time(const char *s) {
    int mo, d, h, mi, se = 0;
    if (sscanf(s, "%d-%d %d:%d:%d", &mo, &d, &h, &mi, &se) < 4) return 0;
    const time_t now = time(NULL) + BJ_OFFSET; struct tm tm; gmtime_r(&now, &tm);
    for (int back = 0; back < 2; back++) {
        struct tm t = { 0 }; t.tm_year = tm.tm_year - back; t.tm_mon = mo - 1; t.tm_mday = d; t.tm_hour = h; t.tm_min = mi; t.tm_sec = se;
        const long v = (long)timegm(&t) - BJ_OFFSET;
        if (v <= (long)time(NULL) + 86400) return v;
    }
    return 0;
}

static long mtime_of(const char *p) { struct stat st; return stat(p, &st) ? 0 : (long)st.st_mtime; }

/* ---- 官方 API 用量: 每个文件扫一遍记下来, 大小与修改时间不变就不再扫(rows/raw.jsonl 带 top-20 logprob, 32 轮合计 600+ MB) ---- */
typedef struct { char path[TR_PATH]; long size, mt; double hit, miss, out; int calls; } use_ent;
#define USE_MAX 4096
static use_ent *g_use; static int g_nuse;
static pthread_mutex_t g_use_mu = PTHREAD_MUTEX_INITIALIZER;

static double num_after(const char *p, const char *end, const char *key) {
    const char *k = strstr(p, key);
    if (!k || k > end) return -1;
    return atof(k + strlen(key));
}
/* 一行里的 usage 对象(json.dumps 的 "usage": {...}; 前面是反斜杠 = 字符串里的字面文本, 不算) */
static bool line_usage(const char *ln, double *hit, double *miss, double *out) {
    for (const char *p = ln; (p = strstr(p, "\"usage\": {")); p++) {
        if (p > ln && p[-1] == '\\') continue;
        const char *end = p + 600, *z = strchr(p, '}'); if (z && z < end) end = z;
        const double h = num_after(p, end, "\"prompt_cache_hit_tokens\": "), m = num_after(p, end, "\"prompt_cache_miss_tokens\": ");
        const double o = num_after(p, end, "\"completion_tokens\": "), pt = num_after(p, end, "\"prompt_tokens\": ");
        if (o < 0) return false;
        *hit += h > 0 ? h : 0; *miss += m >= 0 ? m : (pt > 0 ? pt : 0); *out += o;
        return true;
    }
    return false;
}
static void file_usage(const char *path, double *hit, double *miss, double *out, int *calls, double *cny) {
    struct stat st; if (stat(path, &st)) return;
    use_ent *e = NULL;
    pthread_mutex_lock(&g_use_mu);
    if (!g_use) g_use = calloc(USE_MAX, sizeof *g_use);
    for (int i = 0; g_use && i < g_nuse; i++) if (!strcmp(g_use[i].path, path)) { e = &g_use[i]; break; }
    if (g_use && (!e || e->size != (long)st.st_size || e->mt != (long)st.st_mtime)) {
        if (!e && g_nuse < USE_MAX) { e = &g_use[g_nuse++]; snprintf(e->path, sizeof e->path, "%s", path); }
        if (e) {
            e->hit = e->miss = e->out = 0; e->calls = 0; e->size = (long)st.st_size; e->mt = (long)st.st_mtime;
            FILE *f = fopen(path, "rb"); char *ln = NULL; size_t cap = 0;
            /* .log 里只认 API_USAGE 行(api_round.py 每次调用打一行); jsonl 每行是一次调用 */
            const bool log = strlen(path) > 4 && !strcmp(path + strlen(path) - 4, ".log");
            while (f && getline(&ln, &cap, f) > 0) {
                if (log && strncmp(ln, "API_USAGE ", 10)) continue;
                if (line_usage(ln, &e->hit, &e->miss, &e->out)) e->calls++;
            }
            free(ln); if (f) fclose(f);
        }
    }
    if (e) {
        const double k = peak_bj(e->mt) ? 1.0 : 0.5;
        *hit += e->hit; *miss += e->miss; *out += e->out; *calls += e->calls;
        *cny += k * (e->hit * PRICE_HIT + e->miss * PRICE_MISS + e->out * PRICE_OUT) / 1e6;
    }
    pthread_mutex_unlock(&g_use_mu);
}
/* 一轮目录: 各子目录的 raw.jsonl / api_raw.jsonl(作答、裸续写、思考轨迹、agent 头两步) + 本目录 *.log 的 API_USAGE 行 */
static void round_cost(const char *rdir, rl_round *r) {
    DIR *d = opendir(rdir); struct dirent *de;
    while (d && (de = readdir(d))) {
        if (de->d_name[0] == '.') continue;
        char p[TR_PATH * 2]; snprintf(p, sizeof p, "%s/%s", rdir, de->d_name);
        struct stat st; if (stat(p, &st)) continue;
        if (S_ISDIR(st.st_mode)) {
            static const char *const F[] = { "raw.jsonl", "api_raw.jsonl" };
            for (int i = 0; i < 2; i++) { char q[TR_PATH * 2 + 32]; snprintf(q, sizeof q, "%s/%s", p, F[i]); file_usage(q, &r->hit, &r->miss, &r->out, &r->calls, &r->cny); }
        } else if (strlen(de->d_name) > 4 && !strcmp(de->d_name + strlen(de->d_name) - 4, ".log")) file_usage(p, &r->hit, &r->miss, &r->out, &r->calls, &r->cny);
    }
    if (d) closedir(d);
}

/* ---- 账本 ---- */
static void clear_round(rl_round *r) {
    memset(r, 0, sizeof *r);
    r->kld = r->same = r->smin = r->wkld = r->wsame = r->think = r->dec = RL_NONE;
    r->secs = r->eval0 = r->eval1 = r->gain = r->se = r->gen0 = r->gen1 = r->dec0 = r->dec1 = r->rows = RL_NONE;
}
static double cell_num(const char *c) { return (c && *c && *c != '-' && *c != '?') ? atof(c) : RL_NONE; }
/* ckpt / 起点路径 → 标签: .../pt-vl029/ckpt_e01 → vl029; .../vstep-v17/ckpt_v04 → v17 */
static void path_tag(const char *p, char *out, size_t n) {
    out[0] = 0;
    const char *s = strstr(p, "posttrain/"); if (!s) return;
    s += 10; if (!strncmp(s, "pt-", 3)) s += 3; else if (!strncmp(s, "vstep-", 6)) s += 6;
    snprintf(out, n, "%.*s", (int)strcspn(s, "/"), s);
}
static const char *verdict_key(const char *v) {
    static const char *const M[][2] = { { "接力", "relay" }, { "起点", "chain" }, { "丢", "drop" }, { "留出升", "holdout" }, { "无ckpt", "nockpt" },
                                        { "料失败", "datafail" }, { "prep失败", "prepfail" }, { "当前最好", "base" } };
    for (size_t i = 0; i < sizeof M / sizeof M[0]; i++) if (!strcmp(v, M[i][0])) return M[i][1];
    return "other";
}
static rl_round *round_slot(rl_state *s, int n) {
    for (int i = 0; i < s->nr; i++) if (s->r[i].n == n) return &s->r[i];
    if (s->nr >= RL_MAX_ROUNDS) return NULL;
    rl_round *r = &s->r[s->nr++]; clear_round(r); r->n = n; snprintf(r->tag, sizeof r->tag, "vl%03d", n);
    return r;
}
static void read_ledger(rl_state *s, const char *vl) {
    char p[TR_PATH + 64]; snprintf(p, sizeof p, "%.960s/ledger.tsv", vl);
    FILE *f = fopen(p, "rb"); char *ln = NULL; size_t cap = 0;
    while (f && getline(&ln, &cap, f) > 0) {
        ln[strcspn(ln, "\r\n")] = 0;
        char *c[20] = { 0 }; int nc = 0;
        for (char *q = ln; nc < 20; ) { c[nc++] = q; char *t = strchr(q, '\t'); if (!t) break; *t = 0; q = t + 1; }
        if (nc < 6 || c[0][0] < '0' || c[0][0] > '9') continue;   /* 表头 */
        const int n = atoi(c[0]);
        rl_round *r = n == 0 ? &s->base : round_slot(s, n);
        if (!r) continue;
        if (n == 0) { clear_round(r); s->has_base = true; }
        else { const int keep = r->n; clear_round(r); r->n = keep; }   /* 同一轮重跑过: 后一行为准 */
        snprintf(r->tag, sizeof r->tag, "%s", c[1]); snprintf(r->kind, sizeof r->kind, "%s", c[3]);
        snprintf(r->verdict, sizeof r->verdict, "%s", verdict_key(c[5]));
        const char *ck = nc > 4 ? c[4] : "", *e = strstr(ck, "ckpt_e"); if (e) r->ep = atoi(e + 6);
        if (n == 0) path_tag(ck, r->tag, sizeof r->tag);   /* 第 0 行的标签是写时的 vl000/v17, 按 ckpt 认 */
        if (nc > 6) path_tag(c[6], r->from, sizeof r->from);
        if (nc > 8) { r->kld = cell_num(c[7]); r->same = cell_num(c[8]); }
        if (nc > 11) { r->smin = cell_num(c[9]); r->wkld = cell_num(c[10]); r->wsame = cell_num(c[11]); }
        if (nc > 13) r->end = bj_time(c[13]);
        if (nc > 15) { r->think = cell_num(c[14]); r->dec = cell_num(c[15]); }
    }
    free(ln); if (f) fclose(f);
}

/* loop.log: "[vloop 11-04 16:01:43] == 轮 32(vl032, …" → 这一轮停模型窗口的起点。同一轮起过几次(重跑)取最后一次 */
static void read_loop_log(rl_state *s, const char *vl) {
    char p[TR_PATH + 64]; snprintf(p, sizeof p, "%.960s/loop.log", vl);
    FILE *f = fopen(p, "rb"); char *ln = NULL; size_t cap = 0;
    while (f && getline(&ln, &cap, f) > 0) {
        if (strncmp(ln, "[vloop ", 7)) continue;
        const long t = bj_time(ln + 7);
        if (t && (!s->camp_start || t < s->camp_start)) s->camp_start = t;
        const char *m = strstr(ln, "] == 轮 ");
        if (!m) continue;
        rl_round *r = round_slot(s, atoi(m + strlen("] == 轮 ")));
        if (r && t > r->start) r->start = t;
    }
    free(ln); if (f) fclose(f);
}

/* train.log(core_ptrain.c 的 lf): 同一标签重跑会追加, 遇到新的 "step 0 " 就从头记 */
static void read_train_log(rl_state *s, rl_round *r, rl_step **st, int *ns, int *cap) {
    char p[TR_PATH + 64]; snprintf(p, sizeof p, "%s/pt-%s/train.log", tr_ftd, r->tag);
    long mt = mtime_of(p);
    if (!mt) { snprintf(p, sizeof p, "%s/vstep-%s/train.log", tr_ftd, r->tag); mt = mtime_of(p); }
    FILE *f = mt ? fopen(p, "rb") : NULL;
    if (!f) return;
    const int first = *ns; char ln[512]; double tlast = 0; int ep_seen = 0;
    while (fgets(ln, sizeof ln, f)) {
        unsigned step, ep; float loss, gn, t; double a, b, c, g, se, rows, gk; unsigned nq;
        if (sscanf(ln, "step 0 eval_kl %lf train_kl %lf hold_kl %lf gen_kl %lf", &a, &b, &c, &gk) == 4) {
            *ns = first; r->steps = 0; ep_seen = 0; tlast = 0;
            r->eval0 = a; r->gen0 = gk; r->eval1 = r->gen1 = r->gain = r->se = r->rows = r->dec1 = RL_NONE; r->nq = 0;
        } else if (sscanf(ln, "step %u ep %u loss %f gnorm %f t %f", &step, &ep, &loss, &gn, &t) == 5) {
            if (*ns >= *cap) { *cap = *cap ? *cap * 2 : 4096; rl_step *nst = realloc(*st, (size_t)*cap * sizeof **st); if (!nst) break; *st = nst; }
            (*st)[(*ns)++] = (rl_step){ r->n, (int)step, loss, gn, (long)t };   /* t 先存相对秒, 读完再换墙上时间 */
            r->steps = (int)step; tlast = t;
        } else if (sscanf(ln, "epoch %u eval_kl %lf train_kl %lf hold_kl %lf gain %lf se %lf nq %u rows %lf gen_kl %lf", &ep, &a, &b, &c, &g, &se, &nq, &rows, &gk) == 9) {
            ep_seen = (int)ep;
            if (!r->ep || (int)ep <= r->ep) { r->eval1 = a; r->gain = g; r->se = se; r->gen1 = gk; }   /* 选中的那轮(ckpt_eNN)的读数; 没选中信息就取最后一轮 */
            r->nq += (int)nq; r->rows = (r->rows > RL_NONE / 2 ? r->rows : 0) + rows;
        } else if (sscanf(ln, "decision e%u kl %lf", &ep, &a) == 2) {
            if (ep == 0) r->dec0 = a; else if (!r->ep || (int)ep <= r->ep) r->dec1 = a;
        }
    }
    fclose(f);
    (void)ep_seen;
    r->secs = tlast > 0 ? tlast : RL_NONE; r->train_end = mt;
    for (int i = first; i < *ns; i++) (*st)[i].t = mt - (long)(tlast - (double)(*st)[i].t);   /* 最后一步 ≈ 文件最后写入; 往前按相对秒推 */
    (void)s;
}

/* 料的题数: rows.log "短行: 题 60 / ok 47" ; raw.log "→ 11 篇" */
static void read_data_counts(rl_round *r, const char *rdir) {
    char p[TR_PATH * 2], *t; size_t n;
    snprintf(p, sizeof p, "%s/rows.log", rdir);
    if ((t = tr_slurp(p, &n))) { const char *m = strstr(t, "短行: 题 "); if (m) sscanf(m + strlen("短行: 题 "), "%d / ok %d", &r->q, &r->ok); free(t); }
    snprintf(p, sizeof p, "%s/raw.log", rdir);
    if ((t = tr_slurp(p, &n))) { const char *m = strstr(t, "裸续写流: "); m = m ? strstr(m, "→ ") : NULL; if (m) r->docs = atoi(m + strlen("→ ")); free(t); }
}

static int cmp_round(const void *a, const void *b) { return ((const rl_round *)a)->n - ((const rl_round *)b)->n; }

/* 正在跑的那一轮在哪一步: 料 → (FP 锚) → 装载与第 0 步评估 → 训练 → 判决(轮末评估 + 四把尺)。只看文件, 不看进程内部 */
static void live_phase(rl_state *s, const char *vl) {
    rl_round *r = NULL;
    for (int i = 0; i < s->nr; i++) if (!s->r[i].verdict[0] && (!r || s->r[i].n < r->n)) r = &s->r[i];
    if (!r) return;
    char rd[TR_PATH + 64], p[TR_PATH * 2]; snprintf(rd, sizeof rd, "%.960s/r%03d", vl, r->n);
    s->live_n = r->n; snprintf(s->live_tag, sizeof s->live_tag, "%s", r->tag);
    const char *ph = "data";
    snprintf(p, sizeof p, "%s/GEN_DONE", rd);
    if (mtime_of(p) || (r->train_end && r->train_end >= r->start)) {   /* 复验复用旧料, 没有本轮 GEN_DONE, 但已有训练日志。 */
        ph = "load";
        snprintf(p, sizeof p, "%s/fp/anchor.log", rd); const long an = mtime_of(p);
        snprintf(p, sizeof p, "%s/fp/teacher.ptf", rd);
        if (an && !mtime_of(p) && !r->train_end) ph = "anchor";
        if (r->train_end && (!r->start || r->train_end >= r->start)) {
            ph = r->steps ? (r->eval1 > RL_NONE / 2 ? "judge" : "train") : "eval0";
            for (int i = s->ns - 1; i >= 0; i--) if (s->st[i].n == r->n) { s->live_step = s->st[i].step; s->live_loss = s->st[i].loss; break; }
            s->live_t = r->secs > RL_NONE / 2 ? r->secs : 0;
        }
    }
    snprintf(s->live_phase, sizeof s->live_phase, "%s", ph);
    s->live_since = r->start;
}

void rl_scan(rl_state *s) {
    rl_step *keep = s->st; memset(s, 0, sizeof *s); s->st = keep;
    clear_round(&s->base);
    char vl[TR_PATH + 64]; snprintf(vl, sizeof vl, "%.900s/gguf/go-onebit/evo/vloop", tr_root);
    read_ledger(s, vl);
    /* 盘上已有料目录但账本还没这一轮的(正在跑的、料在并行出的)也列出来: 花费要算进去 */
    DIR *d = opendir(vl); struct dirent *de;
    while (d && (de = readdir(d))) if (de->d_name[0] == 'r' && de->d_name[1] >= '0' && de->d_name[1] <= '9') round_slot(s, atoi(de->d_name + 1));
    if (d) closedir(d);
    read_loop_log(s, vl);
    qsort(s->r, (size_t)s->nr, sizeof s->r[0], cmp_round);
    int cap = 0; s->ns = 0; free(s->st); s->st = NULL;
    for (int i = 0; i < s->nr; i++) {
        rl_round *r = &s->r[i]; char rd[TR_PATH + 64]; snprintf(rd, sizeof rd, "%.960s/r%03d", vl, r->n);
        read_train_log(s, r, &s->st, &s->ns, &cap);
        read_data_counts(r, rd); round_cost(rd, r);
        if (r->n == 1) { char p[TR_PATH * 2]; snprintf(p, sizeof p, "%s/q.log", rd); const long t = mtime_of(p); if (t && (!s->camp_start || t < s->camp_start)) s->camp_start = t; }
    }
    tr_procs p; tr_proc_scan(&p);   /* 单轮复验直接起训练器, 判决时才进入 vstep_spark.sh。 */
    s->running = p.training || tr_proc_kill("vstep_loop_spark.sh", 0) > 0 || tr_proc_kill("vstep_spark.sh", 0) > 0;
    if (s->running) live_phase(s, vl);
}
