/* train_runs.c — ds4-train: 盘上训练产物 → JSON。总述见 train_internal.h。
 * 只读, 不解释: train.log 的 step/epoch 行照抄成数组, 门文件(v41_judge 两臂全文)按 kdpick 的 gate_read 同一口径抽四个数并判过不过,
 * pick.txt 是 kd_pick 写的选轮结果。train.log 是追加写的: 同目录重训时以最后一个 "step 0" 为界只认这一趟(与 kdpick 同规矩)。 */
#include "train_internal.h"
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

char *tr_slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char *b = malloc((size_t)n + 1);
    if (!b) { fclose(f); return NULL; }
    const size_t got = fread(b, 1, (size_t)n, f);
    fclose(f);
    b[got] = 0;
    if (len) *len = got;
    return b;
}

long tr_mem_avail_mb(void) {
#ifdef __APPLE__
    /* Mac 没有 /proc: 按 free + inactive 页算(与 Activity Monitor 的"可用"口径接近); 看门狗/起跑清单在 Mac 上照样有数 */
    vm_size_t pg = 0; vm_statistics64_data_t vs; mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_page_size(mach_host_self(), &pg) != KERN_SUCCESS || host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vs, &cnt) != KERN_SUCCESS) return -1;
    return (long)(((unsigned long long)vs.free_count + vs.inactive_count) * pg / (1024 * 1024));
#endif
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char ln[256]; long kb = -1;
    while (fgets(ln, sizeof ln, f)) if (sscanf(ln, "MemAvailable: %ld kB", &kb) == 1) break;
    fclose(f);
    return kb >= 0 ? kb / 1024 : -1;
}

static time_t mtime_of(const char *path) { struct stat st; return stat(path, &st) == 0 ? st.st_mtime : 0; }
static bool exists(const char *path) { struct stat st; return stat(path, &st) == 0; }

bool tr_run_active(const char *dir) {
    char p[TR_PATH + 32]; snprintf(p, sizeof p, "%s/train.out", dir);
    const time_t m = mtime_of(p);
    return m && time(NULL) - m < 120;
}

/* ptrain.cfg 取键(最后一次出现的生效, 与训练器同) */
static bool cfg_get(const char *dir, const char *key, char *out, size_t n) {
    char p[TR_PATH + 32]; snprintf(p, sizeof p, "%s/ptrain.cfg", dir);
    char *b = tr_slurp(p, NULL);
    if (!b) return false;
    const size_t kl = strlen(key);
    bool got = false;
    for (char *ln = strtok(b, "\n"); ln; ln = strtok(NULL, "\n")) {
        if (!strncmp(ln, key, kl) && ln[kl] == '=') { snprintf(out, n, "%s", ln + kl + 1); got = true; }
    }
    free(b);
    if (!got) out[0] = 0;
    return got;
}

/* 门文件两臂: 先 ② 后 ②+③, 每臂各一行 "分布还原率 Σmin = X" / "Mean KLD = Y (中位 Z" / "Same top token = W%" / "PPL(student) = P" */
static void gate_json(ds4_buf *b, const char *dir, unsigned ep) {
    char p[TR_PATH + 64]; snprintf(p, sizeof p, "%s/eval/gate_ckpt_e%02u.txt", dir, ep);
    char *t = tr_slurp(p, NULL);
    if (!t) { ds4_buf_puts(b, "null"); return; }
    double smin[2] = {0, 0}, kld[2] = {0, 0}, med[2] = {0, 0}, top[2] = {0, 0}, ppl[2] = {0, 0};
    int ns = 0, nk = 0, nt = 0, np = 0; char *sv = NULL;
    for (char *ln = strtok_r(t, "\n", &sv); ln; ln = strtok_r(NULL, "\n", &sv)) {   /* strtok_r: 调用方(log_json)自己也在 strtok 一份缓冲 */
        char *s;
        if ((s = strstr(ln, "Σmin =")) && ns < 2) ns += sscanf(s, "Σmin = %lf", &smin[ns]) == 1;
        else if ((s = strstr(ln, "Mean KLD")) && nk < 2) { if (sscanf(s, "Mean KLD = %lf (中位 %lf", &kld[nk], &med[nk]) >= 1) nk++; }
        else if ((s = strstr(ln, "Same top token")) && nt < 2) nt += sscanf(s, "Same top token = %lf", &top[nt]) == 1;
        else if ((s = strstr(ln, "PPL(student)")) && np < 2) np += sscanf(s, "PPL(student) = %lf", &ppl[np]) == 1;
    }
    free(t);
    if (ns < 2 || nk < 2) { ds4_buf_puts(b, "null"); return; }
    const double d = 100.0 * (smin[1] - smin[0]), r = kld[1] / kld[0] - 1.0;
    ds4_buf_printf(b, "{\"smin\":[%.4f,%.4f],\"dpp\":%.2f,\"kld\":[%.5f,%.5f],\"dkld\":%.1f,\"kldmed\":[%.5f,%.5f],\"top\":[%.2f,%.2f],\"ppl\":[%.4f,%.4f],\"pass\":%s}",
                   smin[0], smin[1], d, kld[0], kld[1], 100.0 * r, med[0], med[1], top[0], top[1], ppl[0], ppl[1], (d >= -0.5 && r <= 0.03) ? "true" : "false");
}

/* train.log → step0 / steps(可选) / epochs / steps_done / last。只认最后一个 "step 0" 之后的行。 */
static void log_json(ds4_buf *b, const char *dir, bool with_steps) {
    char p[TR_PATH + 32]; snprintf(p, sizeof p, "%s/train.log", dir);
    char *t = tr_slurp(p, NULL);
    char *start = t;
    if (t) for (char *q = t; (q = strstr(q, "step 0 ")); q += 7) if (q == t || q[-1] == '\n') start = q;
    ds4_buf_puts(b, "\"step0\":");
    double e0 = 0, t0 = 0, h0 = 0; bool got0 = false;
    unsigned steps_done = 0, last_ep = 0; double last_loss = 0, last_g = 0, last_t = 0; long last_mem = 0;
    ds4_buf steps = {0}, epochs = {0};
    ds4_buf_putc(&steps, '['); ds4_buf_putc(&epochs, '[');
    char *sv = NULL;
    if (t) for (char *ln = strtok_r(start, "\n", &sv); ln; ln = strtok_r(NULL, "\n", &sv)) {
        unsigned st, ep, nq; double loss, g, tt, ev, tr, hd, gain, se; long mem; unsigned long long rows;
        if (sscanf(ln, "step 0 eval_kl %lf train_kl %lf hold_kl %lf", &e0, &t0, &h0) == 3) got0 = true;
        else if (sscanf(ln, "step %u ep %u loss %lf gnorm %lf t %lf mem %ld", &st, &ep, &loss, &g, &tt, &mem) == 6) {
            steps_done = st; last_ep = ep; last_loss = loss; last_g = g; last_t = tt; last_mem = mem;
            if (with_steps) ds4_buf_printf(&steps, "%s[%u,%u,%.5f,%.4g,%.0f,%ld]", steps.len > 1 ? "," : "", st, ep, loss, g, tt, mem);
        } else if (sscanf(ln, "epoch %u eval_kl %lf train_kl %lf hold_kl %lf gain %lf se %lf nq %u rows %llu", &ep, &ev, &tr, &hd, &gain, &se, &nq, &rows) >= 4) {
            ds4_buf_printf(&epochs, "%s{\"n\":%u,\"eval\":%.5f,\"train\":%.5f,\"hold\":%.5f,\"gain\":%.5f,\"se\":%.5f,\"gate\":", epochs.len > 1 ? "," : "", ep, ev, tr, hd, gain, se);
            gate_json(&epochs, dir, ep);
            ds4_buf_putc(&epochs, '}');
        }
    }
    ds4_buf_putc(&steps, ']'); ds4_buf_putc(&epochs, ']');
    if (got0) ds4_buf_printf(b, "{\"eval\":%.5f,\"train\":%.5f,\"hold\":%.5f}", e0, t0, h0); else ds4_buf_puts(b, "null");
    ds4_buf_printf(b, ",\"steps_done\":%u,\"last\":{\"ep\":%u,\"loss\":%.5f,\"gnorm\":%.4g,\"t\":%.0f,\"mem\":%ld},\"epochs\":%s", steps_done, last_ep, last_loss, last_g, last_t, last_mem, epochs.ptr ? epochs.ptr : "[]");
    if (with_steps) ds4_buf_printf(b, ",\"steps\":%s", steps.ptr ? steps.ptr : "[]");
    ds4_buf_free(&steps); ds4_buf_free(&epochs); free(t);
}

/* train.out 里的 "训练题 N"(估总步数用) + 末尾 tail 行(\r 当换行) */
static void out_json(ds4_buf *b, const char *dir, unsigned tail_lines) {
    char p[TR_PATH + 32]; snprintf(p, sizeof p, "%s/train.out", dir);
    size_t n = 0; char *t = tr_slurp(p, &n);
    unsigned ntr = 0;
    if (t) {
        for (size_t i = 0; i < n; i++) if (t[i] == '\r') t[i] = '\n';
        const char *s = strstr(t, "训练题 ");
        if (s) sscanf(s, "训练题 %u", &ntr);
    }
    ds4_buf_printf(b, "\"ntr\":%u,\"tail\":", ntr);
    if (!t || !tail_lines) { ds4_buf_puts(b, "\"\""); free(t); return; }
    size_t pos = n; unsigned lines = 0;
    while (pos > 0 && lines <= tail_lines) { pos--; if (t[pos] == '\n') lines++; }
    if (lines > tail_lines) pos++;
    ds4_json_escape(b, t + pos);
    free(t);
}

/* 状态: running(近两分钟还在写) / done(选了轮) / nopick(③ 落了但没过门或还没选) / failed(日志里有 失败) / stopped */
static const char *run_status(const char *dir) {
    char p[TR_PATH + 32];
    if (tr_job_owns(dir) || tr_run_active(dir)) return "running";   /* 作业线程正做这趟就是 running(门阶段 train.out 不动, 只看文件时间会误报"没选轮"); 老目录重训时上一次的 pick.txt 还在 */
    snprintf(p, sizeof p, "%s/pick.txt", dir);
    if (exists(p)) return "done";
    snprintf(p, sizeof p, "%s/amp_L39.bin", dir);
    if (exists(p)) return "nopick";
    snprintf(p, sizeof p, "%s/train.out", dir);
    char *t = tr_slurp(p, NULL);
    const bool failed = t && (strstr(t, "失败") || strstr(t, "停车"));
    free(t);
    return failed ? "failed" : "stopped";
}

static void run_json(ds4_buf *b, const char *name, bool full) {
    char dir[TR_PATH + 64]; snprintf(dir, sizeof dir, "%s/%s", tr_ftd, name);
    char v[512], p[TR_PATH + 64];
    ds4_buf_puts(b, "{\"name\":"); ds4_json_escape(b, name);
    /* 配置键照抄, epochs 改叫 epochs_cfg: 轮表数组也叫 epochs, 同名会撞 */
    const char *keys[] = { "data", "epochs", "layers", "lr", "batch", "epoch_tok", "init" }, *names[] = { "data", "epochs_cfg", "layers", "lr", "batch", "epoch_tok", "init" };
    for (size_t k = 0; k < sizeof keys / sizeof keys[0]; k++) { cfg_get(dir, keys[k], v, sizeof v); ds4_buf_printf(b, ",\"%s\":", names[k]); ds4_json_escape(b, v); }
    snprintf(p, sizeof p, "%s/ptrain.cfg", dir);
    ds4_buf_printf(b, ",\"started\":%ld,\"status\":\"%s\",\"pick\":", (long)mtime_of(p), run_status(dir));
    snprintf(p, sizeof p, "%s/pick.txt", dir);
    char *pk = tr_slurp(p, NULL);
    if (pk) { char *nl = strchr(pk, '\n'); if (nl) *nl = 0; ds4_json_escape(b, pk); free(pk); } else ds4_buf_puts(b, "null");
    ds4_buf_putc(b, ',');
    log_json(b, dir, full);
    ds4_buf_putc(b, ',');
    out_json(b, dir, full ? 80 : 1);
    if (full) {   /* 探针原文: probe_e00..e99 */
        ds4_buf_puts(b, ",\"probes\":{");
        bool first = true;
        for (unsigned e = 0; e < 100; e++) {
            snprintf(p, sizeof p, "%s/probe_e%02u.txt", dir, e);
            char *t = tr_slurp(p, NULL);
            if (!t) continue;
            ds4_buf_printf(b, "%s\"e%02u\":", first ? "" : ",", e); ds4_json_escape(b, t); free(t); first = false;
        }
        ds4_buf_putc(b, '}');
        snprintf(p, sizeof p, "%s/ptrain.cfg", dir);
        char *cfg = tr_slurp(p, NULL);
        ds4_buf_puts(b, ",\"cfg\":"); ds4_json_escape(b, cfg ? cfg : ""); free(cfg);
    }
    ds4_buf_putc(b, '}');
}

typedef struct { char name[256]; time_t m; } tr_ent;
static int by_mtime_desc(const void *a, const void *b) { const tr_ent *x = a, *y = b; return x->m < y->m ? 1 : x->m > y->m ? -1 : strcmp(x->name, y->name); }

void tr_json_runs(ds4_buf *b) {
    tr_ent *ents = NULL; size_t n = 0;
    DIR *d = opendir(tr_ftd);
    struct dirent *de;
    while (d && (de = readdir(d))) {
        if (strncmp(de->d_name, "kd-", 3)) continue;
        char p[TR_PATH + 320]; snprintf(p, sizeof p, "%s/%s/ptrain.cfg", tr_ftd, de->d_name);
        const time_t m = mtime_of(p);
        if (!m) continue;
        ents = realloc(ents, (n + 1) * sizeof *ents);
        snprintf(ents[n].name, sizeof ents[n].name, "%s", de->d_name); ents[n].m = m; n++;
    }
    if (d) closedir(d);
    if (n) qsort(ents, n, sizeof *ents, by_mtime_desc);
    ds4_buf_putc(b, '[');
    for (size_t i = 0; i < n; i++) { if (i) ds4_buf_putc(b, ','); run_json(b, ents[i].name, false); }
    ds4_buf_putc(b, ']');
    free(ents);
}

bool tr_json_run(ds4_buf *b, const char *name) {
    if (!name[0] || strchr(name, '/') || strstr(name, "..")) return false;
    char p[TR_PATH + 320]; snprintf(p, sizeof p, "%s/%s/ptrain.cfg", tr_ftd, name);
    if (!exists(p)) return false;
    run_json(b, name, true);
    return true;
}

static unsigned count_lines(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    unsigned n = 0; char buf[65536]; size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0) for (size_t i = 0; i < r; i++) n += buf[i] == '\n';
    fclose(f);
    return n;
}
static void jsonl_entry(ds4_buf *b, const char *rel, const char *abs, bool *first) {
    char tb[TR_PATH + 64]; snprintf(tb, sizeof tb, "%s.teacher.bin", abs);
    struct stat st; stat(abs, &st);
    ds4_buf_printf(b, "%s{\"path\":", *first ? "" : ","); ds4_json_escape(b, rel);
    ds4_buf_printf(b, ",\"rows\":%u,\"bytes\":%lld,\"teacher\":%s}", count_lines(abs), (long long)st.st_size, exists(tb) ? "true" : "false");
    *first = false;
}
/* 料清单: $datad 下一层与两层的 *.jsonl(路径相对仓库根, 训练命令直接用), 以及带 chunks/ 的料目录(gen 用) */
void tr_json_data(ds4_buf *b) {
    const char *rel0 = "gguf-tools/data/posttrain";
    ds4_buf_puts(b, "{\"jsonl\":[");
    bool first = true;
    DIR *d = opendir(tr_datad); struct dirent *de;
    ds4_buf dirs = {0}; bool dfirst = true;
    while (d && (de = readdir(d))) {
        if (de->d_name[0] == '.') continue;
        char abs[TR_PATH + 320], rel[TR_PATH + 320];
        snprintf(abs, sizeof abs, "%s/%s", tr_datad, de->d_name); snprintf(rel, sizeof rel, "%s/%s", rel0, de->d_name);
        const size_t L = strlen(de->d_name);
        if (L > 6 && !strcmp(de->d_name + L - 6, ".jsonl")) { jsonl_entry(b, rel, abs, &first); continue; }
        struct stat st;
        if (stat(abs, &st) || !S_ISDIR(st.st_mode)) continue;
        char ch[TR_PATH + 340]; snprintf(ch, sizeof ch, "%s/chunks", abs);
        DIR *cd = opendir(ch); unsigned nch = 0; struct dirent *ce;
        while (cd && (ce = readdir(cd))) { const size_t l = strlen(ce->d_name); nch += l > 4 && !strcmp(ce->d_name + l - 4, ".txt"); }
        if (cd) closedir(cd);
        if (nch) {
            char jl[TR_PATH + 600]; snprintf(jl, sizeof jl, "%s/%s.jsonl", abs, de->d_name);
            ds4_buf_printf(&dirs, "%s{\"path\":", dfirst ? "" : ","); ds4_json_escape(&dirs, rel);
            ds4_buf_printf(&dirs, ",\"chunks\":%u,\"has_jsonl\":%s}", nch, exists(jl) ? "true" : "false"); dfirst = false;
        }
        DIR *sd = opendir(abs); struct dirent *se;   /* 两层: 料目录里 gen 出的 <料名>.jsonl */
        while (sd && (se = readdir(sd))) {
            const size_t l = strlen(se->d_name);
            if (l > 6 && !strcmp(se->d_name + l - 6, ".jsonl")) {
                char a2[TR_PATH + 600], r2[TR_PATH + 600];
                snprintf(a2, sizeof a2, "%s/%s", abs, se->d_name); snprintf(r2, sizeof r2, "%s/%s", rel, se->d_name);
                jsonl_entry(b, r2, a2, &first);
            }
        }
        if (sd) closedir(sd);
    }
    if (d) closedir(d);
    ds4_buf_printf(b, "],\"dirs\":[%s]}", dirs.ptr ? dirs.ptr : "");
    ds4_buf_free(&dirs);
}
