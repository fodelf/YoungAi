/* train_models.c — 工作台"模型"页的接口(2026-10-10, 用户 "少一个模型 tab, 可以下载模型, 跟 unsloth 的客户端页面一样")。总述见 train_internal.h。
 *
 * 本机模型 = gguf/ 下三层以内、≥8 GiB 的 .gguf。小于这个的是 MTP/草稿器之类的部件, 单独装不起来, 列出来只会让人点错。
 * 配套侧车 = 与 GGUF 同目录、名字以 "<GGUF 去掉 .gguf>-" 开头、里面有 gr_L*.bin 或 manifest.txt 的目录。
 *   反修产物和 HF 发布包都是这个命名, 所以按前缀配对, 不需要任何清单。
 * n-gram 表: GGUF 上一级目录里有 deepseek-engram/(下载包的布局)就给起服加 --engram-dir;
 *   没有就用 GGUF 元数据里记的路径(本机自己量化的模型, 那条路径在本机是真的)。
 * 下载 = hf_install.sh download --dir gguf/hub; 进度 = hf_install.sh probe(只看盘不联网)。包里有什么、每件多大只写在那个脚本里,
 *   这里一个数也不抄 —— 抄了迟早和发布包对不上。速度/剩余时间由页面拿两次读数的差自己算。
 * 加载 = 把选择写进 gguf/serve_pick.txt, 再让主进程停了模型子进程重起(serve_1m_spark.sh 不带参数时读它)。
 *   训练中点加载只写文件不打断训练: 训完 train_cycle.sh 装回来的就是新选的这套。 */
#include "train_internal.h"
#include "../common/ds4_gr_fnv.h"   /* ③ 配 ② 的指纹: 与引擎核对、解算器落盘同一份算法 */
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#define MODEL_MIN_BYTES (8ull << 30)
#define MAX_SIDECARS 64
#define MAX_POSTTRAINS 8   /* 聊天页下拉里列最近几趟训练的 ③: 再多就是翻历史, 那在"记录"页 */
/* 指纹扫层上限: 引擎按模型元数据的层数扫, ds4-train 不链引擎拿不到那个数。扫到 64(比 V4.x 的层数都大)结果一样 ——
 * 侧车里没有超出真实层数的文件, 不存在的层号本来就跳过。真对不上时引擎热切前的试探装载还会再拒一次。 */
#define FNV_SCAN_LAYERS 64

tr_live_t tr_live;

static long mem_total_mb(void) {
#ifdef __APPLE__
    uint64_t b = 0; size_t l = sizeof b;
    return sysctlbyname("hw.memsize", &b, &l, NULL, 0) == 0 ? (long)(b >> 20) : -1;
#else
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char ln[256]; long kb = -1;
    while (fgets(ln, sizeof ln, f)) if (sscanf(ln, "MemTotal: %ld kB", &kb) == 1) break;
    fclose(f);
    return kb >= 0 ? kb / 1024 : -1;
#endif
}

/* 绝对路径 → 相对仓库根(页面显示和回传都用相对路径); 不在仓库下的原样给 */
static const char *rel(const char *p) {
    const size_t n = strlen(tr_root);
    return (!strncmp(p, tr_root, n) && p[n] == '/') ? p + n + 1 : p;
}
static bool ends_with(const char *s, const char *suf) { const size_t a = strlen(s), b = strlen(suf); return a >= b && !strcmp(s + a - b, suf); }
static bool is_dir(const char *p) { struct stat st; return stat(p, &st) == 0 && S_ISDIR(st.st_mode); }

static bool is_sidecar_dir(const char *p) {
    DIR *d = opendir(p); if (!d) return false;
    struct dirent *de; bool ok = false;
    while (!ok && (de = readdir(d))) ok = !strncmp(de->d_name, "gr_L", 4) || !strcmp(de->d_name, "manifest.txt");
    closedir(d);
    return ok;
}
static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

/* GGUF 的 n-gram 目录: <GGUF 所在目录>/../deepseek-engram 存在就是它, 否则空串 */
static void engram_of(const char *gguf_abs, char *out, size_t n) {
    char d[TR_PATH * 2 + 8]; snprintf(d, sizeof d, "%s", gguf_abs);
    char *s = strrchr(d, '/'); if (s) *s = 0;
    s = strrchr(d, '/'); if (s) *s = 0;
    snprintf(out, n, "%s/deepseek-engram", d);
    if (!is_dir(out)) out[0] = 0;
}

/* GGUF 的配套侧车: 同目录、"<GGUF 去 .gguf>-" 前缀、里面有 gr_L*.bin / manifest.txt。回填 strdup 的绝对路径(按名排序), 调用方 free */
static int list_sidecars(const char *gguf_abs, char *out[], int max) {
    char dir[TR_PATH * 2 + 8], stem[512]; snprintf(dir, sizeof dir, "%s", gguf_abs);
    char *sl = strrchr(dir, '/'); if (!sl) return 0;
    *sl = 0;
    const char *name = sl + 1; const size_t nl = strlen(name);
    if (nl <= 5) return 0;
    snprintf(stem, sizeof stem, "%.*s-", (int)(nl - 5), name);
    int ns = 0;
    DIR *d = opendir(dir); struct dirent *de;
    while (d && ns < max && (de = readdir(d))) {
        if (strncmp(de->d_name, stem, strlen(stem))) continue;
        char p[TR_PATH * 2 + 600]; snprintf(p, sizeof p, "%s/%s", dir, de->d_name);
        if (is_dir(p) && is_sidecar_dir(p)) out[ns++] = strdup(p);
    }
    if (d) closedir(d);
    qsort(out, (size_t)ns, sizeof *out, cmp_str);
    return ns;
}

static void emit_model(ds4_buf *b, int *n, const char *dir, const char *name, const struct stat *st) {
    char abs[TR_PATH + 600], eg[TR_PATH + 640], hub[TR_PATH + 16];
    snprintf(abs, sizeof abs, "%s/%s", dir, name);
    snprintf(hub, sizeof hub, "%s/gguf/hub/", tr_root);
    engram_of(abs, eg, sizeof eg);
    ds4_buf_puts(b, *n ? "," : ""); (*n)++;
    ds4_buf_puts(b, "{\"path\":"); ds4_json_escape(b, rel(abs));
    ds4_buf_printf(b, ",\"bytes\":%lld,\"mtime\":%ld,\"hub\":%s,\"engram\":", (long long)st->st_size, (long)st->st_mtime, strncmp(abs, hub, strlen(hub)) ? "false" : "true");
    ds4_json_escape(b, eg[0] ? rel(eg) : "");
    ds4_buf_puts(b, ",\"sidecars\":[");
    char *sc[MAX_SIDECARS]; const int ns = list_sidecars(abs, sc, MAX_SIDECARS);
    for (int i = 0; i < ns; i++) { if (i) ds4_buf_putc(b, ','); ds4_json_escape(b, rel(sc[i])); free(sc[i]); }
    ds4_buf_puts(b, "]}");
}

/* 侧车目录的指纹("十六进制 文件数", 与 ③ 的 base.fnv 同写法)。整目录 40 MB 量级, 每 2.5 s 一轮重算不划算: 按目录修改时间缓存 */
static void sidecar_fnv(const char *dir_abs, char *out, size_t n) {
    static struct { char path[TR_PATH * 2 + 8]; time_t mt; char fnv[40]; } cache[MAX_SIDECARS];
    static int nc;
    struct stat st; const time_t mt = stat(dir_abs, &st) == 0 ? st.st_mtime : 0;
    for (int i = 0; i < nc; i++) if (!strcmp(cache[i].path, dir_abs) && cache[i].mt == mt) { snprintf(out, n, "%s", cache[i].fnv); return; }
    unsigned cnt = 0; const uint64_t h = ds4_gr_dir_fnv(dir_abs, FNV_SCAN_LAYERS, &cnt);
    snprintf(out, n, "%016llx %u", (unsigned long long)h, cnt);
    const int k = nc < MAX_SIDECARS ? nc++ : (int)(mt % MAX_SIDECARS);
    snprintf(cache[k].path, sizeof cache[k].path, "%s", dir_abs); cache[k].mt = mt; snprintf(cache[k].fnv, sizeof cache[k].fnv, "%s", out);
}
static void read_fnv_file(const char *dir_abs, char *out, size_t n) {   /* ③ 目录的 base.fnv 第一行; 没有给空串 */
    char p[TR_PATH * 2 + 32]; snprintf(p, sizeof p, "%s/base.fnv", dir_abs);
    unsigned long long h = 0; unsigned c = 0; out[0] = 0;
    FILE *f = fopen(p, "r"); if (!f) return;
    if (fscanf(f, "%llx %u", &h, &c) == 2) snprintf(out, n, "%016llx %u", h, c);
    fclose(f);
}

/* 最近几趟过了门的训练的 ③(<训练目录>/<pick.txt 里的轮>, 要有 base.fnv): 按选轮时间倒序 */
static void emit_posttrains(ds4_buf *b) {
    struct { char path[TR_PATH + 600], run[256], fnv[40]; time_t t; } c[MAX_POSTTRAINS];
    int nc = 0;
    DIR *d = opendir(tr_ftd); struct dirent *de;
    while (d && (de = readdir(d))) {
        if (de->d_name[0] == '.') continue;
        char p[TR_PATH + 600], pk[64] = ""; struct stat st;
        snprintf(p, sizeof p, "%s/%s/pick.txt", tr_ftd, de->d_name);
        FILE *f = stat(p, &st) == 0 ? fopen(p, "r") : NULL;
        if (!f) continue;
        if (!fgets(pk, sizeof pk, f)) pk[0] = 0;
        fclose(f); pk[strcspn(pk, "\n")] = 0;
        if (!pk[0] || !tr_clean(pk, "_")) continue;
        int k = nc;
        if (nc == MAX_POSTTRAINS) { k = 0; for (int i = 1; i < nc; i++) if (c[i].t < c[k].t) k = i; if (c[k].t >= st.st_mtime) continue; }
        else nc++;
        snprintf(c[k].path, sizeof c[k].path, "%s/%s/%s", tr_ftd, de->d_name, pk);
        snprintf(c[k].run, sizeof c[k].run, "%s", de->d_name); c[k].t = st.st_mtime;
        read_fnv_file(c[k].path, c[k].fnv, sizeof c[k].fnv);
        if (!c[k].fnv[0]) { c[k] = c[--nc]; continue; }   /* 没有 base.fnv: 核不了配哪份 ②, 不往下拉里放 */
    }
    if (d) closedir(d);
    ds4_buf_puts(b, ",\"posttrains\":[");
    for (int done = 0; done < nc; done++) {   /* 选择排序出倒序(最多 8 个) */
        int k = -1; for (int i = 0; i < nc; i++) if (c[i].t >= 0 && (k < 0 || c[i].t > c[k].t)) k = i;
        ds4_buf_puts(b, done ? ",{\"path\":" : "{\"path\":"); ds4_json_escape(b, rel(c[k].path));
        ds4_buf_puts(b, ",\"run\":"); ds4_json_escape(b, c[k].run);
        ds4_buf_puts(b, ",\"fnv\":"); ds4_json_escape(b, c[k].fnv);
        ds4_buf_printf(b, ",\"time\":%ld}", (long)c[k].t);
        c[k].t = -1;
    }
    ds4_buf_putc(b, ']');
}

/* gguf/ 往下三层(gguf/hub/model/X.gguf 正好三层)。用 lstat: ./ds4flash.gguf 这类别名软链不重复列, 目录软链也不会绕圈 */
static void scan_models(ds4_buf *b, int *n, const char *dir, int depth) {
    DIR *d = opendir(dir); if (!d) return;
    struct dirent *de;
    while ((de = readdir(d))) {
        if (de->d_name[0] == '.') continue;
        const bool gg = ends_with(de->d_name, ".gguf");
        if (!gg && (de->d_type != DT_DIR && de->d_type != DT_UNKNOWN)) continue;   /* 产物目录里成千上万的 .bin 不逐个 stat */
        char p[TR_PATH + 600]; snprintf(p, sizeof p, "%s/%s", dir, de->d_name);
        struct stat st; if (lstat(p, &st)) continue;
        if (S_ISDIR(st.st_mode)) { if (depth > 0 && !ends_with(de->d_name, "-engine")) scan_models(b, n, p, depth - 1); }
        else if (gg && S_ISREG(st.st_mode) && (unsigned long long)st.st_size >= MODEL_MIN_BYTES) emit_model(b, n, dir, de->d_name, &st);
    }
    closedir(d);
}

/* ui_logs/ 下名字以 prefix 开头的最新一份的尾巴(下载日志带进度条, 跑一整天能到几十 MB, 只读最后 4 KB) */
static time_t log_tail(ds4_buf *b, const char *const prefixes[]) {
    char ld[TR_PATH + 16], best[TR_PATH + 300] = ""; time_t bt = 0;
    snprintf(ld, sizeof ld, "%s/ui_logs", tr_ftd);
    DIR *d = opendir(ld); struct dirent *de;
    while (d && (de = readdir(d))) {
        bool hit = false;
        for (int i = 0; prefixes[i]; i++) hit |= !strncmp(de->d_name, prefixes[i], strlen(prefixes[i]));
        if (!hit) continue;
        char p[TR_PATH + 300]; snprintf(p, sizeof p, "%s/%s", ld, de->d_name);
        struct stat st; if (!stat(p, &st) && st.st_mtime >= bt) { bt = st.st_mtime; snprintf(best, sizeof best, "%s", p); }
    }
    if (d) closedir(d);
    FILE *f = best[0] ? fopen(best, "rb") : NULL;
    if (!f) { ds4_buf_puts(b, "\"\""); return 0; }
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, sz > 4096 ? sz - 4096 : 0, SEEK_SET);
    char t[4097]; const size_t got = fread(t, 1, 4096, f); fclose(f); t[got] = 0;
    for (size_t i = 0; i < got; i++) if (!t[i]) t[i] = ' ';
    ds4_json_escape(b, t);
    return bt;
}

/* hf_install.sh probe → hub 对象。shard0 = 第一片 n-gram 分片的文件名(给"本机已有分片"的提示用) */
static void emit_hub(ds4_buf *b) {
    char cmd[TR_PATH * 2 + 128], ln[1024], shard0[256] = "";
    snprintf(cmd, sizeof cmd, "bash '%s/gguf-tools/scripts/hf_install.sh' probe --dir '%s/gguf/hub' 2>/dev/null", tr_root, tr_root);
    ds4_buf_puts(b, "{\"dir\":\"gguf/hub\",\"sidecars\":[");
    FILE *f = popen(cmd, "r"); int nsc = 0;
    ds4_buf tail = {0};
    while (f && fgets(ln, sizeof ln, f)) {
        char a[512] = "", c[512] = ""; long long x = 0, y = 0;
        if (sscanf(ln, "repo %511s", a) == 1) { ds4_buf_puts(&tail, ",\"repo\":"); ds4_json_escape(&tail, a); }
        else if (sscanf(ln, "shards %255s", shard0) == 1) {}
        else if (sscanf(ln, "gguf %511s %lld %lld", a, &x, &y) == 3) { ds4_buf_puts(&tail, ",\"gguf\":{\"name\":"); ds4_json_escape(&tail, a); ds4_buf_printf(&tail, ",\"bytes\":%lld,\"have\":%lld}", x, y); }
        else if (sscanf(ln, "engram %lld %lld", &x, &y) == 2) ds4_buf_printf(&tail, ",\"engram\":{\"bytes\":%lld,\"have\":%lld}", x, y);
        else if (sscanf(ln, "sidecar %511s %511s %lld", a, c, &x) == 3) {
            ds4_buf_puts(b, nsc++ ? ",{\"domain\":" : "{\"domain\":"); ds4_json_escape(b, a);
            ds4_buf_puts(b, ",\"dir\":"); ds4_json_escape(b, c); ds4_buf_printf(b, ",\"have\":%s}", x ? "true" : "false");
        }
    }
    if (f) pclose(f);
    ds4_buf_putc(b, ']');
    if (tail.len) ds4_buf_append(b, tail.ptr, tail.len);
    ds4_buf_free(&tail);
    /* 本机已经有官方 n-gram 分片(量化时从 HF 下过整份底模)就提示出来: 省 203 GB 下载 */
    char hint[TR_PATH + 300] = "", hd[TR_PATH + 8]; snprintf(hd, sizeof hd, "%s/hf", tr_root);
    DIR *d = shard0[0] ? opendir(hd) : NULL; struct dirent *de;
    while (d && !hint[0] && (de = readdir(d))) {
        if (de->d_name[0] == '.') continue;
        char p[TR_PATH + 600]; struct stat st;
        snprintf(p, sizeof p, "%s/%s/%s", hd, de->d_name, shard0);
        if (!stat(p, &st) && S_ISREG(st.st_mode)) snprintf(hint, sizeof hint, "hf/%s", de->d_name);
    }
    if (d) closedir(d);
    ds4_buf_puts(b, ",\"engram_hint\":"); ds4_json_escape(b, hint);
    ds4_buf_puts(b, "}");
}

/* gguf/serve_pick.txt(格式见 serve_1m_spark.sh pick): 第 1 行 GGUF, 第 2 行侧车或 none, 之后逐行额外参数, 这里只认 --posttrain 后面那项 */
static void read_pick(char *g, size_t gn, char *z, size_t zn, char *pt, size_t pn) {
    char p[TR_PATH + 32], ln[TR_PATH + 64]; snprintf(p, sizeof p, "%s/gguf/serve_pick.txt", tr_root);
    g[0] = z[0] = pt[0] = 0;
    FILE *f = fopen(p, "r"); if (!f) return;
    if (fgets(g, (int)gn, f)) g[strcspn(g, "\n")] = 0;
    if (fgets(z, (int)zn, f)) z[strcspn(z, "\n")] = 0;
    bool want = false;
    while (fgets(ln, sizeof ln, f)) {
        ln[strcspn(ln, "\n")] = 0;
        if (want) { snprintf(pt, pn, "%s", ln); want = false; } else want = !strcmp(ln, "--posttrain");
    }
    fclose(f);
}
static void emit_trio(ds4_buf *b, const char *key, const char *g, const char *z, const char *pt) {
    ds4_buf_printf(b, ",\"%s\":", key);
    if (!g[0]) { ds4_buf_puts(b, "null"); return; }
    ds4_buf_puts(b, "{\"gguf\":"); ds4_json_escape(b, rel(g));
    ds4_buf_puts(b, ",\"zchain\":"); ds4_json_escape(b, (!z[0] || !strcmp(z, "none")) ? "" : rel(z));
    ds4_buf_puts(b, ",\"posttrain\":"); ds4_json_escape(b, pt[0] ? rel(pt) : ""); ds4_buf_putc(b, '}');
}
/* 正在用的 ①②③: 本进程就是服务(钩子在)就问引擎 —— 热切过之后命令行上的 --zchain 已经不是真相; 否则扫 ds4-server 的命令行 */
static void current_trio(char *g, size_t gn, char *z, size_t zn, char *pt, size_t pn, int *pending, char *err, size_t en) {
    *pending = 0; err[0] = 0;
    if (!tr_live.status) { tr_server_model(g, gn, z, zn, pt, pn); return; }
    if (tr_live.gguf[0] == '/') snprintf(g, gn, "%s", tr_live.gguf);
    else snprintf(g, gn, "%s/%s", tr_root, tr_live.gguf);
    tr_live.status(z, zn, pt, pn, pending, err, en);
}

/* 先写临时再改名: 读的一方(起服 / 训练线程 / serve_1m_spark.sh)不会读到半个文件 */
bool tr_pick_write(const char *gguf_abs, const char *zch_abs, const char *pt_abs, char *err, size_t errn) {
    char eg[TR_PATH * 2 + 32], p[TR_PATH + 32], tmp[TR_PATH + 40];
    engram_of(gguf_abs, eg, sizeof eg);
    snprintf(p, sizeof p, "%s/gguf/serve_pick.txt", tr_root); snprintf(tmp, sizeof tmp, "%s.tmp", p);
    FILE *f = fopen(tmp, "w");
    if (!f) { snprintf(err, errn, "写不了 %s", tmp); return false; }
    fprintf(f, "%s\n%s\n", gguf_abs, (zch_abs && zch_abs[0]) ? zch_abs : "none");
    if (eg[0]) fprintf(f, "--engram-dir\n%s\n", eg);
    if (pt_abs && pt_abs[0]) fprintf(f, "--posttrain\n%s\n", pt_abs);
    fclose(f);
    if (rename(tmp, p) != 0) { snprintf(err, errn, "改名 %s 失败", p); return false; }
    return true;
}

/* with_hub = 0(?hub=0): 聊天页每 2.5 s 拉一次只为看侧车与切换状态, 不跑 hf_install.sh probe(每次起一个 bash)、不读日志尾 */
static int api_list(ds4_buf *b, bool with_hub) {
    char g[TR_PATH * 2 + 8], z[TR_PATH * 2 + 8], pt[TR_PATH * 2 + 8], gg[TR_PATH + 16], serr[256];
    int pending = 0;
    tr_procs p; tr_proc_scan(&p);
    ds4_buf_printf(b, "{\"mem_total_mb\":%ld,\"mem_avail_mb\":%ld,\"downloading\":%s,\"live\":%s", mem_total_mb(), tr_mem_avail_mb(),
                   p.dl ? "true" : "false", tr_live.request ? "true" : "false");
    current_trio(g, sizeof g, z, sizeof z, pt, sizeof pt, &pending, serr, sizeof serr); emit_trio(b, "loaded", g, z, pt);
    ds4_buf_printf(b, ",\"switch\":{\"pending\":%s,\"error\":", pending ? "true" : "false"); ds4_json_escape(b, serr); ds4_buf_putc(b, '}');
    /* 装着的模型的侧车指纹: 聊天页按它把 ③ 配到对的 ②(③ 的 base.fnv = 训练时挂的 ② 的指纹) */
    ds4_buf_printf(b, ",\"fnv\":{\"\":\"%016llx 0\"", (unsigned long long)DS4_GR_FNV_SEED);   /* 键 "" = 不挂侧车(对着裸 ① 训的 ③ 配它) */
    if (g[0]) {
        char *sc[MAX_SIDECARS], fv[40]; const int ns = list_sidecars(g, sc, MAX_SIDECARS);
        for (int i = 0; i < ns; i++) { sidecar_fnv(sc[i], fv, sizeof fv); ds4_buf_putc(b, ','); ds4_json_escape(b, rel(sc[i])); ds4_buf_putc(b, ':'); ds4_json_escape(b, fv); free(sc[i]); }
    }
    ds4_buf_putc(b, '}');
    emit_posttrains(b);
    read_pick(g, sizeof g, z, sizeof z, pt, sizeof pt); emit_trio(b, "pick", g, z, pt);
    ds4_buf_puts(b, ",\"local\":[");
    int n = 0; snprintf(gg, sizeof gg, "%s/gguf", tr_root); scan_models(b, &n, gg, 3);
    ds4_buf_putc(b, ']');
    if (!with_hub) { ds4_buf_putc(b, '}'); return 200; }
    ds4_buf_puts(b, ",\"hub\":"); emit_hub(b);
    static const char *const hubp[] = { "hub_", NULL }, *const loadp[] = { "cycle_", "serve_", NULL };
    ds4_buf_puts(b, ",\"hub_log\":"); log_tail(b, hubp);
    ds4_buf_puts(b, ",\"load_log\":"); log_tail(b, loadp);
    ds4_buf_putc(b, '}');
    return 200;
}

static int reply(ds4_buf *out, bool ok, bool deferred, const char *err, int code_fail) {
    ds4_buf_printf(out, "{\"ok\":%s,\"deferred\":%s,\"error\":", ok ? "true" : "false", deferred ? "true" : "false");
    ds4_json_escape(out, err ? err : ""); ds4_buf_putc(out, '}');
    return ok ? 200 : code_fail;
}

static int api_download(const tr_req *rq, ds4_buf *out) {
    char ep[256], eg[TR_PATH], nx[16], hub[TR_PATH + 16], egabs[TR_PATH * 2 + 8];
    tr_body_get(rq, "endpoint", ep, sizeof ep); tr_body_get(rq, "engram_dir", eg, sizeof eg); tr_body_get(rq, "no_xet", nx, sizeof nx);
    snprintf(hub, sizeof hub, "%s/gguf/hub", tr_root);
    if (ep[0] && (strncmp(ep, "https://", 8) || !tr_clean(ep, ":/._-"))) return reply(out, false, false, "镜像地址要是 https://… 且只含字母数字与 : / . _ -", 400);
    if (eg[0] && (!tr_clean(eg, "._/-") || strstr(eg, ".."))) return reply(out, false, false, "n-gram 目录只许字母数字与 . _ / -", 400);
    if (eg[0] == '/') snprintf(egabs, sizeof egabs, "%s", eg);
    else if (eg[0]) snprintf(egabs, sizeof egabs, "%s/%s", tr_root, eg);   /* 相对路径按仓库根算(脚本在仓库根跑, 但给它绝对的最稳) */
    const char *args[10]; int na = 0;
    args[na++] = "--dir"; args[na++] = hub;
    if (ep[0]) { args[na++] = "--endpoint"; args[na++] = ep; }
    if (eg[0]) { args[na++] = "--engram-dir"; args[na++] = egabs; }
    if (!strcmp(nx, "true") || !strcmp(nx, "1")) args[na++] = "--no-xet";
    args[na] = NULL;
    char err[256] = "";
    const bool ok = tr_spawn_download(args, err, sizeof err);
    return reply(out, ok, false, err, 409);
}

/* 加载(只在主进程): 校验 → 写 serve_pick.txt → 按当前状态决定怎么生效 */
static int api_load(const tr_req *rq, ds4_buf *out, int serving, int port) {
    char g[TR_PATH], z[TR_PATH], ga[TR_PATH * 2 + 8], za[TR_PATH * 2 + 8] = "none", err[256] = "";   /* 根 + 相对路径, 两段都可能满 TR_PATH */
    if (serving) return reply(out, false, false, "这是模型子进程; 换模型要从主进程 ds4-train 的页面操作", 409);
    tr_body_get(rq, "gguf", g, sizeof g); tr_body_get(rq, "sidecar", z, sizeof z);
    struct stat st;
    snprintf(ga, sizeof ga, "%s/%s", tr_root, g);
    if (!tr_clean(g, "._/-") || strstr(g, "..") || strncmp(g, "gguf/", 5) || !ends_with(g, ".gguf") || stat(ga, &st) || !S_ISREG(st.st_mode))
        return reply(out, false, false, "模型路径不合法或不存在", 400);
    if (z[0] && strcmp(z, "none")) {
        const char *gs = strrchr(g, '/'), *zs = strrchr(z, '/');
        snprintf(za, sizeof za, "%s/%s", tr_root, z);
        const bool same_dir = zs && gs && zs - z == gs - g && !strncmp(z, g, (size_t)(gs - g));
        const bool prefix = same_dir && !strncmp(zs + 1, gs + 1, strlen(gs + 1) - 5) && zs[1 + strlen(gs + 1) - 5] == '-';
        if (!tr_clean(z, "._/-") || strstr(z, "..") || !prefix || !is_sidecar_dir(za))
            return reply(out, false, false, "侧车目录不是这个模型的(要与 GGUF 同目录、同名前缀, 里面有 gr_L*.bin)", 400);
    }
    const tr_model_phase mp = tr_model_state();
    if (mp == TR_MODEL_STARTING || mp == TR_MODEL_STOPPING) return reply(out, false, false, "正在加载模型, 等它装完", 409);
    char lg[TR_PATH * 2 + 8], lz[TR_PATH * 2 + 8], lp[TR_PATH * 2 + 8], serr[256]; int pending = 0;
    const bool live = tr_model_live();
    current_trio(lg, sizeof lg, lz, sizeof lz, lp, sizeof lp, &pending, serr, sizeof serr);
    if (live && !strcmp(rel(lg), g) && !strcmp(lz[0] ? rel(lz) : "none", strcmp(za, "none") ? z : "none"))
        return reply(out, false, false, "已经在用这一套(只换侧车/后训练件在聊天页切, 不用重装)", 409);
    /* ③ 只在 ①② 没变时留着: 它是对着那份 ② 训的, 换了 ② 还挂着它, 起服会被引擎拒 */
    char pg[TR_PATH * 2 + 8], pz[TR_PATH * 2 + 8], ppt[TR_PATH * 2 + 8];
    read_pick(pg, sizeof pg, pz, sizeof pz, ppt, sizeof ppt);
    const bool keep_pt = !strcmp(pg, ga) && !strcmp(pz, za);
    if (!tr_pick_write(ga, za, keep_pt ? ppt : NULL, err, sizeof err)) return reply(out, false, false, err, 500);
    tr_job_info j; tr_job_status(&j);
    if (j.active) return reply(out, true, true, "", 200);   /* 训练中: 只记下, 训完按它装回来 */
    /* 模型子进程在: 先停它再起新的(只能装一份); 不在: 直接起 */
    const bool ok = tr_model_start(TR_MODEL_PORT(port), live, err, sizeof err);
    return reply(out, ok, false, err, 409);
}

/* 聊天页热切 ②③: 校验(侧车是装着的模型的、③ 的 base.fnv 对得上这份 ②)→ 交给服务排一个切换任务。
 * 真正换在服务端两条请求之间做, 换成了才写 serve_pick.txt(server_plugins.c), 页面轮询 list 的 switch 看结果。 */
static int api_plugins(const tr_req *rq, ds4_buf *out, int serving) {
    if (!serving || !tr_live.request) return reply(out, false, false, "没有装着的模型(只换侧车要服务在跑; 没在跑就去模型页加载)", 409);
    char z[TR_PATH], pt[TR_PATH], za[TR_PATH * 2 + 8] = "", pa[TR_PATH * 2 + 8] = "", err[256] = "";
    tr_body_get(rq, "zchain", z, sizeof z); tr_body_get(rq, "posttrain", pt, sizeof pt);
    char g[TR_PATH * 2 + 8], cz[TR_PATH * 2 + 8], cp[TR_PATH * 2 + 8]; int pending = 0;
    current_trio(g, sizeof g, cz, sizeof cz, cp, sizeof cp, &pending, err, sizeof err);
    if (pending) return reply(out, false, false, "上一次切换还在排队(等当前回答结束)", 409);
    if (z[0] && strcmp(z, "none")) {
        if (!tr_clean(z, "._/-") || strstr(z, "..")) return reply(out, false, false, "侧车路径不合法", 400);
        snprintf(za, sizeof za, "%s/%s", tr_root, z);
        char *sc[MAX_SIDECARS]; const int ns = list_sidecars(g, sc, MAX_SIDECARS); bool mine = false;
        for (int i = 0; i < ns; i++) { mine |= !strcmp(sc[i], za); free(sc[i]); }
        if (!mine) return reply(out, false, false, "这个侧车不是当前模型的(要与 GGUF 同目录、同名前缀)", 400);
    }
    if (pt[0]) {
        struct stat st;
        if (!tr_clean(pt, "._/-") || strstr(pt, "..") || strncmp(pt, "gguf/", 5)) return reply(out, false, false, "后训练件路径不合法", 400);
        snprintf(pa, sizeof pa, "%s/%s", tr_root, pt);
        if (stat(pa, &st) || !S_ISDIR(st.st_mode)) return reply(out, false, false, "后训练件目录不存在", 400);
        char want[40], have[40]; read_fnv_file(pa, want, sizeof want);
        if (za[0]) sidecar_fnv(za, have, sizeof have); else snprintf(have, sizeof have, "%016llx 0", (unsigned long long)DS4_GR_FNV_SEED);
        if (!want[0]) return reply(out, false, false, "后训练件没有 base.fnv, 核不了它配哪份侧车", 400);
        if (strcmp(want, have)) return reply(out, false, false, "这份后训练件是对着另一份侧车训的(base.fnv 对不上), 挂上去引擎会拒", 400);
    }
    if (!strcmp(cz, za) && !strcmp(cp, pa)) return reply(out, false, false, "已经是这一套", 409);
    if (tr_live.request(za, pa, err, sizeof err) != 0) return reply(out, false, false, err, 409);
    return reply(out, true, false, "", 200);
}

int tr_models_api(const tr_req *rq, ds4_buf *out, int serving, int port) {
    if (strncmp(rq->path, "/api/models/", 12)) return -1;
    const char *op = rq->path + 12;
    const bool post = !strcmp(rq->method, "POST");
    char err[256] = "", hub[TR_PATH + 16];
    if (!strcmp(op, "list")) { char h[8]; tr_query_get(rq->query, "hub", h, sizeof h); return api_list(out, strcmp(h, "0") != 0); }
    if (!post) { ds4_buf_puts(out, "{\"error\":\"要 POST\"}"); return 405; }
    if (!strcmp(op, "download")) return api_download(rq, out);
    if (!strcmp(op, "cancel")) { snprintf(hub, sizeof hub, "%s/gguf/hub", tr_root); const bool ok = tr_cancel_download(hub, err, sizeof err); return reply(out, ok, false, err, 409); }
    if (!strcmp(op, "load")) return api_load(rq, out, serving, port);
    if (!strcmp(op, "plugins")) return api_plugins(rq, out, serving);
    ds4_buf_puts(out, "{\"error\":\"没有这个路径\"}");
    return 404;
}
