/* server_plugins.c — 服务里热切换 ②侧车 / ③后训练件, 不重装 113 GB 底座(2026-10-10, 用户 "chat 页面再加一个切换不同侧车的功能,
 * 后训练结束也要一个自动重载的功能")。
 *
 * 链路: 聊天页 POST /api/models/plugins → 训练页模块校验(src/train/train_models.c: 侧车是装着的模型的、③ 的 base.fnv 对得上 ②)
 *   → tr_live.request(本文件)排一个切换任务进请求队列 → worker 轮到它时调 ds4_engine_v41_switch_plugins → 成了再写 serve_pick.txt。
 * 为什么走请求队列: 插件表(路由偏置 / 增益覆盖)是进程级的, 只能在两条请求之间换。单路 worker 按到达顺序取任务, 轮到它时前面的
 *   请求都答完了; 合批调度器取到它就停止接新请求, 等在跑的几路收尾再换(server_sched_v41.c)。
 * 请求立刻返回, 不让浏览器挂着等前面的长回答; 结果记在这里, 页面轮询 /api/models/list 的 switch 字段看。
 * 为什么切完要写 serve_pick.txt: 服务之后任何一次重起(训练回来、崩了重拉)都该是用户最后选的那套, 不是起服命令行上的老侧车。 */
#include "server_internal.h"
#include "../train/train_internal.h"

struct server_plugin_switch { char amp[2056], pt[2056]; };

static server *g_srv;
static const char *g_gguf;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_pending;
static char g_err[256];
/* 当前挂着的 ②③ 的副本: 引擎换目录时会释放老字符串, HTTP 线程直接读引擎的指针可能撞上刚释放的那份, 所以读这里(持锁) */
static char g_amp[2056], g_pt[2056];

static int plugins_request(const char *amp, const char *pt, char *err, size_t errn) {
    pthread_mutex_lock(&g_mu);
    if (g_pending) { pthread_mutex_unlock(&g_mu); snprintf(err, errn, "上一次切换还在排队"); return -1; }
    g_pending = 1; g_err[0] = 0;
    pthread_mutex_unlock(&g_mu);
    job *j = calloc(1, sizeof *j);
    struct server_plugin_switch *sw = calloc(1, sizeof *sw);
    if (!j || !sw) { free(j); free(sw); snprintf(err, errn, "内存不够"); pthread_mutex_lock(&g_mu); g_pending = 0; pthread_mutex_unlock(&g_mu); return -1; }
    snprintf(sw->amp, sizeof sw->amp, "%s", amp ? amp : "");
    snprintf(sw->pt, sizeof sw->pt, "%s", pt ? pt : "");
    j->fd = -1; j->sw = sw;
    pthread_mutex_init(&j->mu, NULL); pthread_cond_init(&j->cv, NULL);
    if (!enqueue(g_srv, j)) {
        pthread_mutex_destroy(&j->mu); pthread_cond_destroy(&j->cv); free(sw); free(j);
        pthread_mutex_lock(&g_mu); g_pending = 0; pthread_mutex_unlock(&g_mu);
        snprintf(err, errn, "服务在停");
        return -1;
    }
    server_log(DS4_LOG_DEFAULT, "ds4-server: 切换插件已排队: 侧车 %s / 后训练 %s(等前面的请求答完)", amp && amp[0] ? amp : "(不挂)", pt && pt[0] ? pt : "(不挂)");
    return 0;
}

static void plugins_status(char *amp, size_t an, char *pt, size_t pn, int *pending, char *err, size_t en) {
    pthread_mutex_lock(&g_mu);
    snprintf(amp, an, "%s", g_amp); snprintf(pt, pn, "%s", g_pt);
    *pending = g_pending; snprintf(err, en, "%s", g_err);
    pthread_mutex_unlock(&g_mu);
}

void server_plugins_init(server *s, const char *gguf) {
    if (!ds4_engine_is_v41(s->engine)) return;   /* V4 没有这套插件: 钩子不挂, 页面就不给切换 */
    g_srv = s; g_gguf = gguf;
    const char *a = ds4_engine_v41_amp_dir(), *p = ds4_engine_v41_posttrain_dir();
    snprintf(g_amp, sizeof g_amp, "%s", a ? a : ""); snprintf(g_pt, sizeof g_pt, "%s", p ? p : "");
    tr_live.gguf = g_gguf; tr_live.request = plugins_request; tr_live.status = plugins_status;
}

void server_plugins_apply(server *s, job *j) {
    (void)s;
    struct server_plugin_switch *sw = j->sw;
    char err[256] = "";
    const int rc = ds4_engine_v41_switch_plugins(sw->amp, sw->pt, err, sizeof err);
    if (rc == 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: 插件已切换: 侧车 %s / 后训练 %s", sw->amp[0] ? sw->amp : "(不挂)", sw->pt[0] ? sw->pt : "(不挂)");
        char gabs[2056], perr[256] = "";
        if (g_gguf[0] == '/') snprintf(gabs, sizeof gabs, "%s", g_gguf); else snprintf(gabs, sizeof gabs, "%s/%s", tr_root, g_gguf);
        if (!tr_pick_write(gabs, sw->amp, sw->pt, perr, sizeof perr))   /* 切成了但没记下: 不回滚(服务已经在用新的), 只说清下次重起会回到老选择 */
            server_log(DS4_LOG_DEFAULT, "ds4-server: ★切换已生效, 但写 serve_pick.txt 失败(%s): 服务下次重起会回到原来的选择★", perr);
    } else server_log(DS4_LOG_DEFAULT, "ds4-server: 插件切换没成: %s", err);
    pthread_mutex_lock(&g_mu);
    if (rc == 0) { snprintf(g_amp, sizeof g_amp, "%s", sw->amp); snprintf(g_pt, sizeof g_pt, "%s", sw->pt); }
    g_pending = 0; snprintf(g_err, sizeof g_err, "%s", rc == 0 ? "" : err);
    pthread_mutex_unlock(&g_mu);
    pthread_mutex_destroy(&j->mu); pthread_cond_destroy(&j->cv);
    free(sw); free(j);
}
