/* row_layout.inc.c — 行布局(域块) → fit/eval 行选取。★唯一实现★, zlayer 与 ds4quant_run 共用。
 *
 * ★为什么不走 env(2026-08-29 用户令"不要环境变量控制逻辑, 每次都这样不是丢了吗")★
 * 用 env 传行掩码 = 下次没人记得设 = 静默走回错的默认。今天两处实撞:
 *   ① LZRANK 未设 → p14:289 静默兜底 16(冠军是 64), 日志里只能从"试过的 k 只有 16/8/4/1"倒推
 *   ② 行掩码把 "32 块 × 256" 抄死在脚本里; 语料换成 8 域 × 1024 行 / 窗 128 后静默错位
 *      ⇒ 反修在拉丁文上拟合、西里尔文上判落地、阿拉伯+中日韩当 held(解码实测三段几乎零重叠)
 *      ⇒ z 落地 378/378 全拒。修掉后同一层 z 立刻复活(L0 组合 held 11.3%)。
 * 所以布局【跟着 ids 走】: <ids>.layout 由 amp_campaign.sh stage_idshalf 在抽样时落盘,
 * 消费方从【自己本来就拿到的 ids 路径】推导出来, 不经过任何开关、任何默认值。
 *
 * 布局文件格式(生产方写, 勿手改):
 *   win <窗宽>              窗与窗之间是源语料里的跳跃 = 上下文断点
 *   <域名> <起始行> <行数>   各域在 8192 行里【连续】铺开
 *
 * 选取规则 — 两个比例来自冠军 r64c 配方, 与语料无关, 故是仅有的两个常数:
 *   剔每个窗的前 25%(冠军 64/256): 窗首是跳跃点, 模型在那儿没有上下文 = "拼接毒"
 *   每个域块末 25% 的窗给 eval(冠军 8/32): ★fit 与 eval 都【从每个域块各取】★,
 *   而不是按行号切 —— 后者在"域连续铺"的布局下必然让两段落到不同的域上。 */

/* 自带头文件: 本片被两个 TU 在【不同位置】include(ds4quant_run 在 anchor 之后 /
 * zlayer 必须在 p1 之前=文件作用域), 不能假设调用方已经引好。重复 include 有守卫, 无害。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 返回 0=成功(out 数组由 malloc 给出, 调用方负责 free); -1=布局文件不在或不合法。
 * 硬失败由调用方决定: 需要分层的路径(反修/sweep 的落地判据)必须停, 不许猜。 */
static int row_layout_split(const char *ids_path,
                            int **fit_out, int *nfit_out, int **ev_out, int *nev_out)
{
    if (!ids_path) return -1;
    char lp[1024];
    snprintf(lp, sizeof lp, "%s.layout", ids_path);
    FILE *f = fopen(lp, "r");
    if (!f) return -1;

    int win = 0, cap = 1024, nf = 0, ne = 0;
    int *fit = (int *)malloc(sizeof(int) * (size_t)cap);
    int *ev  = (int *)malloc(sizeof(int) * (size_t)cap);
    if (!fit || !ev) { free(fit); free(ev); fclose(f); return -1; }

    char line[512];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char nm[128]; long a = 0, b = 0;
        if (sscanf(line, "win %d", &win) == 1) continue;
        if (!strncmp(line, "xshift ", 7)) continue;      /* 行偏移行不是域块(见 row_layout_xshift) */
        if (sscanf(line, "%127s %ld %ld", nm, &a, &b) != 3) continue;
        if (win <= 0 || b <= 0) continue;
        const int off = (int)a, cnt = (int)b;
        int nw = cnt / win; if (nw < 2) nw = 2;          /* 至少留一窗给 eval */
        const int skip = win / 4;                        /* 剔窗首 25% */
        int nev = nw / 4; if (nev < 1) nev = 1;          /* 末 25% 窗给 eval */
        for (int w = 0; w < nw; w++) {
            int x = off + w * win + skip, y = off + (w + 1) * win;
            if (y > off + cnt) y = off + cnt;
            for (int r = x; r < y; r++) {
                int **dst = (w < nw - nev) ? &fit : &ev;
                int *n    = (w < nw - nev) ? &nf  : &ne;
                if (*n >= cap) { cap *= 2;
                    int *t1 = (int *)realloc(fit, sizeof(int) * (size_t)cap);
                    int *t2 = (int *)realloc(ev,  sizeof(int) * (size_t)cap);
                    if (!t1 || !t2) { free(t1 ? t1 : fit); free(t2 ? t2 : ev); fclose(f); return -1; }
                    fit = t1; ev = t2; dst = (w < nw - nev) ? &fit : &ev; }
                (*dst)[(*n)++] = r;
            }
        }
    }
    fclose(f);
    if (win <= 0 || nf < 8 || ne < 2) { free(fit); free(ev); return -1; }
    *fit_out = fit; *nfit_out = nf; *ev_out = ev; *nev_out = ne;
    return 0;
}

/* ★行偏移 "xshift <S0> <N>"(2026-09-08 夜间 z 微调)★ 教师序列 = [前段 S0 行 ‖ 事后上下文 N 行 ‖ 正文],
 * 学生序列(部署口径)没有那 N 行上下文, 所以学生侧的捕获/打分比锚少 N 行, 行号映射:
 *     锚行 r < S0        → 学生行 r
 *     S0 <= r < S0+N     → 学生没有(上下文只给教师看; 这 N 行不在任何域块里 ⇒ 永不是 fit/eval 行)
 *     r >= S0+N          → 学生行 r − N
 * 为什么写进 .layout 而不是命令行: 消费方(zlayer / anchor_metrics)手里只有锚/ids 路径, 偏移与行域
 * 是同一份事实, 分两处传迟早对不上(2026-08-29 行掩码抄死在脚本里的同款事故)。缺此行 = 无偏移。
 * 返回 1=有偏移(*s0,*n 有效), 0=无。 */
static int __attribute__((unused)) row_layout_xshift(const char *ids_path, int *s0, int *n)
{
    if (!ids_path) return 0;
    char lp[1024];
    snprintf(lp, sizeof lp, "%s.layout", ids_path);
    FILE *f = fopen(lp, "r");
    if (!f) return 0;
    int found = 0; long a = 0, b = 0;
    char line[512];
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "xshift %ld %ld", &a, &b) == 2 && a >= 0 && b > 0) { found = 1; break; }
    fclose(f);
    if (!found) return 0;
    *s0 = (int)a; *n = (int)b;
    return 1;
}

/* ★进程内缓存(2026-08-29)★ 布局只解析一次, 反修(p7 的 z 闸/GE 闸)与 sweep(p13)共用同一份
 * 行表 —— 两处用【同一批打分行】才有可比性, 也免得各解析一遍各写一套。
 * 返回 eval 行表(打分行), *n 给长度; 布局缺则返回 NULL, 调用方必须硬停不许回退按行号切。 */
static int *g_rl_fit = NULL, *g_rl_ev = NULL;
static int  g_rl_nfit = 0,   g_rl_nev = 0, g_rl_tried = 0;
static const int *row_layout_ev(const char *base, int *n)
{
    if (!g_rl_tried) { g_rl_tried = 1;
        if (row_layout_split(base, &g_rl_fit, &g_rl_nfit, &g_rl_ev, &g_rl_nev) != 0) {
            g_rl_fit = g_rl_ev = NULL; g_rl_nfit = g_rl_nev = 0; } }
    if (n) *n = g_rl_nev;
    return g_rl_ev;
}
