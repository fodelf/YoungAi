/* asst_internal.h — 工作台的个人助理(2026-10-11, 用户: "对标 muse, 把聊天页面变成个人 Agent 助理" → "安全沙盒第一")。方案稿 docs/assistant.md。
 *
 * 住在 ds4-train 主进程里(主进程永不退 ⇒ 关页继续跑、定时任务才成立); 模型子进程 ds4-server 只按 OpenAI 协议算 token。
 * ★每一次工具调用都在一个新 fork 的子进程里、先上内核沙盒再干活★(src/sandbox/sbx_*.c):
 *   Linux = Landlock(文件按目录白名单, TCP 全禁, 不许给域外进程发信号/ptrace) + seccomp(socket/io_uring/ptrace/mount… 一律 EPERM) + rlimit;
 *   macOS = Seatbelt(deny default, 只开白名单目录)。
 *   为什么不靠路径检查: 路径检查是我写的 C, 有 bug 就漏(软链、..、竞态); 内核规则按真实文件层级判, 工具代码被骗了也出不去。
 *   上锁失败 = 这次工具调用拒绝执行, 从不退回"不上锁照跑"(安全第一, 也是禁兜底铁律)。
 * 守门(sbx_guard.c)决定"要不要问你", 沙盒决定"最多能碰什么" —— 两层各管一件事, 守门放行的东西沙盒照样只给这一次点名的路径。 */
#ifndef DS4_ASST_INTERNAL_H
#define DS4_ASST_INTERNAL_H
#include "../common/ds4_json.h"
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ASST_PATH 1024

/* ---------------- 沙盒(sbx_run.c 跑子进程, sbx_linux.c / sbx_mac.c 上锁) ---------------- */
#define ASST_SBX_MAX_ROOTS 32
#define ASST_SBX_OUT_MAX (256u * 1024u)    /* 子进程交回的结果上限: 工具结果再大模型也看不完, 大文件走分页 */
#define ASST_SBX_FSIZE_MAX (256ull << 20)  /* 子进程单个文件最多写 256 MiB(RLIMIT_FSIZE): 失控的 bash 写不满盘 */
#define ASST_SBX_AS_MAX (8ull << 30)       /* Linux 每个沙盒进程虚存上限 8 GiB: 整机红线 12 GiB 余量里, 模型子进程之外的东西都得有界 */
#define ASST_SBX_NPROC_EXTRA 64            /* 沙盒里最多再开这么多进程(按"当前本用户进程数 + 64"设 RLIMIT_NPROC): fork 炸弹撞墙 */

typedef struct {
    char ro[ASST_SBX_MAX_ROOTS][ASST_PATH]; int nro;   /* 只读根(目录或单个文件), 已 realpath */
    char rw[ASST_SBX_MAX_ROOTS][ASST_PATH]; int nrw;   /* 读写根 */
    bool exec;        /* 允许 exec(bash 才开): 开了会自动加系统只读根(/usr /bin /lib /etc …), 不加用户家目录 */
    int timeout_s;    /* 墙钟超时: 到点整个进程组 SIGKILL */
    int keep_fd;      /* > 2: 这个 fd 留给子进程(make_video 的编码器从它读帧); 0 = 没有。其余继承的 fd 照样全关 */
    bool tts;         /* 要念旁白(make_video): macOS 再放行 say 用到的几个系统语音服务(sbx_mac.c); Linux 用 exec 模式的系统只读根就够 */
} asst_sbx_spec;

/* 子进程里跑的函数: 已上锁, 返回 malloc 的结果文本(NULL = 空)。只许做工具自己的事, 不碰父进程的任何共享状态(那是 fork 出来的副本)。
 * ★ud 指到的东西必须在堆上, 不能在别的线程的栈上★(10-13 spark 实撞): 子进程里 glibc 把"其他线程"(fork 后都不存在了)的栈回收进栈缓存,
 * 子进程一开线程(旁白小模型的卷积多线程)就复用那块栈, 把父进程工人线程栈上的局部变量盖掉 —— make_video 的 aud 就这么被写花, 读到垃圾指针段错误;
 * 单测里同一变量在主线程栈上(主线程栈不进缓存)永远复现不了。 */
typedef char *(*asst_sbx_fn)(void *ud);

typedef enum { ASST_SBX_OK = 0, ASST_SBX_LOCK_FAILED = -1, ASST_SBX_TIMEOUT = -2, ASST_SBX_CRASHED = -3, ASST_SBX_SPAWN_FAILED = -4 } asst_sbx_rc;
/* fork → setsid → 关掉继承的 fd → rlimit → 上锁 → fn → 结果经管道交回 → 进程组全杀。*out 给 malloc 的结果(失败时是一句原因) */
asst_sbx_rc asst_sbx_run(const asst_sbx_spec *sp, asst_sbx_fn fn, void *ud, char **out);
bool asst_sbx_add(asst_sbx_spec *sp, bool rw, const char *path);   /* realpath 后加根; 路径不存在返回 false(Landlock 只能给存在的东西立规则) */
/* 平台上锁(只在子进程里调): 成功 true; 失败把原因写 err。实现见 sbx_linux.c / sbx_mac.c */
bool asst_sbx_lock(const asst_sbx_spec *sp, char *err, size_t errn);
const char *asst_sbx_kind(void);   /* "landlock+seccomp" / "seatbelt": 写进审计与状态页 */

/* ---------------- 小 JSON 取值(asst_json.c): 工具参数与请求体都是扁平对象 ---------------- */
bool asst_jstr(const char *obj, const char *key, char **out);   /* 顶层 key 的字符串值(malloc); 没有/不是字符串 false */
bool asst_jint(const char *obj, const char *key, long *out);
bool asst_jbool(const char *obj, const char *key, bool *out);
bool asst_jraw(const char *obj, const char *key, char **out);   /* 顶层 key 的值原文(对象/数组也行) */

/* ---------------- 文件工具(tool_fs.c): 全部在沙盒子进程里执行 ---------------- */
#define ASST_READ_LINES 400            /* read_file 一次最多给这么多行: 再多模型也读不过来, 用 start 分页 */
#define ASST_LINE_MAX 2000             /* 单行截断(压缩过的 js 一行几 MB) */
#define ASST_READ_BYTES (16u << 10)    /* read_file 一次最多吐这么多字节(≈5~6k token)。只限行数不够: 400 行网页正文 82 KB, 10-11 一次读进去上下文 14 万涨到 19 万、预填几分钟 */
#define ASST_SEARCH_HITS 200
typedef struct { const char *path; long start, lines; } asst_read_args;
typedef struct { const char *path; const char *content; const char *trash; } asst_write_args;   /* trash = 覆盖前旧版本复制去的目录(工作区/.trash) */
typedef struct { const char *path; const char *old_s; const char *new_s; bool all; const char *trash; } asst_edit_args;
typedef struct { const char *root; const char *pattern; const char *glob; } asst_search_args;
char *asst_fs_read(void *a);     /* asst_read_args */
char *asst_fs_list(void *a);     /* const char *path */
char *asst_fs_write(void *a);    /* asst_write_args: 原子写(临时文件 + rename), 父目录不存在就建 */
char *asst_fs_edit(void *a);     /* asst_edit_args: old 必须恰好出现一次(all=true 时全换) */
char *asst_fs_search(void *a);   /* asst_search_args: 递归找含 pattern(子串)的行 */
typedef struct { const char *command; const char *cwd; int timeout_s; } asst_bash_args;
char *asst_bash_exec(void *a);   /* 干净环境(PATH/HOME=工作区/TMPDIR/LANG), stdin=/dev/null, 输出头 64 KB */
char *asst_fs_put_bytes(const char *path, const void *data, size_t n, const char *trash);   /* 二进制产物: 旧版进 .trash + 原子写; 成功 NULL */
bool asst_jarr_at(const char *arr, int i, char **out);   /* 数组第 i 个元素原文(malloc); 越界 false */

/* ---------------- 根目录与存储(asst_store.c): 全是你能打开的纯文本 ---------------- */
#define ASST_ID_LEN 12                 /* 会话 id = 12 个十六进制字符 */
extern char asst_root[ASST_PATH];      /* ~/.ds4/assistant */
extern char asst_ws[ASST_PATH];        /* asst_root/workspace: 助理默认的读写地盘 */
extern char asst_home[ASST_PATH];      /* 用户家目录(getpwuid, 不读 $HOME) */
extern char asst_token[65];            /* 非本机访问口令(asst_root/token, 0600) */
bool asst_init(const char *root, char *err, size_t errn);   /* root = NULL → ~/.ds4/assistant; 建目录、口令 */
bool asst_id_ok(const char *id);
void asst_spath(const char *id, const char *file, char *out, size_t n);   /* sessions/<id>/<file> */
/* system 快照的开关(会话创建时定, 随 system 冻结): PHONE = 用户在手机上(身份段多一句"屏幕小"); ENV = 工具表里有能进用户别的 app/账号的工具,
 * "设备与账号"那段记忆才有意义(照 Cursor 的 Auto Attached: 规则跟着它管的东西走; docs/assistant.md §3.5)。老调用传 bool true = PHONE, 语义不变 */
#define ASST_SYS_PHONE 1
#define ASST_SYS_ENV 2
bool asst_session_new(char id[ASST_ID_LEN + 1], const char *title, const char *first_msg, int sys_flags);   /* first_msg 定会话语言, sys_flags 见上 */
long asst_ev(const char *id, const char *json);   /* 追加一行事件(页面的唯一真相), 返回行号; 广播 asst_cv */
void asst_msg(const char *id, const char *json);  /* 追加一条 OpenAI message */
char *asst_messages(const char *id);              /* "[system, …]": system 取会话创建时冻结的 system.txt */
void asst_audit(const char *json);                /* audit.jsonl 一行(调用方给不带时间的对象体, 这里补 t) */
void asst_sessions_json(ds4_buf *b);              /* [{id,title,updated,state}] 按更新时间倒序 */
bool asst_meta_get(const char *id, const char *key, char **val);   /* meta.json 顶层字段原文 */
void asst_meta_set(const char *id, const char *key, const char *raw);   /* 改/加一个字段(值给 JSON 原文) */
char *asst_slurp(const char *path, size_t *len);  /* 整文件(NUL 结尾); 没有 NULL */
bool asst_write_file(const char *path, const char *data, size_t n, int mode);   /* 临时文件 + rename */
extern pthread_mutex_t asst_mu;        /* 守着运行状态与审批表; 事件追加也在它下面广播 */
extern pthread_cond_t asst_cv;

/* ---------------- 守门(sbx_guard.c): 一个确定性函数, 决策序固定 ---------------- */
typedef enum { ASST_ALLOW, ASST_ASK, ASST_DENY } asst_gdec;
typedef struct {
    asst_gdec dec;
    bool once_only;            /* 只能"仅这次"(bash、污点态、敏感目录): 页面不给"本会话/总是" */
    bool warn;                 /* 硬警告(rm -rf / sudo / 管道进 sh …): 卡片标红 */
    bool rw;                   /* 沙盒里这个目标要读写(否则只读) */
    char reason[256];          /* 拒绝/要问的原因, 原样给模型与页面 */
    char path[ASST_PATH];      /* 解析好的绝对路径(文件类; 网页类是 URL) */
    char target[ASST_PATH];    /* 审批作用域的目标: 目录 / 域名 */
    char cls[8];               /* read / write / exec / web / mem / pay: 审批表按类记(pay 不进审批表, 见 sbx_pay.c) */
    char extra[2048];          /* 审批卡附加 JSON(交易类: 前台页面/截图/落点/金额/命中原因); 空 = 没有 */
} asst_gres;
bool asst_path_resolve(const char *in, char *out, size_t n);   /* ~ / 相对(工作区)/ .. 都解开, 存在的前缀 realpath */
bool asst_under(const char *path, const char *root);
void asst_guard(const char *session, const char *tool, const char *args, asst_gres *r);
void asst_guard_mark(const char *session, const char *cls, const char *path);   /* 执行后记污点(读了工作区外 → priv, 上过网 → web) */
void asst_guard_grant(const char *session, const char *cls, const char *target, const char *scope);   /* session / always */
void asst_guard_spec(const asst_gres *r, bool exec, asst_sbx_spec *sp);   /* 这次调用的沙盒: 工作区 rw + policy 目录 + 这次点名的目标 */
char *asst_policy_json(void);          /* policy.json 原文(没有给缺省) */
bool asst_policy_op(const char *body, char *err, size_t errn);   /* 页面改 policy / 撤销永久授权 */
void asst_approvals_json(ds4_buf *b);
bool asst_secret_like(const char *s);  /* 卡号(15~19 位且过 Luhn)/ 私钥头 / password= / 本机口令原文 */
bool asst_url_host(const char *url, char *host, size_t n);   /* http(s) 地址的主机名(sbx_guard.c; user:pass@ 跳过, [v6] 去括号); 不是 http(s) 回 false */
bool asst_host_private(const char *host);                    /* 本机 / 内网 / 链路本地(sbx_policy.c) */

/* ---------------- 交易审批(sbx_pay.c, 2026-10-16, 方案稿 docs/assistant_approval.md): 云手机上会动手的动作前判"这是不是付款/下单/登录" ---------------- */
typedef struct { char focus[192], hash[32]; bool secure, ime_shown; int ime_top; } asst_phone_win;   /* 前台: 焦点窗口(包/类全名)、窗口编号(mCurrentFocus 里的; 没焦点窗口时空)、FLAG_SECURE、软键盘 */
typedef struct {
    bool ok;                   /* 读到了手机此刻的前台窗口(没读到 → 守门拒, 不放行) */
    bool deny;                 /* 安全窗口上打字: 直接拒(密码不经模型) */
    bool hit;                  /* 交易: 问用户, 只能"仅这次" */
    bool secure, unseen;       /* 焦点窗口禁截屏 / 模型上一帧不是这块屏 */
    char why[256], app[192], text[96], amounts[160], shot[64];   /* 原因 / 前台页面 / 命中的字 / 金额 JSON 数组 / 现场截图(工作区相对) */
    char win[32];              /* 焦点窗口编号("这一页已批"按它认) */
    char prev_shot[64], prev_app[192]; long prev_age;   /* 安全窗口截不到图时: 进它之前的最后一屏 / 那一屏的页面 / 几秒前 */
    int x, y; long age;        /* 落点(没有 = -1) / 模型上一帧是几秒前(没有 = -1) */
} asst_paysig;
const char *asst_phone_win_cmd(void);   /* 读前台的那条 dumpsys(守门与 phone_tap 判键盘共用) */
bool asst_phone_win_parse(const char *win, asst_phone_win *w);   /* 那条命令的输出 → 前台; 没有焦点窗口 false */
bool asst_phone_win_now(asst_phone_win *w, int max_age_ms, char *err, size_t errn);   /* 读手机; max_age_ms > 0 时这么新的上一次结果直接用(守门刚读过, 省一次往返) */
bool asst_pay_relevant(const char *tool, const char *args);   /* phone_tap / phone_swipe / phone_type / phone_key enter */
bool asst_pay_judge(const char *tool, const char *args, const char *win, const char *fxml, const char *ffocus, long fage, asst_paysig *s);   /* 纯函数: 喂 dumpsys 文本 + 上一帧(单测用真机夹具) */
void asst_pay_check(const char *tool, const char *args, asst_paysig *s);   /* 守门用: 读手机 + 判; 命中且能截屏时现场截一张进 .phone/ */
void asst_pay_extra(const asst_paysig *s, char *out, size_t n);           /* → 审批卡附加 JSON {app, secure, screen, at, hit, amounts, frame_age_s, unseen, win, before} */
void asst_pay_approved(const char *sid, const char *extra);               /* 用户在交易卡上点了"仅这次": 安全窗口那张记下"这一页已批" */
bool asst_pay_window_ok(const char *sid, const char *win);                /* 这个会话在这个窗口上已经批过 */

/* ---------------- 官方产出工具(tool_office.c, 方案稿 docs/assistant_office.md): write_document / write_xlsx, 在沙盒子进程里跑 src/office 的写出器 ---------------- */
bool asst_office_tool(const char *name);
char *asst_office_exec(const char *tool, const char *args, const asst_gres *g, const char *trash, int *ok);   /* 已过守门(g->path = 产物绝对路径) */
char *asst_image_exec(const char *args, const asst_gres *g, const char *trash, int *ok);
char *asst_image_view(const char *args, const asst_gres *g, const char *trash, int *ok);   /* view_image(tool_image.c): 只读, 一块放大 + 网格落 .img/ */   /* edit_image(tool_image.c): 已过守门(g->path = 源图绝对路径); 成品 out 默认 <源>-edited.png */

/* ---------------- 记忆(asst_memory.c) ---------------- */
char *asst_system_prompt(const char *first_msg, int sys_flags);   /* 身份段(+手机段) + 规矩 + 关于用户 + 怎么做事 + (ENV 时)设备与账号 + 现在 + 语言句; 中/英按第一条消息选; 会话创建时拍快照 */
bool asst_lang_is_zh(const char *text);
/* 记忆三类(照 ChatGPT 的 Model Set Context / Response Preferences / 设备元数据分段, docs/assistant.md §3.5): kind = fact → profile.md, habit → habits.md, env → env.md */
char *asst_remember(const char *session, const char *kind, const char *text);   /* kind 空 = fact; 不认识的 kind 回 error */
char *asst_forget(const char *text);   /* 三个文件都删 */
void asst_memory_json(ds4_buf *b);     /* {profile, habits, env, rules, workspace, root} */
bool asst_memory_save(const char *body, char *err, size_t errn);

/* ---------------- 工具(asst_tools.c): 一张注册表(总纲 docs/assistant_arch.md §2) + 浏览器(tool_browser.c) ---------------- */
typedef struct { const char *caps, *apps, *task; bool plan; } asst_tctx;   /* 这条消息的上下文: 页面报的设备能力(逗号表) + 能打开的 app(逗号表) + 任务 id(任务会话才有, 否则 NULL) + 这个任务带不带步骤(plan_step 才进表); 工具表按它筛 */

/* ---------------- 任务(asst_task.c, 2026-10-14): 一句话 + 专属会话 + 下次跟进时间; 调度线程到点往会话里发"继续这个任务", 周期由模型用 task_check_in 定 ---------------- */
void asst_task_start(void);                                   /* ds4-train 起时: 建 tasks/ 并开调度线程 */
bool asst_task_of(const char *sid, char tid[ASST_ID_LEN + 1]);   /* 这个会话是任务的吗(meta.task) */
char *asst_task_fact_line(const char *tid, bool zh);          /* "[任务: 第 N 次跟进 · 创建于 … · 上次结束 … · 你上次留的话: …]"(malloc); 放在用户文本前, 页面不显示 */
bool asst_task_tool(const char *name);                        /* task_check_in / task_done */
bool asst_task_build_checkin(const asst_tctx *c, ds4_buf *b);   /* 两个工具的 schema: 只在任务会话里进工具表 */
bool asst_task_build_done(const asst_tctx *c, ds4_buf *b);
char *asst_task_tool_exec(const char *sid, const char *tool, const char *args, int *ok);   /* 主进程直接改任务文件, 不进沙盒 */
void asst_task_run_end(const char *tid, const char *end, const char *last_text);   /* 运行收尾: 次数、上次进展、没定下次就一小时后 */
bool asst_task_add(const char *text, char tid[ASST_ID_LEN + 1], char *err, size_t errn);   /* 建任务 + 会话 + 立刻跑第一次 */
bool asst_task_op(const char *body, char *err, size_t errn);  /* 页面: {op: add|pause|resume|done|run|delete, id, text} */
void asst_tasks_json(ds4_buf *b);                             /* {follow_s, tasks:[{id, session, state, end, created, updated, next, runs, text, last, note, run_state}]} 没完的在前 */
bool asst_task_due(long now, char tid[ASST_ID_LEN + 1], char sid[ASST_ID_LEN + 1], char *text, size_t tn);   /* 调度用: 一个到期且会话空闲的任务 */
bool asst_task_info(const char *tid, char *state, size_t sn, char *text, size_t tn, long *done_at);   /* 状态 + 那句话(截到 tn) + 模型调 task_done 的时刻(没调过 = 0; 可 NULL) */
/* 任务文件的一条记录。计划(asst_plan.c)就是带 steps 的任务, 和一句话任务同一个文件、同一张列表、同一条调度线程。
 * state: proposed(计划等用户按"开始": 调度不叫, 守门只许看) | active | paused | done。summary/steps/questions 是 JSON 原文(steps/questions 为数组, 没有 = NULL)。
 * 读-改-写必须拿着 asst_task_lock(工人线程的工具/收尾、调度线程、接口线程三方都来); 拿着它不许调 asst_run_*(那边拿 asst_mu)。 */
typedef struct { char id[ASST_ID_LEN + 1], session[ASST_ID_LEN + 1], state[12], end[12]; char *text, *last, *note, *summary, *steps, *questions; long created, updated, next, runs, done_at; } asst_task_rec;
void asst_task_lock(void);
void asst_task_unlock(void);
bool asst_task_read(const char *tid, asst_task_rec *t);       /* 拿着锁 */
bool asst_task_write(asst_task_rec *t);                       /* 拿着锁; 顺手改 updated */
void asst_task_free(asst_task_rec *t);
bool asst_task_create(const char *sid, const char *text, const char *state, char tid[ASST_ID_LEN + 1], char *err, size_t errn);   /* 给已有会话建任务文件 + meta.task, 不提交运行 */

/* ---------------- 计划(asst_plan.c, 2026-10-15, 方案稿 docs/assistant_plan.md): 进不进计划由模型调不调 plan_propose 决定, 提示词零改动 ---------------- */
bool asst_plan_tool(const char *name);                        /* plan_propose / plan_step */
bool asst_plan_build_propose(const asst_tctx *c, ds4_buf *b); /* 所有对话都有 */
bool asst_plan_build_step(const asst_tctx *c, ds4_buf *b);    /* 只在带步骤的任务会话里有 */
char *asst_plan_tool_exec(const char *sid, const char *tool, const char *args, int *ok);
bool asst_plan_has(const char *tid);                          /* 这个任务带步骤 */
bool asst_plan_waiting(const char *sid);                      /* 这段对话是个等用户按"开始"的计划(守门据此只许看) */
void asst_plan_event(const char *tid);                        /* 往它的会话写一条 plan 事件(页面计划卡的唯一来源) */
void asst_plan_fact(const asst_task_rec *t, bool zh, ds4_buf *b);   /* "[计划: … · 步骤 2/6 完成 · 下一步: … · 到点: … · 未答: …]"; 没步骤什么也不写 */
long asst_plan_next_at(const char *steps, long now);          /* 最早一个没做完、带时间、还在未来的步骤的时刻; 没有 = 0 */
const char *asst_plan_start_text(const char *title);          /* 按"开始"后发进对话的那句(语言跟标题) */

/* ---------------- 推送(asst_push.c, Web Push; 密码学在 src/push) + 通知规则(asst_notify.c), 2026-10-14 ---------------- */
void asst_push_init(void);                     /* push/vapid.json 没有就生成; ds4-train 起时调 */
const char *asst_push_key(void);               /* VAPID 公钥 base64url(页面订阅用); 没初始化 = "" */
void asst_push_json(ds4_buf *b);               /* {key, proxy, devices:[{id, label, origin, created}]} */
bool asst_push_op(const char *body, char *err, size_t errn);   /* {op: subscribe|unsubscribe|test|proxy, sub, label, origin, endpoint, id, lang, url} */
int asst_push_devices(void);                   /* 开了通知的设备数 */
int asst_push_send_sync(const char *title, const char *body, const char *url, int *held);   /* 推给全部设备, 返回送达数(2xx); 正看着页面的设备不推, 压进 push/held/(*held 计数, 可 NULL); 404/410 的订阅当场删 */
void asst_push_send(const char *title, const char *body, const char *url);       /* 异步版: 起线程, 立刻返回 */
void asst_push_flush(long now);                /* 不再看页面的设备: 把压着的补推出去 */
void asst_push_sub_id(const char *endpoint, char id[17]);   /* 订阅 endpoint → 订阅 id(sha256 前 16 位 hex); 空 endpoint → "" */
bool asst_notify_tool(const char *name);       /* notify */
bool asst_notify_build(const asst_tctx *c, ds4_buf *b);
char *asst_notify_tool_exec(const char *sid, const char *args, int *ok);
void asst_reminders_json(ds4_buf *b);          /* {reminders:[{id, at, created, session, title, text}]} 按时间 */
bool asst_reminder_op(const char *body, char *err, size_t errn);   /* {op: delete, id} */
int asst_notify_tick(long now, long *next);    /* 到点的提醒发出去; 返回发了几条; *next = 下一条的到点时刻(0 = 没有, 可 NULL) */
void asst_notify_start(void);                  /* ds4-train 起时: 开通知线程(到点提醒 + 补推压着的) */
void asst_notify_wake(void);                   /* 叫醒通知线程: 新排了提醒 / 页面报走了 */
void asst_notify_run_end(const char *sid, const char *end, const char *last_text, double secs);   /* 运行收尾: 任务做完 / 跑了很久 → 推一条 */
void asst_notify_approval(const char *sid, const char *tool, const asst_gres *g);   /* 弹了审批卡: 推"需要你批准: …"(正看着页面的设备照在场规则压着) */
bool asst_time_parse(const char *s, time_t *out);   /* 本机时区 "2026-10-20 07:30[:15]"(空格或 T); 解析不了 false */
void asst_presence_set(const char *client, bool visible, const char *endpoint);   /* 页面报"我在看/我走了"; endpoint = 它所在浏览器的推送订阅(没订阅给 NULL/"") */
bool asst_presence_watching(const char *sub, long now);    /* 这台设备(订阅 id)有页面正开着在看(ASST_PRESENCE_S 内报过 visible) */

/* ---------------- 技能(src/skill/skill.c, 方案稿 docs/assistant_skills.md): skills/<名>/SKILL.md + 随附文件; 工具只读, 页面"技能"tab 增删改 ---------------- */
#define ASST_SKILL_NAME_MAX 64
#define ASST_SKILL_MD_MAX (64u << 10)   /* use_skill 一次给模型的正文上限(≈2 万 token); 更长的指路 read_file 分页 */
#define ASST_SKILL_FILES 200            /* 文件表最多列这么多条 */
extern char asst_skills[ASST_PATH];     /* asst_root/skills(asst_init 建) */
bool asst_skill_name_ok(const char *name);
int asst_skill_count(void);
bool asst_skill_build_tool(const asst_tctx *c, ds4_buf *b);   /* use_skill 的 schema: 描述里列已装技能(名字: 描述), name 枚举; 一个没装 → false(工具表里没这行) */
char *asst_skill_load(const char *name);                       /* use_skill 的执行: 文件夹路径 + 文件表 + 正文(malloc) */
void asst_skills_json(ds4_buf *b);                             /* 页面: {root, skills:[{name, enabled, description, md, files:[{path,size}]}]} */
bool asst_skill_op(const char *body, char *err, size_t errn);  /* 页面: {op: save|delete, name, md} / {op: enable, name, on}; save 先过 asst_skill_check */
bool asst_skill_enabled(const char *name);                     /* 装着且开着(文件夹里没有 .disabled); 关掉的不进 use_skill 工具表、点名也拒, 它专属的工具(market_data 等)一并消失 */
bool asst_skill_set_enabled(const char *name, bool on);
bool asst_skill_check(const char *text, char *why, size_t n); /* ★技能不许带脚本★: 围栏代码块 / #! / <script> / 命令行 / 行内代码里的命令或脚本文件名 → false + 第几行什么 */
char *asst_skill_put_file(const char *name, const char *rel, const char *data, size_t n);   /* 页面上传随附文件; NULL = 成功, 否则原因(malloc) */
int asst_skills_seed(const char *src_root, char *msg, size_t msgn);   /* 仓库自带技能(src_root/<名>/)默认装上, 账本 skills.seeded 管升级/删除(skill_seed.c); 返回装/升级了几个 */
/* ---------------- 连接器(asst_conn.c + tools/tool_db.c, 方案稿 docs/assistant_connectors.md): connectors/<名>.json + secrets/conn_<名>; 工具 query_db 只读 ---------------- */
#define ASST_CONN_NAME_MAX 24   /* 密码的 secret 名是 conn_<名>, secrets 名字上限 32 */
typedef struct { char name[ASST_CONN_NAME_MAX + 1], kind[16], host[256], db[128], user[128], auth_db[64], path[ASST_PATH]; int port; bool enabled; long updated; } asst_conn;
bool asst_conn_name_ok(const char *name);
bool asst_conn_load(const char *name, asst_conn *c);          /* 读 json; 不存在或种类不认 false */
bool asst_conn_enabled(const char *name);
bool asst_conn_password(const char *name, char *out, size_t n);   /* secrets/conn_<名>; 调用方用完清零 */
bool asst_conn_build_tool(const asst_tctx *c, ds4_buf *b);   /* query_db 的 schema: 开着的连接器枚举 + 用到的种类的 what 并集; 一个没开 → false */
void asst_conns_json(ds4_buf *b);                              /* 页面: {kinds:[{kind,label,port,fields}], connectors:[{name,kind,host,port,db,user,auth_db,path,has_password,enabled,updated}]}(没有 password) */
bool asst_conn_op(const char *body, char *err, size_t errn);   /* {op: save|enable|delete, …} */
bool asst_conn_test(const char *body, ds4_buf *out);           /* 测试连接 → {ok,line,ms} / {error} */
char *asst_db_exec(const char *sid, const char *args, int *ok);   /* query_db: 主进程直连 TCP(sqlite 在沙盒子进程跑 sqlite3); 头 16 KB 给模型, 全量落工作区 db/ */
bool asst_sqlite_probe(const char *path, char *line, size_t n, char *err, size_t errn);   /* 沙盒里 sqlite3 -readonly 开文件: 版本 + 表数 */
bool asst_toolbox_has(const char *prog);   /* sbx_toolbox.c: 沙盒 PATH 里有没有这个程序 */
char *asst_tools_json(const asst_tctx *ctx);   /* OpenAI tools 数组(malloc): 注册表里 avail 为真的行 */
bool asst_tools_reach(const asst_tctx *ctx);   /* 工具表里有没有能进用户别的 app/账号的工具(share / open_app / 云手机): 决定 system 带不带"设备与账号"记忆 */
char *asst_tool_args_error(const asst_tctx *ctx, const char *tool, const char *args);   /* 参数不合这一轮的 schema → "error: invalid arguments …"(malloc); 合规 NULL */
char *asst_tool_exec(const char *session, const char *tool, const char *args, const asst_gres *g, int *ok);   /* 已过守门; 在沙盒里跑 */
char *asst_tool_post(const char *session, const char *tool, const char *result);   /* 设备工具交回结果后的收尾(locate → 反查地址 + 记进会话); 返回新结果(malloc)或 NULL = 原样 */
const char *asst_toolbox_desc(void);   /* sbx_toolbox.c: run 工具的描述(启动探测本机装了哪些程序, 一次) */
void asst_toolbox_json(ds4_buf *b);    /* {"have":[…],"missing":[…]} 给状态接口 */
#ifdef __APPLE__
#define ASST_SBX_PATH "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Applications/LibreOffice.app/Contents/MacOS"
#else
#define ASST_SBX_PATH "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
#endif
char *asst_web_search(const char *query, char *err, size_t errn);
char *asst_web_visit(const char *url, char *err, size_t errn);
char *asst_web_text(const char *url, char *err, size_t errn);   /* 页面正文原文(document.body.innerText): 给 JSON 接口用, 不走正文抽取 */
char *asst_web_js(const char *url, const char *js, bool await, char *err, size_t errn);   /* 开页面后跑一段 JS 拿字符串; await = 表达式回 Promise 就等它(fetch_image 的 canvas 转码) */
const char *asst_browser_kind(void);   /* "firefox" / "" (本机没有) */
void asst_browser_shutdown(void);
/* 渲染原语(make_video 用): 独立标签, w×h 视口, 预载脚本只挂这个标签 */
bool asst_ff_tab_open(int w, int h, const char *preload_fn, char *ctx, size_t ctxn, char *script, size_t sn, char *err, size_t errn);
char *asst_ff_call(const char *method, const char *params, char *err, size_t errn);   /* BiDi 命令 → result 原文(malloc); 失败 NULL, 读超时会杀掉火狐 */
void asst_ff_tab_close(const char *ctx, const char *script);

/* ---------------- 云手机(方案稿 docs/assistant_phone.md): sbx_adb.c = adb 线协议客户端(直连手机里的 adbd, 不装 adb 命令) ---------------- */
#define ADB_OUT_MAX (64u << 20)   /* 一条 exec 的输出上限 64 MB: 截图 PNG ~1 MB, 这是防对端发疯 */
typedef struct { int fd, timeout_ms; uint32_t next_id, maxdata; char banner[256]; } adb_conn;
typedef bool (*adb_sink)(const void *data, size_t n, void *ud);   /* 返回 false = 不要了(流就此关掉) */
bool adb_connect(adb_conn *c, const char *ip, int port, int timeout_ms, char *err, size_t errn);
void adb_close(adb_conn *c);
bool adb_stream(adb_conn *c, const char *service, const void *in, size_t inlen, adb_sink sink, void *ud, char *err, size_t errn);   /* 一条服务流: in = 喂给 stdin 的字节(可 NULL), 输出逐包给 sink */
bool adb_exec(adb_conn *c, const char *cmd, const void *in, size_t inlen, ds4_buf *out, char *err, size_t errn);   /* "exec:<cmd>" 收满到 out */
char *adb_shell(const char *ip, int port, const char *cmd, int timeout_ms, char *err, size_t errn);   /* 连-跑-关, 文本结果(malloc, 去尾换行); 失败 NULL */

/* sbx_phone.c: 危险模式开关(~/.ds4/studio/danger) + 手机生命周期(sudo -n <仓库根>/ds4-phone start|stop|status) + 内存门 */
#define ASST_PHONE_IP "172.31.250.2"     /* ds4-phone 的固定布局(src/sandbox/phone/phone_internal.h) */
#define ASST_PHONE_PORT 5555
#define ASST_PHONE_MIN_AVAIL_MB 3072     /* 起手机前: MemAvailable − 手机最小顶 ≥ 这个数(和火狐同一条尺; = src/sandbox/phone PH_MEM_HEADROOM_MB) */
#define ASST_PHONE_MIN_CAP_MB 2048       /* 手机的最小内存顶(= src/sandbox/phone PH_MEM_MIN_MB); 真正的顶由 ds4-phone 起时按宿主可用内存定, 状态里 cap_mb 是它 */
typedef struct {
    bool supported, prepared, enabled, running, adb;
    char state[16];                      /* off | starting | ready | stopping | error */
    char err[256];
    long mem_mb, peak_mb, cap_mb, mem_avail_mb, uptime_s;
    int pids, width, height, viewers;
} asst_phone_st;
void asst_phone_init(void);              /* ds4-train 起时: 开关开着就接上已在跑的 / 拉起; 关着就把残留停掉 */
void asst_phone_status(asst_phone_st *st);
bool asst_phone_enable(char *err, size_t errn);
bool asst_phone_disable(char *err, size_t errn);
bool asst_phone_on(void);                /* 开关开 && 手机 ready: 工具表与守门都看它 */
char *asst_phone_shell(const char *cmd, int timeout_ms, char *err, size_t errn);
bool asst_phone_screencap(ds4_buf *png, char *err, size_t errn);
bool asst_phone_install(const void *apk, size_t n, char *pkg, size_t pn, char *err, size_t errn);
const char *asst_phone_apps(void);       /* 逗号表: 已装第三方包 + 常用系统 app; 手机 ready / 装卸后刷新 */
void asst_phone_apps_refresh(void);
void asst_phone_one_app_sh(ds4_buf *b, const char *keep);   /* 一次只开一个 app: 除 keep(NULL = 前台那个)外有进程的第三方包全 force-stop; 接在动作命令后面 */
void asst_phone_open_cmd(ds4_buf *b, const char *pkg);      /* 开 app 的整条命令 = 清场 + am start(工具与 REST 共用这一份) */
/* sbx_phone_video.c: screenrecord 的 H.264 裸流 → 每个看客一条 HTTP 连接(接管 fd); 没人看 5 s 后停录 */
void asst_phone_video_attach(int fd);
int asst_phone_video_viewers(void);
size_t asst_phone_nal_split(const unsigned char *p, size_t n, size_t *starts, size_t max);   /* 起始码位置表(最后一个可能不完整, 调用方自己留尾); 单测用 */
/* tool_phone.c: 模型的六个 phone_* 工具 */
bool asst_phone_tool(const char *name);
bool asst_phone_build_open(const asst_tctx *c, ds4_buf *b);   /* phone_open 的 schema(app 枚举随手机变) */
char *asst_phone_tool_exec(const char *sid, const char *tool, const char *args, int *ok);
char *asst_phone_ui_text(const char *xml);   /* uiautomator 的 XML → 给模型看的元素表(单测用) */
char *asst_phone_ui_json(const char *xml);   /* → REST 的 {elements:[…]} */
int asst_phone_hit(const char *xml, int x, int y, char *text, size_t n);   /* 包含 (x,y) 的最小带字元素的文字(text, 没有用 content-desc); 落在没字的地方 0 */
char *asst_phone_texts(const char *xml);     /* 全部元素的文字, 一行一个(malloc) */
void asst_phone_last_shot(char *rel, size_t rn, char *focus, size_t fn, long *t);   /* 模型最后一次截成功的图(工作区相对)、那时的页面、时刻; 没有 rel 空 */
void asst_phone_frame(char **xml, char *focus, size_t fn, long *t);   /* 模型上一次看到的那一帧(look() 存): 控件树副本(malloc, 没有 NULL)、那时的焦点窗口、时刻(没有 0) */
/* tool_phone_wx.c: 微信写死的按钮表(微信不给 uiautomator 控件树) */
const char *asst_phone_wx_cmd(void);   /* 一条 shell: 版本号 + 弹出窗口数 + 微信的 View 树 */
char *asst_phone_wx_xml(const char *dump, const char *focus, const uint8_t *rgb, int w, int h, char *note, size_t nn);   /* → uiautomator 格式 XML(malloc); rgb = 截图(可 NULL); 版本不对/认不出这页 NULL, note 说原因 */

/* ---------------- 视频(tool_video.c + tool_video_srv.c, 方案稿 docs/assistant_video.md) ---------------- */
char *asst_video_exec(const char *args, const asst_gres *g, const char *trash, int *ok);   /* 已过守门(g->path = .mp4 绝对路径) */
char *asst_ff_eval(const char *ctx, const char *expr, bool await, char *err, size_t errn);   /* 火狐页面里跑一句 JS, 回 value 原文(tool_video.c) */
size_t asst_b64dec(const char *s, uint8_t *o);                                               /* base64 → 字节(截图数据) */
char *asst_render_exec(const char *args, const asst_gres *g, const char *trash, int *ok);   /* render_image(tool_render.c): HTML → 火狐截图 → PNG 落工作区; 已过守门(g->path = .png 绝对路径) */
asst_sbx_rc asst_save_bytes(const asst_gres *g, const char *trash, const uint8_t *data, size_t len, char **out);   /* 字节落到 g->path(沙盒子进程; 建父目录; 旧版进 .trash); *out = created / replaced / error: … */
char *asst_fetch_exec(const char *args, const asst_gres *g, const char *trash, int *ok);    /* fetch_image(tool_fetch.c): 网上找一张照片 / 一个图标, 火狐转码后落工作区 */
typedef struct asst_vsrv asst_vsrv;   /* 渲染期间的本机只读 HTTP 源: 页面从内存给, 别的只给工作区文件 */
asst_vsrv *asst_vsrv_start(const char *page, size_t n, char *url, size_t urln, char *err, size_t errn);   /* url = 页面地址 */
void asst_vsrv_stop(asst_vsrv *s);
void asst_run_progress(const char *text);   /* asst_run.c: 给正在执行的那张工具卡发一条进度(tool_progress 事件) */
bool asst_mkdir_p(const char *dir);   /* 逐级建目录(已存在也算成功) */
void asst_fs_trash(const char *trash, const char *path);   /* 旧版复制进 .trash(覆盖前保全) */
/* tool_video_audio.c: make_video 的旁白 + 背景乐 → AAC 音轨 */
typedef struct { double at; char *text; } asst_vline;
typedef struct { asst_vline *lines; int n; char music[ASST_PATH]; } asst_vaudio;
typedef struct { ds4_buf frames; size_t *off, nf; long samples; int bitrate; } asst_aac_track;   /* 第 i 帧 = frames[off[i] .. off[i+1]) */
bool asst_vaudio_parse(const char *args, asst_vaudio *a, char *err, size_t errn);   /* 父进程: narration / music 参数 → 结构; music 必须是工作区里的 .wav */
void asst_vaudio_free(asst_vaudio *a);
bool asst_vaudio_build(const asst_vaudio *a, double seconds, const char *tmpdir, asst_aac_track *t, ds4_buf *note, char *err, size_t errn);   /* 子进程: 合成 + 编码; note 收给模型看的说明 */
void asst_aac_track_free(asst_aac_track *t);
/* tool_tts.c: 旁白(只在沙盒子进程里调) */
extern char asst_repo[ASST_PATH];   /* 仓库根(发布包 = 解压目录): 随包走的资源(gguf/tts 旁白小模型)从这里找; ds4-train 起时 asst_set_repo(tr_root), 没设 = 当前目录 */
void asst_set_repo(const char *root);
/* 旁白小模型(Piper VITS 转的 GGUF, 各 63 MB): 中文 huayan / 英文 lessac。找两处: 源码树的 gguf/tts/<名>, 以及工作台"模型"页用 install.sh
 * 下到 gguf/hub/model/tts/<名>(HF 发布仓里就放在 tts/ 下, install.sh 的"除分片外全下"顺手带回来)。都没有就退回 espeak 共振峰出声。 */
#define ASST_TTS_ZH_MODEL "zh-huayan.gguf"
#define ASST_TTS_EN_MODEL "en-lessac.gguf"
const char *asst_tts_model(const char *name);   /* ASST_TTS_*_MODEL → 本机全路径; 没有 → "" */
const char *asst_tts_kind(void);   /* "say" / "espeak-ng" / "vits+espeak-ng"(Linux 有小模型) / "" = 本机念不了 */
int16_t *asst_tts(const char *text, const char *tmpdir, size_t *n, char *voice, size_t vn, char *err, size_t errn);   /* 22050 Hz 单声道 16 位(malloc) */
#ifndef __APPLE__
char *asst_tts_phonemes(const char *text);   /* 单测用: espeak 实际念出的 IPA 音标(malloc), 查"念的是不是这门语言" */
char *asst_tts_ids(const char *text);        /* 单测用(Linux 小模型路): 送进模型的音素流(malloc), 句间 " | "; 查标点/分句有没有进模型 */
#endif

/* ---------------- 设备(asst_device.c): 设备事实行 + 设备工具通道(总纲 §3) ---------------- */
char *asst_device_line(const char *device_json, const char *caps, bool zh, const char *sid);   /* "[设备: 手机 · 时间 · 时区 · 可用: … · 上次定位: …]"; 只放事实; sid 可 NULL(不查上次定位) */
char *asst_attach_line(const char *files, bool zh);   /* files = 换行分隔的工作区相对路径(页面上传进 inbox/ 的); 每个存在的文件一行 "[附件: inbox/x (N KB)]"; 没有 → NULL */
/* asst_image.c(2026-10-13): 图片附件 → OpenAI image_url 分块(data: URL), 模型真正看到像素 */
bool asst_image_part(const char *rel, ds4_buf *b);                                 /* 一个工作区路径 → ",{image_url…}" 追加; 不是图/太大 → false */
int asst_image_parts(const char *files, ds4_buf *b);                               /* 附件表里所有图片; 返回张数 */
void asst_content_with_images(ds4_buf *out, const char *text, const char *files);  /* content: 没图 = 字串, 有图 = [text 块, image_url…] */
bool asst_device_request(const char *sid, const char *client, const char *call_id, const char *kind, const char *args, char **out);   /* 写 device 事件, 等页面答复或停止; 返回 true = 设备做成了 */
bool asst_device_answer(const char *sid, const char *call_id, bool ok, const char *data, size_t n, char *err, size_t errn);
bool asst_device_waiting(void);        /* asst_mu 下读 */
bool asst_run_stopping(void);          /* 当前运行被按了停止(asst_mu 下读) */
int asst_device_tool(const char *name);   /* 0 = 不是设备工具; 1 = 要等设备交结果(定位/剪贴板/选文件); 2 = 交出去的动作(分享/打开/导航/拨打/复制/开 app), 页面出按钮, 不等 */

/* ---------------- 地点(tool_geo.c): 高德 Web 服务, key 在 secrets/amap_key ---------------- */
bool asst_secret_get(const char *name, char *out, size_t n);
bool asst_secret_set(const char *name, const char *value, char *err, size_t errn);   /* 空值 = 删 */
bool asst_geo_ready(void);             /* 反查地址可用: 本机有火狐(不要 key: 走高德网页版自用接口) */
bool asst_geo_nearby_ready(void);      /* 周边 POI 可用: 有 key 且有火狐 */
char *asst_geo_regeo(double lat, double lng, char *err, size_t errn);   /* 坐标 → 一句地址(malloc); 失败 NULL */
char *asst_geo_nearby(double lat, double lng, const char *addr, const char *query, long radius, char *err, size_t errn);   /* 没 key 时按 addr 降级成网页搜索 */
char *asst_geo_parse_regeo(const char *json, char *err, size_t errn);   /* 两个解析器单独露出来给单测喂夹具 */
char *asst_geo_parse_around(const char *json, char *err, size_t errn);
char *asst_geo_parse_regeo_free(const char *json, char *err, size_t errn);
bool asst_geo_loc_get(const char *sid, double *lat, double *lng, long *acc, char *addr, size_t n, long *t);   /* 会话里记住的上次定位(meta.loc); 没有 false */
char *asst_geo_locate_post(const char *sid, const char *raw);   /* locate 工具的收尾: 设备交回 {lat,lng,acc} → 反查地址 → 记进 meta.loc → 给模型看的文本 */

/* ---------------- 行情(tool_market.c): market_data, 东方财富公开接口, 经火狐取 ---------------- */
char *asst_market_exec(const char *args, int *ok);
bool asst_mkt_secid(const char *s, char *out, size_t n);   /* 1.600519 / sh600519 / 600519 → secid; 名字、港美股代码 false(走代码搜索) */
char *asst_mkt_parse_suggest(const char *json, char *first, size_t fn, char *first_name, size_t nn);   /* 下面五个解析器露出来给单测喂夹具; 认不出 NULL */
char *asst_mkt_parse_quote(const char *json, bool full);
char *asst_mkt_parse_kline(const char *json, const char *period_cn);
char *asst_mkt_parse_fin(const char *json);
char *asst_mkt_parse_news(const char *text);
char *asst_mkt_parse_notices(const char *json, const char *code);
void asst_mkt_set_fetch(char *(*fn)(const char *url, char *err, size_t errn));   /* 单测换成读夹具; NULL = 回到火狐 */

/* ---------------- 彩票(tool_lottery.c 开奖 / tool_lottery_match.c 竞彩): lottery_data, 体彩中心 + 中彩网官方接口, 火狐开官网页面后在页面里 fetch ---------------- */
char *asst_lottery_exec(const char *args, int *ok);
char *asst_lot_jc(bool results, const char *sport, const char *league, const char *date, long days, char *err, size_t errn);   /* 竞彩在售 / 赛果 */
char *asst_lot_fetch(const char *page, const char *api, char *err, size_t errn);   /* 在 page 里 fetch(api) 的原文 */
char *asst_lot_unexpected(const char *site, const char *raw);   /* 解析不了 → error 文本(认得出防火墙拦截页) */
bool asst_lot_num(const char *s, double *v);                     /* "1,234.5" → 数; "---"/空 false */
char *asst_lot_parse_spt(const char *json, const char *game);   /* 下面四个解析器露出来给单测喂夹具; 认不出 NULL。game = ssq/dlt/… 或中文名 */
char *asst_lot_parse_cwl(const char *json, const char *game);
char *asst_lot_parse_jc_matches(const char *json, bool basketball, const char *league);
char *asst_lot_parse_jc_results(const char *json, bool basketball, const char *league);
void asst_lot_set_fetch(char *(*fn)(const char *page, const char *api, char *err, size_t errn));   /* 单测换成读夹具; NULL = 回到火狐 */

/* ---------------- 交付(asst_file.c): 工作区文件 → 页面下载/分享; 设备传上来的文件 → 工作区 inbox/(方案稿 §13.2) ---------------- */
#define ASST_FILE_MAX (64u << 20)      /* 单个文件 64 MiB: 与工作台上传料上限同 */
bool asst_qget(const char *query, const char *key, char *out, size_t n);   /* 查询串取一项(百分号解码) */
int asst_file_get(const char *path, ds4_buf *out, const char **ctype);     /* 200 + 文件体; 工作区外 403; 没有 404 */
char *asst_file_mentioned(const char *answer);   /* 终稿里写出的、工作区里真有的文件 → JSON 数组(工作区相对路径); 一个没有 → NULL。运行结束发 files 事件, 页面出交付卡 */
char *asst_file_save(const char *name, const char *data, size_t n);        /* 存进 workspace/inbox/<净化过的名字>, 返回给模型看的一句话 */

/* 校验相(做 → 核 → 答)的账本、事实消息与 view_file: 另一个头 asst_verify.h(2026-10-23, 方案稿 docs/assistant_verify.md) */

/* ---------------- 网页预览(asst_preview.c, 方案稿 docs/assistant_page.md): GET /p/<口令>/<路径>, CSP sandbox 不透明源 + 封出网 ---------------- */
#define ASST_PV_MAX 64                 /* 口令表上限: 满了挤最老的 */
#define ASST_PV_TTL_S 3600L            /* 口令 60 分钟过期; 页面每次打开预览重新要 */
const char *asst_web_ctype(const char *path);   /* 按扩展名的网页资源类型(预览路与 make_video 渲染源共用) */
bool asst_preview_path(const char *path);       /* "/p/" 前缀 */
bool asst_preview_issue(const char *path, long now, char *url, size_t un, char *err, size_t errn);   /* 工作区里的 .html → "/p/<口令>/<文件名>" */
int asst_preview_get(const char *method, const char *path, const char *host, long now, ds4_buf *hdr, ds4_buf *body);   /* 纯判定 + 取字节; hdr 收 200 时的类型与安全头(单测直接调) */
void asst_preview_serve(int fd, const char *method, const char *path, const char *host);   /* 主进程接管钩子: 写完整响应并关 fd */
bool asst_page_resolve(const char *in, char *abs, size_t an, char *err, size_t errn);   /* 工作区里、不在 . 开头目录下、真存在的 .html/.htm → 绝对路径 */
bool asst_rand_hex32(char tok[33]);              /* 128 位随机口令(/dev/urandom) */
void asst_url_name(const char *prefix, const char *tok, const char *abs, char *url, size_t un);   /* prefix + 口令 + "/" + 百分号编码的文件名 */
int asst_web_serve(const char *root, bool flat, const char *enc, const char *host, ds4_buf *hdr, ds4_buf *body);   /* root 里的文件 + CSP sandbox 安全头; 越界 404 */
void asst_web_respond(int fd, int code, ds4_buf *hdr, ds4_buf *body);   /* 写完整响应、释放两个 buf、关 fd */

/* ---------------- 网页分享(asst_share.c, 方案稿 docs/assistant_page.md §7): GET /s/<口令>/<路径>, 不要登录、不过期, 只读分享那一刻的快照 ---------------- */
#define ASST_SHARE_MAX_BYTES (64u << 20)   /* 一份快照最多 64 MiB / 2000 个文件: 网页放在大目录里(比如 inbox/)时宁可拒, 不把整个目录发出去 */
#define ASST_SHARE_MAX_FILES 2000
bool asst_share_path(const char *path);          /* "/s/" 前缀 */
bool asst_share_issue(const char *path, char *url, size_t un, char *err, size_t errn);   /* 工作区里的 .html → 拍快照 → "/s/<口令>/<文件名>"; 同一版本再分享复用同一个口令 */
int asst_share_get(const char *method, const char *path, const char *host, ds4_buf *hdr, ds4_buf *body);   /* 纯判定 + 取字节(单测直接调) */
void asst_share_serve(int fd, const char *method, const char *path, const char *host);   /* 主进程接管钩子 */

/* ---------------- 模型客户端(model_llm.c): 127.0.0.1:<模型端口> 的流式 chat/completions ---------------- */
#define ASST_MAX_CALLS 16
typedef struct {
    void (*on_text)(void *ud, const char *s, size_t n);
    void (*on_think)(void *ud, const char *s, size_t n);
    bool (*stop)(void *ud);
    void *ud;
} asst_llm_cb;
typedef struct {
    ds4_buf content;
    ds4_buf reasoning;         /* 思考段原文(reasoning_content 增量拼起来): 同一轮工具调用的后续请求要原样回传, 提示才与模型吐的逐字节同(KV 前缀复用) */
    int ncalls;
    char call_id[ASST_MAX_CALLS][64], call_name[ASST_MAX_CALLS][64];
    ds4_buf call_args[ASST_MAX_CALLS];
    char finish[32], err[512];
    long prompt_tokens, completion_tokens;
    bool stopped;
} asst_llm_out;
bool asst_llm_chat(int port, const char *conv, const char *messages, const char *tools, const asst_llm_cb *cb, asst_llm_out *o);
void asst_llm_out_free(asst_llm_out *o);
bool asst_llm_strip_images(ds4_buf *body);   /* 请求体里的 image_url 块 → 一句"看不到图"的文字; 换了返回 true(model_llm.c, 400 重发用; 单测直接调) */
bool asst_sse_feed(ds4_buf *pending, const char *data, size_t n, const asst_llm_cb *cb, asst_llm_out *o);   /* 喂原始字节, 按 \n\n 切事件解析(单测直接喂录好的流) */

/* ---------------- 运行(asst_run.c): 一条工人线程, FIFO ---------------- */
void asst_run_start(int model_port);
bool asst_run_submit(const char *session, const char *text, const char *device, const char *client, const char *caps, const char *apps, const char *files, bool *ahead, char *err, size_t errn);   /* ahead = 前面有别的会话在跑/在排(页面据此显示排队中); device = 页面报的设备 JSON(可 NULL): 工人线程拼成事实行放在 user content 前(不进页面事件); client = 发消息的页面 id; caps/apps = 它报的能力表与 app 表; files = 附件(换行分隔) */
void asst_run_stop(const char *session);
bool asst_run_answer(const char *session, const char *call_id, const char *decision, char *err, size_t errn);
const char *asst_run_state(const char *session);   /* idle / queued / running / waiting_approval / waiting_device(asst_mu 下读) */

/* ---------------- HTTP(asst_api.c + asst_sse.c): 挂在 ds4-train 主进程上 ---------------- */
typedef struct {
    const char *method, *path, *query, *body;
    size_t body_len;
    bool peer_local;           /* 对端是 127.0.0.1 / ::1 */
    const char *host, *auth, *xhdr;   /* Host / Authorization / X-DS4-Assistant 头(没有 = 空串) */
    bool signed_in;            /* 工作台登录票有效(studio_auth.c): 票走自定义头, 别的网站读不到也带不上, 当口令用 */
} asst_req;
bool asst_api_path(const char *path);
bool asst_authorized(const asst_req *rq, int *code, const char **why);   /* 本机(且 Host 是本机名)或带口令; POST 还要自定义头(挡 CSRF) */
int asst_api(const asst_req *rq, ds4_buf *out, const char **ctype);
bool asst_sse_path(const char *path);
void asst_sse_start(int fd, const asst_req *rq);   /* 接管 fd, 开线程 tail events.jsonl */

#endif
