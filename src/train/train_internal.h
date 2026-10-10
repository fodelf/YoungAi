/* train_internal.h — 工作台 ds4-train(2026-10-10, 用户 "把训练也做一个可视化页面出来, 可以控制和观测" → "ds 服务启动起来就可以选择是跑模型还是跑训练, 集成在一起")。
 *
 * ★主进程永远不退, 模型与训练都是它的子进程★(10-10 用户: "理论上这个主服务是永远不停的, 模型和后训练都是子实现"):
 *   主进程  ds4-train(train_main.c + train_http.c 一层薄 HTTP)占页面端口, 出工作台页与全部 /api/…。
 *   模型    ds4-server 由它直接 fork/exec(train_model.c), 绑 127.0.0.1:<页面端口+1>(TR_MODEL_PORT); 主进程把 /v1/… /monitor /metrics
 *           与要问活引擎的两条(/api/models/plugins 热切、/api/models/list)原字节转给它(train_proxy.c), 页面与 API 客户端只认主进程的端口。
 *   训练    主进程里一条作业线程(train_job.c): 停模型子进程 → fork ./ds4 --ptrain → wt2 门(train_gate.c: fork ./ds4 --score-ids 两臂 +
 *           anchor_metrics 五指标)→ 选轮 → 模型子进程按 gguf/serve_pick.txt 装回来。
 *           (页面上原有的"出题"(文档 → 模型写问答)10-10 按用户要求删掉; 文档上传后直接成原文料 <目录>.text.jsonl 可训。)
 *   ★流程编排全在 C 里, 页面不调任何 shell 脚本★(10-10 用户: "所有功能不应该是 c 代码实现的吗, 为什么一直看你在改 sh 脚本"):
 *   此前停模型/训/门/选轮/起服散在 train_cycle.sh + z_nightly_spark.sh + v41_judge.sh + serve_1m_spark.sh 里, 产品行为(训完挂不挂)埋在脚本,
 *   发布包要带五个脚本, Mac 上跑不了。脚本保留为命令行工具, 产物目录布局/日志格式与它们一致, 记录页两边都能读。
 *   仅剩的脚本调用: hf_install.sh download|probe(纯 hf CLI 包装, 下载本身就是 python 的 hf)。
 * 物理墙: 模型只能装一份(100+ GB), 模型与训练不能同时在, 所以训练前先停模型子进程。
 * 为什么不在进程内卸引擎再重装: CUDA 清理后重初始化从没走过, 崩在中间页面也没了; 进程级起停是天天在用的路。
 * 一份实现: 数据面(train_runs.c, 盘上产物 → JSON)+ 控制面(train_ctl.c /proc 扫 + train_child.c 子进程 + train_model.c/train_job.c)+ 路由(train_api.c)
 *   两个二进制共用: ds4-server 也出同一页(直接开它的端口能聊、能热切), 但发车/换模型/起服这些"管子进程"的操作只在主进程里做(serving=1 时拒)。
 * 接口: GET / 页; GET /api/train/status|runs|run?name=|data|preview; POST /api/train/start|stop|serve|upload?kind=jsonl|doc&name=&dir=(原始体)。
 * 模型页(train_models.c, 10-10 用户 "少一个模型 tab, 可以下载模型, 跟 unsloth 的客户端页面一样"):
 *   GET /api/models/list(本机 GGUF + 配套侧车 + 正在用的 + HF 下载进度); POST /api/models/download|cancel|load|plugins。
 *   加载 = 写 gguf/serve_pick.txt 再(重)起模型子进程。 */
#ifndef DS4_TRAIN_INTERNAL_H
#define DS4_TRAIN_INTERNAL_H
#include "../common/ds4_json.h"
#include <stdbool.h>
#include <stddef.h>
#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TR_PATH 1024
#define TR_UPLOAD_MAX (64u * 1024u * 1024u)   /* 与 ds4-server 的请求体上限同(server_httpd.c max_body) */

typedef struct {
    char method[8];
    char path[512];       /* 不含 ? 之后 */
    char query[512];      /* ? 之后(可空) */
    const char *body;     /* POST 体(可空) */
    size_t body_len;
} tr_req;

/* 路由: 往 out 写响应体, 返回 HTTP 状态码, *ctype 给 Content-Type。serving = 调用方是装着模型的 ds4-server(模型子进程);
 * port = 调用方自己监听的端口(主进程里 +1 就是模型子进程的端口)。 */
int tr_api(const tr_req *rq, ds4_buf *out, const char **ctype, int serving, int port);
/* 一次性设根目录(= 仓库根, z_nightly 的 $ROOT; 两个二进制都在仓库根跑) */
void tr_init(const char *root);
/* 模型子进程的端口 = 页面端口 + 1, 只绑 127.0.0.1: 外面只认主进程一个端口, 子进程换了、重装了, 客户端地址不变 */
#define TR_MODEL_PORT(page_port) ((page_port) + 1)

/* train_http.c + train_proxy.c(只有 ds4-train 用) */
typedef int (*tr_handler)(const tr_req *req, ds4_buf *out, const char **ctype);
int tr_http_serve(const char *host, int port, tr_handler h);   /* 阻塞循环; 监听失败返回非 0 */
bool tr_proxy_path(const char *path);   /* 这条路归模型子进程(/v1/… /monitor /metrics /api/models/plugins|list) */
void tr_proxy_start(int cfd, char *raw, size_t raw_len, int model_port);   /* 接管 cfd 与 raw(请求原字节), 开线程转发, 立刻返回 */
bool tr_query_get(const char *query, const char *key, char *val, size_t n);   /* ?a=b&c=d 取一项(百分号解码) */
bool tr_body_get(const tr_req *rq, const char *key, char *out, size_t n);     /* POST 体扁平 JSON 取一项(数值照原文) */
bool tr_clean(const char *s, const char *extra);   /* 只含字母数字与 extra 里的字符(进 argv 的参数一律先过它) */

/* 盘上根目录(tr_init 设) */
extern char tr_root[TR_PATH];       /* 仓库根 */
extern char tr_ftd[TR_PATH];        /* $ROOT/gguf/v41/posttrain */
extern char tr_datad[TR_PATH];      /* $ROOT/gguf-tools/data/posttrain */

/* train_runs.c: 盘上产物 → JSON */
void tr_json_runs(ds4_buf *b);                 /* [{name, ...摘要}] 按 ptrain.cfg 修改时间倒序 */
bool tr_json_run(ds4_buf *b, const char *name);   /* 一趟全量; 目录不存在返回 false */
void tr_json_data(ds4_buf *b);                 /* {jsonl:[...]} */
long tr_mem_avail_mb(void);                    /* /proc/meminfo MemAvailable(MB); 没有 -1 */
bool tr_run_active(const char *dir);           /* 这趟的 train.out 近两分钟内还在写 */
char *tr_slurp(const char *path, size_t *len); /* 整文件读进 malloc 缓冲(NUL 结尾); 没有 NULL */

/* train_ctl.c: /proc 扫进程(按程序名认, 不在整条命令行里找子串) */
typedef struct { bool scanned, ds4, server, dl; } tr_procs;   /* ds4 = 任何 ./ds4 --cuda|--metal; server = ds4-server; dl = hf_install.sh */
void tr_proc_scan(tr_procs *p);                /* 非 Linux: scanned=false 全 false */
int tr_proc_kill(const char *prog, int sig);   /* 给名叫 prog 的进程(程序本身或 bash 跑的同名脚本)发信号, 返回个数 */
bool tr_server_model(char *gguf, size_t gn, char *zch, size_t zn, char *pt, size_t pn);   /* 跑着的 ds4-server 的 -m / --zchain / --posttrain(没有就空串); 返回有没有扫到 */
bool tr_spawn_download(const char *const args[], char *err, size_t errn);   /* bash hf_install.sh download args…, 日志 ui_logs/hub_*.log */
bool tr_cancel_download(const char *hub, char *err, size_t errn);            /* 杀 hf_install.sh + 它在 hub 下的 hf download; 半截留着续传 */
bool tr_ui_log_path(const char *tag, char *out, size_t n);                   /* ui_logs/<tag>_<时间>.log 的路径(逐层建目录) */

/* train_child.c: 子进程(fork/setsid/exec, 日志重定向, 内存看门狗, 本机 HTTP) */
#define TR_WD_KILL_MB 2500   /* 看门狗红线: MemAvailable 连续两次低于它就杀子进程(08-24 实撞: available → 0 机器假死, 请求中断可重试、假死不可) */
typedef int (*tr_cancel_fn)(void *ud);   /* 返回非 0 = 调用方要求中止 */
pid_t tr_child_spawn(const char *const argv[], const char *log_path, bool append);   /* 自成会话, stdout/stderr → log_path(NULL = 丢弃), fd≥3 不继承; 失败 -1 */
bool tr_child_spawn_detached(const char *const argv[], const char *log_path);        /* 两次 fork 归 init(不等不收): 只给"起了就不管"的下载用 */
int tr_child_wait(pid_t pid, tr_cancel_fn cancel, void *ud, FILE *log);   /* 每秒看一次: 看门狗 + cancel; 返回退出码(被信号杀 = 128+信号, 看门狗杀 = -2) */
void tr_child_kill(pid_t pid);                 /* SIGTERM, 3 秒后还在就 SIGKILL, 然后收尸 */
bool tr_prog_wait_gone(const char *prog, int timeout_s);   /* 等名叫 prog 的进程从 /proc 消失(卸 100 GB 映射要几十秒) */
int tr_http_local(int port, const char *method, const char *path, const char *body, ds4_buf *out, int timeout_s);   /* 127.0.0.1:port 一条请求, 返回状态码(连不上 -1) */
void tr_logf(FILE *f, const char *fmt, ...);   /* "[HH:MM:SS] …" 一行, 立刻 flush */

/* train_model.c: 模型子进程 ds4-server 的生命周期 */
typedef enum { TR_MODEL_IDLE, TR_MODEL_STARTING, TR_MODEL_UP, TR_MODEL_STOPPING } tr_model_phase;
bool tr_model_start(int model_port, bool restart, char *err, size_t errn);   /* 异步: 按 serve_pick.txt 起(restart = 先停在跑的); 已在起 → false */
bool tr_model_start_sync(int model_port, FILE *log);   /* 同步版(作业线程训完装回来用): 起跑清单 → exec → /v1/models 可达 → 余量 → 冒烟 → 看门狗线程 */
void tr_model_stop_sync(FILE *log);            /* 停模型子进程并等它从 /proc 消失(不删任何文件) */
tr_model_phase tr_model_state(void);
bool tr_model_live(void);                      /* 模型子进程在且装完了(冒烟过): 这时才转发 */
bool tr_pick_read(char *g, size_t gn, char *z, size_t zn, char *extra[], int max_extra, char *storage, size_t sn);   /* serve_pick.txt: ①, ②(none→空), 第 3 行起的额外参数 */

/* train_job.c: 训练作业线程(一次一条) */
typedef enum { TR_JOB_NONE, TR_JOB_STOP_MODEL, TR_JOB_TRAIN, TR_JOB_GATE, TR_JOB_RESTART } tr_job_phase;
typedef struct { bool active; tr_job_phase phase; pid_t child; char run[256]; } tr_job_info;   /* run = 训练目录名(kd-…) */
bool tr_job_train(const char *data_rel, const char *epochs, const char *layers, const char *lr, const char *extra, int model_port, char *err, size_t errn);
bool tr_job_stop(char *err, size_t errn);      /* 杀正在跑的 ./ds4; 作业线程收到后照样把模型装回来 */
void tr_job_status(tr_job_info *out);
bool tr_job_owns(const char *run_dir_abs);     /* 这趟目录是正在跑的作业的(记录页判 running 用: 门阶段 train.out 不动, 只看文件时间会误判"没选轮") */
/* 训练/打分共用的 ./ds4 参数前缀: --cuda -m ① [--zchain ②] [--engram-dir …]; 返回 argv 项数 */
int tr_ds4_base_argv(const char *argv[], int max, const char *mdl, const char *zch, char *const extra[], int nextra);

/* train_gate.c: wt2 门 + 选轮 */
#define TR_GATE_NTOK 512          /* 判决料长度: 发布包只带 wt2 512 token 的 FP 教师锚(265 MB) */
#define TR_GATE_SMIN_PP (-0.5)    /* Σmin 相对 ② 态退不过 0.5pp(10-02 定: top-1 一致率太宽松, 主尺换成分布还原率) */
#define TR_GATE_KLD_REL 0.03      /* Mean KLD 涨不过 3% */
/* 对 run_dir 下每个 ckpt_eNN 跑 ② 臂(一次)与 ②+③ 臂, 判决全文落 eval/gate_ckpt_eNN.txt(先 ② 后 ②+③, 记录页同口径读), 过门里留出损失最低者写 pick.txt;
 * 返回选中的 ckpt 名(空串 = 没有一轮过门或失败) */
bool tr_gate_pick(const char *run_dir, const char *mdl, const char *zch, char *const extra[], int nextra, FILE *log, tr_cancel_fn cancel, void *ud, char *pick, size_t pn);

/* train_models.c: 模型页路由; 不是 /api/models/ 开头返回 -1 */
int tr_models_api(const tr_req *rq, ds4_buf *out, int serving, int port);
/* 活引擎的钩子: ds4-server 起服时填(server_plugins.c), ds4-train 里全空 = 没有装着的模型可热切。
 * 热切换 = 不重装底座, 只换 ②侧车 / ③后训练件(聊天页的切换, 10-10 用户 "chat 页面再加一个切换不同侧车的功能")。 */
typedef struct {
    const char *gguf;   /* 服务装着的 GGUF(-m 原样) */
    int (*request)(const char *amp_dir, const char *pt_dir, char *err, size_t errn);   /* 排一个切换任务就返回; 0 = 已排上 */
    /* 当前挂着的 ②③(没挂给空串) + 有没有排着没做的切换 + 上一次切换失败的原因(成功是空串) */
    void (*status)(char *amp, size_t an, char *pt, size_t pn, int *pending, char *err, size_t en);
} tr_live_t;
extern tr_live_t tr_live;
/* 写 gguf/serve_pick.txt(第 1 行 ①, 第 2 行 ② 或 none, 之后每行一个额外引擎参数; serve_1m_spark.sh pick 同一格式): 下次起模型子进程就按它。n-gram 目录自动带 */
bool tr_pick_write(const char *gguf_abs, const char *zch_abs, const char *pt_abs, char *err, size_t errn);

#endif
