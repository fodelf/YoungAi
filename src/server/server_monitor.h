#ifndef DS4_SERVER_MONITOR_H
#define DS4_SERVER_MONITOR_H
/* 监控(GET /monitor 页 + GET /metrics)的内部结构 —— 三个编译单元共用:
 *   server_monitor.c      请求生命周期记录(排队→读提示→生成→收尾)、总计、延迟直方图、/metrics 的 JSON
 *   server_monitor_hw.c   硬件读数(GPU/CPU/内存/磁盘)与 1 Hz 采样线程、60 秒历史
 *   server_monitor_prom.c /metrics 的 Prometheus 文本(vLLM 指标名 + ds4: 前缀的自有指标)
 * 服务端其它文件只走 server_internal.h 里的 mon_* 入口, 不碰这里的结构。
 * 字段名与 Strata(github.com/Niko1221/Strata, serve/server.py 的 metrics())逐个对齐: engine / live / requests /
 * requests_kept / totals / hardware / hardware_static / history / conversation_cache / time —— 两边的页面与看板可互读。 */
#include "server_internal.h"

/* "读不到 / 没到那一步"的哨兵。★不能用 NAN★: 全仓 -ffast-math, 编译器按"没有 NaN"优化, isnan() 可以被折成假, NaN 比较结果不可信
 * (clang 直接给 -Wnan-infinity-disabled)。监控里所有量(百分比 / 字节 / 秒 / 温度 / t/s)都 ≥ 0, 负数就是"没有"。 */
#define MON_NA (-1.0)
static inline bool mon_known(double v) { return v >= 0.0; }

enum {
    MON_HISTORY = 500,        /* 收尾请求保留条数(Strata 的 history deque maxlen) */
    MON_SHOW = 12,            /* /metrics 默认给页面的最近请求条数; ?requests=all 给全部 */
    MON_RATE_RING = 1024,     /* 每条活请求的 (已生成, 时刻) 环: 2 s 窗口按 400 t/s 也装得下 */
    MON_HW_HISTORY = 60,      /* 1 Hz × 60 = 一分钟火花线 */
    MON_BUCKETS = 22,         /* vLLM 的延迟直方图桶数(server_monitor_prom.c 的桶边界) */
};
/* 速率窗口(Strata RATE_WINDOW_S / RATE_MIN_SPAN_S): 2 秒窗内首尾两样本差分; 窗太短(< 0.25 s)退回全程均值 */
#define MON_RATE_WINDOW_S 2.0
#define MON_RATE_MIN_SPAN_S 0.25

typedef enum { MON_QUEUED = 0, MON_READING, MON_GENERATING } mon_state;

typedef struct {
    uint64_t id;
    mon_state state;
    api_style api; req_kind kind;
    bool stream, tools, thinking;
    char path[40];
    double started_at;                  /* 墙钟秒(gettimeofday), 页面显示时刻用 */
    double t_queued, t_read, t_first;   /* 单调钟(now_sec): 入队 / 开始预填 / 首个 token; 0 = 还没到 */
    int prompt_tokens, cached, prompt_read, generated, max_tokens;
    int rn[MON_RATE_RING]; double rt[MON_RATE_RING]; int rhead, rlen;   /* (已生成, 时刻) 环 */
} mon_live;

typedef struct {
    double time, duration_s, first_token_s, prompt_ms, decode_ms;   /* first_token_s/prompt_ms/decode_ms: MON_NA = 没到那一步 */
    char finish[16];
    api_style api; bool stream;
    int prompt_tokens, reused, output_tokens, prompt_read;
    int drafts_offered, drafts_accepted;   /* -1 = 引擎没报(非投机路) */
} mon_done;

typedef struct { uint32_t count[MON_BUCKETS]; double sum; uint32_t n; } mon_hist;
typedef struct { double v[MON_HW_HISTORY]; int head, len; } mon_series;   /* MON_NA = 该秒读不到 */

typedef struct {   /* 一次硬件读数; MON_NA = 读不到(页面显示 "–", Prometheus 不出样本) */
    double gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp, gpu_power, gpu_power_limit;
    double gpu_pcie_gen, gpu_pcie_gen_max, gpu_pcie_width, gpu_pcie_rx_mb, gpu_pcie_tx_mb;
    /* ram_total = 物理(在线)内存, ram_used = ram_total − 可用; ram_kernel_total = 内核自己能分配的总量(Linux MemTotal, 比物理少内核保留的那几 GiB;
     * macOS 两者同)。为什么分两个: GB10 物理 128 GiB, MemTotal 只有 121.7 GiB —— 只报后者用户看到的"显存"就比机器标称少 6 GiB(10-07 实撞)。 */
    double cpu, ram_used, ram_total, ram_kernel_total, disk_read_mb, disk_write_mb;
} mon_hw_now;

typedef struct {
    char gpu_name[128], cpu_name[128];
    int gpu_count, cores, threads;
    double ram_online_bytes;   /* Linux: sysfs 在线内存块之和(= 物理内存, 1023 × 128 MiB 那种); 读不到 = MON_NA, 总量退回 MemTotal */
    /* 显存两项从哪来: "nvml" = 独显的真显存; "ram" = 统一内存机器(GB10)显存就是整机内存; "metal" = 本进程 Metal 工作集 /
     * 设备推荐上限; "" = 读不到。页面按它写副标题, 免得把"本进程工作集"当成"整卡显存"看。 */
    char gpu_mem_source[8];
} mon_hw_static;

enum { MON_S_GPU_UTIL, MON_S_GPU_MEM, MON_S_GPU_TEMP, MON_S_GPU_POWER, MON_S_PCIE_RX, MON_S_CPU, MON_S_RAM,
       MON_S_DISK_READ, MON_S_TOK_S, MON_S_PREFILL, MON_S_COUNT };

struct server_monitor {
    pthread_mutex_t mu;
    server *s;
    double since;                       /* 墙钟: 起服时刻(totals.since) */
    uint64_t next_id;
    mon_live *live; int live_len, live_cap;
    mon_done hist[MON_HISTORY]; int hist_head, hist_len;   /* 环: hist_head = 下一个写位 */
    struct { uint64_t requests, prompt_tokens, reused, output_tokens, drafts_offered, drafts_accepted; double prompt_ms, decode_ms; } totals;
    mon_hist ttft, itl, e2e;            /* 首 token 延迟 / token 间隔 / 整条请求, vLLM 桶 */
    struct { uint64_t parks, restores, evictions; int parked; uint64_t bytes; char last_event[12]; int last_tokens; double last_at; } kv;
    mon_hw_static hw_static; mon_hw_now hw_now; mon_series series[MON_S_COUNT];
    pthread_t sampler; bool sampler_started, stop;
    pthread_cond_t stop_cv;
};

/* server_monitor.c → 采样线程用: 活请求此刻的解码速率(2 s 窗, 多路求和) / 全程均值 / 预填速率; 自己取锁 */
void mon_live_rates(struct server_monitor *m, double *tok_s, double *tok_s_mean, double *prefill_tok_s);
void mon_series_push(mon_series *s, double v);
/* server_monitor_hw.c */
void mon_hw_static_read(mon_hw_static *st);
void mon_hw_sample(mon_hw_now *now, const mon_hw_static *st);
void mon_hw_start(struct server_monitor *m);
void mon_hw_stop(struct server_monitor *m);
#endif /* DS4_SERVER_MONITOR_H */
