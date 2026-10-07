/* server_monitor_hw.c — 监控的硬件读数与 1 Hz 采样线程(2026-10-07)。
 *
 * 读什么、从哪读(与 Strata serve/telemetry.py 同一套来源, 换成 C; 读不到的一律 NAN, 任何一项失败都不影响服务):
 *   GPU  Linux: NVML(libnvidia-ml.so.1, dlopen 运行时装, 不加链接依赖 —— 装了驱动的机器都有它): 负载 / 显存 / 温度 / 功耗 / PCIe 代际与带宽。
 *        统一内存机器(GB10 等, ds4_gpu_unified_memory_host)NVML 不报显存, 显存两项改报整机内存 —— 同一池, 这就是真值(gpu_mem_source="ram")。
 *        macOS: 负载走 IOKit IOAccelerator 的 PerformanceStatistics["Device Utilization %"](活动监视器的同一来源);
 *        显存 = 本进程 Metal 工作集(currentAllocatedSize) / 设备推荐上限(recommendedMaxWorkingSetSize), gpu_mem_source="metal";
 *        温度 / 功耗 / PCIe 没有免 root 的接口(powermetrics 要 sudo), 报 null, 不猜。
 *   CPU  Linux /proc/stat 两次采样差分; macOS host_statistics(HOST_CPU_LOAD_INFO) 差分。
 *   内存 Linux /proc/meminfo: MemTotal − MemAvailable; macOS: hw.memsize − (free + inactive) 页(psutil 的口径, Strata 用 psutil)。
 *   磁盘 Linux /proc/diskstats 整盘(有 /sys/block/<名>)扇区差分 × 512; macOS IOKit IOBlockStorageDriver Statistics 的字节差分。
 * 采样线程每秒一次, 把读数与活请求的 tok/s 一起推进 60 格历史(页面的火花线); 停服时 mon_hw_stop 唤醒并 join 它。 */
#include "server_monitor.h"
#include <sys/utsname.h>
#ifndef DS4_NO_GPU
#include "ds4_gpu.h"
#endif

#if defined(__APPLE__)
#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>
#include <mach/mach.h>
#include <sys/sysctl.h>

static double mac_iokit_number(const char *service, const char *dict_key, const char *key, bool sum_all) {
    io_iterator_t it;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching(service), &it) != KERN_SUCCESS) return MON_NA;
    double out = MON_NA; io_object_t obj;
    CFStringRef cf_dict = CFStringCreateWithCString(kCFAllocatorDefault, dict_key, kCFStringEncodingUTF8);
    CFStringRef cf_key = CFStringCreateWithCString(kCFAllocatorDefault, key, kCFStringEncodingUTF8);
    while ((obj = IOIteratorNext(it))) {
        CFMutableDictionaryRef props = NULL;
        if (IORegistryEntryCreateCFProperties(obj, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS && props) {
            CFDictionaryRef d = CFDictionaryGetValue(props, cf_dict);
            CFNumberRef n = d ? CFDictionaryGetValue(d, cf_key) : NULL;
            double v;
            if (n && CFNumberGetValue(n, kCFNumberDoubleType, &v)) out = mon_known(out) ? out + v : v;
            CFRelease(props);
        }
        IOObjectRelease(obj);
        if (!sum_all && mon_known(out)) break;
    }
    CFRelease(cf_dict); CFRelease(cf_key);
    IOObjectRelease(it);
    return out;
}

static void mac_cpu_ticks(unsigned long long *idle, unsigned long long *total) {
    host_cpu_load_info_data_t cl; mach_msg_type_number_t cnt = HOST_CPU_LOAD_INFO_COUNT;
    *idle = *total = 0;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, (host_info_t)&cl, &cnt) != KERN_SUCCESS) return;
    *idle = cl.cpu_ticks[CPU_STATE_IDLE];
    *total = (unsigned long long)cl.cpu_ticks[CPU_STATE_USER] + cl.cpu_ticks[CPU_STATE_SYSTEM] + cl.cpu_ticks[CPU_STATE_IDLE] + cl.cpu_ticks[CPU_STATE_NICE];
}

static void mac_ram(double *used, double *total) {
    uint64_t mem = 0; size_t len = sizeof mem;
    *used = *total = MON_NA;
    if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) != 0) return;
    vm_statistics64_data_t vm; mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT; vm_size_t page = 0;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &cnt) != KERN_SUCCESS || host_page_size(mach_host_self(), &page) != KERN_SUCCESS) return;
    *total = (double)mem;
    *used = (double)mem - ((double)vm.free_count + (double)vm.inactive_count) * (double)page;
    if (*used < 0) *used = 0;
}
#else
#include <dlfcn.h>
typedef struct { unsigned gpu, memory; } nvml_util_t;
typedef struct { unsigned long long total, free, used; } nvml_mem_t;
static struct {
    void *lib, *dev; int tried, ok; unsigned count; char name[128];
    int (*util)(void *, nvml_util_t *); int (*mem)(void *, nvml_mem_t *);
    int (*temp)(void *, unsigned, unsigned *); int (*pcie_tp)(void *, unsigned, unsigned *);
    int (*power)(void *, unsigned *); int (*plimit)(void *, unsigned *);
    int (*pcie_gen)(void *, unsigned *); int (*pcie_gen_max)(void *, unsigned *); int (*pcie_width)(void *, unsigned *);
} g_nvml;
#define NVML_SYM(field, name) do { *(void **)(&g_nvml.field) = dlsym(g_nvml.lib, name); } while (0)

static int nvml_open(void) {
    if (g_nvml.tried) return g_nvml.ok;
    g_nvml.tried = 1;
    const char *names[] = {"libnvidia-ml.so.1", "libnvidia-ml.so"};
    for (size_t i = 0; i < 2 && !g_nvml.lib; i++) g_nvml.lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
    if (!g_nvml.lib) return 0;
    int (*init)(void) = NULL; int (*get)(unsigned, void **) = NULL; int (*cnt)(unsigned *) = NULL; int (*nm)(void *, char *, unsigned) = NULL;
    *(void **)(&init) = dlsym(g_nvml.lib, "nvmlInit_v2"); if (!init) *(void **)(&init) = dlsym(g_nvml.lib, "nvmlInit");
    *(void **)(&get) = dlsym(g_nvml.lib, "nvmlDeviceGetHandleByIndex_v2"); if (!get) *(void **)(&get) = dlsym(g_nvml.lib, "nvmlDeviceGetHandleByIndex");
    if (!init || !get || init() != 0 || get(0, &g_nvml.dev) != 0) return 0;
    *(void **)(&cnt) = dlsym(g_nvml.lib, "nvmlDeviceGetCount_v2");
    g_nvml.count = 1; if (cnt) (void)cnt(&g_nvml.count);
    *(void **)(&nm) = dlsym(g_nvml.lib, "nvmlDeviceGetName");
    if (nm) (void)nm(g_nvml.dev, g_nvml.name, sizeof g_nvml.name);
    NVML_SYM(util, "nvmlDeviceGetUtilizationRates"); NVML_SYM(mem, "nvmlDeviceGetMemoryInfo");
    NVML_SYM(temp, "nvmlDeviceGetTemperature"); NVML_SYM(pcie_tp, "nvmlDeviceGetPcieThroughput");
    NVML_SYM(power, "nvmlDeviceGetPowerUsage"); NVML_SYM(plimit, "nvmlDeviceGetEnforcedPowerLimit");
    NVML_SYM(pcie_gen, "nvmlDeviceGetCurrPcieLinkGeneration"); NVML_SYM(pcie_gen_max, "nvmlDeviceGetMaxPcieLinkGeneration");
    NVML_SYM(pcie_width, "nvmlDeviceGetCurrPcieLinkWidth");
    g_nvml.ok = 1;
    return 1;
}
static double nvml_u(int (*fn)(void *, unsigned *)) { unsigned v = 0; return fn && fn(g_nvml.dev, &v) == 0 ? (double)v : MON_NA; }
static double nvml_u1(int (*fn)(void *, unsigned, unsigned *), unsigned arg) { unsigned v = 0; return fn && fn(g_nvml.dev, arg, &v) == 0 ? (double)v : MON_NA; }

static void linux_cpu_ticks(unsigned long long *idle, unsigned long long *total) {
    *idle = *total = 0;
    FILE *f = fopen("/proc/stat", "r"); char line[512];
    if (!f) return;
    if (fgets(line, sizeof line, f)) {
        unsigned long long v[10] = {0};
        const int n = sscanf(line, "cpu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9]);
        if (n >= 4) { *idle = v[3] + v[4]; for (int i = 0; i < n; i++) *total += v[i]; }   /* idle + iowait 算空闲(Strata 同) */
    }
    fclose(f);
}
/* 内核报的总量与可用量(kB → 字节); 物理总量由调用方从 sysfs 给(没有就用 MemTotal) */
static void linux_ram(double online_total, double *used, double *total, double *kernel_total) {
    *used = *total = *kernel_total = MON_NA;
    FILE *f = fopen("/proc/meminfo", "r"); char line[256]; unsigned long long tot = 0, avail = 0;
    if (!f) return;
    while (fgets(line, sizeof line, f)) { (void)sscanf(line, "MemTotal: %llu kB", &tot); (void)sscanf(line, "MemAvailable: %llu kB", &avail); }
    fclose(f);
    if (!tot) return;
    *kernel_total = (double)tot * 1024.0;
    *total = mon_known(online_total) && online_total >= *kernel_total ? online_total : *kernel_total;
    *used = *total - (double)avail * 1024.0;   /* 物理总量 − 可用 = 含内核保留在内的"已用" */
    if (*used < 0) *used = 0;
}
/* 物理(在线)内存: /sys/devices/system/memory/block_size_bytes × 在线块数(lsmem 的 "Total online memory" 就是这么算的) */
static double linux_online_memory_bytes(void) {
    FILE *f = fopen("/sys/devices/system/memory/block_size_bytes", "r"); unsigned long long bs = 0;
    if (!f) return MON_NA;
    const int ok = fscanf(f, "%llx", &bs) == 1; fclose(f);
    if (!ok || !bs) return MON_NA;
    unsigned long long online = 0; char p[160];
    for (int i = 0; i < 65536; i++) {   /* 块号可以不连续(热插拔/保留洞), 按编号探 */
        snprintf(p, sizeof p, "/sys/devices/system/memory/memory%d/state", i);
        FILE *s = fopen(p, "r");
        if (!s) { if (i > 4096 && online) break; continue; }
        char st[32] = {0}; if (fgets(st, sizeof st, s) && !strncmp(st, "online", 6)) online++;
        fclose(s);
    }
    return online ? (double)online * (double)bs : MON_NA;
}
static void linux_disk_bytes(double *rd, double *wr) {   /* 整盘累计字节(分区行跳过, 免得算两遍) */
    *rd = *wr = MON_NA;
    FILE *f = fopen("/proc/diskstats", "r"); char line[512];
    if (!f) return;
    double r = 0, w = 0; bool any = false;
    while (fgets(line, sizeof line, f)) {
        char name[64]; unsigned long long sr = 0, sw = 0;
        if (sscanf(line, "%*d %*d %63s %*u %*u %llu %*u %*u %*u %llu", name, &sr, &sw) != 3) continue;
        char path[128]; snprintf(path, sizeof path, "/sys/block/%s", name);
        if (access(path, F_OK) != 0) continue;
        r += (double)sr * 512.0; w += (double)sw * 512.0; any = true;
    }
    fclose(f);
    if (any) { *rd = r; *wr = w; }
}
#endif

void mon_hw_static_read(mon_hw_static *st) {
    memset(st, 0, sizeof *st);
    st->ram_online_bytes = MON_NA;
    st->threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
#if defined(__APPLE__)
    size_t len = sizeof st->cpu_name;
    if (sysctlbyname("machdep.cpu.brand_string", st->cpu_name, &len, NULL, 0) != 0) st->cpu_name[0] = 0;
    int pc = 0; len = sizeof pc;
    if (sysctlbyname("hw.physicalcpu", &pc, &len, NULL, 0) == 0) st->cores = pc;
#ifndef DS4_NO_GPU
    snprintf(st->gpu_name, sizeof st->gpu_name, "%s", ds4_gpu_device_name());
    if (st->gpu_name[0]) { st->gpu_count = 1; snprintf(st->gpu_mem_source, sizeof st->gpu_mem_source, "metal"); }
#endif
#else
    FILE *f = fopen("/proc/cpuinfo", "r"); char line[256];
    if (f) {
        while (fgets(line, sizeof line, f)) {
            if (!strncmp(line, "model name", 10)) {
                const char *c = strchr(line, ':');
                if (c) { snprintf(st->cpu_name, sizeof st->cpu_name, "%s", c + 2); st->cpu_name[strcspn(st->cpu_name, "\n")] = 0; }
                break;
            }
        }
        fclose(f);
    }
    if (!st->cpu_name[0]) { struct utsname u; if (uname(&u) == 0) snprintf(st->cpu_name, sizeof st->cpu_name, "%s", u.machine); }   /* aarch64 的 cpuinfo 没有 model name */
    /* 物理核 = 不同 (package, core_id) 对; 读不到就按线程数 */
    int cores = 0; long long seen[512];
    for (int i = 0; i < st->threads && i < 1024; i++) {
        char p[128]; int core = -1, pkg = -1;
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/core_id", i);
        FILE *fc = fopen(p, "r"); if (fc) { if (fscanf(fc, "%d", &core) != 1) core = -1; fclose(fc); }
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
        FILE *fp = fopen(p, "r"); if (fp) { if (fscanf(fp, "%d", &pkg) != 1) pkg = -1; fclose(fp); }
        if (core < 0) continue;
        const long long key = ((long long)pkg << 32) | (unsigned)core; bool dup = false;
        for (int k = 0; k < cores && !dup; k++) dup = seen[k] == key;
        if (!dup && cores < 512) seen[cores++] = key;
    }
    st->cores = cores > 0 ? cores : st->threads;
    st->ram_online_bytes = linux_online_memory_bytes();
    if (nvml_open()) {
        snprintf(st->gpu_name, sizeof st->gpu_name, "%s", g_nvml.name);
        st->gpu_count = (int)g_nvml.count;
        snprintf(st->gpu_mem_source, sizeof st->gpu_mem_source, "nvml");
    }
#ifndef DS4_NO_GPU
    if (ds4_gpu_unified_memory_host()) {
        snprintf(st->gpu_mem_source, sizeof st->gpu_mem_source, "ram");
        if (!st->gpu_name[0]) { snprintf(st->gpu_name, sizeof st->gpu_name, "%s", ds4_gpu_device_name()); st->gpu_count = st->gpu_name[0] ? 1 : 0; }
    }
#endif
#endif
}

void mon_hw_sample(mon_hw_now *h, const mon_hw_static *st) {
    (void)st;   /* macOS 的 CPU 构建(DS4_NO_GPU)用不到它 */
    static unsigned long long prev_idle, prev_total; static double prev_rd = MON_NA, prev_wr = MON_NA, prev_t;
    const double t = now_sec();
    h->gpu_util = h->gpu_mem_used = h->gpu_mem_total = h->gpu_temp = h->gpu_power = h->gpu_power_limit = MON_NA;
    h->gpu_pcie_gen = h->gpu_pcie_gen_max = h->gpu_pcie_width = h->gpu_pcie_rx_mb = h->gpu_pcie_tx_mb = MON_NA;
    h->cpu = h->ram_used = h->ram_total = h->ram_kernel_total = h->disk_read_mb = h->disk_write_mb = MON_NA;
    unsigned long long idle = 0, total = 0; double rd = MON_NA, wr = MON_NA;
#if defined(__APPLE__)
    mac_cpu_ticks(&idle, &total);
    mac_ram(&h->ram_used, &h->ram_total);
    h->ram_kernel_total = h->ram_total;   /* hw.memsize 就是物理量, 没有第二个口径 */
    h->gpu_util = mac_iokit_number("IOAccelerator", "PerformanceStatistics", "Device Utilization %", false);
    rd = mac_iokit_number("IOBlockStorageDriver", "Statistics", "Bytes (Read)", true);
    wr = mac_iokit_number("IOBlockStorageDriver", "Statistics", "Bytes (Write)", true);
#ifndef DS4_NO_GPU
    if (st->gpu_count) {
        const uint64_t used = ds4_gpu_current_allocated_bytes(), cap = ds4_gpu_recommended_max_working_set_bytes();
        h->gpu_mem_used = (double)used;
        if (cap) h->gpu_mem_total = (double)cap;
    }
#endif
#else
    linux_cpu_ticks(&idle, &total);
    linux_ram(st->ram_online_bytes, &h->ram_used, &h->ram_total, &h->ram_kernel_total);
    linux_disk_bytes(&rd, &wr);
    if (nvml_open()) {
        nvml_util_t u; nvml_mem_t mem;
        if (g_nvml.util && g_nvml.util(g_nvml.dev, &u) == 0) h->gpu_util = (double)u.gpu;
        if (g_nvml.mem && g_nvml.mem(g_nvml.dev, &mem) == 0 && mem.total) { h->gpu_mem_used = (double)mem.used; h->gpu_mem_total = (double)mem.total; }
        h->gpu_temp = nvml_u1(g_nvml.temp, 0u);   /* NVML_TEMPERATURE_GPU */
        const double mw = nvml_u(g_nvml.power); h->gpu_power = mon_known(mw) ? mw / 1000.0 : MON_NA;
        const double lim = nvml_u(g_nvml.plimit); h->gpu_power_limit = mon_known(lim) ? lim / 1000.0 : MON_NA;
        h->gpu_pcie_gen = nvml_u(g_nvml.pcie_gen); h->gpu_pcie_gen_max = nvml_u(g_nvml.pcie_gen_max); h->gpu_pcie_width = nvml_u(g_nvml.pcie_width);
        const double rx = nvml_u1(g_nvml.pcie_tp, 1u), tx = nvml_u1(g_nvml.pcie_tp, 0u);   /* KB/s → MB/s */
        h->gpu_pcie_rx_mb = mon_known(rx) ? rx / 1024.0 : MON_NA; h->gpu_pcie_tx_mb = mon_known(tx) ? tx / 1024.0 : MON_NA;
    }
    if (!strcmp(st->gpu_mem_source, "ram")) { h->gpu_mem_used = h->ram_used; h->gpu_mem_total = h->ram_total; }
#endif
    if (total && prev_total && total > prev_total) {
        const double busy = 1.0 - (double)(idle - prev_idle) / (double)(total - prev_total);
        h->cpu = busy < 0 ? 0 : busy > 1 ? 100 : busy * 100.0;
    }
    prev_idle = idle; prev_total = total;
    if (mon_known(rd) && mon_known(prev_rd) && t > prev_t) {
        h->disk_read_mb = (rd - prev_rd) / (t - prev_t) / 1048576.0;
        h->disk_write_mb = (wr - prev_wr) / (t - prev_t) / 1048576.0;
        if (h->disk_read_mb < 0) h->disk_read_mb = 0;
        if (h->disk_write_mb < 0) h->disk_write_mb = 0;
    }
    prev_rd = rd; prev_wr = wr; prev_t = t;
}

static void *sampler_main(void *arg) {
    struct server_monitor *m = arg;
    for (;;) {
        mon_hw_now now;
        mon_hw_sample(&now, &m->hw_static);
        double tok_s, tok_s_mean, prefill;
        mon_live_rates(m, &tok_s, &tok_s_mean, &prefill);
        pthread_mutex_lock(&m->mu);
        m->hw_now = now;
        mon_series_push(&m->series[MON_S_GPU_UTIL], now.gpu_util);
        mon_series_push(&m->series[MON_S_GPU_MEM], now.gpu_mem_used);
        mon_series_push(&m->series[MON_S_GPU_TEMP], now.gpu_temp);
        mon_series_push(&m->series[MON_S_GPU_POWER], now.gpu_power);
        mon_series_push(&m->series[MON_S_PCIE_RX], now.gpu_pcie_rx_mb);
        mon_series_push(&m->series[MON_S_CPU], now.cpu);
        mon_series_push(&m->series[MON_S_RAM], now.ram_used);
        mon_series_push(&m->series[MON_S_DISK_READ], now.disk_read_mb);
        mon_series_push(&m->series[MON_S_TOK_S], tok_s);
        mon_series_push(&m->series[MON_S_PREFILL], prefill);
        if (!m->stop) {
            struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_sec += 1;
            (void)pthread_cond_timedwait(&m->stop_cv, &m->mu, &ts);
        }
        const bool stop = m->stop;
        pthread_mutex_unlock(&m->mu);
        if (stop) break;
    }
    return NULL;
}

void mon_hw_start(struct server_monitor *m) {
    if (pthread_create(&m->sampler, NULL, sampler_main, m) == 0) m->sampler_started = true;
    else server_log(DS4_LOG_WARNING, "ds4-server: 监控采样线程起不来(%s); /metrics 的硬件读数全为 null", strerror(errno));
}

void mon_hw_stop(struct server_monitor *m) {
    pthread_mutex_lock(&m->mu);
    m->stop = true;
    pthread_cond_broadcast(&m->stop_cv);
    pthread_mutex_unlock(&m->mu);
    if (m->sampler_started) pthread_join(m->sampler, NULL);
    m->sampler_started = false;
}
