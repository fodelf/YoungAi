#include <libproc.h>
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/sysctl.h>
#include <sys/resource.h>
#include <time.h>

int main(int argc, char **argv) {
    int pid = argc > 1 ? atoi(argv[1]) : 0;
    struct rusage_info_v4 ru = {0};
    int proc_ok = pid > 0 && proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t *)&ru) == 0;
    vm_statistics64_data_t vm = {0};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS ||
        host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &count) != KERN_SUCCESS) return 2;
    struct xsw_usage swap = {0};
    size_t len = sizeof(swap);
    if (sysctlbyname("vm.swapusage", &swap, &len, NULL, 0) != 0) return 3;
    int pressure = 0;
    len = sizeof(pressure);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &pressure, &len, NULL, 0) != 0) return 4;
    uint64_t total = 0;
    len = sizeof(total);
    if (sysctlbyname("hw.memsize", &total, &len, NULL, 0) != 0) return 5;
    printf("{\"time\":%lld,\"pid\":%d,\"process_ok\":%s,\"footprint\":%llu,\"rss\":%llu,"
           "\"pressure\":%d,\"swap_used\":%llu,\"swapouts\":%llu,\"compressed\":%llu,"
           "\"free\":%llu,\"speculative\":%llu,\"file_backed\":%llu,\"anonymous\":%llu,\"wired\":%llu,\"total\":%llu}\n",
           (long long)time(NULL), pid, proc_ok ? "true" : "false",
           (unsigned long long)ru.ri_phys_footprint, (unsigned long long)ru.ri_resident_size,
           pressure, (unsigned long long)swap.xsu_used, (unsigned long long)vm.swapouts * page,
           (unsigned long long)vm.compressor_page_count * page,
           (unsigned long long)vm.free_count * page,
           (unsigned long long)vm.speculative_count * page,
           (unsigned long long)vm.external_page_count * page,
           (unsigned long long)vm.internal_page_count * page,
           (unsigned long long)vm.wire_count * page, (unsigned long long)total);
    return 0;
}
