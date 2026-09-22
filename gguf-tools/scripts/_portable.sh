# _portable.sh — 被 source 的函数库(不是可执行脚本)。
#
# 存在理由: 战役流水线原本只跑 macOS, 盘闸/内存看门狗/文件大小全用了 BSD 专有形态
# (df -g /System/Volumes/Data, footprint -p, stat -f %z)。DGX Spark(Linux aarch64)
# 上这些要么报错要么返回空 —— 盘闸拿到空串会当成"不足"直接 exit, 看门狗拿到空串则
# 静默失效(比报错更坏: 你以为有护栏, 其实没有)。
#
# macOS 分支逐字保留原实现, 行为零变化; Linux 分支给等价语义。
#
# 用法: . "$(dirname "$0")/_portable.sh"

if [ "$(uname -s)" = "Darwin" ]; then
    # 可用空间 GiB(整数)。传目录, 内部解析到所在卷。
    disk_free_gb(){ df -g "${1:-$HOME}" | awk 'NR==2{print $4}'; }
    # 进程内存 MiB。macOS 用 footprint(与看门狗历史口径一致: phys_footprint 而非 rss)
    proc_mem_mb(){
        footprint -p "$1" 2>/dev/null | grep -Eo 'Footprint: *[0-9.]+ *[KMG]B' | head -1 \
            | awk '{v=$2;u=$3; if(u=="GB")v*=1024; else if(u=="KB")v/=1024; printf "%d",v}'
    }
    # macOS 的 phys_footprint 本来就不含干净的 file-backed 页 ⇒ 已经是 anon 口径, 直接复用。
    proc_anon_mb(){ proc_mem_mb "$1"; }
    file_apparent_bytes(){ stat -f %z "$1"; }             # 表观大小(含稀疏洞)
    file_real_bytes(){ echo $(( $(stat -f %b "$1") * 512 )); }  # 实占块数×512
    file_mtime(){ stat -f %m "$1"; }
else
    disk_free_gb(){ df -BG "${1:-$HOME}" | awk 'NR==2{gsub(/[A-Za-z]/,"",$4); print $4+0}'; }
    # Linux 没有 footprint。VmRSS 是最接近的常驻口径 —— 对"进程吃爆内存"这个看门狗
    # 真正要拦的场景足够; 与 macOS 的 phys_footprint 不是逐字节同义, 但同量级同趋势。
    proc_mem_mb(){ awk '/^VmRSS:/{printf "%d", $2/1024}' "/proc/$1/status" 2>/dev/null; }
    # ★OOM 闸要用的是这个, 不是 VmRSS(2026-08-28 实撞)★
    # VmRSS = RssAnon + RssFile + RssShmem。本负载把 82GB 的 HF 教师权重和 43 个层文件全
    # mmap 进来, 这些是【干净的 file-backed 页, 内核随时能丢】, 一点 OOM 风险都没有, 却把
    # VmRSS 顶到 112GB —— r30 的看门狗照这个数把一条健康的反修杀在 L40/43(同一时刻
    # MemAvailable 还有 86GB, MemAvailable 地板那道闸根本没响)。
    # RssAnon+RssShmem 才是回收不掉、真会把系统拖进 OOM 的那部分。
    proc_anon_mb(){ awk '/^RssAnon:|^RssShmem:/{s+=$2} END{printf "%d", s/1024}' "/proc/$1/status" 2>/dev/null; }
    file_apparent_bytes(){ stat -c %s "$1"; }
    file_real_bytes(){ echo $(( $(stat -c %b "$1") * 512 )); }
    file_mtime(){ stat -c %Y "$1"; }
fi
