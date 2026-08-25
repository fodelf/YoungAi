#!/bin/sh
# Long-run watchdog/reporter for the dual-host 0731 download. Emits ONE line per
# interval so it can drive a Monitor: progress + rate + free space on both
# disks, and an explicit ALERT line for the states worth acting on (downloader
# gone, no bytes moving, a disk about to fill).
#
# Exits when every placed file is done, so the watch ends by itself.
#
#   ./tools/dl_0731_watch.sh 1800     # report every 30 min
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
INTERVAL=${1:-1800}
NAME=${NAME:-DeepSeek-V4-Flash-0731}
REMOTE_SSH=${REMOTE_SSH:-fodelf@192.168.1.2}
REMOTE_DIR=${REMOTE_DIR:-/Users/fodelf/ds4-main/hf/$NAME}
LOCAL_DIR="$ROOT/hf/$NAME"
PLACED_GIB=${PLACED_GIB:-155.4}
# Free space below this on either disk means the next shard cannot land.
MIN_FREE_GIB=${MIN_FREE_GIB:-3}

SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"

bytes_local()  { find "$LOCAL_DIR" -type f \( -name '*.safetensors' -o -name '*.part' -o -name '*.rr' \) -exec stat -f %z {} + 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
bytes_remote() { $SSH "$REMOTE_SSH" "find '$REMOTE_DIR' -type f \\( -name '*.safetensors' -o -name '*.part' \\) -exec stat -f %z {} + 2>/dev/null | awk '{s+=\$1} END{print s+0}'" </dev/null 2>/dev/null || echo 0; }
free_local()   { df -Pg "$LOCAL_DIR" | awk 'NR==2{print $4}'; }
free_remote()  { $SSH "$REMOTE_SSH" "df -Pg '$REMOTE_DIR'" </dev/null 2>/dev/null | awk 'NR==2{print $4}'; }

prev=$(( $(bytes_local) + $(bytes_remote) ))
while :; do
    sleep "$INTERVAL"
    l=$(bytes_local); r=$(bytes_remote); cur=$((l + r))
    fl=$(free_local); fr=$(free_remote)
    alive=$(pgrep -f download_base_model.sh 2>/dev/null | wc -l | tr -d ' ')
    ndone=$(ls "$LOCAL_DIR"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
    rdone=$($SSH "$REMOTE_SSH" "ls '$REMOTE_DIR'/*.safetensors 2>/dev/null | wc -l" </dev/null 2>/dev/null | tr -d ' ')

    awk -v cur="$cur" -v prev="$prev" -v iv="$INTERVAL" -v placed="$PLACED_GIB" \
        -v nd="${ndone:-0}" -v rd="${rdone:-0}" -v fl="${fl:-0}" -v fr="${fr:-0}" 'BEGIN{
        G=1073741824
        have=cur/G; rate=(cur-prev)/iv/1048576
        printf "0731下载 %.1f/%.1f GiB (%.1f%%) %.2f MB/s", have, placed, have/placed*100, rate
        if (rate > 0.01) printf " 剩余%.1fh", (placed-have)*1024/rate/3600
        printf " | 完整shard M4:%d M1:%d | 余量 M4:%sG M1:%sG\n", nd, rd, fl, fr
    }'

    [ "${alive:-0}" -gt 0 ] || echo "ALERT 下载进程不在了 — 需重跑 ./tools/fetch_0731_dual.sh --force"
    [ "$cur" -gt "$prev" ] || echo "ALERT 一个周期内零字节推进 — 连接可能全卡死"
    awk -v a="${fl:-99}" -v m="$MIN_FREE_GIB" 'BEGIN{exit !(a<m)}' && echo "ALERT M4 磁盘余量 ${fl}G 低于 ${MIN_FREE_GIB}G"
    awk -v a="${fr:-99}" -v m="$MIN_FREE_GIB" 'BEGIN{exit !(a<m)}' && echo "ALERT M1 磁盘余量 ${fr}G 低于 ${MIN_FREE_GIB}G"

    # Done when the downloader has exited AND nothing is still a .part.
    if [ "${alive:-0}" -eq 0 ]; then
        parts=$(find "$LOCAL_DIR" -name '*.part' 2>/dev/null | wc -l | tr -d ' ')
        if [ "${parts:-0}" -eq 0 ]; then echo "0731 本轮下载结束 (M4:$ndone M1:$rdone shards)"; exit 0; fi
    fi
    prev=$cur
done
