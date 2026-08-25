#!/bin/sh
# Progress readout for the dual-host 0731 download: per-host completed bytes,
# live rate over a sampling window, and ETA. Reads file sizes off both disks —
# the downloader itself logs only get/ok lines (curl runs silent, see
# download_base_model.sh), so this is the observability path.
#
#   ./tools/dl_0731_progress.sh          # one snapshot (10 s rate window)
#   ./tools/dl_0731_progress.sh 60       # 60 s window
#   ./tools/dl_0731_progress.sh 30 loop  # repeat forever
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
NAME=${NAME:-DeepSeek-V4-Flash-0731}
REMOTE_SSH=${REMOTE_SSH:-fodelf@192.168.1.2}
REMOTE_DIR=${REMOTE_DIR:-/Users/fodelf/ds4-main/hf/$NAME}
LOCAL_DIR="$ROOT/hf/$NAME"
TOTAL_GIB=${TOTAL_GIB:-155.4}
# What this run can actually place on the two disks. Since M4 freed ~92 GiB on
# 2026-08-02 the whole model fits, so this equals the full size; it stays a knob
# because a tighter disk would again leave a remainder.
PLACED_GIB=${PLACED_GIB:-155.4}
WINDOW=${1:-10}
LOOP=${2:-}

SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"

# Bytes on disk for the model files, complete + in-flight .part alike.
local_bytes() { find "$LOCAL_DIR" -type f \( -name '*.safetensors' -o -name '*.part' -o -name '*.rr' \) -exec stat -f %z {} + 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
remote_bytes() { $SSH "$REMOTE_SSH" "find '$REMOTE_DIR' -type f \\( -name '*.safetensors' -o -name '*.part' \\) -exec stat -f %z {} + 2>/dev/null | awk '{s+=\$1} END{print s+0}'" </dev/null 2>/dev/null || echo 0; }
count_done() { # dir -> number of complete shards (no .part)
    printf '%s' "$1" >/dev/null
}

report() {
    l1=$(local_bytes); r1=$(remote_bytes)
    sleep "$WINDOW"
    l2=$(local_bytes); r2=$(remote_bytes)
    nl=$(ls "$LOCAL_DIR"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
    nr=$($SSH "$REMOTE_SSH" "ls '$REMOTE_DIR'/*.safetensors 2>/dev/null | wc -l" </dev/null 2>/dev/null | tr -d ' ')
    awk -v l1="$l1" -v l2="$l2" -v r1="$r1" -v r2="$r2" -v w="$WINDOW" \
        -v nl="${nl:-0}" -v nr="${nr:-0}" -v tot="$TOTAL_GIB" -v placed="$PLACED_GIB" '
    BEGIN {
        G=1073741824
        lr=(l2-l1)/w/1048576; rr=(r2-r1)/w/1048576
        have=(l2+r2)/G; rate=lr+rr
        printf "M4  %6.2f GiB  %2d shards  %5.2f MB/s\n", l2/G, nl, lr
        printf "M1  %6.2f GiB  %2d shards  %5.2f MB/s\n", r2/G, nr, rr
        printf "合计 %5.2f / %.1f GiB 本轮 (%.1f%%)  %5.2f MB/s", have, placed, have/placed*100, rate
        if (rate > 0.01) printf "  剩余 %.1f h", (placed-have)*1024/rate/3600
        printf "\n"
        printf "     整模型 %.1f GiB, 本轮之外还有 %.1f GiB 待腾空间\n", tot, tot-placed
    }'
}

if [ "$LOOP" = loop ]; then
    while :; do date "+%H:%M:%S"; report; echo; done
else
    report
fi
