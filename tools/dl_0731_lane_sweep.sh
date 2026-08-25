#!/bin/sh
# Find the fastest --lanes for the dual-host 0731 download, then restart the
# download with it.
#
# Why a sweep and not just "set it high": the mirror throttles per connection,
# so more lanes help — until they don't. Two ceilings bite from above:
#   * the mirror starts stalling connections (curl timeouts, lanes idle), and
#   * every REMOTE lane holds its own ssh session; macOS sshd defaults to
#     MaxSessions 10 / MaxStartups 10, past which lanes are simply refused.
# So the useful lane count is measured, not assumed.
#
# Restarting is cheap and lossless: curl -C - resumes each .part in place.
#
#   ./tools/dl_0731_lane_sweep.sh              # sweep 3 6 8 12, then run best
#   ./tools/dl_0731_lane_sweep.sh "4 8 16"     # custom candidates
#   WINDOW=90 ./tools/dl_0731_lane_sweep.sh    # longer measurement window
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CANDIDATES=${1:-"3 6 8 12"}
WARMUP=${WARMUP:-20}
WINDOW=${WINDOW:-60}
LOG=${LOG:-/tmp/dl_0731.log}
REMOTE_SSH=${REMOTE_SSH:-fodelf@192.168.1.2}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"

stop_all() {
    pkill -f download_base_model.sh 2>/dev/null || true
    pkill -f "curl.*hf-mirror" 2>/dev/null || true
    $SSH "$REMOTE_SSH" 'pkill -f "curl.*hf-mirror"' 2>/dev/null || true
    sleep 3
}

start_with() { # lanes
    DL_LANES=$1 nohup "$ROOT/tools/fetch_0731_dual.sh" --force > "$LOG" 2>&1 &
}

best_n=0; best_rate=0
RESULTS=$(mktemp)

for n in $CANDIDATES; do
    stop_all
    start_with "$n"
    sleep "$WARMUP"
    # measure over WINDOW using the progress tool's byte accounting
    out=$("$ROOT/tools/dl_0731_progress.sh" "$WINDOW")
    rate=$(printf '%s\n' "$out" | awk '/合计/ { for (i=1;i<=NF;i++) if ($i=="MB/s") print $(i-1) }')
    [ -n "$rate" ] || rate=0
    # lanes that got refused/stalled show up as fewer live connections
    lc=$(pgrep -f "curl --noproxy" 2>/dev/null | wc -l | tr -d ' ')
    rc=$($SSH "$REMOTE_SSH" 'pgrep -f "^curl -fsSL" 2>/dev/null | wc -l' </dev/null 2>/dev/null | tr -d ' ')
    printf '%s\t%s\t%s\t%s\n' "$n" "$rate" "${lc:-?}" "${rc:-?}" >> "$RESULTS"
    printf 'lanes %-3s => %6s MB/s   (活跃连接 M4:%s M1:%s)\n' "$n" "$rate" "${lc:-?}" "${rc:-?}"
    if awk -v a="$rate" -v b="$best_rate" 'BEGIN{exit !(a>b)}'; then best_rate=$rate; best_n=$n; fi
done

echo
echo "===== 扫描结果 ====="
printf 'lanes\tMB/s\tM4连接\tM1连接\n'
cat "$RESULTS"
rm -f "$RESULTS"
echo
echo "最优: lanes=$best_n  ($best_rate MB/s) — 用它继续下载"
stop_all
start_with "$best_n"
sleep 5
echo "已用 DL_LANES=$best_n 起跑; 进度: ./tools/dl_0731_progress.sh"
