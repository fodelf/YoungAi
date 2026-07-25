#!/usr/bin/env bash
# go2b_migrate_watch.sh — go2b 全量分储搬运器(M4 上跑, emit 期间常驻)。
# M1 盘只有 ~22G 装不下 43×~1G 层产物 → 每层完成即拉到 M4 规范 layers/ 并删 M1 侧(滚动, M1 峰值~5G)。
# 完成判据: L 的 dql+go2b侧车 都在 且 (L+1 的 dql 已出现 或 量化进程已结束) — 保证写完才搬。
# 校验: 字节数两侧一致才删源(scp 后 stat 对表)。日志: /tmp/go2b_migrate.log
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
M1=${M1:-192.168.1.2}; M1DIR=${M1DIR:-/Users/fodelf/ds4-main}
SRC=$M1DIR/gguf/go-onebit/layers
DST=$ROOT/gguf/go-onebit/layers
mkdir -p "$DST"
log(){ echo "[migrate $(date +%H:%M:%S)] $*"; }

pull_one(){   # $1=远端文件名
    local f=$1
    local rsz lsz
    rsz=$(ssh "$M1" "stat -f%z $SRC/$f 2>/dev/null" || echo 0)
    [ "$rsz" -gt 0 ] || return 1
    rsync -qS "$M1:$SRC/$f" "$DST/$f.part" || return 1   # -S 保稀疏(dql 热槽是洞, scp 会实体化多占~7G)
    lsz=$(stat -f%z "$DST/$f.part" 2>/dev/null || echo -1)
    if [ "$lsz" = "$rsz" ]; then
        mv "$DST/$f.part" "$DST/$f"
        case "$f" in dql_L*.bin) python3 "$HERE/sparsify_zeros.py" "$DST/$f" >/dev/null 2>&1 || true ;; esac   # 热槽零块重打洞(网络拷贝实体化, rsync -S 在macOS不保洞)
        ssh "$M1" "rm -f $SRC/$f"
        log "✓ $f (${lsz}B) → M4, M1 已删"
        return 0
    fi
    log "✗ $f 字节不符 remote=$rsz local=$lsz — 保留源, 重试待后"
    rm -f "$DST/$f.part"; return 1
}

log "看守启动: $M1:$SRC → $DST"
DONE_FLAG=0
while :; do
    RUNNING=$(ssh "$M1" "pgrep -f ds4quant_run >/dev/null && echo 1 || echo 0" 2>/dev/null || echo 1)
    LIST=$(ssh "$M1" "ls $SRC 2>/dev/null" || true)
    for L in $(seq 0 42); do
        NN=$(printf "%02d" "$L"); NX=$(printf "%02d" $((L+1)))
        for f in "dql_L$NN.bin" "dql_go2b_L$NN.bin" "opt_L$NN.bin"; do
            echo "$LIST" | grep -qx "$f" || continue
            [ -f "$DST/$f" ] && { ssh "$M1" "rm -f $SRC/$f"; continue; }
            # 完成判据: 下一层 dql 已出现 / 已到 L42 收尾 / 进程结束
            if [ "$L" = 42 ] || echo "$LIST" | grep -qx "dql_L$NX.bin" || [ "$RUNNING" = 0 ]; then
                pull_one "$f" || true
            fi
        done
    done
    N_M4=$(ls "$DST"/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')
    N_G2=$(ls "$DST"/dql_go2b_L*.bin 2>/dev/null | wc -l | tr -d ' ')
    M1FREE=$(ssh "$M1" "df -g /System/Volumes/Data | awk 'NR==2{print \$4}'" 2>/dev/null || echo "?")
    M4FREE=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
    log "M4已收 dql=$N_M4/43 go2b=$N_G2/43 | 盘 M1=${M1FREE}G M4=${M4FREE}G | M1量化进程=$RUNNING"
    if [ "$RUNNING" = 0 ] && [ "$N_M4" -ge 43 ] && [ "$N_G2" -ge 43 ]; then
        log "★全部 43 层已迁移完成, 看守退出★"; break
    fi
    if [ "$RUNNING" = 0 ]; then
        DONE_FLAG=$((DONE_FLAG+1))
        [ "$DONE_FLAG" -ge 10 ] && { log "量化已停但层数不足(dql=$N_M4) — 退出交人工"; break; }
    fi
    sleep 60
done
