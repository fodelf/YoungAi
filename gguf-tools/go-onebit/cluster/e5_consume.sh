#!/bin/sh
# e5_consume.sh — M4 side of the two-stage E5 pipeline: pull each sel_L{n}.bin
# the M1 pump lands (M4→M1 scp; M1 outbound has known quirks), solve it with
# the shard-free --load-sel path, dump z into ZDIR, delete BOTH copies of the
# consumed sel file (they are this pipeline's own intermediates).
#   usage (on M4): e5_consume.sh "23 24 ... 42"
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
ROOT_M1=${ROOT_M1:-/Users/fodelf/ds4-main}
M1=${M1:-192.168.1.2}
SPOOL_M1=${SPOOL_M1:-$ROOT_M1/sel_spool}
SPOOL=${SPOOL:-$ROOT/sel_spool}
ZDIR=${ZDIR:-$ROOT/zdump_rrr}
CFG=${CFG:---solver rrr --rank 64 --lambda 1e-1 --lam-c 0.2 --feat yhat --w-align 1.0 --eig-iters 100 --threads 5}
LAYERS=${1:?layer list}
ZDIR=${2:-$ZDIR}              # positional override beats env-loss-through-nohup
LOCAL=${LOCAL:-0}             # 1 = consume the pump's LOCAL spool on this host
                              # (the load-sel algebra is shard-free: the pump
                              # host can eat its own sel files, no scp at all)
[ "$LOCAL" = "1" ] && SPOOL=$SPOOL_M1
mkdir -p "$SPOOL" "$ZDIR"
cd "$ROOT/gguf-tools" || exit 1
echo "consume: layers=[$LAYERS] zdir=$ZDIR local=$LOCAL spool=$SPOOL"
for L in $LAYERS; do
    [ -f "$ZDIR/z_L$L.bin" ] && continue
    if [ "$LOCAL" = "1" ]; then
        until [ -f "$SPOOL/sel_L$L.bin" ]; do sleep 30; done
        sleep 3    # sel lands via rename (atomic) — tiny grace only
    else
        until scp -o BatchMode=yes -q "$M1:$SPOOL_M1/sel_L$L.bin" "$SPOOL/" 2>/dev/null; do sleep 30; done
    fi
    # shellcheck disable=SC2086
    DS4_Z_DUMP_DIR="$ZDIR" ./calib_run $CFG --load-sel "$SPOOL" --layers "$L" || { echo "CONSUME-FAIL L$L"; continue; }
    [ -f "$ZDIR/z_L$L.bin" ] || { echo "CONSUME-NO-Z L$L (dump dir?)"; continue; }   # never silently drop a solve
    rm -f "$SPOOL/sel_L$L.bin"
    # KEEP_REMOTE=1: steal lane — leave M1's sel copy so the other host's own
    # consumer never blocks on a deleted file (it re-solves at worst: same
    # closed-form config => identical z, harmless duplicate minutes).
    [ "$LOCAL" = "1" ] || [ "${KEEP_REMOTE:-0}" = "1" ] || \
        ssh -o BatchMode=yes "$M1" "rm -f $SPOOL_M1/sel_L$L.bin" 2>/dev/null || true
    # CAP_CLEAN=1: 滚动回收该层 cap npy（本机侧），长跑不再撑爆盘
    if [ "${CAP_CLEAN:-0}" = "1" ] && [ -n "${CAP_DIR:-}" ]; then
        rm -f "$CAP_DIR/ffn_in_L$L.npy" "$CAP_DIR/routed_L$L.npy"               "$CAP_DIR/route_logits_L$L.npy" "$CAP_DIR/route_logits_teacher_L$L.npy"
    fi
    echo "consumed L$L"
done
echo "CONSUME-DONE"
