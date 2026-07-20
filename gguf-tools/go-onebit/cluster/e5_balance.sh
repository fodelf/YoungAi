#!/bin/sh
# e5_balance.sh — dual-host WORK STEALING for the E5 full-layer solve.
# Rule (user, 2026-07-02): when one host finishes its static share it must
# help the other. Feasible direction this round: M1 holds ALL HF shards so it
# can steal M4's remaining layers; M4 lacks shards 14-46 (disk) so the reverse
# is impossible for solve work — encoded here, not assumed.
#
# Convergence: M4's own lane solves L0..12 in ASCENDING order; the stealer
# hands M1 layers from L12 DESCENDING. Completion marker = z_L{n}.bin in M4's
# zdump. Two pointers meet => at most one duplicated layer (same closed-form
# config => identical file, harmless).
#
# Run on M4 (nohup). Env: FEAT/LAMBDA/RANK/WALIGN/NX match e5_solve_dual.sh.
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
ROOT_M1=${ROOT_M1:-/Users/fodelf/ds4-main}
M1=${M1:-192.168.1.2}
ZDIR_M4=${ZDIR_M4:-$ROOT/zdump_rrr}
ZDIR_M1=${ZDIR_M1:-$ROOT_M1/zdump_rrr}
CAP_M1=${CAP_M1:-$ROOT_M1/cap_m1_v2}
FEAT=${FEAT:-yhat}; LAMBDA=${LAMBDA:-1e-1}; RANK=${RANK:-64}; WALIGN=${WALIGN:-1.0}; NX=${NX:-10240}
CFG="--solver rrr --nx $NX --rank $RANK --heldout-frac 0.17 --ntest 1024 --eig-iters 100 --lam-c 0.2 --lambda $LAMBDA --feat $FEAT --w-align $WALIGN"

log() { echo "[balance $(date +%H:%M:%S)] $*"; }

# 1. wait for M1's static lanes to drain (its own 30 layers)
while true; do
    N=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$M1" 'pgrep -f "calib_run --solver rrr" | wc -l' 2>/dev/null | tr -d ' ')
    [ "${N:-1}" = "0" ] && break
    sleep 120
done
log "M1 static lanes drained; stealing M4 leftovers (descending from L12)"

# 2. steal M4's unfinished layers, one at a time, memory-safe single proc
L=12
while [ $L -ge 0 ]; do
    if [ ! -f "$ZDIR_M4/z_L$L.bin" ]; then
        log "steal L$L -> M1"
        ssh -o BatchMode=yes "$M1" "cd $ROOT_M1/gguf-tools && DS4_Z_DUMP_DIR=$ZDIR_M1 ./calib_run $CFG --threads 6 --layers $L --cap $CAP_M1 --hf $ROOT_M1/hf/DeepSeek-V4-Flash-Base" >> /tmp/e5_balance.log 2>&1
        if [ -f "$ZDIR_M4/z_L$L.bin" ]; then
            log "L$L finished locally by M4 while stealing — dropping duplicate"
        else
            scp -q "$M1:$ZDIR_M1/z_L$L.bin" "$ZDIR_M4/" && log "L$L stolen+landed"
        fi
    fi
    L=$((L - 1))
done
log "balance pass complete: $(ls "$ZDIR_M4"/z_L*.bin 2>/dev/null | wc -l | tr -d ' ')/13 shallow layers present"
