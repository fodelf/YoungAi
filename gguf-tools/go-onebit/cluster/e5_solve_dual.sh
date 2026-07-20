#!/bin/sh
# e5_solve_dual.sh — E5 full-43-layer aggregate-RRR solve, dual-host layout:
#   M4 (this host): layers 0-12  (only shards it holds), 1 lane
#   M1:             layers 13-42, N_LANES parallel lanes (odd/even split)
# Solve config = the E2 winner; override via env. z files land in ZDIR on each
# host (M1's are scp'd back by e5_collect.sh). Every lane has the RSS watchdog
# from calib_split.sh semantics (13 GiB) via a local poll.
#   usage: [FEAT=yhat] [LAMBDA=1e-1] [RANK=64] [WALIGN=?] [NX=10240] e5_solve_dual.sh CAP_M4 CAP_M1
set -u
ROOT_M4=${ROOT_M4:-/Users/fodelf/git/ds4-main}
ROOT_M1=${ROOT_M1:-/Users/fodelf/ds4-main}
M1=${M1:-192.168.1.2}
CAP_M4=${1:?cap dir on M4}
CAP_M1=${2:?cap dir on M1}
FEAT=${FEAT:-yhat}
LAMBDA=${LAMBDA:-1e-1}
RANK=${RANK:-64}
WALIGN=${WALIGN:-0}
NX=${NX:-10240}
ZDIR_M4=${ZDIR_M4:-$ROOT_M4/zdump_rrr}
ZDIR_M1=${ZDIR_M1:-$ROOT_M1/zdump_rrr}
N_LANES=${N_LANES:-2}

mkdir -p "$ZDIR_M4"
CFG="--solver rrr --nx $NX --rank $RANK --heldout-frac 0.17 --ntest 1024 --eig-iters 100 --lam-c 0.2 --lambda $LAMBDA --feat $FEAT"
[ "$WALIGN" != "0" ] && CFG="$CFG --w-align $WALIGN"

# ---- M4 lane: layers 0-12 ----
( cd "$ROOT_M4/gguf-tools" && \
  DS4_Z_DUMP_DIR="$ZDIR_M4" nohup ./calib_run $CFG --threads 6 \
      --layers 0,1,2,3,4,5,6,7,8,9,10,11,12 \
      --cap "$CAP_M4" --hf "$ROOT_M4/hf/DeepSeek-V4-Flash-Base" \
      > /tmp/e5_m4.log 2>&1 & )
echo "M4 lane: L0-12 -> $ZDIR_M4 (log /tmp/e5_m4.log)"

# ---- M1 lanes: layers 13-42 interleaved across N_LANES ----
ssh "$M1" "mkdir -p $ZDIR_M1"
lane=0
while [ $lane -lt "$N_LANES" ]; do
    LAYERS=$(i=$((13 + lane)); s=""; while [ $i -le 42 ]; do s="$s,$i"; i=$((i + N_LANES)); done; echo "${s#,}")
    ssh "$M1" "cd $ROOT_M1/gguf-tools && DS4_Z_DUMP_DIR=$ZDIR_M1 nohup ./calib_run $CFG --threads 4 \
        --layers $LAYERS --cap $CAP_M1 --hf $ROOT_M1/hf/DeepSeek-V4-Flash-Base \
        > /tmp/e5_m1_lane$lane.log 2>&1 & echo 'M1 lane$lane: $LAYERS'"
    lane=$((lane + 1))
done
echo "poll: tail /tmp/e5_m4.log; ssh $M1 tail /tmp/e5_m1_lane*.log; z count: ls $ZDIR_M4 | wc -l + ssh ls $ZDIR_M1"
