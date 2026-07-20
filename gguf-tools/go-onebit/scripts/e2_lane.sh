#!/bin/sh
# e2_lane.sh — one parameterized aggregate-RRR lane (either host).
# Fixed defaults = the current E2 protocol (heldout tail chunks, ntest 1024,
# eig 100, lam_c 0.2); everything else via env. Prints the two verdict lines.
#   usage: LAYERS=8 NX=10240 RANK=64 LAMBDA=1e-2 FEAT=x UMODE=rrr \
#          CAP=... HF=... [WSMOOTH=0] [ALPHA=0] [ZDIR=] [GOSTATS=] e2_lane.sh
set -u
ROOT=${ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
GT=$ROOT/gguf-tools
LAYERS=${LAYERS:-8}
NX=${NX:-10240}
RANK=${RANK:-64}
LAMBDA=${LAMBDA:-1e-2}
FEAT=${FEAT:-x}
UMODE=${UMODE:-rrr}
WSMOOTH=${WSMOOTH:-0}
WALIGN=${WALIGN:-0}
ALPHA=${ALPHA:-0}
THREADS=${THREADS:-6}
CAP=${CAP:-$ROOT/cap_m4}
HF=${HF:-$ROOT/hf/DeepSeek-V4-Flash-Base}
ZDIR=${ZDIR:-}
GOSTATS=${GOSTATS:-}

cd "$GT" || exit 1
EXTRA=""
[ -n "$GOSTATS" ] && EXTRA="$EXTRA --go-stats $GOSTATS"
[ "$WSMOOTH" != "0" ] && EXTRA="$EXTRA --w-smooth $WSMOOTH"
[ "$WALIGN" != "0" ] && EXTRA="$EXTRA --w-align $WALIGN"
[ "$ALPHA" != "0" ] && EXTRA="$EXTRA --alpha-router $ALPHA"
echo "=== e2_lane L=$LAYERS nx=$NX k=$RANK lam=$LAMBDA feat=$FEAT umode=$UMODE smooth=$WSMOOTH walign=$WALIGN alpha=$ALPHA gostats=${GOSTATS:-none} ==="
# shellcheck disable=SC2086
DS4_Z_DUMP_DIR="${ZDIR}" ./calib_run --solver rrr --layers "$LAYERS" --nx "$NX" --rank "$RANK" \
    --heldout-frac 0.17 --ntest 1024 --eig-iters 100 --threads "$THREADS" --lam-c 0.2 \
    --lambda "$LAMBDA" --feat "$FEAT" --umode "$UMODE" $EXTRA \
    --cap "$CAP" --hf "$HF" 2>&1 | grep -E "scalar-gain|TR base_cos|assembled|dumped"
