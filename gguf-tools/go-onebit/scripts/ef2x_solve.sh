#!/bin/sh
# ef2x_solve.sh — EF2x round orchestration (run on M4, after teacher_inject
# DONE on M1): solve L0-24 with feat=x (the speed-side of the phi verdict:
# phi=x sidecars decode free at 0.97 t/s; phi=yhat pays a ~2.4x barrier tax)
# on the ENGINE student trajectory (cap_ef2) with engine/teacher-injected
# targets and the fresh harvested corpus.
#
# Lanes (dual-host, even split — pump orders M4's share first so its consumer
# starts immediately; M1 self-consumes its share LOCAL behind the pump):
#   M1: pump sel L0..24 (shard-bound)   -> sel_spool_ef2
#   M4: consume L0-12  (scp mode)       -> zdump_ef2x
#   M1: consume L13-24 (LOCAL mode)     -> zdump_ef2x (M1) -> collected back
# Champion config from E2/E3 except feat: rank 64, lambda 1e-1, lam-c 0.2,
# w-align 1.0, NX 10240 (comparability with all prior rounds).
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
ROOT_M1=${ROOT_M1:-/Users/fodelf/ds4-main}
M1=${M1:-192.168.1.2}
CFG_X="--solver rrr --rank 64 --lambda 1e-1 --lam-c 0.2 --feat x --w-align 1.0 --eig-iters 100 --threads 5"
M4_LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12"
M1_LAYERS="13 14 15 16 17 18 19 20 21 22 23 24"
PUMP_ORDER="$M4_LAYERS $M1_LAYERS"
cd "$ROOT" || exit 1

echo "[ef2x] 1/3 launch M1 pump (sel assembly, shard-bound)" >&2
# < /dev/null: a remote nohup'd job inheriting the ssh channel's stdin keeps
# the ssh session open forever (this exact hang stalled the first run).
if ssh -o BatchMode=yes "$M1" "pgrep -f 'e5_pump.sh' > /dev/null"; then
    echo "[ef2x] pump already running on M1 — not relaunching" >&2
else
    ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && mkdir -p sel_spool_ef2 && \
      nohup env CAP=$ROOT_M1/cap_ef2 SPOOL=$ROOT_M1/sel_spool_ef2 NX=10240 \
      sh gguf-tools/go-onebit/cluster/e5_pump.sh '$PUMP_ORDER' \
      < /dev/null > /tmp/ef2x_pump.log 2>&1 & disown" < /dev/null || exit 1
fi

echo "[ef2x] 2/3 launch M1 LOCAL consumer (L13-24, feat=x)" >&2
if ssh -o BatchMode=yes "$M1" "pgrep -f 'e5_consume.sh' > /dev/null"; then
    echo "[ef2x] M1 consumer already running — not relaunching" >&2
else
    ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && mkdir -p zdump_ef2x && \
      nohup env LOCAL=1 SPOOL_M1=$ROOT_M1/sel_spool_ef2 ROOT=$ROOT_M1 CFG=\"$CFG_X\" \
      sh gguf-tools/go-onebit/cluster/e5_consume.sh '$M1_LAYERS' $ROOT_M1/zdump_ef2x \
      < /dev/null > /tmp/ef2x_consume_m1.log 2>&1 & disown" < /dev/null || exit 1
fi

echo "[ef2x] 3/3 M4 consumer (L0-12, scp mode, feat=x) — foreground of this script" >&2
mkdir -p "$ROOT/zdump_ef2x" "$ROOT/sel_spool_ef2"
env SPOOL_M1="$ROOT_M1/sel_spool_ef2" SPOOL="$ROOT/sel_spool_ef2" CFG="$CFG_X" \
    sh gguf-tools/go-onebit/cluster/e5_consume.sh "$M4_LAYERS" "$ROOT/zdump_ef2x"

echo "[ef2x] M4 lane done; waiting M1 lane + collecting z" >&2
until [ "$(ssh -o BatchMode=yes "$M1" "ls $ROOT_M1/zdump_ef2x/z_L*.bin 2>/dev/null | wc -l" | tr -d ' ')" -ge 12 ]; do
    sleep 60
done
scp -o BatchMode=yes -q "$M1:$ROOT_M1/zdump_ef2x/z_L*.bin" "$ROOT/zdump_ef2x/"
N=$(ls "$ROOT/zdump_ef2x"/z_L*.bin 2>/dev/null | wc -l | tr -d ' ')
echo "EF2X-SOLVE-DONE z_files=$N/25"
