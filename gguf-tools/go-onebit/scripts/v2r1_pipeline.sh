#!/bin/sh
# v2r1_pipeline.sh — v2 Θ_fix round-1 z factory relay (run on M4 after
# cap_v2r1 capture is DONE): transfer raw -> npy -> teacher inject (hash
# branch included) -> interleaved sel pump -> dual-host consumers with
# steal semantics -> collect z -> emit sidecar (phi=x).
# Gates (NLL both heldouts + speed arm) run by hand on completion — judgment.
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
ROOT_M1=${ROOT_M1:-/Users/fodelf/ds4-main}
M1=${M1:-192.168.1.2}
CAP=${CAP:-cap_v2r1}
ZTAG=${ZTAG:-v2r1}
CFG_X="--solver rrr --rank 64 --lambda 1e-1 --lam-c 0.2 --feat x --w-align 1.0 --eig-iters 100 --threads 5"
cd "$ROOT" || exit 1

echo "[v2r1] 1/6 transfer raw shards -> M1" >&2
scp -o BatchMode=yes -rq "$CAP" "$M1:$ROOT_M1/${CAP}_raw" || exit 1

echo "[v2r1] 2/6 convert raw -> npy (M1)" >&2
ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && cap_work/venv/bin/python \
  gguf-tools/go-onebit/calib/pyfwd/cap_raw2npy.py ${CAP}_raw $CAP 0-24 2>&1 | tail -1 && rm -rf ${CAP}_raw" || exit 1

echo "[v2r1] 3/6 teacher inject 25 layers (M1, hash branch for L0-2)" >&2
ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && env DS4_HF=$ROOT_M1/hf/DeepSeek-V4-Flash-Base \
  nice -n 5 cap_work/venv/bin/python gguf-tools/go-onebit/calib/pyfwd/teacher_inject.py \
  --cap $CAP --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 \
  2>&1 | tail -2" || exit 1

echo "[v2r1] 4/6 interleaved pump (M1) + dual consumers" >&2
PUMP_ORDER="0 24 1 23 2 22 3 21 4 20 5 19 6 18 7 17 8 16 9 15 10 14 11 13 12"
ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && mkdir -p sel_spool_$ZTAG zdump_$ZTAG && \
  nohup env CAP=$ROOT_M1/$CAP SPOOL=$ROOT_M1/sel_spool_$ZTAG NX=10240 \
  sh gguf-tools/go-onebit/cluster/e5_pump.sh '$PUMP_ORDER' \
  < /dev/null > /tmp/${ZTAG}_pump.log 2>&1 & disown; echo PUMP-UP" || exit 1
# Two-step launcher: NEVER pass quoted-with-spaces env (CFG) inline through
# ssh — the nested quotes leave the remote shell waiting on an unterminated
# string and the session hangs forever (bit this pipeline once). Write a
# launcher file, scp it, exec it.
cat > /tmp/${ZTAG}_m1_consume.sh <<LAUNCH
#!/bin/sh
cd $ROOT_M1
export LOCAL=1 SPOOL_M1=$ROOT_M1/sel_spool_$ZTAG ROOT=$ROOT_M1
export CFG="$CFG_X"
exec sh gguf-tools/go-onebit/cluster/e5_consume.sh "24 23 22 21 20 19 18 17 16 15 14 13" $ROOT_M1/zdump_$ZTAG
LAUNCH
scp -o BatchMode=yes -q /tmp/${ZTAG}_m1_consume.sh "$M1:/tmp/" || exit 1
ssh -o BatchMode=yes "$M1" "nohup sh /tmp/${ZTAG}_m1_consume.sh < /dev/null > /tmp/${ZTAG}_consume_m1.log 2>&1 & disown; echo M1-CONSUMER-UP" || exit 1
mkdir -p "$ROOT/zdump_$ZTAG" "$ROOT/sel_spool_$ZTAG"
env SPOOL_M1="$ROOT_M1/sel_spool_$ZTAG" SPOOL="$ROOT/sel_spool_$ZTAG" CFG="$CFG_X" KEEP_REMOTE=1 \
    sh gguf-tools/go-onebit/cluster/e5_consume.sh "0 1 2 3 4 5 6 7 8 9 10 11 12" "$ROOT/zdump_$ZTAG"

echo "[v2r1] 5/6 wait M1 lane + collect z" >&2
until [ "$(ssh -o BatchMode=yes "$M1" "ls $ROOT_M1/zdump_$ZTAG/z_L*.bin 2>/dev/null | wc -l" | tr -d ' ')" -ge 12 ]; do
    sleep 90
done
scp -o BatchMode=yes -q "$M1:$ROOT_M1/zdump_$ZTAG/z_L*.bin" "$ROOT/zdump_$ZTAG/"

echo "[v2r1] 6/6 emit sidecar (phi=x)" >&2
cd "$ROOT/gguf-tools" && ./emit_z --out "$ROOT/gguf/ds4-go1b-v2-corr-r1.gguf" \
    --zdir "$ROOT/zdump_$ZTAG" --layers 43 | tail -1
N=$(ls "$ROOT/zdump_$ZTAG"/z_L*.bin 2>/dev/null | wc -l | tr -d ' ')
echo "V2R1-PIPELINE-DONE z=$N/25 sidecar=gguf/ds4-go1b-v2-corr-r1.gguf"
