#!/bin/sh
# v2r2_relay.sh — EF round-2 on v2: student = Θ_fix v2 + r1 sidecar, corpus =
# merged 28k, ALL 43 layers. Consumes the split captures (cap_v2r2a_raw 0-20 +
# cap_v2r2b_raw 21-42 already on M1), then the proven chain: npy → inject →
# interleaved pump → dual consumers (two-step launcher, steal semantics) →
# collect → emit r2 sidecar. Gates run by hand after.
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
ROOT_M1=${ROOT_M1:-/Users/fodelf/ds4-main}
M1=${M1:-192.168.1.2}
CAP=cap_v2r2
ZTAG=v2r2
CFG_X="--solver rrr --rank 64 --lambda 1e-1 --lam-c 0.2 --feat x --w-align 1.0 --eig-iters 100 --threads 5"
cd "$ROOT" || exit 1
say() { echo "[r2 $(date +%H:%M)] $*" >&2; }

NTGT=$(ssh -o BatchMode=yes "$M1" "ls $ROOT_M1/$CAP/routed_L*.npy 2>/dev/null | wc -l" | tr -d ' ')
if [ "${NTGT:-0}" -ge 43 ]; then
    say "1-2/5 已就绪（npy+43目标在位），跳过"
else
say "1/5 npy 转换（两段合一目录）"
ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && \
  cap_work/venv/bin/python gguf-tools/go-onebit/calib/pyfwd/cap_raw2npy.py cap_v2r2a_raw $CAP 0-20 && \
  cap_work/venv/bin/python gguf-tools/go-onebit/calib/pyfwd/cap_raw2npy.py cap_v2r2b_raw $CAP 21-42 && \
  rm -rf cap_v2r2a_raw cap_v2r2b_raw" || exit 1

say "2/5 教师注入 43 层（M1）"
ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && env DS4_HF=$ROOT_M1/hf/DeepSeek-V4-Flash-Base \
  nice -n 5 cap_work/venv/bin/python gguf-tools/go-onebit/calib/pyfwd/teacher_inject.py \
  --cap $CAP --layers $(seq -s, 0 42) " || exit 1

fi

say "3/5 交错泵 + 双泳道（M4 0-21 / M1 42-22）"
PUMP_ORDER=$(python3 -c "
lo,hi=0,42
o=[]
while lo<=hi:
    o.append(str(lo)); lo+=1
    if lo<=hi: o.append(str(hi)); hi-=1
print(' '.join(o))")
ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && mkdir -p sel_spool_$ZTAG zdump_$ZTAG && \
  nohup env CAP=$ROOT_M1/$CAP SPOOL=$ROOT_M1/sel_spool_$ZTAG NX=10240 \
  sh gguf-tools/go-onebit/cluster/e5_pump.sh '$PUMP_ORDER' \
  < /dev/null > /tmp/${ZTAG}_pump.log 2>&1 & disown; echo PUMP-UP" || exit 1
cat > "/tmp/${ZTAG}_m1c.sh" <<LAUNCH
#!/bin/sh
cd $ROOT_M1; export LOCAL=1 SPOOL_M1=$ROOT_M1/sel_spool_$ZTAG ROOT=$ROOT_M1; export CFG="$CFG_X"
exec sh gguf-tools/go-onebit/cluster/e5_consume.sh "42 41 40 39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22" $ROOT_M1/zdump_$ZTAG
LAUNCH
scp -o BatchMode=yes -q "/tmp/${ZTAG}_m1c.sh" "$M1:/tmp/" || exit 1
ssh -o BatchMode=yes "$M1" "nohup sh /tmp/${ZTAG}_m1c.sh < /dev/null > /tmp/${ZTAG}_consume_m1.log 2>&1 & disown; echo M1-UP" || exit 1
mkdir -p "zdump_$ZTAG" "sel_spool_$ZTAG"
env KEEP_REMOTE=1 SPOOL_M1="$ROOT_M1/sel_spool_$ZTAG" SPOOL="$ROOT/sel_spool_$ZTAG" CFG="$CFG_X" \
    sh gguf-tools/go-onebit/cluster/e5_consume.sh "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21" "$ROOT/zdump_$ZTAG"

say "4/5 收集 43 层"
until [ "$(ls zdump_$ZTAG/ 2>/dev/null | grep -c '^z_L')" -ge 43 ]; do
    scp -o BatchMode=yes -q "$M1:$ROOT_M1/zdump_$ZTAG/z_L*.bin" "zdump_$ZTAG/" 2>/dev/null
    sleep 90
done

say "5/5 emit r2 侧车"
( cd gguf-tools && ./emit_z --out "$ROOT/gguf/sidecars/go-r2.gguf" --zdir "../zdump_$ZTAG" --layers 43 | tail -1 )
echo "V2R2-RELAY-DONE $(ls zdump_$ZTAG/ | grep -c '^z_L')/43"
