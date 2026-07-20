#!/bin/sh
# v2r1_deep_relay.sh — 43 层规格补齐：深层 L25-42 的 传输→npy→注入→交错泵→
# 双机窃取求解→合并 emit 43/43。复用 v2r1 全套已验证组件与两步 launcher 模式。
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
ROOT_M1=${ROOT_M1:-/Users/fodelf/ds4-main}
M1=${M1:-192.168.1.2}
CAP=cap_v2r1_deep
ZTAG=v2r1
CFG_X="--solver rrr --rank 64 --lambda 1e-1 --lam-c 0.2 --feat x --w-align 1.0 --eig-iters 100 --threads 5"
cd "$ROOT" || exit 1
say() { echo "[deep $(date +%H:%M)] $*" >&2; }

say "1/5 传输 raw -> M1"
scp -o BatchMode=yes -rq "$CAP" "$M1:$ROOT_M1/${CAP}_raw" || exit 1

say "2/5 npy 转换 (25-42)"
ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && cap_work/venv/bin/python \
  gguf-tools/go-onebit/calib/pyfwd/cap_raw2npy.py ${CAP}_raw $CAP 25-42 2>&1 | tail -1 && rm -rf ${CAP}_raw" || exit 1

say "3/5 教师注入 18 层"
ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && env DS4_HF=$ROOT_M1/hf/DeepSeek-V4-Flash-Base \
  nice -n 5 cap_work/venv/bin/python gguf-tools/go-onebit/calib/pyfwd/teacher_inject.py \
  --cap $CAP --layers 25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42 2>&1 | tail -1" || exit 1

say "4/5 交错泵 + 双泳道 (M4 25-33 / M1 42-34)"
ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && mkdir -p sel_spool_deep && \
  nohup env CAP=$ROOT_M1/$CAP SPOOL=$ROOT_M1/sel_spool_deep NX=10240 \
  sh gguf-tools/go-onebit/cluster/e5_pump.sh '25 42 26 41 27 40 28 39 29 38 30 37 31 36 32 35 33 34' \
  < /dev/null > /tmp/deep_pump.log 2>&1 & disown; echo PUMP-UP" || exit 1
cat > /tmp/deep_m1c.sh <<LAUNCH
#!/bin/sh
cd $ROOT_M1; export LOCAL=1 SPOOL_M1=$ROOT_M1/sel_spool_deep ROOT=$ROOT_M1; export CFG="$CFG_X"
exec sh gguf-tools/go-onebit/cluster/e5_consume.sh "42 41 40 39 38 37 36 35 34" $ROOT_M1/zdump_$ZTAG
LAUNCH
scp -o BatchMode=yes -q /tmp/deep_m1c.sh "$M1:/tmp/" || exit 1
ssh -o BatchMode=yes "$M1" "nohup sh /tmp/deep_m1c.sh < /dev/null > /tmp/deep_consume_m1.log 2>&1 & disown; echo M1-UP" || exit 1
env KEEP_REMOTE=1 SPOOL_M1="$ROOT_M1/sel_spool_deep" SPOOL="$ROOT/sel_spool_deep" CFG="$CFG_X" \
    sh gguf-tools/go-onebit/cluster/e5_consume.sh "25 26 27 28 29 30 31 32 33" "$ROOT/zdump_$ZTAG"

say "5/5 收集 + 合并 emit 43/43"
until [ "$(ls zdump_$ZTAG/ 2>/dev/null | grep -c '^z_L')" -ge 43 ]; do
    scp -o BatchMode=yes -q "$M1:$ROOT_M1/zdump_$ZTAG/z_L*.bin" "zdump_$ZTAG/" 2>/dev/null
    sleep 90
done
( cd gguf-tools && ./emit_z --out "$ROOT/gguf/ds4-go1b-v2-corr-r1.gguf" --zdir "../zdump_$ZTAG" --layers 43 | tail -1 )
echo "DEEP-RELAY-DONE $(ls zdump_$ZTAG/ | grep -c '^z_L')/43"
