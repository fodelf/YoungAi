#!/bin/sh
# v2r2_sentinel.sh — EF 二轮的小步判决门（铁律：慢投入只给已验证方向）。
# 只解 5 个哨兵层（深层赌注 L38/40/42 + 中层刷新 L18/20），拼进 r1 的 z 集,
# 配对短 NLL 对 r1 原版；过门（≤ r1−0.02）才允许全 43 层求解。
# 总成本 ~35 分钟 vs 全量 ~2.5h。
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
ROOT_M1=${ROOT_M1:-/Users/fodelf/ds4-main}
M1=${M1:-192.168.1.2}
CAP=cap_v2r2
SENT="18 20 38 40 42"
SENT_CSV="18,20,38,40,42"
CFG_X="--solver rrr --rank 64 --lambda 1e-1 --lam-c 0.2 --feat x --w-align 1.0 --eig-iters 100 --threads 5"
cd "$ROOT" || exit 1
say() { echo "[sent $(date +%H:%M)] $*" >&2; }

say "1/5 npy 转换（两段合一）"
ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && \
  cap_work/venv/bin/python gguf-tools/go-onebit/calib/pyfwd/cap_raw2npy.py cap_v2r2a_raw $CAP 0-20 2>&1 | tail -1 && \
  cap_work/venv/bin/python gguf-tools/go-onebit/calib/pyfwd/cap_raw2npy.py cap_v2r2b_raw $CAP 21-42 2>&1 | tail -1" || exit 1

say "2/5 哨兵注入 ($SENT_CSV)"
ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && env DS4_HF=$ROOT_M1/hf/DeepSeek-V4-Flash-Base \
  nice -n 5 cap_work/venv/bin/python gguf-tools/go-onebit/calib/pyfwd/teacher_inject.py \
  --cap $CAP --layers $SENT_CSV 2>&1 | tail -1" || exit 1

say "3/5 哨兵泵+解（M1 本地一条龙）"
cat > /tmp/r2sent_m1.sh <<LAUNCH
#!/bin/sh
cd $ROOT_M1
env CAP=$ROOT_M1/$CAP SPOOL=$ROOT_M1/sel_spool_r2s NX=10240 \
  sh gguf-tools/go-onebit/cluster/e5_pump.sh "$SENT" || exit 1
export LOCAL=1 SPOOL_M1=$ROOT_M1/sel_spool_r2s ROOT=$ROOT_M1
export CFG="$CFG_X"
sh gguf-tools/go-onebit/cluster/e5_consume.sh "$SENT" $ROOT_M1/zdump_r2s
LAUNCH
scp -o BatchMode=yes -q /tmp/r2sent_m1.sh "$M1:/tmp/" || exit 1
ssh -o BatchMode=yes "$M1" "mkdir -p $ROOT_M1/zdump_r2s && sh /tmp/r2sent_m1.sh" || exit 1

say "4/5 拼装哨兵侧车（r1 z 集 + 5 哨兵覆盖）"
rm -rf zdump_r2s && mkdir -p zdump_r2s
cp zdump_v2r1/z_L*.bin zdump_r2s/
scp -o BatchMode=yes -q "$M1:$ROOT_M1/zdump_r2s/z_L*.bin" zdump_r2s/ || exit 1
( cd gguf-tools && ./emit_z --out "$ROOT/gguf/sidecars/go-r2sent.gguf" --zdir "../zdump_r2s" --layers 43 | tail -1 )

say "5/5 配对短判（双 heldout）"
export DS4_METAL_EXPERT_PREAD=1 DS4_METAL_EXPERT_PREFETCH_AHEAD=1
A=$(./ds4 -m gguf/ds4-go1b-v2.gguf --corr gguf/sidecars/go-r2sent.gguf --perplexity-file /tmp/heldout_short.txt --metal 2>&1 | grep -oa 'avg_nll=[0-9.]*' | cut -d= -f2)
B=$(./ds4 -m gguf/ds4-go1b-v2.gguf --corr gguf/sidecars/go-r2sent.gguf --perplexity-file /tmp/heldout_new.txt --metal 2>&1 | grep -oa 'avg_nll=[0-9.]*' | cut -d= -f2)
echo "SENTINEL-VERDICT r2sent@old=$A (r1=3.300) r2sent@new=$B (r1 z43@new=5.082)"
