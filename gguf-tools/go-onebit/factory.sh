#!/bin/sh
# factory.sh — go-onebit 工厂总入口（用户可直接跑，双机流水，幂等续跑）
#
#   sh gguf-tools/go-onebit/factory.sh              # 全默认：产出三件套
#   FRESH_MODEL=1 sh .../factory.sh                 # 强制重新量化 Θ_fix
#   KEYWORD=gin FRESH_CORPUS=1 sh .../factory.sh    # 换关键字重抓语料
#
# 产物（三件套 + 后训练入口）：
#   ① Θ_fix 1-bit 模型   gguf/ds4-go1b-v2.gguf        （双机集群量化, L_fix imatrix 在内）
#   ② z 隐变量侧车        gguf/ds4-go1b-v2-corr-r1.gguf（每层 U,V,C,b,β[,δ]）
#   ③ 四损失             L_align/L_smooth/L_cls度量 在求解 CFG；L_fix 在 ①
#   ④ 后训练入口          corpus/{books,method}/ 放料 + 重跑本脚本 = 新一轮 z
# 用法收尾：./ds4 -m gguf/ds4-go1b-v2.gguf --corr gguf/ds4-go1b-v2-corr-r1.gguf -p ...
#
# 每阶段幂等（产物存在即跳过）；双机分工=注册表按 shard 本地性；进度全走 stderr。
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
ROOT_M1=${ROOT_M1:-/Users/fodelf/ds4-main}
M1=${M1:-192.168.1.2}
KEYWORD=${KEYWORD:-go}
MODEL=${MODEL:-$ROOT/gguf/ds4-go1b-v2.gguf}
SIDECAR=${SIDECAR:-$ROOT/gguf/ds4-go1b-v2-corr-r1.gguf}
CAP=${CAP:-cap_v2r1}
ZTAG=${ZTAG:-v2r1}
CORPUS=${CORPUS:-$ROOT/gguf-tools/go-onebit/corpus/build/go_merged_28k.txt}
CFG_X=${CFG_X:-"--solver rrr --rank 64 --lambda 1e-1 --lam-c 0.2 --feat x --w-align 1.0 --eig-iters 100 --threads 5"}
say() { echo "[factory $(date +%H:%M)] $*" >&2; }
cd "$ROOT" || exit 1

# ---- 0. 语料（幂等；FRESH_CORPUS=1 重抓）------------------------------------
if [ ! -f "$CORPUS" ] || [ "${FRESH_CORPUS:-0}" = 1 ]; then
    say "stage0 语料: harvest($KEYWORD) + build 双链路"
    ( cd gguf-tools/go-onebit/corpus && \
      KEYWORD=$KEYWORD OUT=raw python3 harvest_repos.py && \
      KEYWORD=$KEYWORD RAW=raw OUT=build python3 corpus_build.py ) || exit 1
    head -c 98000 "gguf-tools/go-onebit/corpus/build/${KEYWORD}_mixed.txt" > "$CORPUS"
else say "stage0 语料已在: $CORPUS (跳过)"; fi

# ---- 1. Θ_fix 双机集群量化（幂等；FRESH_MODEL=1 重生成）---------------------
if [ ! -f "$MODEL" ] || [ "${FRESH_MODEL:-0}" = 1 ]; then
    say "stage1 双机集群量化 -> $MODEL (~2h)"
    [ -f /tmp/tmpl_hdr.gguf ] || curl -sL -r 0-209715199 \
      "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf" \
      -o /tmp/tmpl_hdr.gguf
    env IMATRIX=/tmp/gostats_full.dat sh gguf-tools/go-onebit/cluster/quant_dual.sh "$MODEL" --experts go1b || exit 1
else say "stage1 模型已在: $(du -h "$MODEL" | cut -f1) (跳过)"; fi

# ---- 2. 引擎采集（幂等）------------------------------------------------------
if [ ! -d "$CAP" ] || [ "$(ls "$CAP" 2>/dev/null | wc -l | tr -d ' ')" -lt 100 ]; then
    say "stage2 引擎采集: $CORPUS (裸 Θ_fix 轨迹, L0-24, ~1h)"
    env MODEL="$MODEL" CTX=32768 sh gguf-tools/go-onebit/scripts/cap_ef2.sh "$CAP" "$CORPUS" - || exit 1
else say "stage2 采集已在: $CAP (跳过)"; fi

# ---- 3. 传输+转换+教师注入（幂等：M1 routed 目标齐则跳）----------------------
NTGT=$(ssh -o BatchMode=yes "$M1" "ls $ROOT_M1/$CAP/routed_L*.npy 2>/dev/null | wc -l" | tr -d ' ')
if [ "${NTGT:-0}" -lt 25 ]; then
    say "stage3 传输->npy->注入 (M1)"
    scp -o BatchMode=yes -rq "$CAP" "$M1:$ROOT_M1/${CAP}_raw"
    ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && cap_work/venv/bin/python \
      gguf-tools/go-onebit/calib/pyfwd/cap_raw2npy.py ${CAP}_raw $CAP 0-24 2>&1 | tail -1 && rm -rf ${CAP}_raw" || exit 1
    ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && env DS4_HF=$ROOT_M1/hf/DeepSeek-V4-Flash-Base \
      nice -n 5 cap_work/venv/bin/python gguf-tools/go-onebit/calib/pyfwd/teacher_inject.py \
      --cap $CAP --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 2>&1 | tail -1" || exit 1
else say "stage3 教师目标已齐 ($NTGT/25) (跳过)"; fi

# ---- 4. 双机窃取式 z 求解（幂等：已解层自动跳；已有泳道在跑则只等）------------
say "stage4 双机求解 (交错泵 + M4 L0-12 / M1 L24-13)"
if ! ssh -o BatchMode=yes "$M1" "pgrep -f e5_pump.sh >/dev/null" && \
   [ "$(ssh -o BatchMode=yes "$M1" "ls $ROOT_M1/sel_spool_$ZTAG/sel_L*.bin 2>/dev/null | wc -l" | tr -d ' ')" -lt 1 ] && \
   [ "$(ls zdump_$ZTAG/z_L*.bin 2>/dev/null | wc -l | tr -d ' ')" -lt 25 ]; then
    ssh -o BatchMode=yes "$M1" "cd $ROOT_M1 && mkdir -p sel_spool_$ZTAG && \
      nohup env CAP=$ROOT_M1/$CAP SPOOL=$ROOT_M1/sel_spool_$ZTAG NX=10240 \
      sh gguf-tools/go-onebit/cluster/e5_pump.sh '0 24 1 23 2 22 3 21 4 20 5 19 6 18 7 17 8 16 9 15 10 14 11 13 12' \
      < /dev/null > /tmp/${ZTAG}_pump.log 2>&1 & disown; echo PUMP-UP"
fi
if ! ssh -o BatchMode=yes "$M1" "pgrep -f e5_consume.sh >/dev/null"; then
    cat > "/tmp/${ZTAG}_m1c.sh" <<LAUNCH
#!/bin/sh
cd $ROOT_M1; export LOCAL=1 SPOOL_M1=$ROOT_M1/sel_spool_$ZTAG ROOT=$ROOT_M1; export CFG="$CFG_X"
exec sh gguf-tools/go-onebit/cluster/e5_consume.sh "24 23 22 21 20 19 18 17 16 15 14 13" $ROOT_M1/zdump_$ZTAG
LAUNCH
    scp -o BatchMode=yes -q "/tmp/${ZTAG}_m1c.sh" "$M1:/tmp/" && \
    ssh -o BatchMode=yes "$M1" "mkdir -p $ROOT_M1/zdump_$ZTAG; nohup sh /tmp/${ZTAG}_m1c.sh < /dev/null > /tmp/${ZTAG}_consume_m1.log 2>&1 & disown; echo M1-CONSUMER-UP"
fi
mkdir -p "zdump_$ZTAG" "sel_spool_$ZTAG"
if ! pgrep -f "e5_consume.sh 0 1 2" >/dev/null; then
    env KEEP_REMOTE=1 SPOOL_M1="$ROOT_M1/sel_spool_$ZTAG" SPOOL="$ROOT/sel_spool_$ZTAG" CFG="$CFG_X" \
        sh gguf-tools/go-onebit/cluster/e5_consume.sh "0 1 2 3 4 5 6 7 8 9 10 11 12" "$ROOT/zdump_$ZTAG"
fi
until [ "$(ssh -o BatchMode=yes "$M1" "ls $ROOT_M1/zdump_$ZTAG/z_L*.bin 2>/dev/null | wc -l" | tr -d ' ')" -ge 12 ]; do
    say "stage4 等 M1 深层泳道: $(ssh -o BatchMode=yes "$M1" "ls $ROOT_M1/zdump_$ZTAG/z_L*.bin 2>/dev/null | wc -l" | tr -d ' ')/12"
    sleep 120
done
scp -o BatchMode=yes -q "$M1:$ROOT_M1/zdump_$ZTAG/z_L*.bin" "zdump_$ZTAG/"

# ---- 5. emit 三件套 + 门 ------------------------------------------------------
say "stage5 emit 侧车 + 短判门"
( cd gguf-tools && ./emit_z --out "$SIDECAR" --zdir "../zdump_$ZTAG" --layers 43 | tail -1 ) || exit 1
export DS4_METAL_EXPERT_PREAD=1 DS4_METAL_EXPERT_PREFETCH_AHEAD=1
N=$(ls "zdump_$ZTAG"/z_L*.bin | wc -l | tr -d ' ')
say "门: 短片配对 NLL (bare vs +z)"
B=$(./ds4 -m "$MODEL" --perplexity-file /tmp/heldout_short.txt --metal 2>&1 | grep -oa 'avg_nll=[0-9.]*')
Z=$(./ds4 -m "$MODEL" --corr "$SIDECAR" --perplexity-file /tmp/heldout_short.txt --metal 2>&1 | grep -oa 'avg_nll=[0-9.]*')
echo "FACTORY-DONE z=$N/25 sidecar=$SIDECAR bare:$B +z:$Z"
echo "run: ./ds4 -m $MODEL --corr $SIDECAR -p '<Go题>' --temp 0 --metal"
