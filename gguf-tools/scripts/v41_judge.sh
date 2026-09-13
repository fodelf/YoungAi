#!/bin/bash
# v41_judge.sh — V4.1 五指标判决(2026-09-11, spark 本机跑)。
#
# 【口径】教师与学生走【同一条前向】(v41_teacher.py), 唯一差别是主干专家有没有过量化 ——
# 差异才纯粹来自权重量化。五指标器复用本仓现役的 gguf-tools/bench/anchor_metrics
# (--ref-raw 与 --student 吃同一种 <i32 S><i32 V><f32 logits> 格式), ★判决器只许有一份★,
# 口径与 V4 时代逐式同源(PPL / Mean KLD / RMS Δp / Same top / Δp(top))。
#
# 【为什么教师要缓存】FP 教师与位宽无关, 跑一次存着; 每加一个位宽只跑学生。教师文件按
# (ids, ntok) 命名, 换语料/换长度会自动另存, 不会拿错锚 —— 锚与 ids 错配会出 PPL 2.4e7
# 这种一眼假的数(V4 时代实撞)。
#
# 用法: v41_judge.sh <ids文件> <ntok> <档位列表...>
#   例: v41_judge.sh gguf/go-onebit/g7/wt2.ids 32 4 3 2 1        标量位宽
#       v41_judge.sh gguf/go-onebit/g7/wt2.ids 512 8:4096         VQ dim:nc
#       v41_judge.sh gguf/go-onebit/g7/wt2.ids 512 amp:<拟合ids>:<拟合ntok>:8:4096:64
#       v41_judge.sh gguf/go-onebit/vqfin41/vqhalf_j.ids 8192 engine engine::<放大器目录>   引擎学生(裸 / 挂放大器)
#       v41_judge.sh ... engine::<放大器目录>:0.25    同上但每层修正缩到 0.25 倍(步长扫描, 2026-09-13)
#       v41_judge.sh gguf/go-onebit/vqfin41/vqhalf_j.ids 8192 engamp:<拟合ids>:8192           引擎上解放大器再判(2026-09-13)
#       v41_judge.sh gguf/go-onebit/vqfin41/vqhalf_j.ids 8192 engkl:<拟合ids>:8192:0.10        蒸馏靶(末层, KL 梯度)解+判(2026-09-13)
#
# 【amp 档在做什么】先用【拟合料】跑一趟在线序贯反修解出 40 层放大器, 再用【判决料】
# 挂着放大器跑一趟打五指标。两份语料必须不同 —— 在判决料上拟合就是自己给自己判卷,
# 09-11 实撞: 拟合料 PPL 比值 1.051 而判决料崩到 55837。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
PY="$HOME/v41env/bin/python"
HF="$ROOT/hf/DeepSeek-V4.1-Flash"
T="$ROOT/gguf-tools/scripts/v41_teacher.py"
AM="$ROOT/gguf-tools/bench/anchor_metrics"
IDS="${1:?ids 文件}"; NTOK="${2:?token 数}"; shift 2
# --act-from <量化份ids>: 用【量化份】语料捕一次激活列权 E[x²], 所有档统一带上 —— 这是
# 量化校准那一环。三份语料各司其职, 缺一环就不是完整配方: 量化份校准 / 反修份拟合放大器 /
# 判决份打分, 谁都不许串岗。
ACTIDS=""
[ "${1:-}" = "--act-from" ] && { ACTIDS="${2:?--act-from 要 ids}"; shift 2; }
OUT="$ROOT/gguf/v41judge"; mkdir -p "$OUT"
# ★语料标识必须带上目录★: gguf/go-onebit/vqhalf/(全域八域: academic/prose/code/euro/
# cyrillic/math/arabic/cjk) 与 gguf/go-onebit/vqfin41/(金融五域) 里的 ids 文件【同名】
# —— 都叫 vqhalf_q/a/j.ids。只取 basename, 两套语料就共用同一个教师锚文件名, 缓存判断
# 是 [ -s "$REF" ] ⇒ 第二套直接拿第一套的锚打分, 不报错, 只出一组看着合理的假数字。
corpus_tag() { echo "$(basename "$(dirname "$1")")_$(basename "$1" .ids)"; }
TAG="$(corpus_tag "$IDS")_n$NTOK"
REF="$OUT/teacher_$TAG.bin"
LOG(){ echo "[judge $(date '+%m-%d %H:%M:%S')] $*"; }

# ★看门狗★: 按 free 的 available 算, 不看 RSS —— 权重是 mmap 的, 那些页算 buff/cache 也计进
# RSS(09-12 凌晨实测: python RSS 报 85G 时 free 的 used 只有 22G), 卡 RSS 95G 实际相当宽松。
# available 掉到 8 GB 以下就是真要吃干 121 GB 了(95GB server 吃干后【假死】只能按电源), 直接杀。
# 写成函数而不是一行内联: 长命令里堆转义塔的看门狗历史上根本没触发过(memory
# watchdog_never_worked_bash32), "从没拦下来"当时被当成"一直很安全"。
WD_MIN_AVAIL_GB=8
WD_LOG=/tmp/v41_mem.log
watchdog() {
    local av
    while sleep 20; do
        av=$(free -g | awk '/^内存|^Mem/{print $7}')
        echo "$(date '+%H:%M:%S') avail=${av}G" >>"$WD_LOG"
        if [ "${av:-99}" -lt "$WD_MIN_AVAIL_GB" ]; then
            echo "★看门狗: available ${av}G < ${WD_MIN_AVAIL_GB}G, 停车★" | tee -a "$WD_LOG"
            pkill -f "v41_teacher[.]py"      # [.] 防自匹配(pkill -f 的模式会命中自己的命令行)
            pkill -f "ds4 -m gguf/v41[/]"; pkill -f "v41_amp_ru[n] "   # 引擎路学生 / 引擎路反修驱动
            return 1
        fi
    done
}
: >"$WD_LOG"; watchdog & WD_PID=$!
trap 'kill $WD_PID 2>/dev/null' EXIT

# ★永不用 grep 包住长任务★(铁律 feedback_progress_observable_mandatory)。
# 09-11 实撞: 教师 10 秒挂在 engram cache 越界(max_seq_len 默认 4096 盖不住 8192 token),
# 而日志里只剩一行"★教师失败★" —— traceback 被 grep 整个吃掉, 退出码判的还是 grep 的
# 不是 python 的, 白白多花一趟去复现。
# 正解: 全量落盘 + 过滤后上屏 + PIPESTATUS 判真实退出码 + 失败时把尾巴打出来。
MODEL_DIR="$HF"     # file: 档临时指向量化目录(v41_quantize 产物, 自带 index.json 与软链)
run_py() {
    local tag="$1" filt="$2"; shift 2
    local lf="$OUT/log_${tag}.txt"
    "$PY" "$T" "$MODEL_DIR" "$@" 2>&1 | tee "$lf" | grep --line-buffered -E "$filt"
    [ "${PIPESTATUS[0]}" = 0 ] && return 0
    LOG "★$tag 失败 —— 全量日志 $lf, 尾部:★"; tail -20 "$lf"; return 1
}

# 引擎学生(2026-09-13): ./ds4 --score-ids 走 V4.1 分块增量前向(部署同路), 可挂 --zchain 放大器目录。ids 截到 NTOK 行
# 落盘再喂引擎(引擎读整个文件)。与 run_py 同一套纪律: 全量日志落盘, 过滤上屏, PIPESTATUS 判真实退出码。
GG_DEFAULT="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-fp4.gguf"
run_eng() {
    local stu="$1" gg="$2" zarg="$3" tag="$4"
    local idsn="$OUT/ids_${TAG}.txt" lf="$OUT/log_${tag}.txt"
    head -n "$NTOK" "$IDS" > "$idsn"
    [ -x "$ROOT/ds4" ] || { LOG "★$ROOT/ds4 没编(make cuda-spark)★"; return 1; }
    file "$ROOT/ds4" | grep -q "ELF.*aarch64" || { LOG "★./ds4 不是 ELF aarch64★"; return 1; }
    "$ROOT/ds4" -m "$gg" --cuda --mem-budget-mb 40000 --score-ids "$idsn" --score-out "$stu" $zarg 2>&1 \
        | tee "$lf" | grep --line-buffered -E "反修|PPL|失败|error|Error|watchdog"
    [ "${PIPESTATUS[0]}" = 0 ] && [ -s "$stu" ] && return 0
    LOG "★$tag 失败 —— 全量日志 $lf, 尾部:★"; tail -20 "$lf"; return 1
}

# ★.so 必须比 .cu 新★: 改了解算器忘重编, 跑的还是上一版 —— 数字全废还看不出来
build_amp_so() {
    local aso="$ROOT/gguf-tools/amp/libv41amp.so"
    if [ ! -s "$aso" ] || [ "$ROOT/gguf-tools/amp/v41_amp_solve.cu" -nt "$aso" ]; then
        LOG "编 libv41amp.so"
        ( cd "$ROOT/gguf-tools/amp" && /usr/local/cuda/bin/nvcc -O3 -fmad=false -shared \
          -Xcompiler -fPIC -o libv41amp.so v41_amp_solve.cu -lcublas -lcusolver ) \
          || { LOG "★libv41amp.so 编译失败★"; return 1; }
    fi
    return 0
}

[ -x "$AM" ] || { LOG "编 anchor_metrics"; make -C "$ROOT/gguf-tools" anchor_metrics || exit 2; }
# ★别信现成的二进制★(09-08 实撞): 整树 scp 过来的可能是 Mac 的 Mach-O, x 位在但一跑就
# "可执行文件格式错误", 而参考前向那时已经跑完 —— 白烧一趟。
file "$AM" | grep -q "ELF.*aarch64" || { LOG "★$AM 不是 ELF aarch64, 重编★";
    rm -f "$AM"; make -C "$ROOT/gguf-tools" anchor_metrics || exit 2; }

if [ ! -s "$REF" ]; then
    LOG "① FP 教师(不量化) → $REF"
    run_py "teacher_$TAG" "配置|完成|PPL|→" --ids "$IDS" --ntok "$NTOK" --out "$REF" || exit 3
else
    LOG "① 教师已在 $(ls -l "$REF" | awk '{printf "%.1f MB", $5/1e6}'), 跳过"
fi

# ---- ⓪ 量化校准: 从【量化份】语料捕激活列权 ----
# 【捕什么】每层 MoE 入口的逐通道能量 E[x²]。VQ 的最近邻按它加权 ⇒ 量化误差优先落在
# 低能量通道上。★这不是训练★: 只统计, 权重语义不动, 无梯度。
# 【为什么 FP 态捕】列权要反映模型真实的激活分布, 拿量化态捕是拿被污染的分布去指导
# 量化自己 —— 自己给自己定标准。
ACTARG=""
if [ -n "$ACTIDS" ]; then
    [ -s "$ACTIDS" ] || { LOG "★量化份 $ACTIDS 不存在★"; exit 4; }
    ACTF="$OUT/act_$(corpus_tag "$ACTIDS")_n${NTOK}.bin"
    if [ ! -s "$ACTF" ]; then
        LOG "⓪ 量化校准: FP 态捕激活 ← $(basename "$ACTIDS") n=$NTOK"
        run_py "act_$(corpus_tag "$ACTIDS")_n$NTOK" "配置|捕获|完成" \
               --ids "$ACTIDS" --ntok "$NTOK" --dump-act "$ACTF" || exit 4
    else
        LOG "⓪ 激活已在 $ACTF, 跳过"
    fi
    ACTARG="--act $ACTF"
fi
# 产物名要能认出带没带校准 —— 带 act 和不带 act 的学生用同一个文件名就是自己骗自己
ACTSFX=""; [ -n "$ACTIDS" ] && ACTSFX="_cal"

# ---- VQ 模式: 参数写成 dim:nc(如 8:4096), 其余当标量位宽 ----
# 【为什么低 bpw 用 dim=8、高 bpw 用 dim=4】索引 bpw = ceil(log2 nc)/dim。dim=8 想上 2.0 bpw
# 要 nc=65536, 码本 1 MB/专家 × 15360 = 16 GB, 光码本就吃掉 0.24 bpw ⇒ 不可行。
# dim=4 在 2~3 bpw 区间码本才几 KB, 可忽略。
CBD="$ROOT/gguf-tools/quantize"
[ -x "$CBD/v41_codebook" ] || { LOG "编 v41_codebook"; ( cd "$CBD" && cc -O3 -std=c99 -o v41_codebook v41_codebook.c -lm ) || exit 2; }
for NB in "$@"; do
  case "$NB" in
    scank:*) # K 扫描诊断: scank:<拟合ids>:<ntok>:<vqdim>:<vqnc>:<跑前几层>
      # 回答"秩该取多少": 每层对一串候选 K 各算一次残差, 末位 K=5120 是完整最小二乘解
      # = 低秩这条路的天花板。只诊断不应用, 不产出放大器。
      IFS=: read -r _ FIT FN VD VN NL <<<"$NB"
      [ -s "$FIT" ] || { LOG "★拟合料 $FIT 不存在★"; continue; }
      ASO="$ROOT/gguf-tools/amp/libv41amp.so"
      if [ ! -s "$ASO" ] || [ "$ROOT/gguf-tools/amp/v41_amp_solve.cu" -nt "$ASO" ]; then
          LOG "编 libv41amp.so"
          ( cd "$ROOT/gguf-tools/amp" && /usr/local/cuda/bin/nvcc -O3 -fmad=false -shared \
            -Xcompiler -fPIC -o libv41amp.so v41_amp_solve.cu -lcublas -lcusolver ) \
            || { LOG "★libv41amp.so 编译失败★"; continue; }
      fi
      LOG "K 扫描: 拟合料 $(basename "$FIT") n=$FN, VQ ${VD}x${VN}, 前 $NL 层"
      run_py "scank_$(corpus_tag "$FIT")_n${FN}_vq${VD}x${VN}_L${NL}" "扫描|\[L[0-9]|^ +K|吃掉" \
             --ids "$FIT" --ntok "$FN" --vq-dim "$VD" --vq-nc "$VN" $ACTARG \
             --amp-scan-k --layers "$NL" || continue
      continue;;
    amp:*) # 反修档: amp:<拟合ids>:<拟合ntok>:<vqdim>:<vqnc>:<K>[:<λ>]
      # λ 不给则用 10.0(held-out K×λ 扫描的峰值)。★别再用 1e-3★: 那是 V4 候选下端的
      # 1/300, 近乎无正则, 实测泛化 −66.5% / 端到端 Σmin 0.6711→0.2431。
      IFS=: read -r _ FIT FN VD VN KK LAM <<<"$NB"
      LAM="${LAM:-10.0}"
      [ -s "$FIT" ] || { LOG "★拟合料 $FIT 不存在★"; continue; }
      # ★.so 必须比 .cu 新★: 改了解算器忘重编, 跑的还是上一版 —— 数字全废还看不出来
      ASO="$ROOT/gguf-tools/amp/libv41amp.so"
      if [ ! -s "$ASO" ] || [ "$ROOT/gguf-tools/amp/v41_amp_solve.cu" -nt "$ASO" ]; then
          LOG "编 libv41amp.so"
          ( cd "$ROOT/gguf-tools/amp" && /usr/local/cuda/bin/nvcc -O3 -fmad=false -shared \
            -Xcompiler -fPIC -o libv41amp.so v41_amp_solve.cu -lcublas -lcusolver ) \
            || { LOG "★libv41amp.so 编译失败★"; continue; }
      fi
      ARM="$OUT/arm_$(corpus_tag "$FIT")_n${FN}_vq${VD}x${VN}${ACTSFX}_k${KK}_lam${LAM}"
      STU="$OUT/stu_${TAG}_vq${VD}x${VN}${ACTSFX}_ampk${KK}_lam${LAM}.bin"
      # ★重跑必须清空放大器目录★: 解算按层落盘, 上一趟解出而这一趟跳过的层会残留,
      # 判决时被 --amp 挂上 —— 混了两代放大器的读数, 查都没法查。
      rm -rf "$ARM"
      LOG "②a 序贯反修: 拟合料 $(basename "$FIT") n=$FN, VQ ${VD}x${VN}, K=$KK λ=$LAM → $ARM"
      run_py "solve_$(basename "$ARM")" "序贯|\[L[0-9]|信任域|完成|PPL" \
             --ids "$FIT" --ntok "$FN" --vq-dim "$VD" --vq-nc "$VN" $ACTARG \
             --amp-online "$ARM" --amp-k "$KK" --amp-lam "$LAM" || continue
      LOG "②b 学生 = VQ ${VD}x${VN} + 放大器 K=$KK λ=$LAM (判决料 $(basename "$IDS") n=$NTOK)"
      run_py "stu_${TAG}_ampk${KK}_lam${LAM}" "反修\]|完成|PPL|路由" \
             --ids "$IDS" --ntok "$NTOK" --vq-dim "$VD" --vq-nc "$VN" $ACTARG \
             --amp "$ARM" --out "$STU" || continue
      LOG "③ 五指标 VQ ${VD}x${VN} + amp K=$KK λ=$LAM"
      "$AM" --ref-raw "$REF" --ids "$IDS" --student "$STU" || LOG "★判决失败★"
      continue;;
    scankfile:*) # 落盘学生的 K×λ 扫描: scankfile:<量化目录>:<拟合ids>:<ntok>:<跑前几层>[:<白化 0/1/2>]
      # 与 scank 同, 只是学生从文件读回, FP 靶从 --fp-dir(HF)装; held-out 按 <拟合ids>.layout 分层
      IFS=: read -r _ MD FIT FN NL WH <<<"$NB"; WH="${WH:-0}"
      [ -s "$MD/model.safetensors.index.json" ] || { LOG "★$MD 不是量化目录★"; continue; }
      [ -s "$FIT" ] || { LOG "★拟合料 $FIT 不存在★"; continue; }
      build_amp_so || continue
      LOG "K×λ 扫描(落盘学生): $(basename "$MD") × 拟合料 $(corpus_tag "$FIT") n=$FN, 前 $NL 层"
      MODEL_DIR="$MD"
      run_py "scank_$(corpus_tag "$FIT")_n${FN}_file_$(basename "$MD")_L${NL}_w${WH}" "扫描|held-out|\[L[0-9]|白化|^ +λ|^ +[0-9.e-]+ |吃掉|最优|失败" \
             --ids "$FIT" --ntok "$FN" --fp-dir "$HF" --amp-scan-k --layers "$NL" --amp-whiten "$WH"; rc=$?
      MODEL_DIR="$HF"
      continue;;
    rowdiagfile:*) # 逐行诊断: rowdiagfile:<量化目录>:<拟合ids>:<ntok>:<层数>:<K>:<λ>[:<白化>]
      # 查"逐层能量增益正 / 端到端负"的分叉: 每层用 (λ,K) 解一次, 报能量增益 vs 逐行/逐通道增益
      IFS=: read -r _ MD FIT FN NL KK LAM WH <<<"$NB"; WH="${WH:-0}"
      [ -s "$MD/model.safetensors.index.json" ] || { LOG "★$MD 不是量化目录★"; continue; }
      [ -s "$FIT.layout" ] || { LOG "★$FIT.layout 缺★"; continue; }
      build_amp_so || continue
      LOG "逐行诊断(落盘学生): $(basename "$MD") × $(corpus_tag "$FIT") n=$FN, 前 $NL 层, K=$KK λ=$LAM"
      MODEL_DIR="$MD"
      run_py "rowdiag_$(corpus_tag "$FIT")_n${FN}_file_$(basename "$MD")_L${NL}_k${KK}_lam${LAM}_w${WH}" "诊断|held-out|\[L[0-9]|拟合:|val |通道:|顶行|失败|Traceback" \
             --ids "$FIT" --ntok "$FN" --fp-dir "$HF" --amp-rowdiag "$KK:$LAM" --layers "$NL" --amp-whiten "$WH"; rc=$?
      MODEL_DIR="$HF"
      continue;;
    t2diagfile:*) # 靶口径诊断: t2diagfile:<量化目录>:<拟合ids>:<ntok>:<层数>:<K>:<λ>
      # 需要教师在同一份 ids 上的 --dump-moe(y_fp(x_fp) 逐层), 缺就先跑一趟 FP 教师(顺带落教师锚)。
      IFS=: read -r _ MD FIT FN NL KK LAM <<<"$NB"
      [ -s "$MD/model.safetensors.index.json" ] || { LOG "★$MD 不是量化目录★"; continue; }
      [ -s "$FIT.layout" ] || { LOG "★$FIT.layout 缺★"; continue; }
      build_amp_so || continue
      DUMP="$OUT/moe_fp_$(corpus_tag "$FIT")_n${FN}"
      FREF="$OUT/teacher_$(corpus_tag "$FIT")_n${FN}.bin"
      if [ ! -s "$DUMP/y_L00.bin" ]; then
          LOG "教师 dump(FP 自然跑, 逐层 x/y) → $DUMP"
          run_py "teacherdump_$(corpus_tag "$FIT")_n${FN}" "配置|反修原料|完成|PPL|→" \
                 --ids "$FIT" --ntok "$FN" --dump-moe "$DUMP" --out "$FREF" || continue
      fi
      LOG "靶口径诊断(落盘学生): $(basename "$MD") × $(corpus_tag "$FIT") n=$FN, 前 $NL 层, K=$KK λ=$LAM"
      MODEL_DIR="$MD"
      run_py "t2diag_$(corpus_tag "$FIT")_n${FN}_file_$(basename "$MD")_L${NL}_k${KK}_lam${LAM}" "诊断|held-out|\[L[0-9]|拟合:|val |失败|Traceback" \
             --ids "$FIT" --ntok "$FN" --fp-dir "$HF" --amp-t2diag "$KK:$LAM:$DUMP" --layers "$NL"; rc=$?
      MODEL_DIR="$HF"
      continue;;
    ampfile:*) # 落盘学生的反修档: ampfile:<量化目录>:<拟合ids>:<拟合ntok>[:<白化 0/1/2>]
      # ②a 在线序贯反修·逐层择优(扫→选→解→应用, 见 v41_amp_hooks.install_online_select)
      # ②b 学生 = 落盘模型 + 放大器, 判决料打分。放大器目录按 manifest 末行"# 完成"判完整:
      # 完整就复用(同一份放大器打第二把尺不重解), 半成品一律 rm -rf 重解(禁两代放大器混装)。
      IFS=: read -r _ MD FIT FN WH <<<"$NB"; WH="${WH:-0}"
      [ -s "$MD/model.safetensors.index.json" ] || { LOG "★$MD 不是量化目录★"; continue; }
      [ -s "$FIT" ] || { LOG "★拟合料 $FIT 不存在★"; continue; }
      [ -s "$FIT.layout" ] || { LOG "★$FIT.layout 缺: 择优路要 layout 分层 held-out★"; continue; }
      build_amp_so || continue
      WSFX=""; [ "$WH" != 0 ] && WSFX="_w$WH"     # 白化与否是两代放大器, 目录分开
      ARM="$ROOT/gguf/v41/$(basename "$MD")-amp-$(corpus_tag "$FIT")_n${FN}${WSFX}"
      if grep -q "^# 完成" "$ARM/manifest.txt" 2>/dev/null; then
          LOG "②a 放大器已在 $ARM (manifest 完成), 复用"
      else
          rm -rf "$ARM"
          LOG "②a 序贯反修·择优: 学生 $(basename "$MD") × 拟合料 $(corpus_tag "$FIT") n=$FN → $ARM"
          MODEL_DIR="$MD"
          run_py "solve_$(basename "$ARM")" "序贯|held-out|\[L[0-9]|信任域|择优|完成|PPL|失败" \
                 --ids "$FIT" --ntok "$FN" --fp-dir "$HF" --amp-select "$ARM" --amp-whiten "$WH"; rc=$?
          MODEL_DIR="$HF"
          [ $rc = 0 ] || continue
      fi
      STU="$OUT/stu_${TAG}_file_$(basename "$MD")_amp_$(corpus_tag "$FIT")_n${FN}${WSFX}.bin"
      LOG "②b 学生 = 落盘模型 + 放大器 (判决料 $(basename "$IDS") n=$NTOK)"
      MODEL_DIR="$MD"
      run_py "stu_${TAG}_file_$(basename "$MD")_amp${WSFX}" "反修\]|完成|PPL|路由|失败" \
             --ids "$IDS" --ntok "$NTOK" --amp "$ARM" --out "$STU"; rc=$?
      MODEL_DIR="$HF"
      [ $rc = 0 ] || continue
      LOG "③ 五指标 落盘模型 + 放大器 ← $ARM"
      "$AM" --ref-raw "$REF" --ids "$IDS" --student "$STU" || LOG "★判决失败★"
      continue;;
    engine|engine:*) # 引擎学生档(2026-09-13): engine[:<gguf>[:<放大器目录>]] —— 学生 = ds4 引擎 --score-ids(部署同路);
      # gguf 空 = 默认 1.5 bpw 文件; 给放大器目录则 --zchain 挂上(目录 manifest 须有 "# 完成", 半成品不判)。
      # 教师锚照旧(FP 教师与学生走哪条路无关); 判决器同一份 anchor_metrics。
      IFS=: read -r _ GG ARMD BETA <<<"$NB"; GG="${GG:-$GG_DEFAULT}"
      [ -s "$GG" ] || { LOG "★$GG 不存在★"; continue; }
      ZARG=""; SFX=""
      if [ -n "${ARMD:-}" ]; then
          grep -q "^# 完成" "$ARMD/manifest.txt" 2>/dev/null || { LOG "★放大器目录 $ARMD 没有完成标记, 不判★"; continue; }
          ZARG="--zchain $ARMD"; SFX="_amp_$(basename "$ARMD")"
          # 第四字段 β(--zchain-scale): 每层修正整体缩到 β 倍 —— 步长扫描(针 0)。空 = 1.0 = 原样。
          [ -n "${BETA:-}" ] && { ZARG="$ZARG --zchain-scale $BETA"; SFX="${SFX}_b$BETA"; }
      fi
      STU="$OUT/stu_${TAG}_eng_$(basename "$GG" .gguf)${SFX}.bin"
      LOG "② 学生 = 引擎 $(basename "$GG")${ARMD:+ + 放大器 $(basename "$ARMD")} (判决料 $(basename "$IDS") n=$NTOK)"
      run_eng "$STU" "$GG" "$ZARG" "stu_${TAG}_eng${SFX}" || continue
      LOG "③ 五指标 引擎 $(basename "$GG" .gguf)${SFX}"
      "$AM" --ref-raw "$REF" --ids "$IDS" --student "$STU" || LOG "★判决失败★"
      continue;;
    engamp:*) # 引擎反修档(2026-09-13): engamp:<拟合ids>:<拟合ntok>[:<gguf>[:<白化 0/1/2>[:<层数>]]]
      # ②a v41_amp_run 在引擎真前向上序贯解放大器(FP 靶用 HF 出厂权重当场算, 解算器与 Python 路同一份 CUDA 源)
      # ②b 学生 = 引擎 + 放大器, 判决料打分。放大器目录按 manifest "# 完成" 判完整: 完整复用, 半成品 rm -rf 重解。
      IFS=: read -r _ FIT FN GG WH NLY <<<"$NB"; GG="${GG:-$GG_DEFAULT}"; WH="${WH:-0}"; NLY="${NLY:-40}"
      [ -s "$GG" ] || { LOG "★$GG 不存在★"; continue; }
      [ -s "$FIT" ] || { LOG "★拟合料 $FIT 不存在★"; continue; }
      [ -s "$FIT.layout" ] || { LOG "★$FIT.layout 缺: 择优路要 layout 分层 held-out★"; continue; }
      WSFX=""; [ "$WH" != 0 ] && WSFX="_w$WH"; LSFX=""; [ "$NLY" != 40 ] && LSFX="_L$NLY"
      ARM="$ROOT/gguf/v41/$(basename "$GG" .gguf)-amp-$(corpus_tag "$FIT")_n${FN}${WSFX}${LSFX}-engine"
      if grep -q "^# 完成" "$ARM/manifest.txt" 2>/dev/null; then
          LOG "②a 放大器已在 $ARM (manifest 完成), 复用"
      else
          rm -rf "$ARM"
          make -C "$ROOT/gguf-tools" v41_amp_run >"$OUT/log_make_v41_amp_run.txt" 2>&1 \
              || { LOG "★v41_amp_run 编译失败★"; tail -20 "$OUT/log_make_v41_amp_run.txt"; continue; }
          LOG "②a 引擎序贯反修·择优: $(basename "$GG") × 拟合料 $(corpus_tag "$FIT") n=$FN, 前 $NLY 层, 白化 $WH → $ARM"
          lf="$OUT/log_solve_$(basename "$ARM").txt"
          "$ROOT/gguf-tools/amp/v41_amp_run" "$GG" "$HF" "$FIT" "$FN" "$ARM" --layers "$NLY" --whiten "$WH" --mem-budget-mb 40000 2>&1 \
              | tee "$lf" | grep --line-buffered -E "held-out|反修|引擎\]|\[L[0-9]|择优|PPL|失败|★|error|Error|watchdog"
          [ "${PIPESTATUS[0]}" = 0 ] || { LOG "★反修失败 —— 全量日志 $lf, 尾部:★"; tail -20 "$lf"; continue; }
      fi
      STU="$OUT/stu_${TAG}_eng_$(basename "$GG" .gguf)_amp_$(basename "$ARM").bin"
      LOG "②b 学生 = 引擎 + 放大器 $(basename "$ARM") (判决料 $(basename "$IDS") n=$NTOK)"
      run_eng "$STU" "$GG" "--zchain $ARM" "stu_${TAG}_eng_amp_$(basename "$ARM")" || continue
      LOG "③ 五指标 引擎 + 放大器 ← $ARM"
      "$AM" --ref-raw "$REF" --ids "$IDS" --student "$STU" || LOG "★判决失败★"
      continue;;
    engkl:*) # 蒸馏靶档(2026-09-13): engkl:<拟合ids>:<拟合ntok>:<eta-rel>[:<层号 默认 39>[:<gguf>]]
      # 与 engamp: 的唯一差别是【靶】: 不再是"这一层像教师"(最小二乘), 而是"让最终 logits 像教师该往哪挪"
      # (KL 对该层输出的梯度, 闭式, 见 amp/v41_kl_target.h)。梯度链止于末层 ⇒ 只解一层。
      # ★教师锚必须是【拟合料】自己的那一份★ —— 拿判决料的锚去解就是自己给自己判卷。
      IFS=: read -r _ FIT FN ETA KLY GG <<<"$NB"; GG="${GG:-$GG_DEFAULT}"; KLY="${KLY:-39}"
      [ -s "$GG" ] || { LOG "★$GG 不存在★"; continue; }
      [ -s "$FIT" ] || { LOG "★拟合料 $FIT 不存在★"; continue; }
      [ -s "$FIT.layout" ] || { LOG "★$FIT.layout 缺: 择优路要 layout 分层 held-out★"; continue; }
      KREF="$OUT/teacher_$(corpus_tag "$FIT")_n${FN}.bin"
      [ -s "$KREF" ] || { LOG "★拟合料教师锚 $KREF 缺 —— 先跑 v41_judge.sh $FIT $FN engine 产一份★"; continue; }
      ARM="$ROOT/gguf/v41/$(basename "$GG" .gguf)-kl-$(corpus_tag "$FIT")_n${FN}_L${KLY}_eta${ETA}-engine"
      if grep -q "^# 完成" "$ARM/manifest.txt" 2>/dev/null; then
          LOG "②a 蒸馏靶放大器已在 $ARM, 复用"
      else
          rm -rf "$ARM"
          make -C "$ROOT/gguf-tools" v41_amp_run >"$OUT/log_make_v41_amp_run.txt" 2>&1 \
              || { LOG "★v41_amp_run 编译失败★"; tail -20 "$OUT/log_make_v41_amp_run.txt"; continue; }
          LOG "②a 蒸馏靶解算: L$KLY, 拟合料 $(corpus_tag "$FIT") n=$FN, eta-rel $ETA, 教师锚 $(basename "$KREF") → $ARM"
          lf="$OUT/log_solve_$(basename "$ARM").txt"
          "$ROOT/gguf-tools/amp/v41_amp_run" "$GG" "$HF" "$FIT" "$FN" "$ARM" --only-layer "$KLY" \
              --target "kl:$KREF" --eta-rel "$ETA" --mem-budget-mb 40000 2>&1 \
              | tee "$lf" | grep --line-buffered -E "held-out|反修|引擎\]|KL 靶|蒸馏靶|\[L[0-9]|择优|PPL|失败|★|error|Error|watchdog"
          [ "${PIPESTATUS[0]}" = 0 ] || { LOG "★蒸馏靶解算失败 —— 全量日志 $lf, 尾部:★"; tail -20 "$lf"; continue; }
      fi
      STU="$OUT/stu_${TAG}_eng_kl_$(basename "$ARM").bin"
      LOG "②b 学生 = 引擎 + 蒸馏靶放大器 $(basename "$ARM") (判决料 $(basename "$IDS") n=$NTOK)"
      run_eng "$STU" "$GG" "--zchain $ARM" "stu_${TAG}_eng_kl_$(basename "$ARM")" || continue
      LOG "③ 五指标 引擎 + 蒸馏靶放大器 ← $ARM"
      "$AM" --ref-raw "$REF" --ids "$IDS" --student "$STU" || LOG "★判决失败★"
      continue;;
    file:*) # 落盘模型档: file:<量化目录>。学生权重从文件读回(VQ 三件走 libv41vq.so 解码,
      # 骨架 FP4 与出厂专家同格式) —— 文件读回来的数才是产物的数, 不是"原地量化-反量化"的数。
      MD="${NB#file:}"
      [ -s "$MD/model.safetensors.index.json" ] || { LOG "★$MD 缺 model.safetensors.index.json, 不是量化目录★"; continue; }
      STU="$OUT/stu_${TAG}_file_$(basename "$MD").bin"
      LOG "② 学生 = 落盘模型 $MD"
      MODEL_DIR="$MD"
      run_py "stu_${TAG}_file_$(basename "$MD")" "索引\]|完成|PPL|路由" \
             --ids "$IDS" --ntok "$NTOK" --out "$STU"; rc=$?
      MODEL_DIR="$HF"
      [ $rc = 0 ] || continue
      LOG "③ 五指标 落盘模型 $(basename "$MD")"
      "$AM" --ref-raw "$REF" --ids "$IDS" --student "$STU" || LOG "★判决失败★"
      continue;;
    *:*)   # VQ 档
      VD="${NB%%:*}"; VN="${NB##*:}"
      STU="$OUT/stu_${TAG}_vq${VD}x${VN}${ACTSFX}.bin"
      LOG "② 学生 VQ dim=$VD nc=$VN"
      run_py "stu_${TAG}_vq${VD}x${VN}${ACTSFX}" "VQ\]|完成|PPL|路由" \
             --ids "$IDS" --ntok "$NTOK" --vq-dim "$VD" --vq-nc "$VN" $ACTARG --out "$STU" || continue
      LOG "③ 五指标 VQ ${VD}x${VN}"
      "$AM" --ref-raw "$REF" --ids "$IDS" --student "$STU" || LOG "★判决失败★"
      continue;;
  esac
    STU="$OUT/stu_${TAG}_b${NB}${ACTSFX}.bin"
    CB="$OUT/cb_b$NB.bin"
    # ★码本必须按真实权重分布求★(2026-09-11): V4.1 专家出厂是 FP4 的 16 个离散值,
    # 拿高斯 Lloyd-Max 去套, 4.25 bpw 都能白吃 KLD 0.41。DP 求的是全局最优, 毫秒级。
    [ -s "$CB" ] || "$CBD/v41_codebook" "$HF" 20 "$NB" "$CB" 8 128 2>&1 | grep -E "\[DP\]|\[码本\]" \
        || { LOG "★码本求解失败 $NB★"; continue; }
    LOG "② 学生 ${NB}bit@blk32 = $(awk "BEGIN{printf \"%.4f\", $NB+8/32}") bpw"
    run_py "stu_${TAG}_b${NB}${ACTSFX}" "量化\]|码本\]|完成|PPL|路由" \
           --ids "$IDS" --ntok "$NTOK" --qnbit "$NB" --qblk 32 --qcb "$CB" $ACTARG --out "$STU" || continue
    LOG "③ 五指标 ${NB}bit"
    "$AM" --ref-raw "$REF" --ids "$IDS" --student "$STU" || LOG "★判决失败★"
done
LOG "收工, 产物在 $OUT"
