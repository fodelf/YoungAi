#!/bin/bash
# q4t_teacher_spark.sh — 引擎原生 Q4_K 教师战役(2026-08-20 用户令"按照这个执行")。
# 背景: ds4quant_run FP 锚教师缺 indexer 层(fable5 08-20 两针尸检), 在线 KL 32-38% 假账。
# 方案落地形态修订: CUDA routed-MoE 无 q8_0 专家路, 但 q4k 朴素批链现成且限 6 专家=主模型
#   形状 → 教师=专家全 Q4_K + 现骨架默认配方(attn q8_0/router+indexer f16)。
#   Q4_K 对 FP 的 KLD≈0.0102(unsloth 表)=尺子噪声地板, 远低于被测 0.5+ 量级。
# 教师与学生同引擎同协议逐字节同构 → 判决尺+解算锚源全换此教师, FP 锚退役。
# 段: quant → smoke(原始输出) → score(wt2+cal12z 教师 logits) → gate(教师体检) → rejudge(换尺重判)
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/scripts"
G7="$ROOT/gguf/go-onebit/g7"
R30="$ROOT/gguf/go-onebit/r30"
MDL="$ROOT/gguf/ds4-q4t.gguf"
TD="$R30/teach"
LOG(){ echo "[q4t $(date +%H:%M:%S)] $*"; }
mkdir -p "$TD"
ST="${1:-all}"

guard_mem(){  # 简易内存卫兵: MemAvailable<8G 杀量化(121G 机, 正常远不触发)
    # ★后台子壳 stdout 必须重定向: $(guard_mem) 命令替换等 stdout 关闭, 不重定向=永久挂死
    ( while true; do
        AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        [ "${AV:-99}" -lt 6 ] && { echo "[q4t][wdog] MemAvailable=${AV}G <6G 杀重载进程"; pkill -9 -x deepseek4-quantize; pkill -9 -x ds4; }
        sleep 5; done ) >/dev/null 2>>/tmp/q4t_wdog.log &
    echo $!
}

stage_quant(){
    [ -f "$MDL.done" ] && { LOG "quant 已完成, 跳过"; return 0; }
    LOG "quant 发车: 专家全 Q4_K + 骨架默认配方(threads 20)"
    local WD; WD=$(guard_mem)
    ( cd "$ROOT/gguf-tools" && ./deepseek4-quantize \
        --hf "$ROOT/hf/DeepSeek-V4-Flash-0731" \
        --template "$R30/template_head.gguf" \
        --out "$MDL" \
        --routed-w1 q4_k --routed-w3 q4_k --routed-w2 q4_k \
        --threads 20 --overwrite )
    local RC=$?
    kill "$WD" 2>/dev/null || true
    [ $RC -eq 0 ] && [ -s "$MDL" ] || { LOG "★quant 失败 rc=$RC★"; exit 2; }
    touch "$MDL.done"
    LOG "quant 收官: $(ls -l "$MDL" | awk '{printf "%.2f GB", $5/1e9}')"
}

stage_smoke(){
    cd "$ROOT"
    local WD; WD=$(guard_mem); trap "kill $WD 2>/dev/null" RETURN
    LOG "smoke 生成(原始输出如下)"
    # (env 大扫除 2026-08-31: CUDA_DIRECT_MODEL 已无读取者, 删)
    timeout --foreground 900 ./ds4 --cuda -m "$MDL" -n 48 \
        -p "Explain what a hash map is in one paragraph." </dev/null 2>&1 \
        | grep -aE "moe-init|t/s|." | tail -12
}

stage_score(){
    cd "$ROOT"
    local WD; WD=$(guard_mem); trap "kill $WD 2>/dev/null" RETURN
    for P in "wt2:$G7/wt2.ids" "cal12z:$G7/cal12z.ids"; do
        local NM="${P%%:*}" IDS="${P#*:}"
        [ -s "$TD/q4t_$NM.bin" ] && { LOG "score $NM 已在, 跳过"; continue; }
        LOG "score $NM 教师 logits"
        timeout --foreground 3000 ./ds4 --cuda -m "$MDL" \
            --score-ids "$IDS" --score-out "$TD/q4t_$NM.bin" </dev/null 2>&1 \
            | grep -aE "完成" | tail -1
        [ -s "$TD/q4t_$NM.bin" ] || { LOG "★score $NM 没落盘★"; exit 3; }
    done
}

stage_gate(){
    cd "$ROOT"
    LOG "教师体检① 语言健康度(旧锚 PPL 4.2365 为参照, 教师应≤)"
    "$(dirname "$0")/../bench/anchor_metrics" --ref-raw "$TD/q4t_wt2.bin" --ids "$G7/wt2.ids" | tail -4
    LOG "教师体检② 旧锚盲区复核(教师应贴学生=会拷贝/会检索)"
    # 原 numpy 盲区细察(anchor_metrics.py)随全仓 Python 清零删除(见 git 历史)。
    # C 版 bench/anchor_metrics 给同尺五指标: 分别以 q4t 教师 dump 与 iq2 学生 dump 作
    # --student 对旧锚判, NLL/Σmin 对比即"教师是否优于旧锚"; 逐位置盲区明细不再输出。
    AM="$ROOT/gguf-tools/bench/anchor_metrics"
    [ -x "$AM" ] || make -C "$ROOT/gguf-tools" anchor_metrics
    echo "== q4t 教师 vs 旧锚 =="
    "$AM" --ref gguf/go-onebit/r30/anchor_wt2_s2653.bin --ids gguf/go-onebit/g7/wt2.ids \
          --student gguf/go-onebit/r30/teach/q4t_wt2.bin
    echo "== iq2 学生 vs 旧锚 =="
    "$AM" --ref gguf/go-onebit/r30/anchor_wt2_s2653.bin --ids gguf/go-onebit/g7/wt2.ids \
          --student /tmp/p2_iq2_wt2.bin
    echo "体检判读: 教师 NLL 应显著低于学生且不劣于旧锚(逐位置盲区细察见 git 历史 py 版)"
}

stage_rejudge(){
    cd "$ROOT"
    LOG "换尺重判(学生 bin 复用 /tmp 现货, 零新引擎跑)"
    for J in "iq2裸 wt2:/tmp/p2_iq2_wt2.bin:wt2" \
             "cal12裸 wt2:/tmp/c86_wt2_base.bin:wt2" \
             "cal12+c86链 wt2:/tmp/c86_wt2_z.bin:wt2" \
             "cal12裸 cal12z:/tmp/c86_cal_base.bin:cal12z" \
             "cal12+c86链 cal12z:/tmp/c86_cal_z.bin:cal12z"; do
        local NM BIN RU
        NM="${J%%:*}"; BIN="$(echo "$J" | cut -d: -f2)"; RU="${J##*:}"
        [ -s "$BIN" ] || { LOG "$NM: 学生bin缺($BIN), 跳过"; continue; }
        echo "══ q4t尺 $NM ══"
        "$(dirname "$0")/../bench/anchor_metrics" --ref-raw "$TD/q4t_$RU.bin" \
            --ids "$G7/$RU.ids" --student "$BIN" --tail 0 2>&1 | tail -6
    done
}

case "$ST" in
    quant) stage_quant ;;
    smoke) stage_smoke ;;
    score) stage_score ;;
    gate)  stage_gate ;;
    rejudge) stage_rejudge ;;
    all) stage_quant && stage_smoke && stage_score && stage_gate && stage_rejudge ;;
    *) echo "用法: $0 [quant|smoke|score|gate|rejudge|all]"; exit 1 ;;
esac
LOG "q4t $ST 收官"
