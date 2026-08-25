#!/bin/bash
# amp86_spark.sh — v6.1 全 q2 放大器战役·完整流水线(2026-08-19 用户令"量化→反修→评分→合并
# 按顺序执行不偷懒"; 依赖序=量化→反修→合并(zchain)→评分)。
# ①量化: 全 q2 基座(quant_allq2_spark.sh, 专家 IQ2_XXS/Q2_K + backbone/attn/shared 全 Q2_K)
# ②反修: 43 层乘性放大器 ELM 闭式(零训练, 全场景 cal12z 锚, gguf-py 读基座)
# ③合并: zrec → DQZ2 zchain 侧车
# ④评分: 引擎重编(dense Q2_K) → wt2 五指标 裸vs+amp → 速度 → pubbench 328(4并发)
# 用法: bash amp86_spark.sh [quant|solve|chain|build|judge|bench|all]
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
ZL="$ROOT/gguf-tools/go-onebit/zlever"
R30="$ROOT/gguf/go-onebit/r30"
G7="$ROOT/gguf/go-onebit/g7"
MDL="$ROOT/gguf/ds4-allq2.gguf"
# ★口径对齐版(2026-08-19 用户令"先把代码对齐再跑"): 解算的 x/路由/yq 全部换成引擎在线
# 的量化链态(DS4_CAP_DIR 捕获→DQA2 链态锚, zlayer XAP 双锚模式), FP 锚只当目标。
# 旧 FP-锚口径产物留在 amp86/ 不动, 对齐版产物进 amp86c/。
AMP="$R30/amp86c"
ANCHOR="$R30/anchor_cal12z_s2048.bin"     # FP 锚 = 解算目标(yfp 侧)
CHA="$AMP/anchor_chain_s2048.bin"         # 量化链态锚 = x_q/路由_q(stage_cap 产)
IDS="$R30/cal12z.ids"                     # 与 FP 锚同序 ids
ZC="$AMP/zchain_amp86c.bin"
export DS4_HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-0731}" OPENBLAS_NUM_THREADS=1
LOG(){ echo "[amp86 $(date +%H:%M:%S)] $*"; }
ST="${1:-all}"
mkdir -p "$AMP"

stage_quant(){   # 幂等: 完成标记在则跳过
    [ -f "$MDL.done" ] && { LOG "①量化 已完成, 跳过"; return 0; }
    LOG "①量化 全q2 发车"
    OUT="$MDL" bash "$SC/quant_allq2_spark.sh" || { LOG "★量化失败★"; exit 2; }
    [ -f "$MDL" ] || { LOG "★模型没落盘★"; exit 2; }
    touch "$MDL.done"
    LOG "①量化 收官: $(ls -l "$MDL" | awk '{printf "%.2f GB", $5/1e9}')"
}

stage_solve(){
    [ -f "$ANCHOR" ] || { LOG "★全场景锚缺 $ANCHOR★"; exit 3; }
    LOG "②反修 43层 乘性放大器(零训练)"
    for L in $(seq 0 42); do
        REC="$AMP/zrec_L$(printf %02d $L).bin"
        [ -f "$REC" ] && { LOG "L$L 已解, 跳过"; continue; }
        ZCF="$AMP/zcache_L$(printf %02d $L).npz"
        [ -f "$ZCF" ] || env DS4_ZL_GGUF="$MDL" DS4_ZL_NTOK=2048 DS4_ZL_GE=0 DS4_ZL_FTA=0 \
            DS4_ZL_ERF=0 DS4_ZL_SWLIM=60 DS4_ZL_GATE=99 \
            python3 -u "$ZL/zlayer.py" "$DS4_HF" "$AMP" "$ANCHOR" $L 1024 0 >/dev/null 2>&1 \
            || { LOG "★L$L zcache 失败★"; exit 3; }
        python3 -u "$ZL/amp_solve.py" "$ANCHOR" "$ZCF" "$REC" 1638 || { LOG "★L$L 解算失败★"; exit 3; }
        rm -f "$ZCF"
    done
    LOG "②反修 43/43 收官"
}

stage_chain(){
    LOG "③合并 zchain"
    "$(dirname "$0")/../calib/zrec_to_zchain" "$AMP" "$ZC" 43 || { LOG "★合并失败★"; exit 4; }
}

stage_build(){
    LOG "④-0 引擎重编(dense Q2_K 支持)"
    cd "$ROOT" && make cuda-spark 2>&1 | tail -1
    # 旧二进制在时 [ -x ds4 ] 会假绿, 必须看 make 自身退出码
    [ "${PIPESTATUS[0]}" -eq 0 ] && [ -x "$ROOT/ds4" ] || { LOG "★引擎编译失败★"; exit 5; }
}

judge_one(){ # $1=标签 $2=zchain(可空) $3=out
    local Z=(); [ -n "${2:-}" ] && Z=(--zchain "$2")
    cd "$ROOT"
    LOG "④评分 wt2 $1"
    # --foreground 必须: GNU timeout 默认 setpgid 子进程成后台组, ds4 一碰 tty 即被
    # SIGTTIN/SIGTTOU 停机(prog86 "score-ids CUDA 卡死"真因, 非 CUDA bug)
    timeout --foreground 3000 ./ds4 --cuda -m "$MDL" "${Z[@]+"${Z[@]}"}" \
        --score-ids "$G7/wt2.ids" --score-out "$3" </dev/null 2>&1 | tail -1
    echo "══ wt2 五指标 $1 ══"
    "$(dirname "$0")/../calib/anchor_metrics" --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" \
        --student "$3" --tail 3 2>&1 | head -12
}

stage_judge(){
    [ -f "$ZC" ] || { LOG "★zchain 缺★"; exit 6; }
    judge_one 裸 "" /tmp/amp86_wt2_base.bin
    judge_one +amp "$ZC" /tmp/amp86_wt2_amp.bin
}

stage_bench(){
    cd "$ROOT"
    LOG "④评分 速度 裸"
    ./ds4 --cuda -m "$MDL" -n 128 -p "Write a Python quicksort function." 2>&1 | grep -aE "t/s" | tail -1
    LOG "④评分 速度 +amp"
    ./ds4 --cuda -m "$MDL" --zchain "$ZC" -n 128 -p "Write a Python quicksort function." 2>&1 | grep -aE "t/s|AMP" | tail -2
    LOG "④评分 server 起(+amp)"
    ./ds4-server --cuda -m "$MDL" --zchain "$ZC" --ctx 16384 > /tmp/amp86_server.log 2>&1 &
    local SRV=$!
    sleep 75
    cd "$SC"
    LOG "smoke 2题"
    python3 pubbench.py --suite humaneval --url http://127.0.0.1:8000 --tag amp86_smoke --limit 2 2>&1 | tail -3
    LOG "Py 164 (4并发)"
    python3 pubbench.py --suite humaneval --url http://127.0.0.1:8000 --tag amp86 --limit 164 --jobs 4 2>&1 | tail -4
    LOG "Go 164 (4并发)"
    python3 pubbench.py --suite humaneval-x-go --url http://127.0.0.1:8000 --tag amp86 --limit 164 --jobs 4 2>&1 | tail -4
    kill $SRV 2>/dev/null || true
}

case "$ST" in
    quant) stage_quant ;;
    solve) stage_solve ;;
    chain) stage_chain ;;
    build) stage_build ;;
    judge) stage_judge ;;
    bench) stage_bench ;;
    all)   stage_quant && stage_solve && stage_chain && stage_build && stage_judge && stage_bench ;;
    *) echo "用法: $0 [quant|solve|chain|build|judge|bench|all]" >&2; exit 1 ;;
esac
LOG "amp86 $ST 收官"
