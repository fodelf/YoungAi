#!/bin/bash
# amp86z_spark.sh — 正序反修流水线驱动(2026-08-19 出生: 路由侧车→链态锚→z 家族解算;
# 2026-08-20 F86_* 全参数化, c86 战役复用)。
# 段(all 顺序): quant(委托 base86p_spark.sh, Q86_* 未设则跳) → merge(委托 merge_base86p.sh,
#   M86_* 未设则跳) → build(引擎重编) → rte(F86_RTE 非空且缺→裸捕获+闭式解算; 空=纯z 跳过)
#   → cap(链态锚: 引擎捕获, RTE 在则 armed) → solve(zside 纯z; GE/FTA/ERF 由 F86_* 定,
#   默认全 0 = z变量+四损失+感知) → chain → judge(五指标 cal12z+wt2 裸/+z/F86_XZC 对照)
#   → bench(速度 + pubbench 328 4并发)
# 用法: bash amp86z_spark.sh [quant|merge|build|rte|cap|solve|chain|judge|bench|all]
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
ZL="$ROOT/gguf-tools/go-onebit/zlever"
R30="$ROOT/gguf/go-onebit/r30"
G7="$ROOT/gguf/go-onebit/g7"
OUT="${F86_OUT:-$R30/full86}"
TAG="$(basename "$OUT")"
MDL="${F86_MDL:-$ROOT/gguf/ds4-allq2.gguf}"
RTE="${F86_RTE-$R30/route86/zchain_route.bin}"      # F86_RTE= (空串)纯z: 全程无路由侧车
FPA="${F86_FPA:-$R30/anchor_cal12z_s2048.bin}"      # FP 锚 = 解算目标
IDS="${F86_IDS:-$G7/cal12z.ids}"
WT2IDS="${F86_WT2IDS:-$G7/wt2.ids}"
WT2ANC="${F86_WT2ANC:-$R30/anchor_wt2_s2653.bin}"
LDIR="${F86_LAYERS:-$OUT/layers}"                   # dql(或空)+zrec/zcache 落盘目录
CHA="$OUT/anchor_chain.bin"                         # 量化链态锚 = x_q/路由_q
CAP="${F86_CAP:-/tmp/cap_$TAG}"
ZC="${F86_ZC:-$OUT/zchain_$TAG.bin}"
XZC="${F86_XZC:-}"                                  # 额外对照 zchain(可空)
LOG(){ echo "[$TAG $(date +%H:%M:%S)] $*"; }
mkdir -p "$OUT"
export DS4_HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-0731}"
ST="${1:-all}"

stage_quant(){
    [ -z "${Q86_OUT:-}" ] && { LOG "quant: Q86_* 未设, 跳过(底座外部给定)"; return 0; }
    LOG "quant 委托 base86p_spark.sh(幂等: 43/43 则秒过; 复现值 cal10 24s/层·cal11 36s/层)"
    bash "$SC/base86p_spark.sh" quant || { LOG "★quant 失败★"; exit 2; }
}

stage_merge(){
    [ -z "${M86_LAYERS:-}" ] && { LOG "merge: M86_* 未设, 跳过"; return 0; }
    M86_MDL="$MDL" bash "$SC/merge_base86p.sh" || { LOG "★merge 失败★"; exit 2; }
    [ -s "$MDL" ] || { LOG "★合并模型没落盘★"; exit 2; }
}

stage_build(){
    LOG "引擎重编"
    cd "$ROOT" && make cuda-spark 2>&1 | tail -1
    # 旧二进制在时 [ -x ds4 ] 会假绿, 必须看 make 自身退出码
    [ "${PIPESTATUS[0]}" -eq 0 ] && [ -x "$ROOT/ds4" ] || { LOG "★引擎编译失败★"; exit 5; }
}

stage_rte(){
    [ -z "$RTE" ] && { LOG "rte: 纯z 战役, 跳过"; return 0; }
    [ -f "$RTE" ] && { LOG "rte 已在, 跳过"; return 0; }
    LOG "rte 裸捕获 → 闭式解算"
    local RCAP="${CAP}_bare" RDIR; RDIR="$(dirname "$RTE")"
    rm -rf "$RCAP"; mkdir -p "$RCAP" "$RDIR"
    cd "$ROOT"
    env DS4_CAP_DIR="$RCAP" DS4_CUDA_NO_TOKEN_GRAPH=1 timeout --foreground 3000 \
        ./ds4 --cuda -m "$MDL" --score-ids "$IDS" --score-out "/tmp/${TAG}_rtecap.bin" \
        </dev/null 2>&1 | tail -1
    RS_OUT="$RDIR" RS_CAP="$RCAP" RS_ANC="$FPA" bash "$SC/rsolve43.sh" 4 \
        || { LOG "★rte 解算失败★"; exit 2; }
    python3 "$ZL/zrec_to_zchain.py" "$RDIR" "$RTE" 43 || { LOG "★rte 链失败★"; exit 2; }
}

stage_cap(){
    # 2026-08-20 判决: 链态锚口径自我拆台(修正生效→下游输入偏离解算假设, 链上复利为负,
    # cal12 实测三链全负 vs FP口径 XZC 正); FP 锚=自洽不动点为默认, F86_CAP=1 才捕获
    [ "${F86_CAP:-0}" != 1 ] && { LOG "cap: FP 锚口径战役, 跳过捕获"; return 0; }
    [ -f "$CHA" ] && { LOG "链态锚已在, 跳过"; return 0; }
    local ZARG=(); [ -n "$RTE" ] && ZARG=(--zchain "$RTE")
    LOG "抓链态锚${RTE:+(route 侧车 armed)}"
    rm -rf "$CAP"; mkdir -p "$CAP"
    cd "$ROOT"
    env DS4_CAP_DIR="$CAP" DS4_CUDA_NO_TOKEN_GRAPH=1 timeout --foreground 3000 \
        ./ds4 --cuda -m "$MDL" "${ZARG[@]+"${ZARG[@]}"}" \
        --score-ids "$IDS" --score-out "/tmp/${TAG}_capscore.bin" </dev/null 2>&1 | tail -1
    python3 "$SC/cap_to_anchor.py" "$CAP" "$IDS" "$CHA" 43 || { LOG "★锚转换失败★"; exit 2; }
}

stage_solve(){
    export DS4_ZL_GE="${F86_GE:-0}" DS4_ZL_FTA="${F86_FTA:-0}" DS4_ZL_ERF="${F86_ERF:-0}"
    export DS4_ZL_GATE="${F86_GATE:-0}"             # >0 即入(cal11 用户令)
    if [ "${F86_AMP:-0}" = 1 ]; then
        # 单遍双解+择优(2026-08-20): 每层 zcache 一次, 加性 zlayer ∥ 乘性 amp_solve, held 定案
        LOG "反修 43 层(单遍双解: 加性 z变量+四损失+感知 ∥ 乘性放大器 ELM, 每层 held 择优)"
        export ZS_AMP=1 ZS_AMPNFIT="${F86_AMPNFIT:-1638}"
    else
        LOG "反修 43 层(纯加性 z变量+四损失+感知; GE=${F86_GE:-0} FTA=${F86_FTA:-0} ERF=${F86_ERF:-0})"
        export ZS_AMP=0
    fi
    # 口径: FP 锚(默认, 自洽不动点); F86_CAP=1 显式捕获战役才走链态锚
    if [ "${F86_CAP:-0}" = 1 ]; then export DS4_ZL_XANCHOR="$CHA"; else unset DS4_ZL_XANCHOR; fi
    [ -n "${F86_ZLGGUF:-}" ] && export DS4_ZL_GGUF="$F86_ZLGGUF"   # 无 dql 的底座(allq2 类)才用
    export ZS_LAYERS="$LDIR" ZS_ANCHOR="$FPA" ZS_NTOK="${F86_NTOK:-2048}" ZS_INJ=2
    export ZS_FIT="${F86_FIT:-0:1638}" ZS_EV="${F86_EV:-1638:2048}"
    bash "$SC/zside_base86p.sh" "${LANES:-4}" || { LOG "★解算失败★"; exit 3; }
}

stage_chain(){
    LOG "合并 zchain${RTE:+(+route)}"
    [ -n "$RTE" ] && cp -f "$(dirname "$RTE")"/zrec_route_L*.bin "$LDIR"/ 2>/dev/null
    python3 "$ZL/zrec_to_zchain.py" "$LDIR" "$ZC" 43 || { LOG "★合并失败★"; exit 4; }
}

judge_one(){ # $1=标签 $2=ids $3=ref锚 $4=zchain(可空) $5=out
    local Z=(); [ -n "${4:-}" ] && Z=(--zchain "$4")
    cd "$ROOT"
    LOG "判决 $1"
    env DS4_CUDA_NO_TOKEN_GRAPH=1 timeout --foreground 3000 ./ds4 --cuda -m "$MDL" \
        "${Z[@]+"${Z[@]}"}" --score-ids "$2" --score-out "$5" </dev/null 2>&1 \
        | grep -aE "zchain loaded|完成" | head -2
    echo "══ 五指标 $1 ══"
    python3 "$SC/anchor_metrics.py" --ref "$3" --ids "$2" --student "$5" --tail 0 2>&1 | tail -7
}

stage_judge(){
    [ -f "$ZC" ] || { LOG "★zchain 缺★"; exit 6; }
    judge_one "cal12z 裸" "$IDS"    "$FPA"    ""    "/tmp/${TAG}_cal_base.bin"
    judge_one "wt2 裸"    "$WT2IDS" "$WT2ANC" ""    "/tmp/${TAG}_wt2_base.bin"
    judge_one "cal12z +z" "$IDS"    "$FPA"    "$ZC" "/tmp/${TAG}_cal_z.bin"
    judge_one "wt2 +z"    "$WT2IDS" "$WT2ANC" "$ZC" "/tmp/${TAG}_wt2_z.bin"
    if [ -n "$XZC" ] && [ -f "$XZC" ]; then
        judge_one "cal12z +z旧离线口径对照" "$IDS"    "$FPA"    "$XZC" "/tmp/${TAG}_cal_xz.bin"
        judge_one "wt2 +z旧离线口径对照"    "$WT2IDS" "$WT2ANC" "$XZC" "/tmp/${TAG}_wt2_xz.bin"
    fi
}

stage_bench(){
    cd "$ROOT"
    LOG "速度 裸"
    ./ds4 --cuda -m "$MDL" -n 128 -p "Write a Python quicksort function." </dev/null 2>&1 \
        | grep -aE "t/s" | tail -1
    LOG "速度 +z"
    ./ds4 --cuda -m "$MDL" --zchain "$ZC" -n 128 -p "Write a Python quicksort function." </dev/null 2>&1 \
        | grep -aE "t/s" | tail -1
    LOG "server 起(+z)"
    ./ds4-server --cuda -m "$MDL" --zchain "$ZC" --ctx 16384 > "/tmp/${TAG}_server.log" 2>&1 &
    local SRV=$!
    sleep 75
    cd "$SC"
    export PUBBENCH_CACHE="$ROOT/gguf-tools/go-onebit/pubbench_data"
    LOG "smoke 2题"
    python3 pubbench.py --suite humaneval --url http://127.0.0.1:8000 --tag "${TAG}_smoke" --limit 2 2>&1 | tail -3
    LOG "Py 164 (4并发)"
    python3 pubbench.py --suite humaneval --url http://127.0.0.1:8000 --tag "$TAG" --limit 164 --jobs 4 2>&1 | tail -4
    LOG "Go 164 (4并发)"
    python3 pubbench.py --suite humaneval-x-go --url http://127.0.0.1:8000 --tag "$TAG" --limit 164 --jobs 4 2>&1 | tail -4
    kill $SRV 2>/dev/null || true
}

case "$ST" in
    quant) stage_quant ;;
    merge) stage_merge ;;
    build) stage_build ;;
    rte)   stage_rte ;;
    cap)   stage_cap ;;
    solve) stage_solve ;;
    chain) stage_chain ;;
    judge) stage_judge ;;
    bench) stage_bench ;;
    all)   stage_quant && stage_merge && stage_build && stage_rte && stage_cap \
           && stage_solve && stage_chain && stage_judge && stage_bench ;;
    *) echo "用法: $0 [quant|merge|build|rte|cap|solve|chain|judge|bench|all]" >&2; exit 1 ;;
esac
LOG "$TAG $ST 收官"
