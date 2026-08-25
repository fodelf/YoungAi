#!/bin/bash
# ampj_spark.sh — 反修百分百还原判决(2026-08-19 用户令"先看百分百还原反修, 整体质量如何")。
# 四配置 × cal12z 语料(解算同料, FP 锚含 logits 作 ref) × 五指标:
#   a 裸          b +amp(在线口径)          c +amp x钉锚(DS4_AMP_ANCHOR)
#   d +amp x+路由全钉(=解算输入条件百分百还原, DS4_AMP_ANCHOR_ROUTE=1)
# 判读: d 恢复量 = 反修产物在自己口径下的链上真值; d−b = 口径漂移的代价;
#       d 仍差 => 反修实现/应用另有病, 继续挖。
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
R30="$ROOT/gguf/go-onebit/r30"
MDL="$ROOT/gguf/ds4-allq2.gguf"
IDS="$ROOT/gguf/go-onebit/g7/cal12z.ids"
ANC="$R30/anchor_cal12z_s2048.bin"
ZC="$R30/amp86/zchain_amp86.bin"
LOG(){ echo "[ampj $(date +%H:%M:%S)] $*"; }

run_one(){ # $1=标签 $2=out.bin $3...=附加 env/参数(K=V 形式在前, -- 后为 ds4 参数)
    local TAG="$1" OUT="$2"; shift 2
    local ENVS=() ARGS=()
    local IN_ARGS=0
    for a in "$@"; do
        [ "$a" = "--" ] && { IN_ARGS=1; continue; }
        [ $IN_ARGS = 1 ] && ARGS+=("$a") || ENVS+=("$a")
    done
    cd "$ROOT"
    LOG "$TAG 发车"
    env DS4_CUDA_NO_TOKEN_GRAPH=1 "${ENVS[@]+"${ENVS[@]}"}" \
        timeout --foreground 3000 ./ds4 --cuda -m "$MDL" "${ARGS[@]+"${ARGS[@]}"}" \
        --score-ids "$IDS" --score-out "$OUT" </dev/null 2>&1 | grep -aE "armed|zchain|完成" | head -4
    echo "══ 五指标 $TAG ══"
    "$(dirname "$0")/../calib/anchor_metrics" --ref "$ANC" --ids "$IDS" --student "$OUT" --tail 0 2>&1 | tail -8
}

run_one "a裸"        /tmp/ampj_a.bin
run_one "b+amp在线"  /tmp/ampj_b.bin -- --zchain "$ZC"
run_one "c+amp x钉锚" /tmp/ampj_c.bin DS4_AMP_ANCHOR="$ANC" -- --zchain "$ZC"
run_one "d全钉(百分百还原)" /tmp/ampj_d.bin DS4_AMP_ANCHOR="$ANC" DS4_AMP_ANCHOR_ROUTE=1 -- --zchain "$ZC"
LOG "ampj 收官"
