#!/bin/bash
# ampj_spark.sh — 反修百分百还原判决(2026-08-19 用户令"先看百分百还原反修, 整体质量如何")。
# 四配置 × cal12z 语料(解算同料, FP 锚含 logits 作 ref) × 五指标:
#   a 裸          b +amp(在线口径)          c +amp x钉锚(DS4_AMP_ANCHOR)
#   d +amp x+路由全钉(=解算输入条件百分百还原, DS4_AMP_ANCHOR_ROUTE=1)
# 判读: d 恢复量 = 反修产物在自己口径下的链上真值; d−b = 口径漂移的代价;
#       d 仍差 => 反修实现/应用另有病, 继续挖。
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/scripts"
R30="$ROOT/gguf/go-onebit/r30"
MDL="$ROOT/gguf/ds4-allq2.gguf"
IDS="$ROOT/gguf/go-onebit/g7/cal12z.ids"
ANC="$R30/anchor_cal12z_s2048.bin"
ZC="$R30/amp86/zchain_amp86.bin"
LOG(){ echo "[ampj $(date +%H:%M:%S)] $*"; }

run_one(){ # $1=标签 $2=out.bin $3...=附加 ds4 参数
    local TAG="$1" OUT="$2"; shift 2
    cd "$ROOT"
    LOG "$TAG 发车"
    # (env 大扫除 2026-08-31: CUDA_NO_TOKEN_GRAPH 已无读取者, 删)
    timeout --foreground 3000 ./ds4 --cuda -m "$MDL" "$@" \
        --score-ids "$IDS" --score-out "$OUT" </dev/null 2>&1 | grep -aE "armed|zchain|完成" | head -4
    echo "══ 五指标 $TAG ══"
    "$(dirname "$0")/../bench/anchor_metrics" --ref "$ANC" --ids "$IDS" --student "$OUT" --tail 0 2>&1 | tail -8
}

run_one "a裸"        /tmp/ampj_a.bin
run_one "b+amp在线"  /tmp/ampj_b.bin --zchain "$ZC"
# ★c/d 腿暂停(2026-08-31 env 大扫除): 判决钩入口原是 DS4_AMP_ANCHOR/_ROUTE, env 已从引擎
# 删除; core_engine_api.c 的 ds4_tool_set_amp_anchor() setter 在, 但 CLI flag 尚未接线。
# 不许静默跑成 b 腿冒充钉锚判决 — 等 --amp-anchor flag 接上后按下面原样恢复:
#   run_one "c+amp x钉锚"        /tmp/ampj_c.bin --amp-anchor "$ANC" --zchain "$ZC"
#   run_one "d全钉(百分百还原)"   /tmp/ampj_d.bin --amp-anchor "$ANC" --amp-anchor-route --zchain "$ZC"
LOG "★c/d 腿跳过: 锚钉 flag 未接线(见上注释)★"
LOG "ampj 收官"
