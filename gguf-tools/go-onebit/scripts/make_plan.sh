#!/bin/bash
# make_plan.sh <模型总体积GiB> [出前缀] — 传体积即出计划表 + 审查 JSON(2026-08-01 用户令)。
#
# 一条命令覆盖全流程: 总体积 → 扣 backbone → 双信号最优分配 → 计划表 + JSON + 最优性/风险审计。
#   ① 难度 d[L]   = 上一轮量化的逐层 held 增量(误差主产地在哪)
#   ② 集中度曲线  = FP 锚每层前 k 名专家的累计路由权重(热专家在这层值不值)
# 目标函数 max Σ d[L]·[cover(hot)·cos_hot + (1-cover)·cos_cold(档)] s.t. Σ字节 ≤ 预算,
# 贪心按【边际精度增益/边际字节】统一竞价 —— 热专家与冷档升级在同一把尺子上, 无人为上限。
#
# 依赖(缺则自动重算/报错):
#   plan/hotcurve.json   锚集中度曲线   ← anchor_hotcurve.py(锚变了才需重算)
#   plan/held_round1.txt 逐层难度       ← 上一轮量化日志(配方大改后应刷新)
#
# 例: make_plan.sh 28      → 27.9x GiB 的最优分配
#     make_plan.sh 34      → 34 GiB 的最优分配(同一套信号, 预算变则分配自动重排)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PLAN="$ROOT/gguf/go-onebit/plan"
SC="$ROOT/gguf-tools/go-onebit/scripts"
BACKBONE_GIB=8.202          # r28_skeleton.gguf 实测(纯 backbone + GGUF 元数据, 不含专家)

TARGET="${1:-}"
[ -n "$TARGET" ] || { echo "用法: make_plan.sh <模型总体积GiB> [出前缀]" >&2; exit 1; }
PREFIX="${2:-r28_rplan_auto}"

CURVE="$PLAN/hotcurve.json"
HELD="$PLAN/held_round1.txt"
[ -f "$CURVE" ] || { echo "★缺 $CURVE — 先跑: anchor_hotcurve.py <anchor.bin> 0.80 $CURVE" >&2; exit 2; }
[ -f "$HELD" ]  || { echo "★缺 $HELD — 从上一轮量化日志提取 'SEARCH L=x QT' 的 held" >&2; exit 2; }

PAYLOAD=$(awk -v t="$TARGET" -v b="$BACKBONE_GIB" 'BEGIN{printf "%.3f", t-b}')
awk -v p="$PAYLOAD" 'BEGIN{exit !(p>0.5)}' || { echo "★总体积 $TARGET GiB 扣 backbone $BACKBONE_GIB 后无有效载荷" >&2; exit 3; }

echo "[make_plan] 目标 ${TARGET} GiB = backbone ${BACKBONE_GIB} + 载荷 ${PAYLOAD} GiB" >&2
python3 "$SC/rplan_solve_v4.py" "$CURVE" "$HELD" "$PAYLOAD" \
        "$ROOT/gguf/go-onebit/${PREFIX}.txt" "$PLAN/${PREFIX}.json"
echo "[make_plan] 计划表 → gguf/go-onebit/${PREFIX}.txt" >&2
echo "[make_plan] 审查JSON → gguf/go-onebit/plan/${PREFIX}.json" >&2
