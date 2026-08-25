#!/bin/bash
# preflight_skeleton.sh — 量化/合并前置闸: 骨架必须干净(2026-08-01 用户令"记为下一次量化前置")。
#
# 背景: r28_skeleton.gguf 抄自冠军 v4bf, exp_probs_b 里烘着 2.5·Δb_champ
#       (skel_bias_probe 实测 k=+2.27 r=+0.94)。用它合并 ⇒ 新模型背着别人的路由药方。
#       fable5 实测迁移他人 Δb 是净负(+0.0003~0.0014, "别人的漂移药方")。
# 本闸: 拿手上所有【他人 Δb】去测骨架相关性, r>THRESH 即判定不干净。
#       干净骨架(自产)对任何他人 Δb 的相关应 ≈0。
#
# 用法: preflight_skeleton.sh <skeleton.gguf> [相关阈值=0.30]
#   通过 → rc=0; 不干净 → rc=1 并打印补救办法(减 Δb 或重抽自产骨架)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SKEL="${1:-$ROOT/gguf/go-onebit/r28_skeleton.gguf}"
THRESH="${2:-0.30}"
G7="$ROOT/gguf/go-onebit/g7"

[ -f "$SKEL" ] || { echo "★骨架不存在: $SKEL" >&2; exit 2; }
# 收集手上所有他人 Δb(冠军系列)
FOREIGN=""
for f in "$G7"/route_bias_v4fix.bin "$G7"/route_bias_ours.bin \
         "$ROOT"/gguf/go-onebit/route_bias_v4fix.bin; do
    [ -f "$f" ] && FOREIGN="$FOREIGN $f"
done
[ -n "$FOREIGN" ] || { echo "[preflight] 无他人 Δb 可比对, 跳过(骨架来源未知, 建议自产)" >&2; exit 0; }

echo "[preflight] 骨架: $SKEL" >&2
OUT=$(python3 "$ROOT/gguf-tools/go-onebit/scripts/skel_bias_probe.py" "$SKEL" $FOREIGN 2>&1)
echo "$OUT" | grep -E "武装槽|斜率|★" >&2

BAD=$(echo "$OUT" | awk -v t="$THRESH" '
  /斜率 k=/ { for(i=1;i<=NF;i++) if($i ~ /^r=/){ split($i,a,"="); v=a[2]+0; if(v<0)v=-v; if(v>t) n++ } }
  END{ print n+0 }')
if [ "$BAD" -gt 0 ]; then
    cat >&2 <<'MSG'

★★骨架不干净 — 它烘着别人的路由偏置★★
  影响: 合并出的模型背着他人 Δb, 与本轮反修的路由基准不一致; 且实测迁移他人 Δb 净负。
  两条补救(选一):
    ① 临时等价: 合并后先减掉他人 Δb 再烘自己的
         r36_rebake_bias.py <模型> <他人Δb> <他人α> <自己Δb> 0.0
         route_alpha_set.py <模型> <自己Δb> 0.0 --snapshot-only
         route_alpha_set.py <模型> <自己Δb> <自己α>
    ② 根治(推荐): 从 ~/ds4-main/hf/DeepSeek-V4-Flash-Base 抽自产骨架
       (需新写 HF→GGUF 骨架工具, 且复现原始 q8_0/f16 编码 — 骨架里 q8_0 占 6.1 GiB)
MSG
    exit 1
fi
echo "[preflight] 骨架干净 ✓ (对所有他人 Δb 相关 |r| <= $THRESH)" >&2
