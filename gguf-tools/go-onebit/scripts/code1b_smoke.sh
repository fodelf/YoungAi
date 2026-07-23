#!/bin/bash
# code1b_smoke.sh — 单机 BASE 裸续写冒烟(自带内存看门狗+超时): 质量+速度一眼判。
# 场景: 合并出的 go-onebit GGUF 首次端到端验证 / 快速质量探针(24-64 tok)。
# 用法: [MODEL=gguf/go-onebit/ds4-code1b.gguf] [NPRED=48] [TIMEOUT_S=300] [MAXMB=11776] \
#       [PROMPT='<BOS>...'] ./gguf-tools/go-onebit/scripts/code1b_smoke.sh
# 输出: 原始生成(不加判读)+ 引擎速度行。默认 prompt = mtp_pipe 同款 twoSum 裸续写(历史可比)。
set -euo pipefail
cd "$(dirname "$0")/../../.."   # → repo 根(ds4 与 metal/*.metal 在此)
MODEL="${MODEL:-gguf/go-onebit/ds4-code1b.gguf}"
NPRED="${NPRED:-48}"; TIMEOUT_S="${TIMEOUT_S:-300}"; MAXMB="${MAXMB:-11776}"
# RESID 口(2026-07-21): 与 pillar_probe.sh 同语义 — 显式空=裸, 非空=挂残差侧车。
# 此前本脚本静默无视 RESID → "带残差"针实际跑裸腿且输出逐字节同裸(已踩)。
if [ -n "${RESID:-}" ]; then export DS4_RESIDUAL="$RESID"; fi
if [ -z "${PROMPT+x}" ]; then
  PROMPT=$(cat <<'PEOF'
<｜begin▁of▁sentence｜>// twoSum returns the indices of the two numbers in nums that add up to target.
func twoSum(nums []int, target int) []int {
PEOF
)
fi
[ -s "$MODEL" ] || { echo "[smoke] 模型缺失: $MODEL" >&2; exit 2; }
[ -x ./ds4 ] || { echo "[smoke] ./ds4 不存在(先 make)" >&2; exit 2; }
OUT=/tmp/code1b_smoke.out; LOG=/tmp/code1b_smoke.log
# 进程纪律: 冒烟前清杀残留大模型进程(实例锁语义: 单机同刻只跑一个大模型)
pkill -9 -x ds4 2>/dev/null || true
# M1 Pro GPU 工作集天花板 ~10.67G: 默认 prefill chunk 4096 的 scratch 池(~4.8G)+骨干 wired 8.2G
# 必穿顶(kIOGPU CB OOM, mtp_pipe 同坑实测)。短 prompt 冒烟 512 足够, 大上下文再显式调。
DS4_MEM_BUDGET_MB="${DS4_MEM_BUDGET_MB:-12000}" \
DS4_METAL_PREFILL_CHUNK="${DS4_METAL_PREFILL_CHUNK:-512}" ./ds4 -m "$MODEL" -n "$NPRED" \
    --temp 0 --seed 1 --nothink -p "$PROMPT" >"$OUT" 2>"$LOG" &
P=$!
trap 'kill -9 "$P" 2>/dev/null || true; echo "[smoke] 中断已清" >&2; exit 130' INT TERM
T0=$(date +%s)
while kill -0 "$P" 2>/dev/null; do
    EL=$(( $(date +%s) - T0 ))
    if [ "$EL" -ge "$TIMEOUT_S" ]; then echo "[watchdog] ${EL}s ≥ ${TIMEOUT_S}s 超时 → kill" >&2; kill -9 "$P" 2>/dev/null || true; break; fi
    MB=$(footprint -p "$P" 2>/dev/null | grep -Eo 'Footprint: *[0-9.]+ *[KMG]B' | head -1 \
         | awk '{v=$2;u=$3; if(u=="GB")v*=1024; else if(u=="KB")v/=1024; printf "%d",v}' || true)
    if [ -n "${MB:-}" ] && [ "$MB" -gt "$MAXMB" ]; then
        echo "[watchdog] 进程 ${MB}MB > ${MAXMB}MB 红线 → kill" >&2; kill -9 "$P" 2>/dev/null || true; exit 9
    fi
    sleep 2
done
# ds4 被杀/非零退出时 wait 返回非零, set -e 会在此静默杀死本脚本(2026-07-21 单机面板
# 3/15 无声死实证) → 显式吞状态
RC=0; wait "$P" 2>/dev/null || RC=$?
echo "===== 原始输出(逐字, 不判读) ====="
cat "$OUT" 2>/dev/null || true
echo ""
echo "===== 引擎速度/内存行 ====="
grep -aE "t/s|tok/s|prefill|decode|budget|resident" "$LOG" 2>/dev/null | tail -8 || true
echo "[smoke] rc=$RC 用时=$(( $(date +%s) - T0 ))s (完整 stderr: $LOG)"
