#!/bin/bash
# rr_verdict.sh [ids] [ntok] — 还原率铁律判决(2026-07-13): 硬文本分布还原率 Σmin/KL, teacher-forced 字节回放
#
# 口径(★还原率评分铁律★): top-1 一致率退役; 只认分布还原率 + 硬多样高PPL文本(默认 /tmp/rr_hard.ids)。
# 机制: 复用 DS4_BF_ONLY 纯回放(SEARCH 跳过, DS4_BF_TERM_MAXP=0 = 不进 sweep/反调/回扫),
#       对 dql 层文件按 op 链字节回放全 43 层 → 与 FP 锚同口径打 VERDICT。只读产物, 不改写。
# 前提: gguf/go-onebit/layers/dql_L*.bin 全齐 — 必须在 merge consume 之前运行!
# 产物: /tmp/rr_verdict.out 全量原始输出; stdout 摘要(VERDICT/回放累积行)。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
QDIR="$ROOT/gguf-tools/go-onebit/quant"
LDIR="$ROOT/gguf/go-onebit/layers"
IDS="${1:-/tmp/rr_hard.ids}"; NTOK="${2:-64}"
NLAY="${DS4_NL:-43}"   # NL 覆盖口(2026-07-16): 小探针标定用 NL=6 局部判决(相对排序口径)
[ -f "$IDS" ] || { echo "[rr_verdict] 语料 $IDS 缺失 — 拒跑" >&2; exit 2; }
CTOK=$(grep -c . "$IDS"); [ "$NTOK" -gt "$CTOK" ] && NTOK=$CTOK
N_HAVE=$(ls "$LDIR"/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ' || true)
# ≥ 而非 =(2026-07-21): 中途部分层判决场景 — 满档跑到 L>NL 时前 NL 层文件已定型可回放
[ "$N_HAVE" -ge "$NLAY" ] || { echo "[rr_verdict] dql 层文件 $N_HAVE/$NLAY 不够(已被 merge consume?) — 拒跑" >&2; exit 3; }
cd "$QDIR"; [ -x ./ds4quant_run ] || ../scripts/quant_verify.sh build

# ---- 内存看门狗(铁律: 吃内存的运行必须自带; 只杀本判决进程, 不动别的) ----
WDOG(){ while true; do
    P=$(pgrep -nf "ds4quant_run $IDS" || true); [ -n "$P" ] || { sleep 5; continue; }
    MB=$(footprint -p "$P" 2>/dev/null | grep -Eo 'Footprint: *[0-9.]+ *[KMG]B' | head -1 \
         | awk '{v=$2;u=$3; if(u=="GB")v*=1024; else if(u=="KB")v/=1024; printf "%d",v}' || true)
    if [ -n "${MB:-}" ] && [ "$MB" -gt 11264 ]; then
        echo "[rr_verdict][watchdog] ${MB}MB >11G → 杀判决进程" >&2; kill -9 "$P" 2>/dev/null || true; fi
    sleep 5; done }
WDOG & WPID=$!
trap 'kill $WPID 2>/dev/null || true' EXIT

# ---- FP 锚(一次性, 尺寸闸防覆盖; 命名与主脚本不冲突) ----
BASE=$(basename "$IDS" .ids)
# NL≠43 时锚名带层数后缀: 不同层数锚尺寸不同, 同名撞防覆盖硬拒(2026-07-16 实证); 43 沿用旧名复用既有锚
if [ "$NLAY" = 43 ]; then ANCH="/tmp/ds4quant_anchor_rr_${BASE}_s${NTOK}.bin"
else ANCH="/tmp/ds4quant_anchor_rr_${BASE}_s${NTOK}_nl${NLAY}.bin"; fi
EXP_SZ=$(( NTOK*(NLAY*81968 + 517120) + 40 ))
if [ -f "$ANCH" ]; then
    ASZ=$(stat -f%z "$ANCH")
    [ "$ASZ" = "$EXP_SZ" ] || { echo "[rr_verdict] 锚 $ANCH 尺寸 $ASZ ≠ $EXP_SZ — 硬拒(防覆盖)" >&2; exit 2; }
else
    echo "[rr_verdict] FP建锚 S=$NTOK → $ANCH (一次性)" >&2
    DS4_ANCHOR="$ANCH" DS4_NL="$NLAY" DS4_FP_ONLY=1 DS4_THREADS="${DS4_THREADS:-6}" \
        ./ds4quant_run "$IDS" "$NTOK" >/tmp/rr_anchor_build.out 2>&1 \
        || { echo "[rr_verdict] 建锚失败, 见 /tmp/rr_anchor_build.out" >&2; exit 2; }
fi

# ---- 纯回放判决 ----
echo "[rr_verdict] 回放判决 S=$NTOK 语料=$IDS (BF_ONLY+MAXP=0: 只回放+VERDICT)" >&2
( export DS4_ANCHOR="$ANCH" DS4_NL="$NLAY" DS4_LCFG=g DS4_COADAPT=1 DS4_LAYER_DIR="$LDIR" \
         DS4_BF_ONLY=1 DS4_BF_TERM_MAXP=0 DS4_THREADS="${DS4_THREADS:-6}"
  exec ./ds4quant_run "$IDS" "$NTOK" ) >/tmp/rr_verdict.out 2>&1 || true
grep -aE "VERDICT|回放累积relL2|Σmin|ratio|KL|ppl|PPL" /tmp/rr_verdict.out || {
    echo "[rr_verdict] 无 VERDICT 行 — 看 /tmp/rr_verdict.out 尾部:" >&2; tail -5 /tmp/rr_verdict.out >&2; exit 1; }
