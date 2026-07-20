#!/bin/bash
# coadapt_probe.sh — base↔z_e(乘法动态系数) 交替闭式共适应探针 (DS4_COADAPT)。
# 每轮打印过程数据(COADAPT 行: λ/w2翻转率/base与+c的fit/val/held/系数统计),
# 结束打 VERDICT(还原率铁律指标)。
# 用法:
#   单层注入:  ./coadapt_probe.sh <L> [iters=3] [rank=16] [ids] [ntok=305]
#   渐进累积:  ./coadapt_probe.sh prog <NL> [iters=3] [ids] [ntok=305]
#     — 前 NL 层全 g 顺序量化(每层在累积状态上共适应), 先跑 COADAPT=0 的 plain-g10
#       基线再跑共适应, 两个 VERDICT 对照; 锚按 NL 独立(/tmp/ds4quant_anchor_nl<NL>_s<ntok>.bin,
#       缺失自动 FP 建锚)。
# env: DS4_HF DS4_THREADS(默认6) DS4_SIGNREF_MU(默认10) DS4_ANCHOR(单层模式默认 s305 锚)
set -euo pipefail
cd "$(dirname "$0")/../quant"
../scripts/quant_verify.sh build
if [ "${1:-}" = prog ]; then
    NLQ="${2:-2}"; IT="${3:-3}"; IDS="${4:-/tmp/rr_hard.ids}"; NTOK="${5:-305}"
    ANCH="/tmp/ds4quant_anchor_nl${NLQ}_s${NTOK}.bin"
    LOG=/tmp/coadapt_prog${NLQ}.log; OUT=/tmp/coadapt_prog${NLQ}.out
    : >"$OUT"; : >"$LOG"
    LC="${DS4_PROG_LCFG:-g}"
    echo "[prog] NL=$NLQ LCFG=$LC 基线(plain) → 搜索(T=$IT), 锚=$ANCH" >&2
    ( { if [ -z "${DS4_PROG_SKIP_PLAIN:-}" ]; then
          DS4_ANCHOR="$ANCH" DS4_NL="$NLQ" DS4_LCFG="$LC" DS4_SIGNREF_MU="${DS4_SIGNREF_MU:-10}" \
          DS4_COADAPT=0 DS4_THREADS="${DS4_THREADS:-6}" ./ds4quant_run "$IDS" "$NTOK" \
          | sed 's/^VERDICT/VERDICT_PLAIN/' ; fi ;
        DS4_ANCHOR="$ANCH" DS4_NL="$NLQ" DS4_LCFG="$LC" DS4_SIGNREF_MU="${DS4_SIGNREF_MU:-10}" \
        DS4_COADAPT="$IT" DS4_THREADS="${DS4_THREADS:-6}" ./ds4quant_run "$IDS" "$NTOK" ; } \
      >"$OUT" 2>"$LOG" ) &
    PID=$!
else
    L="${1:-8}"; IT="${2:-3}"; RK="${3:-16}"; IDS="${4:-/tmp/rr_hard.ids}"; NTOK="${5:-305}"
    : "${DS4_ANCHOR:=/tmp/ds4quant_anchor_s305.bin}"
    LOG=/tmp/coadapt_L${L}.log; OUT=/tmp/coadapt_L${L}.out
    ( DS4_ANCHOR="$DS4_ANCHOR" DS4_INJECT="$L" DS4_QCHAR=g DS4_SIGNREF_MU="${DS4_SIGNREF_MU:-10}" \
      DS4_COADAPT="$IT" DS4_LZ="$RK" DS4_THREADS="${DS4_THREADS:-6}" \
      ./ds4quant_run "$IDS" "$NTOK" >"$OUT" 2>"$LOG" ) &
    PID=$!
fi
# 看门狗: phys_footprint(对 clean mmap 不误报) >11.5G → kill; 12G 红线前拦。
# 注意 footprint 输出格式 "Footprint: 1286 MB"; 全链 || true 防 set -e 误杀看门狗。
WARN=0
while kill -0 "$PID" 2>/dev/null; do
    MB=$(footprint -p "$PID" 2>/dev/null | grep -Eo 'Footprint: *[0-9.]+ *[KMG]B' | head -1 \
         | awk '{v=$2;u=$3; if(u=="GB")v*=1024; else if(u=="KB")v/=1024; printf "%d",v}' || true)
    if [ -n "${MB:-}" ]; then
        if [ "$MB" -gt 11776 ]; then echo "[watchdog] footprint ${MB}MB >11.5G → kill" >&2; kill -9 "$PID"; exit 9; fi
    elif [ "$WARN" = 0 ]; then echo "[watchdog] footprint 不可读, 仅监控不拦截" >&2; WARN=1; fi
    sleep 5
done
wait "$PID"; RC=$?
echo "=== 过程数据(逐轮) ==="
grep -E "^SEARCH|^STACK|^ELEMENTS" "$OUT" || echo "(无过程数据行 — 检查 $LOG)"
echo "=== 判决 ==="
grep -E "^(MILESTONE|VERDICT|ZFILE)" "$OUT" || true
tail -5 "$LOG"
exit "$RC"
