#!/bin/bash
# prog_sweep.sh — 编程全场景 43 针快速验证链(2026-07-22 固化; 用户令: 正向操作全落脚本可链式调用)。
# 链: [FRESH=1 清KV重启栈(防跨模型重放, 每针独立措辞铁律的KV面)] → 15针全域 → 12针算法 →
#     12针四支柱 → 4针行为门 → 全部归档 reports/<面板>_<TAG>_<日期>.report → 汇总行。
# 前提: tools/svc.sh 主线栈可用(默认即冠军配置)。用法: [FRESH=1] [TAG=v3] ./prog_sweep.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
TAG="${TAG:-v3}"; DATE=$(date +%F); RPT="$ROOT/gguf-tools/reports"
mkdir -p "$RPT"

if [ "${FRESH:-0}" = 1 ]; then
    echo "[sweep] FRESH=1: 清 KV 重启栈(防跨模型/跨轮重放污染)" >&2
    "$ROOT/tools/svc.sh" down >/dev/null 2>&1; sleep 3
    rm -f /tmp/ds4-kv-svc/*.kv
    ( "$ROOT/tools/svc.sh" up > /tmp/svc_up_sweep.log 2>&1 & )   # 壳卡死病灶在案: 不等壳, 等信号
    until grep -qa listening /tmp/ds4-svc.log 2>/dev/null && pgrep -f '^\./ds4-server' >/dev/null; do sleep 5; done
fi
pgrep -f '^\./ds4-server' >/dev/null || { echo "[sweep] 栈不在(先 svc.sh up) — 拒跑" >&2; exit 2; }

cd "$HERE"
for C in prog_probes algo_probes pillar_probes; do
    echo "[sweep] ════ $C ════" >&2
    CORPUS="corpus/$C.txt" NPRED=28 PORT="${PORT:-8013}" ./pillar_probe_srv.sh
    SRC=/tmp/${C}_srv.report; [ "$C" = pillar_probes ] && SRC=/tmp/pillar_probe_srv.report
    cp "$SRC" "$RPT/${C}_${TAG}_${DATE}.report" && echo "[sweep] 归档 ${C}_${TAG}_${DATE}.report" >&2
done
echo "[sweep] ════ behavior_gate ════" >&2
rm -f /tmp/behavior_gate_at.report
./behavior_gate_at.sh
cp /tmp/behavior_gate_at.report "$RPT/behavior_gate_${TAG}_${DATE}.report"

echo "[sweep] ★43针全扫完成★ 归档于 $RPT/*_${TAG}_${DATE}.report" >&2
grep -c "════" "$RPT"/prog_probes_"${TAG}"_"${DATE}".report "$RPT"/algo_probes_"${TAG}"_"${DATE}".report \
     "$RPT"/pillar_probes_"${TAG}"_"${DATE}".report "$RPT"/behavior_gate_"${TAG}"_"${DATE}".report 2>/dev/null >&2
