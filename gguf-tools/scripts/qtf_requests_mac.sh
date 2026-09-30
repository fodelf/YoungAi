#!/bin/bash
# qtf_requests_mac.sh — Mac 侧: 从 qtf 容器里按当天口径重建真实请求(大盘 / 个股 CFO), 落到本仓 gguf/v41/night/req/,
# 再推到 spark 同路径, 给 z_nightly_spark.sh 的 sample 段用(2026-09-29, back.md §14.3 逐日回放)。
#
# 为什么在 Mac 上跑: qtf 的库与 crewAI 都在 Mac 的 OrbStack 容器里, 重建必须走 CfoCrew/MarketCrew 本身
# (拿到的是 litellm 真正发出去的字节); spark 只跑模型。请求 JSON 的 _note 里已带打分事实(次日 OHLC / 大盘真值),
# spark 侧不用再连库。
#
# 用法: qtf_requests_mac.sh market <日期>            → req/market_<日期>.json
#       qtf_requests_mac.sh cfo    <日期> <代码|all> → req/cfo_<日期>_<代码>.json(all = 这一天材料齐全的全部股票)
#       qtf_requests_mac.sh inventory                → 逐日盘点(只读)
# 已有的文件不重建(要重建先删)。每次结束 rsync 整个 req/ 到 spark。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CAP="$ROOT/gguf-tools/scripts/qtf_capture_request.py"
REQ="$ROOT/gguf/v41/night/req"; mkdir -p "$REQ"
CTR="quant_trading_flow-main-backend-1"
PY="/app/.venv/bin/python"
run_cap(){ docker exec -i "$CTR" "$PY" - "$@" < "$CAP"; }
docker ps --format '{{.Names}}' | grep -qx "$CTR" || { echo "★qtf 后端容器 $CTR 没起(OrbStack/docker)★"; exit 2; }

build_one(){   # <kind> <date> [symbol]
    local kind="$1" date="$2" sym="${3:-}" out
    if [ "$kind" = market ]; then out="$REQ/market_${date}.json"; else out="$REQ/cfo_${date}_${sym}.json"; fi
    if [ -s "$out" ]; then echo "  已有 $(basename "$out")"; return 0; fi
    echo "  重建 $(basename "$out") …"
    if [ "$kind" = market ]; then run_cap market "$date" > "$out.tmp" 2> "$out.log"
    else run_cap cfo "$date" "$sym" > "$out.tmp" 2> "$out.log"; fi
    if [ -s "$out.tmp" ] && python3 -c 'import json,sys; b=json.load(open(sys.argv[1])); assert b.get("messages") and b.get("_note")' "$out.tmp" 2>/dev/null; then
        mv "$out.tmp" "$out"; rm -f "$out.log"
        python3 - "$out" <<'PY'
import json, sys
b = json.load(open(sys.argv[1])); n = b["_note"]
extra = ""
if n["kind"] == "cfo":
    extra = " 次日OHLC=%s 早盘JSON=%s" % (n["ohlc"][:2] if n.get("ohlc") else n.get("ohlc_error", "无"), n.get("live_json"))
else:
    extra = " 真值=%s" % n.get("truth")
print("    %s: messages=%s stop=%s end_date=%s%s" % (n["kind"], [(m["role"], len(m["content"])) for m in b["messages"]], b.get("stop"), n["end_date"], extra))
PY
    else
        echo "  ★重建失败, 看 $out.log 尾:★"; tail -5 "$out.log" | cut -c1-200; rm -f "$out.tmp"; return 1
    fi
}

case "${1:-}" in
  inventory) run_cap inventory;;
  market) build_one market "${2:?日期}";;
  cfo)
    date="${2:?日期}"; sym="${3:?代码|all}"
    if [ "$sym" = all ]; then
        for s in $(run_cap symbols "$date"); do build_one cfo "$date" "$s"; done
    else build_one cfo "$date" "$sym"; fi;;
  *) echo "用法: $0 inventory | market <日期> | cfo <日期> <代码|all>"; exit 2;;
esac
[ "${1:-}" = inventory ] && exit 0
echo "推到 spark: gguf/v41/night/req/"
rsync -az "$REQ/" spark:ds4-main/gguf/v41/night/req/ && echo "  OK $(ls "$REQ"/*.json 2>/dev/null | wc -l | tr -d ' ') 个请求"
