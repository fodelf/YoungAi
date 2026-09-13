#!/bin/bash
# decision_probe.sh — 决策探针(2026-09-08): 拿夜间样本自己的材料, 只问三个数字(入场/目标/止损),
# 微调前后各打一次, 直接看"同一份材料下决策变没变"。
#
# 为什么不用 CFO 全流程重跑: 实撞两个坑 —— ①温度 0 + CFO 那条"必须详细输出七大节"的长提示,
# 贪心解码进重复环, 单份报告跑到 10950 token 还不停(原报告约 4900 token), 一只 10 分钟起;
# ②思考档下长提示会出现 thinking 不闭合 ⇒ content 为空 ⇒ crewAI 静默重试, 再烧 10 分钟。
# 对比只需要"同材料同问法, 前后两版模型各答一次", 三个数字就够判决策变化, 30 秒一只。
# (CFO 全流程重跑的接口仍在 quant_trading_flow /api/review/replay, 想要完整报告时再用。)
#
# 用法: decision_probe.sh <标签> [后端=http://192.168.2.203:8001] [服务=http://127.0.0.1:8000]
#   标签 = z0(微调前) / z1(微调后); 结果落 gguf/go-onebit/vqnight/probe_<标签>.json
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
TAG="${1:?标签(z0/z1)}"
BACKEND="${2:-http://192.168.2.203:8001}"
SRV="${3:-http://127.0.0.1:8000}"
OUT="$ROOT/gguf/go-onebit/vqnight/probe_$TAG.json"
mkdir -p "$(dirname "$OUT")"

curl -sf -m 60 "$BACKEND/api/review/finetune-samples?limit=20&full=true" -o /tmp/probe_samples.json \
    || { echo "★拉样本失败(Mac 后端没起?)★"; exit 1; }

python3 - "$SRV" "$OUT" <<'PY'
import json, sys, urllib.request

srv, out = sys.argv[1], sys.argv[2]
items = json.load(open("/tmp/probe_samples.json"))["items"]
# 问法固定: 只给决策日的材料, 只要三个数字。措辞一个字都不许在前后两版之间变 ——
# 变了就分不清"决策变化"来自模型还是来自问法。
ASK = ("\n以上是决策日可得的全部材料。请只输出一行 JSON, 不要任何解释、不要 markdown 围栏, 字段固定为:\n"
       '{"symbol": "股票代码", "entry_price": 入场价, "target_price": 目标价, "stop_loss": 止损位}\n')

res = []
for it in items:
    body = (it.get("body_prompt") or "").strip()
    if not body:
        continue
    req = urllib.request.Request(
        srv.rstrip("/") + "/v1/chat/completions",
        json.dumps({"model": "deepseek-chat", "temperature": 0, "max_tokens": 200,
                    "messages": [{"role": "user", "content": body + ASK}]}).encode(),
        {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900))
        txt = r["choices"][0]["message"].get("content", "").strip()
        usage = r.get("usage", {})
    except Exception as e:
        txt, usage = "★请求失败: %s★" % e, {}
    row = {"symbol": it["symbol"], "symbol_code": it.get("symbol_code", ""),
           "added_date": it.get("added_date", ""), "raw": txt,
           "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
    # 解析三个数字; 解析不出就留空, 原文照留(判读交给人)
    try:
        s = txt[txt.index("{"): txt.rindex("}") + 1]
        j = json.loads(s.replace("'", '"'))
        row["got_symbol"] = str(j.get("symbol", ""))
        for k in ("entry_price", "target_price", "stop_loss"):
            row[k] = float(j.get(k) or 0)
    except Exception:
        pass
    res.append(row)
    print("%s %s 入场 %s 目标 %s 止损 %s%s" % (
        row["symbol"], row["symbol_code"], row.get("entry_price"), row.get("target_price"), row.get("stop_loss"),
        "" if row.get("got_symbol", row["symbol"]) == row["symbol"] else
        "  ★串台: 报告里的代码是 %s★" % row.get("got_symbol")))
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
print("→", out)
PY
