#!/bin/bash
# ctx_1m_memory_gate.sh — 1M 上下文档的内存门(2026-09-22, 用户令"上下文设置为 1m")。
#
# 为什么要这把尺: V4.1 的请求状态是**按 ctx 分配的, 与提示多长无关**(core_v41.h 的账:
# iscore[512][ctx] + cand[512][ctx] + 候选暂存 ⇒ 524288 约 2.4 GiB, 1048576 约 4.8 GiB)。
# 所以把上限从 500k 抬到 1M, 每条请求都多吃 2.4 GiB —— 121 GB 的机器上装着 113.6 GB 的模型,
# 这 2.4 GiB 是从 MemAvailable 里出的。不量就上线 = 赌看门狗不会在早盘把服务杀掉。
#
# 跑什么: 发一条真实规模的长提示(默认 13 万 token, 与产品最长的那条选股请求同量级), 解码只要几个 token,
# 期间每 10 秒记一次 MemAvailable, 最后报最低点与看门狗红线的距离。
#
# 用法: ctx_1m_memory_gate.sh [host:port] [提示 token 数] [红线 MB]
set -uo pipefail
HOSTPORT="${1:-127.0.0.1:8000}"
NTOK="${2:-130000}"
RED_MB="${3:-2500}"
REQ=/tmp/ctx1m_req.json

python3 - "$NTOK" "$REQ" <<'PY'
import json, sys
n, path = int(sys.argv[1]), sys.argv[2]
# 每段约 40 token 的中文, 拼到目标 token 数(按 0.9 token/字 估, 宁可多不可少)
seg = "第{}条：市场消息，某公司发布公告称将调整生产计划并披露最新经营数据，请阅读后备用。\n"
text = "".join(seg.format(i) for i in range(n // 35))
req = {"model": "deepseek-chat", "temperature": 0, "max_tokens": 8,
       "messages": [{"role": "user", "content": text + "\n上面一共有多少条消息？只回答数字。"}]}
open(path, "w").write(json.dumps(req, ensure_ascii=False))
print(f"[gate] 提示文件 {path} 约 {len(text)} 字")
PY

LOW=999999
( while :; do
    A=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    echo "$A" >> /tmp/ctx1m_mem.txt
    sleep 10
  done ) &
SAMPLER=$!
: > /tmp/ctx1m_mem.txt
echo "[gate $(date +%H:%M:%S)] 发请求(解码只要 8 个 token, 时间基本都在预填)"
T0=$(date +%s)
OUT=$(curl -s -m 3000 "http://$HOSTPORT/v1/chat/completions" -H 'Content-Type: application/json' --data-binary @"$REQ")
T1=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "[gate] 用时 $((T1-T0)) 秒"
echo "$OUT" | python3 -c '
import json, sys
r = json.loads(sys.stdin.read())
u = r.get("usage", {})
print("  prompt_tokens =", u.get("prompt_tokens"), "completion_tokens =", u.get("completion_tokens"),
      "finish =", r["choices"][0].get("finish_reason"))
print("  回答:", repr((r["choices"][0]["message"].get("content") or "")[:60]))' || { echo "  ★服务没给正常 JSON★: $(echo "$OUT" | head -c 300)"; exit 1; }
LOW=$(sort -n /tmp/ctx1m_mem.txt | head -1)
echo "[gate] MemAvailable 最低 ${LOW} MB (采样 $(wc -l < /tmp/ctx1m_mem.txt) 次, 看门狗红线 ${RED_MB} MB)"
if [ "$LOW" -le "$RED_MB" ]; then echo "  ✗ 摸到红线, 1M 档在这台机器上不安全"; exit 1; fi
echo "  ✓ 距红线还有 $((LOW-RED_MB)) MB"
