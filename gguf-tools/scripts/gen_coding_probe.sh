#!/bin/bash
# gen_coding_probe.sh — 泛化编码探针(2026-07-25): 非针集、真实中长编码任务, 直接抽样"最终编码场景"。
# 背景: 熵门在 6 针回归集上定操作点后, 用户担忧针外泛化 → 本探针 = 3 个从未当过针的
# 300-500 token 生成任务(三语言), 判据 = 乱码启发式计数(下划线链/非拉丁突发/整块复读) + 原始输出归档。
# 用法: [PORT=8013] [MAXTOK=500] [TAG=xxx] ./gen_coding_probe.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
PORT="${PORT:-8013}"; MAXTOK="${MAXTOK:-500}"; TAG="${TAG:-gen}"
RPT="$ROOT/gguf-tools/reports/gen_coding_${TAG}_$(date +%F).report"
: > "$RPT"
export no_proxy='*' NO_PROXY='*'; unset http_proxy https_proxy 2>/dev/null || true

TASKS=(
"Write a complete LRU cache class in Go with Get and Put methods, using a doubly linked list and a map. Include brief comments."
"Write a Python HTTP file server using http.server that serves files from a directory and logs each request with timestamp and path."
"Write a Rust worker pool: spawn N threads consuming jobs from a channel, collect results, and shut down cleanly."
)
i=0
for T in "${TASKS[@]}"; do
    i=$((i+1))
    echo "════ 任务 $i ════" >> "$RPT"
    echo "── 提示: $T" >> "$RPT"
    echo "── 回答(原始):" >> "$RPT"
    RESP=$(curl -s -m 900 -X POST "http://127.0.0.1:$PORT/v1/messages" \
        -H 'Content-Type: application/json' \
        -d "$(python3 - "$T" "$MAXTOK" <<'PEOF'
import json, sys
print(json.dumps({"model":"ds4","max_tokens":int(sys.argv[2]),"mode":"code",
                  "messages":[{"role":"user","content":sys.argv[1]}]}))
PEOF
)")
    python3 - "$RESP" >> "$RPT" <<'PEOF'
import json, sys, re
try:
    r = json.loads(sys.argv[1])
    txt = "".join(b.get("text","") for b in r.get("content",[]))
except Exception as e:
    txt = f"[解析失败: {e}] {sys.argv[1][:200]}"
print(txt)
# 乱码启发式(附注, 非判决): 下划线链≥4段 / 非拉丁非中文突发 / 连续整行复读
u = len(re.findall(r"(?:\w+_){4,}\w+", txt))
nl = len(re.findall(r"[֐-ࣿ가-힯]{2,}", txt))     # 希伯来/阿拉伯/韩文等突发
lines = [l for l in txt.splitlines() if l.strip()]
rep = sum(1 for a, b in zip(lines, lines[1:]) if a == b)
print(f"\n[启发式计数] 下划线链={u} 非拉丁突发={nl} 相邻整行复读={rep}")
PEOF
    echo >> "$RPT"
    echo "[gen-probe $i/3] 完成" >&2
done
echo "[gen-probe] 归档 → $RPT" >&2
