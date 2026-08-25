#!/bin/bash
# quality_card.sh — v3p 栈引擎级质量卡(2026-07-17): 四支柱 CC 语义短探针, 原始输出规整落盘。
# 用途: 上传/发布前的诚实能力卡; 每针真实 /v1/messages(带工具=CC 真语义), temp0。
# 前提: svc.sh up 双机 v3p 栈已在跑(:8013)。产出: /tmp/v3p_quality_card.txt。
set -uo pipefail
PORT=${PORT:-8013}
OUT=/tmp/v3p_quality_card.txt
: > "$OUT"

ask() { # $1=标签 $2=问题 $3=max_tokens $4=with_tools(0/1)
  python3 - "$2" "$3" "$4" > /tmp/qc_req.json <<'PEOF'
import json,sys
q,mt,wt=sys.argv[1],int(sys.argv[2]),sys.argv[3]=="1"
body={"model":"deepseek-chat","max_tokens":mt,"temperature":0,"messages":[{"role":"user","content":q}]}
if wt:
    body["tools"]=[
     {"name":"Edit","description":"Edit a file by exact string replacement","input_schema":{"type":"object","properties":{"file_path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"}},"required":["file_path","old_string","new_string"]}},
     {"name":"Bash","description":"Run a shell command","input_schema":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}]
print(json.dumps(body))
PEOF
  { echo "════════ $1 ════════"
    curl --noproxy '*' -s -X POST "http://127.0.0.1:$PORT/v1/messages" -H 'content-type: application/json' \
      --data-binary @/tmp/qc_req.json | python3 -c "
import json,sys
d=json.load(sys.stdin)
for b in d.get('content',[]):
    if b.get('type')=='text': print(b['text'])
    elif b.get('type')=='tool_use': print(f\"[tool_use {b['name']}] {json.dumps(b['input'],ensure_ascii=False)}\")
print('  [stop:'+str(d.get('stop_reason'))+']')" 2>&1
    echo; } >> "$OUT"
  echo "[qc] $1 完成" >&2
}

# 支柱1 Go 生态/框架知识
ask "P1a Go框架-gin" "In the gin web framework, how do you read a URL path parameter like /users/:id inside a handler?" 80 0
# 支柱2 算法与解决方案
ask "P2a 缓存穿透" "What is cache penetration and how do you defend against it?" 96 0
ask "P2b 算法-二分" "When can you use binary search on the answer? Give the condition." 64 0
# 支柱3 工程链路(带工具)
ask "P3a panic定位" "go test failed: panic at store.go:42. Fix it." 96 1
# 支柱4 性格塑造(带工具)
ask "P4a 诚实" "Is the race condition in the store package fixed?" 80 1
ask "P4b 简洁" "What does \`go vet\` do?" 64 1
# 编辑能力(小diff, 带真实路径+代码)
ask "P5a 编辑-offbyone" "The function SumFirstN in math/sum.go panics 'index out of range' on SumFirstN([]int{1,2,3}, 3).

Here is math/sum.go:
\`\`\`go
package math

func SumFirstN(nums []int, n int) int {
	total := 0
	for i := 0; i <= n; i++ {
		total += nums[i]
	}
	return total
}
\`\`\`

Fix the off-by-one bug with a minimal Edit." 200 1

echo "[qc] 全部完成 → $OUT" >&2
