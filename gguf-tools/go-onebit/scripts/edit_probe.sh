#!/bin/bash
# edit_probe.sh — Edit 通路决胜针(分钟级): 常驻栈上测"读过的文件+明确修复指令→正确 edit 调用"。
# 判决口径(07-16 Clamp 三连): old_string 对=值拷贝通; new_string 对=写码链路打穿;
# new==old 或抄文件后续=已知吸引子(契约约束未落地的预期形态)。
# 用法: [PORT=8013] [MAXTOK=200] [FORCE=1] ./edit_probe.sh   输出: /tmp/edit_probe.report
#   FORCE=1 → tool_choice={"type":"tool","name":"Edit"} (修② tool_choice 具名强制口径)
set -uo pipefail
PORT="${PORT:-8013}"
MAXTOK="${MAXTOK:-200}"
REPORT=/tmp/edit_probe.report
PORT="$PORT" MAXTOK="$MAXTOK" FORCE="${FORCE:-}" python3 - <<'PEOF' | tee "$REPORT"
import json,os,urllib.request
port=os.environ["PORT"]; mt=int(os.environ["MAXTOK"])
force=os.environ.get("FORCE")=="1"
filebody="""package clamp

// clamp limits v to the range [lo, hi].
func clamp(v, lo, hi int) int {
	if v > hi {
		return hi
	}
	if v < lo {
		return hi
	}
	return v
}"""
user=("Here is clamp.go:\n```go\n"+filebody+"\n```\n"
      "Bug: when v < lo, clamp returns hi but it should return lo. "
      "Fix it using the Edit tool: replace the wrong return with the correct one.")
body={"model":"deepseek-chat","max_tokens":mt,"temperature":0,
 "messages":[{"role":"user","content":user}],
 **({"tool_choice":{"type":"tool","name":"Edit"}} if force else {}),
 "tools":[{"name":"Edit","description":"Replace old_string with new_string in the file",
   "input_schema":{"type":"object","properties":{
     "file_path":{"type":"string"},
     "old_string":{"type":"string"},
     "new_string":{"type":"string"}},
     "required":["file_path","old_string","new_string"]}}]}
req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/messages",
 data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(req,timeout=1800) as r:
    d=json.loads(r.read())
for blk in d.get("content",[]):
    if blk.get("type")=="text": print("[text]",blk["text"])
    elif blk.get("type")=="tool_use": print("[tool_use Edit]",json.dumps(blk["input"],ensure_ascii=False))
print("[stop_reason]",d.get("stop_reason"),d.get("usage"))
PEOF
