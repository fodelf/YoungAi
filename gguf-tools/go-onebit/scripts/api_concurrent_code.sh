#!/usr/bin/env bash
# 8 道代码题并发实测。
# 口径(2026-08-20 已定): allq2 族丢了"思考退出"能力, chat 口会一直思考不出码;
# 正解是 /v1/completions BOS 裸续写(BASE 口径)+ amp42 反修链。
# 用法: [PORT=8099] [N=8] [MAXTOK=220] [API=completions|messages] bash api_concurrent_code.sh
set -uo pipefail
PORT=${PORT:-8099}
N=${N:-8}
MAXTOK=${MAXTOK:-220}
API=${API:-completions}
OUT=${OUT:-/tmp/api_code}
mkdir -p "$OUT"; rm -f "$OUT"/*.json "$OUT"/*.txt "$OUT"/timing.txt

# 裸续写口径: 给函数签名+docstring, 模型接着写函数体(HumanEval 同款)
PROMPTS=(
'def merge_intervals(intervals):
    """Merge overlapping intervals. intervals is a list of [start, end]. Return the merged list sorted by start."""
'
'def binary_search(arr, target):
    """Return the index of target in the sorted list arr, or -1 if absent."""
'
'def word_frequencies(text):
    """Return a dict mapping each lowercase word in text to how many times it occurs."""
'
'def is_balanced(s):
    """Return True if the brackets in s (), [], {} are balanced, else False."""
'
'def longest_common_prefix(strs):
    """Return the longest common prefix among the strings in strs, or "" if none."""
'
'def flatten(nested):
    """Flatten an arbitrarily nested list of lists into a single flat list."""
'
'def two_sum(nums, target):
    """Return indices [i, j] such that nums[i] + nums[j] == target, or [] if none."""
'
'def rotate_matrix(matrix):
    """Rotate an n x n matrix 90 degrees clockwise in place and return it."""
'
)

one() {
  local i=$1 p="$2"
  python3 - "$PORT" "$MAXTOK" "$OUT/$i.json" "$API" "$p" <<'PY'
import json,sys,urllib.request
port,maxtok,path,api,prompt = sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5]
if api=="completions":
    url=f"http://127.0.0.1:{port}/v1/completions"
    body={"model":"deepseek-v4-flash","prompt":prompt,"max_tokens":maxtok,"temperature":0}
else:
    url=f"http://127.0.0.1:{port}/v1/messages"
    body={"model":"deepseek-v4-flash","max_tokens":maxtok,"temperature":0,"think":False,
          "messages":[{"role":"user","content":prompt}]}
req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
try:
    with urllib.request.urlopen(req,timeout=900) as r: open(path,"wb").write(r.read())
except Exception as ex: open(path,"w").write(json.dumps({"error":str(ex)}))
PY
}

S=$(python3 -c 'import time;print(time.time())')
for i in $(seq 0 $((N-1))); do one "$i" "${PROMPTS[$i]}" & done
wait
E=$(python3 -c 'import time;print(time.time())')
python3 - "$OUT" "$S" "$E" "$API" <<'PY'
import json,sys,glob,os
out,S,E,api=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),sys.argv[4]
tot=0
for p in sorted(glob.glob(os.path.join(out,"*.json")), key=lambda x:int(os.path.basename(x)[:-5])):
    i=os.path.basename(p)[:-5]; d=json.load(open(p))
    if "error" in d: print(f"[{i}] 失败: {d['error']}"); continue
    if api=="completions":
        txt=d["choices"][0]["text"]; ot=d.get("usage",{}).get("completion_tokens",0)
        fin=d["choices"][0].get("finish_reason")
    else:
        txt="".join(b.get("text","") for b in d.get("content",[])); ot=d.get("usage",{}).get("output_tokens",0)
        fin=d.get("stop_reason")
    tot+=ot
    open(os.path.join(out,f"{i}.txt"),"w").write(txt)
    print(f"[{i}] tokens={ot} finish={fin}")
wall=E-S
print(f"\n合计 {tot} token / 墙钟 {wall:.2f}s ⇒ 聚合 {tot/wall:.2f} t/s")
PY
