#!/usr/bin/env bash
# 并发吞吐实测: N 路同时打同一个 server, 量聚合 token/s 与单路 token/s。
# 引擎若无请求批处理, 聚合 ≈ 单路(纯排队); 有批处理则聚合应随 N 上升。
set -uo pipefail
PORT=${PORT:-8099}
N=${N:-4}
MAXTOK=${MAXTOK:-96}
PROMPTS=("Write a Python function that reverses a string." "Explain how a hash map works." "Write a bash script that finds duplicate files." "Describe the CAP theorem.")
run_one() {
  local i=$1 p="$2" t0 t1
  t0=$(python3 -c 'import time;print(time.time())')
  local out
  out=$(curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"deepseek-v4-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":$MAXTOK,\"temperature\":0}")
  t1=$(python3 -c 'import time;print(time.time())')
  local ct
  ct=$(printf '%s' "$out" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("usage",{}).get("completion_tokens",0))' 2>/dev/null || echo 0)
  echo "$i $ct $(python3 -c "print($t1-$t0)")" >> /tmp/conc_res.txt
}
rm -f /tmp/conc_res.txt
echo "=== 单路基准 ==="
S=$(python3 -c 'import time;print(time.time())')
run_one 0 "${PROMPTS[0]}"
E=$(python3 -c 'import time;print(time.time())')
python3 - <<PY
rows=[l.split() for l in open("/tmp/conc_res.txt")]
tok=sum(int(r[1]) for r in rows); wall=$E-$S
print(f"单路: {tok} token / {wall:.2f}s = {tok/wall:.2f} t/s")
PY
rm -f /tmp/conc_res.txt
echo "=== $N 路并发 ==="
S=$(python3 -c 'import time;print(time.time())')
for i in $(seq 0 $((N-1))); do run_one "$i" "${PROMPTS[$((i % 4))]}" & done
wait
E=$(python3 -c 'import time;print(time.time())')
python3 - <<PY
rows=[l.split() for l in open("/tmp/conc_res.txt")]
tok=sum(int(r[1]) for r in rows); wall=$E-$S
per=[f"{r[1]}tok/{float(r[2]):.1f}s" for r in rows]
print(f"{len(rows)} 路: 合计 {tok} token / {wall:.2f}s = {tok/wall:.2f} t/s (聚合)")
print("  各路:", " ".join(per))
PY
