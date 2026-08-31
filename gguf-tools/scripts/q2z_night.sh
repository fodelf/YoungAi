#!/bin/bash
# q2z_night.sh — 夜跑编排器(2026-08-17): 串行保内存账(单任务峰值≤100G, 永不叠加)。
#   ①锚(CPU 4G) ②v2 基线基准: 328 题 + 单场景速度 + 4 并发速度(server 90G)
#   ③q2z 量化 43 层(CPU ≤60G) ④z 侧车。明早无论到哪步, 前面阶段的账都完整落盘。
# 用法: tmux new -d -s night "bash q2z_night.sh > /tmp/q2z_night.log 2>&1"
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/scripts"
MDL="$ROOT/gguf/ds4-final86v2.gguf"
RPT=/tmp/night_report
mkdir -p "$RPT"
LOG(){ echo "[night $(date +%H:%M:%S)] $*"; }

# ---------- ① 锚 ----------
LOG "阶段① 锚"
bash "$SC/q2z_spark.sh" anchor || { LOG "★锚失败★"; exit 1; }

# ---------- ② v2 基线: 基准 + 速度 ----------
LOG "阶段② v2 基线基准(server 起)"
pkill -x ds4-server 2>/dev/null; sleep 2; rm -f /tmp/ds4.lock
cd "$ROOT"
setsid nohup ./ds4-server -m "$MDL" --ctx 8192 > /tmp/ds4_server_bench.log 2>&1 < /dev/null &
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 5; done
curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 || { LOG "★server 没起来★"; tail -5 /tmp/ds4_server_bench.log; exit 2; }

LOG "② 单场景速度(server 侧 3 段生成计时)"
for i in 1 2 3; do
  python3 - <<'PYEOF' >> "$RPT/speed_single.txt"
import json, time, urllib.request
body = json.dumps({"model":"ds4","prompt":"Write a Go function that returns the sum of a slice of ints.\n","max_tokens":200,"temperature":0}).encode()
t0=time.time()
r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/completions",body,{"Content-Type":"application/json"}),timeout=600)
d=json.load(r); dt=time.time()-t0
n=d.get("usage",{}).get("completion_tokens",200)
print(f"single: {n} tok in {dt:.2f}s = {n/dt:.2f} t/s")
PYEOF
done
cat "$RPT/speed_single.txt"

LOG "② 4 并发速度(4×200 token 同发)"
python3 - <<'PYEOF' > "$RPT/speed_c4.txt"
import json, time, threading, urllib.request
res=[]
def one(i):
    p=["Write a Go function that reverses a string.\n","Write a Python function that checks prime.\n",
       "Write a JS function that flattens an array.\n","Write a C function that swaps two ints.\n"][i]
    body=json.dumps({"model":"ds4","prompt":p,"max_tokens":200,"temperature":0}).encode()
    t0=time.time()
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/completions",body,{"Content-Type":"application/json"}),timeout=1200)
    d=json.load(r); dt=time.time()-t0
    n=d.get("usage",{}).get("completion_tokens",200)
    res.append((i,n,dt))
t0=time.time()
th=[threading.Thread(target=one,args=(i,)) for i in range(4)]
[t.start() for t in th]; [t.join() for t in th]
wall=time.time()-t0
tot=sum(n for _,n,_ in res)
for i,n,dt in sorted(res): print(f"c4 stream{i}: {n} tok {dt:.1f}s = {n/dt:.2f} t/s")
print(f"c4 aggregate: {tot} tok wall {wall:.1f}s = {tot/wall:.2f} t/s agg = {tot/wall/4:.2f} t/s/流")
PYEOF
cat "$RPT/speed_c4.txt"

LOG "② 328 题基准"
cd "$ROOT/gguf-tools"
export PATH="$HOME/opt/go/bin:$PATH"
# (2026-08-31 env 大扫除: DS4_URL/PUBBENCH_CACHE → --url/--cache-dir; --api completions=真代码基准口径铁律)
"$(dirname "$0")/../bench/pubbench" --url http://127.0.0.1:8000 --cache-dir "$PWD/bench/data" --suite humaneval --limit 164 --tag f86v2 --api completions 2>&1 | tail -6 | tee "$RPT/bench_py.txt"
"$(dirname "$0")/../bench/pubbench" --url http://127.0.0.1:8000 --cache-dir "$PWD/bench/data" --suite humaneval-x-go --limit 164 --tag f86v2 --api completions 2>&1 | tail -6 | tee "$RPT/bench_go.txt"
LOG "② 基准完, 杀 server"
pkill -x ds4-server 2>/dev/null; sleep 3; rm -f /tmp/ds4.lock

# ---------- ③ q2z 量化 ----------
LOG "阶段③ q2z 量化 43 层"
bash "$SC/q2z_spark.sh" quant || { LOG "★量化失败(锚/基准账已保全)★"; exit 3; }

# ---------- ④ z 侧车 ----------
LOG "阶段④ z 侧车"
bash "$SC/q2z_spark.sh" zside || { LOG "★z侧车失败★"; exit 4; }
LOG "夜跑全部收官"
