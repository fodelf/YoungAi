#!/usr/bin/env bash
# dsml_scale_sweep.sh — corr 侧车缩放甜点扫描。
# 已知: 1.0=乱码, 0.25=连贯但值=占位符(=baseline)。扫中间档找"连贯且改善填值"。
# 每档: 缩放 corr_C → 同步 worker → 重启双端 → 单条探针取 file_path 值。
#
# 判读: 值=真路径(/data/pipeline/server.go 类) → 甜点(侧车有效);
#       值=$占位符 → 连贯但无改善; 词汤(petabits/瘫) → 该档仍过大。
#
# 用法: dsml_scale_sweep.sh 0.4 0.5 0.6   (默认这三档)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
M1=192.168.1.2; M1DIR=/Users/fodelf/ds4-main; DPORT=5599; PORT=8013
BASE=${BASE:-/tmp/dsml_L21_30.gguf}   # L20 已排除的 v2 侧车
ENVSTR="DS4_DIST_REVERSE_CONNECT=1 DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=2048 DS4_DIST_PREFILL_CAP=2048 DS4_METAL_EXPERT_GATHER_THREADS=8 DS4_METAL_NO_MODEL_WARMUP=1 DS4_MEM_BUDGET_MB=12000"
PROBE='{"model":"ds4","max_tokens":128,"temperature":0,"tools":[{"name":"Read","description":"Reads a file from the local filesystem.","input_schema":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}],"messages":[{"role":"user","content":"Open /data/pipeline/server.go and inspect validateToken."}]}'
SCALES=("$@"); [ ${#SCALES[@]} -eq 0 ] && SCALES=(0.4 0.5 0.6)

for s in "${SCALES[@]}"; do
  echo "===== scale $s ====="
  python3 "$HERE/dsml_sidecar_scale.py" "$BASE" /tmp/dsml_sweep.gguf "$s" >/dev/null
  ssh "$M1" "pkill -f 'role worker'" 2>/dev/null; sleep 2
  scp -q /tmp/dsml_sweep.gguf "$M1:$M1DIR/dsml_sweep.gguf"
  ssh "$M1" "( cd $M1DIR && $ENVSTR nohup ./ds4 -m gguf/ds4-mono-mixed.gguf --corr dsml_sweep.gguf --role worker --listen $M1 $DPORT --layers 20:output -c 65536 --temp 0 --nothink ) > /tmp/ds4_worker_svc.log 2>&1 < /dev/null & echo ok" >/dev/null
  until ssh "$M1" "grep -q 'waiting for coordinator' /tmp/ds4_worker_svc.log" 2>/dev/null; do sleep 3; done
  # 内存红线守护 (worker 越 12G 同杀, OOM 铁律)
  rkb=$(ssh "$M1" "pgrep -f 'role worker' | head -1 | xargs -I{} ps -o rss= -p {}" 2>/dev/null | tr -d ' ')
  echo "  worker RSS $(( ${rkb:-0} / 1048576 ))G"
  pkill -f "ds4-server.*$PORT" 2>/dev/null; sleep 2
  ( cd "$ROOT" && env $ENVSTR nohup ./ds4-server -m gguf/ds4-mono-mixed.gguf --corr /tmp/dsml_sweep.gguf --role coordinator --coordinator "$M1" "$DPORT" --layers 0:19 -c 65536 --port "$PORT" --kv-disk-dir /tmp/ds4-kv-sweep --kv-disk-space-mb 8192 --max-output-tokens 512 --nothink --tool-primer > /tmp/ds4-sweep-svc.log 2>&1 & )
  until grep -qE "listening|refusing|error" /tmp/ds4-sweep-svc.log 2>/dev/null; do sleep 3; done
  out=$(curl -s --noproxy '*' -m 300 "http://127.0.0.1:$PORT/v1/messages" -H 'content-type: application/json' -d "$PROBE")
  val=$(echo "$out" | python3 -c "import sys,json
try:
    d=json.load(sys.stdin); c=d.get('content',[])
    tu=[b for b in c if b.get('type')=='tool_use']
    tx=[b for b in c if b.get('type')=='text']
    if tu: print('tool_use file_path =', repr(tu[0]['input'].get('file_path')))
    elif tx: print('TEXT(乱码?):', repr(tx[0]['text'][:120]))
    else: print('RAW:', json.dumps(d)[:200])
except Exception as e: print('PARSE-ERR:', e, sys.stdin.read()[:120])")
  echo "  scale $s → $val"
  echo "SWEEP_RESULT scale=$s $val"
done
echo "===== 扫描完成 ====="
