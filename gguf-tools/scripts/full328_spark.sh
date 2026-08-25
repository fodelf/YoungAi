#!/bin/bash
# full328_spark.sh — Spark(GB10) 全量代码题验证: HumanEval-Py 164 + HumanEval-X-Go 164。
# 用法: MODEL=gguf/ds4-final86v4.gguf TAG=v4full bash full328_spark.sh
# 进度: /tmp/full328.log; 逐题 jsonl 在 gguf-tools/reports/pubbench/。
set -u
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
MODEL="${MODEL:-gguf/ds4-final86v4.gguf}"
TAG="${TAG:-v4full}"
PORT="${PORT:-8000}"
LIMIT="${LIMIT:-164}"
SUITES="${SUITES:-humaneval humaneval-x-go}"
LOG(){ echo "[full328 $(date +%H:%M:%S)] $*"; }

cd "$ROOT"
pkill -9 -x ds4-server 2>/dev/null; pkill -9 -x ds4 2>/dev/null; sleep 2; rm -f /tmp/ds4.lock
LOG "起 server: $MODEL port=$PORT"
./ds4-server -m "$MODEL" --port "$PORT" --ctx 8192 --nothink > /tmp/full328_srv.log 2>&1 &
SP=$!
up=0
for _ in $(seq 1 200); do
    curl --noproxy '*' -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" && { up=1; break; }
    kill -0 "$SP" 2>/dev/null || break
    sleep 2
done
[ "$up" = 1 ] || { LOG "★server 未就绪★ $(tail -2 /tmp/full328_srv.log)"; exit 1; }
cd gguf-tools/scripts
for S in $SUITES; do
    LOG "server 就绪, 起 $S ×$LIMIT"
    DS4_URL="http://127.0.0.1:$PORT" PUBBENCH_CACHE="$ROOT/gguf-tools/bench/data" \
        SUITE="$S" TAG="$TAG" LIMIT="$LIMIT" SMOKE=0 bash pubbench.sh 2>&1
done
LOG "跑批收官"
pkill -9 -x ds4-server 2>/dev/null
for S in $SUITES; do
    J="../reports/pubbench/pubbench_${S}_${TAG}.jsonl"
    [ -f "$J" ] && python3 - "$J" <<'PYEOF'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
p=sum(1 for r in rows if r.get("pass"))
print(f"{sys.argv[1]}: {p}/{len(rows)} pass")
PYEOF
done
