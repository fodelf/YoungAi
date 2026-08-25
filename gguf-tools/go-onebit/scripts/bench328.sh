#!/bin/bash
# bench328.sh — 328 题终判发射器(模型/链参数化; amp86_spark.sh bench 段独立版)。
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
MDL="${B_MDL:-$ROOT/gguf/ds4-allq2.gguf}"
ZC="${B_ZC:-$ROOT/gguf/go-onebit/r30/full86/zchain_noge.bin}"
TAG="${B_TAG:-noge}"
LOG(){ echo "[b328 $(date +%H:%M:%S)] $*"; }
cd "$ROOT"
LOG "速度 (${TAG})"
./ds4 --cuda -m "$MDL" --zchain "$ZC" -n 128 -p "Write a Python quicksort function." </dev/null 2>&1 | grep -aE "t/s"
LOG "server 起"
./ds4-server --cuda -m "$MDL" --zchain "$ZC" --ctx 16384 > /tmp/b328_server.log 2>&1 &
SRV=$!
sleep 75
cd "$SC"
LOG "smoke 2题"
"$(dirname "$0")/../calib/pubbench" --suite humaneval --url http://127.0.0.1:8000 --tag ${TAG}_smoke --limit 2 2>&1 | tail -2
LOG "Py 164 (4并发)"
"$(dirname "$0")/../calib/pubbench" --suite humaneval --url http://127.0.0.1:8000 --tag $TAG --limit 164 --jobs 4 2>&1 | tail -3
LOG "Go 164 (4并发)"
"$(dirname "$0")/../calib/pubbench" --suite humaneval-x-go --url http://127.0.0.1:8000 --tag $TAG --limit 164 --jobs 4 2>&1 | tail -3
kill $SRV 2>/dev/null || true
LOG "b328 收官"
