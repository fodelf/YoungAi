#!/bin/bash
# cli_probe_m4_single.sh — M4 单机 CLI 裸续写行为门(2026-07-28, v4bf 合一文件)。
# 背景: 双机 lane 需 M1 侧模型文件, M1 已无(盘不容 53G) → 行为门走 M4 单机
# (合一 VQ GGUF 引擎自动加载, pubbench 时代已验证单机路径)。
# 口径与 cli_coding_probe.sh 同源: go_013 LRU 契约前缀, BOS 裸续写 greedy temp0,
# 判定=组装文件可编译+过官方 solution_test.go。数值 flag 与 dual_vq.sh NUM_FLAGS 同。
# 用法: [MODEL=gguf/go-onebit/ds4-vq4bf.gguf] [NPRED=360] [TAG=v4bf] ./cli_probe_m4_single.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
MODEL="${MODEL:-gguf/go-onebit/ds4-vq4bf.gguf}"
NPRED="${NPRED:-360}"; TAG="${TAG:-v4bf}"; CTX="${CTX:-2048}"
RUN_TIMEOUT="${RUN_TIMEOUT:-1800}"; MAXMB="${MAXMB:-12288}"
RPT="$ROOT/gguf-tools/reports/cli_single_${TAG}_$(date +%F_%H%M).report"
WORK=/tmp/cli_probe_single_$TAG; mkdir -p "$WORK" "$(dirname "$RPT")"

PREFIX='// Package solution implements an LRU (least-recently-used) cache with
// O(1) Get and Put, using a map plus a container/list doubly linked list.
//
// Required API (must match exactly):
//
//	func NewLRUCache(capacity int) *LRUCache
//	func (c *LRUCache) Get(key int) (int, bool)   // (0,false) on miss
//	func (c *LRUCache) Put(key int, value int)    // evict LRU key over capacity
package solution

import "container/list"

// entry is stored in each list element: key lets eviction delete the map slot.
type entry struct {
	key, value int
}

// LRUCache evicts the least-recently-used key once len(items) exceeds capacity.
type LRUCache struct {'
printf '%s' "$PREFIX" > "$WORK/prefix.txt"

cd "$ROOT"
: > "$RPT"
echo "════ go_013 LRU (M4 单机裸续写 $MODEL NPRED=$NPRED) ════" >> "$RPT"
# (env 大扫除 2026-08-31: MATH_SAFE 三件套成组升格为 --strict-fp; REPEAT_FREQ 引擎已无
#  此路; VQ_GPU 诊断口已删)
./ds4 -m "$MODEL" --strict-fp --prefill-chunk 8 --mem-budget-mb "$MAXMB" \
    -c "$CTX" -n "$NPRED" --temp 0 --seed 1 --nothink \
    -p "<｜begin▁of▁sentence｜>${PREFIX}" > "$WORK/gen.out" 2> "$WORK/gen.log" &
DPID=$!
T0=$(date +%s)
while kill -0 "$DPID" 2>/dev/null; do
  EL=$(( $(date +%s) - T0 ))
  [ "$EL" -ge "$RUN_TIMEOUT" ] && { echo "[watchdog] 超时 ${EL}s 杀" >> "$RPT"; kill -9 "$DPID"; break; }
  MB=$(footprint -p "$DPID" 2>/dev/null | grep -Eo 'Footprint: *[0-9.]+ *[KMG]B' | head -1 \
       | awk '{v=$2;u=$3; if(u=="GB")v*=1024; else if(u=="KB")v/=1024; printf "%d",v}')
  # fail-closed: footprint 读不出=失明, 按危险杀(旧版读空跳过判断=看门狗静默缴械)
  [ -z "${MB:-}" ] && { echo "[watchdog] footprint 读取失败(失明)=按危险处理 杀" >> "$RPT"; kill -9 "$DPID"; break; }
  [ "$MB" -gt "$MAXMB" ] && { echo "[watchdog] ${MB}MB>红线 杀" >> "$RPT"; kill -9 "$DPID"; break; }
  sleep 3
done
wait "$DPID" 2>/dev/null; RC=$?
echo "── 续写(原始, rc=$RC):" >> "$RPT"
cat "$WORK/gen.out" >> "$RPT"
grep -aE "prefill:.*generation|t/s" "$WORK/gen.log" | tail -2 >> "$RPT"

# 组装 + 编译 + 官方测试(与 cli_coding_probe.sh 同判定)
python3 - "$WORK" <<'PEOF'
import sys, os, re
work = sys.argv[1]
raw = open(os.path.join(work, 'gen.out')).read()
prefix = open(os.path.join(work, 'prefix.txt')).read()
full = prefix + raw
lines = full.splitlines()
depth = 0; last_ok = 0
for i, l in enumerate(lines):
    depth += l.count('{') - l.count('}')
    if depth == 0 and re.search(r'}\s*$', l):
        last_ok = i + 1
code = '\n'.join(lines[:last_ok]) + '\n' if last_ok else full
open(os.path.join(work, 'solution.go'), 'w').write(code)
print(f"[组装] 全文 {len(lines)} 行, 配平截断至 {last_ok} 行", file=sys.stderr)
PEOF
GODIR="$ROOT/benchmarks/go-bench/problems/go_013"
if [ -d "$GODIR" ]; then
  rm -rf "$WORK/go_013"; cp -r "$GODIR" "$WORK/go_013"
  cp "$WORK/solution.go" "$WORK/go_013/solution.go"
  # 夹具无 go.mod → 建模块壳(canonical/plus 与生成解冲突, 移开只留官方 test)
  [ -f "$WORK/go_013/go.mod" ] || ( cd "$WORK/go_013" && go mod init solution >/dev/null 2>&1 )
  mv "$WORK/go_013/canonical.go" "$WORK/go_013/canonical.go.bak" 2>/dev/null
  mv "$WORK/go_013/solution_plus_test.go" "$WORK/go_013/solution_plus_test.go.bak" 2>/dev/null
  ( cd "$WORK/go_013" && go test . > "$WORK/gotest.out" 2>&1 )
  GRC=$?
  echo "── go test rc=$GRC:" >> "$RPT"; tail -5 "$WORK/gotest.out" >> "$RPT"
else
  echo "── go_013 题库缺, 只出编译判定:" >> "$RPT"
  ( cd "$WORK" && go vet ./solution.go > "$WORK/gotest.out" 2>&1 ); echo "vet rc=$?" >> "$RPT"
fi
echo "REPORT=$RPT"
cat "$RPT"
