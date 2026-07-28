#!/bin/bash
# cli_coding_probe.sh — CLI 双机裸续写版编码探针(2026-07-26)。
# 背景: server /v1/messages 路对 VQ 模型产词沙拉(prog_probes_vqfinal), CLI 双机裸续写正常
#       (twoSum/LRU 连贯) → 用户裁决: 可用性判定一律走 CLI, 不走 server。
# 口径: go_013 (LRU Cache, benchmarks/go-bench/problems/go_013) 判定 = 生成代码可编译+过基础测试。
#   BASE 模型 → BOS+代码前缀裸续写(与 dual_vq.sh twoSum 同机制), greedy temp0。
# 用法: [NPRED=360] [VQ_GPU=0|1] [TAG=vq] ./cli_coding_probe.sh
#   产物: reports/cli_coding_<TAG>_<date>.report (原始输出) + go test 判定行。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
NPRED="${NPRED:-360}"; TAG="${TAG:-vq}"
# ctx 必须容纳 prompt+NPRED+投机批 KV 行开销; 512 默认在 ~400 token 处撞压缩 KV 容量墙(v4 实证)。
export CTX="${CTX:-2048}"
RPT="$ROOT/gguf-tools/go-onebit/reports/cli_coding_${TAG}_$(date +%F).report"
WORK="${WORK:-/tmp/cli_coding_probe_$TAG}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"

# go_013 LRU 裸续写前缀: prompt.md 的签名约束翻译成代码上下文(package solution + container/list)。
# v2: 固定 API 契约进包注释(真实工程文件头形式) — 无此契约模型自发 NewLRU/Get(int)->int,
#     代码自洽可编译但过不了官方 solution_test.go(NewLRUCache/(int,bool)) = 口径失配非质量问题。
# CONTRACT=0 切回无契约自由发挥口径(判"模型自发能否写出自洽完整 LRU", 官方 test 不作数)。
API_BLOCK='
//
// Required API (must match exactly):
//
//	func NewLRUCache(capacity int) *LRUCache
//	func (c *LRUCache) Get(key int) (int, bool)   // (0,false) on miss
//	func (c *LRUCache) Put(key int, value int)    // evict LRU key over capacity'
[ "${CONTRACT:-1}" = 0 ] && API_BLOCK=''
PREFIX='// Package solution implements an LRU (least-recently-used) cache with
// O(1) Get and Put, using a map plus a container/list doubly linked list.'"$API_BLOCK"'
package solution

import "container/list"

// entry is stored in each list element: key lets eviction delete the map slot.
type entry struct {
	key, value int
}

// LRUCache evicts the least-recently-used key once len(items) exceeds capacity.
type LRUCache struct {'
PROMPT="<｜begin▁of▁sentence｜>${PREFIX}"

mkdir -p "$(dirname "$RPT")" "$WORK"
printf '%s' "$PREFIX" > "$WORK/prefix.txt"
: > "$RPT"
echo "════ go_013 LRU (CLI 双机裸续写, NPRED=$NPRED VQ_GPU=${VQ_GPU:-0}) ════" >> "$RPT"
echo "── 前缀:" >> "$RPT"; printf '%s\n' "$PREFIX" >> "$RPT"
echo "── 续写(原始):" >> "$RPT"

PROMPT="$PROMPT" NPRED="$NPRED" RUN_TIMEOUT="$RUN_TIMEOUT" VQ_GPU="${VQ_GPU:-0}" \
  "$ROOT/tools/dual_vq.sh" 2> "$WORK/dual_vq.stderr"
RC=$?
# 取证保全: dual_vq.sh 的日志是固定 /tmp 路径, 下一次运行即覆盖 — 立刻快照进 WORK。
cp /tmp/dual_vq_coord.log "$WORK/coord.log" 2>/dev/null
cp /tmp/dual_vq_coord.out "$WORK/coord.out" 2>/dev/null
ssh "${REMOTE:-192.168.1.2}" "cat /tmp/dual_vq_worker.log" > "$WORK/worker.log" 2>/dev/null
GEN=$(cat /tmp/dual_vq_coord.out 2>/dev/null)
printf '%s\n' "$GEN" >> "$RPT"
echo "[dual_vq rc=$RC]" >> "$RPT"
grep -a "prefill:.*generation" /tmp/dual_vq_coord.log 2>/dev/null | tail -1 >> "$RPT"

# 组装完整文件: 前缀 + 续写, 截到花括号配平的最长前缀(去掉尾部未完成函数)
python3 - "$WORK" "$RPT" <<'PEOF'
import sys, os, re
work, rpt = sys.argv[1], sys.argv[2]
raw = open('/tmp/dual_vq_coord.out').read() if os.path.exists('/tmp/dual_vq_coord.out') else ''
prefix = open(os.path.join(work, 'prefix.txt')).read() if os.path.exists(os.path.join(work,'prefix.txt')) else ''
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

# 判定: 官方 solution_test.go, go build + go test
cp "$ROOT/benchmarks/go-bench/problems/go_013/solution_test.go" "$WORK/" 2>/dev/null
( cd "$WORK" && rm -f go.mod && go mod init solution >/dev/null 2>&1
  echo "── go build:" >> "$RPT"
  if go build ./... >> "$RPT" 2>&1; then echo "PASS(可编译)" >> "$RPT"; else echo "FAIL(编译错)" >> "$RPT"; fi
  echo "── go test (官方 solution_test.go):" >> "$RPT"
  if go test -run . -count=1 ./... >> "$RPT" 2>&1; then echo "PASS(测试过)" >> "$RPT"; else echo "FAIL(测试挂)" >> "$RPT"; fi )
echo "[cli-probe] 归档 → $RPT (组装文件 $WORK/solution.go)" >&2
tail -6 "$RPT" >&2
