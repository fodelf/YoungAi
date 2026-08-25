#!/bin/bash
# pubbench_serial.sh — VQ lane 逐题隔离驱动(2026-07-27)。
# 背景: 常驻 server 跨请求 live-KV rewind 在分布式 VQ lane 断路(实证链:
#   rewind→worker KV prefix hash mismatch(设计内拒绝)→回退全量 rebuild→路由挂死/连接互断)。
#   根修=worker 侧 KV rewind 协议, 归 v2.3。本驱动绕过而不掩盖: 每题重启 server →
#   fresh session 全量 prefill(与 CLI dual_vq 已验证语义一致, 零 rewind 零 forget)。
#   worker 常驻不动(svc.sh 设计本意: server 可独立重启, worker 断开自动清 session 等重连)。
# 用法: [SUITE=humaneval|humaneval-x-go] [TAG=vq22] [N=20] ./pubbench_serial.sh
# 产物: reports/pubbench/pubbench_${SUITE}_${TAG}.jsonl (逐题合并) + 每题独立 _t<k>.jsonl
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
SUITE="${SUITE:-humaneval}"; TAG="${TAG:-vq22}"; N="${N:-20}"; PORT="${PORT:-8013}"
# VQ lane 已验证 env(与 CLI dual_vq 同源): span/批 ≤8(VQ scratch 墙), GPU F16W 路,
# 关 q2 IO 杠杆, 无残差侧车, PIPE_CHUNK 关(VQ lane 未验证)。
# ★域中立判定(2026-08-05 用户铁律: 域注入/定型配方全清, 裸模型裸判)★
export RESID= CTX=4096 PIPE_CHUNK= SOUL= KNOWLEDGE=
# 2026-07-28: DS4_VQ_DIR 摘除 — v4bf 起 blob 内嵌合一卷(引擎自动装载), 侧车目录已清
export EXTRA_ENV="DS4_PRIMER_BATCH_INJECT=0 DS4_METAL_PREFILL_CHUNK=${PB_CHUNK:-8} DS4_DIST_PREFILL_CAP=${PB_CHUNK:-8} DS4_VQ_GPU=1 DS4_METAL_EXPERT_PREAD=0 DS4_METAL_EXPERT_PREFETCH_AHEAD=0 DS4_METAL_EXPERT_EVENT_DRAIN=0 ${PB_EXTRA:-}"
OUT="$ROOT/gguf-tools/go-onebit/reports/pubbench"; mkdir -p "$OUT"
FINAL="$OUT/pubbench_${SUITE}_${TAG}.jsonl"; : > "$FINAL"
REQ_FAIL=0
# OFFSETS="1 4 11" 指定题号复测(2026-07-28, 断环器删除后贪心真相针); 缺省=0..N-1 全量
for k in ${OFFSETS:-$(seq 0 $((N-1)))}; do
  pkill -f "^\./ds4-server .*--port $PORT" 2>/dev/null; sleep 2
  "$ROOT/tools/svc.sh" up > /tmp/pubbench_serial_up.log 2>&1 || {
    echo "[serial] svc up 失败@题$k:" >&2; tail -3 /tmp/pubbench_serial_up.log >&2; exit 1; }
  echo "[serial] 题 $((k+1))/$N (offset=$k) 生成中…" >&2
  python3 "$HERE/pubbench.py" --suite "$SUITE" --url "http://127.0.0.1:$PORT" \
    --tag "${TAG}_t$k" --offset "$k" --limit 1 --api completions 2>&1 | grep -vE "^\[fetch\]" >&2
  TJ="$OUT/pubbench_${SUITE}_${TAG}_t$k.jsonl"
  [ -f "$TJ" ] && cat "$TJ" >> "$FINAL"
  # 前 2 题即冒烟: request 层错误连出 2 次 = server/协议问题, 不再空烧
  if grep -q '"err": "request:' "$TJ" 2>/dev/null; then
    REQ_FAIL=$((REQ_FAIL+1))
    [ "$k" -le 1 ] && [ "$REQ_FAIL" -ge 2 ] && { echo "[serial] 前2题全 request 错误 — 中止" >&2; exit 1; }
  fi
done
python3 - "$FINAL" <<'PEOF' >&2
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
p = sum(bool(r["pass"]) for r in rows)
print(f"[serial done] pass@1 = {p}/{len(rows)}  ({sys.argv[1]})")
for r in rows:
    print(f"  {r['task_id']}: {'PASS' if r['pass'] else 'FAIL'}"
          + (f" ({(r['err'].splitlines() or [''])[-1][:70]})" if r["err"] else ""))
PEOF
