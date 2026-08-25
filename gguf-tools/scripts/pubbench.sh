#!/bin/bash
# pubbench.sh — 公共标尺一键驱动: 先 2 题冒烟(最小端到端铁律), 过了再放全 20 题。
# 用法:
#   DS4_URL=http://127.0.0.1:8080 TAG=vq14 ./pubbench.sh            # HumanEval-20 Python
#   SUITE=humaneval-x-go TAG=vq14 ./pubbench.sh                     # Go 20 题(judge=experimental)
#   ./pubbench.sh compare BASE.jsonl QUANT.jsonl                    # delta 表 + 判决闸(顺序=基线在前)
#     判决闸输出两个独立决策: [GOAI 参赛闸](goaihz.com, 三赛道截止 08-16) 与 [买机闸];
#     阈值=决策带(经得起±2题噪声), 编码 07-25 对话决策逻辑, 非大赛评审标准。
# 判决口径: 同套题跑 TAG=base(基线部署) 与 TAG=vq14(量化部署) 各一遍, compare 出 delta。
# 原始输出: gguf-tools/reports/pubbench/*.jsonl (逐题 prompt/completion/raw 全量)。
set -u
cd "$(dirname "$0")"

if [ "${1:-}" = "compare" ]; then
  exec "$(dirname "$0")/../bench/pubbench" --compare "$2" "$3"
fi

SUITE="${SUITE:-humaneval}"
TAG="${TAG:-run}"
LIMIT="${LIMIT:-20}"
URL="${DS4_URL:-http://127.0.0.1:8080}"
SMOKE="${SMOKE:-1}"

echo "[pubbench] suite=$SUITE tag=$TAG url=$URL limit=$LIMIT" >&2

if [ "$SMOKE" = "1" ]; then
  echo "[pubbench] smoke: 2 题先行, 判据=请求通+抽取出代码体+judge 正常给出 PASS/FAIL" >&2
  "$(dirname "$0")/../bench/pubbench" --suite "$SUITE" --url "$URL" --tag "${TAG}_smoke" --limit 2 || {
    echo "[pubbench] smoke FAILED — 不放全量, 先查 server 字段/抽取/judge" >&2
    exit 1
  }
  # 请求层错误(HTTP 500/超时)也必须拦: run_suite 对 request 错误记 FAIL 但退出码仍 0
  # (2026-07-27 实证: 20 题全 500 仍"smoke 通过"放行全量 → 空跑)。
  SMOKE_JSONL="../reports/pubbench/pubbench_${SUITE}_${TAG}_smoke.jsonl"
  if grep -q '"err": "request:' "$SMOKE_JSONL" 2>/dev/null; then
    echo "[pubbench] smoke FAILED — 存在 request 层错误(server 不通/500), 不放全量" >&2
    exit 1
  fi
  echo "[pubbench] smoke 通过, 放全量 $LIMIT 题" >&2
fi

exec "$(dirname "$0")/../bench/pubbench" --suite "$SUITE" --url "$URL" --tag "$TAG" --limit "$LIMIT"
