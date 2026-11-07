#!/bin/bash
# 选轮会改变后续训练起点；在临时数据上验证小增益、倒退、舍入和损坏输入边界。
set -euo pipefail
cd "$(dirname "$0")/.."
source gguf-tools/scripts/vloop_gate.sh
vloop_better 13201 13202 15360 0.20733 0.20734
for pair in '13201 13201 15360 0.20733 0.20000' '13201 13200 15360 0.20733 0.10000' \
            '13201 13202 15360 0.20733 0.20834' '13201 15361 15360 0.20733 0.20733' \
            '13201 13202 15360 NaN 0.20733'; do
    if vloop_better $pair; then echo "错误接纳: $pair"; exit 1; fi
done
TASK_DIR=$(mktemp -d); trap 'rm -rf "$TASK_DIR"' EXIT
printf '0 0.2 0.8 1\n1 0.3 0.7 0\n2 0.1 0.9 1\n' > "$TASK_DIR/rows"
test "$(vloop_metrics "$TASK_DIR/rows" 3)" = '2 3 0.2'
if vloop_metrics "$TASK_DIR/rows" 4; then exit 1; fi
printf '0 0.1 0.9 1\n' >> "$TASK_DIR/rows"
if vloop_metrics "$TASK_DIR/rows" 4; then exit 1; fi
printf '0 NaN 0.9 1\n' > "$TASK_DIR/rows"
if vloop_metrics "$TASK_DIR/rows" 1; then exit 1; fi
printf '2 0.1 0.9 1\n' > "$TASK_DIR/rows"
if vloop_metrics "$TASK_DIR/rows" 1; then exit 1; fi
echo 'vloop monotonic gate: PASS'
