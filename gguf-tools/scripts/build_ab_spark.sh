#!/bin/bash
# build_ab_spark.sh — spark 本机: 同一份源码编两份二进制: ds4/ds4-bench(token graph 开) +
# ds4.direct(图关), 供 tokgraph_ab_spark.sh 逐位对拍(图 vs 直发)与 kernel_parity 对拍(新 vs 老)。
#
# 为什么: 图开/图关只差 cuda_graphcap.inc.cu 里 `g_tok_graph_on = 1;` 一处; 手工 sed 来回改
# 容易把 ds4-bench 编成直发却当图开测(09-05 撞过: 曲线是图关的, 探针是图开的, 白比一轮)。
# 这里直发版只留 ds4.direct 一个文件, 其余二进制永远是图开态。
# 用法: build_ab_spark.sh [标签]  → ds4 ds4-bench …(图开); ds4.direct(图关); 有标签则再存 ds4.<标签>
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
TAG="${1:-}"
F=src/cuda/cuda_graphcap.inc.cu
LOG(){ echo "[build $(date +%H:%M:%S)] $*"; }
grep -q "g_tok_graph_on = 1;" "$F" || { LOG "★源码不是图开态($F 没有 'g_tok_graph_on = 1;')★"; exit 2; }
BUSY=$(for p in ds4 ds4-bench ds4-server ds4quant_run zlayer vq_merge_v4; do pgrep -x "$p"; done)
[ -z "$BUSY" ] || { LOG "★机器非空: $BUSY★"; exit 3; }
# ① 直发版: 开关翻 0 编一遍, 只留 ds4.direct, 源码立刻翻回(trap 保证异常退出也翻回)
cp "$F" /tmp/graphcap.on.bak
trap 'cp /tmp/graphcap.on.bak "$F"' EXIT
sed -i 's/g_tok_graph_on = 1;/g_tok_graph_on = 0;/' "$F"
make cuda-spark > /tmp/build_direct.log 2>&1; rc=$?
cp /tmp/graphcap.on.bak "$F"
[ $rc -eq 0 ] || { LOG "★直发版编译失败 rc=$rc★"; grep -m8 -iE "error|错误" /tmp/build_direct.log; exit 4; }
cp ds4 ds4.direct
# ② 图开版(全部二进制)
make cuda-spark > /tmp/build_graph.log 2>&1; rc=$?
[ $rc -eq 0 ] || { LOG "★图开版编译失败 rc=$rc★"; grep -m8 -iE "error|错误" /tmp/build_graph.log; exit 5; }
[ -n "$TAG" ] && cp ds4 "ds4.$TAG"
LOG "ok: ds4/ds4-bench(图开) ds4.direct(图关)${TAG:+ ds4.$TAG}"
ls -la --time-style=+%H:%M ds4 ds4-bench ds4.direct
