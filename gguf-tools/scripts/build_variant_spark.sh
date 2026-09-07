#!/bin/bash
# build_variant_spark.sh — spark 本机: 对一个源文件做一处 sed 改动, 编出 ds4.<标签>, 源码立刻还原。
#
# 为什么: 消融/隔离对拍经常要"只关掉一条路"编一份二进制(例如把分行注意力入口改成直接
# return -1, 让老核接管), 手工 sed+make+cp+还原四步容易漏还原, 留下被改坏的源码继续被
# 后面的 build 用。这里 trap 保证任何退出路径都还原; 只产出 ds4.<标签>, 不动 ds4/ds4-bench。
# 用法: build_variant_spark.sh <文件> <sed 表达式> <标签>
#   例: build_variant_spark.sh src/cuda/cuda_api_attention_4.inc.cu \
#         's/if (n_head > 64u || (n_head \& 7u) != 0u) return -1;/return -1;/' iso
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
F="${1:?源文件}"; EXPR="${2:?sed 表达式}"; TAG="${3:?标签}"
LOG(){ echo "[variant $(date +%H:%M:%S)] $*"; }
[ -f "$F" ] || { LOG "★文件缺 $F★"; exit 2; }
BUSY=$(for p in ds4 ds4-bench ds4-server ds4quant_run zlayer vq_merge_v4; do pgrep -x "$p"; done)
[ -z "$BUSY" ] || { LOG "★机器非空: $BUSY★"; exit 3; }
cp "$F" "/tmp/variant.$(basename "$F").bak"
trap 'cp "/tmp/variant.$(basename "$F").bak" "$F"' EXIT
sed -i "$EXPR" "$F"
cmp -s "$F" "/tmp/variant.$(basename "$F").bak" && { LOG "★sed 没改到任何东西: $EXPR★"; exit 4; }
make cuda-spark > "/tmp/build_variant_$TAG.log" 2>&1; rc=$?
[ $rc -eq 0 ] || { LOG "★变体编译失败 rc=$rc★"; grep -m8 -iE " error|错误" "/tmp/build_variant_$TAG.log"; exit 5; }
cp ds4 "ds4.$TAG"
LOG "ok: ds4.$TAG ($F: $EXPR); 源码已还原, ds4/ds4-bench 仍是变体, 下一次 build 覆盖"
