#!/bin/bash
# validate_build_spark.sh — spark 本机: 一次引擎改动的标准验收链(速度战役 09-05 定型)。
#
# 顺序(每步都是既有脚本): ①build_ab(图开 ds4/ds4-bench + 图关 ds4.direct + 备份 ds4.<标签>)
# ②短提示 图 vs 直发 逐字节对拍 ③长提示(4300 token, 稀疏 indexer 路)图 vs 直发 对拍
# ④旧二进制 vs 新 64 token 对拍(只看是否分叉+采样序列是否同) ⑤kernel_parity: 新两遍决定论 +
# 新/旧各对 FP 锚五指标 + 新 vs 旧 ⑥speed_champ 曲线(2048..8192) ⑦bench 路 4096 ctx nsys 剖面。
# 为什么固定成链: 每轮改核都要过同一套尺, 手拼命令链两次都拼错过(parity 参数路径错/
# ds4-bench 编成直发)。任一步失败链不停(后面的读数仍有用), 靠输出里的 ★ 行人工判。
# 用法: validate_build_spark.sh <标签> [旧二进制=ds4.gfix4] [ctxmax=8192]
# 产物: speed/{tokgraph_<标签>{s,l,old}_*, parity_<标签>.txt, champ86amp_<标签>_zchain.csv, prof_bench_<标签>_4096.*}
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
TAG="${1:?标签}"; OLD="${2:-./ds4.gfix4}"; CTXMAX="${3:-8192}"
S=gguf-tools/scripts; SP=gguf/go-onebit/vqhalf/champ86amp/speed
LOG(){ echo "[validate $(date +%H:%M:%S)] $*"; }
[ -x "$OLD" ] || { LOG "★旧二进制缺 $OLD★"; exit 2; }
[ -s "$SP/prompt_long15k.txt" ] || head -c 15500 speed-bench/promessi_sposi.txt > "$SP/prompt_long15k.txt"
LOG "== ① build $TAG"; bash $S/build_ab_spark.sh "$TAG" || { LOG "★build 失败, 链停★"; exit 3; }
LOG "== ② 短提示 图 vs 直发"; bash $S/tokgraph_ab_spark.sh ./ds4.direct ./ds4 64 "${TAG}s"
LOG "== ③ 长提示 图 vs 直发"; bash $S/tokgraph_ab_spark.sh ./ds4.direct ./ds4 64 "${TAG}l" "$SP/prompt_long15k.txt"
LOG "== ④ 旧 $OLD vs 新 64 token"; bash $S/tokgraph_ab_spark.sh "$OLD" ./ds4 64 "${TAG}old"
if cmp -s "$SP/tokgraph_${TAG}old_off.json" "$SP/tokgraph_${TAG}old_on.json"; then
    LOG "旧/新 logits 逐字节全同"
else
    a=$(grep -o '"token":[0-9]*' "$SP/tokgraph_${TAG}old_off.json" | md5sum | cut -c1-8)
    b=$(grep -o '"token":[0-9]*' "$SP/tokgraph_${TAG}old_on.json" | md5sum | cut -c1-8)
    LOG "旧/新 logits 有序差; 采样序列 $([ "$a" = "$b" ] && echo 全同 || echo ★不同★)"
fi
LOG "== ⑤ parity(FP 锚)"; bash $S/kernel_parity_spark.sh "$OLD" ./ds4 gguf/go-onebit/g7/wt2.ids gguf/go-onebit/r30/anchor_wt2_s2653.bin "$TAG"
LOG "== ⑥ speed"; bash $S/speed_champ_spark.sh champ86amp "champ86amp_$TAG" "$CTXMAX"
LOG "== ⑦ prof bench 4096"; bash $S/prof_bench_spark.sh champ86amp "bench_${TAG}_4096" 4096 64
LOG "VALIDATE_${TAG}_DONE"
