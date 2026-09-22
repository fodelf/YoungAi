#!/bin/bash
# chain_speed_variant_spark.sh — 引擎速度改动的一条龙验收(spark 本机跑, nohup 发车)。
#
# 为什么要有它: 每次动解码路的 kernel/发射方式, 验收都是同一套四件事, 手敲一次错一次:
#   ① 编译并留一份带标签的二进制(ds4.<tag>)  ② decode parity: 新二进制 vs 基线二进制, 同 ids
#   同 FP 锚, 期望 new vs old KLD 0(纯发射/调度改动逐位不许变)  ③ 8192 快照曲线(speed_champ)
#   ④ 2048 ctx nsys 剖面 + 解码段逐核账(prof_decode_account.sh: 周期/忙/空隙/每核 ms)。
# 用法: chain_speed_variant_spark.sh <tag> <基线二进制 如 ./ds4.cr> [工作区=champ86q4k]
# 产物: gguf/go-onebit/vqhalf/<ws>/speed/{parity_<tag>.txt, <ws>_<tag>.csv, prof_<tag>_2048.*}
# 内存账: 同 speed_champ_spark.sh / prof_bench_spark.sh(--mem-budget-mb 110000); 全程单进程串行。
set -uo pipefail
TAG="${1:?tag}"; BASE="${2:?基线二进制}"; WS="${3:-champ86q4k}"
ROOT="$HOME/ds4-main"; S="$ROOT/gguf-tools/scripts"; VQH="$ROOT/gguf/go-onebit/vqhalf"
cd "$ROOT" || exit 1
LOG(){ echo "[chain-$TAG $(date +%H:%M:%S)] $*"; }
[ -x "$BASE" ] || { LOG "★基线二进制缺 $BASE★"; exit 2; }
LOG "== build $TAG"
make cuda-spark > "/tmp/build_$TAG.log" 2>&1 || { LOG "BUILD_FAIL"; grep -m8 -iE " error|ptxas error" "/tmp/build_$TAG.log"; exit 1; }
cp ds4 "ds4.$TAG"; cp ds4-bench "ds4-bench.$TAG"; LOG "build ok → ds4.$TAG"
LOG "== decode parity $BASE vs ./ds4"
bash "$S/kernel_parity_spark.sh" "$BASE" ./ds4 gguf/go-onebit/g7/wt2.ids gguf/go-onebit/r30/anchor_wt2_s2653.bin "$TAG" "$WS" decode
# --score-ids 是喂 token 的裸通道, 测不到贪心生成路(token 图重放/预发射): 再走一遍 --dump-logprobs
# 贪心 256 token 逐字节对拍(短提示 + 4300 token 长提示两态)。
LOG "== greedy A/B $BASE vs ./ds4 (短提示)"
bash "$S/tokgraph_ab_spark.sh" "$BASE" ./ds4 256 "${TAG}_s" "" "$WS" | grep -E "rc=|全同|分叉|★"
[ -s speed-bench/promessi_long.txt ] || head -c 24000 speed-bench/promessi_sposi.txt > speed-bench/promessi_long.txt   # ~6k token: 盖过 4096 原始窗 + 稀疏 indexer 路
LOG "== greedy A/B $BASE vs ./ds4 (长提示 ~6k)"
bash "$S/tokgraph_ab_spark.sh" "$BASE" ./ds4 64 "${TAG}_l" speed-bench/promessi_long.txt "$WS" | grep -E "rc=|全同|分叉|★"
LOG "== 8192 快照曲线"
bash "$S/speed_champ_spark.sh" "$WS" "${WS}_$TAG" 8192
LOG "== 2048 剖面"
bash "$S/prof_bench_spark.sh" "$WS" "${TAG}_2048" 2048 64 > "/tmp/prof_${TAG}_2048.out" 2>&1
grep -E "bench rc" "/tmp/prof_${TAG}_2048.out"
bash "$S/prof_decode_account.sh" "$VQH/$WS/speed/prof_${TAG}_2048.nsys-rep" 4 64 24
LOG "CHAIN_${TAG}_END"
