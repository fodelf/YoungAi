#!/bin/bash
# v41_engine_parity_spark.sh — 引擎 V4.1 批前向 vs Python 学生(同一文件)逐位置 logits 对拍(2026-09-12 战役 P2 判决)。
#
# 【干什么】①引擎 ./ds4 --score-ids 出 <i32 S><i32 V><f32> ②anchor_metrics 以 Python 学生 logits 为"参考"
# 打 KL/Σmin/top1 —— 两边同一个量化文件, 差异只来自实现(GEMM 序/舍入点/索引选择), 目标 KL 1e-4~1e-3。
# 【内存账】模型 103 GiB mmap; 引擎按 VQ blob 强制 offload(骨架 ~4.5 GB 常驻注册), 专家 blob 每层注册→算→注销
# →DONTNEED, 稳态 < 15 GB; 看门狗 available < 8 GB 杀进程(free 口径, 不看 RSS)。
# 用法: v41_engine_parity_spark.sh <ids文件> <ntok> <python学生logits.bin> [out.bin] [--engram]
#   默认 --v41-no-engram(与 Python --no-engram 参考同口径); 给 --engram 则带 engram 跑(参考也得是带 engram 的)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
IDS="${1:?ids}"; NTOK="${2:?ntok}"; REF="${3:?python 学生 logits}"; OUT="${4:-$ROOT/gguf/v41judge/eng_$(basename "$IDS" .ids)_n$NTOK.bin}"
NOENG="--v41-no-engram"; [ "${5:-}" = "--engram" ] && NOENG=""
GG="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-fp4.gguf"
LOG(){ echo "[parity $(date '+%m-%d %H:%M:%S')] $*"; }
head -n "$NTOK" "$IDS" > "$OUT.ids"
watchdog() {
    while sleep 10; do
        local av; av=$(free -g | awk '/^内存|^Mem/{print $7}')
        echo "$(date '+%H:%M:%S') avail=${av}G" >> "$OUT.mem"
        if [ "${av:-99}" -lt 8 ]; then echo "★看门狗: available ${av}G < 8G, 停车★" | tee -a "$OUT.mem"; pkill -f "ds4 -m gguf/v41[/]"; return 1; fi
    done
}
: > "$OUT.mem"; watchdog & WD=$!; trap 'kill $WD 2>/dev/null' EXIT
LOG "引擎批前向 n=$NTOK → $OUT"
./ds4 -m "$GG" --cuda --mem-budget-mb 40000 --score-ids "$OUT.ids" --score-out "$OUT" $NOENG 2>&1 | tee "$OUT.log" | grep --line-buffered -E "v41|V4.1|offload|ds4:|PPL|失败|error|Error"
[ -s "$OUT" ] || { LOG "★引擎没出文件★"; exit 2; }
LOG "五指标: 参考=Python 学生 $REF, 学生=引擎 $OUT"
./gguf-tools/bench/anchor_metrics --ref-raw "$REF" --ids "$OUT.ids" --student "$OUT"
LOG "收工; 内存峰值: $(sort -t= -k2 -n "$OUT.mem" | head -1)"
