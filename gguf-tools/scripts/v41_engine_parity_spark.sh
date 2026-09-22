#!/bin/bash
# v41_engine_parity_spark.sh — 引擎 V4.1 批前向 vs Python 学生(同一文件)逐位置 logits 对拍(2026-09-12 战役 P2 判决)。
#
# 【干什么】①引擎 ./ds4 --score-ids 出 <i32 S><i32 V><f32> ②anchor_metrics 以 Python 学生 logits 为"参考"
# 打 KL/Σmin/top1 —— 两边同一个量化文件, 差异只来自实现(GEMM 序/舍入点/索引选择), 目标 KL 1e-4~1e-3。
# 【内存账】模型 103 GiB mmap; 引擎按 VQ blob 强制 offload(骨架 ~4.5 GB 常驻注册), 专家 blob 每层注册→算→注销
# →DONTNEED, 稳态 < 15 GB; 看门狗 available < 8 GB 杀进程(free 口径, 不看 RSS)。
# 用法: v41_engine_parity_spark.sh <ids文件> <ntok> <python学生logits.bin> [out.bin] [--engram] [--gguf 文件]
#   默认 --v41-no-engram(与 Python --no-engram 参考同口径); 给 --engram 则带 engram 跑(参考也得是带 engram 的)。
#   --gguf: 换被测 GGUF(2026-09-20 加, 用来对拍 q4_K 骨架那份); 不给就是现役 vq8x4096-fp4。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
GG="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-fp4.gguf"
NOENG="--v41-no-engram"
POS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --engram) NOENG=""; shift ;;
        --gguf) GG="${2:?--gguf 后面要跟文件}"; shift 2 ;;
        *) POS+=("$1"); shift ;;
    esac
done
IDS="${POS[0]:?ids}"; NTOK="${POS[1]:?ntok}"; REF="${POS[2]:?python 学生 logits}"
OUT="${POS[3]:-$ROOT/gguf/v41judge/eng_$(basename "$IDS" .ids)_n$NTOK.bin}"
[ -f "$GG" ] || { echo "★被测 GGUF 不存在: $GG★"; exit 1; }
LOG(){ echo "[parity $(date '+%m-%d %H:%M:%S')] $*"; }
head -n "$NTOK" "$IDS" > "$OUT.ids"
watchdog() {
    while sleep 10; do
        local av; av=$(free -g | awk '/^内存|^Mem/{print $7}')
        echo "$(date '+%H:%M:%S') avail=${av}G" >> "$OUT.mem"
        # ★模式里带 .*★(2026-09-20 修): GG 是绝对路径, 旧模式 "ds4 -m gguf/v41[/]" 对 "ds4 -m /home/…/gguf/v41/…"
        # 永远不中 —— 这个看门狗以前一次都没真拦过。[/] 是防 pkill 匹配到自己。
        if [ "${av:-99}" -lt 8 ]; then echo "★看门狗: available ${av}G < 8G, 停车★" | tee -a "$OUT.mem"; pkill -f "ds4 -m .*gguf/v41[/]"; return 1; fi
    done
}
: > "$OUT.mem"; watchdog & WD=$!; trap 'kill $WD 2>/dev/null' EXIT
LOG "引擎批前向 n=$NTOK → $OUT  (被测 $(basename "$GG"), ${NOENG:-带 engram})"
./ds4 -m "$GG" --cuda --mem-budget-mb 40000 --score-ids "$OUT.ids" --score-out "$OUT" $NOENG 2>&1 | tee "$OUT.log" | grep --line-buffered -E "v41|V4.1|offload|ds4:|PPL|失败|error|Error"
[ -s "$OUT" ] || { LOG "★引擎没出文件★"; exit 2; }
LOG "五指标: 参考=Python 学生 $REF, 学生=引擎 $OUT"
./gguf-tools/bench/anchor_metrics --ref-raw "$REF" --ids "$OUT.ids" --student "$OUT"
LOG "收工; 内存峰值: $(sort -t= -k2 -n "$OUT.mem" | head -1)"
