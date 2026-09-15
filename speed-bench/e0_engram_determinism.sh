#!/usr/bin/env bash
# e0_engram_determinism.sh — 段 0: 定位"温度 0 下引擎不确定"的病灶在哪一段(2026-09-15)
#
# 现象(09-15 第四轮挖出): 同一份二进制、同一份输入、温度 0, 四次跑出四个 PPL
# (15.43 / 15.59 / 15.65 / 15.58)。--v41-no-engram 时逐字同 ⇒ 病在 engram 路。
# 已排除: 流序(--v41-prof 逐层同步仍错)、页缓存(engram 表已 O_DIRECT 仍错)、
# 预填融合核(3 token 提示根本不走它)。
#
# 这个脚本干什么: 用同一个 3 token 提示跑 N 遍, 每遍收 `--v41-prof` 打出的 engram 三段指纹
#   rows = 行号(主机纯计算: 哈希 + st->hist)
#   raw  = 从 203 GB 表 pread 回来的原始字节
#   hc   = engram 门写完之后的 hc(GPU 侧出口)
# 然后逐遍比。怎么读结果:
#   rows 变        → 病在哈希/hist 路(主机纯计算变了, 说明有人在读写 hist 或常量表时被踩)
#   rows 同 raw 变 → 病在盘读路(pread/O_DIRECT/bounce)
#   rows raw 都同、hc 变 → 病在 GPU 路(dequant 核 / wkv 走 mmap 的脏读 / 门核)
#   三个全同但生成的字还是不同 → 病根本不在 engram 层, 前面的定位要推翻
#
# 出错会怎样: 看不到 "engram 指纹" 行 = 二进制不是带探针的那份(要先重编);
# 看到 "拿不到 O_DIRECT" = 文件系统不支持, 那一路的结论作废要单说。
#
# 用法: ./speed-bench/e0_engram_determinism.sh [跑几遍, 默认 8]

set -u
cd "$(dirname "$0")/.." || exit 1

RUNS="${1:-8}"
MODEL=gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-fp4.gguf
AMP=gguf/v41/gr-fin-40-fp4
OUT=/tmp/e0-engram
PROMPT="你好世界"          # 3 token: 短到不走任何预填融合核, 只过 embed + 40 层 + 出口

mkdir -p "$OUT"
# ★必须先清★: 下面用 run*.out 通配符比对, 上一轮(可能是另一个二进制)留下的文件会混进来,
# 结果就是"引擎不确定"的假警报 —— 09-15 实撞过一次, 白白二分了一轮。
rm -f "$OUT"/run*.out "$OUT"/run*.err "$OUT"/fp*.txt
[ -f "$MODEL" ] || { echo "★没有模型 $MODEL★"; exit 1; }
[ -x ./ds4 ]    || { echo "★没有 ./ds4, 先 make cuda-spark★"; exit 1; }

echo "== 段 0 engram 确定性探针: $RUNS 遍, 提示「$PROMPT」"
for i in $(seq 1 "$RUNS"); do
  ./ds4 -m "$MODEL" --zchain "$AMP" --v41-prof --temp 0 --seed 1 -n 24 \
        -p "$PROMPT" >"$OUT/run$i.out" 2>"$OUT/run$i.err"
  grep -a "engram 指纹" "$OUT/run$i.err" > "$OUT/fp$i.txt"
  printf "run %-2s 指纹行 %-3s 生成 md5 %s\n" "$i" \
         "$(wc -l < "$OUT/fp$i.txt")" "$(md5sum < "$OUT/run$i.out" | cut -c1-16)"
done

echo "== 逐遍与 run1 比"
for i in $(seq 2 "$RUNS"); do
  if diff -q "$OUT/fp1.txt" "$OUT/fp$i.txt" >/dev/null; then
    echo "run $i 指纹 == run1"
  else
    echo "run $i 指纹 ★不同★:"
    diff "$OUT/fp1.txt" "$OUT/fp$i.txt" | head -8
  fi
done

echo "== 非有限值扫描(--v41-prof 每层后扫 hc)"
grep -ah "非有限值" "$OUT"/run*.err | sort | uniq -c | head
echo "== 生成文本是否全同"
md5sum "$OUT"/run*.out | awk '{print $1}' | sort -u | wc -l | \
  xargs -I{} sh -c '[ {} = 1 ] && echo "★全同★" || echo "★{} 个不同的输出★"'
