#!/usr/bin/env bash
# sampling_loop_probe.sh — 解码采样的打转探针(2026-09-21, 113-1.md §4): 同一条提示、同一个模型 + 反修,
# 温 0 一趟(必须与改采样之前的输出逐字节同 = 采样接线没碰裸路)+ 温 T 若干个 seed 各一趟, 全部走
# d1_kv_ring_gate.sh 的 loop 模式(打转尺: "约？"次数 / 4-gram 重复 / 分段 / 解码 t/s), 这里只串起来, 不另造尺。
#
# 为什么要多个 seed: 温 0 是单轨迹(一次 argmax 翻面定整篇), 采样更是; 一条提示三个 seed 才看得出
# "偶尔打转"还是"总打转"。判读一律看原文($OUT/loop_*.out), 数字只是索引。
#
# 用法: ./speed-bench/sampling_loop_probe.sh <模型.gguf> <反修目录|none> [温度, 默认 0.6] [seed 列表, 默认 "1 2 3"] [提示文件] [温 0 参照 .out|none] [附加采样参数]
#   温 0 参照 .out = 改采样之前同提示同模型跑出的 loop_*.out; 给了就 cmp, 不同 = 采样接线碰了裸路, 直接停; none = 跳过温 0 那趟。
#   附加采样参数 = 原样传给 ds4 的其余采样开关, 如 "--top-p 0.95 --min-p 0"(DeepSeek 对推理模型的官方口径是温 0.6 + top-p 0.95;
#   本引擎默认 top_p 1 + min_p 0.05 是相对概率过滤, 在复读区(冠军 token 90% 以上)会把所有备选都滤掉 = 事实上退回贪心, 出不来)。
# 出错会怎样: 任一趟 ds4 非 0 退出就停在那一趟(看 $OUT/<tag>.err); 温 0 cmp 不同退 3。
set -u
cd "$(dirname "$0")/.." || exit 1
MODEL="${1:?模型}"; AMP="${2:?反修目录或 none}"; TEMP="${3:-0.6}"; SEEDS="${4:-1 2 3}"
P="${5:-speed-bench/fin_chat_prompt.txt}"; REF="${6:-}"; XS="${7:-}"
OUT=/tmp/d1-kv-ring
NGEN=1400
base="loop_$(basename "$MODEL" .gguf)_$(basename "$AMP")_$(basename "$P" .txt)"

if [ "$REF" != none ]; then
  echo "== 温 0(裸 argmax 回归)"
  ./speed-bench/d1_kv_ring_gate.sh none "$MODEL" "$NGEN" loop "$AMP" "$P" "--temp 0" || exit 2
  if [ -n "$REF" ]; then
    if cmp -s "$REF" "$OUT/${base}_xtemp0.out"; then echo "  温 0 输出与参照逐字节同 ✓ ($REF)"
    else echo "  ★温 0 输出与参照不同★ ($REF vs $OUT/${base}_xtemp0.out) —— 采样接线碰了裸路, 停"; exit 3; fi
  fi
fi
for s in $SEEDS; do
  echo "== 温 $TEMP ${XS:+$XS }seed $s"
  ./speed-bench/d1_kv_ring_gate.sh none "$MODEL" "$NGEN" loop "$AMP" "$P" "--temp $TEMP ${XS:+$XS }--seed $s" || exit 2
done
echo "== 汇总(解码 t/s 一列看采样每步多付了多少; 分段一行看从哪段起死循环)"
for f in "$OUT/${base}_xtemp0.err" "$OUT"/${base}_xtemp*seed*.err; do
  [ -f "$f" ] || continue
  printf '  %s: %s\n' "$(basename "$f" .err | sed "s/^${base}_x//")" "$(grep -a -h 'decode .* token' "$f" | tail -1)"
done
