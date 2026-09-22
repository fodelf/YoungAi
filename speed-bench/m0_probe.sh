#!/usr/bin/env bash
# m0_probe.sh — mtp-2.md M0 的探针驱动: 在**同一组提示**上把引擎自带的三个探针一次跑齐。
#
# 为什么单独一个文件而不是每次手敲 d0a: 这三个数是"下一刀该不该动、怎么动"的依据, 每落一刀就要
# 同口径复量一次。参数(提示、截多少字符、生成几个 token、用哪个 GGUF)写死在这里, 两次跑才可比 ——
# 手敲的话迟早一次截 8000 一次截 50000, 然后拿两个数去比, 得出的结论全是假的。
# 它本身不判速度(prof 开着会逐层 flush, 慢 15%), 判速度用 d1_kv_ring_gate.sh。
#
# 三个探针(都只在 --v41-prof 下出数, 实现分别在 cuda_vq_decode.inc.cu 与 core_v41_forward.c):
#   [f16-range] 激活(x)与中间量(h)有没有超出 f16 的指数范围 —— 定专家核张量核版走 f16 还是 TF32(mtp-2 §5.3)
#               判据: **溢出必须是 0**(f16 溢出 → inf → 0×inf = NaN, 静默毁一整行), 次正规占比要可忽略
#   [moe-uniq]  主干验证批的唯一专家数 / n×6 —— 验证多一位要多付多少专家字节(mtp-2 §2.2 的 u(n))
#   [mtp-uniq]  草稿塔的唯一专家数 / n×3 —— 塔专家并集那一刀值不值(mtp-2 §6.1)
#
# 用法: ./speed-bench/m0_probe.sh [模型] [生成几个 token, 默认 24]
# 出错会怎样: 表里一行 [mtp-uniq] 都没有 = 投机没武装(GGUF 没带三塔, 或者调度器一直判亏本歇着) ——
# 那时 f16 的"中间量(h)"一栏也只有纯解码那一档的量, 别拿它去判验证批。
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="${1:-gguf/v41/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf}"
NGEN="${2:-24}"
OUT=/tmp/m0-probe
mkdir -p "$OUT"
[ -f "$MODEL" ] || { echo "★没有模型 $MODEL★"; exit 1; }

# 三条提示与判决尺同源: 2K/12k 英文来自 d1 门用的那份 README×80, 难文本用意大利语长篇
# (它在这个底座上接受率最低 —— 09-16 实测 0.78/3, 而 README 是 1.39/5)。
head -c  8000 speed-bench/readme_en_x80.txt  > "$OUT/p2k.txt"  || exit 1
head -c 50000 speed-bench/readme_en_x80.txt  > "$OUT/p12k.txt" || exit 1
head -c  8000 speed-bench/promessi_sposi.txt > "$OUT/phard.txt" || exit 1

for p in p2k p12k phard; do
  echo "=========== $p ($(wc -c < "$OUT/$p.txt") 字符)"
  # ★等上一趟的进程真退干净★(2026-09-17 实撞): 引擎有意留着单实例锁(memory: instance lock is
  # intentional), 而 110 GB 的映射卸载要几秒 —— 上一趟 shell 已经返回, 内核还在收页。
  # 不等就是第二、三趟直接 "another ds4 process is already running; refusing to start",
  # 而脚本照样往下走、日志里只少几行, 一眼看不出这一趟根本没跑。
  for _ in $(seq 1 60); do pgrep -f "[d]s4 -m gguf" >/dev/null || break; sleep 2; done
  ./speed-bench/d0a_decode_profile.sh "$NGEN" specno "$MODEL" "$OUT/$p.txt" 100000 \
      > "$OUT/$p.log" 2>&1
  # d0a 的 OUT 是 /tmp/d0a-decode-<bin>, 每趟会被下一趟覆盖 —— 把这一趟的 prof.err 留一份
  cp -f /tmp/d0a-decode-ds4/prof.err "$OUT/$p.err" 2>/dev/null
  sed -n '/\[f16-range\]/,$p' "$OUT/$p.log" | head -24
  grep -a -h "DSpark:\|一轮 \|★中位" "$OUT/$p.log" | sed 's/^/  /'
done
echo
echo "== 判据: f16 溢出 == 0 且次正规可忽略 ⇒ 专家核张量核版走 f16; 否则走 TF32(mtp-2 §5.7)"
