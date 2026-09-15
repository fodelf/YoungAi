#!/bin/bash
# v41_topk_selftest.sh — anchor_metrics --ref-topk 的合成夹具自检(2026-09-13)。
#
# 为什么要它: 这把尺(决策点 argmax 命中率)是后训练的主判据, 而它读的是一段自己约定的
# 二进制。约定读错不会报错, 只会给出一个安静的错数 —— 所以先用"答案已知"的假文件验一遍:
# 6 行, 人为安排 3 行 argmax 命中、3 行不命中, 其中 4 行与另一版不同(=决策点), 决策点里
# 2 行已站到目标一侧。期望输出 50.00% / 50.00%。
#
# 用法: v41_topk_selftest.sh   (不需要模型, 秒级)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BEN="$ROOT/gguf-tools/bench"
TMP="${TMPDIR:-/tmp}/v41_topk_selftest.$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

[ -x "$BEN/anchor_metrics" ] || make -C "$ROOT/gguf-tools" anchor_metrics || exit 1

python3 - "$TMP" <<'PY' || exit 1
import struct, sys, os
d = sys.argv[1]
K, S, V = 4, 6, 100
# 每行: (目标 token, top-K 的 (id,p) 列表)。前 3 行让目标排第一, 后 3 行让别人排第一。
rows = [
    (10, [(10, .6), (11, .2), (12, .1), (13, .1)]),
    (20, [(20, .5), (21, .3), (22, .1), (23, .1)]),
    (30, [(30, .7), (31, .1), (32, .1), (33, .1)]),
    (40, [(41, .6), (40, .2), (42, .1), (43, .1)]),
    (50, [(51, .5), (50, .3), (52, .1), (53, .1)]),
    (60, [(61, .4), (60, .3), (62, .2), (63, .1)]),
]
with open(os.path.join(d, "topk.bin"), "wb") as f:
    f.write(struct.pack("<4I", 0x44475445, K, S, V))
    for i, (tgt, top) in enumerate(rows):
        tp = dict(top).get(tgt, 0.0)
        mass = sum(p for _, p in top)
        f.write(struct.pack("<Iiff", i, tgt, tp, mass))
        f.write(struct.pack("<%di" % K, *[t for t, _ in top]))
        f.write(struct.pack("<%df" % K, *[p for _, p in top]))
# 另一版序列: 位置 i 的另一版 token = alt[i+1]。让第 0/1/3/4 行成为决策点(与目标不同),
# 第 2/5 行两版相同(不是决策点)。决策点里第 0、1 行 argmax 命中目标 ⇒ 2/4 = 50%。
alt = [0, 99, 98, 30, 97, 96, 60]
open(os.path.join(d, "alt.txt"), "w").write("\n".join(map(str, alt)) + "\n")
PY

OUT="$TMP/out.txt"
if ! "$BEN/anchor_metrics" --ref-topk "$TMP/topk.bin" --alt "$TMP/alt.txt" > "$OUT" 2>&1; then
    cat "$OUT"; echo "★anchor_metrics 失败★"; exit 1
fi
cat "$OUT"
grep -q "argmax 命中目标\[全段\]: 3/6 = 50.00%" "$OUT" || { echo "★命中率不是期望的 3/6★"; exit 1; }
grep -q "决策点(两版不同的位置) 4 个: argmax 已站到目标一侧 2 个 = 50.00%" "$OUT" || { echo "★决策点统计不对★"; exit 1; }

# ---- 自检 2 的读表器(--predict): 解算器预测 vs 真前向 ----
# 同一张假表, 造一份 predict.txt 把 m_pred 填成【真值】(ln p(a) − ln p(b) 手算),
# 期望相关 1.0000、一致 100%。验的是"这个读表器有没有按约定读": 行号/token 列错位
# 会让相关掉下来, 而不会报错。
python3 - "$TMP" <<'PY2' || exit 1
import math, os, sys
d = sys.argv[1]
# (行号, a, b, 真 m) —— p 值取自上面那张表
rows = [(0, 10, 11, math.log(.6 / .2)), (1, 20, 22, math.log(.5 / .1)),
        (3, 41, 40, math.log(.6 / .2)), (4, 50, 51, math.log(.3 / .5))]
with open(os.path.join(d, "predict.txt"), "w") as f:
    f.write("# 段 样本 行号 token_a token_b m0 dm m_pred\n")
    for r, a, b, m in rows:
        f.write("dec_fit 0 %d %d %d %.6f 0.000000 %.6f\n" % (r, a, b, m, m))
    f.write("ctr 1 2 30 31 0.000000 0.000000 0.000000\n")   # 别的样本: 必须被滤掉
PY2
OUT2="$TMP/out2.txt"
if ! "$BEN/anchor_metrics" --ref-topk "$TMP/topk.bin" --predict "$TMP/predict.txt" --predict-sample 0 > "$OUT2" 2>&1; then
    cat "$OUT2"; echo "★--predict 失败★"; exit 1
fi
cat "$OUT2"
grep -q "4 行, 相关 1.0000, 翻/不翻一致 100.00%" "$OUT2" || { echo "★自检2 读表器读出来的不是期望值(应 4 行/相关 1/一致 100%)★"; exit 1; }

# ---- 守门 1 的两态对比(--vs-topk) ----
# 造第二张表: 第 0、1 行把榜首换人(argmax 变了), 其余照抄。6 行里 2 行变 ⇒ 期望"没变 4/6 = 66.67%"。
python3 - "$TMP" <<'PY3' || exit 1
import struct, sys, os
d = sys.argv[1]
K, S, V = 4, 6, 100
rows = [
    (10, [(11, .6), (10, .2), (12, .1), (13, .1)]),   # 榜首 10 → 11, 变了
    (20, [(21, .5), (20, .3), (22, .1), (23, .1)]),   # 榜首 20 → 21, 变了
    (30, [(30, .7), (31, .1), (32, .1), (33, .1)]),
    (40, [(41, .6), (40, .2), (42, .1), (43, .1)]),
    (50, [(51, .5), (50, .3), (52, .1), (53, .1)]),
    (60, [(61, .4), (60, .3), (62, .2), (63, .1)]),
]
with open(os.path.join(d, "topk2.bin"), "wb") as f:
    f.write(struct.pack("<4I", 0x44475445, K, S, V))
    for i, (tgt, top) in enumerate(rows):
        tp = dict(top).get(tgt, 0.0)
        f.write(struct.pack("<Iiff", i, tgt, tp, sum(p for _, p in top)))
        f.write(struct.pack("<%di" % K, *[t for t, _ in top]))
        f.write(struct.pack("<%df" % K, *[p for _, p in top]))
PY3
OUT3="$TMP/out3.txt"
if ! "$BEN/anchor_metrics" --ref-topk "$TMP/topk.bin" --vs-topk "$TMP/topk2.bin" > "$OUT3" 2>&1; then
    cat "$OUT3"; echo "★--vs-topk 失败★"; exit 1
fi
cat "$OUT3"
grep -q "argmax 没变 4/6 = 66.67%" "$OUT3" || { echo "★两态对比读出来的不是期望的 4/6★"; exit 1; }
echo "v41_topk_selftest: PASS"
