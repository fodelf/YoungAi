#!/usr/bin/env python3
"""p2_copysrc.py — 验证"锚盲区"假说(2026-08-20 第2针): KL爆炸位置是否=远距离可拷贝重复。
对每个位置 i, 找最长后缀 ids[i-m+1..i] 在历史中出现且后继=ids[i+1] 的拷贝源, 记录
匹配长度与回看距离。若爆炸位置的拷贝源距离系统性大(超出锚 FP 前向的注意力跨度),
而平静位置无此形态 → 锚 dchunk 盲区实锤。
用法: p2_copysrc.py <ref_anchor.bin> <ids.txt> <student.bin>
"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from anchor_metrics import read_anchor_logits, read_student_logits, log_softmax

ref, meta = read_anchor_logits(sys.argv[1])
ids = np.array([int(t) for t in open(sys.argv[2]).read().split()], dtype=np.int64)
stu = read_student_logits(sys.argv[3])
S = meta['S']
lref = log_softmax(ref); lstu = log_softmax(stu)
pref = np.exp(lref)
kl = np.empty(S)
for lo in range(0, S, 256):
    hi = min(lo + 256, S)
    kl[lo:hi] = (pref[lo:hi] * (lref[lo:hi] - lstu[lo:hi])).sum(axis=1)

def copy_source(i):
    """最长匹配拷贝源: 返回(匹配长度, 回看距离src_end→i)"""
    best = (0, -1)
    for j in range(i):          # 候选源结束位 j: ids[j+1]==ids[i+1] 且后缀匹配
        if ids[j + 1] != ids[i + 1]:
            continue
        m = 0
        while m < j + 1 and m < i + 1 and ids[j - m] == ids[i - m]:
            m += 1
        if m > best[0]:
            best = (m, i - j)
    return best

hotN = 50
hot = [int(x) for x in np.argsort(kl)[::-1][:hotN] if x + 1 < len(ids)]
rng = np.random.default_rng(7)
cold = [int(x) for x in rng.choice(S - 1, size=hotN, replace=False)]

for name, group in [("爆炸top50", hot), ("随机对照50", cold)]:
    mlen, dist = [], []
    for i in group:
        m, d = copy_source(i)
        mlen.append(m); dist.append(d)
    mlen = np.array(mlen); dist = np.array(dist, dtype=float)
    has = mlen >= 4
    print(f"== {name}: 有≥4-gram拷贝源比例 {has.mean()*100:.0f}%  "
          f"匹配长度中位 {np.median(mlen):.0f}  "
          f"拷贝距离中位 {np.median(dist[has]) if has.any() else float('nan'):.0f}")
print()
print("爆炸位置明细: pos, KL, 匹配长度, 拷贝距离, ln p_ref(true), ln p_stu(true)")
for i in hot[:15]:
    m, d = copy_source(i)
    t = ids[i + 1]
    print(f"  {i:5d} {kl[i]:6.1f}  m={m:3d} d={d:5d}  {lref[i, t]:8.3f} {lstu[i, t]:8.3f}")
