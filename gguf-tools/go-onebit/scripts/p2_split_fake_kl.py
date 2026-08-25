#!/usr/bin/env python3
"""p2_split_fake_kl.py — 把在线 KL 拆成"教师盲区(假)"与"学生损伤(真)"两份(2026-08-20)。
教师盲区判据: lnp_stu(true) - lnp_ref(true) > 2 nats(学生比FP教师更懂真值=教师瞎);
学生损伤判据: 反向 > 2 nats。其余=中性背景。
用法: p2_split_fake_kl.py <ref_anchor.bin> <ids.txt> <student.bin> [...]
"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from anchor_metrics import read_anchor_logits, read_student_logits, log_softmax

ref, meta = read_anchor_logits(sys.argv[1])
ids = np.array([int(t) for t in open(sys.argv[2]).read().split()])
S = meta['S']
lref = log_softmax(ref)
pref = np.exp(lref)
for sp in sys.argv[3:]:
    lstu = log_softmax(read_student_logits(sp))
    kl = np.empty(S)
    for lo in range(0, S, 256):
        hi = min(lo + 256, S)
        kl[lo:hi] = (pref[lo:hi] * (lref[lo:hi] - lstu[lo:hi])).sum(axis=1)
    n = S - 1
    tid = ids[1:S]
    adv = lstu[np.arange(n), tid] - lref[np.arange(n), tid]  # 学生-教师 真值优势
    blind = adv > 2.0          # 教师盲区
    hurt = adv < -2.0          # 学生损伤
    mid = ~blind & ~hurt
    k = kl[:n]
    print(f"===== {sp}")
    print(f"  教师盲区: {blind.sum()}位置({blind.mean()*100:.1f}%) 贡献KL {k[blind].sum()/k.sum()*100:.1f}%  该区KL均值 {k[blind].mean():.2f}")
    print(f"  学生损伤: {hurt.sum()}位置({hurt.mean()*100:.1f}%) 贡献KL {k[hurt].sum()/k.sum()*100:.1f}%  该区KL均值 {k[hurt].mean():.2f}")
    print(f"  中性背景: {mid.sum()}位置 KL均值 {k[mid].mean():.4f} 中位 {np.median(k[mid]):.4f}")
    print(f"  全段KL {k.mean():.4f} → 剔除教师盲区后 {k[~blind].mean():.4f}")
