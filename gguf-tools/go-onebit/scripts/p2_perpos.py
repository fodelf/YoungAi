#!/usr/bin/env python3
"""p2_perpos.py — 第2针逐位置 KL 分布分析(2026-08-20)。
定位在线 KLD 膨胀的位置形态: 头部集中=BOS/协议错位; 均匀=量化底噪;
尾段集中=长上下文/indexer; mod-128 周期(score 分块128)=score 分块缝 bug。
用法: p2_perpos.py <ref_anchor.bin> <ids.txt> <student.bin> [student2 ...]
"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from anchor_metrics import read_anchor_logits, read_student_logits, log_softmax

ref_path, ids_path = sys.argv[1], sys.argv[2]
ref, meta = read_anchor_logits(ref_path)
S = meta['S']
lref = log_softmax(ref)
pref = np.exp(lref)
for stu_path in sys.argv[3:]:
    stu = read_student_logits(stu_path)
    kl = np.empty(S, dtype=np.float64)
    for lo in range(0, S, 256):
        hi = min(lo + 256, S)
        lstu = log_softmax(stu[lo:hi])
        kl[lo:hi] = (pref[lo:hi] * (lref[lo:hi] - lstu)).sum(axis=1)
    print(f"===== {stu_path}  mean={kl.mean():.4f} median={np.median(kl):.4f} p95={np.percentile(kl,95):.3f}")
    print("  位置分桶:", "  ".join(
        f"[{a}:{b})={kl[a:b].mean():.3f}" for a, b in
        [(0, 16), (16, 128), (128, 512), (512, 1024), (1024, 2048), (2048, S)]))
    m = np.array([kl[np.arange(S) % 128 == r].mean() for r in range(128)])
    print(f"  mod128: r0-3={m[:4].round(3).tolist()} 中段r8-119均值={m[8:120].mean():.3f}"
          f" r124-127={m[124:].round(3).tolist()}")
    top = np.argsort(kl)[::-1][:15]
    print(f"  top15爆炸位置: {[(int(i), round(float(kl[i]), 2)) for i in top]}")
    n5 = max(1, int(S * 0.05))
    print(f"  top5%位置承担KL份额: {np.sort(kl)[::-1][:n5].sum() / kl.sum() * 100:.1f}%")
