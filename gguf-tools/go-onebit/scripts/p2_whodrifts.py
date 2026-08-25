#!/usr/bin/env python3
"""p2_whodrifts.py — 在 KL 爆炸位置上判定"锚 vs 学生谁偏离真值"(2026-08-20 第2针)。
teacher-forced 真值 = ids[i+1]。若学生 p(true) 好而锚 p(true) 差 → 锚(FP前向/协议)有病;
反之 → 学生真差; 双方都糊 → 真困难位置。
用法: p2_whodrifts.py <ref_anchor.bin> <ids.txt> <student.bin>
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

top = np.argsort(kl)[::-1][:15]
print("pos    KL    true_id  |  ref_top1(p)         stu_top1(p)        | lnp_ref(true) lnp_stu(true)")
for i in top:
    i = int(i)
    if i + 1 >= len(ids): continue
    t = ids[i + 1]
    rt, st = int(np.argmax(lref[i])), int(np.argmax(lstu[i]))
    print(f"{i:5d} {kl[i]:6.1f}  {t:7d} | {rt:7d}({np.exp(lref[i,rt]):.3f})"
          f"{'✓' if rt==t else ' '}  {st:7d}({np.exp(lstu[i,st]):.3f})"
          f"{'✓' if st==t else ' '} | {lref[i,t]:9.3f}  {lstu[i,t]:9.3f}")

# 聚合: top5% KL 位置上的 teacher-forced NLL, 锚 vs 学生
n5 = max(1, int(S * 0.05))
hot = np.argsort(kl)[::-1][:n5]
hot = hot[hot + 1 < len(ids)]
tid = ids[hot + 1]
nll_ref = -lref[hot, tid].mean()
nll_stu = -lstu[hot, tid].mean()
cold = np.argsort(kl)[:n5]
cold = cold[cold + 1 < len(ids)]
cid = ids[cold + 1]
print(f"\ntop5%爆炸位置  NLL: 锚={nll_ref:.3f}  学生={nll_stu:.3f}   (谁小谁贴真值)")
print(f"bottom5%平静位置 NLL: 锚={-lref[cold,cid].mean():.3f}  学生={-lstu[cold,cid].mean():.3f}")
print(f"全段 NLL: 锚={-lref[np.arange(S-1),ids[1:S]].mean():.3f}  学生={-lstu[np.arange(S-1),ids[1:S]].mean():.3f}")
