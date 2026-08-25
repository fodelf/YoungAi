#!/usr/bin/env python3
"""zlever/kl_forensic.py — 兑现链 S4 取证: 逐位置 KL 分解(诊断, 2026-08-24 夜)。

对已落盘的判决 logits 做逐位置 KL(ref‖stu), 输出 ΔKL=stu_B−stu_A 的结构:
按参考 NLL 分桶(易/难 token)、最差位置贡献占比、top 罪犯清单。
回答: 放大器伤害/收益集中在哪类位置 —— 引擎链均匀失真 vs 尾部选择性伤害, 一测便知。

用法: kl_forensic.py <anchor_wt2.bin> <ids.txt> <stuA.bin> <stuB.bin> [标签A 标签B]
"""
import sys, os
import numpy as np

anc_p, ids_p, pa, pb = sys.argv[1:5]
la = sys.argv[5] if len(sys.argv) > 5 else "A"
lb = sys.argv[6] if len(sys.argv) > 6 else "B"
V = 129280
ids = np.array([int(x) for x in open(ids_p).read().split()], dtype=np.int64)
S = len(ids)
anc_sz = os.path.getsize(anc_p)
ref = np.memmap(anc_p, dtype=np.float32, mode="r", offset=anc_sz - S*V*4, shape=(S, V))

def stu_map(p):
    off = os.path.getsize(p) - S*V*4   # dump 可能带小头(8B), 从尾对齐
    assert off in (0, 8), f"{p} 尺寸异常 off={off}"
    return np.memmap(p, dtype=np.float32, mode="r", offset=off, shape=(S, V))

A = stu_map(pa); B = stu_map(pb)

def logsm(M):
    M = M.astype(np.float64)
    M -= M.max(1, keepdims=True)
    return M - np.log(np.exp(M).sum(1, keepdims=True))

n = S - 1   # teacher-forced: 位置 i 预测 ids[i+1]
klA = np.empty(n); klB = np.empty(n); nllR = np.empty(n)
CH = 128
for i0 in range(0, n, CH):
    i1 = min(i0 + CH, n)
    lr = logsm(ref[i0:i1]); pr = np.exp(lr)
    laa = logsm(A[i0:i1]); lbb = logsm(B[i0:i1])
    klA[i0:i1] = (pr * (lr - laa)).sum(1)
    klB[i0:i1] = (pr * (lr - lbb)).sum(1)
    tgt = ids[i0+1:i1+1]
    nllR[i0:i1] = -lr[np.arange(i1-i0), tgt]

d = klB - klA
print(f"n={n}  KL[{la}]={klA.mean():.5f}  KL[{lb}]={klB.mean():.5f}  ΔKL(mean)={d.mean():+.5f}  Δ(中位)={np.median(d):+.5f}")
# 按参考 NLL 分桶(token 对 FP 模型的难度)
edges = [0, 0.1, 0.5, 1.5, 3.0, 6.0, 99]
print(f"{'ref-NLL桶':>12} {'n':>5} {la+'-KL':>9} {lb+'-KL':>9} {'ΔKL均值':>9} {'Δ总量占比':>9}")
tot = d.sum()
for j in range(len(edges)-1):
    m = (nllR >= edges[j]) & (nllR < edges[j+1])
    if m.sum() == 0: continue
    print(f"{edges[j]:>5}-{edges[j+1]:<6} {m.sum():>5} {klA[m].mean():>9.4f} {klB[m].mean():>9.4f} {d[m].mean():>+9.4f} {d[m].sum()/tot*100 if tot!=0 else 0:>8.1f}%")
# 最差位置集中度
o = np.argsort(d)[::-1]
for frac in (0.01, 0.05):
    k = max(1, int(n*frac))
    print(f"最差{frac*100:.0f}%位置({k}个)承担 Δ 总量的 {d[o[:k]].sum()/tot*100 if tot!=0 else 0:.1f}%")
print("top10 罪犯位置(pos tgt refNLL KLa→KLb):")
for i in o[:10]:
    print(f"  pos={i} tgt={ids[i+1]} refNLL={nllR[i]:.2f} {klA[i]:.3f}→{klB[i]:.3f} (Δ{d[i]:+.3f})")
