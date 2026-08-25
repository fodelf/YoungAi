#!/usr/bin/env python3
"""probe_z_structure.py — 动态 z 的最优结构实测(2026-08-08 用户令"验证更优的结构")。

已知(实测): 一层全共享基 → 挽回 39%(k=128); 每专家独享基 → 80% 但体积×256 杠杆<1。
本工具找中间最优, 并做 held-out(前面 39% 是训练集内, 数字可能虚高):

  ① 分组共享: G 个专家共用一个基, 扫 G=1/2/4/8 → 效率 vs 体积曲线
  ② held-out : 用 A 组专家学基, 测 B 组(完全未见)的挽回 = 真实泛化
  ③ 共享基+每专家对角: B 共享, 每专家只存 k 个标量 d_e(体积几乎不变)
用法: probe_z_structure.py <hf> <layers_dir> <anchor> <层> [专家数=16] [tok=512]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_behavior_spectrum import st_index, st_mxfp4, vq_slot, vq_dequant, anchor_fin

MOEI, D, NEXP, NL = 2048, 4096, 256, 43

def rec(basis, dY, k):
    P = basis[:k]
    return float(((dY @ P.T @ P) ** 2).sum() / (dY ** 2).sum())

def joint_basis(dYs):
    return np.linalg.svd(np.vstack([d.astype(np.float64) for d in dYs]), full_matrices=False)[2]

def main():
    hf, ld, ap, L = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    nexp = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    ntok = int(sys.argv[6]) if len(sys.argv) > 6 else 512
    wmap = st_index(hf); X, DIM = anchor_fin(ap, L, ntok)
    blob = open(os.path.join(ld, f"dql_vq_L{L:02d}.bin"), "rb").read()
    dYs, ids = [], []
    for e in range(0, 256, max(1, 256 // nexp)):
        off = vq_slot(blob, e, 0); tn = f"layers.{L}.ffn.experts.{e}.w1.weight"
        if not off or tn not in wmap: continue
        W = st_mxfp4(hf, wmap, tn); Wq = vq_dequant(blob, off)
        if W.shape != Wq.shape: W = W.T
        dYs.append((X @ (W - Wq).T).astype(np.float64)); ids.append(e)
        if len(dYs) >= nexp: break
    N = len(dYs)
    print(f"L{L}: {N} 个专家(w1), ΔY [{ntok}×{MOEI}]\n")
    KS = (16, 32, 64, 128)

    own = []
    for d in dYs:
        s = np.linalg.svd(d, compute_uv=False) ** 2
        own.append(np.cumsum(s) / s.sum())
    ownk = {k: float(np.mean([o[k-1] for o in own])) for k in KS}

    print("① 分组共享: G 个专家共用一个基(组内自评)")
    print(f"{'组大小':>7} " + "  ".join(f"k={k}" for k in KS) + "     每层 z 体积(三矩阵)")
    for G in (1, 2, 4, 8, N):
        if G > N: continue
        accs = {k: [] for k in KS}
        for g0 in range(0, N, G):
            grp = dYs[g0:g0+G]
            if not grp: continue
            B = joint_basis(grp)
            for k in KS:
                accs[k] += [rec(B, d, k) for d in grp]
        ngrp = (NEXP + G - 1) // G
        vol = {k: 3 * k * (D + MOEI) * 2 * ngrp / 2**20 for k in KS}
        print(f"{G:>7} " + "  ".join(f"{np.mean(accs[k])*100:5.1f}%" for k in KS) +
              f"    k=64: {vol[64]:.1f} MiB/层 ({ngrp} 组)")

    print(f"\n② held-out 泛化: 前 {N//2} 个专家学基 → 后 {N//2} 个(完全未见)")
    Btr = joint_basis(dYs[:N//2])
    print(f"{'k':>5} {'训练集内':>9} {'held-out':>10} {'自有基上界':>11}")
    for k in KS:
        tr = np.mean([rec(Btr, d, k) for d in dYs[:N//2]])
        ho = np.mean([rec(Btr, d, k) for d in dYs[N//2:]])
        print(f"{k:>5} {tr*100:>8.1f}% {ho*100:>9.1f}% {ownk[k]*100:>10.1f}%")

    print("\n③ 共享基 + 每专家对角 d_e(体积仅 +k 标量/专家)")
    B = joint_basis(dYs)
    print(f"{'k':>5} {'纯共享':>8} {'+对角':>8} {'增益':>7}")
    for k in KS:
        P = B[:k]
        plain, diag = [], []
        for d in dYs:
            C = d @ P.T                      # [ntok,k] 系数
            plain.append(float(((C @ P)**2).sum() / (d**2).sum()))
            # 对角 d_e: 逐模式最小二乘缩放(闭式) —— 对投影后残差再拟合一次幅度
            num = (C * C).sum(0); den = num + 1e-12
            de = num / den                   # 投影已是最优 ⇒ 对角=1, 增益来自跨模式再平衡
            diag.append(float((((C * de) @ P)**2).sum() / (d**2).sum()))
        print(f"{k:>5} {np.mean(plain)*100:>7.1f}% {np.mean(diag)*100:>7.1f}% {(np.mean(diag)-np.mean(plain))*100:>6.2f}pp")

if __name__ == "__main__":
    main()
