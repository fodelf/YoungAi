#!/usr/bin/env python3
"""probe_subspace_share.py — 行为误差主子空间的【跨专家共享度】实测(2026-08-08)。

判什么: 同层不同专家的 ΔY 主方向是否张在同一个低维子空间。
  共享 ⇒ 一层一份小 z 即可覆盖全部 256 专家(杠杆 200×+, 用户理论完整成立);
  不共享 ⇒ z 必须每专家一份(杠杆掉到 1.2-1.6×)。

度量:
  ① 交叉挽回率: 用专家 A 的前 k 主方向去投影专家 B 的 ΔY, 看能量挽回多少
     (对照 B 自己的前 k 挽回率 —— 比值 = 共享度)
  ② 联合基: 把 N 个专家的 ΔY 拼起来求前 k 主方向, 看对每个专家的平均挽回
     (这才是"一层一份 z"的真实工况)
用法: probe_subspace_share.py <hf> <layers_dir> <anchor> <层> [专家数=8] [tok=512]
"""
import json, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_behavior_spectrum import st_index, st_mxfp4, vq_slot, vq_dequant, anchor_fin

def main():
    hf, ld, ap, L = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    nexp = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    ntok = int(sys.argv[6]) if len(sys.argv) > 6 else 512
    wmap = st_index(hf)
    X, DIM = anchor_fin(ap, L, ntok)
    blob = open(os.path.join(ld, f"dql_vq_L{L:02d}.bin"), "rb").read()

    dYs, ids = [], []
    step = max(1, 256 // nexp)
    for e in range(0, 256, step):
        off = vq_slot(blob, e, 0)          # w1
        tn = f"layers.{L}.ffn.experts.{e}.w1.weight"
        if not off or tn not in wmap: continue
        W = st_mxfp4(hf, wmap, tn); Wq = vq_dequant(blob, off)
        if W.shape != Wq.shape: W = W.T
        dYs.append(X @ (W - Wq).T); ids.append(e)
        if len(dYs) >= nexp: break
    print(f"L{L}: 取 {len(dYs)} 个专家(w1), ΔY 各 [{ntok}×{dYs[0].shape[1]}]\n")

    KS = (8, 16, 32, 64, 128)
    # 每专家自有基(上界)
    bases, self_rec = [], []
    for dY in dYs:
        U, S, Vt = np.linalg.svd(dY.astype(np.float64), full_matrices=False)
        bases.append(Vt)                      # 行空间主方向 [r, 2048]
        e2 = S**2; self_rec.append(np.cumsum(e2)/e2.sum())

    def rec_by(basis_Vt, dY, k):
        """用给定的前 k 个方向投影 dY, 返回挽回能量比"""
        P = basis_Vt[:k]                      # [k, 2048]
        proj = dY @ P.T @ P
        return float((proj**2).sum() / (dY**2).sum())

    print("① 交叉挽回(专家 A 的基 → 专家 B 的误差) vs B 自己的基")
    print(f"{'k':>5} {'自有基(上界)':>13} {'邻居基':>10} {'共享度':>8}")
    for k in KS:
        own = np.mean([sr[k-1] for sr in self_rec])
        cross = np.mean([rec_by(bases[i], dYs[j], k)
                         for i in range(len(dYs)) for j in range(len(dYs)) if i != j])
        print(f"{k:>5} {own*100:>12.1f}% {cross*100:>9.1f}% {cross/own*100:>7.1f}%")

    print("\n② 联合基(全部专家 ΔY 拼接求主方向 = 一层一份 z 的真实工况)")
    cat = np.vstack([d.astype(np.float64) for d in dYs])
    Uc, Sc, Vtc = np.linalg.svd(cat, full_matrices=False)
    print(f"{'k':>5} {'联合基挽回':>11} {'自有基上界':>11} {'效率':>8}   z体积/层(三矩阵fp16)")
    for k in KS:
        joint = np.mean([rec_by(Vtc, d, k) for d in dYs])
        own = np.mean([sr[k-1] for sr in self_rec])
        vol = 3*k*(DIM + dYs[0].shape[1])*2/2**20
        print(f"{k:>5} {joint*100:>10.1f}% {own*100:>10.1f}% {joint/own*100:>7.1f}%   {vol:>6.2f} MiB")

if __name__ == "__main__":
    main()
