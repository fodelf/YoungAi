#!/usr/bin/env python3
"""probe_layer_z_validate.py — 每层动态 z 的两道终验(2026-08-08)。

前序: 整层行为误差 ΔH 极低维(k=1 抓 49.3%, k=32 抓 70.4%), 但那是训练集内。
本工具把它坐实(或证伪):
  ① held-out 基泛化: 前半 token 学基 B, 测后半(未见 token)的挽回
  ② 动态可预测性  : z 必须是 x 的函数 —— 系数 C=ΔH·Bᵀ 能否由 x 线性预测(C ≈ x·P)?
     这才是"动态 z"能落地的前提(否则只是事后拟合, 推理时算不出来)。
     报 held-out R²(未见 token 上的解释率)与最终"端到端挽回"(x·P·B vs ΔH)。
  ③ 跨层复用性  : 同一个 z 结构在不同层各自训练, 报各层挽回(体积按每层一份计)
用法: probe_layer_z_validate.py <hf> <layers_dir> <anchor> <层...> [--tok=512]
"""
import os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_behavior_spectrum import st_index, st_mxfp4, vq_slot, vq_dequant
from probe_layer_behavior import anchor_layer, swiglu

D = 4096

def build_dH(hf, wmap, ld, ap, L, ntok):
    X, ridx, rw, NACT = anchor_layer(ap, L, ntok)
    blob = open(os.path.join(ld, f"dql_vq_L{L:02d}.bin"), "rb").read()
    need = sorted(set(int(e) for e in ridx.reshape(-1)))
    Hfp = np.zeros((ntok, D)); Hq = np.zeros((ntok, D))
    for i, e in enumerate(need):
        rows, slots = np.where(ridx == e)
        if len(rows) == 0: continue
        xs = X[rows].astype(np.float32); w = rw[rows, slots].astype(np.float64)[:, None]
        P = {}
        for nm, which in (("w1",0),("w3",1),("w2",2)):
            Wf = st_mxfp4(hf, wmap, f"layers.{L}.ffn.experts.{e}.{nm}.weight")
            off = vq_slot(blob, e, which); Wq = vq_dequant(blob, off) if off else None
            if Wq is not None and Wf.shape != Wq.shape: Wf = Wf.T
            P[nm] = (Wf, Wq if Wq is not None else Wf)
        Hfp[rows] += w * (swiglu(xs @ P["w1"][0].T, xs @ P["w3"][0].T) @ P["w2"][0].T)
        Hq[rows]  += w * (swiglu(xs @ P["w1"][1].T, xs @ P["w3"][1].T) @ P["w2"][1].T)
        if (i+1) % 50 == 0: print(f"    …{i+1}/{len(need)} 专家", file=sys.stderr)
    return X.astype(np.float64), (Hfp - Hq), Hfp

def main():
    hf, ld, ap = sys.argv[1], sys.argv[2], sys.argv[3]
    layers = [int(a) for a in sys.argv[4:] if not a.startswith("--")]
    ntok = 512
    for a in sys.argv:
        if a.startswith("--tok="): ntok = int(a.split("=")[1])
    wmap = st_index(hf)
    KS = (4, 8, 16, 32, 64, 128)
    for L in layers:
        print(f"\n{'='*72}\nL{L}  (tok={ntok}, 前半训练 / 后半 held-out)")
        X, dH, Hfp = build_dH(hf, wmap, ld, ap, L, ntok)
        n = ntok // 2
        Xtr, Xho = X[:n], X[n:]
        Dtr, Dho = dH[:n], dH[n:]
        rel = np.linalg.norm(dH)/np.linalg.norm(Hfp)
        print(f"  ΔH 相对误差 {rel:.4f}")
        # 训练集基
        Vt = np.linalg.svd(Dtr, full_matrices=False)[2]
        print(f"  {'k':>5} {'训练内':>8} {'held-out':>9} {'动态R²(ho)':>11} {'端到端挽回(ho)':>14} {'z体积/层':>10}")
        for k in KS:
            B = Vt[:k]                                   # [k, 4096] 输出基(共享)
            tr = float(((Dtr @ B.T @ B)**2).sum()/(Dtr**2).sum())
            ho = float(((Dho @ B.T @ B)**2).sum()/(Dho**2).sum())
            # 动态: 用训练集学 P: minimize ||Xtr·P − Ctr||, Ctr = Dtr·Bᵀ
            Ctr = Dtr @ B.T                              # [n,k]
            Pmat, *_ = np.linalg.lstsq(Xtr, Ctr, rcond=None)   # [4096,k]
            Cho_hat = Xho @ Pmat
            Cho = Dho @ B.T
            ss = ((Cho - Cho_hat)**2).sum(); tt = (Cho**2).sum()
            r2 = 1 - ss/tt
            # 端到端: 用 x 预测的系数重建修正量, 看真实挽回
            rec = Cho_hat @ B
            e2e = 1 - ((Dho - rec)**2).sum()/(Dho**2).sum()
            vol = k*(D+D)*2/2**20
            print(f"  {k:>5} {tr*100:>7.1f}% {ho*100:>8.1f}% {r2*100:>10.1f}% {e2e*100:>13.1f}% {vol:>9.2f}M")

if __name__ == "__main__":
    main()
