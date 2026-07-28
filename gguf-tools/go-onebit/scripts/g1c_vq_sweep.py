#!/usr/bin/env python3
"""g1c_vq_sweep.py — 码本尺寸等体积扫描(held-out, 2026-07-25)。
冷: dim8 nc∈{256,512,1024} → 1.00/1.13/1.25 bpw(生产1.0625); 热: dim4 nc∈{256,512} → 2.00/2.25(生产2.125)。
附 GPTQ 式误差反馈 VQ(列组 Hessian 反馈, 分配仍最近邻码字)对照。
用法: g1c_vq_sweep.py --layer L --x X.npy [--ne 2]
"""
import os, sys, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "quant"))
from g1a_lever_probe import open_shard_for_layer, walsh, rot_groups
from g1b_stack_probe import hot_vq, metrics, cold_C0

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", file=sys.stderr, flush=True)


def _assign_chunk(V, C, bs=16384):
    """内存安全最近邻: 分块 [bs,nc] 距离, 峰值 ~bs*nc*4B。"""
    out = np.empty(len(V), np.int64)
    c2 = (C*C).sum(1)
    for i in range(0, len(V), bs):
        v = V[i:i+bs]
        d = v@C.T                                   # [bs,nc]
        d *= -2.0; d += c2[None,:]                  # ||v-c||² 去掉常数项 v²
        out[i:i+bs] = d.argmin(1)
    return out

def vq_gptq(W, Xf, dim=8, nc=256, grp=128):
    """VQ + 列组误差反馈: 逐 dim-块最近邻后, 残差经 Hinv 反馈到后续列(GPTQ 思想的 VQ 版)。"""
    import math
    rows, cols = W.shape
    rng = np.random.default_rng(0x5EED)
    V = W.reshape(-1, dim)
    C = V[rng.choice(len(V), nc, replace=False)].copy()
    sub = V[rng.choice(len(V), min(len(V), 200000), replace=False)]
    for _ in range(8):
        a2 = _assign_chunk(sub, C)
        for c in range(nc):
            m = a2==c
            if m.any(): C[c]=sub[m].mean(0)
    Wq = np.empty_like(W); Wk = W.copy()
    for j0 in range(0, cols, grp):
        g = min(grp, cols-j0)
        Xb = Xf[:, j0:j0+g]; H = Xb.T@Xb
        H[np.diag_indices(g)] += 0.02*(np.trace(H)/g+1e-9)
        try: Hi = np.linalg.inv(H)
        except np.linalg.LinAlgError: Hi = np.eye(g)
        for jj in range(0, g, dim):
            seg = Wk[:, j0+jj:j0+jj+dim]                       # [rows,dim]
            q = C[_assign_chunk(seg, C)]
            Wq[:, j0+jj:j0+jj+dim] = q
            err = seg - q                                       # [rows,dim]
            hjj = np.maximum(np.abs(np.diag(Hi)[jj:jj+dim]), 1e-30)
            if j0+jj+dim < j0+g:
                Wk[:, j0+jj+dim:j0+g] -= (err/hjj) @ Hi[jj:jj+dim, jj+dim:g]
    p = Xf @ Wq.T; y = Xf @ W.T
    sp2=(p*p).sum(0); spy=(p*y).sum(0)
    lam=np.maximum(sp2/(len(Xf)+1.0),1e-3*sp2+1e-9)
    g2=np.clip((spy+lam*1.0)/(sp2+lam),0.25,4.0)
    import math
    return Wq*g2[:,None].astype(np.float32), math.log2(nc)/dim+0.002

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--x", required=True)
    ap.add_argument("--hf", default="/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base")
    ap.add_argument("--hot-table", default=os.path.join(HERE,"..","corpus","prog_active_top64.txt"))
    ap.add_argument("--ne", type=int, default=2)
    a = ap.parse_args(); L = a.layer
    X = np.load(a.x).astype(np.float32)
    Xf, Xe = X[0::2], X[1::2]
    hot = {}
    for ln in open(a.hot_table):
        p2 = ln.split(":"); hot[int(p2[0][1:])] = [int(x) for x in p2[1].split()]
    cold_all = [e for e in range(256) if e not in set(hot[L])]
    cids = cold_all[:a.ne]; hids = hot[L][:a.ne]
    sh = open_shard_for_layer(a.hf, L)
    Hseq = walsh(256, True)
    import collections; agg = collections.defaultdict(list)
    for lane, ids in (("cold", cids), ("hot", hids)):
        for e in ids:
            W = sh.expert_w(L, e, "w1"); Ye = Xe @ W.T
            Wq,_ = cold_C0(W, Xf) if lane=="cold" else (None,None)
            if lane=="cold":
                rel,_ = metrics(Ye, Xe@Wq.T); agg[(lane,"prod_1.0625")].append(rel)
            variants = ([("vq256_1.00",8,256),("vq512_1.13",8,512),("vq1024_1.25",8,1024)] if lane=="cold"
                        else [("vq256_2.00",4,256),("vq512_2.25",4,512)])
            for tag,dim,nc in variants:
                Wqv,_ = hot_vq(W, Xf, dim=dim, nc=nc)
                rel,_ = metrics(Ye, Xe@Wqv.T); agg[(lane,tag)].append(rel)
            tag,dim,nc = ("gptqvq256_1.00",8,256) if lane=="cold" else ("gptqvq256_2.00",4,256)
            Wqv,_ = vq_gptq(W, Xf, dim=dim, nc=nc)
            rel,_ = metrics(Ye, Xe@Wqv.T); agg[(lane,tag)].append(rel)
            log(f"e{e}({lane}) done")
    print(f"\n== L{L} held-out relF 均值 ==")
    for k in sorted(agg): print(f"{k[0]:4s} {k[1]:16s} relF={np.mean(agg[k]):.4f} n={len(agg[k])}")

if __name__ == "__main__":
    main()
