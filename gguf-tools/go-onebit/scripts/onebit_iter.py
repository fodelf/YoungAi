#!/usr/bin/env python3
"""onebit_iter.py — 守住1-bit, 迭代算法抬方向(cos): per-block 输出最优 scale (GO1B格式内, 零额外字节)。

洞察: GO1B 每行本就 16 block(各带 fp16 scale), 现在把 per-row 一个 scale 复制进所有 block=浪费格式。
改每 block 各自求【联合输出最优】scale (per row 解 16 变量 LS 使 Σ_b s_b·(B_b·x) ≈ w·x), 不同 block 不同
scale → 改变 1-bit 能表达的方向 → 抬 cos。仍是 1-bit codes, 字节不变。
对比 nblk∈{1(现状),4,16} 的 held-out 专家输出 cos + rel_L2, 看方向墙能否在 1-bit 内破。

用法(M1): DS4_HF=... python3 onebit_iter.py --cap /tmp/capcodeF --layers 8,26 [--ntok 800]
"""
import argparse
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "calib", "pyfwd"))
sys.path.insert(0, os.path.join(_here, "..", "quant"))
import dsv4_fwd as F   # noqa: E402

D = 4096
NACT = 6


def go1b_row(W, X):
    """现状: per-row 输出最优 scale (一个 scale/行, 复制进 block)。"""
    S = np.sign(W).astype(np.float32); S[S == 0] = 1
    Pt = X @ W.T; Ps = X @ S.T
    num = (Pt * Ps).sum(0); den = (Ps * Ps).sum(0)
    s = np.where(den > 0, num / np.maximum(den, 1e-12), np.abs(W).mean(1))
    return S * s[:, None].astype(np.float32)


def go1b_block(W, X, nblk):
    """per-block 输出最优 scale: per row 联合解 nblk 个 scale 使 Σ_b s_b·(B_b·x) ≈ w·x。"""
    if nblk == 1:
        return go1b_row(W, X)
    dout, din = W.shape
    bs = din // nblk
    B = np.sign(W).astype(np.float32); B[B == 0] = 1
    Y = X @ W.T                                    # [T,dout] teacher 预激活
    T = X.shape[0]
    P = np.empty((T, dout, nblk), np.float32)
    for b in range(nblk):
        c = slice(b * bs, (b + 1) * bs)
        P[:, :, b] = X[:, c] @ B[:, c].T           # Σ_{c∈b} B_ic x_tc
    A = np.einsum('tib,tic->ibc', P, P)            # [dout,nblk,nblk]
    rhs = np.einsum('tib,ti->ib', P, Y)            # [dout,nblk]
    tr = np.trace(A, axis1=1, axis2=2) / nblk + 1e-9
    A[:, np.arange(nblk), np.arange(nblk)] += (1e-3 * tr)[:, None]
    S = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]   # [dout,nblk]
    return B * np.repeat(S, bs, axis=1).astype(np.float32)


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def rel(a, b):
    return float(np.linalg.norm(b - a) / (np.linalg.norm(b) + 1e-12))


def routed_out(X, ids, wts, L, Xcal, nblk=0):
    """nblk=0: fp8 teacher; nblk>=1: 1-bit per-block(nblk) scale。"""
    fe = ids.reshape(-1); fw = wts.reshape(-1)
    tok = np.repeat(np.arange(len(X)), NACT)
    out = np.zeros((len(X), D), dtype=np.float32)
    for e in np.unique(fe):
        if e < 0: continue
        m = fe == e; t = tok[m]; ww = fw[m][:, None]; Xt = X[t]
        w1, w3, w2 = (F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w{k}.weight") for k in (1, 3, 2))
        if nblk >= 1:
            Xh = Xcal if len(Xcal) >= 32 else np.repeat(Xcal, 8, 0)[:256]
            q1 = go1b_block(w1, Xh, nblk); q3 = go1b_block(w3, Xh, nblk)
            g = Xh @ q1.T; u = Xh @ q3.T; h = (g / (1 + np.exp(-g))) * u
            q2 = go1b_block(w2, h, nblk)
            w1, w3, w2 = q1, q3, q2
        np.add.at(out, t, F.expert_fp(Xt, w1, w3, w2, ww))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--layers", default="8,26")
    ap.add_argument("--ntok", type=int, default=800)
    ap.add_argument("--nblks", default="1,4,16")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    nblks = [int(x) for x in a.nblks.split(",")]
    print(f"{'层':>4} {'方案':>12} {'cos':>8} {'rel_L2':>8}   (守1-bit)")
    for L in layers:
        X = np.fromfile(f"{a.cap}/raw_ffn_in_L{L}", dtype="<f2").reshape(-1, D).astype(np.float32)
        ids = np.fromfile(f"{a.cap}/raw_route_L{L}", dtype="<i2").reshape(-1, NACT).astype(np.int64)
        wts = np.fromfile(f"{a.cap}/raw_route_w_L{L}", dtype="<f2").reshape(-1, NACT).astype(np.float32)
        ok = np.isfinite(X).all(1)
        X, ids, wts = X[ok][:a.ntok], ids[ok][:a.ntok], wts[ok][:a.ntok]
        nh = len(X) // 2
        Xtr, Xte = X[:nh], X[nh:2*nh]; itr = ids[:nh]; ite = ids[nh:2*nh]
        wtr = wts[:nh]; wte = wts[nh:2*nh]
        F.load_layer(L)
        yr = routed_out(Xte, ite, wte, L, Xtr, nblk=0)         # fp8 teacher (held-out)
        for nblk in nblks:
            yq = routed_out(Xte, ite, wte, L, Xtr, nblk=nblk)
            tag = "per-row" if nblk == 1 else f"per-blk{nblk}"
            print(f"L{L:>3} {tag:>12} {cos(yq, yr):>8.4f} {rel(yq, yr):>8.4f}", flush=True)


if __name__ == "__main__":
    main()
