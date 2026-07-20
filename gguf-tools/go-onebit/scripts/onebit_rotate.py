#!/usr/bin/env python3
"""onebit_rotate.py — 守1-bit破方向墙候选: Hadamard 旋转非相干处理 (QuIP/QuaRot 式)。★held-out★

原理: y = W x = (W H)(Hᵀ x), H 正交(Hadamard, Hᵀ=H)。硬层权重/激活有 outlier(massive activation),
1-bit 抓不住; 旋转后 W'=WH 分布更均匀(高斯化)→ 1-bit 量化误差降 → cos 涨。仍1bit(在线FWHT)。
每专家: train token 标定 1-bit block-scale, test token 求值 (held-out, 与 onebit_iter 同口径, 防过拟合虚高)。
对比 per-blk16 (无旋转) vs per-blk16+Hadamard。

用法(M1): DS4_HF=... python3 onebit_rotate.py --cap /tmp/capcodeF --layers 8,26 [--ntok 600]
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


def fwht(x):
    """快速 Walsh-Hadamard, 沿最后轴 (n=2^k), 归一化正交。"""
    x = x.astype(np.float32).copy(); n = x.shape[-1]; h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :]; b = x[..., 1, :]
        x = np.concatenate([a + b, a - b], axis=-1).reshape(*x.shape[:-3], n); h *= 2
    return x / np.sqrt(n)


def go1b_block(W, X, nblk):
    """per-block 输出最优 scale (X=标定激活)。nblk=1 即 per-row。"""
    dout, din = W.shape
    B = np.sign(W).astype(np.float32); B[B == 0] = 1
    if nblk == 1:
        Pt = X @ W.T; Ps = X @ B.T
        num = (Pt * Ps).sum(0); den = (Ps * Ps).sum(0)
        s = np.where(den > 0, num / np.maximum(den, 1e-12), np.abs(W).mean(1))
        return B * s[:, None].astype(np.float32)
    bs = din // nblk; Y = X @ W.T; T = X.shape[0]
    P = np.empty((T, dout, nblk), np.float32)
    for b in range(nblk):
        c = slice(b * bs, (b + 1) * bs)
        P[:, :, b] = X[:, c] @ B[:, c].T
    A = np.einsum('tib,tic->ibc', P, P); rhs = np.einsum('tib,ti->ib', P, Y)
    tr = np.trace(A, axis1=1, axis2=2) / nblk + 1e-9
    A[:, np.arange(nblk), np.arange(nblk)] += (1e-3 * tr)[:, None]
    S = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
    return B * np.repeat(S, bs, axis=1).astype(np.float32)


def silu_mid(X, q1, q3):
    g = X @ q1.T; u = X @ q3.T
    return (g / (1 + np.exp(-g))) * u


def expert_ho(Xcal, Xte, w1, w3, w2, wte, nblk, rot):
    """held-out 专家输出: Xcal=全体train token 标定 1-bit scale(充分), Xte=该专家test token 求值。"""
    if rot:
        Xcal, Xte = fwht(Xcal), fwht(Xte)        # Hᵀx
        w1, w3 = fwht(w1), fwht(w3)              # WH (沿 din)
    q1 = go1b_block(w1, Xcal, nblk); q3 = go1b_block(w3, Xcal, nblk)
    hcal = silu_mid(Xcal, q1, q3); hte = silu_mid(Xte, q1, q3)
    if rot:
        hcal, hte = fwht(hcal), fwht(hte); w2 = fwht(w2)
    q2 = go1b_block(w2, hcal, nblk)
    return (hte @ q2.T) * wte


def routed_ho(X, ids, wts, L, nh, nblk, rot):
    """1-bit scale 用【全体 train token X[:nh]】标定(充分, 同 onebit_iter 口径), test 行求值。"""
    fe = ids.reshape(-1); fw = wts.reshape(-1)
    tok = np.repeat(np.arange(len(X)), NACT)
    Xcal = X[:nh]
    out = np.zeros((len(X), D), np.float32)
    for e in np.unique(fe):
        if e < 0: continue
        m = fe == e; t = tok[m]; ww = fw[m][:, None]
        te = t >= nh
        if te.sum() == 0: continue
        Xte = X[t[te]]; wte = ww[te]
        w1, w3, w2 = (F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w{k}.weight") for k in (1, 3, 2))
        if nblk == 0:
            y = F.expert_fp(Xte, w1, w3, w2, wte)
        else:
            y = expert_ho(Xcal, Xte, w1, w3, w2, wte, nblk, rot)
        np.add.at(out, t[te], y)
    return out


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def rel(a, b):
    return float(np.linalg.norm(b - a) / (np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--layers", default="8,26")
    ap.add_argument("--ntok", type=int, default=600)
    ap.add_argument("--nblk", type=int, default=16)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    print(f"{'层':>4} {'方案':>18} {'cos':>8} {'rel_L2':>8}   (held-out, 守1-bit)")
    for L in layers:
        X = np.fromfile(f"{a.cap}/raw_ffn_in_L{L}", dtype="<f2").reshape(-1, D).astype(np.float32)
        ids = np.fromfile(f"{a.cap}/raw_route_L{L}", dtype="<i2").reshape(-1, NACT).astype(np.int64)
        wts = np.fromfile(f"{a.cap}/raw_route_w_L{L}", dtype="<f2").reshape(-1, NACT).astype(np.float32)
        ok = np.isfinite(X).all(1)
        X, ids, wts = X[ok][:a.ntok], ids[ok][:a.ntok], wts[ok][:a.ntok]
        nh = len(X) // 2
        F.load_layer(L)
        yr = routed_ho(X, ids, wts, L, nh, 0, False)      # fp8 teacher (test 行)
        for rot, tag in ((False, f"per-blk{a.nblk}"), (True, f"per-blk{a.nblk}+Hadam")):
            yq = routed_ho(X, ids, wts, L, nh, a.nblk, rot)
            # 只在 test 行比较 (train 行都是 0)
            mask = np.abs(yr).sum(1) > 0
            print(f"L{L:>3} {tag:>18} {cos(yq[mask], yr[mask]):>8.4f} {rel(yq[mask], yr[mask]):>8.4f}", flush=True)


if __name__ == "__main__":
    main()
