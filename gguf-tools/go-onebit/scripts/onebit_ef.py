#!/usr/bin/env python3
"""onebit_ef.py — 守1-bit破方向墙: EF/GPTQ sign 重选 (换掉 naive sign(W))。★held-out★

naive base 用 sign(W); 但输出最优的符号未必是 sign(W)。GPTQ 误差反馈按 Hessian(激活二阶矩)逐列
重选符号最小化输出重建误差 → 直接改方向 → 抬 cos。这才是 1-bit 专用的方向工具(Hadamard 是4-bit的)。
复用 go2b_encode._gptq_assign, 2 level {-s,+s}。对比 per-blk16(naive sign) vs EF-sign+per-blk16。

用法(M1): DS4_HF=... python3 onebit_ef.py --cap /tmp/capcodeF --layers 8,26 [--ntok 600]
"""
import argparse
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "calib", "pyfwd"))
sys.path.insert(0, os.path.join(_here, "..", "quant"))
import dsv4_fwd as F   # noqa: E402
from go2b_encode import _gptq_assign   # noqa: E402  复用 GPTQ 误差反馈

D = 4096
NACT = 6


def perrow_scale(W, X):
    B = np.sign(W).astype(np.float32); B[B == 0] = 1
    Pt = X @ W.T; Ps = X @ B.T
    num = (Pt * Ps).sum(0); den = (Ps * Ps).sum(0)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.abs(W).mean(1)).astype(np.float32)


def block_scale(B, Wtrue, X, nblk):
    """给定符号 B, per-block 输出最优 scale 使 Σ_b s_b·(B_b·x) ≈ w·x。"""
    dout, din = B.shape; Y = X @ Wtrue.T
    if nblk == 1:
        Ps = X @ B.T; num = (Y * Ps).sum(0); den = (Ps * Ps).sum(0)
        s = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        return B * s[:, None].astype(np.float32)
    bs = din // nblk; T = X.shape[0]
    P = np.empty((T, dout, nblk), np.float32)
    for b in range(nblk):
        c = slice(b * bs, (b + 1) * bs); P[:, :, b] = X[:, c] @ B[:, c].T
    A = np.einsum('tib,tic->ibc', P, P); rhs = np.einsum('tib,ti->ib', P, Y)
    tr = np.trace(A, axis1=1, axis2=2) / nblk + 1e-9
    A[:, np.arange(nblk), np.arange(nblk)] += (1e-3 * tr)[:, None]
    S = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
    return B * np.repeat(S, bs, axis=1).astype(np.float32)


def ef_signs(W, Xcal, grp=128, ridge=0.02):
    """GPTQ 误差反馈重选 1-bit 符号 (2 level {-s,+s})。"""
    s0 = perrow_scale(W, Xcal)
    L2 = np.stack([-s0, s0], axis=1).astype(np.float32)      # [rows,2]
    Wq, _ = _gptq_assign(W, L2, Xcal, grp, ridge)
    B = np.sign(Wq).astype(np.float32); B[B == 0] = 1
    return B


def quant_expert(w1, w3, w2, Xcal, Xte, nblk, ef):
    def q(W, Xc):
        B = ef_signs(W, Xc) if ef else _sgn(W)
        return block_scale(B, W, Xc, nblk)
    q1 = q(w1, Xcal); q3 = q(w3, Xcal)
    hcal = _silu(Xcal, q1, q3); hte = _silu(Xte, q1, q3)
    q2 = q(w2, hcal)
    return q1, q3, q2, hte


def _silu(X, q1, q3):
    g = X @ q1.T; u = X @ q3.T
    return (g / (1 + np.exp(-g))) * u


def _sgn(W):
    B = np.sign(W).astype(np.float32); B[B == 0] = 1; return B


def routed_ho(X, ids, wts, L, nh, nblk, ef):
    fe = ids.reshape(-1); fw = wts.reshape(-1)
    tok = np.repeat(np.arange(len(X)), NACT); Xcal = X[:nh]
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
            q1, q3, q2, hte = quant_expert(w1, w3, w2, Xcal, Xte, nblk, ef)
            y = (hte @ q2.T) * wte
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
        yr = routed_ho(X, ids, wts, L, nh, 0, False)
        mask = np.abs(yr).sum(1) > 0
        for ef, tag in ((False, f"per-blk{a.nblk}"), (True, f"EF-sign+blk{a.nblk}")):
            yq = routed_ho(X, ids, wts, L, nh, a.nblk, ef)
            print(f"L{L:>3} {tag:>18} {cos(yq[mask], yr[mask]):>8.4f} {rel(yq[mask], yr[mask]):>8.4f}", flush=True)


if __name__ == "__main__":
    main()
