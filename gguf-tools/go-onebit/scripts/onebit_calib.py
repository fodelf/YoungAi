#!/usr/bin/env python3
"""onebit_calib.py — 探: 更多标定数据能否抬 per-block-16 (固定test集, 变标定量)。★held-out★

full_optimize 现在只用 64 token 标定 per-block scale; capture 有 ~1341 token。若更多标定数据抬 cos,
则"scale 在 capture 大集标定 / 前向仍少 token"解耦 = 免费提质不拖慢前向。
固定最后 150 token 作 test, per-blk16 scale 标定在 N∈{80,300,1000} 训练 token, 测同一 test cos。

用法(M1): DS4_HF=... python3 onebit_calib.py --cap /tmp/capcodeF --layers 8,26
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


def go1b_block(W, X, nblk=16):
    dout, din = W.shape
    B = np.sign(W).astype(np.float32); B[B == 0] = 1
    bs = din // nblk; Y = X @ W.T; T = X.shape[0]
    P = np.empty((T, dout, nblk), np.float32)
    for b in range(nblk):
        c = slice(b * bs, (b + 1) * bs); P[:, :, b] = X[:, c] @ B[:, c].T
    A = np.einsum('tib,tic->ibc', P, P); rhs = np.einsum('tib,ti->ib', P, Y)
    tr = np.trace(A, axis1=1, axis2=2) / nblk + 1e-9
    A[:, np.arange(nblk), np.arange(nblk)] += (1e-3 * tr)[:, None]
    S = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
    return B * np.repeat(S, bs, axis=1).astype(np.float32)


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def routed(Xcal, Xte, ite, wte, L, quant):
    """quant=None: fp8 teacher(Xte). 否则 1-bit per-blk16(Xcal标定, Xte求值)。"""
    fe = ite.reshape(-1); fw = wte.reshape(-1)
    tok = np.repeat(np.arange(len(Xte)), NACT)
    out = np.zeros((len(Xte), D), np.float32)
    for e in np.unique(fe):
        if e < 0: continue
        m = fe == e; t = tok[m]; ww = fw[m][:, None]; Xt = Xte[t]
        w1, w3, w2 = (F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w{k}.weight") for k in (1, 3, 2))
        if quant is None:
            np.add.at(out, t, F.expert_fp(Xt, w1, w3, w2, ww))
        else:
            q1 = go1b_block(w1, Xcal); q3 = go1b_block(w3, Xcal)
            hc = (lambda g, u: (g / (1 + np.exp(-g))) * u)(Xcal @ q1.T, Xcal @ q3.T)
            q2 = go1b_block(w2, hc)
            np.add.at(out, t, F.expert_fp(Xt, q1, q3, q2, ww))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--layers", default="8,26")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    print(f"{'层':>4} {'标定N':>7} {'cos':>8}   (固定test150, per-blk16)")
    for L in layers:
        X = np.fromfile(f"{a.cap}/raw_ffn_in_L{L}", dtype="<f2").reshape(-1, D).astype(np.float32)
        ids = np.fromfile(f"{a.cap}/raw_route_L{L}", dtype="<i2").reshape(-1, NACT).astype(np.int64)
        wts = np.fromfile(f"{a.cap}/raw_route_w_L{L}", dtype="<f2").reshape(-1, NACT).astype(np.float32)
        ok = np.isfinite(X).all(1); X, ids, wts = X[ok], ids[ok], wts[ok]
        # 固定 test = 最后 150; 训练池 = 其余
        Xte = X[-150:]; ite = ids[-150:]; wte = wts[-150:]
        Xpool = X[:-150]
        yr = routed(None, Xte, ite, wte, L, None)
        for N in (80, 300, 1000):
            if N > len(Xpool): N = len(Xpool)
            Xcal = Xpool[:N]
            yq = routed(Xcal, Xte, ite, wte, L, quant=True)
            print(f"L{L:>3} {N:>7} {cos(yq, yr):>8.4f}", flush=True)


if __name__ == "__main__":
    main()
