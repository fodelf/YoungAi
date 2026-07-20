#!/usr/bin/env python3
"""dcoef_eval.py — 文档记录的正解: 乘法动态系数 D0→D1 单层评分表 (z-dynamic-redesign.md 2026-07-05)。

治"质量差的层": 根因=幅度塌缩(学生/教师范数比~0.43×), 不是方向。加法残差 z 已证伪(天花板0.545,
数据/rank 都不救)。乘法动态系数原生表达缩放。★关键: cos 对幅度失明, 必须用幅度敏感指标 rel_L2/范数比★。

阶梯(全闭式零训练, held-out 评):
  base   : ŷ = 1-bit output-optimal
  D0     : y'=g_L·ŷ, g_L=⟨y*,ŷ⟩/⟨ŷ,ŷ⟩ 全层标量 (零字节, 折进 down scale)
  D0ch   : y'=g⊙ŷ, g_j=Σy*_j ŷ_j/Σŷ_j² 每输出通道 (=w2 行 scale)
  D1     : y'=g(x)·ŷ, 每token最优增益 g*_t=⟨y*_t,ŷ_t⟩/‖ŷ_t‖² 对 x 岭回归 → 动态标量
指标: cos(方向) / rel_L2=‖y*−y'‖/‖y*‖(幅度敏感) / 范数比‖y'‖/‖y*‖。

用法(M1): DS4_HF=... python3 dcoef_eval.py --cap /tmp/capcodeF --layers 3,8,26 [--ntok 800]
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


def go1b_quant(W, X):
    S = np.sign(W).astype(np.float32); S[S == 0] = 1
    Pt = X @ W.T; Ps = X @ S.T
    num = (Pt * Ps).sum(0); den = (Ps * Ps).sum(0)
    s = np.where(den > 0, num / np.maximum(den, 1e-12), np.abs(W).mean(1))
    return S * s[:, None].astype(np.float32)


def routed_out(X, ids, wts, L, Xcal, quant=False):
    fe = ids.reshape(-1); fw = wts.reshape(-1)
    tok = np.repeat(np.arange(len(X)), NACT)
    out = np.zeros((len(X), D), dtype=np.float32)
    for e in np.unique(fe):
        if e < 0:
            continue
        m = fe == e; t = tok[m]; ww = fw[m][:, None]; Xt = X[t]
        w1 = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight")
        w3 = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight")
        w2 = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight")
        if quant:
            Xh = Xcal if len(Xcal) >= 32 else np.repeat(Xcal, 8, 0)[:256]
            w1 = go1b_quant(w1, Xh); w3 = go1b_quant(w3, Xh)
            g = Xh @ w1.T; u = Xh @ w3.T; h = (g / (1 + np.exp(-g))) * u
            w2 = go1b_quant(w2, h)
        np.add.at(out, t, F.expert_fp(Xt, w1, w3, w2, ww))
    return out


def metrics(yp, yr):
    cos = float((yp.ravel() @ yr.ravel()) / (np.linalg.norm(yp) * np.linalg.norm(yr) + 1e-12))
    rel = float(np.linalg.norm(yr - yp) / (np.linalg.norm(yr) + 1e-12))
    nr = float(np.linalg.norm(yp) / (np.linalg.norm(yr) + 1e-12))
    return cos, rel, nr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--layers", default="3,8,26")
    ap.add_argument("--ntok", type=int, default=800)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    for L in layers:
        X = np.fromfile(f"{a.cap}/raw_ffn_in_L{L}", dtype="<f2").reshape(-1, D).astype(np.float32)
        ids = np.fromfile(f"{a.cap}/raw_route_L{L}", dtype="<i2").reshape(-1, NACT).astype(np.int64)
        wts = np.fromfile(f"{a.cap}/raw_route_w_L{L}", dtype="<f2").reshape(-1, NACT).astype(np.float32)
        ok = np.isfinite(X).all(1)
        X, ids, wts = X[ok][:a.ntok], ids[ok][:a.ntok], wts[ok][:a.ntok]
        n = len(X); nh = n // 2
        Xtr, Xte = X[:nh], X[nh:2*nh]
        itr, ite = ids[:nh], ids[nh:2*nh]
        wtr, wte = wts[:nh], wts[nh:2*nh]
        F.load_layer(L)

        yr_tr = routed_out(Xtr, itr, wtr, L, Xtr)
        yr_te = routed_out(Xte, ite, wte, L, Xtr)
        yb_tr = routed_out(Xtr, itr, wtr, L, Xtr, quant=True)
        yb_te = routed_out(Xte, ite, wte, L, Xtr, quant=True)

        print(f"\n=== L{L} 动态系数阶梯 (held-out n={nh}, ★rel_L2越小越好, 范数比→1★) ===", flush=True)
        print(f"{'方案':<10}{'cos':>8}{'rel_L2':>9}{'范数比':>8}   说明")
        c, r, nr = metrics(yb_te, yr_te)
        print(f"{'base':<10}{c:>8.4f}{r:>9.4f}{nr:>8.3f}   1-bit output-optimal", flush=True)

        # D0 全层标量增益
        g0 = float((yr_tr * yb_tr).sum() / ((yb_tr**2).sum() + 1e-9))
        c, r, nr = metrics(g0 * yb_te, yr_te)
        print(f"{'D0标量':<10}{c:>8.4f}{r:>9.4f}{nr:>8.3f}   g_L={g0:.3f} (零字节)", flush=True)

        # D0ch 每输出通道增益
        gch = (yr_tr * yb_tr).sum(0) / ((yb_tr**2).sum(0) + 1e-9)
        c, r, nr = metrics(gch[None, :] * yb_te, yr_te)
        print(f"{'D0通道':<10}{c:>8.4f}{r:>9.4f}{nr:>8.3f}   |g|~{np.abs(gch).mean():.3f}±{gch.std():.3f}", flush=True)

        # D1 动态标量: 每token最优增益 g*_t 对 x 岭回归
        gstar = (yr_tr * yb_tr).sum(1) / ((yb_tr**2).sum(1) + 1e-9)   # [nh]
        Xa = np.hstack([np.ones((nh, 1), np.float32), Xtr])            # 加偏置
        lam = 1e-2 * (Xa**2).sum() / Xa.shape[1]
        A = Xa.T @ Xa; A[np.diag_indices(A.shape[0])] += lam
        wcoef = np.linalg.solve(A, Xa.T @ gstar)                       # [d+1]
        gte = np.hstack([np.ones((nh, 1), np.float32), Xte]) @ wcoef   # [nh]
        c, r, nr = metrics(gte[:, None] * yb_te, yr_te)
        print(f"{'D1动态':<10}{c:>8.4f}{r:>9.4f}{nr:>8.3f}   g(x)均{gte.mean():.3f} (侧车{ (D+1)*4//1024 }KB)", flush=True)

        # 参考: 学生/教师范数比 (幅度塌缩证据)
        ratio = float(np.linalg.norm(yb_te) / (np.linalg.norm(yr_te) + 1e-12))
        print(f"  [幅度塌缩证据] base 范数比={ratio:.3f} (文档预期~0.43×, 需增益~{1/max(ratio,1e-3):.2f}×)", flush=True)


if __name__ == "__main__":
    main()
