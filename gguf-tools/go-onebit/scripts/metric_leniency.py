#!/usr/bin/env python3
"""metric_leniency.py — 坐实"top-1一致率虚高": 量化只动 routed 专家, 但传播出去的 MoE 输出里
shared 专家(高精不量化)占大头, routed 的 1-bit 误差被淹没 → 下游 top-1 几乎不掉 → 指标宽松。

对每层报:
  ||routed_fp8|| vs ||shared||        (routed 在 MoE 输出里占多大)
  cos(routed_1bit, routed_fp8)         (专家层真·1-bit 保真度, 低)
  cos(MoE_1bit, MoE_fp8)               (真正传播出去的量, 被 shared 淹没后有多高)
  cos 差 = 指标宽松度证据

用法(M1): DS4_HF=... python3 metric_leniency.py --cap /tmp/capcodeF --layers 0,3,26 [--ntok 500]
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


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def routed_out(X, ids, wts, L, quant=False):
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
            Xh = Xt if len(Xt) >= 32 else np.repeat(Xt, 8, 0)[:256]
            w1 = go1b_quant(w1, Xh); w3 = go1b_quant(w3, Xh)
            g = Xh @ w1.T; u = Xh @ w3.T; h = (g / (1 + np.exp(-g))) * u
            w2 = go1b_quant(w2, h)
        np.add.at(out, t, F.expert_fp(Xt, w1, w3, w2, ww))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--layers", default="0,3,26")
    ap.add_argument("--ntok", type=int, default=500)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    print(f"{'层':>4} {'||routed||':>10} {'||shared||':>10} {'routed占比':>9} "
          f"{'cos(专家1b)':>11} {'cos(MoE传播)':>12} {'宽松度Δ':>8}")
    for L in layers:
        X = np.fromfile(f"{a.cap}/raw_ffn_in_L{L}", dtype="<f2").reshape(-1, D).astype(np.float32)
        ids = np.fromfile(f"{a.cap}/raw_route_L{L}", dtype="<i2").reshape(-1, NACT).astype(np.int64)
        wts = np.fromfile(f"{a.cap}/raw_route_w_L{L}", dtype="<f2").reshape(-1, NACT).astype(np.float32)
        ok = np.isfinite(X).all(1)
        X, ids, wts = X[ok][:a.ntok], ids[ok][:a.ntok], wts[ok][:a.ntok]
        W = F.load_layer(L)
        shared = F.expert_fp(X, W['s1'], W['s3'], W['s2']).astype(np.float32)
        r_fp8 = routed_out(X, ids, wts, L, quant=False)
        r_1b = routed_out(X, ids, wts, L, quant=True)
        moe_fp8 = shared + r_fp8
        moe_1b = shared + r_1b
        nr = float(np.sqrt((r_fp8**2).mean()))
        ns = float(np.sqrt((shared**2).mean()))
        frac = nr / (nr + ns + 1e-9)
        ce = cos(r_1b, r_fp8)            # 专家层 1-bit 保真度 (真实, 低)
        cm = cos(moe_1b, moe_fp8)        # 传播出去的 MoE 输出 (被 shared 淹没后)
        print(f"L{L:>3} {nr:>10.3f} {ns:>10.3f} {frac:>8.1%} "
              f"{ce:>11.4f} {cm:>12.4f} {cm-ce:>+8.4f}", flush=True)
        del W


if __name__ == "__main__":
    main()
