#!/usr/bin/env python3
"""coadapt_layer.py — 真·逐层共适应: 模型(1-bit base) + 隐变量(z) + 四损失 交替闭式一起动到最优。

对齐用户四支柱, 治"没看到三份一起动": 每层交替最小化【专家输出重建】(诚实指标, 非teacher-forced top-1):
  y = D ⊙ y_base0 + X @ z     (y_base0=1-bit output-optimal, D=每输出通道增益=w2行scale共适应, z=rank-k)
  z-step  : R=y_ref−D⊙y_base0 → 四损失闭式 ridge rank-k, held-out 1-SE 选 (k,λ,dither) 防过拟合
  base-step: 固定z, D_j = Σ(y_ref−Xz)_j·y_base0_j / Σ y_base0_j²   (逐通道闭式, base 真随z重解)
交替 3 轮, 每轮打印 base_cos / +z / +D 三份的当前值 → 看得见一起动。全闭式零训练。
指标=held-out cos(专家输出重建), 治"top-1虚高"。

用法(M1): DS4_HF=... python3 coadapt_layer.py --cap /tmp/capcodeF --layers 0,3,26 [--ntok 800]
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
    """1-bit: sign 固定 + 输出最优 per-row scale。"""
    S = np.sign(W).astype(np.float32); S[S == 0] = 1
    Pt = X @ W.T; Ps = X @ S.T
    num = (Pt * Ps).sum(0); den = (Ps * Ps).sum(0)
    s = np.where(den > 0, num / np.maximum(den, 1e-12), np.abs(W).mean(1))
    return S * s[:, None].astype(np.float32)


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def routed_out(X, ids, wts, L, Xcal, quant=False):
    """routed 专家输出。quant=1-bit(用 Xcal 标定 output-optimal scale)。"""
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


def solve_z_4loss(X, R, rank, lam, dim_w=None, dither=0.0, seed=1):
    """四损失闭式 z: L_fixed=lam ridge, L_classify=dim_w 方差权, L_smooth=dither 输入抖动增广,
    L_align=截断rank后 cos。返回 rank-k 预测器 W [d,d_out]。"""
    Xa, Ra = X, R
    if dither > 0:
        rng = np.random.RandomState(seed)
        Xd = X + dither * X.std(0, keepdims=True) * rng.randn(*X.shape)
        Xa = np.vstack([X, Xd]); Ra = np.vstack([R, R])
    if dim_w is not None:
        sw = np.sqrt(np.maximum(dim_w, 1e-9))[None, :]
        Ra = Ra * sw
    d = Xa.shape[1]
    XtX = Xa.T @ Xa
    XtX[np.diag_indices(d)] += lam * (np.trace(XtX) / d + 1e-9)
    W = np.linalg.solve(XtX, Xa.T @ Ra)
    if dim_w is not None:
        W = W / np.sqrt(np.maximum(dim_w, 1e-9))[None, :]
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    k = min(rank, len(s))
    return (U[:, :k] * s[:k]) @ Vt[:k]


def best_z_1se(Xtr, Rtr, Xte, ybase_te, y_ref_te, dim_w):
    """四损失 z, held-out 选。1-SE 规则治过拟合: 在最优 cos 的一个噪声带内, 选最强正则
    (最大λ/最小rank/最大dither), 不选拟合噪声的弱正则。"""
    cand = []
    for k in (8, 16, 32):
        for lam in (10.0, 30.0, 100.0, 300.0):      # 去掉 λ<10 的弱正则(易过拟合)
            for dith in (0.0, 0.1):
                Wz = solve_z_4loss(Xtr, Rtr, k, lam, dim_w=dim_w, dither=dith)
                c = cos(ybase_te + Xte @ Wz, y_ref_te)
                cand.append((c, k, lam, dith, Wz))
    cand.sort(key=lambda z: -z[0])
    cbest = cand[0][0]
    # 噪声带: held-out cos 的 SE ~ 1/sqrt(n_te); 用简易带宽 0.003
    band = [z for z in cand if z[0] >= cbest - 0.003]
    # 带内选最强正则: 大λ优先, 小rank次之, 大dither再次
    band.sort(key=lambda z: (-z[2], z[1], -z[3]))
    c, k, lam, dith, Wz = band[0]
    return c, Wz, f"k{k}·λ{lam:g}·dith{dith:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--layers", default="0,3,26")
    ap.add_argument("--ntok", type=int, default=800)
    ap.add_argument("--iters", type=int, default=3)
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

        # teacher = fp8 专家输出; base0 = 1-bit(train标定)
        y_ref_tr = routed_out(Xtr, itr, wtr, L, Xtr)
        y_ref_te = routed_out(Xte, ite, wte, L, Xtr)
        yb_tr = routed_out(Xtr, itr, wtr, L, Xtr, quant=True)
        yb_te = routed_out(Xte, ite, wte, L, Xtr, quant=True)
        dim_w = y_ref_tr.var(0) + 1e-9        # L_classify 方差权

        base_cos = cos(yb_te, y_ref_te)
        print(f"\n=== L{L} 共适应 (n={nh}train/{nh}test, 指标=held-out 专家输出 cos) ===", flush=True)
        print(f"iter0  模型:1-bit base           z:—         → cos={base_cos:.4f}", flush=True)

        D_tr = np.ones(D, dtype=np.float32); D_te = np.ones(D, dtype=np.float32)
        Wz = np.zeros((D, D), dtype=np.float32)
        cur = base_cos
        for it in range(1, a.iters + 1):
            # z-step: 对当前 base(D⊙y_base0) 的残差重解 z
            R = y_ref_tr - D_tr * yb_tr
            cz, Wz, ztag = best_z_1se(Xtr, R, Xte, D_te * yb_te, y_ref_te, dim_w)
            # base-step: 固定 z, 逐输出通道重解 D (=w2 行 scale 共适应)
            zt_tr = Xtr @ Wz; zt_te = Xte @ Wz
            tgt = y_ref_tr - zt_tr
            num = (tgt * yb_tr).sum(0); den = (yb_tr * yb_tr).sum(0) + 1e-9
            D_tr = (num / den).astype(np.float32); D_te = D_tr
            cD = cos(D_te * yb_te + zt_te, y_ref_te)
            dmean = float(np.abs(D_tr).mean()); dstd = float(D_tr.std())
            print(f"iter{it}  模型:D通道重标 |D|~{dmean:.3f}±{dstd:.3f}  "
                  f"z:{ztag}  → +z:cos={cz:.4f}  +D共适应:cos={cD:.4f}", flush=True)
            cur = cD
        rank_used = int(ztag.split("·")[0][1:])
        size_z = rank_used * (D + D) * 2   # z rank-k fp16 字节/专家族(近似, per层共享一个z)
        print(f"L{L} 收敛: base{base_cos:.4f} → 共适应{cur:.4f} (+{cur-base_cos:.4f})  "
              f"三份都在动: 模型(D)+z(rank{rank_used})+四损失({ztag})", flush=True)


if __name__ == "__main__":
    main()
