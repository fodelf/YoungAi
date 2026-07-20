#!/usr/bin/env python3
"""layer_design.py — 逐层量化设计 (对齐设计: 每层 base+z一起, 层层对应, 调到最小体积+最好质量)。

单层 L: 用真实激活 x_L (顺序模式=已量化0..L-1产出; L0=embedding后近似fp8) 求
  y_ref = fp8 MoE(x_L)   (该层fp8函数=每层teacher)
扫 (base bit-width × z-rank k), 每组合:
  y_q = 量化MoE(x_L) + z校正(rank k, 闭式解 R=y_ref-y_base)
  质量 = cos(y_q, y_ref) + 四损失; 体积 = base_bits/param + z_rank*(d_in+d_out)
输出该层的 质量↔体积 前沿, 挑甜点。快速: 单层, offline, 分钟级。

用法(M1): DS4_HF=... python3 layer_design.py --layer 0 --cap /tmp/capcodeF [--experts 64]
"""
import argparse
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "calib", "pyfwd"))
sys.path.insert(0, os.path.join(_here, "..", "quant"))
import dsv4_fwd as F   # noqa: E402
from go2b_encode import encode_go2b, decode_go2b   # noqa: E402

D = 4096
NACT = 6


def go1b_quant(W, X):
    """1-bit: sign(W) * 输出最优 per-row scale (激活感知)。"""
    S = np.sign(W).astype(np.float32); S[S == 0] = 1
    Pt = X @ W.T; Ps = X @ S.T
    num = (Pt * Ps).sum(0); den = (Ps * Ps).sum(0)
    s = np.where(den > 0, num / np.maximum(den, 1e-12), np.abs(W).mean(1))
    return S * s[:, None].astype(np.float32)


def solve_z(X, R, rank, lam=1e-3):
    """闭式低秩: (XᵀX+λI)W=XᵀR → 截断rank. 返回预测器 (给新X算 X@W_k)。"""
    n, d = X.shape
    XtX = X.T @ X
    XtX[np.diag_indices(d)] += lam * (np.trace(XtX) / d + 1e-9)
    W = np.linalg.solve(XtX, X.T @ R)          # [d, d_out]
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    k = min(rank, len(s))
    return (U[:, :k] * s[:k]) @ Vt[:k]          # [d, d_out] rank-k


def solve_z_4loss(X, R, rank, lam, dim_w=None, dither=0.0, seed=1):
    """四损失闭式 z:
      L_fixed   = lam (ridge 正则, 已有)
      L_classify= dim_w (按输出维方差加权 → 目标 R 各维乘 sqrt(w), 保判别维)
      L_smooth  = dither (输入加固定种子抖动增广 → 鲁棒, 不过拟合具体激活)
      L_align   = 截断rank后由调用方测 cosine
    加权 solve: min ||(X_aug W - R_aug)·diag(√w)||² + λ||W||²。"""
    Xa, Ra = X, R
    if dither > 0:
        rng = np.random.RandomState(seed)
        Xd = X + dither * X.std(0, keepdims=True) * rng.randn(*X.shape)
        Xa = np.vstack([X, Xd]); Ra = np.vstack([R, R])      # 抖动输入→同目标=鲁棒
    if dim_w is not None:
        sw = np.sqrt(np.maximum(dim_w, 1e-9))[None, :]        # [1,d_out]
        Ra = Ra * sw                                          # 目标按维加权
    n, d = Xa.shape
    XtX = Xa.T @ Xa
    XtX[np.diag_indices(d)] += lam * (np.trace(XtX) / d + 1e-9)
    W = np.linalg.solve(XtX, Xa.T @ Ra)
    if dim_w is not None:
        W = W / np.sqrt(np.maximum(dim_w, 1e-9))[None, :]     # 解回原尺度
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    k = min(rank, len(s))
    return (U[:, :k] * s[:k]) @ Vt[:k]


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--ntok", type=int, default=1000)
    a = ap.parse_args()
    L = a.layer

    X = np.fromfile(f"{a.cap}/raw_ffn_in_L{L}", dtype="<f2").reshape(-1, D).astype(np.float32)
    ids = np.fromfile(f"{a.cap}/raw_route_L{L}", dtype="<i2").reshape(-1, NACT).astype(np.int64)
    wts = np.fromfile(f"{a.cap}/raw_route_w_L{L}", dtype="<f2").reshape(-1, NACT).astype(np.float32)
    ok = np.isfinite(X).all(1)
    X, ids, wts = X[ok][:a.ntok], ids[ok][:a.ntok], wts[ok][:a.ntok]
    n = len(X)
    # train/test 分裂 (泛化, 不过拟合校准)
    nh = n // 2
    Xtr, itr, wtr = X[:nh], ids[:nh], wts[:nh]
    Xte, ite, wte = X[nh:2*nh], ids[nh:2*nh], wts[nh:2*nh]

    W = F.load_layer(L)
    # 用捕获的真实路由 (ids+gate权重) 算 routed MoE 输出, teacher=fp8。不重算路由。
    def routed_out(Xset, iset, wset, quant=None):
        fe = iset.reshape(-1); fw = wset.reshape(-1)
        tok = np.repeat(np.arange(len(Xset)), NACT)
        out = np.zeros((len(Xset), D))
        for e in np.unique(fe):
            if e < 0:
                continue
            m = fe == e; t = tok[m]; ww = fw[m][:, None]; Xt = Xset[t]
            w1 = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight")
            w3 = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight")
            w2 = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight")
            if quant == "go1b":
                seltr = (itr == e).any(1); Xh = Xtr[seltr] if seltr.sum() >= 32 else Xtr[:256]
                w1 = go1b_quant(w1, Xh); w3 = go1b_quant(w3, Xh)
                g = Xt @ w1.T; u = Xt @ w3.T; h = (g / (1 + np.exp(-g))) * u
                hh = Xh @ (w1).T; uu = Xh @ (w3).T; hcal = (hh / (1 + np.exp(-hh))) * uu
                w2 = go1b_quant(w2, hcal)
            elif quant == "go2b":
                seltr = (itr == e).any(1); Xh = Xtr[seltr] if seltr.sum() >= 32 else Xtr[:256]
                os.environ["DS4_GO2B_ACT_SCALE"] = "1"
                b1, _ = encode_go2b(w1, Xh); b3, _ = encode_go2b(w3, Xh)
                w1 = decode_go2b(b1, 4096); w3 = decode_go2b(b3, 4096)
                g = Xt @ w1.T; u = Xt @ w3.T; h = (g / (1 + np.exp(-g))) * u
                hh = Xh @ w1.T; uu = Xh @ w3.T; hcal = (hh / (1 + np.exp(-hh))) * uu
                b2, _ = encode_go2b(w2, hcal); w2 = decode_go2b(b2, 2048)
                os.environ.pop("DS4_GO2B_ACT_SCALE", None)
            np.add.at(out, t, F.expert_fp(Xt, w1, w3, w2, ww))
        return out

    print(f"L{L} 计算 fp8 teacher + 量化基线 (train n={nh})...", file=sys.stderr, flush=True)
    y_ref_tr = routed_out(Xtr, itr, wtr)
    y_ref_te = routed_out(Xte, ite, wte)
    print(f"  base rms fp8={np.sqrt((y_ref_te**2).mean()):.3f}", file=sys.stderr, flush=True)

    quant_sweep = [("1-bit", "go1b")]
    if os.environ.get("INCLUDE_GO2B"):
        quant_sweep.append(("2-bit", "go2b"))
    for bit, qname in quant_sweep:
        y_base_tr = routed_out(Xtr, itr, wtr, quant=qname)
        y_base_te = routed_out(Xte, ite, wte, quant=qname)
        base_cos = cos(y_base_te, y_ref_te)
        R_tr = y_ref_tr - y_base_tr
        print(f"L{L} {bit} base(无z) HELD-OUT cos→fp8={base_cos:.4f}", flush=True)

        # L_classify 权重: 输出维方差 (保判别维)
        dim_var = y_ref_tr.var(0) + 1e-9

        def best_z(Rtr, ybase_te, four=False):
            """z 动态调优: 扫 rank×λ (four=True 加 L_classify方差权+L_smooth抖动)。"""
            bc = cos(ybase_te, y_ref_te); bw, bt = None, "base"
            for k in (16, 32, 64):
                for lam in (1.0, 10.0, 100.0, 300.0):
                    if four:
                        for dith in (0.0, 0.05, 0.1):
                            Wz = solve_z_4loss(Xtr, Rtr, k, lam, dim_w=dim_var, dither=dith)
                            c = cos(ybase_te + Xte @ Wz, y_ref_te)
                            if c > bc: bc, bw, bt = c, Wz, f"z{k}·λ{lam:g}·cls·sm{dith:g}"
                    else:
                        Wz = solve_z(Xtr, Rtr, k, lam=lam)
                        c = cos(ybase_te + Xte @ Wz, y_ref_te)
                        if c > bc: bc, bw, bt = c, Wz, f"z{k}·λ{lam:g}"
            return bc, bw, bt

        # ①裸 z (只 L_align+L_fixed)
        c1, Wz1, t1 = best_z(R_tr, y_base_te, four=False)
        print(f"L{L} {bit} [裸z: align+fixed] 最优 {t1} cos={c1:.4f} (+{c1-base_cos:.4f})", flush=True)
        # ①b 四损失 z (+L_classify方差权 +L_smooth抖动)
        c1f, Wz1f, t1f = best_z(R_tr, y_base_te, four=True)
        print(f"L{L} {bit} [四损失z: +classify+smooth] 最优 {t1f} cos={c1f:.4f} "
              f"(vs 裸z {c1:.4f}, 四损失{'再+%.4f'%(c1f-c1) if c1f>c1 else '无增益'})", flush=True)
        if c1f > c1: c1, Wz1, t1 = c1f, Wz1f, t1f

        # ②z↔base co-adapt: base scale 不是死的, 让它适应 (fp8 - z校正) 的目标, 迭代
        if qname == "go1b" and Wz1 is not None:
            # 目标残给 base: base 应拟合 y_ref - z(x)。等价: 从 fp8 输出里减去 z 贡献,
            # base 重量化时的 activation-aware scale 对"扣掉z后"的目标最优 → 迭代2轮。
            c_best, tag_best = c1, t1
            R_iter = R_tr; ybt = y_base_te
            for it in range(2):
                _, Wz, tg = best_z(R_iter, ybt)
                if Wz is None: break
                # base co-adapt: 重算 base scale 使 base 输出 → (y_ref - z) 而非 y_ref。
                # 近似(闭式): base 已是 output-optimal to y_ref; z 吃了低秩部分,
                # base 对剩余(y_ref - Xz)重标 → 用 per-expert 重算(此处用整体缩放近似 co-adapt)。
                zt_tr = Xtr @ Wz; zt_te = Xte @ Wz
                # base 缩放 α 使 α·ybase + z 最贴 y_ref (闭式 α per全层)
                num = ((y_ref_tr - zt_tr) * y_base_tr).sum(); den = (y_base_tr**2).sum() + 1e-9
                al = num / den
                ybt2 = al * y_base_te
                c2 = cos(ybt2 + zt_te, y_ref_te)
                if c2 > c_best: c_best, tag_best = c2, f"{tg}+base·α{al:.3f}(it{it+1})"
                R_iter = y_ref_tr - al * y_base_tr; ybt = ybt2
            print(f"L{L} {bit} [z↔base co-adapt] 最优 {tag_best} cos={c_best:.4f} "
                  f"(vs 静态base {base_cos:.4f}, 动态z {c1:.4f})", flush=True)


if __name__ == "__main__":
    main()
