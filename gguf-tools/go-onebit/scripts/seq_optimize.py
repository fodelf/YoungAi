#!/usr/bin/env python3
"""seq_optimize.py — 单遍顺序量化-优化引擎 (对齐设计的核心)。

逐层推进, 每层在【上游已量化的传播输入 Fin_q】上优化 (=每层更新数据):
  teacher = fp8 MoE(Fin_q)              (fp8 在该真实输入上的输出=局部 teacher)
  student = 量化 MoE(Fin_q) + z(Fin_q)  (base 1-bit + z 动态调优)
  优化 z (held-out 调 λ/rank) 使 student≈teacher → 传播优化后输出给下游
最后 head → 比 fp8 final logits → 整体 top-1 还原率。

一遍完成: 产出优化模型 + 测还原率 + 每层用更新数据。对比 restore_rate 贪心基线 79.5%。
纯量化零训练 (z 闭式解)。

用法(M1): DS4_HF=... python3 seq_optimize.py --ids CODE.ids [--ntok 128] [--z 1] [--layers 0-42]
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


def go1b_q(W, X):
    S = np.sign(W).astype(np.float32); S[S == 0] = 1
    Pt = X @ W.T; Ps = X @ S.T
    num = (Pt * Ps).sum(0); den = (Ps * Ps).sum(0)
    s = np.where(den > 0, num / np.maximum(den, 1e-12), np.abs(W).mean(1))
    return S * s[:, None].astype(np.float32)


def solve_z(X, R, rank, lam):
    d = X.shape[1]
    XtX = X.T @ X
    XtX[np.diag_indices(d)] += lam * (np.trace(XtX) / d + 1e-9)
    W = np.linalg.solve(XtX, X.T @ R)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    k = min(rank, len(s))
    return (U[:, :k] * s[:k]) @ Vt[:k]


class ZSolver:
    """动态 per-layer λ/rank, 对偶形式 (nh 训练样本 << d): eigh 只做 nh×nh 的 XXᵀ (近瞬时,
    非 O(d³) 的 XtX)。ridge 对偶: W_λ = Xᵀ (XXᵀ+λ̄I)⁻¹ R = Xᵀ α_λ, 秩≤nh。
    rank-k 截断: qr(Xᵀ) 一次 → SVD 小的 [nh,dout]。扫 λ×rank 几乎不加时间。"""
    def __init__(self, X, R):
        self.X = X                                         # [nh, d]
        self.nh, self.d = X.shape
        G = X @ X.T                                        # [nh, nh] 小
        self.tr = np.trace(G) / self.d + 1e-9
        self.mu, self.V = np.linalg.eigh(G)                # nh×nh eigh, 瞬时
        self.VtR = self.V.T @ R                            # [nh, dout]
        self.Qx, self.Rx = np.linalg.qr(X.T)               # Xᵀ=Qx Rx, [d,nh][nh,nh] 一次

    def solve(self, rank, lam):
        alpha = self.V @ ((1.0 / (self.mu + lam * self.tr))[:, None] * self.VtR)  # [nh,dout] 便宜
        # W_λ = Xᵀ α = Qx (Rx α); rank-k 截断 = SVD 小的 (Rx α) [nh,dout]
        Ub, s, Vt = np.linalg.svd(self.Rx @ alpha, full_matrices=False)
        k = min(rank, len(s), self.nh)
        U = self.Qx @ Ub[:, :k]                            # [d, k]
        return (U * s[:k]) @ Vt[:k]                          # [d, dout] rank-k


def layer_cache_clear():
    pass   # 缓存一整层专家 ~19GB 爆 16GB → 不缓存, 直接读 (I/O 代价换内存安全)


def _read_expert(L, e):
    return (F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight"),
            F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight"),
            F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight"))


def routed(Fin, Ids, W, L, quant=False, both=False):
    """routed MoE 输出。both=True 一趟(读一次专家)同时算 fp8 teacher + 1-bit student,
    返回 (y_fp8, y_quant)；否则返回 (y, idx)。融合+缓存省重复专家 I/O。"""
    idx, wt, _ = F.gate_route(Fin, W, Ids)
    share = F.expert_fp(Fin, W['s1'], W['s3'], W['s2'])   # shared 高精不量化
    yf = share.copy(); yq = share.copy() if (both or quant) else None
    fe = idx.reshape(-1); fw = wt.reshape(-1)
    tok = np.repeat(np.arange(len(Fin)), NACT)
    for e in np.unique(fe):
        if e < 0: continue
        m = fe == e; t = tok[m]; ww = fw[m][:, None]; Xt = Fin[t]
        w1, w3, w2 = _read_expert(L, e)
        if both or not quant:
            np.add.at(yf, t, F.expert_fp(Xt, w1, w3, w2, ww))
        if both or quant:
            Xc = Xt if len(Xt) >= 32 else np.repeat(Xt, 8, 0)[:256]
            q1 = go1b_q(w1, Xc); q3 = go1b_q(w3, Xc)
            g = Xt @ q1.T; u = Xt @ q3.T; h = (g / (1 + np.exp(-g))) * u
            q2 = go1b_q(w2, h)
            np.add.at(yq, t, F.expert_fp(Xt, q1, q3, q2, ww))
    if both:
        return yf.astype(np.float32), yq.astype(np.float32)
    return (yq if quant else yf).astype(np.float32), idx


def layer_attn_to_fin(H, W, L):
    """H → (Fin=ffn输入, 传播上下文用于重组)。复刻 dsv4_fwd.run 的层内 attn 段。"""
    resid = H
    y, post, comb = F.hc_pre(H, W['hc_attn_fn'], W['hc_attn_scale'], W['hc_attn_base'])
    a = F.attention(F.rms(y, W['an']), W, L)
    h = F.hc_post(a, resid, post, comb)
    resid2 = h
    y2, post2, comb2 = F.hc_pre(h, W['hc_ffn_fn'], W['hc_ffn_scale'], W['hc_ffn_base'])
    fin = F.rms(y2, W['fn'])
    return fin, (resid2, post2, comb2)


def head_logits(H):
    hcfn = F.R.get("hc_head_fn"); hcb = F.R.get("hc_head_base"); hcs = F.R.get("hc_head_scale")
    fnorm = F.R.get("norm.weight"); hw = F.R.get("head.weight")
    S = H.shape[0]; x = H.reshape(S, -1)
    rsq = np.reciprocal(np.sqrt(np.mean(x * x, -1, keepdims=True) + F.EPS))
    mixes = (x @ hcfn.T) * rsq
    pre = F.sigmoid(mixes * hcs + hcb) + F.HCEPS
    y = (pre[:, :, None] * H).sum(1)
    y = F.rms(y, fnorm)
    return y @ hw.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ntok", type=int, default=128)
    ap.add_argument("--z", type=int, default=1)             # 1=优化z, 0=贪心base(对照79.5%)
    ap.add_argument("--zk", type=int, default=64)           # z rank (固定, L0验证)
    ap.add_argument("--zlam", type=float, default=100.0)    # z 正则 (固定强正则防过拟合)
    ap.add_argument("--layers", default="0-42")
    a = ap.parse_args()
    lo, hi = map(int, a.layers.split("-"))
    qlayers = set(range(lo, hi + 1))
    ids = np.array([int(x) for x in open(a.ids) if x.strip()], dtype=np.int64)[:a.ntok]
    Ids = ids
    print(f"tokens={len(ids)} z={'开' if a.z else '关(贪心对照)'} 量化层={a.layers}", file=sys.stderr, flush=True)

    emb = F.R.get("embed.weight")
    HCM = F.HCM
    H_fp8 = np.repeat(emb[ids][:, None, :], HCM, 1).astype(np.float32)
    H_q = H_fp8.copy()
    nh = len(ids) // 2   # train/test 分裂 (z held-out 调优)

    for L in range(F.NL):
        W = F.load_layer(L)
        import time; t0 = time.time()
        # fp8 轨迹
        Fin_f, ctx_f = layer_attn_to_fin(H_fp8, W, L)
        Fout_f, _ = routed(Fin_f, Ids, W, L, quant=False)
        r2, p2, c2 = ctx_f; H_fp8 = F.hc_post(Fout_f, r2, p2, c2)
        # 量化轨迹 (传播输入 Fin_q = 上游已量化)
        Fin_q, ctx_q = layer_attn_to_fin(H_q, W, L)
        if L in qlayers:
            teach, stud = routed(Fin_q, Ids, W, L, both=True)     # 一趟: fp8 teacher + 1-bit student
            Fout_q = stud
            zinfo = "base"
            if a.z:
                R = teach - stud
                # ★动态 per-layer λ/rank★: eigendecomp 一次, 扫 (rank×λ) held-out 选最优
                # (随机SVD, 几乎不加时间)。有效性 gate: 最优仍不降 err 才弃(防过拟合)。
                zs = ZSolver(Fin_q[:nh], R[:nh])
                err_b = float(np.abs(stud[nh:] - teach[nh:]).mean())
                best_err, best_W, best_tag = err_b, None, "z弃(过拟合)"
                # rank 上限=nh(训练样本), rank<nh 是第二正则; λ 主旋钮。全对偶几乎零成本。
                for k in (8, 16, 32, 64):
                    for lam in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
                        Wz = zs.solve(k, lam)
                        e = float(np.abs(stud[nh:] + Fin_q[nh:] @ Wz - teach[nh:]).mean())
                        if e < best_err:
                            best_err, best_W, best_tag = e, Wz, f"z{k}·λ{lam:g}✓"
                if best_W is not None:
                    Fout_q = stud + Fin_q @ best_W
                zinfo = best_tag
        else:
            Fout_q, _ = routed(Fin_q, Ids, W, L, quant=False)
            zinfo = "fp8"
        r2q, p2q, c2q = ctx_q; H_q = F.hc_post(Fout_q, r2q, p2q, c2q)
        # 每层兑现质量: 早退 head → 至此层量化 vs fp8 的 top-1 一致率 (保真度轨迹) + 隐状态 cos。
        # 看得见哪层伤质量、z 有没有救回来 (44min 前就有逐层判决, 不用等最后)。
        lf_i = head_logits(H_fp8); lq_i = head_logits(H_q)
        ag_i = float(np.mean(np.argmax(lf_i[:-1], 1) == np.argmax(lq_i[:-1], 1)))
        hc = float((H_fp8 * H_q).sum() / (np.linalg.norm(H_fp8) * np.linalg.norm(H_q) + 1e-9))
        print(f"L{L:2d} {zinfo:12s} {time.time()-t0:5.1f}s  质量:top1={ag_i:.3f} cos={hc:.4f}", flush=True)
        layer_cache_clear()   # 释放本层专家缓存, 不跨层累积
        del W

    lf = head_logits(H_fp8); lq = head_logits(H_q)
    pf = np.argmax(lf[:-1], 1); pq = np.argmax(lq[:-1], 1)
    agree = float(np.mean(pf == pq))
    print(f"SEQOPT z={'开' if a.z else '关'} 层={a.layers}: top-1一致率={agree:.4f} (n={len(pf)})")


if __name__ == "__main__":
    main()
