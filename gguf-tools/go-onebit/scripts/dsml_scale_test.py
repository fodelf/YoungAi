#!/usr/bin/env python3
"""dsml_scale_test.py — 杠杆① 离线验证: 激活感知 scale vs mean|w| (1-bit)。

问题: mono 的 GO1B 用 mean(|w|) (权重空间最优 scale)。posttrain 模块的
输出最优 scale s = Σ(w·x)(b·x)/Σ(b·x)² 声称把 cosine 0.55→0.70。本脚本在
本模型真实专家 + 真实 Go 标定激活上复验: 同样的 sign 位, 只换 scale, 完整
专家前向 (w1/w3/silu/w2) 后对 fp8 参考的 cosine 谁高。

关键: 单行输出 cos 对 scale 不变 → 必须测完整专家输出 (w2 求和后幅度才生效)。
数值全部复用 dsv4_fwd.expert_fp + ds4reader e4m3 dequant, 零新数值代码。

用法 (M1): DS4_HF=/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base \
           python3 dsml_scale_test.py --cap /tmp/scaletest --layer 20 --experts 24
输入: {cap}/raw_ffn_in_L{L} (f16 [n,4096]) + {cap}/raw_route_L{L} (i16 [n,6])
"""
import argparse
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "calib", "pyfwd"), os.path.join(_here, "pyfwd")):
    if os.path.isfile(os.path.join(_cand, "dsv4_fwd.py")):
        sys.path.insert(0, _cand)
        break
import dsv4_fwd as F   # noqa: E402

D = 4096
NACT = 6


def meanabs_scale(W):
    """mean(|w|) per row  -> [rows,1]  (mono GO1B 现用法)"""
    return np.abs(W).mean(axis=1, keepdims=True)


def act_scale(W, X):
    """完整 Gram 输出最优 per-row scale: s_i = Σ_x (w_i·x)(b_i·x) / Σ_x (b_i·x)².
    X: [n,d] 标定激活; 返回 [rows,1]。b = sign(W)。含跨列相关。"""
    S = np.sign(W).astype(np.float32)
    S[S == 0] = 1.0
    Ptrue = X @ W.T          # [n,rows] 真 pre-activation
    Psign = X @ S.T          # [n,rows] 1-bit pre-activation
    num = (Ptrue * Psign).sum(axis=0)
    den = (Psign * Psign).sum(axis=0)
    s = np.where(den > 0, num / np.maximum(den, 1e-12), np.abs(W).mean(axis=1))
    return s[:, None].astype(np.float32)


def diag_scale(W, X):
    """对角 imatrix 加权 scale (go1b_blk_quantize_imat 现用法):
    s_i = Σ_j ew[j]|w_ij| / Σ_j ew[j],  ew[j]=Σ_x x_j² (每列激活能量)。
    只用对角, 不含跨列相关 —— 便宜且是已落地代码。"""
    ew = (X * X).sum(axis=0)                     # [d]
    num = np.abs(W) @ ew                         # [rows]
    den = ew.sum()
    s = num / max(den, 1e-12)
    return s[:, None].astype(np.float32)


def onebit(W, scale):
    """sign(W) * scale  (scale: [rows,1])"""
    S = np.sign(W).astype(np.float32)
    S[S == 0] = 1.0
    return S * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--experts", type=int, default=24)
    ap.add_argument("--ntok", type=int, default=2048)
    a = ap.parse_args()
    L = a.layer

    X = np.fromfile(f"{a.cap}/raw_ffn_in_L{L}", dtype="<f2").reshape(-1, D).astype(np.float32)
    ids = np.fromfile(f"{a.cap}/raw_route_L{L}", dtype="<i2").reshape(-1, NACT)
    # 只用非零行 (L20 首 chunk 零行陷阱同款防御)
    ok = np.abs(X).max(axis=1) > 0
    X, ids = X[ok], ids[ok]
    if len(X) > a.ntok:
        X, ids = X[:a.ntok], ids[:a.ntok]

    # 选路由命中最多的前 experts 个 (代表真实 Go 负载)
    from collections import Counter
    freq = Counter(ids.reshape(-1).tolist())
    picks = [e for e, _ in freq.most_common(a.experts) if e >= 0]

    def flatcos(u, v):
        u, v = u.ravel(), v.ravel()
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

    def expert_out(w1, w3, w2, sfn):
        q1 = onebit(w1, sfn(w1, Xe)); q3 = onebit(w3, sfn(w3, Xe))
        gate = Xe @ q1.T; up = Xe @ q3.T
        h = (gate / (1.0 + np.exp(-gate))) * up                  # silu(gate)*up
        q2 = onebit(w2, sfn(w2, h))
        return F.expert_fp(Xe, q1, q3, q2)

    cos_mean, cos_diag, cos_act = [], [], []
    for e in picks:
        sel = (ids == e).any(axis=1)
        Xe = X[sel]
        if len(Xe) < 8:
            continue
        w1 = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight")
        w3 = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight")
        w2 = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight")
        out_ref = F.expert_fp(Xe, w1, w3, w2)                    # fp8 参考
        cos_mean.append(flatcos(expert_out(w1, w3, w2, lambda W, _X: meanabs_scale(W)), out_ref))
        cos_diag.append(flatcos(expert_out(w1, w3, w2, diag_scale), out_ref))
        cos_act.append(flatcos(expert_out(w1, w3, w2, act_scale), out_ref))

    cm, cd, ca = np.mean(cos_mean), np.mean(cos_diag), np.mean(cos_act)
    print(f"L{L}  experts={len(cos_mean)}  ntok={len(X)}", file=sys.stderr)
    print(f"  mean|w|       cosine→fp8: {cm:.4f}", file=sys.stderr)
    print(f"  diag-imatrix  cosine→fp8: {cd:.4f}   (Δ={cd-cm:+.4f})  [已落地代码]", file=sys.stderr)
    print(f"  full-Gram act cosine→fp8: {ca:.4f}   (Δ={ca-cm:+.4f})", file=sys.stderr)
    print(f"SCALETEST L{L} mean={cm:.4f} diag={cd:.4f} act={ca:.4f}")


if __name__ == "__main__":
    main()
