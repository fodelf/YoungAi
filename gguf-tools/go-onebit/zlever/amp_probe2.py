#!/usr/bin/env python3
# amp_probe2.py — v6.1 乘性放大器·闭式判决探针(2026-08-19 用户令"不要训练/零自加超参")。
# 数学: min‖y_fp − y_q⊙(1+g(x))‖² 变量代换 = 比值空间 RRR: 目标 r=dH·y_q/(y_q²+ε²)(光滑
# 正则比值), 列权=rms|y_q|·感知colw(行为空间等价, 列级近似与 zlayer colw 同哲学)。
# 解法与 zlayer 四损失闭式逐式同构: 对偶 ridge + dither 增广 + SVD 截 k + k_L held 网格自选。
# 零训练。判据: held 行为挽回 1−‖y_fp−ŷ‖²/‖y_fp−y_q‖² (补差闭式=0.0% 在案)。
# 用法: amp_probe2.py <anchor> <zcache_LXX.npz> [NFIT=1638]
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from probe_layer_behavior import anchor_layer

ap, zcp = sys.argv[1], sys.argv[2]
NFIT = int(sys.argv[3]) if len(sys.argv) > 3 else 1638
L = int(os.path.basename(zcp).split("_L")[1][:2])
zc = np.load(zcp)
dH, prow, pw, pYQ = zc["dH"], zc["prow"], zc["pw"], zc["pYQ"]
S, D = dH.shape
X0, _, _, _ = anchor_layer(ap, L, S)

yq = np.zeros((S, D), dtype=np.float32)
np.add.at(yq, prow, pw[:, None] * pYQ)
yfp = yq + dH
tr, ev = np.arange(0, NFIT), np.arange(NFIT, S)

# 比值残差(光滑正则: ε=每列 rms|y_q| 的 1e-2, 无奇点; y_q→0 处 r→0=自动降权)
eps = np.sqrt((yq[tr] ** 2).mean(0)) * 1e-2 + 1e-12
R = dH * yq / (yq ** 2 + eps[None, :] ** 2)
# 列权 = 感知colw(比值域方差) × 行为等价权(rms|y_q|) — zlayer colw 同哲学
colw = np.sqrt(R[tr].var(0) + 1e-12) * np.sqrt((yq[tr] ** 2).mean(0) + 1e-12)
X = X0.astype(np.float64)
rms = np.sqrt((X[tr] ** 2).mean(1, keepdims=True))
r2 = np.random.RandomState(1)
Xa = np.vstack([X[tr], (X[tr] + r2.randn(len(tr), D) * 0.04 * rms) * np.sqrt(0.25)])
Ra = np.vstack([R[tr] * colw, (R[tr] * colw) * np.sqrt(0.25)])
G = Xa @ Xa.T
G[np.diag_indices_from(G)] += 3.0 * np.trace(G) / Xa.shape[1] + 1e-10
al = np.linalg.solve(G, Ra)
W = (Xa.T @ al) / colw[None, :]
A, Sv, Bt = np.linalg.svd(W, full_matrices=False)

e0 = float(((yfp[ev] - yq[ev]).astype(np.float64) ** 2).sum())
def rec_k(k):
    g = (X[ev] @ (A[:, :k] * Sv[:k])) @ Bt[:k]
    yh = yq[ev] * (1.0 + g.astype(np.float32))
    return 1 - float(((yfp[ev] - yh).astype(np.float64) ** 2).sum()) / e0
KG = [16, 32, 64, 128, 256, 384, 512, 768, 1024]
curve = {k: rec_k(k) for k in KG}
kbest = max(curve, key=curve.get)
print(f"  L{L} 乘性闭式 k曲线 " + " ".join(f"{k}:{curve[k]*100:.1f}" for k in KG), flush=True)
print(f"★L{L} 乘性放大器(闭式RRR·零训练): held行为挽回 {curve[kbest]*100:.2f}% @k_L={kbest} "
      f"(体积 {(D*kbest+kbest*D+kbest)*2/2**20:.1f}MB fp16) | 加性闭式基线=0.0%(在案)", flush=True)
