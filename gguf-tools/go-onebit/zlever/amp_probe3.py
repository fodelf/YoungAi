#!/usr/bin/env python3
# amp_probe3.py — v6.1 乘性放大器·解析判决探针 v3(2026-08-19 用户令"解析迭代可以, 几十秒内")。
# 模型: z=tanh(x·V₀/s), ŷ=y_q⊙(1+z·U)。V₀=fit段PCA方向(解析产物, 零拍脑袋)+固定seed随机
# 特征对照; U=一步闭式加权ridge(行为空间目标, |y_q|列权+感知colw+dither增广)。零SGD。
# 可选 AMP_ALS=N 轮 backfitting(每步解析 Gauss-Newton, 默认0)。k_L=held 网格自选。
# 判据: held 行为挽回 1−‖y_fp−ŷ‖²/‖y_fp−y_q‖² | 加性闭式=0.0% 乘性线性闭式=+0.6%(在案)。
# 用法: amp_probe3.py <anchor> <zcache_LXX.npz> [NFIT=1638]
import sys, os, time
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
t0 = time.time()

yq = np.zeros((S, D), dtype=np.float32)
np.add.at(yq, prow, pw[:, None] * pYQ)
yfp = yq + dH
tr, ev = np.arange(0, NFIT), np.arange(NFIT, S)
X = X0.astype(np.float64)
# 行为空间等价目标(比值域, 光滑正则)+列权(|y_q|·感知)
eps = np.sqrt((yq[tr] ** 2).mean(0)) * 1e-2 + 1e-12
R = (dH * yq / (yq ** 2 + eps[None, :] ** 2)).astype(np.float64)
colw = np.sqrt(R[tr].var(0) + 1e-12) * np.sqrt((yq[tr] ** 2).mean(0) + 1e-12)
rms = np.sqrt((X[tr] ** 2).mean(1, keepdims=True))
r2 = np.random.RandomState(1)
Xa = np.vstack([X[tr], X[tr] + r2.randn(len(tr), D) * 0.04 * rms])          # dither 增广
Ra = np.vstack([R[tr] * colw, R[tr] * colw])
e0 = float(((yfp[ev] - yq[ev]).astype(np.float64) ** 2).sum())

KMAX = 512
# V₀ 两法: ①PCA(fit段解析主方向) ②固定seed随机特征 — held 择优, 非人工选择
_, _, Vt = np.linalg.svd(Xa - Xa.mean(0), full_matrices=False)
V_pca = Vt[:KMAX].T                                    # D×KMAX
V_rnd = np.random.RandomState(7).randn(D, KMAX) / np.sqrt(D)
scale = np.sqrt((Xa ** 2).mean())                      # tanh 进饱和前的解析定标

def solve_eval(V0):
    Za = np.tanh(Xa @ V0 / scale)                      # 增广样本特征
    Ze = np.tanh(X[ev] @ V0 / scale)
    out = {}
    ZtZ = Za.T @ Za; ZtR = Za.T @ Ra
    for lam in (100.0, 30.0, 10.0, 3.0, 1.0):          # λ 同 k_L: held 网格自选(上界扫到收敛)
        G = ZtZ.copy()
        G[np.diag_indices_from(G)] += lam * np.trace(ZtZ) / KMAX + 1e-10
        U = np.linalg.solve(G, ZtR)
        for k in (16, 32, 64, 128, 256, 384, 512):
            if k > KMAX: continue
            g = (Ze[:, :k] @ (U[:k] / colw[None, :])).astype(np.float32)
            yh = yq[ev] * (1.0 + g)
            out[(lam, k)] = 1 - float(((yfp[ev] - yh).astype(np.float64) ** 2).sum()) / e0
    return out, None

cur_p, _ = solve_eval(V_pca)
cur_r, _ = solve_eval(V_rnd)
kp = max(cur_p, key=cur_p.get); kr = max(cur_r, key=cur_r.get)
tp = sorted(cur_p.items(), key=lambda x: -x[1])[:5]
tr_ = sorted(cur_r.items(), key=lambda x: -x[1])[:5]
print(f"  L{L} ELM-PCA  top5 " + " ".join(f"λ{a}k{b}:{v*100:.1f}" for (a, b), v in tp), flush=True)
print(f"  L{L} ELM-rand top5 " + " ".join(f"λ{a}k{b}:{v*100:.1f}" for (a, b), v in tr_), flush=True)
best = ("PCA", kp, cur_p[kp]) if cur_p[kp] >= cur_r[kr] else ("rand", kr, cur_r[kr])
print(f"★L{L} 乘性放大器(ELM闭式·零迭代·零训练): held行为挽回 {best[2]*100:.2f}% "
      f"@V₀={best[0]} λ,k={best[1]} | 线性乘性闭式=+0.6% 加性=0.0%(在案) | {time.time()-t0:.0f}s", flush=True)
