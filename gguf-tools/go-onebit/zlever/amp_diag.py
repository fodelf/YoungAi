#!/usr/bin/env python3
# amp_diag.py — 放大器闸层机制审计探针(2026-08-20 用户令"闸就该停了, 肯定有问题")。
# 单层 zcache 上量化四件事, 把 held≈0 拆到根:
#  ① 残差能量按 |y_q| 分布: 乘性 ŷ=y_q(1+g) 只能修 |y_q| 有量的坐标
#  ② oracle_form: g 不设限时形式天花板(只有 y_q==0 坐标不可修)
#  ③ oracle_eps: ratio 目标 R=dH·yq/(yq²+eps²) 完美拟合时的天花板(eps 阻尼的结构损失)
#  ④ 同特征 z=tanh(x·V₀/s): 乘性 ELM held vs 加性 ELM held —— 形式受限 vs 求解器病
# 用法: DS4_ZL_XANCHOR=链态锚 amp_diag.py <FP锚> <zcache_LXX.npz> [NFIT=1638]
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
XAP = os.getenv("DS4_ZL_XANCHOR")
X0, _, _, _ = anchor_layer(XAP if XAP else ap, L, S)

yq = np.zeros((S, D), dtype=np.float32)
np.add.at(yq, prow, pw[:, None] * pYQ)
yfp = yq + dH
tr, ev = np.arange(0, NFIT), np.arange(NFIT, S)
e0 = float(((yfp[ev] - yq[ev]).astype(np.float64) ** 2).sum())

# ① 残差能量按 |y_q| 分位分布(ev 段)
a = np.abs(yq[ev]).ravel(); e = (dH[ev].astype(np.float64) ** 2).ravel()
qs = np.quantile(a, [0.10, 0.25, 0.50, 0.75])
et = e.sum()
frac = [float(e[a <= q].sum() / et) for q in qs]
print(f"L{L} ① dH²能量落在 |y_q| 最小10/25/50/75分位内的占比: "
      f"{frac[0]*100:.1f}% {frac[1]*100:.1f}% {frac[2]*100:.1f}% {frac[3]*100:.1f}%")

# ②③ 形式/目标天花板
eps = np.sqrt((yq[tr] ** 2).mean(0)) * 1e-2 + 1e-12
res_eps = dH[ev] * (eps[None, :] ** 2) / (yq[ev] ** 2 + eps[None, :] ** 2)
oracle_eps = 1 - float((res_eps.astype(np.float64) ** 2).sum()) / e0
zero_mask = (yq[ev] == 0)
oracle_form = 1 - float((dH[ev][zero_mask].astype(np.float64) ** 2).sum()) / e0
print(f"L{L} ② oracle_form(乘性形式无限容量天花板) = {oracle_form*100:.2f}%")
print(f"L{L} ③ oracle_eps (ratio目标完美拟合天花板)  = {oracle_eps*100:.2f}%")

# ④ 同特征 乘性 vs 加性 ELM(与 amp_solve 同式: PCA+dither, λ 网格, KMAX=512)
KMAX = 512
try:
    import cupy as _cp
    _GPU = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _cp = None; _GPU = False
xp = _cp if _GPU else np
_A = (lambda x: xp.asarray(x)) if _GPU else (lambda x: x)
X = X0.astype(np.float64)
R = (dH * yq / (yq ** 2 + eps[None, :] ** 2)).astype(np.float64)
colw = np.sqrt(R[tr].var(0) + 1e-12) * np.sqrt((yq[tr] ** 2).mean(0) + 1e-12)
rms = np.sqrt((X[tr] ** 2).mean(1, keepdims=True))
r2 = np.random.RandomState(1)
Xa = np.vstack([X[tr], X[tr] + r2.randn(len(tr), D) * 0.04 * rms])
Xa_ = _A(Xa); Xev_ = _A(X[ev]); colw_ = _A(colw)
yq_ev = _A(yq[ev]); yfp_ev = _A(yfp[ev])
Ra_ = _A(np.vstack([R[tr] * colw, R[tr] * colw]))
dcolw = np.sqrt(dH[tr].var(0) + 1e-12)                       # 加性目标列权(同风格)
Da_ = _A(np.vstack([dH[tr] / dcolw, dH[tr] / dcolw]).astype(np.float64))
dcolw_ = _A(dcolw)
_, _, Vt = xp.linalg.svd(Xa_ - Xa_.mean(0), full_matrices=False)
V0 = Vt[:KMAX].T
scale = float(xp.sqrt((Xa_ ** 2).mean()))
Za = xp.tanh(Xa_ @ V0 / scale); Ze = xp.tanh(Xev_ @ V0 / scale)
ZtZ = Za.T @ Za
_diag = xp.arange(KMAX)
bm = ba = 0.0
for lam in (1000.0, 300.0, 100.0, 30.0, 10.0, 3.0):
    G = ZtZ.copy(); G[_diag, _diag] += lam * float(xp.trace(ZtZ)) / KMAX + 1e-10
    Um = xp.linalg.solve(G, Za.T @ Ra_)
    Ua = xp.linalg.solve(G, Za.T @ Da_)
    for k in (128, 256, 384, 512):
        g = (Ze[:, :k] @ (Um[:k] / colw_[None, :])).astype(xp.float32)
        rec_m = 1 - float(((yfp_ev - yq_ev * (1.0 + g)).astype(xp.float64) ** 2).sum()) / e0
        add = (Ze[:, :k] @ (Ua[:k] * dcolw_[None, :])).astype(xp.float32)
        rec_a = 1 - float(((yfp_ev - (yq_ev + add)).astype(xp.float64) ** 2).sum()) / e0
        bm = max(bm, rec_m); ba = max(ba, rec_a)
print(f"L{L} ④ 同特征 held: 乘性 ELM {bm*100:.2f}%  vs  加性 ELM {ba*100:.2f}%")
print(f"L{L} 判读: 加性>>乘性 且 oracle 高 → 形式受限(残差在小|y_q|坐标, 乘性无从下手);"
      f" oracle 低 → ratio 目标 eps 结构损失; 两者都高而双 ELM 低 → 特征/求解器病")
