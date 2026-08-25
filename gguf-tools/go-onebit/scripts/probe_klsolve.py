#!/usr/bin/env python3
# probe_klsolve.py — 10分钟针(2026-08-16 用户令"快速验证我的真实设计"):
# 命题: 解算目标从层内MSE换成"终局KL行加权"(行为放大器口径), 解是否显著不同且KL口径更优。
# 纯解算对比, 零注入零模型改动。用 L00 zcache(INJ=2 产) + cal9 学生/FP logits 行权重。
# 判据: ①两解余弦(≈1=前提死) ②KL加权EV误差: KL解 vs MSE解(赢=前提立)。
import sys, os, struct
import numpy as np

R30 = "/Users/fodelf/ds4-main/gguf/go-onebit/r30"
zc = np.load(f"{R30}/en86/layers/zcache_L00.npz")
dH = zc["dH"].astype(np.float64)          # [S,D] 层残差目标
S, D = dH.shape

# 锚布局读 X0(层0输入) 与 FP logits 不需要 — 行KL从 student vs anchor logits 顶层取
def anchor_layer_fin(ap, L, ntok):
    hd = struct.unpack("<8I", open(ap, "rb").read(32))
    Sa, DIM, NL = hd[1], hd[3], hd[4]
    off = 40 + (L * Sa) * DIM * 4
    f = open(ap, "rb"); f.seek(off)
    return np.frombuffer(f.read(ntok * DIM * 4), dtype=np.float32).reshape(ntok, DIM).copy()

AP = f"{R30}/anchor_wtcal9_s2906.bin"
X = anchor_layer_fin(AP, 0, S).astype(np.float64)

# 行KL权重: student(/tmp/cal9_student.bin) vs 锚FP logits(锚文件尾部 logits 块)
def read_logits_rows(path, S, V):
    raw = open(path, "rb").read()
    n, v = struct.unpack("<ii", raw[:8])
    assert v == V, (v, V)
    return np.frombuffer(raw, dtype=np.float32, count=n*v, offset=8).reshape(n, v)

sys.path.insert(0, "/Users/fodelf/ds4-main/gguf-tools/go-onebit/scripts")
from anchor_metrics import read_anchor_logits
ref, meta = read_anchor_logits(AP)
Sa, V = meta["S"], meta["VOCAB"]
stu = read_logits_rows("/tmp/cal9_student.bin", Sa, V)

def kl_rows(ref, stu):
    r = ref - ref.max(1, keepdims=True); s = stu - stu.max(1, keepdims=True)
    pr = np.exp(r); pr /= pr.sum(1, keepdims=True)
    ls = s - np.log(np.exp(s).sum(1, keepdims=True))
    lr = r - np.log(np.exp(r).sum(1, keepdims=True))
    return (pr * (lr - ls)).sum(1)

w_kl = kl_rows(ref.astype(np.float64), stu.astype(np.float64))
w_kl = np.clip(w_kl, 1e-4, None); w_kl = (w_kl / w_kl.mean())

FR = [(0,975),(1141,1590),(1668,1899),(1971,2429),(2577,2836)]
EV = [(975,1141),(1590,1668),(1899,1971),(2429,2577),(2836,2906)]
tr = np.concatenate([np.arange(a,b) for a,b in FR])
ev = np.concatenate([np.arange(a,b) for a,b in EV])

def solve(weights):
    Xa = X[tr] * weights[tr][:, None] ** 0.5
    Ra = dH[tr] * weights[tr][:, None] ** 0.5
    G = Xa.T @ Xa; G[np.diag_indices_from(G)] += 3.0 * np.trace(G) / Xa.shape[0] + 1e-10
    return np.linalg.solve(G, Xa.T @ Ra)   # W: [D,D] 线性 z 映射(探针口径, 不截秩)

W_mse = solve(np.ones(S))
W_kl  = solve(w_kl)

cos = float((W_mse*W_kl).sum() / (np.linalg.norm(W_mse)*np.linalg.norm(W_kl) + 1e-12))
def ev_err(W, wts):
    E = dH[ev] - X[ev] @ W
    return float(((E**2).sum(1) * wts[ev]).sum() / ((dH[ev]**2).sum(1) * wts[ev]).sum())
r_mse_on_kl = ev_err(W_mse, w_kl)
r_kl_on_kl  = ev_err(W_kl,  w_kl)
r_mse_plain = ev_err(W_mse, np.ones(S))
r_kl_plain  = ev_err(W_kl,  np.ones(S))
print(f"两解余弦 = {cos:.4f}  (≈1 → 换目标无杠杆, 前提死)")
print(f"KL加权EV残差:  MSE解 {r_mse_on_kl:.4f}  vs  KL解 {r_kl_on_kl:.4f}  (KL解更小=前提立)")
print(f"平权EV残差:    MSE解 {r_mse_plain:.4f}  vs  KL解 {r_kl_plain:.4f}  (KL解的平权代价)")
