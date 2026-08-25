#!/usr/bin/env python3
# probe_condz.py — 条件化针(2026-08-16 用户令"试一下"):
# 问题: 用 student 自身信号(本层激活专家集 pe/pw)做条件, dH 的可预测比例能否显著抬升?
# 对照: ①全局线性(昨针口径, L00≈4-8%) ②student路由条件化(部署可得) ③FP路由条件化(神谕上界)
# 纯离线解算, 零注入。数据: zcache_L{L}.npz(dH/prow/pe/pw/pDY) + 锚(fin/ridx/rw)。
# 用法: probe_condz.py <L>
import sys, struct
import numpy as np

L = int(sys.argv[1])
R30 = "/Users/fodelf/ds4-main/gguf/go-onebit/r30"
AP = f"{R30}/anchor_wtcal9_s2906.bin"
NE = 256

hd = struct.unpack("<8I", open(AP, "rb").read(32))
_, S, HCM, DIM, NL, V, NACT, _ = hd
FR = [(0,975),(1141,1590),(1668,1899),(1971,2429),(2577,2836)]
EVR = [(975,1141),(1590,1668),(1899,1971),(2429,2577),(2836,2906)]
tr = np.concatenate([np.arange(a,b) for a,b in FR])
ev = np.concatenate([np.arange(a,b) for a,b in EVR])

z = np.load(f"{R30}/en86/layers/zcache_L{L:02d}.npz")
dH = z["dH"].astype(np.float64)
prow, pe, pw = z["prow"], z["pe"], z["pw"].astype(np.float64)
pDY = z["pDY"].astype(np.float64)

def afin(l):
    f = open(AP, "rb"); f.seek(40 + (np.int64(l)*S)*DIM*4)
    return np.frombuffer(f.read(S*DIM*4), dtype=np.float32).reshape(S, DIM).astype(np.float64)
def aroute(l):
    base = 40 + np.int64(NL)*S*DIM*4
    f = open(AP, "rb"); f.seek(base + np.int64(l)*S*NACT*4)
    ri = np.frombuffer(f.read(S*NACT*4), dtype=np.int32).reshape(S, NACT).copy()
    f.seek(base + np.int64(NL)*S*NACT*4 + np.int64(l)*S*NACT*4)
    rw = np.frombuffer(f.read(S*NACT*4), dtype=np.float32).reshape(S, NACT).astype(np.float64)
    return ri, rw

X = afin(L)
fri, frw = aroute(L)

den_ev = float((dH[ev]**2).sum())
def evfrac(pred):                       # EV 上被解释的能量比例
    return 1.0 - float(((dH[ev]-pred[ev])**2).sum())/den_ev

# ① 全局线性(昨针口径)
Xa, Ra = X[tr], dH[tr]
G = Xa.T @ Xa; G[np.diag_indices_from(G)] += 3.0*np.trace(G)/Xa.shape[0] + 1e-10
W = np.linalg.solve(G, Xa.T @ Ra)
lin_pred = X @ W

# 门矩阵: A[S,NE] 稀疏门权(student自身 / FP神谕)
def gate_mat(rows_idx, e_ids, wts):
    A = np.zeros((S, NE))
    np.add.at(A, (rows_idx, e_ids), wts)
    return A
A_st = gate_mat(prow, pe, pw)
A_fp = gate_mat(np.repeat(np.arange(S), NACT), fri.ravel(), frw.ravel())

def route_bias(A, resid):               # 残差上闭式解 per-expert 修正向量 C[NE,D]
    At = A[tr]
    Gg = At.T @ At; Gg[np.diag_indices_from(Gg)] += 1e-3*np.trace(Gg)/NE + 1e-10
    C = np.linalg.solve(Gg, At.T @ resid[tr])
    return A @ C

rb_st  = route_bias(A_st, dH)                       # 路由条件单独
rb_fp  = route_bias(A_fp, dH)
seq_st = lin_pred + route_bias(A_st, dH - lin_pred) # 线性之上再叠路由条件
seq_fp = lin_pred + route_bias(A_fp, dH - lin_pred)

# 路由一致率(student 6选 vs FP 6选, 集合交/6)
agree = np.mean([len(set(pe[prow==r]) & set(fri[r]))/NACT for r in range(0,S,7)])

# dH 里 MoE 份额: pick级 pw·pDY 重建 vs dH 的逐行余弦
M = np.zeros_like(dH)
np.add.at(M, prow, pw[:,None]*pDY)
cs = (M*dH).sum(1)/(np.linalg.norm(M,axis=1)*np.linalg.norm(dH,axis=1)+1e-12)

print(f"L{L:02d}  S={S}  路由一致率(student vs FP)={agree:.3f}  dH~MoE重建余弦(中位)={np.median(cs):.3f}")
print(f"EV解释比例:  全局线性        = {evfrac(lin_pred)*100:6.2f}%")
print(f"             路由条件(student)= {evfrac(rb_st)*100:6.2f}%   线性+路由(student)= {evfrac(seq_st)*100:6.2f}%")
print(f"             路由条件(FP神谕) = {evfrac(rb_fp)*100:6.2f}%   线性+路由(FP神谕) = {evfrac(seq_fp)*100:6.2f}%")
