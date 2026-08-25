#!/usr/bin/env python3
# route_probe.py — 路由闭式侧车探针(2026-08-19 用户令"路由走体积和算法路线"):
# δlogits(x_q) = U·tanh(V₀ᵀ x_q / s), ELM 闭式(amp_probe3 同框架, 零训练),
# 目标 t = logits_fp(FP链) − logits_q(引擎在线, DS4_CAP_DIR 捕获), λ/k held 网格自选。
# 判决尺 = held 段 top6 命中率(sigmoid(logits)+bias 口径) 修正前→后 vs FP 真值(锚 ridx),
# 附 logits MSE 恢复%。线性 gate(x) 历史零增益 ≠ 非线性判决(v6.1 ELM 判例同构)。
# 用法: route_probe.py <hf> <cap_dir> <fp_anchor> <层号> [NFIT=1638]
import os, sys, struct
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from probe_layer_behavior import anchor_layer
from probe_behavior_spectrum import st_index, st_raw

hf, cap, ap, L = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
NFIT = int(sys.argv[5]) if len(sys.argv) > 5 else 1638
D, NE, NACT = 4096, 256, 6

def st_any(hf, wmap, name):
    dt, sh, raw = st_raw(hf, wmap, name)
    if dt in ("F32", "float32"):
        f = np.frombuffer(raw, dtype=np.float32)
    elif dt in ("BF16", "bfloat16"):
        f = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        f = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    return np.ascontiguousarray(f.reshape(sh))

wmap = st_index(hf)
Wfp = st_any(hf, wmap, f"layers.{L}.ffn.gate.weight")        # [256,4096]
try:
    bias = st_any(hf, wmap, f"layers.{L}.ffn.gate.bias").reshape(-1)  # [256]
except KeyError:
    bias = np.zeros(NE, dtype=np.float32)   # 浅层无 bias 条目

xq = np.fromfile(os.path.join(cap, f"raw_ffn_in_L{L}"), dtype=np.float16).astype(np.float32).reshape(-1, D)
lq = np.fromfile(os.path.join(cap, f"raw_route_logits_L{L}"), dtype=np.float16).astype(np.float32).reshape(-1, NE)
S = xq.shape[0]
Xfp, ridx_fp, _, _ = anchor_layer(ap, L, S)
lfp = Xfp.astype(np.float32) @ Wfp.T                          # FP 链 raw logits

def top6(logits):
    sc = 1.0 / (1.0 + np.exp(-logits)) + bias[None, :]
    return np.argsort(sc, axis=1)[:, -NACT:]

def hit(sel, ref):
    h = 0
    for i in range(sel.shape[0]):
        h += len(set(sel[i].tolist()) & set(ref[i].tolist()))
    return h / (sel.shape[0] * NACT) * 100.0

tr, ev = np.arange(0, NFIT), np.arange(NFIT, S)
san = hit(top6(lfp[ev]), ridx_fp[ev])                         # 口径 sanity: FP logits 复算 vs 锚
h0 = hit(top6(lq[ev]), ridx_fp[ev])                           # 修正前命中
t = (lfp - lq).astype(np.float64)
e0 = float((t[ev] ** 2).sum())

X = xq.astype(np.float64)
rms = np.sqrt((X[tr] ** 2).mean(1, keepdims=True))
r2 = np.random.RandomState(1)
Xa = np.vstack([X[tr], X[tr] + r2.randn(len(tr), D) * 0.04 * rms])
Ta = np.vstack([t[tr], t[tr]])
KMAX = 512
_, _, Vt = np.linalg.svd(Xa - Xa.mean(0), full_matrices=False)
CANDS = {"PCA": Vt[:KMAX].T, "rand": np.random.RandomState(7).randn(D, KMAX) / np.sqrt(D)}
scale = float(np.sqrt((Xa ** 2).mean()))

best = (h0, 0.0, None)   # (held命中, mse恢复, cfg)
for nm, V0 in CANDS.items():
    Za = np.tanh(Xa @ V0 / scale)
    Ze = np.tanh(X[ev] @ V0 / scale)
    ZtZ = Za.T @ Za; ZtT = Za.T @ Ta
    for lam in (30.0, 10.0, 3.0, 1.0):
        G = ZtZ.copy()
        G[np.diag_indices_from(G)] += lam * np.trace(ZtZ) / KMAX + 1e-10
        U = np.linalg.solve(G, ZtT)
        for k in (16, 32, 64, 128, 256, 512):
            d = Ze[:, :k] @ U[:k]
            rec = 1 - float(((t[ev] - d) ** 2).sum()) / e0
            h1 = hit(top6(lq[ev] + d.astype(np.float32)), ridx_fp[ev])
            if h1 > best[0]: best = (h1, rec, (nm, lam, k))
print(f"★L{L} 路由ELM: sanity(FP复算)={san:.1f}%  修正前top6={h0:.2f}%  "
      f"修正后top6={best[0]:.2f}% (+{best[0]-h0:.2f})  logitsMSE恢复={best[1]*100:.1f}% "
      f"@{best[2]}  S={S}", flush=True)
