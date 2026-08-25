#!/usr/bin/env python3
# route_solve.py — 路由闭式侧车逐层解算(2026-08-19 用户令"路由走体积和算法路线"):
# δlogits(x_q) = U·tanh(V₀ᵀ x_q / s), ELM 闭式零训练(route_probe.py 产品化)。
# 目标 t = logits_fp(FP链, HF router 离线复算) − logits_q(引擎在线, DS4_CAP_DIR 捕获)。
# 择优尺 = held top6 命中率(sigmoid+bias 口径, 锚 FP 路由为真值); 层闸 = 提升≤0.1 空记录。
# 产物: zrec_route_LXX.bin "zl.RTE"(type8): u32 k | f32 scale | u32 din=4096 | u32 dout=256
#       | fp16 z[k](全1) | U[256*k](hU[j*k+c]) | V[4096*k](hV[j*k+c])
# 用法: route_solve.py <hf> <cap_dir> <fp_anchor> <out_dir> <层号> [NFIT=1638]
import os, sys, struct, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from probe_layer_behavior import anchor_layer
from probe_behavior_spectrum import st_index, st_raw

hf, cap, ap, outd, L = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
NFIT = int(sys.argv[6]) if len(sys.argv) > 6 else 1638
D, NE, NACT = 4096, 256, 6
outp = os.path.join(outd, f"zrec_route_L{L:02d}.bin")
if os.path.exists(outp):
    print(f"★L{L} 路由侧车: 已存在, 跳过", flush=True); sys.exit(0)
t0 = time.time()

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
Wfp = st_any(hf, wmap, f"layers.{L}.ffn.gate.weight")
try:
    bias = st_any(hf, wmap, f"layers.{L}.ffn.gate.bias").reshape(-1)
except KeyError:
    bias = np.zeros(NE, dtype=np.float32)

xq = np.fromfile(os.path.join(cap, f"raw_ffn_in_L{L}"), dtype=np.float16).astype(np.float32).reshape(-1, D)
lq = np.fromfile(os.path.join(cap, f"raw_route_logits_L{L}"), dtype=np.float16).astype(np.float32).reshape(-1, NE)
S = xq.shape[0]
Xfp, ridx_fp, _, _ = anchor_layer(ap, L, S)
lfp = Xfp.astype(np.float32) @ Wfp.T

def top6(logits):
    sc = 1.0 / (1.0 + np.exp(-logits)) + bias[None, :]
    return np.argsort(sc, axis=1)[:, -NACT:]

def hit(sel, ref):
    h = 0
    for i in range(sel.shape[0]):
        h += len(set(sel[i].tolist()) & set(ref[i].tolist()))
    return h / (sel.shape[0] * NACT) * 100.0

tr, ev = np.arange(0, NFIT), np.arange(NFIT, S)
san = hit(top6(lfp[ev]), ridx_fp[ev])
h0 = hit(top6(lq[ev]), ridx_fp[ev])
if san < 80.0:
    open(outp, 'wb').write(b"")
    print(f"★L{L} 路由侧车: sanity={san:.1f}%<80 口径不符, 空记录跳过 | {time.time()-t0:.0f}s", flush=True)
    sys.exit(0)

t = (lfp - lq).astype(np.float64)
X = xq.astype(np.float64)
rms = np.sqrt((X[tr] ** 2).mean(1, keepdims=True))
r2 = np.random.RandomState(1)
Xa = np.vstack([X[tr], X[tr] + r2.randn(len(tr), D) * 0.04 * rms])
Ta = np.vstack([t[tr], t[tr]])
KMAX = 512
_, _, Vt = np.linalg.svd((Xa - Xa.mean(0)).astype(np.float32), full_matrices=False)
CANDS = {"PCA": Vt[:KMAX].T.astype(np.float64),
         "rand": np.random.RandomState(7).randn(D, KMAX) / np.sqrt(D)}
scale = float(np.sqrt((Xa ** 2).mean()))

best = (h0 + 0.1, None)   # 层闸: held top6 提升 ≤0.1 → 空记录
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
            h1 = hit(top6(lq[ev] + d.astype(np.float32)), ridx_fp[ev])
            if h1 > best[0]: best = (h1, (nm, lam, k, U[:k].copy()))

def hdr(nm, psz):
    h = bytearray(116); h[0:len(nm)] = nm.encode()
    struct.pack_into('<Q', h, 88, psz); struct.pack_into('<i', h, 112, 1)
    return bytes(h)

if best[1] is None:
    open(outp, 'wb').write(b"")
    print(f"★L{L} 路由侧车: held top6 {h0:.2f}% 无提升 → 空记录(层闸) | {time.time()-t0:.0f}s", flush=True)
else:
    nm, lam, k, U = best[1]
    Ueng = np.ascontiguousarray(U.T.astype(np.float16))                # [256][k]
    Veng = np.ascontiguousarray(CANDS[nm][:, :k].astype(np.float16))   # [4096][k]
    z1 = np.ones(k, dtype=np.float16)
    pay = struct.pack('<IfII', k, scale, D, NE) + z1.tobytes() + Ueng.tobytes() + Veng.tobytes()
    open(outp, 'wb').write(hdr("zl.RTE", len(pay)) + pay)
    print(f"★L{L} 路由侧车: top6 {h0:.2f}→{best[0]:.2f}% (+{best[0]-h0:.2f}) @V₀={nm} λ={lam} k={k} "
          f"体积 {len(pay)/2**20:.1f}MB | {time.time()-t0:.0f}s", flush=True)
