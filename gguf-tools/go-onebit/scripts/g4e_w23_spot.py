#!/usr/bin/env python3
"""g4e_w23_spot.py — w3/w2 抽查针: 验证 g4d w1 口径外推(2026-07-28)。
g4d 全层扫是 w1; w3 同形同输入(共享 X 基底, 预期同 w1); w2 输入=专家门控中间态 Z
(per-expert 流形!) 且历史抗码本 — 是 5.02 GiB 账的最大敞口。
抽 3 层(L5 贵中段 / L20 便宜中段 / L38 便宜深尾) × 2hot+2cold:
  w3: 锚=vq8(生产), 流形档 r384(0.379)/r768(0.754), X 基底层共享(g4d 同款)
  w2: 锚=signref 行scale(生产 w2 路), 流形档 r192(0.379)/r384(0.754) [cols=2048]
      w2sh: 基底=4专家池化 Z 的 SVD(体积可行架构) / w2pe: per-expert Z 基底(天花板参照,
      基底自重 ~2×payload, 仅作诊断)
用法(M1): g4e_w23_spot.py [--out rpt]
"""
import os, sys, time, argparse, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "quant"))
from g1a_lever_probe import open_shard_for_layer
from g1b_stack_probe import metrics, cold_C0

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", file=sys.stderr, flush=True)

RNG = np.random.default_rng(0x5EED)

def style_of(name):
    n = name[5:] if name.startswith("held_") else name
    return "method" if n == "method" else n.split(".")[1]

def row_act_scale(Wq, W, Xf):
    p = Xf @ Wq.T; y = Xf @ W.T
    sp2 = (p*p).sum(0); spy = (p*y).sum(0)
    lam = np.maximum(sp2/(len(Xf)+1.0), 1e-3*sp2+1e-9)
    g = np.clip((spy+lam*1.0)/(sp2+lam), 0.25, 4.0)
    return Wq * g[:, None].astype(np.float32)

def _assign(V, C, bs=16384):
    out = np.empty(len(V), np.int64); c2 = (C*C).sum(1)
    for i in range(0, len(V), bs):
        d = V[i:i+bs] @ C.T; d *= -2.0; d += c2[None, :]
        out[i:i+bs] = d.argmin(1)
    return out

def vq8(W, Xf):
    V = W.reshape(-1, 8)
    sub = V[RNG.choice(len(V), min(len(V), 200000), replace=False)]
    C = sub[RNG.choice(len(sub), 256, replace=False)].copy()
    for _ in range(8):
        a = _assign(sub, C)
        for c in range(256):
            m = a == c
            if m.any(): C[c] = sub[m].mean(0)
    return row_act_scale(C[_assign(V, C)].reshape(W.shape).astype(np.float32), W, Xf)

def coeff_q4(A):
    s = np.abs(A).max(1)/7.0 + 1e-20
    return np.round(A/s[:, None])*s[:, None]

def mrung(W, Xf, Xe, Ye, Vt, r):
    B = Vt[:r].T
    Wq = row_act_scale((coeff_q4(W @ B) @ B.T).astype(np.float32), W, Xf)
    return metrics(Ye, Xe @ Wq.T)[1]

def silu(z): return z/(1+np.exp(-np.clip(z, -30, 30)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="")
    ap.add_argument("--cells", default="/tmp/g4c_cells.tsv")
    ap.add_argument("--hf", default="/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base")
    ap.add_argument("--hot-table", default=os.path.join(HERE, "..", "corpus", "prog_active_top64.txt"))
    a = ap.parse_args()
    hot = {}
    for ln in open(a.hot_table):
        p = ln.split(":"); hot[int(p[0][1:])] = [int(x) for x in p[1].split()]
    out = []
    for L in (5, 20, 38):
        X = np.load(f"/tmp/g4c_x_L{L:02d}.npy").astype(np.float32)
        st = np.empty(len(X), dtype=object)
        for ln in open(a.cells):
            name, _, s, e2 = ln.split("\t")
            st[int(s):int(e2)] = style_of(name)
        code = np.isin(st, ("repo", "tut", "test", "contract")); idx = np.arange(len(X))
        Xf, Xe = X[code & (idx % 2 == 0)], X[code & (idx % 2 == 1)]
        _, _, VtX = np.linalg.svd(Xf, full_matrices=False)
        cold_all = [e for e in range(256) if e not in set(hot[L])]
        picks = hot[L][:2] + [cold_all[0], cold_all[-1]]
        sh = open_shard_for_layer(a.hf, L)
        W1 = {e: sh.expert_w(L, e, "w1") for e in picks}
        W3 = {e: sh.expert_w(L, e, "w3") for e in picks}
        Zf = {e: (silu(Xf @ W1[e].T)*(Xf @ W3[e].T)).astype(np.float32) for e in picks}
        Ze = {e: (silu(Xe @ W1[e].T)*(Xe @ W3[e].T)).astype(np.float32) for e in picks}
        _, _, VtZsh = np.linalg.svd(np.concatenate([Zf[e][::4] for e in picks]), full_matrices=False)
        t0 = time.time()
        for e in picks:
            # w3: X 基底(层共享)
            W = W3[e]; Ye = Xe @ W.T
            anc = metrics(Ye, Xe @ vq8(W, Xf).T)[1]
            out.append((f"L{L:02d}", "w3", f"e{e}", f"{anc:.4f}",
                        f"{mrung(W, Xf, Xe, Ye, VtX, 384):.4f}", f"{mrung(W, Xf, Xe, Ye, VtX, 768):.4f}"))
            # w2: Z 输入(per-expert 流形), 锚=signref
            W = sh.expert_w(L, e, "w2"); Ye2 = Ze[e] @ W.T
            anc2 = metrics(Ye2, Ze[e] @ cold_C0(W, Zf[e])[0].T)[1]
            _, _, VtZpe = np.linalg.svd(Zf[e], full_matrices=False)
            r1, r2 = 192, 384                        # cols=2048 → 0.379/0.754 bpw
            csh = [f"{mrung(W, Zf[e], Ze[e], Ye2, VtZsh, r):.4f}" for r in (r1, r2)]
            cpe = [f"{mrung(W, Zf[e], Ze[e], Ye2, VtZpe, r):.4f}" for r in (r1, r2)]
            out.append((f"L{L:02d}", "w2sh", f"e{e}", f"{anc2:.4f}", csh[0], csh[1]))
            out.append((f"L{L:02d}", "w2pe", f"e{e}", f"{anc2:.4f}", cpe[0], cpe[1]))
            log(f"L{L} e{e} 完成")
        log(f"L{L} 全部 {time.time()-t0:.0f}s")
    print("\nlayer kind expert anchor_cos r@0.379 r@0.754")
    for r in out: print(" ".join(r))
    agg = collections.defaultdict(lambda: [[], [], []])
    for r in out:
        agg[(r[0], r[1])][0].append(float(r[3])); agg[(r[0], r[1])][1].append(float(r[4])); agg[(r[0], r[1])][2].append(float(r[5]))
    print("\n== 汇总(cos 均值: anchor / 0.379 / 0.754) ==")
    for k in sorted(agg):
        v = agg[k]
        print(f"{k[0]} {k[1]:5s} anchor={np.mean(v[0]):.4f} r379={np.mean(v[1]):.4f} r754={np.mean(v[2]):.4f}")
    if a.out:
        with open(a.out, "w") as f:
            f.write("layer kind expert anchor_cos r_0379 r_0754\n")
            for r in out: f.write(" ".join(r) + "\n")

if __name__ == "__main__":
    main()
