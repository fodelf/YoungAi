#!/usr/bin/env python3
"""d2_probe.py — D2 单层探针: 每专家系数 g_e (静态版, 动态 z_e(x) 的第一级).
模型: y ≈ Σ_pick w_e·g_e·ŷ_e, 对 g∈R^256 全局 LS (256×256 正规方程, 闭式).
对照三档: baseline(g=1) / D0 全局标量 g_L / D2 每专家 g_e — TR/TE 各报 cos + rel(幅度敏感).
用法: python3 d2_probe.py --gguf v3.gguf --cap /tmp/cap_v3prep --layer 8 [--swlim 10.0]
"""
import os, sys, argparse
import numpy as np

def tok_cos(A, B):
    return float(np.mean(np.sum(A*B,1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+1e-20)))
def rel(A, B):  # ||err||/||ref|| 均值 — 幅度敏感
    return float(np.mean(np.linalg.norm(A-B,axis=1)/(np.linalg.norm(A,axis=1)+1e-20)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--swlim", type=float, default=10.0)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--heldout-frac", type=float, default=0.17)
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gptq1_rewrite import parse_gguf, read_expert_blocks, blocks_dequant, deq, swiglu
    L = args.layer

    X = np.load(f"{args.cap}/ffn_in_L{L}.npy").astype(np.float32)
    route = np.load(f"{args.cap}/route_L{L}.npy")
    rw = np.load(f"{args.cap}/route_w_L{L}.npy").astype(np.float32)
    Yt = np.load(f"{args.cap}/routed_L{L}.npy").astype(np.float32)
    n, K = route.shape
    D = Yt.shape[1]

    f = open(args.gguf, "rb")
    tens, data0 = parse_gguf(f)
    T = {}
    for kind in ("gate","up","down"):
        ne, ty, off = tens[f"blk.{L}.ffn_{kind}_exps.weight"]
        assert ty == 40
        T[kind] = (int(ne[0]), int(ne[1]), off)
    def ew(kind, e):
        cols, rows, toff = T[kind]
        raw,_ = read_expert_blocks(f, data0, toff, e, rows, cols)
        s,B = blocks_dequant(raw, cols)
        return deq(s,B)

    # 每 (token,pick) 学生输出, f16 暂存 [n,K,D] ≈ 1.3GB
    C = np.zeros((n, K, D), dtype=np.float16)
    for e in range(256):
        sel = np.nonzero(route == e)
        if sel[0].size == 0: continue
        Xt = X[sel[0]]
        w1, w3, w2 = ew("gate",e), ew("up",e), ew("down",e)
        h = swiglu(Xt @ w1.T, Xt @ w3.T, args.swlim)
        C[sel[0], sel[1]] = (h @ w2.T).astype(np.float16)
        if e % 64 == 0: print(f"L{L} contrib e{e}/256", flush=True)

    # heldout 按 chunk 边界
    nch = n // args.chunk; nte_ch = max(1, int(args.heldout_frac*nch+0.5))
    pool = (nch-nte_ch)*args.chunk
    tr = slice(0, pool); te = slice(pool, n)

    W = rw[..., None]                                   # [n,K,1]
    def assemble(g):                                    # g[256] -> y' [n,D]
        ge = g[route]                                   # [n,K]
        return np.einsum("nk,nk,nkd->nd", rw, ge, C.astype(np.float32), optimize=True)
    Yb = np.einsum("nk,nkd->nd", rw, C.astype(np.float32), optimize=True)   # baseline g=1

    # D0 全局标量 (train 拟合)
    gl = float(np.sum(Yt[tr]*Yb[tr]) / (np.sum(Yb[tr]*Yb[tr])+1e-20))

    # D2 每专家 g_e: A g = b, A_ee' = Σ_t w_e w_e' <c_e,c_e'>, b_e = Σ_t w_e <y*, c_e>
    A = np.zeros((256,256), dtype=np.float64); b = np.zeros(256, dtype=np.float64)
    step = 4096
    for i0 in range(0, pool, step):
        i1 = min(i0+step, pool)
        Cb = C[i0:i1].astype(np.float32) * W[i0:i1]     # [m,K,D] 已含 w
        G = np.einsum("mkd,mjd->mkj", Cb, Cb)           # [m,K,K]
        yv = np.einsum("mkd,md->mk", Cb, Yt[i0:i1])     # [m,K]
        ids = route[i0:i1]
        for k in range(K):
            np.add.at(b, ids[:,k], yv[:,k])
            for j in range(K):
                np.add.at(A, (ids[:,k], ids[:,j]), G[:,k,j])
    A[np.diag_indices(256)] += 1e-3*np.trace(A)/256
    ge = np.linalg.solve(A, b).astype(np.float32)
    print(f"L{L} g_e: mean={ge.mean():.3f} p10={np.percentile(ge,10):.3f} p90={np.percentile(ge,90):.3f}", flush=True)

    Y0 = Yb; Yg = Yb*gl; Ye = assemble(ge)
    for name, Yp in (("baseline g=1", Y0), (f"D0 global g={gl:.3f}", Yg), ("D2 per-expert g_e", Ye)):
        print(f"L{L} {name:22s} TR cos={tok_cos(Yt[tr],Yp[tr]):.4f} rel={rel(Yt[tr],Yp[tr]):.4f} | "
              f"TE cos={tok_cos(Yt[te],Yp[te]):.4f} rel={rel(Yt[te],Yp[te]):.4f}", flush=True)

if __name__ == "__main__":
    main()
