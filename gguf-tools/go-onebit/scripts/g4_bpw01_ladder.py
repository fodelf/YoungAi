#!/usr/bin/env python3
"""g4_bpw01_ladder.py — 0.1 bpw 可能性梯子短针(2026-07-28)。
背景: v4bf 定版(动态层体积=历史最好)后, 用户令探索 0.1 bpw。物理: bpw=log2(nc)/dim;
dim 须整除 cols=4096 → 0.1 档取 dim64/nc64 = 6/64 = 0.094 bpw。码本自重在 0.1 档不可
忽略 → 层共享码本是唯一诚实架构(per-expert dim64 自重 +0.031bpw=33%超编; 层 256 专家
共享 → ~0.0001)。奖品: 专家 0.1bpw → 全模型专家 ~1GiB → 全驻留 RAM → 消灭 SSD 流墙。

梯子(held-out 偶/奇分半, 真实激活, w1):
  prod_sign      1.0625  生产冷路基线(cold_C0)
  vq{dim}[c{nc}] per-expert 码本 (8,256)/(16,256)/(32,256)/(64,256)/(64,64) → 1.0/0.5/0.25/0.125/0.094
  shvq{...}      同上但码本跨采样专家池共享(层共享架构模拟) (16,256)/(64,256)/(64,64)
  signvq64       0.125   二值码本 VQ(sign 矩阵 8× 压缩)+行 act scale — 方向信息存活率针
  randvq64       0.125   随机高斯码本(免 kmeans)对照 — 机会地板
判据: held-out relF(越小越好)/cos; bpw 均含行乘子, 码本自重单列显式。
用法(M1): g4_bpw01_ladder.py --layer L --x /tmp/xr_x_L{L:02d}.npy [--nh 2 --nc 2] [--out rpt]
"""
import os, sys, time, argparse, math, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "quant"))
from g1a_lever_probe import open_shard_for_layer
from g1b_stack_probe import metrics, cold_C0

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", file=sys.stderr, flush=True)

RNG = np.random.default_rng(0x5EED)

def _assign(V, C, bs=16384):
    out = np.empty(len(V), np.int64); c2 = (C*C).sum(1)
    for i in range(0, len(V), bs):
        d = V[i:i+bs] @ C.T; d *= -2.0; d += c2[None, :]
        out[i:i+bs] = d.argmin(1)
    return out

def kmeans(sub, nc, iters=8, binary=False):
    C = sub[RNG.choice(len(sub), nc, replace=False)].copy()
    for _ in range(iters):
        a = _assign(sub, C)
        for c in range(nc):
            m = a == c
            if m.any(): C[c] = sub[m].mean(0)
        if binary:
            C = np.where(C >= 0, 1.0, -1.0).astype(np.float32)
    return C

def row_act_scale(Wq, W, Xf):
    """行级闭式 ridge act 乘子(g1b 同款 1 DOF)。"""
    p = Xf @ Wq.T; y = Xf @ W.T
    sp2 = (p*p).sum(0); spy = (p*y).sum(0)
    lam = np.maximum(sp2/(len(Xf)+1.0), 1e-3*sp2+1e-9)
    g = np.clip((spy+lam*1.0)/(sp2+lam), 0.25, 4.0)
    return Wq * g[:, None].astype(np.float32)

def vq_apply(W, Xf, C, dim):
    V = W.reshape(-1, dim)
    Wq = C[_assign(V, C)].reshape(W.shape).astype(np.float32)
    return row_act_scale(Wq, W, Xf)

def subsample(W, dim, cap=200000):
    V = W.reshape(-1, dim)
    if len(V) <= cap: return V.copy()
    return V[RNG.choice(len(V), cap, replace=False)]

def bpw_of(dim, nc, rows, cols, shared):
    idx = math.log2(nc)/dim                      # 索引流
    cb = nc*dim*16.0/(rows*cols)                 # f16 码本自重(per-expert 口径)
    gr = 16.0/cols                               # 行乘子 f16
    return idx+gr+(0.0 if shared else cb), cb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--x", required=True)
    ap.add_argument("--hf", default="/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base")
    ap.add_argument("--hot-table", default=os.path.join(HERE, "..", "corpus", "prog_active_top64.txt"))
    ap.add_argument("--nh", type=int, default=2); ap.add_argument("--nc", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args(); L = a.layer
    X = np.load(a.x).astype(np.float32)
    Xf, Xe = X[0::2], X[1::2]                    # 偶=fit 奇=held-out(g1b 纠偏口径)
    hot = {}
    for ln in open(a.hot_table):
        p = ln.split(":"); hot[int(p[0][1:])] = [int(x) for x in p[1].split()]
    cold_all = [e for e in range(256) if e not in set(hot[L])]
    picks = [(e, "hot") for e in hot[L][:a.nh]] + \
            [(e, "cold") for e in (cold_all[0], cold_all[-1])[:a.nc]]
    sh = open_shard_for_layer(a.hf, L)
    log(f"G4 L{L}: picks={picks} X={X.shape}")

    Ws = {e: sh.expert_w(L, e, "w1") for e, _ in picks}
    rows, cols = next(iter(Ws.values())).shape
    LADDER = [(8, 256), (16, 256), (32, 256), (64, 256), (64, 64)]
    SHARED = [(16, 256), (64, 256), (64, 64)]
    # 共享码本: 跨采样专家池(每专家 60K 向量)训练 — 层共享架构的 4 专家模拟
    shared_C = {}
    for dim, nc in SHARED:
        pool = np.concatenate([subsample(W, dim, 60000) for W in Ws.values()])
        t0 = time.time(); shared_C[(dim, nc)] = kmeans(pool, nc)
        log(f"共享码本 dim{dim}/nc{nc} 训练 {time.time()-t0:.0f}s (pool={len(pool)})")

    out = []
    def emit(e, lane, tag, bpw, cb, rel, cos):
        r = (f"L{L}", "w1", f"e{e}", lane, tag, f"{bpw:.4f}", f"{cb:.4f}", f"{rel:.4f}", f"{cos:.5f}")
        out.append(r); log("  " + " ".join(r))

    for e, lane in picks:
        W = Ws[e]; Ye = Xe @ W.T; t0 = time.time()
        Wq, bpw = cold_C0(W, Xf)
        rel, cos = metrics(Ye, Xe @ Wq.T); emit(e, lane, "prod_sign", bpw, 0, rel, cos)
        for dim, nc in LADDER:
            C = kmeans(subsample(W, dim), nc)
            Wq = vq_apply(W, Xf, C, dim)
            rel, cos = metrics(Ye, Xe @ Wq.T)
            bpw, cb = bpw_of(dim, nc, rows, cols, False)
            tag = f"vq{dim}" if nc == 256 else f"vq{dim}c{nc}"
            emit(e, lane, tag, bpw, cb, rel, cos)
        for dim, nc in SHARED:
            Wq = vq_apply(W, Xf, shared_C[(dim, nc)], dim)
            rel, cos = metrics(Ye, Xe @ Wq.T)
            bpw, _ = bpw_of(dim, nc, rows, cols, True)
            tag = f"shvq{dim}" if nc == 256 else f"shvq{dim}c{nc}"
            emit(e, lane, tag, bpw, 0, rel, cos)
        # signvq64: sign 矩阵二值码本 8× 压缩 + 行 scale(方向存活率)
        S = np.where(W >= 0, 1.0, -1.0).astype(np.float32)
        Cb = kmeans(subsample(S, 64), 256, binary=True)
        Sq = Cb[_assign(S.reshape(-1, 64), Cb)].reshape(W.shape).astype(np.float32)
        s0 = np.abs(W).mean(1)
        Wq = row_act_scale(Sq * s0[:, None], W, Xf)
        rel, cos = metrics(Ye, Xe @ Wq.T)
        emit(e, lane, "signvq64", 8/64+16.0/cols, 0, rel, cos)
        # randvq64: 随机码本机会地板
        V = W.reshape(-1, 64)
        Cr = (RNG.standard_normal((256, 64)) * V.std()).astype(np.float32)
        Wq = vq_apply(W, Xf, Cr, 64)
        rel, cos = metrics(Ye, Xe @ Wq.T)
        emit(e, lane, "randvq64", 8/64+16.0/cols, 0, rel, cos)
        log(f"e{e}({lane}) 完成 {time.time()-t0:.0f}s")

    print("\nlayer kind expert lane variant bpw cb_overhead relF cos")
    for r in out: print(" ".join(r))
    agg = collections.defaultdict(list)
    for r in out: agg[r[4]].append((float(r[7]), float(r[8])))
    print("\n== 汇总(均值, relF 越小越好) ==")
    for k in sorted(agg, key=lambda k: np.mean([v[0] for v in agg[k]])):
        vs = agg[k]
        print(f"{k:12s} relF={np.mean([v[0] for v in vs]):.4f} cos={np.mean([v[1] for v in vs]):.4f} n={len(vs)}")
    if a.out:
        with open(a.out, "w") as f:
            f.write("layer kind expert lane variant bpw cb_overhead relF cos\n")
            for r in out: f.write(" ".join(r) + "\n")
        log(f"报告 → {a.out}")

if __name__ == "__main__":
    main()
