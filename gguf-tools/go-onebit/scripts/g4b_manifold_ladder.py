#!/usr/bin/env python3
"""g4b_manifold_ladder.py — 0.1 bpw 第二支短针: 激活流形基底编码(2026-07-28)。
g4a 判决: 权重空间 VQ 各档恰落高斯率失真极限 cos=sqrt(1-2^-2R), learned≈random →
无结构可榨。但 R-D 墙只锁"重建 W"; 域行为 Y=XW^T 只活在激活流形 span(X) 上
(rank≤265 ≪ 4096)。本针: 系数编码在流形基底 B=V_r(X_fit SVD 右奇异向量, 层共享,
闭式派生隐变量零训练)中进行, 同 bpw 对照 g4a:
  mf16_r24    0.094  f16 系数 rank24        mf16_r64   0.25  f16 rank64
  mq4_r96     0.094  int4 系数 rank96       ms1_r128   0.031 1-bit 系数 rank128
  ms1_r256    0.063  1-bit 系数 rank256(fit 满秩)
  rs1_r256    0.063  随机正交基对照(流形结构 vs 任意低秩 分离)
系数行 scale=act ridge(生产同款); 1-bit 加全局每维 scale d_j=mean|A_j|(f16, 自重~0.0007)。
判据: held-out(奇行) relF/cos — held 行不在 fit 流形内 ⇒ 泛化真考。
用法(M1): g4b_manifold_ladder.py --layer L --x /tmp/xr_x_L{L:02d}.npy [--out rpt]
"""
import os, sys, time, argparse, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "quant"))
from g1a_lever_probe import open_shard_for_layer
from g1b_stack_probe import metrics

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", file=sys.stderr, flush=True)

RNG = np.random.default_rng(0x5EED)

def f16(a): return a.astype(np.float16).astype(np.float32)

def row_act_scale(Wq, W, Xf):
    p = Xf @ Wq.T; y = Xf @ W.T
    sp2 = (p*p).sum(0); spy = (p*y).sum(0)
    lam = np.maximum(sp2/(len(Xf)+1.0), 1e-3*sp2+1e-9)
    g = np.clip((spy+lam*1.0)/(sp2+lam), 0.25, 4.0)
    return Wq * g[:, None].astype(np.float32)

def coeff_code(A, mode):
    """A[rows,r] → 量化系数 + 每权重附加 bpw(不含基底/行乘子)。"""
    r = A.shape[1]
    if mode == "f16":
        return f16(A), 16.0*r
    if mode == "q4":
        s = np.abs(A).max(1)/7.0 + 1e-20
        return np.round(A/s[:, None])*s[:, None], 4.0*r
    if mode == "s1":
        d = f16(np.abs(A).mean(0) + 1e-20)                  # 全局每维 scale
        return np.where(A >= 0, 1.0, -1.0).astype(np.float32)*d[None, :], 1.0*r
    raise SystemExit(mode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--x", required=True)
    ap.add_argument("--hf", default="/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base")
    ap.add_argument("--hot-table", default=os.path.join(HERE, "..", "corpus", "prog_active_top64.txt"))
    ap.add_argument("--out", default="")
    a = ap.parse_args(); L = a.layer
    X = np.load(a.x).astype(np.float32)
    Xf, Xe = X[0::2], X[1::2]
    hot = {}
    for ln in open(a.hot_table):
        p = ln.split(":"); hot[int(p[0][1:])] = [int(x) for x in p[1].split()]
    cold_all = [e for e in range(256) if e not in set(hot[L])]
    picks = [(e, "hot") for e in hot[L][:2]] + [(e, "cold") for e in (cold_all[0], cold_all[-1])]
    sh = open_shard_for_layer(a.hf, L)

    # 层共享流形基底: X_fit SVD 右奇异向量(rank ≤ len(Xf)); 随机正交基对照
    U, S, Vt = np.linalg.svd(Xf, full_matrices=False)        # Vt [265,4096]
    en = (S*S).cumsum()/(S*S).sum()
    log(f"G4b L{L}: X={X.shape} fit-rank={len(S)} 能量: r24={en[23]:.3f} r64={en[63]:.3f} r128={en[127]:.3f} r256={en[min(255,len(S)-1)]:.3f}")
    Q, _ = np.linalg.qr(RNG.standard_normal((4096, 256)).astype(np.float32))
    RB = Q.astype(np.float32)                                # 随机正交 4096×256

    RUNGS = [("mf16_r24", "f16", 24), ("mf16_r64", "f16", 64), ("mq4_r96", "q4", 96),
             ("ms1_r128", "s1", 128), ("ms1_r256", "s1", 256), ("rs1_r256", "s1", 256)]
    out = []
    for e, lane in picks:
        W = sh.expert_w(L, e, "w1"); Ye = Xe @ W.T
        cols = W.shape[1]; t0 = time.time()
        for tag, mode, r in RUNGS:
            B = RB[:, :r] if tag.startswith("rs1") else Vt[:r].T   # [4096,r]
            A = W @ B
            Aq, cbits = coeff_code(A, mode)
            Wq = row_act_scale((Aq @ B.T).astype(np.float32), W, Xf)
            rel, cos = metrics(Ye, Xe @ Wq.T)
            bpw = cbits/cols + 16.0/cols                     # 系数+行乘子; 共享基底~0
            row = (f"L{L}", "w1", f"e{e}", lane, tag, f"{bpw:.4f}", f"{rel:.4f}", f"{cos:.5f}")
            out.append(row); log("  " + " ".join(row))
        log(f"e{e}({lane}) 完成 {time.time()-t0:.0f}s")

    print("\nlayer kind expert lane variant bpw relF cos")
    for r in out: print(" ".join(r))
    agg = collections.defaultdict(list)
    for r in out: agg[r[4]].append((float(r[6]), float(r[7])))
    print("\n== 汇总(均值, relF 越小越好) ==")
    for k in sorted(agg, key=lambda k: np.mean([v[0] for v in agg[k]])):
        vs = agg[k]
        print(f"{k:12s} relF={np.mean([v[0] for v in vs]):.4f} cos={np.mean([v[1] for v in vs]):.4f} n={len(vs)}")
    if a.out:
        with open(a.out, "w") as f:
            f.write("layer kind expert lane variant bpw relF cos\n")
            for r in out: f.write(" ".join(r) + "\n")
        log(f"报告 → {a.out}")

if __name__ == "__main__":
    main()
