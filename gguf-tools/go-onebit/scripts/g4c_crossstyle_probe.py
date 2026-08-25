#!/usr/bin/env python3
"""g4c_crossstyle_probe.py — 流形基底跨风格泛化 + rank 天花板针(2026-07-28)。
g4b 判决: 同 0.098bpw 流形基底 cos 0.77-0.83 vs 权重空间 R-D 墙 0.39; 但 held-out
是同捕获奇偶行(同风格内)。本针用 v5 语料 FP 态 X(g4c_capture_v5x.sh 产)回答两问:
  ①跨风格泛化: 基底 fit 在部分风格, eval 整格陌生风格(界外滑坡多陡?)
    切分: insty(奇偶行, g4b 口径参照) / xtut(fit=repo+test+contract, eval=tut 全格,
    Go 崩塌流形) / xcontract(fit=repo+tut+test) / xmethod(eval=非代码 method 尾巴,最远界外)
  ②rank 天花板: fit-rank ~3500 后 mq4 r∈{96,256,512,1024} (bpw 0.098/0.25/0.50/1.00)
    + ms1_r256(0.066) + rs1_r256 随机基对照(insty only)。
判据: relF/cos on Y=X_eval·W^T, 2 hot+2 cold 专家 w1。
用法(M1): g4c_crossstyle_probe.py --layer L --x /tmp/g4c_x_L{L:02d}.npy [--out rpt]
"""
import os, sys, time, argparse, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "quant"))
from g1a_lever_probe import open_shard_for_layer
from g1b_stack_probe import metrics

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", file=sys.stderr, flush=True)

RNG = np.random.default_rng(0x5EED)

def style_of(name):
    n = name[5:] if name.startswith("held_") else name
    if n == "method": return "method"
    return n.split(".")[1]                              # lang.style

def row_act_scale(Wq, W, Xf):
    p = Xf @ Wq.T; y = Xf @ W.T
    sp2 = (p*p).sum(0); spy = (p*y).sum(0)
    lam = np.maximum(sp2/(len(Xf)+1.0), 1e-3*sp2+1e-9)
    g = np.clip((spy+lam*1.0)/(sp2+lam), 0.25, 4.0)
    return Wq * g[:, None].astype(np.float32)

def coeff_q4(A):
    s = np.abs(A).max(1)/7.0 + 1e-20
    return np.round(A/s[:, None])*s[:, None]

def coeff_s1(A):
    d = (np.abs(A).mean(0) + 1e-20).astype(np.float16).astype(np.float32)
    return np.where(A >= 0, 1.0, -1.0).astype(np.float32)*d[None, :]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--x", required=True)
    ap.add_argument("--cells", default="/tmp/g4c_cells.tsv")
    ap.add_argument("--hf", default="/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base")
    ap.add_argument("--hot-table", default=os.path.join(HERE, "..", "corpus", "prog_active_top64.txt"))
    ap.add_argument("--out", default="")
    a = ap.parse_args(); L = a.layer
    X = np.load(a.x).astype(np.float32)
    rows_style = np.empty(len(X), dtype=object)
    for ln in open(a.cells):
        name, _, s, e2 = ln.split("\t")
        rows_style[int(s):int(e2)] = style_of(name)
    assert not (rows_style == None).any(), "cells.tsv 未覆盖全部行"
    code = np.isin(rows_style, ("repo", "tut", "test", "contract"))
    SPLITS = {
        "insty":     (np.arange(len(X)) % 2 == 0) & code,
        "xtut":      np.isin(rows_style, ("repo", "test", "contract")),
        "xcontract": np.isin(rows_style, ("repo", "tut", "test")),
        "xmethod":   code,
    }
    EVALS = {
        "insty":     (np.arange(len(X)) % 2 == 1) & code,
        "xtut":      rows_style == "tut",
        "xcontract": rows_style == "contract",
        "xmethod":   rows_style == "method",
    }
    hot = {}
    for ln in open(a.hot_table):
        p = ln.split(":"); hot[int(p[0][1:])] = [int(x) for x in p[1].split()]
    cold_all = [e for e in range(256) if e not in set(hot[L])]
    picks = [(e, "hot") for e in hot[L][:2]] + [(e, "cold") for e in (cold_all[0], cold_all[-1])]
    sh = open_shard_for_layer(a.hf, L)
    Ws = {e: sh.expert_w(L, e, "w1") for e, _ in picks}
    cols = next(iter(Ws.values())).shape[1]

    out = []
    for split in ("insty", "xtut", "xcontract", "xmethod"):
        Xf, Xe = X[SPLITS[split]], X[EVALS[split]]
        t0 = time.time()
        U, S, Vt = np.linalg.svd(Xf, full_matrices=False)
        log(f"L{L} {split}: fit={len(Xf)} eval={len(Xe)} svd {time.time()-t0:.0f}s fit-rank={len(S)}")
        RUNGS = [("mq4_r96", "q4", 96), ("mq4_r256", "q4", 256),
                 ("mq4_r512", "q4", 512), ("mq4_r1024", "q4", 1024), ("ms1_r256", "s1", 256)]
        if split == "insty": RUNGS.append(("rs1_r256", "s1", 256))
        for e, lane in picks:
            W = Ws[e]; Ye = Xe @ W.T
            for tag, mode, r in RUNGS:
                if r > len(S): continue
                if tag.startswith("rs1"):
                    B, _ = np.linalg.qr(RNG.standard_normal((cols, r)).astype(np.float32)); B = B.astype(np.float32)
                else:
                    B = Vt[:r].T
                A = W @ B
                Aq = coeff_q4(A) if mode == "q4" else coeff_s1(A)
                Wq = row_act_scale((Aq @ B.T).astype(np.float32), W, Xf)
                rel, cos = metrics(Ye, Xe @ Wq.T)
                bpw = (4.0 if mode == "q4" else 1.0)*r/cols + 16.0/cols
                row = (f"L{L}", split, f"e{e}", lane, tag, f"{bpw:.4f}", f"{rel:.4f}", f"{cos:.5f}")
                out.append(row); log("  " + " ".join(row))

    print("\nlayer split expert lane variant bpw relF cos")
    for r in out: print(" ".join(r))
    agg = collections.defaultdict(list)
    for r in out: agg[(r[1], r[4])].append(float(r[7]))
    print("\n== 汇总(cos 均值) ==")
    for k in sorted(agg):
        print(f"{k[0]:10s} {k[1]:10s} cos={np.mean(agg[k]):.4f} n={len(agg[k])}")
    if a.out:
        with open(a.out, "w") as f:
            f.write("layer split expert lane variant bpw relF cos\n")
            for r in out: f.write(" ".join(r) + "\n")
        log(f"报告 → {a.out}")

if __name__ == "__main__":
    main()
