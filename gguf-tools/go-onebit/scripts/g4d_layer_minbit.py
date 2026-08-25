#!/usr/bin/env python3
"""g4d_layer_minbit.py — 每层最小 bit 短针扫描(用户方案, 2026-07-28)。
思路(用户): 每层跑 bpw 梯子短针 → 每层最小 bit → 求和 = 最小体积(直接喂动态层体积架构)。
质量锚: 该层 vq8×256(v2.2 生产冷路, 1.0bpw) held-out cos — v4bf 端到端行为门已验证可用,
        "同层内误差不劣于生产"即该层可换; min bpw = 达锚的最低档。
梯子(g4a/b 判决后的有效前沿=流形系数族): 按 bpw 升序
  mq4_r48(0.051) ms1_r256(0.066) mq4_r96(0.098) mq4_r192(0.191) mq4_r384(0.379)
  mq4_r768(0.754) → 全不达锚则 min=vq8(1.0, 维持生产)。
口径: w1, 2 hot+2 cold 专家, code 行(repo/tut/test/contract 含 held)偶=fit 奇=eval;
      跨风格泛化系数另由 g4c_crossstyle_probe 判(本扫是层内同风格口径)。
用法(M1): g4d_layer_minbit.py --l0 0 --l1 14 --out /tmp/g4d_lane0.rpt
"""
import os, sys, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "quant"))
from g1a_lever_probe import open_shard_for_layer
from g1b_stack_probe import metrics

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

def vq8_anchor(W, Xf):
    """生产冷路 vq8×256 (1.0bpw) — 层内质量锚。"""
    V = W.reshape(-1, 8)
    sub = V[RNG.choice(len(V), min(len(V), 200000), replace=False)]
    C = sub[RNG.choice(len(sub), 256, replace=False)].copy()
    for _ in range(8):
        a = _assign(sub, C)
        for c in range(256):
            m = a == c
            if m.any(): C[c] = sub[m].mean(0)
    Wq = C[_assign(V, C)].reshape(W.shape).astype(np.float32)
    return row_act_scale(Wq, W, Xf)

def coeff_q4(A):
    s = np.abs(A).max(1)/7.0 + 1e-20
    return np.round(A/s[:, None])*s[:, None]

def coeff_s1(A):
    d = (np.abs(A).mean(0) + 1e-20).astype(np.float16).astype(np.float32)
    return np.where(A >= 0, 1.0, -1.0).astype(np.float32)*d[None, :]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l0", type=int, required=True); ap.add_argument("--l1", type=int, required=True)
    ap.add_argument("--cells", default="/tmp/g4c_cells.tsv")
    ap.add_argument("--hf", default="/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base")
    ap.add_argument("--hot-table", default=os.path.join(HERE, "..", "corpus", "prog_active_top64.txt"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hot = {}
    for ln in open(a.hot_table):
        p = ln.split(":"); hot[int(p[0][1:])] = [int(x) for x in p[1].split()]

    RUNGS = [("mq4_r48", "q4", 48, 0.0508), ("ms1_r256", "s1", 256, 0.0664),
             ("mq4_r96", "q4", 96, 0.0977), ("mq4_r192", "q4", 192, 0.1914),
             ("mq4_r384", "q4", 384, 0.3789), ("mq4_r768", "q4", 768, 0.7539)]
    rows_out = []
    for L in range(a.l0, a.l1 + 1):
        t0 = time.time()
        X = np.load(f"/tmp/g4c_x_L{L:02d}.npy").astype(np.float32)
        st = np.empty(len(X), dtype=object)
        for ln in open(a.cells):
            name, _, s, e2 = ln.split("\t")
            st[int(s):int(e2)] = style_of(name)
        code = np.isin(st, ("repo", "tut", "test", "contract"))
        idx = np.arange(len(X))
        Xf, Xe = X[code & (idx % 2 == 0)], X[code & (idx % 2 == 1)]
        _, S, Vt = np.linalg.svd(Xf, full_matrices=False)
        cold_all = [e for e in range(256) if e not in set(hot[L])]
        picks = hot[L][:2] + [cold_all[0], cold_all[-1]]
        sh = open_shard_for_layer(a.hf, L)
        anc, rung_cos = [], {t: [] for t, _, _, _ in RUNGS}
        for e in picks:
            W = sh.expert_w(L, e, "w1"); Ye = Xe @ W.T
            _, c0 = metrics(Ye, Xe @ vq8_anchor(W, Xf).T); anc.append(c0)
            for tag, mode, r, _ in RUNGS:
                B = Vt[:r].T
                A = W @ B
                Aq = coeff_q4(A) if mode == "q4" else coeff_s1(A)
                Wq = row_act_scale((Aq @ B.T).astype(np.float32), W, Xf)
                _, c = metrics(Ye, Xe @ Wq.T); rung_cos[tag].append(c)
        anc_m = float(np.mean(anc))
        pick_tag, pick_bpw = "vq8", 1.0078
        for tag, _, _, bpw in RUNGS:                     # bpw 升序 → 首个达锚
            if np.mean(rung_cos[tag]) >= anc_m: pick_tag, pick_bpw = tag, bpw; break
        row = [f"L{L:02d}", f"{anc_m:.4f}"] + [f"{np.mean(rung_cos[t]):.4f}" for t, _, _, _ in RUNGS] \
              + [pick_tag, f"{pick_bpw:.4f}"]
        rows_out.append(row)
        log(f"L{L:02d} 完成 {time.time()-t0:.0f}s anchor={anc_m:.4f} → {pick_tag}({pick_bpw})")

    hdr = ["layer", "vq8_anchor"] + [t for t, _, _, _ in RUNGS] + ["min_variant", "min_bpw"]
    with open(a.out, "w") as f:
        f.write(" ".join(hdr) + "\n")
        for r in rows_out: f.write(" ".join(r) + "\n")
    log(f"报告 → {a.out}")

if __name__ == "__main__":
    main()
