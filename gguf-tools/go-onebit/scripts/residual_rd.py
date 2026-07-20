#!/usr/bin/env python3
"""residual_rd.py — 权重空间残差 率失真探针 (方向杠杆的决定性实验)。

判决背景: 全动态标量修正(full_optimize)在 hard-text 只到分布还原 0.545/PPL×2.71,
根因 yq=share+gx·rb 只缩放 1-bit 输出幅度、方向被符号钉死(cos 动不了)。
behavior-space 全部标量/低秩杠杆已判死。唯一未测=权重空间残差 R=W-Q1(W)。

本探针不碰 behavior-space, 直接测权重残差恢复的率失真:
  对 hard 层的真实路由专家权重 w1/w3/w2, 用真实激活 X 度量【专家矩阵输出方向 cos】:
    Q1     = 1-bit per-block(≈1.06 bit/w)         → cos0
    Q1+Q2  = 二次 1-bit 残差(≈2.12 bit/w)          → cos2   (=有效2-bit)
    Q1+lowrank_r = R 的秩-r SVD(+ r·(din+dout)·16 bit) → cos_r
  能量加权聚合(专家×矩阵), 打印率失真表: bit/w → cos。
回答: 把 hard-text 专家输出 cos 从 ~0.5 抬到 ~0.9 需要多少 bit/权重
     (低秩是否可行 / 还是必须逼近 2-bit)。零训练, 纯闭式。

用法(M1): DS4_HF=... python3 residual_rd.py --ids CODE.ids [--ntok 64]
          [--probe-layers 2,20,40] [--experts 8]
"""
import argparse
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "calib", "pyfwd"))
sys.path.insert(0, os.path.join(_here, "..", "quant"))
import dsv4_fwd as F   # noqa: E402
from full_optimize import go1b_q, layer_attn_to_fin, NACT   # noqa: E402


def _cos(Y, Yh):
    a = Y.reshape(-1); b = Yh.reshape(-1)
    d = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float((a * b).sum() / d)


def _bits_per_w(din, dout, base=True, second=False, rank=0, sparse_f=0.0):
    """1-bit per-block16: signs=1 bit/w + scales 16·dout·16bit。二次同。
    低秩 fp16: rank·(din+dout)·16。稀疏(index计费): f·(log2(din)+16)。"""
    tot = din * dout
    b = 0.0
    if base:     b += 1.0 + (16 * dout * 16) / tot
    if second:   b += 1.0 + (16 * dout * 16) / tot
    if rank:     b += (rank * (din + dout) * 16) / tot
    if sparse_f: b += sparse_f * (np.log2(din) + 16)
    return b


def probe_matrix(W, Xcal, Xte, ranks, sparse_fracs):
    """率失真点(★held-out★): go1b_q 在 Xcal(标定,前半全token,贴生产)标定, cos 在 Xte(held-out路由token)评估。
    返回 dict{config:(cos,bit/w,energy)}。残差变体: 二次1bit / 低秩SVD / top-k稀疏(index计费)。"""
    dout, din = W.shape
    Yte = Xte @ W.T
    energy = float((Yte ** 2).sum())
    Q1 = go1b_q(W, Xcal)
    R = W - Q1
    out = {"Q1": (_cos(Yte, Xte @ Q1.T), _bits_per_w(din, dout, base=True), energy)}
    Q2 = go1b_q(R, Xcal)
    out["Q1+Q2"] = (_cos(Yte, Xte @ (Q1 + Q2).T),
                    _bits_per_w(din, dout, base=True, second=True), energy)
    try:
        U, s, Vt = np.linalg.svd(R, full_matrices=False)
        for r in ranks:
            if r >= len(s): continue
            Rr = (U[:, :r] * s[:r]) @ Vt[:r]
            out[f"Q1+lr{r}"] = (_cos(Yte, Xte @ (Q1 + Rr).T),
                                _bits_per_w(din, dout, base=True, rank=r), energy)
    except np.linalg.LinAlgError:
        pass
    # top-k 稀疏残差(唯一可能 <2bit 的候选): 每行保留 |R| 最大的 f·din 项 fp16 + index
    aR = np.abs(R)
    for f in sparse_fracs:
        k = max(1, int(round(f * din)))
        thr = np.partition(aR, din - k, axis=1)[:, din - k][:, None]
        Rs = np.where(aR >= thr, R, 0.0).astype(np.float32)
        out[f"Q1+sp{f:g}"] = (_cos(Yte, Xte @ (Q1 + Rs).T),
                              _bits_per_w(din, dout, base=True, sparse_f=f), energy)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ntok", type=int, default=64)
    ap.add_argument("--probe-layers", default="2,20,40")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--ranks", default="16,64,256")
    ap.add_argument("--sparse", default="0.03,0.06,0.12")   # top-k 稀疏残差比例
    ap.add_argument("--fin-cache", default="/tmp/rd_fin.npz")  # Fin 快照缓存(跳过前向)
    a = ap.parse_args()
    probe = [int(x) for x in a.probe_layers.split(",")]
    ranks = [int(x) for x in a.ranks.split(",")]
    sparse_fracs = [float(x) for x in a.sparse.split(",") if x.strip()]
    ids = np.array([int(x) for x in open(a.ids) if x.strip()], dtype=np.int64)[:a.ntok]
    Ids = ids
    emb = F.R.get("embed.weight"); HCM = F.HCM
    H0 = np.repeat(emb[ids][:, None, :], HCM, 1).astype(np.float32)

    # ── fp8 前向, 在 probe 层快照 Fin(=MoE 输入); 有缓存则跳过(快速迭代率失真变体) ──
    if os.path.exists(a.fin_cache):
        print(f"载入 Fin 缓存 {a.fin_cache}(跳过前向)", file=sys.stderr, flush=True)
        z = np.load(a.fin_cache)
        fin_at = {int(k[1:]): z[k] for k in z.files}
    else:
        print(f"前向到 {max(probe)} 层取 Fin...", file=sys.stderr, flush=True)
        H = H0.copy(); fin_at = {}
        for L in range(max(probe) + 1):
            W = F.load_layer(L); t0 = time.time()
            Fin_f, ctx = layer_attn_to_fin(H, W, L)
            if L in probe:
                fin_at[L] = Fin_f.copy()
            share = F.expert_fp(Fin_f, W['s1'], W['s3'], W['s2'])
            idx, wt, _ = F.gate_route(Fin_f, W, Ids)
            out = share.copy(); fe = idx.reshape(-1); fw = wt.reshape(-1)
            tok = np.repeat(np.arange(len(Fin_f)), NACT)
            for e in np.unique(fe):
                if e < 0: continue
                m = fe == e; t = tok[m]; ww = fw[m][:, None]
                w1, w3, w2 = (F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w{k}.weight") for k in (1, 3, 2))
                np.add.at(out, t, F.expert_fp(Fin_f[t], w1, w3, w2, ww))
            H = F.hc_post(out.astype(np.float32), *ctx)
            print(f"[前向] L{L:2d} {time.time()-t0:5.1f}s", file=sys.stderr, flush=True)
            del W
        np.savez(a.fin_cache, **{f"L{L}": fin_at[L] for L in probe})
        print(f"Fin 已缓存 → {a.fin_cache}", file=sys.stderr, flush=True)

    # ── 每 probe 层: top-N 专家, ★held-out★(前半标定/后半评估)逐矩阵率失真, 能量加权 ──
    for L in probe:
        Fin = fin_at[L]; W = F.load_layer(L); nh = len(Fin) // 2
        Fin_cal = Fin[:nh]                       # 标定输入(前半全token, 贴生产 go1b_q(w,Fin))
        idx, wt, _ = F.gate_route(Fin, W, Ids)
        fe = idx.reshape(-1); tok = np.repeat(np.arange(len(Fin)), NACT)
        uniq, cnt = np.unique(fe[fe >= 0], return_counts=True)
        top = uniq[np.argsort(cnt)[::-1][:a.experts]]
        hid_cal_cache = {}
        agg = {}   # config -> [cos*energy 累计, energy 累计, bit/w]
        for e in top:
            m = fe == e; t = tok[m]; te = t[t >= nh]           # held-out 路由 token
            if len(te) < 2: continue
            Xte = Fin[te]
            w1, w3, w2 = (F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w{k}.weight") for k in (1, 3, 2))
            def _hid(Xf):
                gf = Xf @ w1.T; uf = Xf @ w3.T
                return (gf / (1 + np.exp(-gf))) * uf
            hid_cal = _hid(Fin_cal); hid_te = _hid(Xte)         # down-proj 输入(标定/held-out)
            for W_, Xc_, Xt_ in ((w1, Fin_cal, Xte), (w3, Fin_cal, Xte), (w2, hid_cal, hid_te)):
                for cfg, (c, bpw, en) in probe_matrix(W_, Xc_, Xt_, ranks, sparse_fracs).items():
                    s = agg.setdefault(cfg, [0.0, 0.0, bpw])
                    s[0] += c * en; s[1] += en
        print(f"\n=== L{L} (top{len(top)}专家 × w1/w3/w2, 能量加权 cos) ===", flush=True)
        order = sorted(agg.items(), key=lambda kv: kv[1][2])
        for cfg, (cw, ew, bpw) in order:
            print(f"  {cfg:10s} bit/w={bpw:5.2f}  cos={cw/ew:.4f}", flush=True)
        del W


if __name__ == "__main__":
    main()
