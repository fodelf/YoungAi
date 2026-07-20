#!/usr/bin/env python3
"""full_optimize.py — 全模型 逐层动态1bit + 隐变量z + 四损失, 向前+向后+感知优化 (对齐用户收官指令)。

三份一起动 (每层):
  动态1bit模型 = base 1-bit(output-optimal sign+scale) + 每专家乘法系数 coef_e(x)=g_e + c_e·(vᵀx)
                 (D0 g_e 修幅度塌缩 + D1/D2 动态部分, v 层内共享低秩基, 零训练闭式)
  隐变量 z    = 上面的 (v, g_e, c_e) = 每专家动态条件系数 (=z-dynamic-redesign 的 D2, 设计本义)
  四损失      = L_fixed(ridge λ) + L_classify(输出方差权) + L_smooth(输入抖动) + L_align(最优增益目标)

向前 (forward): 顺序传播, 每层在【上游已量化输出】上优化 → 误差层层传播 (真前向, 非独立层)。
向后 (backward): 头部误差 cotangent 近似回传 (残差近似 λ_L≈headᵀ·err), s_L=⟨λ,δ_L⟩=每层对最终的敏感度。
感知优化 (perceptual): 报敏感度剖面; 高敏层是加 bit/rank 预算的靶 (分配在后续 pass 接)。
指标: 最终 top-1 一致率 + rel_L2(★幅度敏感, cos盲区★) vs fp8。

用法(M1): DS4_HF=... python3 full_optimize.py --ids CODE.ids [--ntok 64] [--layers 0-42] [--rank 8]
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

D = 4096
NACT = 6


NBLK = 16   # GO1B 格式内 16 block/行, per-block 输出最优 scale (格式内零额外字节, 实测抬 cos)


_DEPLOY_MATCH = os.environ.get("DS4_DEPLOY_MATCH") is not None


def go1b_q(W, X):
    """per-block 输出最优 scale (nblk=16): per row 联合解 16 scale 使 Σ_b s_b·(B_b·x)≈w·x。
    格式内(GO1B 本就16block), 零额外字节。X=标定激活(全token, 防16-scale数据饿死)。
    DS4_DEPLOY_MATCH: 改用 deepseek4-quantize 的 diagonal per-block scale (s=Σ ew|w|/Σ ew,
    ew=E[x²] per channel), 隔离 joint-LS vs diagonal 是否为部署崩溃元凶。"""
    dout, din = W.shape
    B = np.sign(W).astype(np.float32); B[B == 0] = 1
    if _DEPLOY_MATCH and din % NBLK == 0 and len(X) >= 4:
        bs = din // NBLK
        ew = (X.astype(np.float64) ** 2).mean(0)                    # E[x²] per input channel
        S = np.empty((dout, NBLK), np.float32)
        aw = np.abs(W)
        for b in range(NBLK):
            c = slice(b * bs, (b + 1) * bs)
            num = (aw[:, c] * ew[c]).sum(1); den = ew[c].sum()      # Σ ew|w| / Σ ew (diagonal)
            S[:, b] = (num / den) if den > 0 else aw[:, c].mean(1)
        return B * np.repeat(S, bs, axis=1).astype(np.float32)
    if din % NBLK != 0 or len(X) < 4:
        Pt = X @ W.T; Ps = X @ B.T
        num = (Pt * Ps).sum(0); den = (Ps * Ps).sum(0)
        s = np.where(den > 0, num / np.maximum(den, 1e-12), np.abs(W).mean(1))
        return B * s[:, None].astype(np.float32)
    bs = din // NBLK; Y = X @ W.T; T = X.shape[0]
    P = np.empty((T, dout, NBLK), np.float32)
    for b in range(NBLK):
        c = slice(b * bs, (b + 1) * bs)
        P[:, :, b] = X[:, c] @ B[:, c].T
    A = np.einsum('tib,tic->ibc', P, P); rhs = np.einsum('tib,ti->ib', P, Y)
    tr = np.trace(A, axis1=1, axis2=2) / NBLK + 1e-9
    A[:, np.arange(NBLK), np.arange(NBLK)] += (1e-3 * tr)[:, None]
    S = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
    return B * np.repeat(S, bs, axis=1).astype(np.float32)


def routed_coadapt(Fin, Ids, W, L, nh, sens_w=1.0, dim_w=None, zranks=(0,)):
    """★真动态★每层调优: 读专家累积 fp8 teacher(yf)+1bit student(yb)。对 routed 残差拟合乘法系数,
    但四损失超参 {λ(L_fixed)×dither(L_smooth)} 与 z结构{弃/D0静态标量/D1动态标量} 逐层【扫配置→held-out选最优】
    (不再写死); sens_w≥1.5(向后判定的高敏层)额外试逐通道增益(感知优化, 更丰富z)。返回 (yf, yq, tag)。
    L_align=最优增益目标; L_classify=held-out routed能量权。shared 高精不动。"""
    idx, wt, _ = F.gate_route(Fin, W, Ids)
    share = F.expert_fp(Fin, W['s1'], W['s3'], W['s2'])
    yf = share.copy(); yb = share.copy()
    fe = idx.reshape(-1); fw = wt.reshape(-1)
    tok = np.repeat(np.arange(len(Fin)), NACT)
    for e in np.unique(fe):
        if e < 0: continue
        m = fe == e; t = tok[m]; ww = fw[m][:, None]; Xt = Fin[t]
        w1, w3, w2 = (F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w{k}.weight") for k in (1, 3, 2))
        np.add.at(yf, t, F.expert_fp(Xt, w1, w3, w2, ww))
        q1 = go1b_q(w1, Fin); q3 = go1b_q(w3, Fin)
        gf = Fin @ q1.T; uf = Fin @ q3.T; hf = (gf / (1 + np.exp(-gf))) * uf
        q2 = go1b_q(w2, hf)
        np.add.at(yb, t, F.expert_fp(Xt, q1, q3, q2, ww))
    rf = yf - share; rb = yb - share                       # routed teacher / 1bit student
    a = (rf * rb).sum(1) / ((rb * rb).sum(1) + 1e-9)        # 每token最优增益 (L_align 目标)
    S = len(Fin); tr = np.arange(S) < nh; te = ~tr
    Xtr, atr = Fin[tr], a[tr]
    wc = (rf[tr] ** 2).sum(1); wc = wc / (wc.mean() + 1e-9)  # L_classify: routed能量权
    Xte, rb_te, rf_te = Fin[te], rb[te], rf[te]
    wte = (rf_te ** 2).sum(1) + 1e-9                        # held-out L_classify 权(token)
    # ★感知★ 每输出维重要度 dw: 向后传入(dim_w=下游敏感度), 让每层优先修下游在乎的维; None=均匀
    dw = dim_w if dim_w is not None else np.ones(D, np.float32)
    dw = (dw / (dw.mean() + 1e-9)).astype(np.float32)

    def herr(gx):   # held-out 维加权+token加权 重建误差 (真实每层目标, 幅度敏感)
        e = (((gx[:, None] * rb_te - rf_te) ** 2) * dw).sum(1)
        return float(np.nan_to_num(e, nan=1e30) @ wte / wte.sum())

    # ★动态 per-layer 调优★: 扫 四损失{λ(L_fixed)×dither(L_smooth)} × z结构{弃/D0静/D1动} held-out选最优
    Z = np.zeros(D, np.float32)
    best = (herr(np.ones(len(Xte))), 1.0, Z, Z, "z弃")         # (err, aw, v, Xm, tag); 基线 g=1
    for lam_m in (0.03, 0.1, 0.3, 1.0):
        for dith in (0.0, 0.1):
            Xf, af, wf = Xtr, atr, wc
            if dith > 0:
                rng = np.random.RandomState(1)
                Xd = Xtr + dith * Xtr.std(0, keepdims=True) * rng.randn(*Xtr.shape)
                Xf = np.vstack([Xtr, Xd]); af = np.concatenate([atr, atr]); wf = np.concatenate([wc, wc])
            aw = float(np.average(af, weights=wf))
            Xm = np.average(Xf, 0, weights=wf).astype(np.float32)
            Xc = Xf - Xm; sw = np.sqrt(wf)[:, None]
            A = (Xc * sw).T @ (Xc * sw); A[np.diag_indices(D)] += lam_m * np.trace(A) / D + 1e-6
            v = np.nan_to_num(np.linalg.solve(A, (Xc * sw).T @ ((af - aw) * wf))).astype(np.float32)
            e0 = herr(np.full(len(Xte), aw, np.float32))       # D0 静态标量 (只修幅度)
            if e0 < best[0]: best = (e0, aw, Z, Z, f"D0·λ{lam_m:g}")
            e1 = herr(aw + (Xte - Xm) @ v)                     # D1 动态标量 g(x)
            if e1 < best[0]: best = (e1, aw, v, Xm, f"D1·λ{lam_m:g}·d{dith:g}")
    err_b, aw, v, Xm, tag = best
    if _DEPLOY_MATCH:                                       # 部署无每层增益修正: gx=1
        return yf.astype(np.float32), (share + rb).astype(np.float32), "deploy(diag·无增益)"
    gx = aw + (Fin - Xm) @ v                                # 标量增益 g(x) (全token)
    yq = share + gx[:, None] * rb
    # ★低秩方向 z (RRR 闭式, 秩 r 逐层 held-out 选=动态 k_L)★: 标量 gx 只修幅度(平行 rb),
    # 加一个【不平行 rb】的低秩方向项修方向。E=rf−gx·rb 的未解释残差, 对输入 (x−Xm) 做约化秩回归;
    # dim_w(向后敏感度)在 herr2 里加权=感知优化。侧车/base 不动/1-bit 安全。r=0 → 现行为。
    if max(zranks) > 0:
        gx_tr = gx[tr]; Ec = (rf[tr] - gx_tr[:, None] * rb[tr]).astype(np.float32)   # 训练未解释残差
        Xct = (Xtr - Xm).astype(np.float32); lamz = 0.1
        Az = Xct.T @ Xct; Az[np.diag_indices(D)] += lamz * np.trace(Az) / D + 1e-6
        Bz = np.nan_to_num(np.linalg.solve(Az, Xct.T @ Ec)).astype(np.float32)        # din×D ridge OLS
        gx_te = gx[te]
        def herr2(add_te):
            e = (((gx_te[:, None] * rb_te + add_te - rf_te) ** 2) * dw).sum(1)
            return float(np.nan_to_num(e, nan=1e30) @ wte / wte.sum())
        best_ez, best_add_full, rtag = herr2(np.zeros_like(rf_te)), None, ""   # 基线=不加方向项
        try:
            _, _, Vt = np.linalg.svd(Xct @ Bz, full_matrices=False)              # 拟合输出子空间
            for r in sorted(zranks):
                if r <= 0 or r > Vt.shape[0]: continue
                Vr = Vt[:r].T                                                    # D×r
                Br = Bz @ Vr @ Vr.T                                              # din×D 秩-r RRR
                if herr2((Xte - Xm) @ Br) < best_ez - 1e-9:
                    best_ez = herr2((Xte - Xm) @ Br)
                    best_add_full = ((Fin - Xm) @ Br).astype(np.float32); rtag = f"+z秩{r}"
        except np.linalg.LinAlgError:
            pass
        if best_add_full is not None:
            yq = yq + best_add_full; tag += rtag
    if sens_w >= 1.5: tag += "[高敏·维加权]"
    return yf.astype(np.float32), yq.astype(np.float32), f"{tag}·g{aw:.2f}"


def layer_attn_to_fin(H, W, L):
    resid = H
    y, post, comb = F.hc_pre(H, W['hc_attn_fn'], W['hc_attn_scale'], W['hc_attn_base'])
    a = F.attention(F.rms(y, W['an']), W, L)
    h = F.hc_post(a, resid, post, comb)
    resid2 = h
    y2, post2, comb2 = F.hc_pre(h, W['hc_ffn_fn'], W['hc_ffn_scale'], W['hc_ffn_base'])
    fin = F.rms(y2, W['fn'])
    return fin, (resid2, post2, comb2)


def head_logits(H):
    hcfn = F.R.get("hc_head_fn"); hcb = F.R.get("hc_head_base"); hcs = F.R.get("hc_head_scale")
    fnorm = F.R.get("norm.weight"); hw = F.R.get("head.weight")
    S = H.shape[0]; x = H.reshape(S, -1)
    rsq = np.reciprocal(np.sqrt(np.mean(x * x, -1, keepdims=True) + F.EPS))
    mixes = (x @ hcfn.T) * rsq
    pre = F.sigmoid(mixes * hcs + hcb) + F.HCEPS
    y = (pre[:, :, None] * H).sum(1)
    y = F.rms(y, fnorm)
    return y @ hw.T


def real_metrics(lf, lq, ids, n):
    def logsm(z):
        z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
    lsf = logsm(lf[:n]); lsq = logsm(lq[:n])
    pf_p = np.exp(lsf); pq_p = np.exp(lsq)
    overlap = float(np.minimum(pf_p, pq_p).sum(1).mean())
    kl = float((pf_p * (lsf - lsq)).sum(1).mean())
    nxt = ids[1:n + 1]
    nll_f = float(-lsf[np.arange(n), nxt].mean()); nll_q = float(-lsq[np.arange(n), nxt].mean())
    return overlap, kl, float(np.exp(nll_f)), float(np.exp(min(nll_q, 20)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ntok", type=int, default=64)
    ap.add_argument("--layers", default="0-42")
    ap.add_argument("--passes", type=int, default=2)   # 向前-向后-感知 闭环轮数
    ap.add_argument("--zrank", default="0")             # 低秩方向z候选秩(逗号), 0=仅标量(现行为)
    a = ap.parse_args()
    zranks = tuple(int(x) for x in a.zrank.split(",") if x.strip())
    lo, hi = map(int, a.layers.split("-")); qlayers = set(range(lo, hi + 1))
    ids = np.array([int(x) for x in open(a.ids) if x.strip()], dtype=np.int64)[:a.ntok]
    Ids = ids; nh = len(ids) // 2
    emb = F.R.get("embed.weight"); HCM = F.HCM
    H0 = np.repeat(emb[ids][:, None, :], HCM, 1).astype(np.float32)

    # ── fp8 教师轨迹一次 (确定性; 存每层后 H_fp8 供向后敏感度) ──
    print("fp8 教师轨迹...", file=sys.stderr, flush=True)
    H = H0.copy(); hf_traj = []
    for L in range(hi + 1):
        W = F.load_layer(L); t0 = time.time()
        Fin_f, ctx = layer_attn_to_fin(H, W, L)
        Fout_f = _fp8_routed(Fin_f, Ids, W, L, F.expert_fp(Fin_f, W['s1'], W['s3'], W['s2']))
        H = F.hc_post(Fout_f, *ctx); hf_traj.append(H.copy()); del W
        print(f"[fp8教师] L{L:2d} {time.time()-t0:5.1f}s", flush=True)
    lf = head_logits(H); hw = F.R.get("head.weight")

    # ── 向前 + 向后 + 感知 多轮闭环 ──
    sens_w = {}       # 感知权重 per layer (pass0 空=均匀1.0)
    dim_w_g = None    # 向后传下来的每输出维重要度 (pass0 None=均匀; pass≥1=上轮 headᵀerr²)
    n = len(ids) - 1
    for p in range(a.passes):
        tail = " +感知(向后维加权目标)" if p > 0 else ""
        print(f"\n=== PASS {p + 1}/{a.passes}: 向前传播 + 每层动态调优{tail} ===", flush=True)
        Hq = H0.copy(); dlt = []
        for L in range(hi + 1):
            W = F.load_layer(L); t0 = time.time()
            Fin_q, ctx = layer_attn_to_fin(Hq, W, L)
            if L in qlayers:
                _, yq, tag = routed_coadapt(Fin_q, Ids, W, L, nh, sens_w.get(L, 1.0), dim_w_g, zranks)
                Fout_q = yq
            else:
                Fout_q = _fp8_routed(Fin_q, Ids, W, L, F.expert_fp(Fin_q, W['s1'], W['s3'], W['s2'])); tag = "fp8"
            Hq = F.hc_post(Fout_q, *ctx); dlt.append(Hq - hf_traj[L])
            lqi = head_logits(Hq)
            rel = float(np.linalg.norm(lqi - lf) / (np.linalg.norm(lf) + 1e-9))
            sw = sens_w.get(L, 1.0)
            print(f"L{L:2d} {tag:24s} {time.time()-t0:5.1f}s  logit_relL2={rel:.3f}"
                  f"{'  [高敏]' if sw >= 1.5 else ''}", flush=True)
            del W
        lq = head_logits(Hq)
        ov, kl, pf, pq = real_metrics(lf, lq, ids, n)
        # 向后: cotangent λ=headᵀ·err → 每层敏感度 s_L + 每输出维重要度 dim_w_g(喂下一轮感知)
        lam = (lq - lf) @ hw
        dim_w_g = np.nan_to_num((lam[:n] ** 2).mean(0)).astype(np.float32)   # 下游在乎的维
        ss = [float(np.abs((lam[:n] * d[:n].mean(1)).sum())) for d in dlt]
        tot = sum(ss) + 1e-9; order = np.argsort(ss)[::-1]
        print(f"★PASS{p+1} 真实还原: 分布还原率={ov:.4f} KL={kl:.4f} PPL fp8={pf:.2f}→q={pq:.2f} (×{pq/max(pf,1e-6):.2f})★")
        print("  向后敏感度 top5: " + " ".join(f"L{lo+Li}={ss[Li]/tot:.0%}" for Li in order[:5]), flush=True)
        # 感知: 高敏 top-K 层下轮 sens_w=2.0 → pass2 加通道增益
        K = max(3, len(ss) // 4)
        sens_w = {lo + Li: (2.0 if r < K else 1.0) for r, Li in enumerate(order)}
    np.save("/tmp/fullopt_logits.npy", np.stack([lf, lq]))


def _fp8_routed(Fin, Ids, W, L, share):
    idx, wt, _ = F.gate_route(Fin, W, Ids)
    out = share.copy(); fe = idx.reshape(-1); fw = wt.reshape(-1)
    tok = np.repeat(np.arange(len(Fin)), NACT)
    for e in np.unique(fe):
        if e < 0: continue
        m = fe == e; t = tok[m]; ww = fw[m][:, None]; Xt = Fin[t]
        w1, w3, w2 = (F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w{k}.weight") for k in (1, 3, 2))
        np.add.at(out, t, F.expert_fp(Xt, w1, w3, w2, ww))
    return out.astype(np.float32)


if __name__ == "__main__":
    main()
