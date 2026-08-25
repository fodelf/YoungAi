#!/usr/bin/env python3
"""probe_layer_behavior.py — 【整层行为误差】ΔH 的谱实测(2026-08-08)。

前序实测已否决"专家级 z": 各专家 ΔY_e 子空间互不共享(held-out = 随机基)。
但用户设计的是【每层一份】z —— 整层 MoE 输出 = Σ_e w_e·Expert_e(x)(6/256 稀疏激活,
经 SwiGLU + down + 路由加权), 各专家误差在求和中可能坍缩到低维。
ΔH = MoE_fp(x) − MoE_quant(x) 的谱, 才是"高维行为"的正确定义, 也只需要一份 z。

数据: 锚 fin(层输入)+ ridx/rw(真实路由与权重)+ HF 原始权重 + vq 侧车量化权重。
用法: probe_layer_behavior.py <hf> <layers_dir> <anchor> <层> [tok=256]
"""
import os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_behavior_spectrum import st_index, st_mxfp4, vq_slot, vq_dequant

MOEI, D = 2048, 4096

def anchor_layer(ap, L, ntok):
    """返回 fin[ntok,D], ridx[ntok,NACT], rw[ntok,NACT]"""
    with open(ap, "rb") as f:
        hd = struct.unpack("<8I", f.read(32))
        S, HCM, DIM, NL, V, NACT = hd[1], hd[2], hd[3], hd[4], hd[5], hd[6]
        fin_off = 40
        ridx_off = fin_off + NL*S*DIM*4
        rw_off = ridx_off + NL*S*NACT*4
        f.seek(fin_off + (L*S)*DIM*4)
        fin = np.frombuffer(f.read(ntok*DIM*4), dtype=np.float32).reshape(ntok, DIM)
        f.seek(ridx_off + (L*S)*NACT*4)
        ridx = np.frombuffer(f.read(ntok*NACT*4), dtype=np.int32).reshape(ntok, NACT)
        f.seek(rw_off + (L*S)*NACT*4)
        rw = np.frombuffer(f.read(ntok*NACT*4), dtype=np.float32).reshape(ntok, NACT)
    return fin.copy(), ridx.copy(), rw.copy(), NACT

def swiglu(g, u, lim=None):
    # ★解算截断=加权旋钮(2026-08-13 实测翻案): ±10=回放对齐, ±60=旧行为——A尺反而最好
    # (宽截断=解算目标强调大激活行=更好的尾部加权)。DS4_ZL_SWLIM 扫参, 默认 10(产线对齐)。
    if lim is None:
        lim = float(os.getenv("DS4_ZL_SWLIM", "10"))
    if lim > 0:
        g = np.clip(g, -lim, lim); u = np.clip(u, -lim, lim)
    return (g / (1.0 + np.exp(-np.clip(g, -60, 60)))) * u

def main():
    hf, ld, ap, L = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    ntok = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    wmap = st_index(hf)
    X, ridx, rw, NACT = anchor_layer(ap, L, ntok)
    blob = open(os.path.join(ld, f"dql_vq_L{L:02d}.bin"), "rb").read()
    need = sorted(set(int(e) for e in ridx.reshape(-1)))
    print(f"L{L}: {ntok} token × top-{NACT} → 触碰 {len(need)} 个专家; 逐专家 FP/量化两路前向…")

    Hfp = np.zeros((ntok, D), dtype=np.float64)
    Hq  = np.zeros((ntok, D), dtype=np.float64)
    done = 0
    for e in need:
        rows, slots = np.where(ridx == e)
        if len(rows) == 0: continue
        xs = X[rows].astype(np.float32)                    # [n, D]
        w  = rw[rows, slots].astype(np.float64)[:, None]   # [n,1]
        parts = {}
        for nm, which in (("w1", 0), ("w3", 1), ("w2", 2)):
            tn = f"layers.{L}.ffn.experts.{e}.{nm}.weight"
            Wf = st_mxfp4(hf, wmap, tn)
            off = vq_slot(blob, e, which)
            Wq = vq_dequant(blob, off) if off else None
            if Wq is not None and Wf.shape != Wq.shape: Wf = Wf.T
            parts[nm] = (Wf, Wq)
        # FP 路
        g = xs @ parts["w1"][0].T; u = xs @ parts["w3"][0].T
        Hfp[rows] += w * (swiglu(g, u) @ parts["w2"][0].T)
        # 量化路(冷 w2 无 vq 槽 ⇒ 用 FP w2, 只隔离 w1/w3 的量化影响)
        W1q = parts["w1"][1] if parts["w1"][1] is not None else parts["w1"][0]
        W3q = parts["w3"][1] if parts["w3"][1] is not None else parts["w3"][0]
        W2q = parts["w2"][1] if parts["w2"][1] is not None else parts["w2"][0]
        gq = xs @ W1q.T; uq = xs @ W3q.T
        Hq[rows] += w * (swiglu(gq, uq) @ W2q.T)
        done += 1
        if done % 32 == 0: print(f"  …{done}/{len(need)}", file=sys.stderr)

    dH = Hfp - Hq
    rel = np.linalg.norm(dH) / (np.linalg.norm(Hfp) + 1e-12)
    s = np.linalg.svd(dH, compute_uv=False); e2 = s**2; cum = np.cumsum(e2)/e2.sum()
    print(f"\n整层 MoE 输出误差 ΔH [{ntok}×{D}]  相对误差 {rel:.4f}")
    KS = [1,2,4,8,16,32,64,128,256]
    KS = [k for k in KS if k <= len(s)]
    print("  k       " + "  ".join(f"{k:>5}" for k in KS))
    print("  挽回    " + "  ".join(f"{cum[k-1]*100:4.1f}%" for k in KS))
    print("  随机基  " + "  ".join(f"{min(k/D,1)*100:4.1f}%" for k in KS))
    print(f"\n(对照: 单专家 ΔY 谱 k=8 24.8% / k=128 80.3%, 但跨专家不共享 held-out=随机)")

if __name__ == "__main__" and "--export" not in sys.argv:
    main()

# ── 数据导出(z_leverage_test.c 用): X.bin / R.bin ──
def export_for_z(hf, ld, ap, L, ntok, outdir):
    """整层 X(激活) 与 R(FP−量化 的层输出修正目标)导出为 f32 二进制"""
    import numpy as np, os
    wmap = st_index(hf)
    X, ridx, rw, NACT = anchor_layer(ap, L, ntok)
    blob = open(os.path.join(ld, f"dql_vq_L{L:02d}.bin"), "rb").read()
    need = sorted(set(int(e) for e in ridx.reshape(-1)))
    Hfp = np.zeros((ntok, D)); Hq = np.zeros((ntok, D))
    for i, e in enumerate(need):
        rows, slots = np.where(ridx == e)
        if len(rows) == 0: continue
        xs = X[rows].astype(np.float32); w = rw[rows, slots].astype(np.float64)[:, None]
        P = {}
        for nm, which in (("w1",0),("w3",1),("w2",2)):
            Wf = st_mxfp4(hf, wmap, f"layers.{L}.ffn.experts.{e}.{nm}.weight")
            off = vq_slot(blob, e, which); Wq = vq_dequant(blob, off) if off else None
            if Wq is not None and Wf.shape != Wq.shape: Wf = Wf.T
            P[nm] = (Wf, Wq if Wq is not None else Wf)
        Hfp[rows] += w * (swiglu(xs @ P["w1"][0].T, xs @ P["w3"][0].T) @ P["w2"][0].T)
        Hq[rows]  += w * (swiglu(xs @ P["w1"][1].T, xs @ P["w3"][1].T) @ P["w2"][1].T)
        if (i+1) % 50 == 0: print(f"    …{i+1}/{len(need)}", file=sys.stderr)
    os.makedirs(outdir, exist_ok=True)
    X.astype(np.float32).tofile(os.path.join(outdir, "X.bin"))
    (Hfp - Hq).astype(np.float32).tofile(os.path.join(outdir, "R.bin"))
    print(f"导出 {outdir}: X[{ntok}×{D}] R[{ntok}×{D}]  ΔH相对={np.linalg.norm(Hfp-Hq)/np.linalg.norm(Hfp):.4f}")

if __name__ == "__main__" and "--export" in sys.argv:
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    export_for_z(a[0], a[1], a[2], int(a[3]), int(a[4]) if len(a) > 4 else 512,
                 a[5] if len(a) > 5 else "/tmp/zdata")
