#!/usr/bin/env python3
"""g1b_stack_probe.py — 杠杆栈 held-out 探针(2026-07-25 纠偏版)。
★方法论修正: X 按行分半 fit/eval, 一切指标只报 held-out(eval 行)——杜绝 blk 式 in-sample 假增益★
测设计原样杠杆(v2.1 §1), 不测简化版:
  冷栈: C0 行scale(生产) → C1 权重驱动块scale+行级激活乘子(1 act-DOF, 防过拟合版 B0b)
        → C2 C1+幅度分段(行内按|w|降序 4 段 rank 分组, 段界存 3×u8/行) → C3 C2+显著列1%q8 钉扎
  热栈: H0 go2b(生产) → H1 块对角Walsh+go2b → H2 块对角Walsh+4维VQ-256码本(2bpw, E8/TCQ 类代表)
判据: held-out 行输出 relF + cos; 等体积 bpw 全部显式列出。
用法: g1b_stack_probe.py --layer L --x X.npy [--nh 2 --nc 3] [--out rpt]
"""
import os, sys, json, struct, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "quant"))
from go2b_encode import encode_go2b
from g1a_lever_probe import Shard, open_shard_for_layer, walsh, rot_groups, f8lut

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", file=sys.stderr, flush=True)

def metrics(Y, Yq):
    d = Y - Yq
    return (float(np.linalg.norm(d) / (np.linalg.norm(Y) + 1e-20)),
            float((Y * Yq).sum() / (np.linalg.norm(Y) * np.linalg.norm(Yq) + 1e-20)))

# ---------- 冷栈 ----------
def cold_C0(W, Xf):                          # 生产行 scale: sign + act 锚定单 scale
    S = np.where(W >= 0, 1, -1).astype(np.float32)
    s0 = np.abs(W).mean(1)
    p = Xf @ S.T; y = Xf @ W.T
    sp2 = (p*p).sum(0); spy = (p*y).sum(0)
    lam = np.maximum(sp2/(len(Xf)+1.0), 1e-3*sp2+1e-9)
    s = np.clip((spy+lam*s0)/(sp2+lam), 0, 4*s0)
    return S*s[:,None].astype(np.float32), 1.0625

def cold_C1(W, Xf):
    """权重驱动块 scale(mean|w_b|, 零校准依赖) × 行级激活乘子 g_r(唯一 act-DOF)。"""
    rows, cols = W.shape; nb = cols//256
    S = np.where(W >= 0, 1, -1).astype(np.float32)
    sb = np.stack([np.abs(W[:,b*256:(b+1)*256]).mean(1) for b in range(nb)],1)  # [rows,nb] 权重驱动
    Wq0 = S*np.repeat(sb,256,1)
    p = Xf @ Wq0.T; y = Xf @ W.T                                   # 行级 g 闭式
    sp2=(p*p).sum(0); spy=(p*y).sum(0)
    lam=np.maximum(sp2/(len(Xf)+1.0),1e-3*sp2+1e-9)
    g=np.clip((spy+lam*1.0)/(sp2+lam),0.25,4.0)
    return Wq0*g[:,None].astype(np.float32), 1.0625

def cold_C2(W, Xf, nseg=4):
    """C1 + 行内幅度分段: 每行按|w|四分位划 4 个幅度段(成员=rank 序, 段界 3×u8/行≈0.006bpw),
    每段权重驱动 scale mean|w_seg| × 行级 act 乘子。段成员按列 rank 需存 2bit/元? 不——
    段=|w| 分位阈值上的集合, 成员由阈值+|w|本身推出?? 反量化端无|w| ⇒ 成员必须显式。
    诚实版: 段成员 2bit/元 → bpw 1.06+2 ✗。改用位置分段: 每 64 列一段(4段/256块, 免费),
    scale=mean|w_seg| 权重驱动 → 真等体积(块内 4 子 scale 全 f16 会超; 用 d×比率: 存 f16 主 scale
    + 3×u4 对数比率/块 = +1.5B/块 → 35.5B/256 = 1.109 bpw)。"""
    rows, cols = W.shape; nb = cols//256
    S = np.where(W >= 0, 1, -1).astype(np.float32)
    Wq0 = np.empty_like(W)
    for b in range(nb):
        blk = W[:, b*256:(b+1)*256]
        for s4 in range(4):
            seg = np.abs(blk[:, s4*64:(s4+1)*64]).mean(1)          # [rows] 权重驱动子 scale
            # u4 对数比率量化(相对主 scale)模拟存储代价
            main = np.abs(blk).mean(1)+1e-12
            ratio = np.clip(seg/main, 2**-4, 2**3.5)
            q = np.round(np.log2(ratio)*2)/2                        # 半档对数 u4
            segq = main*(2.0**q)
            Wq0[:, b*256+s4*64:b*256+(s4+1)*64] = S[:, b*256+s4*64:b*256+(s4+1)*64]*segq[:,None]
    p = Xf @ Wq0.T; y = Xf @ W.T
    sp2=(p*p).sum(0); spy=(p*y).sum(0)
    lam=np.maximum(sp2/(len(Xf)+1.0),1e-3*sp2+1e-9)
    g=np.clip((spy+lam*1.0)/(sp2+lam),0.25,4.0)
    return Wq0*g[:,None].astype(np.float32), 1.1094

def pin_top(W, Xf, frac=0.01):
    imp = (Xf*Xf).mean(0)[:W.shape[1]]*(W*W).sum(0)
    k = max(1,int(W.shape[1]*frac)); idx = np.argsort(imp)[-k:]
    Wp = W.copy(); Wp[:,idx]=0.0
    q8 = np.zeros((W.shape[0],k),np.float32)
    for i,j in enumerate(idx):
        s=np.abs(W[:,j]).max()/127.0+1e-20; q8[:,i]=np.round(W[:,j]/s)*s
    return Wp, idx, q8, 0.0002+frac*8

# ---------- 热栈: 4 维 VQ-256 码本(2bpw, 码本 256×4×f16=2KB/矩阵≈0.002bpw) ----------
def hot_vq(W, Xf, dim=4, nc=256, iters=8):
    import math
    rows, cols = W.shape
    V = W.reshape(-1, dim)                                        # [N,4]
    # k-means++ 简化: 随机取样初始化(固定种子) + Lloyd
    rng = np.random.default_rng(0x5EED)
    C = V[rng.choice(len(V), nc, replace=False)].copy()
    sub = V[rng.choice(len(V), min(len(V), 200000), replace=False)]
    def _assign(Vv, bs=16384):
        out = np.empty(len(Vv), np.int64); c2=(C*C).sum(1)
        for i in range(0, len(Vv), bs):
            d = Vv[i:i+bs]@C.T; d *= -2.0; d += c2[None,:]
            out[i:i+bs] = d.argmin(1)
        return out
    for _ in range(iters):
        a = _assign(sub)
        for c in range(nc):
            m = a==c
            if m.any(): C[c]=sub[m].mean(0)
    Wq = C[_assign(V)].reshape(rows, cols).astype(np.float32)
    # 行级 act 乘子(与冷同款 1 DOF)
    p = Xf @ Wq.T; y = Xf @ W.T
    sp2=(p*p).sum(0); spy=(p*y).sum(0)
    lam=np.maximum(sp2/(len(Xf)+1.0),1e-3*sp2+1e-9)
    g=np.clip((spy+lam*1.0)/(sp2+lam),0.25,4.0)
    return Wq*g[:,None].astype(np.float32), math.log2(nc)/dim+0.002

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--x", required=True)
    ap.add_argument("--hf", default="/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base")
    ap.add_argument("--hot-table", default=os.path.join(HERE,"..","corpus","prog_active_top64.txt"))
    ap.add_argument("--nh", type=int, default=2); ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args(); L = a.layer
    X = np.load(a.x).astype(np.float32)
    n = len(X); ifit = np.arange(0,n,2); ieval = np.arange(1,n,2)   # 偶=fit 奇=eval
    Xf, Xe = X[ifit], X[ieval]
    hot = {}
    for ln in open(a.hot_table):
        p2 = ln.split(":"); hot[int(p2[0][1:])] = [int(x) for x in p2[1].split()]
    hids = hot[L][:a.nh]
    cold_all = [e for e in range(256) if e not in set(hot[L])]
    cids = [cold_all[0], cold_all[len(cold_all)//2], cold_all[-1]][:a.nc]
    sh = open_shard_for_layer(a.hf, L)
    Hseq = walsh(256, True)
    out=[]
    def emit(r): out.append(r); log("  "+" ".join(map(str,r)))
    log(f"held-out 栈探针 L{L}: hot={hids} cold={cids} fit={len(Xf)} eval={len(Xe)}")
    os.environ["DS4_GO2B_ACT_SCALE"]="1"
    for e in cids:
        W = sh.expert_w(L, e, "w1"); Ye = Xe @ W.T
        for tag, fn in (("C0_row", cold_C0), ("C1_wblk", cold_C1), ("C2_wseg", cold_C2)):
            Wq, bpw = fn(W, Xf)
            rel, cos = metrics(Ye, Xe @ Wq.T)
            emit((f"L{L}","w1",f"e{e}","cold",tag,f"{bpw:.4f}",f"{rel:.4f}",f"{cos:.5f}"))
        Wqv, bpwv = hot_vq(W, Xf, dim=8, nc=256)               # 冷二值码本类: 8bit/8w=1.0bpw
        rel, cos = metrics(Ye, Xe @ Wqv.T)
        emit((f"L{L}","w1",f"e{e}","cold","C4_vq8x256",f"{bpwv:.4f}",f"{rel:.4f}",f"{cos:.5f}"))
        Wr2 = rot_groups(W, Hseq); Xfr2 = rot_groups(Xf, Hseq); Xer2 = rot_groups(Xe, Hseq)
        Wqv, bpwv = hot_vq(Wr2, Xfr2, dim=8, nc=256)
        rel, cos = metrics(Ye, Xer2 @ Wqv.T)
        emit((f"L{L}","w1",f"e{e}","cold","C5_walsh_vq8x256",f"{bpwv:.4f}",f"{rel:.4f}",f"{cos:.5f}"))
        Wp, idx, q8, extra = pin_top(W, Xf)
        Wq, bpw = cold_C2(Wp, Xf)
        rel, cos = metrics(Ye, Xe @ Wq.T + Xe[:,idx] @ q8.T)
        emit((f"L{L}","w1",f"e{e}","cold","C3_wseg+pin1%",f"{bpw+extra:.4f}",f"{rel:.4f}",f"{cos:.5f}"))
    for e in hids:
        W = sh.expert_w(L, e, "w1"); Ye = Xe @ W.T
        blk, Wq = encode_go2b(W, Xh=Xf, mode="nf")
        rel, cos = metrics(Ye, Xe @ Wq.T)
        emit((f"L{L}","w1",f"e{e}","hot","H0_go2b","2.1250",f"{rel:.4f}",f"{cos:.5f}"))
        Wr = rot_groups(W, Hseq); Xfr = rot_groups(Xf, Hseq); Xer = rot_groups(Xe, Hseq)
        blk, Wqr = encode_go2b(Wr, Xh=Xfr, mode="nf")
        rel, cos = metrics(Ye, Xer @ Wqr.T)
        emit((f"L{L}","w1",f"e{e}","hot","H1_walsh_go2b","2.1250",f"{rel:.4f}",f"{cos:.5f}"))
        Wq2, bpw = hot_vq(Wr, Xfr)
        rel, cos = metrics(Ye, Xer @ Wq2.T)
        emit((f"L{L}","w1",f"e{e}","hot","H2_walsh_vq4x256",f"{bpw:.4f}",f"{rel:.4f}",f"{cos:.5f}"))
    print("\nlayer kind expert lane variant bpw relF_heldout cos")
    for r in out: print(" ".join(map(str,r)))
    import collections
    agg = collections.defaultdict(list)
    for r in out: agg[(r[3],r[4])].append(float(r[6]))
    print("\n== held-out 汇总(relF 均值) ==")
    for k in sorted(agg): print(f"{k[0]:4s} {k[1]:16s} relF={np.mean(agg[k]):.4f} n={len(agg[k])}")
    if a.out:
        with open(a.out,"w") as f:
            f.write("layer kind expert lane variant bpw relF_heldout cos\n")
            for r in out: f.write(" ".join(map(str,r))+"\n")
        log(f"报告 → {a.out}")

if __name__ == "__main__":
    main()
