#!/usr/bin/env python3
"""free3part_poc.py — 三段式非线性行为生成器 POC 复现(2026-08-08)。

严格按 wave-171(2026-06-26)已验证方案复现, 非新设计:
  ①Dyn(x, z_e): 共享非线性动态网络, per-expert 嵌入 z_e 作条件
  ②Fixed:      共享固定输出头(小输出 init)
  目标 = 修正量 ΔY(x,e) = FP 专家输出 − q2 专家输出(拟合行为, 不拟合权重 →
         不受权重秩/正交墙约束 — wave-171 的关键机制)
  四损失: task=MSE + align(方向) + smooth(输入扰动稳定) + fixed(dither 增广)
  权重 1.0/0.5/0.15/0.05(与量化器 z_pick_rank 同比例)
  训练纪律(wave-171 原文): lr≤5e-4, grad-clip, 小输出 init; 容量勿大(emb512/h2048 曾过拟合)
切分: 按 token 切(前 80% token 训练, 后 20% token 的全部对 held-out = 真泛化口径)。
用法: free3part_poc.py <pairs_dir> [epochs=40] [device=mps]
"""
import os, sys, math
import numpy as np
import torch, torch.nn as nn

D = 4096

def main():
    pd = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    dev = sys.argv[3] if len(sys.argv) > 3 else ("mps" if torch.backends.mps.is_available() else "cpu")
    X = np.fromfile(f"{pd}/X.bin", dtype=np.float32).reshape(-1, D)
    E = np.fromfile(f"{pd}/E.bin", dtype=np.int32)
    Y = np.fromfile(f"{pd}/DY.bin", dtype=np.float32).reshape(-1, D)
    N = len(E)
    # 按 token 切: 对级数据由逐专家拼接而成, token 身份=X 行的重复模式 → 用 X 哈希分组
    key = (X[:, :8] * 1e4).round().astype(np.int64).sum(1)
    uniq = np.unique(key)
    rng = np.random.RandomState(0); rng.shuffle(uniq)
    tr_tok = set(uniq[:int(len(uniq)*0.8)].tolist())
    tr = np.array([k in tr_tok for k in key]); ho = ~tr
    print(f"对 {N}(token {len(uniq)}) → 训练 {tr.sum()} / held-out {ho.sum()}  设备={dev}")

    class Gen(nn.Module):
        def __init__(s, emb=128, h=512):
            super().__init__()
            s.e = nn.Embedding(256, emb)
            s.dyn = nn.Sequential(nn.Linear(D+emb, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
            s.out = nn.Linear(h, D)
            nn.init.normal_(s.out.weight, std=1e-3); nn.init.zeros_(s.out.bias)   # 小输出 init
        def forward(s, x, e):
            return s.out(s.dyn(torch.cat([x, s.e(e)], -1)))

    m = Gen().to(dev)
    npar = sum(p.numel() for p in m.parameters())
    print(f"生成器参数 {npar/1e6:.2f}M(fp16 侧车 ≈ {npar*2/2**20:.1f} MiB/层)")
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)
    Xt = torch.tensor(X[tr], device=dev); Et = torch.tensor(E[tr].astype(np.int64), device=dev)
    Yt = torch.tensor(Y[tr], device=dev)
    Xh = torch.tensor(X[ho], device=dev); Eh = torch.tensor(E[ho].astype(np.int64), device=dev)
    Yh = torch.tensor(Y[ho], device=dev)
    ycal = float((Yt**2).mean().sqrt())
    bs = 1024
    for ep in range(epochs):
        perm = torch.randperm(len(Et), device=dev)
        tot = 0.0
        for i in range(0, len(Et), bs):
            idx = perm[i:i+bs]
            x, e, y = Xt[idx], Et[idx], Yt[idx]
            rms = x.pow(2).mean(-1, keepdim=True).sqrt()
            delta = torch.randn_like(x) * 0.04 * rms                      # dither 同口径
            p = m(x, e)
            task = ((p - y)**2).mean()
            cos = nn.functional.cosine_similarity(p, y, dim=-1)
            align = (1 - cos).mean()
            p2 = m(x + delta, e)
            smooth = ((p2 - p)**2).mean()
            fixed = ((p2 - y)**2).mean()                                  # dither 增广行同目标
            loss = 1.0*task + 0.5*align*ycal**2 + 0.15*smooth + 0.05*fixed
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); tot += float(task)*len(idx)
        if (ep+1) % 5 == 0 or ep == 0:
            with torch.no_grad():
                ph = m(Xh, Eh)
                res = ((Yh-ph)**2).sum(); base = (Yh**2).sum()
                rec = float(1 - res/base)
                cosh = float(nn.functional.cosine_similarity(ph, Yh, dim=-1).mean())
            print(f"ep{ep+1:>3}  train task={tot/len(Et):.5f}  held-out 挽回={rec*100:5.1f}%  cos={cosh:.3f}")
    # 终报: 修正前后专家输出相对误差
    with torch.no_grad():
        ph = m(Xh, Eh)
        rec = float(1 - ((Yh-ph)**2).sum()/(Yh**2).sum())
    print(f"\n★ 终判(held-out): 误差能量挽回 {rec*100:.1f}%  |  生成器 {npar*2/2**20:.1f} MiB(fp16)/层")
    print(f"  等效 Δbpw = {0.5*math.log2(1/(1-rec)) if 0<rec<1 else 0:.3f}(基础 2.26bpw 之上)")

if __name__ == "__main__":
    main()
