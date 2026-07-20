#!/usr/bin/env python3
# Proper deep per-layer distillation probe (discovery; Python+training OK per user).
# Tests the user's claim: programming rigor => a real (complex, nonlinear) regularity
# exists; a PROPER deep net (not linear/sklearn) should learn each layer's Go function
# ffn_in -> ffn_out far better than the weak baselines. Each layer gets its OWN model.
# Pure cap_m1 (no model run). Run in M1 cap_work venv (needs torch; uses MPS if present).
import os, math, numpy as np, torch, torch.nn as nn

CAP = os.environ.get("CAP", "/Users/fodelf/ds4-main/cap_m1")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "0,12,21").split(",")]
NTR = int(os.environ.get("NTR", "10000"))
H   = int(os.environ.get("H", "2048"))
NB  = int(os.environ.get("NB", "3"))
EP  = int(os.environ.get("EP", "200"))
BS  = int(os.environ.get("BS", "256"))
dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device={dev}  H={H} NB={NB} EP={EP} NTR={NTR}", flush=True)

class ResMLP(nn.Module):
    def __init__(self, d=4096, h=H, nb=NB):
        super().__init__()
        self.inp = nn.Linear(d, h)
        drop = float(os.environ.get("DROP", "0.0"))
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, h))
            for _ in range(nb)])
        self.out = nn.Linear(h, d)
    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = x + b(x)
        return self.out(x)

def rowcos(P, Y):  # mean per-row cosine, raw scale, numpy float64
    P = P.astype(np.float64); Y = Y.astype(np.float64)
    num = (P*Y).sum(1); dp = np.linalg.norm(P, axis=1); dy = np.linalg.norm(Y, axis=1)
    m = (dp > 0) & (dy > 0)
    return float((num[m]/(dp[m]*dy[m])).mean())

print(f"{'L':>3} {'best_cos':>9} {'R2':>7} {'epoch':>6}", flush=True)
for L in LAYERS:
    X = np.load(f"{CAP}/ffn_in_L{L}.npy").astype(np.float32)
    Y = np.load(f"{CAP}/ffn_out_L{L}.npy").astype(np.float32)
    TEST_FROM = int(os.environ.get("TEST_FROM", str(NTR)))   # fixed held-out for learning curves
    Xtr, Xte = X[:NTR], X[TEST_FROM:]; Ytr, Yte = Y[:NTR], Y[TEST_FROM:]
    xm, xs = Xtr.mean(0), Xtr.std(0)+1e-6
    ym, ys = Ytr.mean(0), Ytr.std(0)+1e-6
    Xtr2 = ((Xtr-xm)/xs); Xte2 = ((Xte-xm)/xs)
    Ytr2 = ((Ytr-ym)/ys)                              # train target standardized (stable)
    xt = torch.tensor(Xtr2, device=dev); yt = torch.tensor(Ytr2, device=dev)
    xv = torch.tensor(Xte2, device=dev)
    net = ResMLP().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=float(os.environ.get("WD","1e-4")))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP)
    lossf = nn.MSELoss()
    n = xt.shape[0]; best = -1.0; bep = 0
    for ep in range(EP):
        net.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]
            opt.zero_grad(); out = net(xt[idx]); loss = lossf(out, yt[idx])
            loss.backward(); opt.step()
        sched.step()
        if ep % 10 == 9 or ep == EP-1:
            net.eval()
            with torch.no_grad():
                pv = net(xv).cpu().numpy()
            pv_raw = pv*ys + ym                       # un-standardize -> raw scale
            c = rowcos(pv_raw, Yte)
            if c > best: best = c; bep = ep
    # final R2 on raw
    ss_res = ((pv_raw-Yte)**2).sum(); ss_tot = ((Yte-Ytr.mean(0))**2).sum()
    r2 = 1.0 - ss_res/ss_tot
    print(f"{L:>3} {best:>9.3f} {r2:>7.3f} {bep:>6}", flush=True)
