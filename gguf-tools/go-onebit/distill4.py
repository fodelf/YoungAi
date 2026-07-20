#!/usr/bin/env python3
# Four-loss per-layer Go distiller (ALGORITHM.md §4): re-incorporates the user's
# original four losses into the validated ffn_in->ffn_out distillation.
#   align    = 1 - cos(y_hat, y)          (direction = the held-out metric)
#   classify = MSE on standardized y      (value/magnitude -> downstream routing)
#   smooth   = ||M(x+delta) - M(x)||^2     (input robustness / no jumps)
#   fixed    = weight-decay + dropout + fixed-seed input dither (anti-overfit)
# Pure cap_m1 (no model run). Run in M1 cap_work venv (torch + MPS).
import os, numpy as np, torch, torch.nn as nn

CAP = os.environ.get("CAP", "/Users/fodelf/ds4-main/cap_m1")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "0,21").split(",")]
NTR = int(os.environ.get("NTR", "11000"))
TEST_FROM = int(os.environ.get("TEST_FROM", "11500"))
H, NB = int(os.environ.get("H", "2048")), int(os.environ.get("NB", "3"))
EP, BS = int(os.environ.get("EP", "250")), int(os.environ.get("BS", "256"))
DROP, WD = float(os.environ.get("DROP", "0.1")), float(os.environ.get("WD", "1e-3"))
W_A = float(os.environ.get("W_A", "1.0"))    # align
W_C = float(os.environ.get("W_C", "1.0"))    # classify (mse)
W_S = float(os.environ.get("W_S", "0.1"))    # smooth
DITH = float(os.environ.get("DITH", "0.05")) # fixed/smooth dither std (standardized space)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
print(f"device={dev} H={H} NB={NB} EP={EP} W_A={W_A} W_C={W_C} W_S={W_S} DROP={DROP} WD={WD}", flush=True)

class ResMLP(nn.Module):
    def __init__(self, d=4096, h=H, nb=NB):
        super().__init__()
        self.inp = nn.Linear(d, h); self.out = nn.Linear(h, d)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(DROP), nn.Linear(h, h))
            for _ in range(nb)])
    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks: x = x + b(x)
        return self.out(x)

def rowcos(P, Y):
    P, Y = P.astype(np.float64), Y.astype(np.float64)
    num = (P*Y).sum(1); dp = np.linalg.norm(P, 1, axis=1) if False else np.linalg.norm(P, axis=1)
    dy = np.linalg.norm(Y, axis=1); m = (dp > 0) & (dy > 0)
    return float((num[m]/(dp[m]*dy[m])).mean())

print(f"{'L':>3} {'best_cos':>9} {'epoch':>6}", flush=True)
for L in LAYERS:
    X = np.load(f"{CAP}/ffn_in_L{L}.npy").astype(np.float32)
    Y = np.load(f"{CAP}/ffn_out_L{L}.npy").astype(np.float32)
    Xtr, Xte = X[:NTR], X[TEST_FROM:]; Ytr, Yte = Y[:NTR], Y[TEST_FROM:]
    xm, xs = Xtr.mean(0), Xtr.std(0)+1e-6; ym, ys = Ytr.mean(0), Ytr.std(0)+1e-6
    xt = torch.tensor((Xtr-xm)/xs, device=dev); yt = torch.tensor((Ytr-ym)/ys, device=dev)
    xv = torch.tensor((Xte-xm)/xs, device=dev)
    net = ResMLP().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=WD)     # fixed: weight decay
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP)
    n = xt.shape[0]; best, bep = -1.0, 0
    for ep in range(EP):
        net.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]; xb = xt[idx]; yb = yt[idx]
            xb = xb + torch.randn_like(xb)*DITH            # fixed: input dither augmentation
            out = net(xb)
            cos = (out*yb).sum(1) / (out.norm(dim=1)*yb.norm(dim=1) + 1e-8)
            L_align = (1 - cos).mean()                      # align
            L_classify = ((out - yb)**2).mean()             # classify (value match)
            out2 = net(xb + torch.randn_like(xb)*DITH)
            L_smooth = ((out2 - out)**2).mean()             # smooth (input robustness)
            loss = W_A*L_align + W_C*L_classify + W_S*L_smooth
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 10 == 9 or ep == EP-1:
            net.eval()
            with torch.no_grad(): pv = net(xv).cpu().numpy()
            c = rowcos(pv*ys + ym, Yte)
            if c > best: best, bep = c, ep
    print(f"{L:>3} {best:>9.3f} {bep:>6}", flush=True)
