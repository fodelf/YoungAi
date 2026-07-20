#!/usr/bin/env python3
# Option-2: train one small Go generator per layer  G_L: ffn_in -> ffn_out  (replaces the
# MoE for the Go domain), with the four losses, save weights (numpy) for the forward + later GGUF.
#   align=1-cos | classify=MSE | smooth=||G(x+δ)-G(x)||² | fixed=weight-decay+dropout+dither
# Pure cap_m1 (no model run). Run in M4 /tmp/go_venv (torch + MPS).
import os, numpy as np, torch, torch.nn as nn
CAP = os.environ.get("CAP", "/private/tmp/m1_ds4/cap_m1")
OUT = os.environ.get("OUT", "/tmp/gen"); os.makedirs(OUT, exist_ok=True)
H, NB = int(os.environ.get("H", "2048")), int(os.environ.get("NB", "3"))
EP, BS = int(os.environ.get("EP", "120")), 256
NTR = int(os.environ.get("NTR", "11500"))      # train tokens; rest held-out for the per-layer report
DROP, WD, DITH = 0.1, 1e-3, 0.05
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
print(f"gen_train dev={dev} H={H} NB={NB} EP={EP} -> {OUT}", flush=True)

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
    n = (P*Y).sum(1); dp = np.linalg.norm(P, axis=1); dy = np.linalg.norm(Y, axis=1)
    m = (dp > 0) & (dy > 0); return float((n[m]/(dp[m]*dy[m])).mean())

for L in range(43):
    X = np.load(f"{CAP}/ffn_in_L{L}.npy").astype(np.float32)
    Y = np.load(f"{CAP}/ffn_out_L{L}.npy").astype(np.float32)
    Xtr, Xte = X[:NTR], X[NTR:]; Ytr, Yte = Y[:NTR], Y[NTR:]
    xm, xs = Xtr.mean(0), Xtr.std(0)+1e-6; ym, ys = Ytr.mean(0), Ytr.std(0)+1e-6
    xt = torch.tensor((Xtr-xm)/xs, device=dev); yt = torch.tensor((Ytr-ym)/ys, device=dev)
    xv = torch.tensor((Xte-xm)/xs, device=dev)
    net = ResMLP().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP)
    n = xt.shape[0]; best = -1.0
    for ep in range(EP):
        net.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]; xb = xt[idx] + torch.randn_like(xt[idx])*DITH; yb = yt[idx]
            out = net(xb)
            cos = (out*yb).sum(1)/(out.norm(dim=1)*yb.norm(dim=1)+1e-8)
            loss = (1-cos).mean() + ((out-yb)**2).mean()
            out2 = net(xb + torch.randn_like(xb)*DITH); loss = loss + 0.1*((out2-out)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep % 20 == 19 or ep == EP-1:
            net.eval()
            with torch.no_grad(): pv = net(xv).cpu().numpy()*ys+ym
            best = max(best, rowcos(pv, Yte))
    # save weights as numpy (for the numpy forward + later GGUF emit)
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    np.savez(f"{OUT}/gen_L{L}.npz", xm=xm, xs=xs, ym=ym, ys=ys, **sd)
    print(f"L{L:2d} held-out cos={best:.3f}  saved", flush=True)
print("ALL 43 GENERATORS TRAINED", flush=True)
