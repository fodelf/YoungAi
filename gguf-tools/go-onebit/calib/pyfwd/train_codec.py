#!/usr/bin/env python3
# Step 2: DEEP Go-customized restoration codec (NOT a replacement model).
#   o_restored = o_base + D(LN(o_base), [LN(x)], z^L)   -- anchored on the 1-bit base output ô.
#   D = shared deep nonlinear net (the Go-customized high->low->high transformation).
#   z^L = tiny per-layer hidden variable (Embedding). Fit with the four losses, held-out.
# Capability stays in the original (ô); D+z restore the precision 1-bit discarded.
# Two variants via USE_X: D(ô,z) pure restoration vs D(ô,x,z). Runs on M1 (torch+MPS).
import os, numpy as np, torch, torch.nn as nn
CAP   = os.environ.get("CAP",   "/Users/fodelf/ds4-main/cap_m1")
OBASE = os.environ.get("OBASE", "/Users/fodelf/ds4-main/obase")
OUT   = os.environ.get("OUT",   "/Users/fodelf/ds4-main/codec"); os.makedirs(OUT, exist_ok=True)
d_model = 4096
DZ    = int(os.environ.get("DZ", "64"))
H     = int(os.environ.get("H", "2048"))
USE_X = os.environ.get("USE_X", "1") == "1"
N     = int(os.environ.get("N", "3000"))
NTR   = int(N * 0.8)
EP    = int(os.environ.get("EP", "80"))
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
print(f"train_codec dev={dev} USE_X={USE_X} DZ={DZ} H={H} N={N} EP={EP}", flush=True)

OB=[]; X=[]; O=[]
for L in range(43):
    OB.append(np.load(f"{OBASE}/obase_L{L}.npy")[:N].astype(np.float16))
    X.append(np.load(f"{CAP}/ffn_in_L{L}.npy")[:N].astype(np.float16))
    O.append(np.load(f"{CAP}/ffn_out_L{L}.npy")[:N].astype(np.float16))
OB=np.stack(OB); X=np.stack(X); O=np.stack(O)              # [43,N,4096] fp16

class Codec(nn.Module):
    def __init__(self):
        super().__init__()
        self.z=nn.Embedding(43,DZ)
        self.ln_ob=nn.LayerNorm(d_model); self.ln_x=nn.LayerNorm(d_model)
        ind=d_model+DZ+(d_model if USE_X else 0)
        self.net=nn.Sequential(nn.Linear(ind,H),nn.GELU(),nn.LayerNorm(H),
                               nn.Linear(H,H),nn.GELU(),nn.LayerNorm(H),
                               nn.Linear(H,d_model))
    def forward(self,ob,x,Lidx):
        z=self.z(Lidx); feats=[self.ln_ob(ob),z]+([self.ln_x(x)] if USE_X else [])
        return ob + self.net(torch.cat(feats,-1))          # anchor + learned correction

def cosfn(a,b): return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)+1e-6)
net=Codec().to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=1.5e-3,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EP)

def batch(split,bs=4096):
    Ls=np.random.randint(0,43,bs); ts=np.random.randint(0,NTR,bs) if split=="tr" else np.random.randint(NTR,N,bs)
    g=lambda A: torch.tensor(A[Ls,ts].astype(np.float32),device=dev)
    return g(OB),g(X),g(O),torch.tensor(Ls,device=dev)

for ep in range(EP):
    net.train()
    for it in range(60):
        ob,x,o,L=batch("tr"); orr=net(ob,x,L)
        align=(1-cosfn(orr,o)).mean()                                   # 1) alignment (direction)
        classify=((orr-o)**2).sum(-1).mean()/((o**2).sum(-1).mean()+1e-6) # 2) value/routing fidelity
        orr2=net(ob+torch.randn_like(ob)*0.05*ob.std(),x+torch.randn_like(x)*0.05*x.std(),L)
        smooth=((orr2-orr)**2).mean()/((orr**2).mean()+1e-6)            # 3) smoothness
        fixed=(net.z(L)**2).mean()                                      # 4) fixed (z regularization) + wd
        (align+0.5*classify+0.1*smooth+1e-3*fixed).backward(); opt.step(); opt.zero_grad()
    sch.step()
    if ep%10==9 or ep==EP-1:
        net.eval()
        with torch.no_grad():
            b=[]; c=[]
            for L in range(43):
                ob=torch.tensor(OB[L,NTR:N].astype(np.float32),device=dev)
                x=torch.tensor(X[L,NTR:N].astype(np.float32),device=dev)
                o=torch.tensor(O[L,NTR:N].astype(np.float32),device=dev)
                Lt=torch.full((N-NTR,),L,device=dev,dtype=torch.long)
                b.append(cosfn(ob,o).mean().item()); c.append(cosfn(net(ob,x,Lt),o).mean().item())
            print(f"ep{ep} held-out cos: base={np.mean(b):.3f} -> codec={np.mean(c):.3f}  (L0 {b[0]:.2f}->{c[0]:.2f} L21 {b[21]:.2f}->{c[21]:.2f} L42 {b[42]:.2f}->{c[42]:.2f})",flush=True)
torch.save(net.state_dict(),f"{OUT}/codec_{'x' if USE_X else 'nox'}_dz{DZ}.pt")
print("ALL DONE saved",flush=True)
