#!/usr/bin/env python3
# train_z_sentinel.py — 训练 z 可行性哨兵（体积锁死下的破墙判决）。
# 读闭式求解器的 sel_L{n}.bin（同数据同切分=严格可比），在 TE 上对比：
#   arm0 base        : cos(Ŷ, Y*)                       —— 1-bit 裸误差
#   arm1 linear-SGD  : rank-64 线性 (U,V) 梯度训练       —— 排除"闭式没解好"假设
#   arm2 MLP-trained : 4096->256->4096 GELU 非线性校正器 —— 破线性秩墙的主判
# 预注册判据：arm2 TE cos ≥ 闭式 TE cos + 0.05 → GO（墙可破，值得建运行时 kernel）
# 用法: venv/bin/python train_z_sentinel.py sel_L8.bin [--epochs 40]
import sys, struct
import numpy as np, torch, torch.nn as nn

p = sys.argv[1]
epochs = int(sys.argv[sys.argv.index('--epochs')+1]) if '--epochs' in sys.argv else 40
raw = open(p, 'rb').read()
magic, L, ntr, nte, DM, TOPK = struct.unpack_from('<6i', raw, 0)
assert magic == 0x4C455352, 'bad RSEL'
n = ntr + nte
off = 24
def take(cnt):
    global off
    a = np.frombuffer(raw, dtype=np.float32, count=cnt, offset=off).copy()
    off += cnt * 4
    return a
X    = take(n * DM).reshape(n, DM)
Yhat = take(n * DM).reshape(n, DM)
Yref = take(n * DM).reshape(n, DM)
T = Yref - Yhat
Xtr, Xte = X[:ntr], X[ntr:]
Ttr, Tte = T[:ntr], T[ntr:]
Yh_te, Yr_te = Yhat[ntr:], Yref[ntr:]

def te_cos(F_te):
    a = Yh_te + F_te; b = Yr_te
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return float((num / den).mean())

print(f'L{L} ntr={ntr} nte={nte}  base TE cos={te_cos(np.zeros_like(Tte)):.4f}', flush=True)

dev = 'cpu'
xt = torch.from_numpy(Xtr); tt = torch.from_numpy(Ttr)
xe = torch.from_numpy(Xte)

def train(model, tag):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    best, best_state = -1.0, None
    nb = max(1, ntr // 512)
    for ep in range(epochs):
        perm = torch.randperm(ntr)
        model.train()
        for i in range(nb):
            idx = perm[i*512:(i+1)*512]
            opt.zero_grad()
            loss = ((model(xt[idx]) - tt[idx])**2).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            c = te_cos(model(xe).numpy())
        if c > best: best, best_state = c, {k: v.clone() for k, v in model.state_dict().items()}
        print(f'  {tag} ep{ep+1} TE cos={c:.4f} (best {best:.4f})', flush=True)
        if ep > 8 and c < best - 0.01: break   # simple early stop
    return best

lin = nn.Sequential(nn.Linear(DM, 64, bias=False), nn.Linear(64, DM, bias=True))
b1 = train(lin, 'linear64-SGD')
mlp = nn.Sequential(nn.Linear(DM, 256), nn.GELU(), nn.Linear(256, DM))
b2 = train(mlp, 'MLP256')
print(f'SENTINEL-RESULT L{L} base={te_cos(np.zeros_like(Tte)):.4f} linSGD={b1:.4f} MLP={b2:.4f}', flush=True)
