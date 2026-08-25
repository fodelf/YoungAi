#!/usr/bin/env python3
# amp_probe.py — v6.1 放大器判决探针(2026-08-19 用户令"放大器而不是补差")。
# 命题: 行为空间目标(对齐 FP 输出)+乘性调制场, 在深层(补差恒零处)能否拿到 held 正挽回。
# 数据=zlayer zcache(dH/pYQ/pw/prow)+锚 X: y_q=Σw·Yq, y_fp=y_q+dH(行为目标, 非残差)。
# 模型: z=tanh(V·x) (k维隐变量), ŷ=y_q⊙(1+U_g·z)+U_a·z (乘性主体+加性辅助)。
# 四损失: L1 行为fit(colw感知加权) + L2 dither稳定; 判据=held 1−‖y_fp−ŷ‖²/‖y_fp−y_q‖²。
# 用法: amp_probe.py <anchor> <zcache_LXX.npz> [K=64] [NFIT=1638] [steps=3000]
import sys, os, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from probe_layer_behavior import anchor_layer

ap, zcp = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 64
NFIT = int(sys.argv[4]) if len(sys.argv) > 4 else 1638
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
L = int(os.path.basename(zcp).split("_L")[1][:2])
zc = np.load(zcp)
dH, prow, pw, pYQ = zc["dH"], zc["prow"], zc["pw"], zc["pYQ"]
S, D = dH.shape
X0, _, _, _ = anchor_layer(ap, L, S)

yq = np.zeros((S, D), dtype=np.float32)
np.add.at(yq, prow, pw[:, None] * pYQ)          # y_q = Σ 命中对的加权量化输出
yfp = yq + dH                                    # 行为目标 = FP 的 MoE 块输出
tr, ev = np.arange(0, NFIT), np.arange(NFIT, S)

dev = "cpu"
X = torch.tensor(X0, dtype=torch.float32, device=dev)
Yq = torch.tensor(yq, device=dev); Yfp = torch.tensor(yfp, device=dev)
colw = torch.tensor(np.sqrt(dH[tr].var(0) + 1e-12), device=dev)   # 感知加权(与 zlayer 同式)
e0 = ((Yfp[ev] - Yq[ev]) ** 2).sum().item()      # 基线误差(=纯量化行为差)

g = torch.Generator().manual_seed(7)
V  = (torch.randn(D, K, generator=g) * 0.02).requires_grad_()
Ug = torch.zeros(K, D, requires_grad=True)       # 乘性头: 0 初始=恒等出发(安全)
Ua = torch.zeros(K, D, requires_grad=True)       # 加性辅助头
_MUL=int(os.getenv("AMP_MUL_ONLY","0"))   # 1=纯乘性(加性头冻结; L35首针: 加性=毒 乘性=+13.2%)
opt = torch.optim.Adam([V, Ug] if _MUL else [V, Ug, Ua], lr=float(os.getenv("AMP_LR","2e-3")))
rms = X[tr].pow(2).mean(1, keepdim=True).sqrt()
t0 = time.time()
for it in range(STEPS):
    idx = torch.randint(0, len(tr), (256,), generator=g)
    xb = X[tr][idx]
    if it % 2 == 1:                              # L2 dither 稳定(zlayer 同幅 0.04·rms)
        xb = xb + torch.randn(xb.shape, generator=g) * 0.04 * rms[idx]
    z = torch.tanh(xb @ V)
    yhat = Yq[tr][idx] * (1 + z @ Ug) + z @ Ua
    loss = (((yhat - Yfp[tr][idx]) * colw) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if (it + 1) % 1000 == 0:
        with torch.no_grad():
            ze = torch.tanh(X[ev] @ V)
            yh = Yq[ev] * (1 + ze @ Ug) + ze @ Ua
            rec = 1 - ((yh - Yfp[ev]) ** 2).sum().item() / e0
        print(f"  L{L} step {it+1} held行为挽回={rec*100:.2f}%", flush=True)
with torch.no_grad():
    z = torch.tanh(X[ev] @ V)
    yh = Yq[ev] * (1 + z @ Ug) + z @ Ua
    rec = 1 - ((yh - Yfp[ev]) ** 2).sum().item() / e0
    zt = torch.tanh(X[tr] @ V)
    yt = Yq[tr] * (1 + zt @ Ug) + zt @ Ua
    rft = 1 - ((yt - Yfp[tr]) ** 2).sum().item() / ((Yfp[tr] - Yq[tr]) ** 2).sum().item()
    # 消融: 纯乘性(关加性头)
    yh2 = Yq[ev] * (1 + z @ Ug)
    rec_mul = 1 - ((yh2 - Yfp[ev]) ** 2).sum().item() / e0
vol = (D * K + 2 * K * D) * 2 / 2 ** 20
print(f"★L{L} 放大器探针: held行为挽回 {rec*100:.2f}% (纯乘性 {rec_mul*100:.2f}%, fit {rft*100:.2f}%) "
      f"k={K} 体积{vol:.1f}MB fp16 | 补差基线=0.0%(判决在案) | {time.time()-t0:.0f}s", flush=True)
