#!/usr/bin/env python3
# go_input_rank.py — measure the effective rank of the Go FFN-input subspace
# (cap ffn_in_L{L}.npy) to size the low-rank Go residual. If the Go inputs live in
# a low-dim subspace (d_go small), the residual R·x = (R·V_go)·(V_go^T x) is low-rank
# → a genuinely small Go-specific residual. Usage: go_input_rank.py cap_dir L [L ...]
import numpy as np, sys, os

cap = sys.argv[1] if len(sys.argv) > 1 else "cap_m4"
layers = [int(a) for a in sys.argv[2:]] or [8]
print(f"cap={cap}  layers={layers}")
print(f"{'L':>3} | {'d90':>5} {'d95':>5} {'d99':>5} | {'top8':>6} {'top32':>6} {'top128':>7} | {'nsamp':>6}")
for L in layers:
    f = os.path.join(cap, f"ffn_in_L{L}.npy")
    if not os.path.exists(f):
        print(f"{L:>3} | missing {f}"); continue
    X = np.load(f).astype(np.float64)          # [n_samples, d_model]
    X = X - X.mean(0)
    n = X.shape[0]
    C = (X.T @ X) / n                          # [d_model, d_model] covariance
    ev = np.linalg.eigvalsh(C)[::-1]           # descending eigenvalues
    ev = ev[ev > 0]
    cum = np.cumsum(ev) / ev.sum()
    d90 = int(np.searchsorted(cum, 0.90)) + 1
    d95 = int(np.searchsorted(cum, 0.95)) + 1
    d99 = int(np.searchsorted(cum, 0.99)) + 1
    t8   = cum[7]   if len(cum) > 7   else 1.0
    t32  = cum[31]  if len(cum) > 31  else 1.0
    t128 = cum[127] if len(cum) > 127 else 1.0
    print(f"{L:>3} | {d90:>5} {d95:>5} {d99:>5} | {t8:>6.3f} {t32:>6.3f} {t128:>7.3f} | {n:>6}")
print("\n读: d95 小(~几十-一两百) => Go 输入低维 => 低秩 Go 残差可行且极小; top8/32 高 => 更省")
