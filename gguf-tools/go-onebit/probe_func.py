#!/usr/bin/env python3
# Functional-compressibility probe (discovery, Python OK per user).
# THE decisive test of "Go computation is structured => compressible":
# can a SMALL net learn a layer's Go MoE function  ffn_in -> ffn_out  on held-out tokens?
# If held-out cosine is high, the 256-expert layer reduces to a tiny Go-specific function.
# Pure cap_m1 (no model run). Run in M1 cap_work venv (needs scikit-learn).
import os, numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CAP  = os.environ.get("CAP", "/Users/fodelf/ds4-main/cap_m1")
NTR  = int(os.environ.get("NTR", "9000"))            # train tokens; rest held-out
HID  = int(os.environ.get("HID", "512"))
LAYERS = [int(x) for x in os.environ.get("LAYERS", "0,12,21,42").split(",")]

def cos_rows(A, B):
    num = (A*B).sum(1)
    da = np.linalg.norm(A, axis=1); db = np.linalg.norm(B, axis=1)
    m = (da > 0) & (db > 0)
    return float((num[m]/(da[m]*db[m])).mean())

print(f"functional probe  ffn_in -> ffn_out  | NTR={NTR} HID={HID}")
print(f"{'L':>3} {'ridge_cos':>9} {'MLP_cos':>8} {'MLP_R2':>7}   (held-out; baseline=identity ffn_in->ffn_out cos)")
for L in LAYERS:
    X = np.load(f"{CAP}/ffn_in_L{L}.npy").astype(np.float32)
    Y = np.load(f"{CAP}/ffn_out_L{L}.npy").astype(np.float32)
    Xtr, Xte = X[:NTR], X[NTR:]
    Ytr, Yte = Y[:NTR], Y[NTR:]
    # identity baseline: how aligned is the raw input to the output (sanity floor)
    base = cos_rows(Xte, Yte)
    sx = StandardScaler().fit(Xtr)
    Xtr2, Xte2 = sx.transform(Xtr), sx.transform(Xte)
    rg = Ridge(alpha=10.0).fit(Xtr2, Ytr)
    cr = cos_rows(rg.predict(Xte2), Yte)
    hidden = tuple(int(h) for h in str(os.environ.get("HIDDEN", str(HID))).split(","))
    mlp = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                       max_iter=int(os.environ.get("MAXITER","60")), early_stopping=True,
                       n_iter_no_change=8, alpha=1e-4, random_state=0).fit(Xtr2, Ytr)
    Pm = mlp.predict(Xte2)
    cm = cos_rows(Pm, Yte)
    # R2 (variance explained, held-out)
    ss_res = ((Pm-Yte)**2).sum(); ss_tot = ((Yte-Ytr.mean(0))**2).sum()
    r2 = 1.0 - ss_res/ss_tot
    print(f"{L:>3} {cr:>9.3f} {cm:>8.3f} {r2:>7.3f}   (id-base cos={base:.3f})")
