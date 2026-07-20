#!/usr/bin/env python3
# Decisive info test: are the magnitudes |W| that 1-bit discards LOW-DIM predictable (cheap z
# recovers -> user right) or HIGH-DIM independent (must be stored -> physical wall)?
# 1-bit keeps sign(W) + one row scale (= |W| approximated by its row mean = rank-1, col-constant).
# Replace that with a rank-k approx of the FULL |W| matrix (cheap structured magnitude), keep exact
# signs, measure per-expert SwiGLU output cosine on Go activations vs full precision.
#   rank-k small & cos -> ~1  => magnitudes low-dim predictable (small z works).
#   need large k        => magnitudes high-dim, must store the residual (info wall).
import os, sys
os.environ.setdefault("DS4_HF", "/private/tmp/claude-501/-Users-fodelf-git-ds4-main/24503593-c406-4203-a34b-b2d8ea433b47/scratchpad/hf_all")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, ds4reader as R
CAP = os.environ.get("CAP", "/private/tmp/m1_ds4/cap_m1")

def silu(z): return z / (1.0 + np.exp(-z))
def expert_out(g, u, d, x):                 # x[N,4096] g,u[2048,4096] d[4096,2048]
    return (silu(x @ g.T) * (x @ u.T)) @ d.T
def cosrows(A, B):
    n = (A*B).sum(1); da = np.linalg.norm(A, axis=1); db = np.linalg.norm(B, axis=1)
    m = (da > 0) & (db > 0); return float((n[m] / (da[m]*db[m])).mean())
def onebit(W):                              # uniform row scale (what 1-bit stores today)
    return (np.abs(W).mean(1, keepdims=True) * np.sign(W)).astype(np.float32)
def mag_rankk(W, k):                        # |W| ~= rank-k structured magnitude, x exact signs
    A = np.abs(W).astype(np.float32)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    Ak = (U[:, :k] * S[:k]) @ Vt[:k]
    return (Ak * np.sign(W)).astype(np.float32)

RANKS = [1, 8, 64, 256]
EXPERTS = [0, 50, 128, 200]
for L in [0, 21, 42]:
    x = np.load(f"{CAP}/ffn_in_L{L}.npy").astype(np.float32)[:512]
    acc = {f"1bit": [], **{f"magR{k}": [] for k in RANKS}}
    for e in EXPERTS:
        g = R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight")
        u = R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight")
        d = R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight")
        of = expert_out(g, u, d, x)
        acc["1bit"].append(cosrows(of, expert_out(onebit(g), onebit(u), onebit(d), x)))
        for k in RANKS:
            oa = expert_out(mag_rankk(g, k), mag_rankk(u, k), mag_rankk(d, k), x)
            acc[f"magR{k}"].append(cosrows(of, oa))
    msg = " | ".join(f"{tag}={np.mean(v):.3f}" for tag, v in acc.items())
    print(f"L{L:2d} (avg {len(EXPERTS)} experts, Go out cos): {msg}", flush=True)
print("DONE", flush=True)
