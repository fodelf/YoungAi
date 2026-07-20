#!/usr/bin/env python3
# FULL-STACK end-to-end: Go-aware 1-bit base + per-layer HIDDEN VARIABLE (four-loss
# closed-form correction) applied in the forward. Answers: does adding the hidden
# variable + losses to the end-to-end actually recover Go quality over the bare base?
# Per-layer correction z^L: regularized low-rank ridge  delta = (y_full - y_1bit) ~ M_L x,
#  align(cosine via least-sq) + fixed(ridge LAM + mean-pin) + smooth(rank truncation) + classify(value MSE).
# Pure numpy, reuses dsv4_fwd (load_layer/moe_all/expert_fp). Run in M4 /tmp/go_venv.
import os, sys
os.environ.setdefault("DS4_HF", "/private/tmp/claude-501/-Users-fodelf-git-ds4-main/24503593-c406-4203-a34b-b2d8ea433b47/scratchpad/hf_all")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, ds4reader as R
CAP  = os.environ.get("CAP", "/private/tmp/m1_ds4/cap_m1")
NSUB = int(os.environ.get("NSUB", "800"))     # cap_m1 tokens to fit each layer's hidden var
RANK = int(os.environ.get("RANK", "48"))
LAM  = float(os.environ.get("LAM", "20.0"))
USE_HV = os.environ.get("USE_HV", "1") == "1"

_V = {}
def vlayer(L):
    if L not in _V:
        x = np.load(f"{CAP}/ffn_in_L{L}.npy").astype(np.float32); _V[L] = x.var(0) + 1e-8
    return _V[L]
def go1b(w, v):                                # Go-aware diagonal scale (gate/up)
    return ((np.abs(w) @ v)[:, None] / v.sum() * np.sign(w)).astype(np.float32)
def gen1b(w):
    return (np.abs(w).mean(1, keepdims=True) * np.sign(w)).astype(np.float32)

QON = [False]                                  # routed-expert 1-bit toggle
_orig = R.read_weight
def maybe(name, *a, **k):
    w = _orig(name, *a, **k)
    if QON[0] and ".ffn.experts." in name:
        p = name.split('.'); L = int(p[1]); mat = p[-2]
        return go1b(w, vlayer(L)) if mat in ("w1", "w3") else gen1b(w)
    return w
R.read_weight = maybe
import dsv4_fwd as F
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.path.join(R.HF, "tokenizer.json"))
corpus_ids = np.array(tk.encode(open("/private/tmp/m1_ds4/cap_work/gocorpus_big.txt").read()).ids, dtype=np.int64)

CORR = {}
def fit_layer(L):
    x = np.load(f"{CAP}/ffn_in_L{L}.npy").astype(np.float32)[:NSUB]
    yf = np.load(f"{CAP}/ffn_out_L{L}.npy").astype(np.float32)[:NSUB]
    ids = corpus_ids[:NSUB]
    W = F.load_layer(L)
    QON[0] = True
    y1, _ = F.moe_all(x, ids, W, L)            # Go-aware 1-bit MoE output on cap_m1
    QON[0] = False
    del W
    delta = (yf - y1).astype(np.float32)
    xm, dm = x.mean(0), delta.mean(0)
    xc, dc = x - xm, delta - dm
    d = x.shape[1]
    G = xc.T @ xc + LAM * np.eye(d, dtype=np.float32)     # fixed: ridge LAM
    M = np.linalg.solve(G, xc.T @ dc).astype(np.float32)  # [d,d]
    U, S, Vt = np.linalg.svd(M, full_matrices=False)      # smooth: rank-RANK truncation
    A = (U[:, :RANK] * S[:RANK]).astype(np.float32)       # [d,RANK] low-rank factors (small)
    B = Vt[:RANK].astype(np.float32)                      # [RANK,d]
    b = (dm - (xm @ A) @ B).astype(np.float32)
    CORR[L] = (A, B, b)
    # in-sample sanity
    pred = (xc @ A) @ B + dm
    cs = float((np.sum((y1+pred)*yf,1) / (np.linalg.norm(y1+pred,axis=1)*np.linalg.norm(yf,axis=1)+1e-8)).mean())
    cb = float((np.sum(y1*yf,1) / (np.linalg.norm(y1,axis=1)*np.linalg.norm(yf,axis=1)+1e-8)).mean())
    print(f"  fit L{L}: base_cos={cb:.3f} +hv_cos={cs:.3f} (in-sample)", flush=True)

# wrap moe_all to apply the hidden-var correction
_orig_moe = F.moe_all
def moe_hv(Fin, Ids, W, L):
    Fout, Idx = _orig_moe(Fin, Ids, W, L)
    if USE_HV and L in CORR:
        A, B, b = CORR[L]
        Fout = (Fout + (Fin @ A) @ B + b).astype(np.float32)
    return Fout, Idx
F.moe_all = moe_hv

if __name__ == "__main__":
    print(f"=== precompute hidden var (NSUB={NSUB} RANK={RANK} LAM={LAM}) over 43 layers ===", flush=True)
    import time; t0 = time.time()
    for L in range(43):
        fit_layer(L)
    print(f"precompute done {time.time()-t0:.0f}s", flush=True)
    code = ('package main\n\nimport "fmt"\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n\n'
            'func main() {\n\tfmt.Println(Add(3, 4))\n}\n')
    ids = np.array(tk.encode(code).ids, dtype=np.int64)[:int(os.environ.get("S","72"))]
    QON[0] = True                              # 1-bit routed experts in the eval forward
    print(f"=== FULL STACK eval: Go-aware 1-bit + hidden-var (USE_HV={USE_HV}) S={len(ids)} ===", flush=True)
    F.run([ids], [ids], "/tmp/fs_out", capture=False)
