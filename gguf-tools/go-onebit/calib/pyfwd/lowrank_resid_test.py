#!/usr/bin/env python3
# Recovery-vs-SIZE curve for a SMALL derived hidden variable (iron law: recover original, no training).
# Hidden var = per-expert LOW-RANK residual  R = W - Q1(W) ~= U_d V_d  (rank-d, derived via randomized SVD).
# Stored size = 0.406*d GB on top of the 33G 1-bit base. Sweep RANK to map small-hidden-var recovery.
#   RANK=0 -> bare 1-bit (0.222) ; full 1-bit residual plane -> 0.778 ; tiny shared rank-48 -> 0.028 (full_stack).
import os, sys
os.environ.setdefault("DS4_HF", "/private/tmp/claude-501/-Users-fodelf-git-ds4-main/24503593-c406-4203-a34b-b2d8ea433b47/scratchpad/hf_all")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, ds4reader as R
np.random.seed(0)

RANK = int(os.environ.get("RANK", "16"))
def lowrank(Rm, d):                 # randomized rank-d approx of residual matrix Rm
    if d <= 0: return np.zeros_like(Rm)
    n = Rm.shape[1]; G = np.random.randn(n, d + 8).astype(np.float32)
    Y = Rm @ G; Q, _ = np.linalg.qr(Y); B = Q.T @ Rm
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    return ((Q @ Ub[:, :d]) * S[:d]) @ Vt[:d]
def qhv(w, d):                      # 1-bit base + low-rank residual hidden var
    s = np.abs(w).mean(1, keepdims=True); q1 = s * np.sign(w)
    return (q1 + lowrank((w - q1).astype(np.float32), d)).astype(np.float32)
_orig = R.read_weight
def patched(name, *a, **k):
    w = _orig(name, *a, **k)
    if ".ffn.experts." in name:
        return qhv(w, RANK)
    return w
R.read_weight = patched

import dsv4_fwd as F
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.path.join(R.HF, "tokenizer.json"))
code = ('package main\n\nimport "fmt"\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n\n'
        'func main() {\n\tfmt.Println(Add(3, 4))\n}\n')
ids = np.array(tk.encode(code).ids, dtype=np.int64)[:int(os.environ.get("S", "72"))]
gb = 0.406 * RANK
print(f"=== LOW-RANK HIDDEN-VAR RANK={RANK} (residual ~{gb:.1f}G on top of 33G base = {33+gb:.0f}G) S={len(ids)} | 1bit=0.222 fullplane=0.778 ===", flush=True)
F.run([ids], [ids], "/tmp/lr_out", capture=False)
