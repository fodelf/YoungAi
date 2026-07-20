#!/usr/bin/env python3
# WEIGHT-SPACE recovery (iron law: recover the ORIGINAL model's capability, NO self-trained
# replacement). Hidden variable = DERIVED residual quantization Q2(W - Q1(W)): 1-bit base +
# residual planes reconstruct the original expert weights -> original capability. No training.
# Sweep QBITS (planes): 1 = bare 1-bit base (0.222), 2 = 1-bit + 1-bit residual, etc.
import os, sys
os.environ.setdefault("DS4_HF", "/private/tmp/claude-501/-Users-fodelf-git-ds4-main/24503593-c406-4203-a34b-b2d8ea433b47/scratchpad/hf_all")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, ds4reader as R

QBITS = int(os.environ.get("QBITS", "2"))
def qresid(w, nbits):           # K-plane residual quant: W ~= sum_k s_k*sign(r_k); derived from W, no training
    out = np.zeros_like(w, dtype=np.float32); r = w.astype(np.float32).copy()
    for _ in range(nbits):
        s = np.abs(r).mean(1, keepdims=True); b = np.sign(r); out = out + s*b; r = r - s*b
    return out
_orig = R.read_weight
def patched(name, *a, **k):
    w = _orig(name, *a, **k)
    if ".ffn.experts." in name:          # routed experts only; shared+backbone full
        return qresid(w, QBITS)
    return w
R.read_weight = patched

import dsv4_fwd as F
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.path.join(R.HF, "tokenizer.json"))
F.TK = tk
code = ('package main\n\nimport "fmt"\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n\n'
        'func main() {\n\tfmt.Println(Add(3, 4))\n}\n')
ids = np.array(tk.encode(code).ids, dtype=np.int64)[:int(os.environ.get("S", "72"))]
print(f"=== RESIDUAL RECOVERY QBITS={QBITS} (1-bit base + {QBITS-1}-bit residual hidden-var) S={len(ids)} | full=0.833 1bit=0.222 ===", flush=True)
F.run([ids], [ids], "/tmp/resid_out", capture=False)
