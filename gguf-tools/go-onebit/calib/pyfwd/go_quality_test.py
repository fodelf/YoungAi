#!/usr/bin/env python3
# Minimal Go output-quality test: next-token top1 accuracy on a Go snippet,
# full-precision experts vs strict-1-bit (go1b) routed experts.
# Backbone + attention + shared expert stay full precision (only ROUTED experts -> 1-bit),
# matching the design. Reuses the validated numpy forward (dsv4_fwd.py).
import os, sys
HFALL = "/private/tmp/claude-501/-Users-fodelf-git-ds4-main/24503593-c406-4203-a34b-b2d8ea433b47/scratchpad/hf_all"
os.environ.setdefault("DS4_HF", HFALL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ds4reader as R

ONEBIT  = os.environ.get("ONEBIT", "0") == "1"
GOAWARE = os.environ.get("GOAWARE", "0") == "1"   # (placeholder; generic go1b for now)

CAP = os.environ.get("CAP", "/private/tmp/m1_ds4/cap_m1")
_V = {}                         # layer -> per-dim Go input variance (gate/up Go-aware scale)
def vlayer(L):
    if L not in _V:
        x = np.load(f"{CAP}/ffn_in_L{L}.npy").astype(np.float32)
        _V[L] = x.var(0) + 1e-8
    return _V[L]

def go1b(w, v=None):
    if v is not None:           # Go-aware diagonal output-optimal scale: |w| weighted by Go input var
        s = (np.abs(w) @ v)[:, None] / v.sum()
    else:                       # generic weight-L2 scale = mean(|w|)
        s = np.abs(w).mean(1, keepdims=True)
    return (s * np.sign(w)).astype(np.float32)

_orig = R.read_weight
def patched(name, *a, **k):
    w = _orig(name, *a, **k)
    # only routed experts (layers.L.ffn.experts.E.wX.weight); NOT shared_experts, NOT backbone
    if ONEBIT and ".ffn.experts." in name:
        if GOAWARE:
            parts = name.split('.'); L = int(parts[1]); mat = parts[-2]
            if mat in ("w1", "w3"):           # gate/up take x=ffn_in -> Go-aware scale
                return go1b(w, vlayer(L))
            return go1b(w)                     # w2/down takes h (not captured) -> generic for now
        return go1b(w)
    return w
R.read_weight = patched

import dsv4_fwd as F
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.path.join(R.HF, "tokenizer.json"))

code = ('package main\n\nimport "fmt"\n\n'
        'func Add(a, b int) int {\n\treturn a + b\n}\n\n'
        'func main() {\n\tfmt.Println(Add(3, 4))\n}\n')
ids = np.array(tk.encode(code).ids, dtype=np.int64)
S = int(os.environ.get("S", "72")); ids = ids[:S]
print(f"=== GO QUALITY  ONEBIT={ONEBIT}  S={len(ids)}  (routed experts {'1-bit' if ONEBIT else 'full'}; shared+backbone full) ===", flush=True)
print("prompt:", repr(tk.decode([int(i) for i in ids])[:160]), flush=True)
F.run([ids], [ids], "/tmp/go_qual_out", capture=False)
