#!/usr/bin/env python3
# Step 1 for the deep restoration codec: compute the 1-bit BASE output o_base (ô) on Go data,
# per layer, on M4 (all shards reachable). Saved to the M1-NFS path so M1 can train the codec.
# Anchor for restoration: o_restored = ô + D(ô, x, z^L). Generic 1-bit experts = the fixed Θ.
import os, sys, numpy as np
os.environ.setdefault("DS4_HF", "/private/tmp/claude-501/-Users-fodelf-git-ds4-main/24503593-c406-4203-a34b-b2d8ea433b47/scratchpad/hf_all")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds4reader as R
CAP  = os.environ.get("CAP", "/private/tmp/m1_ds4/cap_m1")
OUT  = os.environ.get("OUT", "/private/tmp/m1_ds4/obase")   # M4 writes via NFS -> M1 reads local
NSUB = int(os.environ.get("NSUB", "5000"))
os.makedirs(OUT, exist_ok=True)

def go1b(w): return (np.abs(w).mean(1, keepdims=True) * np.sign(w)).astype(np.float32)
QON = [False]; _orig = R.read_weight
def patched(name, *a, **k):
    w = _orig(name, *a, **k)
    if QON[0] and ".ffn.experts." in name: return go1b(w)
    return w
R.read_weight = patched
import dsv4_fwd as F
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.path.join(R.HF, "tokenizer.json"))
corpus_ids = np.array(tk.encode(open("/private/tmp/m1_ds4/cap_work/gocorpus_big.txt").read()).ids, dtype=np.int64)

print(f"precompute o_base NSUB={NSUB} -> {OUT}", flush=True)
for L in range(43):
    x = np.load(f"{CAP}/ffn_in_L{L}.npy").astype(np.float32)[:NSUB]
    ids = corpus_ids[:NSUB]
    W = F.load_layer(L)
    QON[0] = True
    o1, _ = F.moe_all(x, ids, W, L)            # 1-bit base MoE output on Go tokens
    QON[0] = False
    np.save(f"{OUT}/obase_L{L}.npy", o1.astype(np.float32))
    del W
    print(f"L{L:2d} obase saved {o1.shape}", flush=True)
print("ALL OBASE SAVED", flush=True)
