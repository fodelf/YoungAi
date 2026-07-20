#!/usr/bin/env python3
# Apply the trained per-layer Go generators in the numpy forward (each replaces its MoE),
# measure end-to-end Go next-token accuracy. Validates option-2 BEFORE the ds4 GGUF/runtime build.
import os, sys
os.environ.setdefault("DS4_HF", "/private/tmp/claude-501/-Users-fodelf-git-ds4-main/24503593-c406-4203-a34b-b2d8ea433b47/scratchpad/hf_all")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, ds4reader as R
import dsv4_fwd as F
from tokenizers import Tokenizer

GENDIR = os.environ.get("GENDIR", "/private/tmp/m1_ds4/gen")
NB = int(os.environ.get("NB", "3"))
GEN = {}
for L in range(43):
    p = f"{GENDIR}/gen_L{L}.npz"
    if os.path.exists(p):
        z = np.load(p); GEN[L] = {k: z[k] for k in z.files}
print(f"loaded {len(GEN)}/43 generators from {GENDIR}", flush=True)

def gelu(x): return 0.5*x*(1.0+np.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
def ln(x, w, b): m=x.mean(-1,keepdims=True); v=x.var(-1,keepdims=True); return (x-m)/np.sqrt(v+1e-5)*w+b
def gen_fwd(x, P):
    x = (x - P['xm'])/P['xs']
    x = x @ P['inp.weight'].T + P['inp.bias']
    for i in range(NB):
        h = ln(x, P[f'blocks.{i}.0.weight'], P[f'blocks.{i}.0.bias'])
        h = gelu(h @ P[f'blocks.{i}.1.weight'].T + P[f'blocks.{i}.1.bias'])
        h = h @ P[f'blocks.{i}.4.weight'].T + P[f'blocks.{i}.4.bias']
        x = x + h
    return ((x @ P['out.weight'].T + P['out.bias']) * P['ys'] + P['ym']).astype(np.float32)

_om = F.moe_all
def moe_gen(Fin, Ids, W, L):
    if L in GEN:                       # generator replaces the whole MoE (routed+shared) for this layer
        return gen_fwd(Fin.astype(np.float32), GEN[L]), np.zeros((Fin.shape[0], 6), np.int64)
    return _om(Fin, Ids, W, L)
F.moe_all = moe_gen

tk = Tokenizer.from_file(os.path.join(R.HF, "tokenizer.json"))
code = ('package main\n\nimport "fmt"\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n\n'
        'func main() {\n\tfmt.Println(Add(3, 4))\n}\n')
ids = np.array(tk.encode(code).ids, dtype=np.int64)[:int(os.environ.get("S", "72"))]
print(f"=== GENERATOR FORWARD (MoE -> trained gen) S={len(ids)} | vs full=0.833 base1bit=0.222 ===", flush=True)
F.run([ids], [ids], "/tmp/gen_eval_out", capture=False)
