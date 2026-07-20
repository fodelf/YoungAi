#!/usr/bin/env python3
# teacher_nll.py — the ORIGINAL model's teacher-forced avg_nll on a raw text
# slice, tokenized exactly like ds4 --perplexity-file's stream (BOS-less raw
# continuation over one contiguous chunk). Anchors the restoration ratio:
#   restore% = (nll_bare − nll_corr) / (nll_bare − nll_teacher)
# Usage: DS4_HF=... venv/bin/python teacher_nll.py <text file> [S=699]
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds4reader as R
import dsv4_fwd as F
from tokenizers import Tokenizer

path = sys.argv[1]
S = int(sys.argv[2]) if len(sys.argv) > 2 else 727
tk = Tokenizer.from_file(os.path.join(R.HF, "tokenizer.json"))
ids = np.array(tk.encode(open(path).read()).ids[:S], dtype=np.int64)
print(f"teacher_nll: {len(ids)} tokens from {path}", flush=True)

# forward once; recompute head logits (mirror of run()'s head block)
emb = R.get("embed.weight")
H = [np.repeat(emb[ids][:, None, :], F.HCM, 1).astype(np.float32)]
h = H[0]
for L in range(F.NL):
    W = F.load_layer(L)
    resid = h
    y, post, comb = F.hc_pre(h, W['hc_attn_fn'], W['hc_attn_scale'], W['hc_attn_base'])
    a = F.attention(F.rms(y, W['an']), W, L)
    h = F.hc_post(a, resid, post, comb)
    resid2 = h
    y2, post2, comb2 = F.hc_pre(h, W['hc_ffn_fn'], W['hc_ffn_scale'], W['hc_ffn_base'])
    fin = F.rms(y2, W['fn'])
    Fout, _ = F.moe_all(fin, ids, W, L)
    h = F.hc_post(Fout, resid2, post2, comb2)
    print(f"L{L} done", flush=True)
    del W

hcfn = R.get("hc_head_fn"); hcb = R.get("hc_head_base"); hcs = R.get("hc_head_scale")
fnorm = R.get("norm.weight"); hw = R.get("head.weight")
Sn = h.shape[0]; x = h.reshape(Sn, -1)
rsq = np.reciprocal(np.sqrt(np.mean(x * x, -1, keepdims=True) + F.EPS))
mixes = (x @ hcfn.T) * rsq
pre = F.sigmoid(mixes * hcs + hcb) + F.HCEPS
y = (pre[:, :, None] * h).sum(1)
y = F.rms(y, fnorm)
logits = (y @ hw.T).astype(np.float64)

lp = logits[:-1] - logits[:-1].max(1, keepdims=True)
lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
tgt = ids[1:]
nll = -lp[np.arange(len(tgt)), tgt]
np.save("/tmp/teacher_nll_pertok.npy", nll.astype(np.float32))
np.save("/tmp/teacher_nll_ids.npy", ids)
print(f"tokens={len(ids)} scored={len(tgt)} nll={nll.sum():.6f} avg_nll={nll.mean():.6f} ppl={np.exp(nll.mean()):.6f}", flush=True)
w = nll[31:]  # ds4 --perplexity-file 从第32个token起分, 对齐窗口
print(f"ds4-window scored={len(w)} avg_nll={w.mean():.6f}", flush=True)
