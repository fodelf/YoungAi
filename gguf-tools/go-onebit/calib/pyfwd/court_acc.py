#!/usr/bin/env python3
# court_acc.py — recover next-token top1 acc from a capture dir's final_topk_idx
# (row t's top1 vs the true token at t+1). Usage: court_acc.py CAP_DIR CORPUS S
import sys, numpy as np, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds4reader as R
from tokenizers import Tokenizer
cap, corpus, S = sys.argv[1], sys.argv[2], int(sys.argv[3])
tk = Tokenizer.from_file(os.path.join(R.HF, "tokenizer.json"))
ids = np.array(tk.encode(open(corpus).read()).ids[:S], dtype=np.int64)
top = np.load(f"{cap}/final_topk_idx.npy")[:len(ids)]
pred = top[:-1, 0]; tgt = ids[1:]
print(f"{cap}: top1 acc = {float((pred == tgt).mean()):.4f}  (n={len(tgt)})")
