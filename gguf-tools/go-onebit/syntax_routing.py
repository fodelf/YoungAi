#!/usr/bin/env python3
# Syntax-conditioned routing analysis (discovery, Python OK per user).
# Tests: does the model's expert routing organize by Go SYNTAX (the "语法糖")?
# Pure cap_m1 + re-tokenize gocorpus (no model run). Run in M1 cap_work venv.
import os, collections, math
import numpy as np
from tokenizers import Tokenizer

HF     = os.environ.get("HF",  "/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base")
CORPUS = os.environ.get("CORPUS", "/Users/fodelf/ds4-main/cap_work/gocorpus_big.txt")
CAP    = os.environ.get("CAP", "/Users/fodelf/ds4-main/cap_m1")
NTOK   = 12288

tk = Tokenizer.from_file(os.path.join(HF, "tokenizer.json"))
ids_all = tk.encode(open(CORPUS).read()).ids
assert len(ids_all) >= NTOK, f"corpus only {len(ids_all)} tokens < {NTOK}"
ids = ids_all[:NTOK]                                  # first 24 chunks of 512 = cap_m1 row order

# byte-level BPE -> readable text
def clean(t):
    if t is None: return ""
    return (t.replace('Ġ',' ').replace('Ċ','\n').replace('ĉ','\t')
             .replace('č','\r'))
texts = [clean(tk.id_to_token(i)) for i in ids]

# --- alignment / decode sanity ---
print("=== DECODE SANITY (first 24 tokens; corpus starts '// Copyright 2009 The Go Authors') ===")
print(repr("".join(texts[:24])))
print()

GO_KW = set("break default func interface select case defer go map struct chan else goto "
            "package switch const fallthrough if range type continue for import return var "
            "bool string int int8 int16 int32 int64 uint byte rune float64 error nil true false".split())
def classify(s):
    st = s.strip(" \t\r\n")
    if st == "":            return "ws/nl"
    if st in GO_KW:         return "keyword"
    if st in ("{","}"):     return "brace"
    if st in ("(",")"):     return "paren"
    if st in ("[","]"):     return "bracket"
    if all((not c.isalnum()) and c != "_" for c in st): return "operator"
    if st[0].isdigit():     return "number"
    if st[0].isalpha() or st[0] == "_": return "ident"
    return "other"
cats = [classify(t) for t in texts]
print("=== token category counts ===")
for c,n in collections.Counter(cats).most_common(): print(f"  {c:10} {n}")
print()

def entropy(counter):
    tot = sum(counter.values())
    return -sum((c/tot)*math.log2(c/tot) for c in counter.values()) if tot else 0.0

CATS = ["keyword","brace","paren","bracket","operator","ident","number","ws/nl"]
LAYERS = [int(x) for x in os.environ.get("LAYERS","0,6,12,21,30,42").split(",")]

for L in LAYERS:
    route = np.load(f"{CAP}/route_L{L}.npy")[:NTOK].astype(np.int64)   # (NTOK,6) top-6 ids
    prim = route[:,0]                                                  # primary (highest) expert
    overall_H = entropy(collections.Counter(prim.tolist()))
    print(f"\n=== Layer {L}  (primary-expert entropy: overall={overall_H:.2f} bits, max=8.0) ===")
    print(f"{'category':10} {'ntok':>6} {'H_prim':>7} {'top-primary-experts (coverage%)'}")
    # mutual information I(category; primary expert)
    mi = 0.0; N = NTOK
    pe = collections.Counter(prim.tolist())
    pc = collections.Counter(cats)
    joint = collections.Counter(zip(cats, prim.tolist()))
    for (c,e),n in joint.items():
        mi += (n/N)*math.log2((n/N)/((pc[c]/N)*(pe[e]/N)))
    for cat in CATS:
        idx = [i for i in range(NTOK) if cats[i]==cat]
        if not idx: continue
        cprim = collections.Counter(prim[idx].tolist())
        H = entropy(cprim)
        top = cprim.most_common(5)
        cov = sum(n for _,n in top)/len(idx)*100
        tops = ",".join(str(e) for e,_ in top)
        print(f"{cat:10} {len(idx):>6} {H:>7.2f}  [{tops}] cov={cov:.0f}%")
    print(f"  --> I(category ; primary_expert) = {mi:.3f} bits   (overall H={overall_H:.2f})")
