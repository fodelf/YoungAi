#!/usr/bin/env python3
# check_capture.py — E1 capture-extension smoke gate.
# Compares a fresh capture (--new, e.g. 1 chunk) against the existing gold cap
# (--old) on the shared token prefix, and self-checks the new signals:
#   ffn_in/route identical (deterministic replay)  → pipeline unchanged
#   routed close to old gold                       → new routed-only = old tool's
#   route_w rows sum to 1.5 (norm_topk × scaling)  → weight formula intact
#   route_logits→sqrtsoftplus+gbias→top6 == route  → logits are the real raw scores
# Usage: DS4_HF=... python3 check_capture.py --new DIR --old DIR --layers 3,8,20,42
import sys, numpy as np, os
import ds4reader as R

def arg(k, d=None):
    return sys.argv[sys.argv.index(k)+1] if k in sys.argv else d

new, old = arg("--new"), arg("--old")
layers = [int(t) for t in arg("--layers", "3,8").split(",")]
ok_all = True
for L in layers:
    fi_n = np.load(f"{new}/ffn_in_L{L}.npy"); n = fi_n.shape[0]
    fi_o = np.load(f"{old}/ffn_in_L{L}.npy")[:n]
    rt_n = np.load(f"{new}/route_L{L}.npy"); rt_o = np.load(f"{old}/route_L{L}.npy")[:n]
    rd_n = np.load(f"{new}/routed_L{L}.npy").astype(np.float32)
    rd_o = np.load(f"{old}/routed_L{L}.npy")[:n].astype(np.float32)
    rw = np.load(f"{new}/route_w_L{L}.npy").astype(np.float32)
    rl = np.load(f"{new}/route_logits_L{L}.npy").astype(np.float32)

    in_eq = np.array_equal(fi_n, fi_o)
    rt_eq = np.array_equal(np.sort(rt_n, 1), np.sort(rt_o, 1))
    num = np.linalg.norm(rd_n - rd_o); den = np.linalg.norm(rd_o) + 1e-9
    rd_rel = float(num / den)
    wsum = float(np.abs(rw.sum(1) - 1.5).max())
    # raw logits → selection replay (learned-routing layers only; L0-2 are hash)
    sel_ok = True
    if L >= 3:
        gb = R.get(f"layers.{L}.ffn.gate.bias").astype(np.float32)
        sc = np.sqrt(np.log1p(np.exp(rl))) + gb[None]
        top6 = np.argpartition(-sc, 5, axis=1)[:, :6]
        sel_ok = np.array_equal(np.sort(top6, 1), np.sort(rt_n.astype(np.int64), 1))
    ok = in_eq and rt_eq and rd_rel < 0.02 and wsum < 1e-2 and sel_ok
    ok_all &= ok
    print(f"L{L}: ffn_in_eq={in_eq} route_eq={rt_eq} routed_rel={rd_rel:.5f} "
          f"|w_sum-1.5|max={wsum:.2e} logits_top6_eq={sel_ok}  {'OK' if ok else 'FAIL'}")
print("CAPTURE-CHECK", "PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
