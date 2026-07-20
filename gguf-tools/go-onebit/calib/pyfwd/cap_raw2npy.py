#!/usr/bin/env python3
# cap_raw2npy.py — convert the engine batch-capture raw shards (DS4_CAP_DIR)
# into the standard cap npy schema consumed by the whole calibration chain.
#   raw_ffn_in_L{L}       f16 [n×4096] -> ffn_in_L{L}.npy       (float16)
#   raw_route_L{L}        i16 [n×6]    -> route_L{L}.npy        (int16)
#   raw_route_logits_L{L} f16 [n×256]  -> route_logits_L{L}.npy (float16)
#   raw_route_w_L{L}      f16 [n×6]    -> route_w_L{L}.npy      (float16)
# Token count is inferred from byte size; all four widths must agree per layer.
# Usage: cap_raw2npy.py RAW_DIR OUT_DIR [layers e.g. 20-42]
import sys, os
import numpy as np

raw, out = sys.argv[1], sys.argv[2]
rng = sys.argv[3] if len(sys.argv) > 3 else "0-42"
lo, hi = (int(t) for t in rng.split("-"))
os.makedirs(out, exist_ok=True)
DM, NE, KU = 4096, 256, 6
kinds = [("raw_ffn_in", "ffn_in", np.float16, DM),
         ("raw_route", "route", np.int16, KU),
         ("raw_route_logits", "route_logits", np.float16, NE),
         ("raw_route_w", "route_w", np.float16, KU)]
for L in range(lo, hi + 1):
    ns = []
    for rk, ok_, dt, w in kinds:
        p = f"{raw}/{rk}_L{L}"
        if not os.path.exists(p):
            ns = None; break
        a = np.fromfile(p, dtype=dt)
        if a.size % w:
            raise SystemExit(f"L{L} {rk}: {a.size} not divisible by {w}")
        ns.append(a.size // w)
        np.save(f"{out}/{ok_}_L{L}.npy", a.reshape(-1, w))
    if ns is None:
        print(f"L{L}: missing shards — skipped"); continue
    if len(set(ns)) != 1:
        raise SystemExit(f"L{L}: token counts disagree {ns}")
    print(f"L{L}: {ns[0]} tokens converted")
print("RAW2NPY DONE")
