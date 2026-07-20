#!/usr/bin/env python3
# merge_caps.py — concatenate two chunk-parallel capture dirs (a: chunks LO..M,
# b: chunks M..HI) into one, preserving chunk order. Chunks are independent
# 512-token sequences, so row-wise concat per layer file is exact.
# Usage: merge_caps.py A_DIR B_DIR OUT_DIR N_LAYERS
import sys, os
import numpy as np

a, b, out, nl = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
os.makedirs(out, exist_ok=True)
kinds = ["ffn_in", "ffn_out", "route", "route_logits", "route_w", "routed"]
for L in range(nl):
    for k in kinds:
        fa, fb = f"{a}/{k}_L{L}.npy", f"{b}/{k}_L{L}.npy"
        va, vb = np.load(fa), np.load(fb)
        np.save(f"{out}/{k}_L{L}.npy", np.concatenate([va, vb], 0))
    print(f"L{L} merged", flush=True)
for k in ["final_topk_idx", "final_topk_val"]:
    va, vb = np.load(f"{a}/{k}.npy"), np.load(f"{b}/{k}.npy")
    np.save(f"{out}/{k}.npy", np.concatenate([va, vb], 0))
print("MERGE DONE", flush=True)
