#!/usr/bin/env python3
# gen_active_topk.py — 从 capture 的 route_L*.npy 点火计数生成 emit_residual
# 的 --active-experts 文件（"L{n}: e0 e1 ..."，每层按点火数 top-K）。
# 用法: gen_active_topk.py CAP_DIR K OUT [--layers lo-hi]
import sys, glob
import numpy as np

cap, K, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lo, hi = 0, 42
if "--layers" in sys.argv:
    lo, hi = map(int, sys.argv[sys.argv.index("--layers") + 1].split("-"))
files = {int(f.split("_L")[1].split(".")[0]): f for f in glob.glob(cap + "/route_L*.npy")}
lines = []
for L in range(lo, hi + 1):
    if L not in files:
        print(f"L{L}: 无 route npy，跳过", file=sys.stderr)
        continue
    ids = np.load(files[L]).astype(np.int64).ravel()
    cnt = np.bincount(ids, minlength=256)
    top = sorted(int(i) for i in np.argsort(-cnt)[:K])
    lines.append(f"L{L}: " + " ".join(map(str, top)))
open(out, "w").write("\n".join(lines) + "\n")
print(f"{out}: {len(lines)} 层 × top-{K}", file=sys.stderr)
