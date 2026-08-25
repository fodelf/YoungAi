#!/usr/bin/env python3
"""zlever/trace_ladder.py — 单token逐层分叉梯(诊断, 2026-08-24 夜)。
读 traceH_*.bin(int32 L, int32 row, HCM*DIM f32 逐条追加) + wt2 锚 H(FP 参考),
对每个追踪行输出逐层: relL2(armed vs bare) / relL2(bare vs FP) / relL2(armed vs FP),
并对齐 trace_clean.log 里该行每层的 |zd| 注入量。
用法: trace_ladder.py <traceH_bare> <traceH_armed> <anchor_wt2> <trace_clean.log>
"""
import sys, os, re
import numpy as np

pb, pa, anc, lg = sys.argv[1:5]
D = 4096; HCM = 4; ROW = HCM * D; NL = 43; S = 2653

def load(p):
    d = {}
    with open(p, "rb") as f:
        while True:
            hd = f.read(8)
            if len(hd) < 8: break
            L, r = np.frombuffer(hd, dtype=np.int32)
            d[(int(L), int(r))] = np.frombuffer(f.read(ROW*4), dtype=np.float32).astype(np.float64)
    return d

B = load(pb); A = load(pa)
rows = sorted({r for (_, r) in B})
# 锚 H: hdr40 + fin[NL][S][D] + ridx/rw 2*[NL][S][6] + H[NL][S][HCM*D]
base = 40 + NL*S*D*4 + 2*NL*S*6*4
anch = np.memmap(anc, dtype=np.float32, mode="r", offset=base, shape=(NL, S, ROW))
# zd 注入行: [TRACE] L=%d row=%d type6 |zd|=%f |routed|=%f sc2=%f
zd = {}
for m in re.finditer(r"\[TRACE\] L=(\d+) row=(\d+) type6 \|zd\|=([\d.]+) \|routed\|=([\d.]+) sc2=([\d.]+)", open(lg, errors="replace").read()):
    zd[(int(m[1]), int(m[2]))] = (float(m[3]), float(m[4]), float(m[5]))

def rel(x, y):
    return np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-30)

for r in rows:
    print(f"===== row {r} =====")
    print(f"{'L':>3} {'armed-vs-bare':>13} {'bare-vs-FP':>11} {'armed-vs-FP':>12} {'|zd|':>8} {'|routed|':>9} {'sc2':>5}")
    prev = 0.0
    for L in range(NL):
        if (L, r) not in B: continue
        ab = rel(A[(L, r)], B[(L, r)])
        bf = rel(B[(L, r)], np.asarray(anch[L, r], dtype=np.float64))
        af = rel(A[(L, r)], np.asarray(anch[L, r], dtype=np.float64))
        z = zd.get((L, r))
        mark = " ◀跳变" if ab - prev > 0.05 else ""
        print(f"{L:>3} {ab:>13.4f} {bf:>11.4f} {af:>12.4f} " +
              (f"{z[0]:>8.3f} {z[1]:>9.2f} {z[2]:>5.2f}" if z else " "*24) + mark)
        prev = ab
