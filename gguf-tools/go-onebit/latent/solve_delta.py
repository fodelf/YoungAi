#!/usr/bin/env python3
# solve_delta.py — L_cls δ, closed form: per layer (learned routing only),
#   δ[e] = w · mean_t( teacher_raw_logit[t,e] − student_raw_logit[t,e] )
# positions align because teacher capture and student trajectory ran the same
# teacher-forced token stream. Patches δ into z_L{L}.bin files in place
# (header + U,V,C,b,beta,delta layout from calib_run's dump).
# Usage: solve_delta.py --teacher TEACHER_CAP --student STUDENT_CAP \
#            --zdir ZDIR --layers 3,4,...,42 [--w 1.0]
import os, sys, struct
import numpy as np

def arg(k, d=None):
    return sys.argv[sys.argv.index(k)+1] if k in sys.argv else d

tcap, scap, zdir = arg("--teacher"), arg("--student"), arg("--zdir")
tname = arg("--teacher-name", "route_logits")   # engine-EF cap: teacher logits live in
sname = arg("--student-name", "route_logits")   # the same dir as route_logits_teacher_L
w = float(arg("--w", "1.0"))
layers = [int(t) for t in arg("--layers").split(",")]
for L in layers:
    if L < 3:
        print(f"L{L}: hash layer, δ not applicable — skip"); continue
    zp = f"{zdir}/z_L{L}.bin"
    if not os.path.exists(zp):
        print(f"L{L}: no z file — skip"); continue
    tr = np.load(f"{tcap}/{tname}_L{L}.npy").astype(np.float64)
    sr = np.load(f"{scap}/{sname}_L{L}.npy").astype(np.float64)
    n = min(len(tr), len(sr))
    delta = w * (tr[:n] - sr[:n]).mean(0)                     # [256]
    with open(zp, "r+b") as f:
        d, ne, dl = struct.unpack("<iii", f.read(12))
        off = 12 + 4 * (d*dl + dl*d + ne*dl + d + ne)         # -> delta block
        f.seek(off)
        f.write(delta.astype("<f4").tobytes())
    print(f"L{L}: δ patched  |δ|max={np.abs(delta).max():.4f} mean={np.abs(delta).mean():.4f}", flush=True)
print("SOLVE-DELTA DONE", flush=True)
