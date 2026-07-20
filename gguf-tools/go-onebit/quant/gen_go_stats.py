#!/usr/bin/env python3
# gen_go_stats.py — L_fix statistics: per-expert Go-domain activation second
# moments -> llama.cpp-style imatrix .dat consumed by deepseek4-quantize
# --imatrix (per-expert segmented entries, names = GGUF tensor names).
#
# For each layer L and routed expert e (firing tokens T_e from cap route_L):
#   gate/up (w1/w3, input x, 4096):   E[x_j^2]     over T_e
#   down    (w2, input h, 2048):      E[(h_j*w)^2] over T_e   — h = the CLAMPED
#     SwiGLU mid with the gate weight w baked in, exactly what the runtime's
#     down projection consumes (moe.metal: mid = silu(min(g,10))*clip(u,±10)*w).
# Never-fired experts fall back to the layer mean so Θ_fix never regresses.
#
# Entries (nval = 256*ncols → imatrix_find per-expert slice, ncall=1):
#   blk.{L}.ffn_gate_exps.weight  [256*4096]
#   blk.{L}.ffn_up_exps.weight    [256*4096]  (same values as gate: same input x)
#   blk.{L}.ffn_down_exps.weight  [256*2048]
#
# Usage:
#   DS4_HF=... python3 gen_go_stats.py --cap CAPDIR --layers 5,20,35 --out gostats.dat
# Gate weights: uses cap route_w_L{L}.npy when present (post-E1 capture),
# else recomputes sqrtsoftplus@ids/norm/x1.5 from the HF router gate.
import os, sys, struct, time
import numpy as np
import ds4reader as R

NEXP, TOPK, DM, DF = 256, 6, 4096, 2048
SWLIM = 10.0

def silu(x): return x / (1.0 + np.exp(-x))

def parse_args(argv):
    a = {"cap": None, "layers": None, "out": "gostats.dat"}
    i = 1
    while i < len(argv):
        if argv[i] == "--cap": a["cap"] = argv[i+1]; i += 2
        elif argv[i] == "--layers": a["layers"] = argv[i+1]; i += 2
        elif argv[i] == "--out": a["out"] = argv[i+1]; i += 2
        else: raise SystemExit(f"unknown arg {argv[i]}")
    if not a["cap"] or not a["layers"]: raise SystemExit("need --cap and --layers")
    return a

def gate_weights(x, ids, gate_w):
    """sqrtsoftplus at the selected ids, normalized over top-6, x1.5 —
    identical to the capture/runtime weight formula (hash layers included:
    their selection differed, but selection is given by ids)."""
    raw = x @ gate_w.T                       # [n,256]
    sc = np.sqrt(np.log1p(np.exp(raw)))
    w = np.take_along_axis(sc, ids, 1)
    w = w / w.sum(1, keepdims=True) * 1.5
    return w.astype(np.float32)

def layer_stats(cap, L):
    x_all = np.load(f"{cap}/ffn_in_L{L}.npy").astype(np.float32)     # [N,4096]
    ids = np.load(f"{cap}/route_L{L}.npy").astype(np.int64)          # [N,6]
    n = x_all.shape[0]
    rw_path = f"{cap}/route_w_L{L}.npy"
    if os.path.exists(rw_path):
        rw = np.load(rw_path).astype(np.float32)
        src = "captured route_w"
    else:
        gate_w = R.get(f"layers.{L}.ffn.gate.weight").astype(np.float32)
        rw = gate_weights(x_all, ids, gate_w)
        src = "recomputed"
    ex2 = np.zeros((NEXP, DM), np.float64)
    eh2 = np.zeros((NEXP, DF), np.float64)
    fired = np.zeros(NEXP, bool)
    t0 = time.time()
    flat_e = ids.reshape(-1); flat_w = rw.reshape(-1)
    tok = np.repeat(np.arange(n), TOPK)
    for e in np.unique(flat_e):
        sel = flat_e == e
        t = tok[sel]; w = flat_w[sel][:, None]
        xe = x_all[t]                                                # [m,4096]
        ex2[e] = np.mean(xe.astype(np.float64) ** 2, 0)
        w1 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight")
        w3 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight")
        g = xe @ w1.T; u = xe @ w3.T
        g = np.minimum(g, SWLIM); u = np.clip(u, -SWLIM, SWLIM)
        h = silu(g) * u * w                                          # weighted mid = w2 input
        eh2[e] = np.mean(h.astype(np.float64) ** 2, 0)
        fired[e] = True
    nf = int(fired.sum())
    if nf < NEXP:   # never-fired: layer-mean fallback (encoder also self-guards)
        mx = ex2[fired].mean(0); mh = eh2[fired].mean(0)
        ex2[~fired] = mx; eh2[~fired] = mh
    print(f"L{L}: fired {nf}/256  weights={src}  {time.time()-t0:.0f}s", flush=True)
    return ex2.astype(np.float32), eh2.astype(np.float32)

def write_dat(path, entries):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(entries)))
        for name, vals in entries:
            nb = name.encode()
            f.write(struct.pack("<i", len(nb))); f.write(nb)
            f.write(struct.pack("<ii", 1, vals.size))   # ncall=1: values stored final
            f.write(vals.astype("<f4").tobytes())

def main():
    a = parse_args(sys.argv)
    layers = [int(t) for t in a["layers"].split(",")]
    entries = []
    for L in layers:
        ex2, eh2 = layer_stats(a["cap"], L)
        entries.append((f"blk.{L}.ffn_gate_exps.weight", ex2.reshape(-1)))
        entries.append((f"blk.{L}.ffn_up_exps.weight",   ex2.reshape(-1)))
        entries.append((f"blk.{L}.ffn_down_exps.weight", eh2.reshape(-1)))
    write_dat(a["out"], entries)
    print(f"wrote {a['out']}: {len(entries)} entries ({len(layers)} layers)", flush=True)

if __name__ == "__main__":
    main()
