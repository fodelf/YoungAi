#!/usr/bin/env python3
"""deep_survey_and_blob.py — 深五层(L38-42)同代真尺普查 + L38 残差平面 go1b blob 打包.
前提: /tmp/cap_v3prep 已换装 deep 代 route/route_w/routed_paired (L38-42), ffn_in 本就 deep 代.
输出: 每层一行 [同代真尺] 底座还原度; L38 残差 blob /tmp/l38_residual_go1b.bin.
"""
import numpy as np, json, sys, os
sys.path.insert(0, "/Users/fodelf/git/ds4-main/gguf-tools/go-onebit/quant")
sys.path.insert(0, "/Users/fodelf/git/ds4-main/gguf-tools/go-onebit/calib/pyfwd")
from gptq1_rewrite import parse_gguf, read_expert_blocks, blocks_dequant, deq, swiglu
from ds4reader import LUT

CAP = "/tmp/cap_v3prep"
g = open("/Users/fodelf/git/ds4-main/gguf/ds4-go1b-v3.gguf", "rb")
tens, data0 = parse_gguf(g)

def T(L):
    d = {}
    for kind in ("gate", "up", "down"):
        ne, ty, off = tens[f"blk.{L}.ffn_{kind}_exps.weight"]
        d[kind] = (int(ne[0]), int(ne[1]), off)
    return d

def base_w(TT, kind, e):
    cols, rows, toff = TT[kind]
    raw, _ = read_expert_blocks(g, data0, toff, e, rows, cols)
    s, B = blocks_dequant(raw, cols)
    return deq(s, B)

gains = {}
for L in range(38, 43):
    yt = np.load(f"{CAP}/routed_paired_L{L}.npy").astype(np.float32)
    X = np.load(f"{CAP}/ffn_in_L{L}.npy").astype(np.float32)
    route = np.load(f"{CAP}/route_L{L}.npy")
    rw = np.load(f"{CAP}/route_w_L{L}.npy").astype(np.float32)
    TT = T(L)
    n = min(len(yt), len(route))
    y = np.zeros((n, 4096), dtype=np.float32)
    for e in range(256):
        sel = np.nonzero(route[:n] == e)
        if sel[0].size == 0:
            continue
        h = swiglu(X[sel[0]] @ base_w(TT, "gate", e).T, X[sel[0]] @ base_w(TT, "up", e).T, 10.0)
        np.add.at(y, sel[0], (rw[:n][sel][:, None] * (h @ base_w(TT, "down", e).T)))
    nch = n // 512; nte = max(1, int(0.17 * nch + 0.5)); cut = (nch - nte) * 512
    num = np.sum(yt[:cut] * y[:cut], 1); den = np.sum(y[:cut] * y[:cut], 1) + 1e-20
    gr = float(np.median(num / den)); gains[L] = gr
    e2 = yt[cut:] - gr * y[cut:]
    cos = float(np.mean(np.sum(yt[cut:] * y[cut:], 1) /
                        (np.linalg.norm(yt[cut:], axis=1) * np.linalg.norm(y[cut:], axis=1) + 1e-20)))
    r2 = (1 - float(np.sum(e2 * e2)) / float(np.sum(yt[cut:] * yt[cut:]))) * 100
    print(f"L{L}[同代真尺] 底座: g={gr:.2f} cos={cos:.3f} restore={r2:.1f}%", flush=True)
    np.save(f"{CAP}/obase_v3_L{L}.npy", y.astype(np.float16))
json.dump(gains, open("/Users/fodelf/git/ds4-main/gguf/v3-artifacts/gains_deep_true.json", "w"), indent=1)

# L38 残差平面 blob (对当前部署字节的 Q1 残差, go1b 34B 块, kind-major expert-major)
L = 38
meta = json.load(open("/tmp/cd_L38.meta.json")); K = meta["kinds"]
w8 = open("/tmp/cd_L38.w8", "rb"); si = open("/tmp/cd_L38.si", "rb")
def hf_w(ki, kind, e):
    r, c = K[kind]["rows"], K[kind]["cols"]; sr, scn = K[kind]["si_shape"]
    off = sum(K[k]["rows"] * K[k]["cols"] * 256 for k in ("gate", "up", "down")[:ki])
    w8.seek(off + e * r * c)
    a = np.frombuffer(w8.read(r * c), dtype=np.uint8).reshape(r, c)
    soff = sum(int(np.prod(K[k]["si_shape"])) * 4 * 256 for k in ("gate", "up", "down")[:ki])
    si.seek(soff + e * sr * scn * 4)
    sc = np.frombuffer(si.read(sr * scn * 4), dtype=np.float32).reshape(sr, scn)
    return LUT[a] * np.repeat(np.repeat(sc, 128, axis=0), 128, axis=1)[:r, :c]
TT = T(L)
out = open("/Users/fodelf/git/ds4-main/gguf/v3-artifacts/l38_residual_go1b.bin", "wb")
tot = 0
for ki, kind in ((0, "gate"), (1, "up"), (2, "down")):
    cols, rows, toff = TT[kind]; nblk = cols // 256
    for e in range(256):
        raw, _ = read_expert_blocks(g, data0, toff, e, rows, cols)
        s, B = blocks_dequant(raw, cols)
        R = hf_w(ki, kind, e) - deq(s, B)
        sr_ = np.abs(R).mean(axis=1).astype(np.float16)
        blk = np.zeros((rows, nblk, 34), dtype=np.uint8)
        blk[:, :, 0:2] = np.repeat(sr_.view(np.uint8).reshape(rows, 1, 2), nblk, axis=1)
        bits = (R >= 0).astype(np.uint8).reshape(rows, nblk, 256)
        blk[:, :, 2:34] = np.packbits(bits, axis=2, bitorder="little")
        out.write(blk.tobytes()); tot += blk.nbytes
    print(f"resblob {kind} done", flush=True)
out.close()
print(f"RESBLOB-OK bytes={tot} ({tot/2**20:.0f} MiB) -> /Users/fodelf/git/ds4-main/gguf/v3-artifacts/l38_residual_go1b.bin", flush=True)
