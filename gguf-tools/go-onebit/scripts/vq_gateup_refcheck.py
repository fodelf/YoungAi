#!/usr/bin/env python3
"""fused gateup 对拍: 引擎 dump(/tmp/vq_x.bin, vq_h.bin, vq_sel.bin) vs 层件 blob 参考。"""
import struct, sys
import numpy as np
LAYER = sys.argv[1] if len(sys.argv) > 1 else "/home/fodelf/ds4-main/gguf/go-onebit/r30/full/layers/dql_vq_L00.bin"
blob = open(LAYER, "rb").read()
x = np.fromfile("/tmp/vq_x.bin", dtype=np.float32)
h_eng = np.fromfile("/tmp/vq_h.bin", dtype=np.float32)[:2048]
raw = open("/tmp/vq_sel.bin", "rb").read()
sel = struct.unpack("<6i", raw[:24]); wts = struct.unpack("<6f", raw[24:48])
print("sel", sel, "w", ["%.3f" % w for w in wts])

def f16(b):
    return np.frombuffer(b, dtype=np.float16).astype(np.float32)

def slot(e, which):
    o, = struct.unpack("<Q", blob[16 + (e * 3 + which) * 8: 24 + (e * 3 + which) * 8])
    return o

def dequant(o, rows, cols):
    mg, dim, nc = struct.unpack("<IHH", blob[o:o+8])
    r, c = struct.unpack("<II", blob[o+8:o+16])
    assert (r, c) == (rows, cols), (r, c)
    cb = f16(blob[o+16 : o+16+nc*dim*2]).reshape(nc, dim)
    gro = o + 16 + nc*dim*2
    gr = f16(blob[gro : gro+rows*2])
    ixo = gro + rows*2
    nbit = max(1, (nc-1).bit_length())
    nidx = rows * cols // dim
    # 位流解码
    bits = np.unpackbits(np.frombuffer(blob[ixo: ixo + (nidx*nbit+7)//8], dtype=np.uint8), bitorder="little")
    idx = np.zeros(nidx, dtype=np.int64)
    for b in range(nbit):
        idx |= bits[b::nbit][:nidx].astype(np.int64) << b
    W = cb[idx].reshape(rows, cols // dim, dim).reshape(rows, cols)
    return W * gr[:, None]

e0 = sel[0]
G = dequant(slot(e0, 0), 2048, 4096)
U = dequant(slot(e0, 1), 2048, 4096)
g = G @ x; u = U @ x
h_ref = (g / (1 + np.exp(-np.clip(g, -30, 30)))) * u
def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-30))
print(f"h cos = {cos(h_ref, h_eng):.6f}  ref[:4]={h_ref[:4]}  eng[:4]={h_eng[:4]}")

# down 段对拍: out_ref = Σ_e w_e · D_e @ h_e (h 用引擎 dump 的 6 pair)
h6 = np.fromfile("/tmp/vq_h.bin", dtype=np.float32).reshape(6, 2048)
try:
    out_eng = np.fromfile("/tmp/vq_out.bin", dtype=np.float32)
    out_ref = np.zeros(4096, dtype=np.float64)
    for k in range(6):
        D = dequant(slot(sel[k], 2), 4096, 2048)
        out_ref += wts[k] * (D @ h6[k])
    out_ref = out_ref.astype(np.float32)
    print(f"down out cos = {cos(out_ref, out_eng):.6f}  ref[:3]={out_ref[:3]}  eng[:3]={out_eng[:3]}")
except FileNotFoundError as e:
    print("down skip:", e)
