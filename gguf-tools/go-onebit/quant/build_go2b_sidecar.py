#!/usr/bin/env python3
"""build_go2b_sidecar.py — 为最差层建 go2b(type41) overlay 侧车 (运行时 --residual 加载,
全层替换 base 专家)。逐层流式 HF pack (M1) → encode_go2b(直接NF, 无EF) → 写 GGUF。
侧车格式=残差侧车契约: blk.{L}.ffn_{gate,up,down}_exps_res.weight(go2b) + ffn_res_lut(F32恒等) + ds4.residual.present.
用法: build_go2b_sidecar.py OUT.gguf L1,L2,...  (pack 需 /tmp/sc_L{L}.{w8,si,meta.json} 就位)
"""
import os, sys, json, struct
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib", "pyfwd"))
from go2b_encode import encode_go2b
from ds4reader import LUT

GO2B, F32 = 41, 0
DIMS = {"gate": (2048, 4096), "up": (2048, 4096), "down": (4096, 2048)}  # (rows, cols)

def hf_pack(pk, ki, kind, e):
    meta = json.load(open(f"{pk}.meta.json")); K = meta["kinds"]
    r, c = K[kind]["rows"], K[kind]["cols"]; sr, scn = K[kind]["si_shape"]
    with open(f"{pk}.w8", "rb") as w8:
        off = sum(K[k]["rows"]*K[k]["cols"]*256 for k in ("gate","up","down")[:ki])
        w8.seek(off+e*r*c); a = np.frombuffer(w8.read(r*c), dtype=np.uint8).reshape(r, c)
    with open(f"{pk}.si", "rb") as si:
        soff = sum(int(np.prod(K[k]["si_shape"]))*4*256 for k in ("gate","up","down")[:ki])
        si.seek(soff+e*sr*scn*4); sc = np.frombuffer(si.read(sr*scn*4), dtype=np.float32).reshape(sr, scn)
    return LUT[a]*np.repeat(np.repeat(sc, 128, axis=0), 128, axis=1)[:r, :c]

def main():
    out_p = sys.argv[1]
    layers = [int(x) for x in sys.argv[2].split(",")]
    # 张量表: 每层 3 个 go2b + 1 个 F32 lut
    tens = []
    for L in layers:
        for kind in ("gate", "up", "down"):
            r, c = DIMS[kind]
            tens.append((f"blk.{L}.ffn_{kind}_exps_res.weight", [c, r, 256], GO2B))
        tens.append((f"blk.{L}.ffn_res_lut.weight", [256], F32))
    kvs = [("ds4.residual.present", 7, struct.pack("<B", 1))]  # type 7 = bool(uint8)
    for L in layers:
        kvs.append((f"ds4.residual.layer.{L}", 7, struct.pack("<B", 1)))

    out = open(out_p, "wb")
    def w(b): out.write(b)
    def wstr(s): b = s.encode(); w(struct.pack("<Q", len(b))); w(b)
    w(struct.pack("<IIQQ", 0x46554747, 3, len(tens), len(kvs)))
    for k, t, v in kvs:
        wstr(k); w(struct.pack("<I", t)); w(v)
    def go2b_sz(ne): return ne[1] * (ne[0] // 256) * 68 * ne[2]
    def f32_sz(ne):
        n = 1
        for x in ne: n *= x
        return n * 4
    align = 32; roff = 0; sizes = []
    for nm, ne, ty in tens:
        sz = go2b_sz(ne) if ty == GO2B else f32_sz(ne)
        sizes.append(sz)
        wstr(nm); w(struct.pack("<I", len(ne)))
        for d in ne: w(struct.pack("<Q", d))
        w(struct.pack("<IQ", ty, roff))
        roff += (sz + align - 1) // align * align
    hdr = out.tell(); data0 = (hdr + align - 1) // align * align
    w(b"\0" * (data0 - hdr))

    ti = 0
    for L in layers:
        pk = f"/tmp/sc_L{L}"
        assert os.path.exists(f"{pk}.w8"), f"missing pack {pk}"
        for ki, kind in ((0, "gate"), (1, "up"), (2, "down")):
            r, c = DIMS[kind]
            for e in range(256):
                W = hf_pack(pk, ki, kind, e)
                blk, _ = encode_go2b(W, None, mode='nf')   # 直接NF, 无EF
                out.write(blk.astype(np.uint8).tobytes())
            pad = (align - (out.tell() - data0) % align) % align; w(b"\0" * pad)
            print(f"L{L} {kind} done ({out.tell()>>20}MiB)", flush=True)
        lut = np.arange(256, dtype=np.float32)   # 恒等: 专家 e → 槽 e (全层替换)
        out.write(lut.tobytes())
        pad = (align - (out.tell() - data0) % align) % align; w(b"\0" * pad)
    out.close()
    print(f"SIDECAR-OK {out_p} {os.path.getsize(out_p)>>20}MiB layers={layers}", flush=True)

if __name__ == "__main__":
    main()
