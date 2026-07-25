#!/usr/bin/env python3
"""go2b_overlay_from_sidecars.py — dql_go2b_L*.bin(量化器侧车) → 引擎残差格式 overlay GGUF。
零再编码: 68B 块字节直拷。格式与 build_go2b_hot.py 输出逐字段同(引擎 07-24 已验加载路径):
  blk.L.ffn_{gate,up,down}_exps_res.weight [GO2B(41), ne=[cols,rows,Khot]] + blk.L.ffn_res_lut.weight [256,F32]
  KV: ds4.residual.present + ds4.residual.layer.L
侧车头: 'DQG2' u32|ver u32|L u32|khot u32|mean_cos f32|rsv u32|ids u16[khot]|pad8 → gate[k×szG] up[k×szG] down[k×szD]
用法: go2b_overlay_from_sidecars.py OUT.gguf SIDECAR_DIR [L0-L42]
"""
import os, sys, struct
import numpy as np

GO2B, F32 = 41, 0
SZG = 2048 * 16 * 68          # 每专家 gate/up 字节(2048行×16块×68B)
SZD = 4096 * 8 * 68           # 每专家 down

def read_hdr(p):
    f = open(p, "rb")
    mg, ver, L, kh = struct.unpack("<IIII", f.read(16))
    assert mg == 0x32475144, f"{p}: magic 不对"
    struct.unpack("<fI", f.read(8))
    ids = struct.unpack(f"<{kh}H", f.read(2 * kh))
    hdr = (24 + 2 * kh + 7) // 8 * 8
    return f, L, kh, ids, hdr

def main():
    out_p, sdir = sys.argv[1], sys.argv[2]
    layers = list(range(43))
    meta = {}
    for L in layers:
        p = f"{sdir}/dql_go2b_L{L:02d}.bin"
        f, Lh, kh, ids, hdr = read_hdr(p)
        assert Lh == L
        need = hdr + 2 * kh * SZG + kh * SZD
        assert os.path.getsize(p) >= need, f"{p}: 尺寸不足"
        meta[L] = (p, kh, ids, hdr); f.close()
    tens = []
    for L in layers:
        _, kh, _, _ = meta[L]
        tens.append((f"blk.{L}.ffn_gate_exps_res.weight", [4096, 2048, kh], GO2B))
        tens.append((f"blk.{L}.ffn_up_exps_res.weight",   [4096, 2048, kh], GO2B))
        tens.append((f"blk.{L}.ffn_down_exps_res.weight", [2048, 4096, kh], GO2B))
        tens.append((f"blk.{L}.ffn_res_lut.weight", [256], F32))
    kvs = [("ds4.residual.present", 7, b"\x01")] + \
          [(f"ds4.residual.layer.{L}", 7, b"\x01") for L in layers]
    out = open(out_p, "wb")
    def w(b): out.write(b)
    def wstr(s): b = s.encode(); w(struct.pack("<Q", len(b))); w(b)
    w(struct.pack("<IIQQ", 0x46554747, 3, len(tens), len(kvs)))
    for k, t, v in kvs: wstr(k); w(struct.pack("<I", t)); w(v)
    def sz_of(ne, ty): return ne[1] * (ne[0] // 256) * 68 * ne[2] if ty == GO2B else int(np.prod(ne)) * 4
    align = 32; roff = 0
    for nm, ne, ty in tens:
        wstr(nm); w(struct.pack("<I", len(ne)))
        for d in ne: w(struct.pack("<Q", d))
        w(struct.pack("<IQ", ty, roff)); roff += (sz_of(ne, ty) + align - 1) // align * align
    hdrsz = out.tell(); data0 = (hdrsz + align - 1) // align * align; w(b"\0" * (data0 - hdrsz))
    for L in layers:
        p, kh, ids, hdr = meta[L]
        src = open(p, "rb"); src.seek(hdr)
        for region in (kh * SZG, kh * SZG, kh * SZD):   # gate → up → down 连续区直拷
            left = region
            while left > 0:
                b = src.read(min(left, 64 << 20)); w(b); left -= len(b)
            pad = (align - (out.tell() - data0) % align) % align; w(b"\0" * pad)
        src.close()
        lut = np.full(256, -1.0, dtype=np.float32)
        for slot, e in enumerate(ids): lut[e] = float(slot)
        w(lut.tobytes())
        pad = (align - (out.tell() - data0) % align) % align; w(b"\0" * pad)
        print(f"L{L:02d} khot={kh} ✓", flush=True)
    out.close()
    print(f"OVERLAY-OK {out_p} {os.path.getsize(out_p) >> 20}MiB", flush=True)

if __name__ == "__main__":
    main()
