#!/usr/bin/env python3
"""vq_overlay_from_sidecars.py — dql_vq_L*.bin(战役侧车) → 引擎 VQ overlay GGUF(v2.2)。
零再编码: 每层一个 opaque blob 张量 = 侧车文件原字节(hdr16 + 256×3 u64 表 + DQVQ 载荷)。
  blk.L.ffn_exps_vq.blob [nbytes] type=VQ_BLOB(42)
  KV: ds4.vq.present=true + ds4.vq.layer.L=true
引擎侧: mmap blob → 读内嵌表 → 按 (e,which) 取 DQVQ 载荷 → gather 时 dequant→f16。
用法: vq_overlay_from_sidecars.py OUT.gguf SIDECAR_DIR [NL=43]
"""
import os, sys, struct

VQ_BLOB = 42
GGUF_MAGIC = 0x46554747

def main():
    out_p, sdir = sys.argv[1], sys.argv[2]
    nl = int(sys.argv[3]) if len(sys.argv) > 3 else 43
    blobs = []
    for L in range(nl):
        p = f"{sdir}/dql_vq_L{L:02d}.bin"
        sz = os.path.getsize(p)
        with open(p, "rb") as f:
            mg, ver, Lh, ne = struct.unpack("<4I", f.read(16))
        assert mg == 0x4C565144 and Lh == L and ne == 256, f"{p}: 头不对 {hex(mg)} L{Lh} ne{ne}"
        blobs.append((p, sz))
        print(f"L{L:02d} {sz/1048576:.1f}MiB", file=sys.stderr)
    tens = [(f"blk.{L}.ffn_exps_vq.blob", [blobs[L][1]], VQ_BLOB) for L in range(nl)]
    kvs = [("ds4.vq.present", 7, b"\x01")] + [(f"ds4.vq.layer.{L}", 7, b"\x01") for L in range(nl)]
    out = open(out_p, "wb")
    def w(b): out.write(b)
    def wstr(s):
        b = s.encode(); w(struct.pack("<Q", len(b))); w(b)
    w(struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tens), len(kvs)))
    for k, t, v in kvs:
        wstr(k); w(struct.pack("<I", t)); w(v)
    off, align = 0, 4096
    offs = []
    for name, ne, ty in tens:
        wstr(name); w(struct.pack("<I", len(ne)))
        for d in ne: w(struct.pack("<Q", d))
        w(struct.pack("<IQ", ty, off))
        off = (off + ne[0] + align - 1) // align * align
        offs.append(off)
    pos = out.tell()
    pad = (pos + align - 1) // align * align - pos
    w(b"\x00" * pad)
    base = out.tell()
    for i, (p, sz) in enumerate(blobs):
        cur = out.tell() - base
        want = (offs[i - 1] if i > 0 else 0)
        if cur < want: w(b"\x00" * (want - cur))
        with open(p, "rb") as f:
            while True:
                chunk = f.read(1 << 22)
                if not chunk: break
                w(chunk)
    out.close()
    print(f"→ {out_p} ({os.path.getsize(out_p)/2**30:.2f} GiB, {nl} 层)", file=sys.stderr)

if __name__ == "__main__":
    main()
