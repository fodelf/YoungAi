#!/usr/bin/env python3
# gguf_range_stream.py — 把一个 GGUF 里若干层的 ffn_*_exps 专家张量字节以
# [u64 off][u64 size][bytes] 帧流发送/接收，用于把 v2 底座专家字节 splice 进
# 另一台机器的同布局稀疏切片文件（quant_assemble 系）。
#   send: gguf_range_stream.py send FILE LO-HI            (帧流 → stdout)
#   recv: gguf_range_stream.py recv FILE                  (stdin 帧流 → 按偏移写入)
# 偏移是文件绝对偏移（含 data 段基址），两侧文件布局必须同源。
import sys, struct

def parse_header(f):
    def rd(n):
        b = f.read(n)
        assert len(b) == n, "short read"
        return b
    magic, ver, n_t, n_kv = struct.unpack("<IIQQ", rd(24))
    assert magic == 0x46554747, "not gguf"
    def rstr():
        n, = struct.unpack("<Q", rd(8))
        return rd(n).decode()
    def skipv(t):
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8: rstr()
        elif t == 9:
            et, = struct.unpack("<I", rd(4)); n, = struct.unpack("<Q", rd(8))
            if et == 8:
                for _ in range(n): rstr()
            else: rd(sz[et]*n)
        else: rd(sz[t])
    for _ in range(n_kv):
        rstr(); t, = struct.unpack("<I", rd(4)); skipv(t)
    tens = {}
    for _ in range(n_t):
        nm = rstr(); nd, = struct.unpack("<I", rd(4))
        ne = struct.unpack("<%dQ" % nd, rd(8*nd))
        ty, = struct.unpack("<I", rd(4)); off, = struct.unpack("<Q", rd(8))
        tens[nm] = (ty, ne, off)
    data0 = (f.tell() + 31) // 32 * 32
    return tens, data0

def go1b_bytes(ne):  # ne=(cols, rows, nexp), 34B/256块
    return ne[0] // 256 * 34 * ne[1] * ne[2]

if __name__ == "__main__" and len(sys.argv) < 2:
    sys.exit("usage: send FILE LO-HI | recv FILE")

if __name__ != "__main__":
    pass
elif sys.argv[1] == "send":
    path, rng = sys.argv[2], sys.argv[3]
    lo, hi = map(int, rng.split("-"))
    f = open(path, "rb")
    tens, data0 = parse_header(f)
    out = sys.stdout.buffer
    total = 0
    for L in range(lo, hi + 1):
        for kind in ("gate", "up", "down"):
            nm = f"blk.{L}.ffn_{kind}_exps.weight"
            ty, ne, off = tens[nm]
            assert ty == 40, f"{nm} type {ty} != go1b"
            size, ab = go1b_bytes(ne), data0 + off
            out.write(struct.pack("<QQ", ab, size))
            f.seek(ab)
            left = size
            while left:
                chunk = f.read(min(left, 1 << 24))
                out.write(chunk); left -= len(chunk)
            total += size
        print(f"send L{L} 完成 (累计 {total>>20} MiB)", file=sys.stderr, flush=True)
elif sys.argv[1] == "recv":
    path = sys.argv[2]
    f = open(path, "r+b")
    src = sys.stdin.buffer
    n = 0
    while True:
        hdr = src.read(16)
        if not hdr: break
        assert len(hdr) == 16
        off, size = struct.unpack("<QQ", hdr)
        f.seek(off)
        left = size
        while left:
            chunk = src.read(min(left, 1 << 24))
            assert chunk, "stream truncated"
            f.write(chunk); left -= len(chunk)
        n += 1
        print(f"recv frame {n} off={off} {size>>20} MiB", file=sys.stderr, flush=True)
    f.close()
    print(f"recv done: {n} frames", file=sys.stderr)
else:
    sys.exit("usage: send FILE LO-HI | recv FILE")
