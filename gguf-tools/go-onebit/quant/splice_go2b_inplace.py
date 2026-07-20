#!/usr/bin/env python3
"""splice_go2b_inplace.py — 把 go2b EF bin 原地覆写进单块 mono 的对应层 expert 张量。
EF go2b 与 direct-NF go2b 同为 68B 块、同偏移 → 原地覆写零额外空间, mono 就地升级成 EF 版。
bin 布局: kind-major(gate,up,down) expert-major, 68B 块。
用法: splice_go2b_inplace.py --gguf mono.gguf --layer L --bin go2b_ef_L{L}.bin
"""
import sys, struct, argparse, os

def parse_off(f, L):
    def rd(n):
        b = f.read(n); assert len(b) == n; return b
    magic, ver, n_t, n_kv = struct.unpack("<IIQQ", rd(24)); assert magic == 0x46554747
    def rstr(): n, = struct.unpack("<Q", rd(8)); return rd(n).decode()
    def skip(t):
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8: rstr(); return
        if t == 9:
            et, = struct.unpack("<I", rd(4)); n, = struct.unpack("<Q", rd(8))
            [skip(et) for _ in range(n)]; return
        rd(sz[t])
    for _ in range(n_kv): rstr(); t, = struct.unpack("<I", rd(4)); skip(t)
    offs = {}
    for _ in range(n_t):
        nm = rstr(); nd, = struct.unpack("<I", rd(4))
        ne = struct.unpack("<%dQ" % nd, rd(8*nd))
        ty, = struct.unpack("<I", rd(4)); off, = struct.unpack("<Q", rd(8))
        offs[nm] = (list(ne), ty, off)
    data0 = (f.tell() + 31)//32*32
    return offs, data0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--bin", required=True)
    args = ap.parse_args()
    L = args.layer
    f = open(args.gguf, "r+b")
    offs, data0 = parse_off(f, L)
    DIMS = {"gate": (2048, 4096), "up": (2048, 4096), "down": (4096, 2048)}
    binf = open(args.bin, "rb")
    for kind in ("gate", "up", "down"):
        nm = f"blk.{L}.ffn_{kind}_exps.weight"
        ne, ty, off = offs[nm]
        assert ty == 41, f"{nm} 不是 go2b(type {ty}); mono 该层应已是 go2b"
        rows, cols = DIMS[kind]; nblk = cols // 256
        nbytes = rows * nblk * 68 * 256   # 256 experts
        data = binf.read(nbytes)
        assert len(data) == nbytes, f"{kind} bin 短读 {len(data)}!={nbytes}"
        f.seek(data0 + off); f.write(data)
    binf.close(); f.close()
    print(f"SPLICE-INPLACE-OK L{L} <- {os.path.basename(args.bin)}", flush=True)

if __name__ == "__main__":
    main()
