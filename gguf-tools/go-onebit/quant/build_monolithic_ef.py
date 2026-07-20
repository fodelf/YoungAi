#!/usr/bin/env python3
"""build_monolithic_ef.py — 单块混合模型组装 (go2b 层取自逐层 EF .bin, 非侧车 GGUF)。
v3(go1b全层) 的 overlay 层 expert 张量 → 换成 EFDIR/go2b_ef_L{L}.bin (68B块, kind-major expert-major)。
顺序写 stdout (管道给目标机避免共存)。overlay 层集合 = EFDIR 里有 bin 的层。
用法: build_monolithic_ef.py V3.gguf EFDIR > mono.gguf   (或 | ssh M1 'cat>...')
"""
import sys, os, struct, glob
import numpy as np

def parse(path):
    f = open(path, "rb")
    def rd(n):
        b = f.read(n); assert len(b) == n; return b
    magic, ver, n_t, n_kv = struct.unpack("<IIQQ", rd(24)); assert magic == 0x46554747
    def rstr(): n, = struct.unpack("<Q", rd(8)); return rd(n).decode()
    kv = []
    def rdval(t):
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8: n, = struct.unpack("<Q", rd(8)); return ("s", rd(n))
        if t == 9:
            et, = struct.unpack("<I", rd(4)); n, = struct.unpack("<Q", rd(8))
            return ("a", et, n, [rdval(et) for _ in range(n)])
        return ("r", rd(sz[t]))
    for _ in range(n_kv):
        k = rstr(); t, = struct.unpack("<I", rd(4)); kv.append((k, t, rdval(t)))
    tens = []
    for _ in range(n_t):
        nm = rstr(); nd, = struct.unpack("<I", rd(4))
        ne = struct.unpack("<%dQ" % nd, rd(8*nd))
        ty, = struct.unpack("<I", rd(4)); off, = struct.unpack("<Q", rd(8))
        tens.append([nm, list(ne), ty, off])
    data0 = (f.tell() + 31)//32*32
    return f, kv, tens, data0

def tsize(ne, ty):
    n = 1
    for d in ne: n *= d
    if ty == 41: return ne[1]*(ne[0]//256)*68*ne[2]
    if ty == 40: return ne[1]*(ne[0]//256)*34*ne[2]
    if ty == 0:  return n*4
    if ty == 1:  return n*2
    if ty == 26: return n*4
    if ty == 8:  return (n//32)*34
    raise SystemExit(f"unknown type {ty}")

def main():
    v3p, efdir = sys.argv[1], sys.argv[2]
    OV = sorted(int(os.path.basename(p).split("_L")[1].split(".")[0])
                for p in glob.glob(f"{efdir}/go2b_ef_L*.bin"))
    print(f"overlay layers from EF bins: {OV}", file=sys.stderr, flush=True)
    vf, vkv, vtens, vdata0 = parse(v3p)
    out_t = []
    for nm, ne, ty, off in vtens:
        parts = nm.split(".")
        is_ov = (parts[0] == "blk" and "_exps.weight" in nm and "ffn_" in nm and
                 int(parts[1]) in OV and any(k in nm for k in ("gate","up","down")))
        if is_ov:
            L = int(parts[1]); kind = "gate" if "gate" in nm else ("up" if "up" in nm else "down")
            binp = f"{efdir}/go2b_ef_L{L}.bin"
            # go2b bin: kind-major expert-major. 该 kind 在 bin 内的字节偏移
            r, c = (2048, 4096) if kind in ("gate", "up") else (4096, 2048)
            nb = r*(c//256)*68*256
            koff = {"gate": 0, "up": nb, "down": 2*nb}[kind] if kind != "down" else (2048*(4096//256)*68*256)*2
            # 精确: gate,up 同尺寸, down 不同 → 累加
            szg = 2048*(4096//256)*68*256
            koff = {"gate": 0, "up": szg, "down": 2*szg}[kind]
            out_t.append((nm, ne, 41, binp, koff, nb))
        else:
            out_t.append((nm, ne, ty, "v3", vdata0+off, tsize(ne, ty)))
    o = sys.stdout.buffer
    def w(b): o.write(b)
    def wstr(s): b = s.encode(); w(struct.pack("<Q", len(b))); w(b)
    w(struct.pack("<IIQQ", 0x46554747, 3, len(out_t), len(vkv)))
    for k, t, v in vkv:
        wstr(k); w(struct.pack("<I", t))
        if v[0] == "s": w(struct.pack("<Q", len(v[1]))); w(v[1])
        elif v[0] == "a":
            _, et, n, ps = v; w(struct.pack("<I", et)); w(struct.pack("<Q", n))
            for p in ps: (w(struct.pack("<Q", len(p[1]))), w(p[1])) if p[0]=="s" else w(p[1])
        else: w(v[1])
    align = 32; roff = 0
    for nm, ne, ty, src, so, nb in out_t:
        wstr(nm); w(struct.pack("<I", len(ne)))
        for d in ne: w(struct.pack("<Q", d))
        w(struct.pack("<IQ", ty, roff)); roff += (nb+align-1)//align*align
    written = 24
    for k, t, v in vkv:
        written += 8+len(k.encode())+4
        written += (8+len(v[1])) if v[0]=="s" else (4+8+sum((8+len(p[1])) if p[0]=="s" else len(p[1]) for p in v[3]) if v[0]=="a" else len(v[1]))
    for nm, ne, ty, src, so, nb in out_t:
        written += 8+len(nm.encode())+4+8*len(ne)+4+8
    data0 = (written+align-1)//align*align; w(b"\0"*(data0-written))
    pos = 0
    for nm, ne, ty, src, so, nb in out_t:
        f = vf if src == "v3" else open(src, "rb")
        f.seek(so); remain = nb
        while remain > 0:
            c = f.read(min(1<<24, remain)); w(c); remain -= len(c)
        if src != "v3": f.close()
        pos += nb; pad = (align-pos%align)%align; w(b"\0"*pad); pos += pad
    o.flush()
    print(f"MONO-EF-DONE tensors={len(out_t)} overlay={OV} bytes={data0+pos}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
