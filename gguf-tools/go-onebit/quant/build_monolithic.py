#!/usr/bin/env python3
"""build_monolithic.py — 把 v3(go1b全层) + go2b侧车(16最差层) 合成单块混合模型, 顺序写 stdout。
overlay 层的 expert 张量 go1b→go2b(type41, 取自侧车), 其余张量原样拷 v3。丢侧车的 lut(单块里 expert 即 go2b)。
消 overlay 层的 go1b 死重 → ~59GB。用法: build_monolithic.py V3.gguf SIDECAR.gguf > mono.gguf (或 | ssh M1 'cat>...')
"""
import sys, struct
import numpy as np

def parse(path):
    f = open(path, "rb")
    def rd(n):
        b = f.read(n); assert len(b) == n; return b
    magic, ver, n_t, n_kv = struct.unpack("<IIQQ", rd(24))
    assert magic == 0x46554747
    def rstr(): n, = struct.unpack("<Q", rd(8)); return rd(n).decode()
    kv_raw = []
    def rdval(t):
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8:
            n, = struct.unpack("<Q", rd(8)); return ("s", rd(n))
        if t == 9:
            et, = struct.unpack("<I", rd(4)); n, = struct.unpack("<Q", rd(8))
            parts = [rdval(et) for _ in range(n)]
            return ("a", et, n, parts)
        return ("r", rd(sz[t]))
    for _ in range(n_kv):
        k = rstr(); t, = struct.unpack("<I", rd(4)); kv_raw.append((k, t, rdval(t)))
    tens = []
    for _ in range(n_t):
        nm = rstr(); nd, = struct.unpack("<I", rd(4))
        ne = struct.unpack("<%dQ" % nd, rd(8*nd))
        ty, = struct.unpack("<I", rd(4)); off, = struct.unpack("<Q", rd(8))
        tens.append([nm, list(ne), ty, off])
    data0 = (f.tell() + 31)//32*32
    return f, kv_raw, tens, data0

def tsize(ne, ty):
    n = 1
    for d in ne: n *= d
    if ty == 41: return ne[1]*(ne[0]//256)*68*ne[2]      # go2b
    if ty == 40: return ne[1]*(ne[0]//256)*34*ne[2]      # go1b
    if ty == 0:  return n * 4                             # F32
    if ty == 1:  return n * 2                             # F16
    if ty == 26: return n * 4                             # I32 (tid2eid)
    if ty == 8:  return (n // 32) * 34                    # Q8_0
    raise SystemExit(f"unknown backbone type {ty} ne={ne}")

def main():
    v3p, scp = sys.argv[1], sys.argv[2]
    vf, vkv, vtens, vdata0 = parse(v3p)
    sf, skv, stens, sdata0 = parse(scp)
    sidx = {t[0]: t for t in stens}
    # overlay 层集合
    OV = sorted({int(nm.split(".")[1]) for nm, _, _, _ in stens if "_exps_res" in nm})
    # v3 张量原字节大小(相邻 offset 差, 末个到文件尾)
    vsz = {}
    order = sorted(vtens, key=lambda t: t[3])
    import os as _os
    vend = (_os.path.getsize(v3p))
    for i, t in enumerate(order):
        nxt = order[i+1][3] if i+1 < len(order) else (vend - vdata0)
        vsz[t[0]] = nxt - t[3]                            # 含对齐 pad, 拷贝时用真实 tsize 截断
    # 输出张量表 = v3 顺序, overlay 层 expert 张量替换成 go2b
    out_t = []                                            # (name, ne, ty, src, src_off, nbytes)
    for nm, ne, ty, off in vtens:
        is_ov_exp = ("_exps.weight" in nm and "ffn_" in nm and
                     nm.split(".")[0] == "blk" and int(nm.split(".")[1]) in OV and
                     any(k in nm for k in ("gate", "up", "down")))
        if is_ov_exp:
            L = int(nm.split(".")[1]); kind = "gate" if "gate" in nm else ("up" if "up" in nm else "down")
            st = sidx[f"blk.{L}.ffn_{kind}_exps_res.weight"]
            nb = tsize(st[1], 41)
            out_t.append((nm, st[1], 41, "sc", sdata0 + st[3], nb))
        else:
            nb = tsize(ne, ty)
            out_t.append((nm, ne, ty, "v3", vdata0 + off, nb))

    o = sys.stdout.buffer
    def w(b): o.write(b)
    def wstr(s): b = s.encode(); w(struct.pack("<Q", len(b))); w(b)
    w(struct.pack("<IIQQ", 0x46554747, 3, len(out_t), len(vkv)))
    for k, t, v in vkv:                                   # KV 原样(v3 的, 含 arch/tokenizer)
        wstr(k); w(struct.pack("<I", t))
        if v[0] == "s": w(struct.pack("<Q", len(v[1]))); w(v[1])
        elif v[0] == "a":
            _, et, n, parts = v; w(struct.pack("<I", et)); w(struct.pack("<Q", n))
            for p in parts:
                if p[0] == "s": w(struct.pack("<Q", len(p[1]))); w(p[1])
                else: w(p[1])
        else: w(v[1])
    align = 32; roff = 0
    for nm, ne, ty, src, so, nb in out_t:
        wstr(nm); w(struct.pack("<I", len(ne)))
        for d in ne: w(struct.pack("<Q", d))
        w(struct.pack("<IQ", ty, roff)); roff += (nb + align - 1)//align*align
    # stdout 是管道不可 seek → 手动数已写字节求 data0
    print("HEADER-WRITTEN", file=sys.stderr, flush=True)
    written = 24
    for k, t, v in vkv:
        written += 8 + len(k.encode()) + 4
        if v[0]=="s": written += 8 + len(v[1])
        elif v[0]=="a":
            written += 4 + 8
            for p in v[3]:
                written += (8+len(p[1])) if p[0]=="s" else len(p[1])
        else: written += len(v[1])
    for nm, ne, ty, src, so, nb in out_t:
        written += 8 + len(nm.encode()) + 4 + 8*len(ne) + 4 + 8
    data0 = (written + align - 1)//align*align
    w(b"\0" * (data0 - written))
    pos = 0
    for nm, ne, ty, src, so, nb in out_t:
        f = vf if src == "v3" else sf
        f.seek(so); remain = nb
        while remain > 0:
            chunk = f.read(min(1<<24, remain)); w(chunk); remain -= len(chunk)
        pos += nb; pad = (align - pos % align) % align; w(b"\0"*pad); pos += pad
    o.flush()
    print(f"MONO-DONE tensors={len(out_t)} overlay_layers={OV} bytes={data0+pos}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
