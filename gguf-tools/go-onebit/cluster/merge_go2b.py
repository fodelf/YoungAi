#!/usr/bin/env python3
# merge_go2b.py — R5-C C1: 把 v2 底座(go1b, 每专家 ±s1) 与热残差侧车(go1b, ±s2)
# 离线合并为 go2b 热侧车（每 256-block: [f16 s1][f16 s2][32B base signs][32B res signs]
# = 68B, 2.125bpw, 数学上与 base+residual 两遍相加逐位等价）。
# 张量名/LUT/KV 沿用 _res 侧车约定, 仅类型 40→41: 运行时 residual_load 零改动,
# metal 按类型分支走"热/冷两源拆分"路。
# 用法: merge_go2b.py BASE_V2.gguf RES_SIDECAR.gguf OUT_GO2B.gguf
import sys, struct, mmap
import numpy as np

GO1B, GO2B, F32 = 40, 41, 0
QK = 256

def parse(f):
    def rd(n):
        b = f.read(n); assert len(b) == n; return b
    magic, ver, n_t, n_kv = struct.unpack("<IIQQ", rd(24))
    assert magic == 0x46554747
    def rstr():
        n, = struct.unpack("<Q", rd(8)); return rd(n).decode()
    kvs = []
    def rdval(t):
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8: return rstr()
        if t == 9:
            et, = struct.unpack("<I", rd(4)); n, = struct.unpack("<Q", rd(8))
            return [rdval(et) for _ in range(n)]
        raw = rd(sz[t])
        return raw
    for _ in range(n_kv):
        k = rstr(); t, = struct.unpack("<I", rd(4)); kvs.append((k, t, rdval(t)))
    tens = []
    for _ in range(n_t):
        nm = rstr(); nd, = struct.unpack("<I", rd(4))
        ne = struct.unpack("<%dQ" % nd, rd(8*nd))
        ty, = struct.unpack("<I", rd(4)); off, = struct.unpack("<Q", rd(8))
        tens.append([nm, list(ne), ty, off])
    data0 = (f.tell() + 31) // 32 * 32
    return kvs, tens, data0

base_p, res_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
bf, rf = open(base_p, "rb"), open(res_p, "rb")
bkv, btens, bdata0 = parse(bf)
rkv, rtens, rdata0 = parse(rf)
bmap = mmap.mmap(bf.fileno(), 0, prot=mmap.PROT_READ)
rmap = mmap.mmap(rf.fileno(), 0, prot=mmap.PROT_READ)
bidx = {t[0]: t for t in btens}

def g1_bytes(ne):  # go1b 张量字节: 每行 cols/256 块 × 34B
    return ne[0] // QK * 34 * ne[1] * (ne[2] if len(ne) > 2 else 1)

# 输出张量表: _res 张量 40→41(块 68B), LUT F32 原样, 逐块交错重排
out_tens = []   # (name, ne, type, src_kind)
for nm, ne, ty, off in rtens:
    if ty == GO1B and "_exps_res" in nm:
        out_tens.append((nm, ne, GO2B, "merge", off))
    elif "_res_lut" in nm:
        out_tens.append((nm, ne, F32, "copy", off))
    else:
        out_tens.append((nm, ne, ty, "copy", off))

def go2b_sz(ne): return ne[0] // QK * 68 * ne[1] * ne[2]
def f32_sz(ne):
    n = 1
    for x in ne: n *= x
    return n * 4

# ---- 写输出 ----
out = open(out_p, "wb")
def w(b): out.write(b)
def wstr(s):
    b = s.encode(); w(struct.pack("<Q", len(b))); w(b)
w(struct.pack("<IIQQ", 0x46554747, 3, len(out_tens), len(rkv)))
for k, t, v in rkv:   # KV 原样复制(present/sparse/n_present/layer.i/arch)
    wstr(k); w(struct.pack("<I", t))
    if t == 8: wstr(v)
    else: w(v)
align = 32
roff = 0
sizes = []
for nm, ne, ty, kind, soff in out_tens:
    sz = go2b_sz(ne) if ty == GO2B else (f32_sz(ne) if ty == F32 else g1_bytes(ne))
    sizes.append(sz)
    wstr(nm); w(struct.pack("<I", len(ne)))
    for d in ne: w(struct.pack("<Q", d))
    w(struct.pack("<IQ", ty, roff))
    roff += (sz + align - 1) // align * align
hdr = out.tell()
data0 = (hdr + align - 1) // align * align
w(b"\0" * (data0 - hdr))

nmerged = 0
for (nm, ne, ty, kind, soff), sz in zip(out_tens, sizes):
    pos = out.tell()
    if kind == "copy":
        out.write(rmap[rdata0 + soff: rdata0 + soff + sz])
    else:
        # 逐专家合并: res 槽 s ↔ base 专家 e (LUT 反查)
        L = int(nm.split(".")[1])
        lut_t = next(t for t in rtens if t[0] == f"blk.{L}.ffn_res_lut.weight")
        lut = struct.unpack("<256f", rmap[rdata0 + lut_t[3]: rdata0 + lut_t[3] + 1024])
        slot2e = {}
        for e, s in enumerate(lut):
            if s >= 0: slot2e[int(s)] = e
        kind3 = "gate" if "_gate_" in nm else ("up" if "_up_" in nm else "down")
        bt = bidx[f"blk.{L}.ffn_{kind3}_exps.weight"]
        cols, rows, K = ne[0], ne[1], ne[2]
        bpr = cols // QK                      # 块/行
        b_row = bpr * 34
        b_exp = rows * b_row                  # base 每专家字节
        r_exp = b_exp                         # res 同为 go1b
        nblk = rows * bpr
        for s in range(K):
            e = slot2e[s]
            bo = bdata0 + bt[3] + e * b_exp
            ro = rdata0 + soff + s * r_exp
            bb = np.frombuffer(bmap, dtype=np.uint8, count=b_exp, offset=bo).reshape(nblk, 34)
            rb = np.frombuffer(rmap, dtype=np.uint8, count=r_exp, offset=ro).reshape(nblk, 34)
            eb = np.empty((nblk, 68), dtype=np.uint8)
            eb[:, 0:2]   = bb[:, 0:2]     # d1
            eb[:, 2:4]   = rb[:, 0:2]     # d2
            eb[:, 4:36]  = bb[:, 2:34]    # base signs
            eb[:, 36:68] = rb[:, 2:34]    # res signs
            out.write(eb.tobytes())
        nmerged += 1
        print(f"{nm}: merged K={K} ({sz>>20} MiB)", file=sys.stderr, flush=True)
    pad = (align - (out.tell() - data0) % align) % align
    w(b"\0" * pad)
print(f"wrote {out_p}: {nmerged} merged tensors, {out.tell()>>20} MiB total", file=sys.stderr)
