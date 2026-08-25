#!/usr/bin/env python3
"""r30_blob_patch.py — 原位替换合并 GGUF 里的 blk.N.ffn_exps_vq.blob 段(2026-08-03 抢救)。
背景: student2 事故用 top64 热布局重写了 L00-L05 侧车头, merge 按旧账截断 → 前几层 blob 槽表越界。
blob 与反修无关(vq_keep 从不改载荷), 纯量化重跑的侧车与原版逐字节等价 → 同尺寸原位覆盖即恢复。
用法: r30_blob_patch.py <merged.gguf> <layers_dir> <L0> <L1>   # 闭区间
"""
import struct, sys, os

gguf, ldir, l0, l1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
f = open(gguf, "r+b")
f.read(8); nt, = struct.unpack("<q", f.read(8)); nkv, = struct.unpack("<q", f.read(8))
def rstr():
    n, = struct.unpack("<q", f.read(8)); return f.read(n).decode(errors="replace")
align = 32
def rv(t):
    if t in (0,1,7): f.read(1); return None
    if t in (2,3): f.read(2); return None
    if t in (4,5): return struct.unpack("<i", f.read(4))[0]
    if t == 6: f.read(4); return None
    if t in (10,11,12): f.read(8); return None
    if t == 8: return rstr()
    if t == 9:
        et, = struct.unpack("<i", f.read(4)); n, = struct.unpack("<q", f.read(8))
        for _ in range(n): rv(et)
        return None
for _ in range(nkv):
    k = rstr(); t, = struct.unpack("<i", f.read(4)); v = rv(t)
    if k == "general.alignment" and v: align = v
tgt = {}
for _ in range(nt):
    nm = rstr(); nd, = struct.unpack("<i", f.read(4))
    dims = struct.unpack("<%dq" % nd, f.read(8*nd))
    ty, = struct.unpack("<i", f.read(4)); off, = struct.unpack("<q", f.read(8))
    tgt[nm] = (dims[0], off)
data0 = (f.tell() + align - 1) // align * align
for L in range(l0, l1+1):
    nm = f"blk.{L}.ffn_exps_vq.blob"
    blen, off = tgt[nm]
    src = os.path.join(ldir, f"dql_vq_L{L:02d}.bin")
    sb = open(src, "rb").read()
    assert len(sb) == blen, f"L{L} 尺寸不符 {len(sb)} vs {blen}"
    f.seek(data0 + off); f.write(sb)
    # 校验: 重读槽表最大偏移必须 < blob 长
    f.seek(data0 + off + 16); tab = struct.unpack("<768Q", f.read(768*8))
    mx = max(tab)
    print(f"L{L:02d} 替换 ✓ {blen} B, 槽表max={mx} {'OK' if mx < blen else '★仍越界★'}")
f.close()
print("done")
