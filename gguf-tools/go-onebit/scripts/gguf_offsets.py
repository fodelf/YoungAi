#!/usr/bin/env python3
"""gguf_offsets.py — 解析 GGUF v3 头, 输出每个 tensor 的绝对数据偏移表(splice/直写用)。
用法: python3 gguf_offsets.py model.gguf > offsets.txt
每行: name type_id abs_offset n_elements
(go1b 块类型 type=40: nbytes = n_elements/256*34, 由消费端按需计算)"""
import sys, struct

def rd(f, fmt):
    sz = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(sz))

def rstr(f):
    (n,) = rd(f, "<Q")
    return f.read(n).decode("utf-8", "replace")

def skip_kv_value(f, t):
    scal = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t in scal: f.seek(scal[t], 1)
    elif t == 8: rstr(f)                      # string
    elif t == 9:                              # array
        (et,) = rd(f, "<i"); (n,) = rd(f, "<Q")
        if et == 8:
            for _ in range(n): rstr(f)
        elif et == 9:
            for _ in range(n): skip_kv_value(f, 9)
        else:
            f.seek(scal[et] * n, 1)
    else: raise SystemExit(f"unknown kv type {t}")

def main(path):
    f = open(path, "rb")
    magic = f.read(4)
    assert magic == b"GGUF", f"not GGUF: {magic!r}"
    (ver,) = rd(f, "<I")
    (n_tensors,) = rd(f, "<q")
    (n_kv,) = rd(f, "<q")
    align = 32
    for _ in range(n_kv):
        k = rstr(f); (t,) = rd(f, "<i")
        if k == "general.alignment":
            if t == 4: (align,) = rd(f, "<I")
            elif t == 5: (align,) = rd(f, "<i")
            else: skip_kv_value(f, t)
        else:
            skip_kv_value(f, t)
    infos = []
    for _ in range(n_tensors):
        name = rstr(f)
        (nd,) = rd(f, "<I")
        dims = [rd(f, "<Q")[0] for _ in range(nd)]
        (ty,) = rd(f, "<i")
        (off,) = rd(f, "<Q")
        nel = 1
        for d in dims: nel *= d
        infos.append((name, ty, off, nel))
    data_start = (f.tell() + align - 1) // align * align
    for name, ty, off, nel in infos:
        print(f"{name} {ty} {data_start + off} {nel}")

if __name__ == "__main__":
    main(sys.argv[1])
