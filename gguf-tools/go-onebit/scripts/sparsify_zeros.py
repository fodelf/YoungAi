#!/usr/bin/env python3
"""sparsify_zeros.py — 把文件中的全零块变成 APFS 稀疏洞(dql 热槽位是洞, 网络拷贝会实体化)。
做法: 逐 1MiB 块读, 全零→seek 跳过(成洞), 非零→写; 末尾 truncate 保逻辑长度; os.replace 原子换。
用法: sparsify_zeros.py FILE...   (打印 每文件 回收字节)
"""
import os, sys

CH = 1 << 18   # 256KiB: 专家槽 1.0625MiB 与 1MiB 块不对齐, 粗粒度打不净; APFS 洞粒度 4K
Z = bytes(CH)

def sparsify(p):
    sz = os.path.getsize(p)
    tmp = p + ".sparse_tmp"
    saved = 0
    with open(p, "rb") as f, open(tmp, "wb") as g:
        while True:
            b = f.read(CH)
            if not b: break
            if b == Z[:len(b)]:
                g.seek(len(b), 1); saved += len(b)
            else:
                g.write(b)
        g.truncate(sz)
    if os.path.getsize(tmp) != sz:
        os.remove(tmp); print(f"{p}: 尺寸不符, 放弃"); return
    os.replace(tmp, p)
    print(f"{p}: 逻辑{sz>>20}MiB 洞{saved>>20}MiB", flush=True)

for p in sys.argv[1:]:
    sparsify(p)
