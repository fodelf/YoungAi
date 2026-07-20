#!/usr/bin/env python3
"""dsml_sidecar_ablate.py — 生成层消融侧车变体 (保留 keep 列表, 其余层校正置零)。

置零 corr_C ⇒ 该层校正 = Σ U@(0⊙Vx) + n·b(=0) + Σbeta(=0) = 精确 no-op,
不改张量结构 (corr_load/present 判定不变), 挂载路径零改动。
用途: 定位"侧车整体挂载→乱码"是单层失稳(如 L20 大幅校正)还是运行时应用 bug。

用法: python3 dsml_sidecar_ablate.py IN.gguf OUT.gguf 30[,29,…]
"""
import shutil
import struct
import sys

import numpy as np


def main():
    src, dst, keep = sys.argv[1], sys.argv[2], {int(x) for x in sys.argv[3].split(",")}
    shutil.copyfile(src, dst)
    f = open(dst, "r+b")
    struct.unpack("<II", f.read(8))
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

    def rstr():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode()

    for _ in range(n_kv):
        rstr()
        t, = struct.unpack("<I", f.read(4))
        if t == 8: rstr()
        elif t == 7: f.read(1)
        elif t in (4, 5, 10, 11): f.read(8)
        else: f.read(4)
    metas = []
    for _ in range(n_tensors):
        name = rstr()
        nd, = struct.unpack("<I", f.read(4))
        ne = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        dt, off = struct.unpack("<IQ", f.read(12))
        metas.append((name, ne, dt, off))
    base = (f.tell() + 31) // 32 * 32
    for name, ne, dt, off in metas:
        if not name.endswith(".corr_C"):
            continue
        L = int(name.split(".")[1])
        if L in keep:
            print(f"{name}: 保留")
            continue
        n = 1
        for e in ne: n *= e
        f.seek(base + off)
        f.write(b"\x00" * (n * 4))
        print(f"{name}: 置零")
    f.close()


if __name__ == "__main__":
    main()
