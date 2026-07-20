#!/usr/bin/env python3
"""dsml_sidecar_scale.py — 就地把 corr 侧车整份校正乘一个标量 (幅度诊断/调优)。

校正 = U@(C[e]⊙Vx) + b + beta; 当前 b/beta/delta 全零, 故缩放 corr_C 即等价
缩放整份校正 (线性)。用于测"乱码是纯幅度问题还是应用根上错": s<1 若恢复连贯
=幅度可调, 否则=根上问题。

用法: python3 dsml_sidecar_scale.py IN.gguf OUT.gguf 0.25
"""
import shutil
import struct
import sys

import numpy as np


def main():
    src, dst, s = sys.argv[1], sys.argv[2], float(sys.argv[3])
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
        n = 1
        for e in ne: n *= e
        f.seek(base + off)
        a = np.frombuffer(f.read(n * 4), dtype="<f4") * s
        f.seek(base + off)
        f.write(a.astype("<f4").tobytes())
        print(f"{name}: ×{s}")
    f.close()
    print(f"scaled all corr_C by {s} in {dst}")


if __name__ == "__main__":
    main()
