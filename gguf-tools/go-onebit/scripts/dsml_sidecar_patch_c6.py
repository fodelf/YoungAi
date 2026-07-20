#!/usr/bin/env python3
"""dsml_sidecar_patch_c6.py — 就地把 corr 侧车的 corr_C 除以 n_expert_used(6)。

背景: kernel_dsv4_corr_apply 对每个选中专家各累加一次 U@(C[e]⊙Vx), zsolve v0
按"每 token 一次"求解并把同一 z 写满 256 行 → 运行时 6 倍过冲 (实测乱码)。
b/beta/delta 均为零, 只需 C/6。与用修复后 zsolve 重解完全等价 (线性缩放)。
zsolve.c 写出器已同步修复; 本脚本只为已产出的侧车做等价修补。

用法: python3 dsml_sidecar_patch_c6.py SIDECAR.gguf
"""
import struct
import sys

import numpy as np

N_USED = 6.0


def main():
    path = sys.argv[1]
    f = open(path, "r+b")
    magic, ver = struct.unpack("<II", f.read(8))
    assert magic == 0x46554747, "不是 GGUF"
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
    patched = 0
    for name, ne, dt, off in metas:
        if not name.endswith(".corr_C"):
            continue
        assert dt == 0, (name, dt)
        n = 1
        for e in ne: n *= e
        f.seek(base + off)
        a = np.frombuffer(f.read(n * 4), dtype="<f4") / N_USED
        f.seek(base + off)
        f.write(a.astype("<f4").tobytes())
        patched += 1
        print(f"{name}: /{int(N_USED)} 完成")
    f.close()
    print(f"patched {patched} tensors in {path}")


if __name__ == "__main__":
    main()
