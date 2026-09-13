#!/usr/bin/env python3
"""v41_head_tail_probe.py — 查 V4.1 GGUF 里 output.weight(fp4x32)尾部几行的源字节(2026-09-12 NaN 定罪夹具, 只读数)。
用法: v41_head_tail_probe.py <gguf> [起始行=129200]
打: 每行的 e8m0 scale 字节分布(max/是否有 0xFF)、解出的最大 |值|(超 f16 65504 就是 inf 的来源)。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v41_gguf_parity as P  # noqa: E402

FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)

kv, ts, data0, mm = P.read_gguf(sys.argv[1])
start = int(sys.argv[2]) if len(sys.argv) > 2 else 129200
ty, ne, off0 = ts["output.weight"]
print("output.weight: type", ty, "ne", ne, "off", off0)
D, V = 5120, 129280
nblk = D // 32
base = data0 + off0
for row in range(start, V):
    off = base + row * nblk * 17
    blkb = np.frombuffer(mm[off:off + nblk * 17], dtype=np.uint8).reshape(nblk, 17)
    sc = blkb[:, 16].astype(np.int32)
    scale = np.exp2(sc - 127.0)
    nib = blkb[:, :16]
    vals = np.concatenate([FP4[nib & 0x0F], FP4[nib >> 4]], axis=1) * scale[:, None]
    mx = np.abs(vals).max()
    flag = " ★scale=0xFF★" if (sc == 255).any() else (" ★>f16★" if mx > 65504 else "")
    print(f"行 {row}: scale max 0x{sc.max():02x} min 0x{sc.min():02x} | max|w| {mx:.4g}{flag}")
