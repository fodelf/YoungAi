#!/usr/bin/env python3
"""gen_pinned.py — 从 FP 锚生成 DS4_METAL_EXPERT_POOL_PINNED 白名单(2026-07-14 速度)。

活跃专家进 RAM = 消除 decode 的 SSD fault。用户物理洞察: top-k 活跃集塞得进 24G。
按机器层范围切分(coordinator/worker 各自只钉自己持有的层), 紧凑格式(逗号分专家、
分号分层、无空格)便于 env 传递。

用法: gen_pinned.py ANCHOR.bin OUT_PREFIX [topk=64] [split=20]
  → OUT_PREFIX.coord.pinned (层 0..split-1) + OUT_PREFIX.worker.pinned (层 split..NL-1)
  + stderr 每机字节估算(× 均值 MB/专家, 供池尺寸/内存红线核对)
"""
import struct
import sys

import numpy as np

anc, pref = sys.argv[1], sys.argv[2]
topk = int(sys.argv[3]) if len(sys.argv) > 3 else 64
split = int(sys.argv[4]) if len(sys.argv) > 4 else 20

with open(anc, "rb") as f:
    hd = struct.unpack("<8I", f.read(32)); f.read(8)
    magic, S, HCM, DIM, NL, VOCAB, NACT = hd[:7]
    assert magic == 0x32415144, f"bad anchor magic {magic:#x}"
    f.seek(40 + NL * S * DIM * 4)
    ridx = np.frombuffer(f.read(NL * S * NACT * 4), dtype=np.int32).reshape(NL, S, NACT)
    rw = np.frombuffer(f.read(NL * S * NACT * 4), dtype=np.float32).reshape(NL, S, NACT)

def layer_top(L):
    mass = np.zeros(256, dtype=np.float64)
    np.add.at(mass, ridx[L].reshape(-1), np.abs(rw[L].reshape(-1)))
    return np.argsort(-mass)[:topk], mass

MB_PER_EXPERT = 3.11  # 实测均值(15.94G/20层/256)
for tag, lo, hi in (("coord", 0, split), ("worker", split, NL)):
    lines = []
    n_exp = 0
    for L in range(lo, hi):
        top, _ = layer_top(L)
        lines.append(f"L{L}:" + ",".join(str(int(e)) for e in top))
        n_exp += len(top)
    out = f"{pref}.{tag}.pinned"
    with open(out, "w") as fo:
        fo.write(";".join(lines) + "\n")
    est_gb = n_exp * MB_PER_EXPERT / 1024.0
    print(f"[gen_pinned] {out}: 层 {lo}..{hi-1} × top{topk} = {n_exp} 专家 ≈ {est_gb:.2f} GiB 池", file=sys.stderr)
