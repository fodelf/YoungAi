#!/usr/bin/env python3
"""gen_active_from_anchor.py — 从 ds4quant FP 锚提取每层热专家表(2026-07-14, 方向A产线①)。

锚里存着整段校准语料的 FP 路由(ridx/rw, [NL][S][NACT]) — 免再跑 HF 前向。
按路由权重质量累加排序, 输出 emit_residual --active-experts 格式: "L{n}: e0 e1 ...".

用法: python3 gen_active_from_anchor.py ANCHOR.bin OUT.txt [topk=32]
"""
import struct
import sys

import numpy as np

anc, out = sys.argv[1], sys.argv[2]
topk = int(sys.argv[3]) if len(sys.argv) > 3 else 32

with open(anc, "rb") as f:
    hd = struct.unpack("<8I", f.read(32))
    f.read(8)  # idh u64
    magic, S, HCM, DIM, NL, VOCAB, NACT = hd[0], hd[1], hd[2], hd[3], hd[4], hd[5], hd[6]
    assert magic == 0x32415144, f"bad anchor magic {magic:#x}"
    f.seek(40 + NL * S * DIM * 4)  # skip fin
    ridx = np.frombuffer(f.read(NL * S * NACT * 4), dtype=np.int32).reshape(NL, S, NACT)
    rw = np.frombuffer(f.read(NL * S * NACT * 4), dtype=np.float32).reshape(NL, S, NACT)

lines = []
for L in range(NL):
    mass = np.zeros(256, dtype=np.float64)
    np.add.at(mass, ridx[L].reshape(-1), np.abs(rw[L].reshape(-1)))
    top = np.argsort(-mass)[:topk]
    cov = mass[top].sum() / max(mass.sum(), 1e-12)
    lines.append(f"L{L}: " + " ".join(str(int(e)) for e in top))
    print(f"L{L:02d} top{topk} 权重覆盖={cov*100:.1f}% 非零专家={int((mass>0).sum())}")

with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"[gen_active] {out}: {NL} 层 × top{topk} (S={S}, NACT={NACT})")
