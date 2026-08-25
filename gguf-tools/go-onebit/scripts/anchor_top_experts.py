#!/usr/bin/env python3
# anchor_top_experts.py — 从 R30 锚(DQA2)统计每层专家激活频次(按路由权重加权), 产 top-N 热表。
# 用法: anchor_top_experts.py <anchor.bin> <N> <out.txt>
# 表格式与 prog_active_top64.txt 同款: "L%d: id id ..."(按权重降序 = 排名序)。
# 2026-08-04 r60 战役: 60G 全口径贴满 → HOT=88, 表必须出自 0731 源+v5mini 语料的真实路由(锚)。
import struct, sys
import numpy as np

ap, N, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
f = open(ap, 'rb')
hd = struct.unpack('<8I', f.read(32))
assert hd[0] == 0x32415144, 'bad magic'
S, HCM, DIM, NL, VOCAB, NACT = hd[1], hd[2], hd[3], hd[4], hd[5], hd[6]
f.read(8)  # idh
f.seek(NL * S * DIM * 4, 1)                      # skip fin
ridx = np.frombuffer(f.read(NL * S * NACT * 4), dtype=np.int32).reshape(NL, S, NACT)
rw   = np.frombuffer(f.read(NL * S * NACT * 4), dtype=np.float32).reshape(NL, S, NACT)
with open(outp, 'w') as o:
    for L in range(NL):
        w = np.zeros(256, dtype=np.float64)
        ids = ridx[L].ravel(); ws = rw[L].ravel()
        ok = (ids >= 0) & (ids < 256)
        np.add.at(w, ids[ok], ws[ok])
        top = np.argsort(-w)[:N]
        o.write(f"L{L}: " + " ".join(str(int(e)) for e in top) + "\n")
        if L == 0:
            cov = w[top].sum() / max(w.sum(), 1e-9)
            print(f"L0 top{N} 权重覆盖率={cov*100:.1f}%  S={S} NACT={NACT} NL={NL}", file=sys.stderr)
print(f"top{N} 表 → {outp}", file=sys.stderr)
