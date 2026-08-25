#!/usr/bin/env python3
# cap_to_anchor.py — 引擎轨迹捕获(DS4_CAP_DIR raw 分片) → DQA2 锚(量化链态锚)。
# 口径对齐战役(2026-08-19): 放大器解算的 x/路由必须与引擎在线一致(量化链态),
# 引擎侧零新代码 — DS4_CAP_DIR 已抓 raw_ffn_in(x̂=ffn_norm后)/raw_route(pre-remap id)/
# raw_route_w(门权重), 本脚本转成 anchor_layer() 可读的 DQA2 前三段(fin/ridx/rw)。
# H/logits 段不写(zlever 链只 seek 前三段; ds4quant_run 的 anchor_load 不吃此锚)。
# 用法: cap_to_anchor.py <cap_dir> <ids文件> <out_anchor.bin> [NL=43]
import os, struct, sys
import numpy as np

cap, idsp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
NL = int(sys.argv[4]) if len(sys.argv) > 4 else 43
DIM, NACT, HCM, VOCAB = 4096, 6, 4, 129280

ids = [int(t) for t in open(idsp).read().split()]
S = len(ids)

def idh_fnv(ids):
    h = 1469598103934665603
    for v in ids:
        for b in range(8):
            h ^= (v >> (8 * b)) & 0xFF
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

fin = np.zeros((NL, S, DIM), dtype=np.float32)
ridx = np.zeros((NL, S, NACT), dtype=np.int32)
rw = np.zeros((NL, S, NACT), dtype=np.float32)
for L in range(NL):
    x = np.fromfile(os.path.join(cap, f"raw_ffn_in_L{L}"), dtype=np.float16)
    r = np.fromfile(os.path.join(cap, f"raw_route_L{L}"), dtype=np.int16)
    w = np.fromfile(os.path.join(cap, f"raw_route_w_L{L}"), dtype=np.float16)
    n = x.size // DIM
    assert n == S and r.size // NACT == S and w.size // NACT == S, \
        f"L{L}: 行数 {n}/{r.size//NACT}/{w.size//NACT} != ids {S} (捕获目录未清空/跑串?)"
    fin[L] = x.reshape(S, DIM).astype(np.float32)
    ridx[L] = r.reshape(S, NACT).astype(np.int32)
    rw[L] = w.reshape(S, NACT).astype(np.float32)

with open(outp, "wb") as f:
    f.write(struct.pack("<8I", 0x32415144, S, HCM, DIM, NL, VOCAB, NACT, 0))
    f.write(struct.pack("<Q", idh_fnv(ids)))
    f.write(fin.tobytes()); f.write(ridx.tobytes()); f.write(rw.tobytes())
print(f"→ {outp} S={S} NL={NL} ({(40+fin.nbytes+ridx.nbytes+rw.nbytes)/2**30:.2f} GiB) "
      f"链态锚: fin/ridx/rw = 引擎在线口径", flush=True)
