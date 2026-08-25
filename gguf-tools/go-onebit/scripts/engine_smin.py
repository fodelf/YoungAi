#!/usr/bin/env python3
# engine_smin.py — 引擎口径还原率终审(2026-07-31 用户令"看看是哪里导致输出不正确"):
#   引擎 DS4_EVAL_IDS 吐的 logits vs 锚内量化器 FP logits → held 区 Σmin/KL。
#   对账口径 = 量化器终值 0.6875/1.0672(同 ids 同 held 划分)。
# 用法: engine_smin.py <engine_logits.bin> <anchor.bin> [nfit=933]
import struct, sys
import numpy as np

eng_p, anc_p = sys.argv[1], sys.argv[2]
NFIT = int(sys.argv[3]) if len(sys.argv) > 3 else 933

f = open(anc_p, 'rb')
hd = struct.unpack('<8I', f.read(32))
S, HCM, DIM, NL, VOCAB, NACT = hd[1], hd[2], hd[3], hd[4], hd[5], hd[6]
f.read(8)
fin_b = NL*S*DIM*4; ridx_b = NL*S*NACT*4
H_b = NL*S*HCM*DIM*4
f.seek(32+8+fin_b+2*ridx_b+H_b)
fp = np.frombuffer(f.read(S*VOCAB*4), dtype=np.float32).reshape(S, VOCAB)

eng = np.fromfile(eng_p, dtype=np.float32)
assert eng.size == S*VOCAB, f'引擎 logits 尺寸 {eng.size} != {S*VOCAB}'
eng = eng.reshape(S, VOCAB)

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=-1, keepdims=True)

sm_tot = kl_tot = 0.0; n = 0
for s in range(NFIT, S-1):
    p = softmax(fp[s]); q = softmax(eng[s])
    sm_tot += np.minimum(p, q).sum()
    kl_tot += float((p * (np.log(p+1e-12) - np.log(q+1e-12))).sum())
    n += 1
print(f'held 行={n}  ★引擎口径 Σmin={sm_tot/n:.4f}  KL={kl_tot/n:.4f}★  (量化器口径对账值: 0.6875/1.0672)')
