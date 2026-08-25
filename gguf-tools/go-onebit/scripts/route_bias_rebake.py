#!/usr/bin/env python3
# r36_rebake_bias.py — 让运行时路由 == 量化时路由(2026-07-31 实锤不一致):
#   合并骨架抄自冠军 ⇒ exp_probs_b 里是【冠军的 α·Δb】; 而量化链上用的是【R36 自己的 Δb×α】
#   (内存态, 不落文件) ⇒ 两者错配。本工具: 原位 bias −= 2.5·Δb_champ, += α·Δb_r36。
# 用法: r36_rebake_bias.py <model.gguf> <champ_rb.bin> <champ_alpha> <r36_rb.bin> <r36_alpha>
import struct, sys

NL, NEXP, RB_MINCNT = 43, 256, 8

def load_rb(path):
    with open(path, 'rb') as f:
        hd = struct.unpack('<4I', f.read(16))
        assert hd[0] == 0x41494252 and hd[1] == NL and hd[2] == NEXP, f'{path} RBIA 头不对'
        acc = struct.unpack(f'<{NL*NEXP}f', f.read(NL*NEXP*4))
        cnt = struct.unpack(f'<{NL*NEXP}I', f.read(NL*NEXP*4))
    return acc, cnt

path, cp, ca, rp, ra = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], float(sys.argv[5])
cacc, ccnt = load_rb(cp)
racc, rcnt = load_rb(rp)
# ★口径差: 冠军侧车存"累计和"(消费端 acc[i] 直接×α, 见 vq_merge_v4.load_route_bias);
#   R36 侧车由 rb_save 落盘的是 RB_APPLY(已 /cnt 的均值) ⇒ 不再除 cnt。
f = open(path, 'r+b')
f.read(8); n_t, n_kv = struct.unpack('<QQ', f.read(16))
def rs():
    n = struct.unpack('<Q', f.read(8))[0]; return f.read(n)
def sv(t):
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t in sz: f.seek(sz[t], 1); return
    if t == 8: rs(); return
    if t == 9:
        et, n = struct.unpack('<IQ', f.read(12))
        for _ in range(n): sv(et)
align = 32
for _ in range(n_kv):
    k = rs(); t = struct.unpack('<I', f.read(4))[0]
    if k == b'general.alignment': align = struct.unpack('<I', f.read(4))[0]
    else: sv(t)
tgt = {}
for _ in range(n_t):
    name = rs().decode()
    nd = struct.unpack('<I', f.read(4))[0]; f.seek(8*nd, 1)
    ty, off = struct.unpack('<IQ', f.read(12))
    if name.endswith('.exp_probs_b.bias'):
        tgt[int(name.split('.')[1])] = (off, ty)
data_start = (f.tell() + align - 1)//align*align
nsub = nadd = 0
for L, (off, ty) in sorted(tgt.items()):
    assert ty == 0, f'L{L} bias 非 f32'
    f.seek(data_start + off)
    vals = list(struct.unpack(f'<{NEXP}f', f.read(NEXP*4)))
    for e in range(NEXP):
        i = L*NEXP + e
        if ccnt[i] >= RB_MINCNT and cacc[i] != 0.0:
            vals[e] -= ca * cacc[i]; nsub += 1
        if racc[i] != 0.0:                       # RB_APPLY 已含 mincnt 门(不满门=0)
            vals[e] += ra * racc[i]; nadd += 1
    f.seek(data_start + off)
    f.write(struct.pack(f'<{NEXP}f', *vals))
f.close()
print(f'REBAKE 冠军偏置减 {nsub} 槽(α={ca}) / R36 偏置加 {nadd} 槽(α={ra}) → 运行时路由 == 量化时路由')
