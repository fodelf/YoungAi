"""zlever/zrec_to_zchain.py — zrec_LXX.bin(zlayer INJ=2 外挂记录)→ 引擎 DQZ2 zchain。
zrec 内容 = 116B 头记录串(bf.GE/zl.RRR), 载荷逐字节直通; 空文件=该层被闸(skip)。
用法: zrec_to_zchain.py <layers_dir> <out.zchain.bin> [nl=43]
"""
import os, sys, struct
ld, outp = sys.argv[1], sys.argv[2]
NL = int(sys.argv[3]) if len(sys.argv) > 3 else 43
# ZC_SKIP="5,7": 合并时按类型过滤(5=GE 6=z^L 7=AMP 8=RTE 9=AMPD动态z), 组合裁剪对照用
# ★AMPD 必须在 AMP 之前判★: 'zl.AMP' 是 'zl.AMPD' 的子串, 顺序反了会把动态 z 侧车
# 错标成 type7 —— 载荷 A|U|V 按 z|U|V 解析, 引擎跑出垃圾且不报错(2026-08-22)。
SKIP = set(int(x) for x in os.getenv("ZC_SKIP", "").split(",") if x.strip())
out = bytearray(); out += struct.pack('<II', 0x325A5144, NL)
tot = {5: 0, 6: 0, 7: 0, 8: 0, 9: 0}; miss = []   # 9 = zl.AMPD 动态 z
for L in range(NL):
    ops = []
    seen_any = False
    # 主记录 zrec_LXX.bin + 路由侧车 zrec_route_LXX.bin(type8, 2026-08-19) 两路合并
    for fn in (f'zrec_L{L:02d}.bin', f'zrec_route_L{L:02d}.bin'):
        p = os.path.join(ld, fn)
        if not os.path.exists(p):
            continue
        seen_any = True
        if os.path.getsize(p) == 0:
            continue
        raw = open(p, 'rb').read(); off = 0
        while off + 116 <= len(raw):
            nm = raw[off:off+16].split(b'\0')[0].decode('ascii', 'replace')
            psz, = struct.unpack_from('<Q', raw, off+88)
            pay = raw[off+116:off+116+psz]; off += 116 + psz
            ty = 5 if 'bf.GE' in nm and psz >= 512 else \
                 6 if 'zl.RRR' in nm and psz >= 16 else \
                 9 if 'zl.AMPD' in nm and psz >= 16 else \
                 7 if 'zl.AMP' in nm and psz >= 16 else \
                 8 if 'zl.RTE' in nm and psz >= 16 else 0
            if ty and ty not in SKIP: ops.append((ty, pay)); tot[ty] += 1
    if not seen_any:
        miss.append(L)
    out += struct.pack('<II', L, len(ops))
    for ty, pay in ops:
        out += struct.pack('<II', ty, len(pay)) + pay
open(outp, 'wb').write(bytes(out))
print(f'→ {outp} ({len(out)/1e6:.1f} MB) GE={tot[5]} z^L={tot[6]} AMP={tot[7]} RTE={tot[8]} AMPD={tot[9]}' + (f' ★缺层 {miss}★' if miss else ''))
