#!/usr/bin/env python3
# ops_to_zchain.py — dql_ops 侧车(DQO2, 反修 op 唯一权威)→ DQZ2 运行时链。
#
# 背景(2026-08-06 用户令"修通op"): 8-04 平行架构后反修 op 全进侧车, zchain_write 旧线
# 断供(收官只写空链 352B)⇒ 运行时(--zchain)拿不到反修效果。本工具从侧车直转:
# 记录名→type 映射与量化器 parse_op_rec 完全同规则, 载荷逐字节直通(格式二者一致)。
# 用法: ops_to_zchain.py <layers_dir> <out.zchain.bin> [nl=43]
import struct, sys, os

layers, outp = sys.argv[1], sys.argv[2]
NL = int(sys.argv[3]) if len(sys.argv) > 3 else 43
# 消融过滤(2026-08-07 行为毒源归因): 第4参数=逗号分隔 type 白名单(如 "1,4"), 缺省全族
ONLY = set(int(x) for x in sys.argv[4].split(',')) if len(sys.argv) > 4 and sys.argv[4] != 'all' else None
# GL 界闸(2026-08-07 级联62倍定谳): 第5参数=clamp 界(如 "0.85,1.15"), 把 type1 g 拉回设计网格界
CLAMP = tuple(float(x) for x in sys.argv[5].split(',')) if len(sys.argv) > 5 else None
REC_HDR = 116   # nm[16] al[64] vol@80 psz@88 m4@108 vd@112 pay@116

def op_type(nm, psz):
    if 'GLhc' in nm: return 0          # type7: 合并器折 ge, DQZ2 引擎无此 type — 外挂剔除
                                        # (且 '.GL' 子串会把它误编 type1 载荷错位 — 显式先判)
    if 'GLdyn2' in nm and psz >= 16: return 2
    if 'GLdyn8' in nm and psz >= 36: return 3
    if 'bf.GE' in nm and psz >= 512: return 5
    if 'zl.RRR' in nm and psz >= 16: return 6
    if '.GL' in nm and psz >= 4: return 1
    if ('TREF' in nm or 'xlayer' in nm) and psz >= 4: return 4
    return 0

out = bytearray()
out += struct.pack('<II', 0x325A5144, NL)
tot = {1:0,2:0,3:0,4:0,5:0,6:0}
for L in range(NL):
    p = os.path.join(layers, f'dql_ops_L{L:02d}.bin')
    ops = []
    if os.path.exists(p) and os.path.getsize(p) >= 12:
        raw = open(p, 'rb').read()
        assert raw[:4] == b'DQO2', f'{p} magic'
        nrec, = struct.unpack_from('<I', raw, 8)
        off = 12
        for _ in range(nrec):
            if off + REC_HDR > len(raw): break
            nm = raw[off:off+16].split(b'\0')[0].decode('ascii', 'replace')
            psz, = struct.unpack_from('<Q', raw, off+88)
            vd, = struct.unpack_from('<i', raw, off+112)
            pay = raw[off+REC_HDR:off+REC_HDR+psz]
            off += REC_HDR + psz
            if vd != 1: continue                     # 只搬落地链末回放 op
            ty = op_type(nm, psz)
            if ONLY is not None and ty not in ONLY: ty = 0
            if ty == 1 and CLAMP:
                g, = struct.unpack_from('<f', pay, 0)
                g2 = min(max(g, CLAMP[0]), CLAMP[1])
                if g2 != g: pay = struct.pack('<f', g2) + pay[4:]
            if ty == 0: continue
            ops.append((ty, pay)); tot[ty] += 1
    out += struct.pack('<II', L, len(ops))
    for ty, pay in ops:
        out += struct.pack('<II', ty, len(pay)) + pay
open(outp, 'wb').write(out)
print(f'[ops_to_zchain] → {outp} ({len(out)/1e6:.2f} MB) ops: ' +
      ' '.join(f'type{t}={n}' for t, n in tot.items() if n))
