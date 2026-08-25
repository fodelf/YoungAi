"""zlever/dql_to_zchain.py — 从 dql 主文件内嵌 vd=1 记录(zlayer 注入的 bf.GE/zl.RRR)
抽出引擎 DQZ2 外挂 zchain(冠军侧车挂载形态)。载荷逐字节直通。
用法: dql_to_zchain.py <layers_dir> <out.zchain.bin> [nl=43]
"""
import os, sys, struct
ld, outp = sys.argv[1], sys.argv[2]
NL = int(sys.argv[3]) if len(sys.argv) > 3 else 43
out = bytearray(); out += struct.pack('<II', 0x325A5144, NL)
tot = {5: 0, 6: 0}
for L in range(NL):
    p = os.path.join(ld, f'dql_L{L:02d}.bin')
    ops = []
    raw = open(p, 'rb').read()
    assert raw[:4] == b'DQL2', p
    nrec, = struct.unpack_from('<I', raw, 8)
    off = 12
    for _ in range(nrec):
        nm = raw[off:off+16].split(b'\0')[0].decode('ascii', 'replace')
        psz, = struct.unpack_from('<Q', raw, off+88)
        vd, = struct.unpack_from('<i', raw, off+112)
        pay = raw[off+116:off+116+psz]
        off += 116 + psz
        if vd != 1: continue
        if 'bf.GE' in nm and psz >= 512: ops.append((5, pay)); tot[5] += 1
        elif 'zl.RRR' in nm and psz >= 16: ops.append((6, pay)); tot[6] += 1
    out += struct.pack('<II', L, len(ops))
    for ty, pay in ops:
        out += struct.pack('<II', ty, len(pay)) + pay
open(outp, 'wb').write(bytes(out))
print(f'→ {outp} ({len(out)/1e6:.1f} MB) GE={tot[5]} z^L={tot[6]}')
