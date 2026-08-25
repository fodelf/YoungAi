#!/usr/bin/env python3
# dql_strip.py — 从 DQL2 层文件剥掉 "1bit" 基座记录, 产纯部署文件(R24 真体积口径)。
# 保留: g2hot(热go2b) / z·loss·bwd 小记录。
# 用法: dql_strip.py <in.dql> <out.dql>   → stdout: 每记录名+字节 与 总字节
import struct, sys, os

# ★哨兵拦截(2026-07-30 用户令"跑完量化→反修→合并"): NO_STRIP 存在时拒绝剥壳 —
#   反修/合并需要完整层文件(1bit 记录=lfile 回放基座), 剥壳只在全流程收官后手动执行。
#   (r36full 收官段的 `strip && mv` 短路, 层文件原样保留)
if os.path.exists(os.path.expanduser('~/ds4-main/gguf/go-onebit/r36/NO_STRIP')):
    print('STRIP 被哨兵拦截(r36/NO_STRIP): 保留完整层文件给反修/合并', file=sys.stderr)
    sys.exit(3)

# --keep-w2: "1bit" 记录不丢弃, 改为截取其 D 段(w2 signref 字节)重打包成 "w2sr" 记录
#            (R36 口径: 冷 w2=signref 全精, G/U 段=稀疏洞不入部署体积)
KEEP_W2 = '--keep-w2' in sys.argv
W2_BYTES = 285212672          # 256×szD = 256×4096×272 (1.0625bpw 行字节, 冠军同款)
args = [a for a in sys.argv[1:] if a != '--keep-w2']
src, dst = args[0], args[1]
f = open(src, 'rb')
magic, L, nrec = struct.unpack('<III', f.read(12))
assert magic == 0x324C5144, 'DQL2 魔数不符'
keep = []
dropped = 0
for _ in range(nrec):
    hdr = f.read(116)
    name = hdr[:16].split(b'\0')[0].decode()
    (psz,) = struct.unpack('<Q', hdr[88:96])
    pay = f.read(psz)
    assert len(pay) == psz, f'{name} 载荷截断'
    if name == '1bit':
        if KEEP_W2:
            d = pay[-W2_BYTES:]                     # 载荷尾部 = D 段(w2 signref)
            nh = b'w2sr'.ljust(16, b'\0') + hdr[16:80] \
                 + struct.pack('<QQ', W2_BYTES, W2_BYTES) + hdr[96:]
            keep.append(('w2sr', nh, d))
            dropped += 116 + psz - (116 + W2_BYTES)
        else:
            dropped += 116 + psz
        continue
    keep.append((name, hdr, pay))
f.close()
o = open(dst, 'wb')
o.write(struct.pack('<III', magic, L, len(keep)))
tot = 12
for name, hdr, pay in keep:
    o.write(hdr); o.write(pay)
    tot += 116 + len(pay)
    if len(pay) > 1 << 20:
        print(f'  keep {name:12s} {len(pay)} B ({len(pay)/2**20:.1f} MiB)')
o.close()
print(f'STRIP L={L} 记录 {nrec}→{len(keep)} 剥1bit={dropped} B  部署文件={tot} B ({tot/2**20:.2f} MiB, {tot/2**30:.4f} GiB)')
