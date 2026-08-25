#!/usr/bin/env python3
# zrollback.py — 按 zinject_manifest.txt 账本回滚反修注入(截断 dql_LXX.bin + 恢复 nrec),
# 旧账本改名留档。用法: zrollback.py <layers_dir> [留档后缀=bak]
import struct, os, sys

ld = sys.argv[1]
suf = sys.argv[2] if len(sys.argv) > 2 else "bak"
man = os.path.join(ld, "zinject_manifest.txt")
if not os.path.exists(man):
    print("无账本, 层已是裸态"); sys.exit(0)
n = 0
for ln in open(man):
    L, osz, n0 = (int(x) for x in ln.split())
    if osz < 0: continue   # 不注入层无动作
    p = os.path.join(ld, f"dql_L{L:02d}.bin")
    cur = os.path.getsize(p)
    assert cur >= osz, (L, cur, osz)
    f = open(p, "r+b"); f.truncate(osz); f.seek(8); f.write(struct.pack("<I", n0)); f.close()
    n += 1
os.rename(man, f"{man}.{suf}")
for f in os.listdir(ld):
    if f.startswith("zcache_"): os.remove(os.path.join(ld, f))
# 验证: 无 zl./bf. 残留
bad = 0
for L in range(43):
    p = os.path.join(ld, f"dql_L{L:02d}.bin")
    if not os.path.exists(p): continue
    raw = open(p, "rb").read()
    nr, = struct.unpack_from("<I", raw, 8); off = 12; names = []
    for _ in range(nr):
        nm = raw[off:off+16].split(b"\0")[0].decode("ascii", "replace")
        psz, = struct.unpack_from("<Q", raw, off+88); off += 116+psz; names.append(nm)
    if off > len(raw) or any(("zl." in x or "bf." in x) for x in names):
        bad += 1; print("BAD", L, nr, names)
print(f"回滚 {n} 层, 验证 {'all-clean' if bad == 0 else str(bad)+' bad'}")
sys.exit(1 if bad else 0)
