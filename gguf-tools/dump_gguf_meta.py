#!/usr/bin/env python3
"""Memory-safe GGUF metadata dumper. Reads ONLY the header/KV/tensor-info region
(never tensor data). Prints KV keys (+scalar values / array summaries) and tensor
infos (focusing on expert tensors). Usage: dump_gguf_meta.py FILE.gguf"""
import struct, sys

GGUF_TYPES = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',
              8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}
SCALAR_FMT = {0:('B',1),1:('b',1),2:('H',2),3:('h',2),4:('I',4),5:('i',4),
              6:('f',4),7:('?',1),10:('Q',8),11:('q',8),12:('d',8)}

def main(path):
    f = open(path, 'rb')
    magic = f.read(4)
    assert magic == b'GGUF', f'bad magic {magic!r}'
    version, = struct.unpack('<I', f.read(4))
    n_tensors, = struct.unpack('<Q', f.read(8))
    n_kv, = struct.unpack('<Q', f.read(8))
    print(f'== {path}')
    print(f'version={version} n_tensors={n_tensors} n_kv={n_kv}')

    def rstr():
        ln, = struct.unpack('<Q', f.read(8))
        return f.read(ln).decode('utf-8', 'replace')

    def rval(t):
        if t in SCALAR_FMT:
            fmt, sz = SCALAR_FMT[t]
            return struct.unpack('<'+fmt, f.read(sz))[0]
        if t == 8:
            return rstr()
        if t == 9:
            et, = struct.unpack('<I', f.read(4))
            cnt, = struct.unpack('<Q', f.read(8))
            if et == 8:
                vals = [rstr() for _ in range(cnt)]
            else:
                fmt, sz = SCALAR_FMT[et]
                raw = f.read(sz*cnt)
                vals = list(struct.unpack('<'+fmt*cnt, raw))
            return ('arr', GGUF_TYPES.get(et,et), cnt, vals)
        raise ValueError(f'unknown type {t}')

    print('--- KV ---')
    for _ in range(n_kv):
        key = rstr()
        t, = struct.unpack('<I', f.read(4))
        v = rval(t)
        if isinstance(v, tuple) and v[0] == 'arr':
            _, et, cnt, vals = v
            head = vals[:12]
            print(f'  {key} : arr<{et}>[{cnt}] {head}{" ..." if cnt>12 else ""}')
        else:
            sv = v if not isinstance(v, str) else (v[:80] + ('...' if len(v)>80 else ''))
            print(f'  {key} : {GGUF_TYPES.get(t,t)} = {sv}')

    stats = '--stats' in sys.argv
    print('--- TENSORS (type stats)' if stats else '--- TENSORS (expert + sample) ---')
    shown = 0
    infos = []
    for _ in range(n_tensors):
        name = rstr()
        nd, = struct.unpack('<I', f.read(4))
        dims = list(struct.unpack('<'+'Q'*nd, f.read(8*nd)))
        ttype, = struct.unpack('<I', f.read(4))
        off, = struct.unpack('<Q', f.read(8))
        infos.append((off, name, dims, ttype))
        if not stats and ((('_exps.' in name and ('blk.0.' in name or 'blk.1.' in name)) or shown < 6)):
            print(f'  {name}  dims={dims} type={ttype} off={off}')
            shown += 1
    if '--vqhead' in sys.argv:
        import os
        # data 区起点 = 对齐后的 info 区尾; alignment 默认 32(general.alignment 可覆盖)
        align = 32
        base = f.tell()
        if base % align: base += align - (base % align)
        for off, name, dims, tt in infos:
            if 'ffn_exps_vq.blob' not in name: continue
            f.seek(base + off)
            hdr = f.read(16 + 256*3*8)
            mg, ver, L, nexp = struct.unpack('<IIII', hdr[:16])
            tab = struct.unpack('<' + 'Q'*(256*3), hdr[16:])
            print(f'  {name}: magic={mg:08x} ver={ver} L={L} nexp={nexp} bytes={dims}')
            for which, wn in ((0,'w1'), (1,'w3'), (2,'w2')):
                po = tab[0*3 + which]
                if not po: print(f'    {wn}: 槽缺席(off=0)'); continue
                f.seek(base + off + po)
                ph = f.read(16)
                pmg, dim, nc = struct.unpack('<IHH', ph[:8])
                rows, cols = struct.unpack('<II', ph[8:16])
                print(f'    {wn}: magic={pmg:08x} dim={dim} nc={nc} rows={rows} cols={cols}')
            # 槽在位统计: fused2 的 probe 要求全 256 专家×3 槽非 0
            zeros = [0, 0, 0]
            ncs = {}
            for e in range(256):
                for w in range(3):
                    o = tab[e*3 + w]
                    if not o: zeros[w] += 1; continue
                    f.seek(base + off + o)
                    ph = f.read(8)
                    _pm, d, n = struct.unpack('<IHH', ph)
                    ncs[(w, d, n)] = ncs.get((w, d, n), 0) + 1
            print(f'    槽缺席: w1={zeros[0]} w3={zeros[1]} w2={zeros[2]} (fused2 要求全 0)')
            for (w, d, n), c in sorted(ncs.items()):
                print(f'    形态 {["w1","w3","w2"][w]}: dim={d} nc={n} × {c} 个专家')
            break
        f.close(); return
    if stats:
        import os
        # 字节数 = 相邻 offset 差(GGUF 张量数据按 info 顺序紧密排布); 末尾用文件尾。
        infos.sort()
        end = os.path.getsize(path)
        # 数据区起点未知也无妨: 只用差值, 末尾张量用 (文件尾 - 数据区总跨度) 近似。
        agg = {}
        for i, (off, name, dims, tt) in enumerate(infos):
            nxt = infos[i+1][0] if i+1 < len(infos) else None
            nb = (nxt - off) if nxt is not None else 0
            grp = 'routed_exps' if '_exps.' in name else ('attn' if '.attn' in name else
                  ('ffn_shared' if 'ffn' in name else 'other'))
            k = (tt, grp)
            a = agg.setdefault(k, [0, 0])
            a[0] += 1; a[1] += nb
        tot = sum(v[1] for v in agg.values())
        print(f'  {"type":>5} {"组":<12} {"张量数":>7} {"GiB":>9} {"占比":>7}')
        for (tt, grp), (c, nb) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
            print(f'  {tt:>5} {grp:<12} {c:>7} {nb/2**30:>9.2f} {100.0*nb/max(tot,1):>6.1f}%')
        print(f'  合计(按 offset 差) = {tot/2**30:.2f} GiB, 文件 {end/2**30:.2f} GiB')
    f.close()

if __name__ == '__main__':
    main(sys.argv[1])
