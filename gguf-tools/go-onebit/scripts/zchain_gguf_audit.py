#!/usr/bin/env python3
"""zchain_gguf_audit.py — 审计: 优化链(DQZ2 侧车)是否逐字节并入合一 GGUF 的 blk.*.opt_* 张量。
用法: python3 zchain_gguf_audit.py zchain_all.bin model.gguf [opt_LNN.bin ...]
逻辑: 重放 deepseek4-quantize zchain_in_load 的打包规则(16 float/op; GE 取最后一条 type5;
无 V8 的 dyn8 丢弃; V8 层内块序)→ 生成每层期望字节 → 与 GGUF 张量区实际字节比对。
可选再传若干 opt_LNN.bin(单层 DQZ2), 校验其与 zchain_all.bin 对应层一致(同源同终值)。
每行: L nops_raw nops_packed chain[=/≠/缺] ge[=/≠/缺/无] v8[=/≠/缺/无]; 末行 VERDICT。"""
import sys, struct

def parse_dqz2(path):
    """→ {L: [(type, payload_bytes), ...]}"""
    with open(path, "rb") as f:
        magic, nl = struct.unpack("<II", f.read(8))
        assert magic == 0x325A5144, f"{path}: bad DQZ2 magic"
        layers = {}
        for _ in range(nl):
            L, nops = struct.unpack("<II", f.read(8))
            ops = []
            for _ in range(nops):
                ty, psz = struct.unpack("<II", f.read(8))
                ops.append((ty, f.read(psz)))
            layers[L] = ops
    return layers

def pack_layer(ops):
    """重放量化器口径 → (chain_bytes, ge_bytes, v8_bytes, zlm_bytes, nops_raw, nops_packed)"""
    chain, v8, ge, zlm = b"", b"", None, None
    nblk = 0
    for ty, pay in ops:
        if ty == 5 and len(pay) >= 2:            # GE: 取最后一条, fp16→f32
            ne = len(pay) // 2
            import numpy as np
            ge = np.frombuffer(pay[:ne*2], dtype=np.float16).astype(np.float32).tobytes()
            continue
        if ty == 6 and len(pay) >= 16:           # 冻结 z^L(2026-07-14): 槽{6,tr,k}+opt_zlm 张量
            zk, tr, din, dout = struct.unpack("<IfII", pay[:16])
            nh = zk + zk*din + zk*dout
            if zk and din == dout and len(pay) >= 16 + nh*2:
                f = [0.0]*16; f[0] = 6.0; f[1] = tr; f[2] = float(zk); f[15] = 0.0
                chain += struct.pack("<16f", *f)
                zlm = pay[16:16+nh*2]
            continue
        if not (1 <= ty <= 4):
            continue
        f = [0.0]*16; f[0] = float(ty); f[15] = -1.0
        if ty == 1 and len(pay) >= 4:
            f[1] = struct.unpack("<f", pay[:4])[0]
        elif ty == 2 and len(pay) >= 16:
            f[2:6] = struct.unpack("<4f", pay[:16])
        elif ty == 3 and len(pay) >= 36:
            f[6:15] = struct.unpack("<9f", pay[:36])
            if len(pay) > 36:
                v8 += pay[36:]
                f[15] = float(nblk); nblk += 1
            else:
                continue                          # 无 V8 的 dyn8 = no-op, 量化器丢弃
        elif ty == 4 and len(pay) >= 4:
            f[1] = struct.unpack("<f", pay[:4])[0]
        else:
            continue
        chain += struct.pack("<16f", *f)
    return chain, ge, v8, zlm, len(ops), len(chain)//64

def gguf_tensors(path):
    """→ {name: (type_id, abs_off, n_el)} — 与 gguf_offsets.py 同口径"""
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    struct.unpack("<I", f.read(4))
    n_tensors, n_kv = struct.unpack("<qq", f.read(16))
    align = 32
    def rstr():
        (n,) = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
    def skip_val(t):
        scal = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t in scal: f.seek(scal[t], 1)
        elif t == 8: rstr()
        elif t == 9:
            (et,) = struct.unpack("<i", f.read(4)); (n,) = struct.unpack("<Q", f.read(8))
            if et == 8:
                for _ in range(n): rstr()
            else: f.seek(scal[et]*n, 1)
        else: raise SystemExit(f"kv type {t}?")
    for _ in range(n_kv):
        k = rstr(); (t,) = struct.unpack("<i", f.read(4))
        if k == "general.alignment" and t in (4,5): (align,) = struct.unpack("<I" if t==4 else "<i", f.read(4))
        else: skip_val(t)
    infos = []
    for _ in range(n_tensors):
        name = rstr(); (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack(f"<{nd}Q", f.read(8*nd))
        ty, off = struct.unpack("<iQ", f.read(12))
        nel = 1
        for d in dims: nel *= d
        infos.append((name, ty, off, nel))
    data0 = (f.tell() + align - 1)//align*align
    out = {n: (t, data0+o, e) for n, t, o, e in infos}
    return f, out

def main():
    dump_L = None
    argv = sys.argv[1:]
    if argv and argv[0] == "--dump":            # --dump N: 逐 op 打印期望 vs GGUF 的 16-float 记录
        dump_L = int(argv[1]); argv = argv[2:]
    zc, gguf = argv[0], argv[1]
    layers = parse_dqz2(zc)
    fh, tens = gguf_tensors(gguf)
    if dump_L is not None:
        chain, ge, v8, zlm, raw, packed = pack_layer(layers[dump_L])
        name = f"blk.{dump_L}.opt_chain.weight"
        _, off, nel = tens[name]
        fh.seek(off); got = fh.read(nel*4)
        print(f"L{dump_L:02d} raw_ops(侧车)={raw}:", " ".join(f"ty{t}(psz{len(p)})" for t, p in layers[dump_L]))
        for i in range(packed):
            e = struct.unpack("<16f", chain[i*64:(i+1)*64])
            g = struct.unpack("<16f", got[i*64:(i+1)*64])
            mark = "  ✓" if e == g else "  ✗ 异字段:" + ",".join(f"f[{j}] exp={e[j]:.6g} got={g[j]:.6g}" for j in range(16) if e[j] != g[j])
            print(f"  op{i:02d} ty={int(e[0])}{mark}")
        return
    def read_t(name, elsize):
        if name not in tens: return None
        _, off, nel = tens[name]
        fh.seek(off); return fh.read(nel*elsize)
    bad = 0
    for L in sorted(layers):
        chain, ge, v8, zlm, raw, packed = pack_layer(layers[L])
        row = [f"L{L:02d} raw={raw:3d} packed={packed:3d}"]
        for tag, exp, name, esz in (("chain", chain, f"blk.{L}.opt_chain.weight", 4),
                                    ("ge", ge, f"blk.{L}.opt_ge.weight", 4),
                                    ("v8", v8, f"blk.{L}.opt_v8.weight", 2),
                                    ("zlm", zlm, f"blk.{L}.opt_zlm.weight", 2)):
            got = read_t(name, esz)
            if not exp:
                st = "无" if got is None else "≠(侧车无但GGUF有)"
            elif got is None: st = "缺!"
            elif got == exp: st = "="
            else: st = f"≠! (exp {len(exp)}B vs got {len(got)}B{'' if len(exp)!=len(got) else ' 同长异字节'})"
            if st not in ("=", "无"): bad += 1
            row.append(f"{tag}{st}")
        print(" ".join(row))
    for extra in sys.argv[3:]:
        ol = parse_dqz2(extra)
        for L, ops in ol.items():
            same = ops == layers.get(L)
            print(f"{extra} L{L:02d}: {'≡ zchain 同层' if same else '≠ zchain!'}")
            if not same: bad += 1
    print(f"VERDICT {'✓ 优化链已逐字节并入GGUF' if bad==0 else f'✗ {bad} 处不一致/缺失'}")
    sys.exit(0 if bad == 0 else 1)

main()
