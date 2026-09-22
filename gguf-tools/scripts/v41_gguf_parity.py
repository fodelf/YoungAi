#!/usr/bin/env python3
"""v41_gguf_parity.py — 转换后 GGUF 的逐字节回读对拍(2026-09-12, P0 判决)。金标夹具, 不进数值链。

对拍项(字节级, 不是容差级):
  ① fp4x32 块: 第 r 行第 b 块的 16 B nibble == HF weight[r, 16b:16b+16], 第 17 B == HF scale[r, b]
  ② VQ blob: 头/表/每载荷的 dim/nc/rows/cols 与 码本/行增益/索引字节 == 量化目录三件
  ③ bf16→f32: 位左移 16 逐位同
  ④ engram wkv f16: 与 fp8×ue8m0 按 f16(RNE) 舍入逐位同(numpy astype float16 = RNE)
用法: v41_gguf_parity.py <量化目录> <x.gguf> [层号=2]
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v41_hf_io import build_index, _mm  # noqa: E402

GT = {12: ("q4_K", 256, 144), 30: ("bf16", 1, 2), 0: ("f32", 1, 4), 1: ("f16", 1, 2), 26: ("i32", 1, 4), 27: ("i64", 1, 8), 42: ("vqblob", 1, 1), 43: ("fp4x32", 32, 17)}


def read_gguf(path):
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    b = bytes(mm[: 64 << 20])                          # 头远小于 64 MB(★别 mm.tobytes(): 那是把整个 110 GB 文件读进内存★)
    pos = 0
    def u32():
        nonlocal pos; v = struct.unpack_from("<I", b, pos)[0]; pos += 4; return v
    def u64():
        nonlocal pos; v = struct.unpack_from("<Q", b, pos)[0]; pos += 8; return v
    def s():
        nonlocal pos; n = u64(); v = b[pos:pos + n].decode(); pos += n; return v
    assert b[:4] == b"GGUF"; pos = 4
    ver, nt, nkv = u32(), u64(), u64()
    kv = {}
    sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    def val(t):
        nonlocal pos
        if t == 8: return s()
        if t == 9:
            et, n = u32(), u64()
            return [val(et) for _ in range(n)]
        v = struct.unpack_from(fmt[t], b, pos)[0]; pos += sz[t]; return v
    for _ in range(nkv):
        k = s(); t = u32(); kv[k] = val(t)
    tens = {}
    for _ in range(nt):
        name = s(); nd = u32(); ne = [u64() for _ in range(nd)]; ty = u32(); off = u64()
        tens[name] = (ty, ne, off)
    align = kv.get("general.alignment", 32)
    data0 = (pos + align - 1) // align * align
    return kv, tens, data0, mm


def tbytes(mm, data0, t):
    ty, ne, off = t
    _, blk, tsz = GT[ty]
    n = int(np.prod(ne)); nb = n // blk * tsz
    return np.asarray(mm[data0 + off: data0 + off + nb])


def main():
    qd, gg = Path(sys.argv[1]), sys.argv[2]
    L = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    idx = build_index(qd)
    kv, tens, data0, mm = read_gguf(gg)
    print(f"[gguf] {len(tens)} 张量, {len(kv)} 键, 数据区起点 {data0}, 变体 {kv.get('deepseek4.variant')}, 压缩比 {kv['deepseek4.attention.compress_ratios'][:6]}…")
    bad = 0
    def raw(name):
        p, off, dt, shp = idx[name]
        if dt == "Q4_K":   # 块格式: shape 记逻辑形状, 字节按 144/256 算(按 shape 算会短 4.5 倍)
            nbytes = shp[0] * (shp[1] // 256) * 144
        else:
            nbytes = {"I8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F16": 2, "F32": 4, "U8": 1}[dt] * int(np.prod(shp))
        return np.frombuffer(_mm(p)[off:off + nbytes], dtype=np.uint8), shp
    # ① 骨架: 按量化目录里登记的 dtype 分流(fp4x32 或 q4_K), 两者都必须逐字节搬过来
    for gn, hn in ((f"blk.{L}.attn_q_a.weight", f"layers.{L}.attn.wq_a.weight"), (f"blk.{L}.ffn_down_shexp.weight", f"layers.{L}.ffn.shared_experts.w2.weight"), ("token_embd.weight", "embed.weight")):
        if idx[hn][2] == "Q4_K":
            w, shp = raw(hn)
            got = tbytes(mm, data0, tens[gn])
            ok = np.array_equal(got, w)
            print(f"  q4_K {gn}: {'✓ 逐字节同' if ok else '★不同★'}  ({shp[0]}×{shp[1] // 256} 块, {len(w)} B)"); bad += not ok
            continue
        w, shp = raw(hn); sc, sshp = raw(hn[:-6] + "scale")
        rows, cb = shp[0], shp[1] // 16
        blocks = tbytes(mm, data0, tens[gn]).reshape(rows, cb, 17)
        ok = np.array_equal(blocks[:, :, :16].reshape(rows, -1), w.reshape(rows, -1)) and np.array_equal(blocks[:, :, 16], sc.reshape(rows, cb))
        print(f"  fp4x32 {gn}: {'✓ 逐字节同' if ok else '★不同★'}  ({rows}×{cb} 块)"); bad += not ok
    # ③ bf16 → f32
    for gn, hn in ((f"blk.{L}.attn_norm.weight", f"layers.{L}.attn_norm.weight"), (f"blk.{L}.ffn_gate_inp.weight", f"layers.{L}.ffn.gate.weight")):
        w, shp = raw(hn)
        ref = (w.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        got = tbytes(mm, data0, tens[gn]).view(np.float32)
        ok = np.array_equal(ref, got); print(f"  bf16→f32 {gn}: {'✓' if ok else '★不同★'}"); bad += not ok
    # ② VQ blob
    blob = tbytes(mm, data0, tens[f"blk.{L}.ffn_exps_vq.blob"]).tobytes()
    mg, ver, lid, nexp = struct.unpack_from("<4I", blob, 0)
    assert mg == 0x4C565144 and lid == L, (hex(mg), lid)
    tab = np.frombuffer(blob[16:16 + nexp * 24], dtype=np.uint64).reshape(nexp, 3)
    nchk = 0
    for e in (0, 1, nexp // 2, nexp - 1):
        cbw, _ = raw(f"layers.{L}.ffn.experts.{e}.vq.cb")
        for wi, m in enumerate(("w1", "w3", "w2")):
            off = int(tab[e, wi]); pm, dim, nc, rows, cols = struct.unpack_from("<IHHII", blob, off)
            assert pm == 0x51565144
            ix, ishp = raw(f"layers.{L}.ffn.experts.{e}.{m}.vq.idx"); g, _ = raw(f"layers.{L}.ffn.experts.{e}.{m}.vq.gain")
            p = off + 16
            ok = (blob[p:p + nc * dim * 2] == cbw.tobytes()); p += nc * dim * 2
            ok &= (blob[p:p + rows * 2] == g.tobytes()); p += rows * 2
            ok &= (blob[p:p + ix.size] == ix.tobytes())
            ok &= rows == ishp[0] and cols == ishp[1] * 8 // 12 * dim
            bad += not ok; nchk += 1
            if not ok: print(f"  ★VQ L{L} e{e} {m} 不同★")
    print(f"  VQ blob L{L}: ver {ver} nexp {nexp}, 抽查 {nchk} 载荷 {'全逐字节同 ✓' if not bad else '有差'}")
    # ④ engram wkv f16
    for i, EL in enumerate(kv["deepseek4.engram.layer_ids"]):
        w, shp = raw(f"layers.{EL}.engram.wkv.weight"); sc, sshp = raw(f"layers.{EL}.engram.wkv.scale")
        rows, cols = shp
        import torch
        wf = torch.from_numpy(w.copy()).view(torch.float8_e4m3fn).float().reshape(rows, cols)
        sf = torch.from_numpy(sc.copy()).view(torch.float8_e8m0fnu).float().reshape(sshp[0], sshp[1])
        ref = (wf.reshape(rows // 32, 32, cols // 32, 32) * sf[:, None, :, None]).reshape(rows, cols).numpy().astype(np.float16)
        got = tbytes(mm, data0, tens[f"blk.{EL}.engram_wkv.weight"]).view(np.float16).reshape(rows, cols)
        ok = np.array_equal(ref.view(np.uint16), got.view(np.uint16))
        print(f"  engram wkv L{EL} f16: {'✓ 逐位同' if ok else '★不同★'} (max|ref| {np.abs(ref.astype(np.float32)).max():.3g})"); bad += not ok
    # 常量
    tm = tbytes(mm, data0, tens["engram.token_map"]).view(np.int32)
    print(f"  engram.token_map: {tm.size} 项, 压缩 id 范围 [{tm.min()}, {tm.max()}] (声明 {kv['deepseek4.engram.compressed_vocab_size']})")
    bad += not (tm.max() + 1 == kv["deepseek4.engram.compressed_vocab_size"])
    print("★对拍失败★" if bad else "对拍全绿")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
