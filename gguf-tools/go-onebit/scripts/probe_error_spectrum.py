#!/usr/bin/env python3
"""probe_error_spectrum.py — 量化误差矩阵的奇异值谱实测(2026-08-08 用户令"客观评估
99% 还原, rank 不许盲目设定")。

判什么: ΔW = W_原始 − W_量化 的能量在前 k 个奇异值上的累积占比。
  强结构谱(前 k 占 90%+) ⇒ 低秩 z 是 99% 还原的有效杠杆, rank 由挽回率反推;
  白噪声谱(前 k ≈ k/rank_full) ⇒ 低秩路线当场否决, z 必须换形式。

数据源(都是真实产物, 不做任何近似):
  W_原始  = HF safetensors 的 MXFP4(I8 nibble + E8M0 1×32 scale), 解码与 st_read.c 逐位一致
  W_量化  = r86 的 dql_vq 侧车 DQVQ 载荷(含 GPTQ 补偿的真实量化结果), 解码与 vq_fmt.h 一致

用法: probe_error_spectrum.py <hf_dir> <layers_dir> <层号...>   (每层取 3 个专家: 首/中/末)
"""
import json, os, struct, sys
import numpy as np

FP4T = np.array([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.], dtype=np.float32)

# ── HF safetensors: 找张量所在 shard + 读 MXFP4 ──
def st_index(hf):
    p = os.path.join(hf, "model.safetensors.index.json")
    return json.load(open(p))["weight_map"]

def st_read_raw(hf, wmap, name):
    """返回 (dtype, shape, bytes) — 只读, 不解码"""
    shard = wmap[name]
    p = os.path.join(hf, shard)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        meta = hdr[name]
        s, e = meta["data_offsets"]
        f.seek(8 + n + s)
        return meta["dtype"], meta["shape"], f.read(e - s)

def st_mxfp4(hf, wmap, name):
    """MXFP4 专家权重 → f32 [R, C]; scale 张量名 = .weight→.scale"""
    dt, sh, raw = st_read_raw(hf, wmap, name)
    assert dt == "I8", f"{name}: dtype={dt} 非 MXFP4 容器"
    R, Cc = sh
    Cin = Cc * 2
    nblk = Cin // 32
    sname = name.replace(".weight", ".scale")
    dt2, sh2, sraw = st_read_raw(hf, wmap, sname)
    assert dt2 == "F8_E8M0" and tuple(sh2) == (R, nblk), f"{sname}: {dt2} {sh2}"
    e = np.frombuffer(sraw, dtype=np.uint8).reshape(R, nblk).astype(np.uint32)
    u = np.where(e == 0, np.uint32(0x00400000), e << 23)
    scale = u.view(np.float32).reshape(R, nblk) if u.dtype == np.uint32 else u.astype(np.float32)
    scale = np.frombuffer(u.astype(np.uint32).tobytes(), dtype=np.float32).reshape(R, nblk)
    b = np.frombuffer(raw, dtype=np.uint8).reshape(R, nblk, 16)
    lo = FP4T[b & 0x0F]           # [R,nblk,16]
    hi = FP4T[(b >> 4) & 0x0F]
    w = np.empty((R, nblk, 32), dtype=np.float32)
    w[:, :, 0::2] = lo
    w[:, :, 1::2] = hi
    w *= scale[:, :, None]
    return w.reshape(R, Cin)

# ── DQVQ 载荷解码(与 vq_fmt.h ds4vq_dequant_f32 同口径) ──
def vq_slot(blob, e, which):
    return struct.unpack_from("<Q", blob, 16 + (e * 3 + which) * 8)[0]

def vq_dequant(blob, off):
    mg, = struct.unpack_from("<I", blob, off)
    assert mg == 0x51565144, f"DQVQ 魔数 {mg:#x}"
    dim, nc = struct.unpack_from("<HH", blob, off + 4)
    rows, cols = struct.unpack_from("<II", blob, off + 8)
    p = off + 16
    cb = np.frombuffer(blob, dtype=np.float16, count=nc * dim, offset=p).astype(np.float32).reshape(nc, dim)
    p += nc * dim * 2
    gr = np.frombuffer(blob, dtype=np.float16, count=rows, offset=p).astype(np.float32)
    p += rows * 2
    nbit = max(1, (nc - 1).bit_length())
    nidx = rows * cols // dim
    if nbit == 8:
        idx = np.frombuffer(blob, dtype=np.uint8, count=nidx, offset=p).astype(np.int32)
    else:  # 位流 LE
        nby = (nidx * nbit + 7) // 8 + 1
        bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8, count=nby, offset=p), bitorder="little")
        idx = np.zeros(nidx, dtype=np.int32)
        for b in range(nbit):
            idx |= bits[b::0 + 1][0:0] if False else 0
        # 逐索引取位(nbit 非 8 时索引跨字节, 用 stride 取)
        pos = np.arange(nidx) * nbit
        for b in range(nbit):
            idx |= bits[pos + b].astype(np.int32) << b
    w = (cb[idx] * gr.repeat(cols // dim)[:, None]).reshape(rows, cols)
    return w, dim, nc

def spectrum(dW, ks=(4, 8, 16, 32, 64, 128, 256)):
    s = np.linalg.svd(dW.astype(np.float64), compute_uv=False)
    e = s ** 2
    tot = e.sum()
    cum = np.cumsum(e) / tot
    return {k: float(cum[min(k, len(cum)) - 1]) for k in ks}, len(s)

def main():
    hf, ld = sys.argv[1], sys.argv[2]
    layers = [int(x) for x in sys.argv[3:]] or [0, 21, 42]
    wmap = st_index(hf)
    print(f"{'层':>3} {'专家':>5} {'矩阵':>3} {'档':>10} {'相对误差':>9} " +
          "  ".join(f"k={k}" for k in (4, 8, 16, 32, 64, 128, 256)))
    for L in layers:
        blob = open(os.path.join(ld, f"dql_vq_L{L:02d}.bin"), "rb").read()
        assert blob[:4] == b"DQVL", "非 DQVL blob"
        for e in (0, 128, 255):
            for wi, (which, hfnm) in enumerate(((0, "w1"), (1, "w3"), (2, "w2"))):
                off = vq_slot(blob, e, which)
                if not off:
                    print(f"L{L:02d} e{e:>4} {hfnm:>4}  (无 vq 槽=冷 signref, 跳过)")
                    continue
                Wq, dim, nc = vq_dequant(blob, off)
                nm = f"layers.{L}.ffn.experts.{e}.{hfnm}.weight"   # 0731 命名
                if nm not in wmap:
                    print(f"  {nm} 不在 index, 跳过"); continue
                W = st_mxfp4(hf, wmap, nm)
                if W.shape != Wq.shape:
                    W = W.T if W.T.shape == Wq.shape else W
                dW = W - Wq
                rel = np.linalg.norm(dW) / (np.linalg.norm(W) + 1e-12)
                cum, nfull = spectrum(dW)
                cols = "  ".join(f"{cum[k]*100:5.1f}%" for k in (4, 8, 16, 32, 64, 128, 256))
                print(f"L{L:02d} e{e:>4} {hfnm:>4} vq{dim}x{nc:<5} {rel:>8.4f}  {cols}   (满秩={nfull})")

if __name__ == "__main__":
    main()
