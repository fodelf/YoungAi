#!/usr/bin/env python3
"""probe_behavior_spectrum.py — 【行为空间】误差谱实测(2026-08-08 用户纠正: z 是每层
动态修正、目标是高维行为, 不是权重残差)。

理论: 权重残差 ΔW 满秩白噪声(已实测), 但模型只在激活流形上工作 —
      ΔY = X·ΔWᵀ 的秩 ≤ min(rank ΔW, rank X), 由【激活有效维度】封顶。
      若 ΔY 谱陡 ⇒ 每层一份的小 z 足以还原行为(用户理论成立), 且 rank 由谱反推。

数据源: 锚 fin(每层真实 MoE 输入, FP 口径, 部署同源) + HF 原始权重 + vq 侧车量化权重。
用法: probe_behavior_spectrum.py <hf> <layers_dir> <anchor.bin> <层> [专家数=8] [tok=512]
"""
import json, os, struct, sys
import numpy as np

FP4T = np.array([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.], dtype=np.float32)

def st_index(hf):
    return json.load(open(os.path.join(hf, "model.safetensors.index.json")))["weight_map"]

def st_raw(hf, wmap, name):
    with open(os.path.join(hf, wmap[name]), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
        s, e = meta["data_offsets"]
        f.seek(8 + n + s)
        return meta["dtype"], meta["shape"], f.read(e - s)

def st_mxfp4(hf, wmap, name):
    dt, sh, raw = st_raw(hf, wmap, name)
    assert dt == "I8", f"{name}: {dt}"
    R, Cc = sh; Cin = Cc * 2; nblk = Cin // 32
    _, sh2, sraw = st_raw(hf, wmap, name.replace(".weight", ".scale"))
    e = np.frombuffer(sraw, dtype=np.uint8).reshape(R, nblk).astype(np.uint32)
    u = np.where(e == 0, np.uint32(0x00400000), e << 23).astype(np.uint32)
    scale = np.frombuffer(u.tobytes(), dtype=np.float32).reshape(R, nblk)
    b = np.frombuffer(raw, dtype=np.uint8).reshape(R, nblk, 16)
    w = np.empty((R, nblk, 32), dtype=np.float32)
    w[:, :, 0::2] = FP4T[b & 0x0F]; w[:, :, 1::2] = FP4T[(b >> 4) & 0x0F]
    w *= scale[:, :, None]
    return w.reshape(R, Cin)

def vq_slot(blob, e, w):
    return struct.unpack_from("<Q", blob, 16 + (e * 3 + w) * 8)[0]

def vq_dequant(blob, off):
    assert struct.unpack_from("<I", blob, off)[0] == 0x51565144
    dim, nc = struct.unpack_from("<HH", blob, off + 4)
    rows, cols = struct.unpack_from("<II", blob, off + 8)
    p = off + 16
    cb = np.frombuffer(blob, dtype=np.float16, count=nc*dim, offset=p).astype(np.float32).reshape(nc, dim); p += nc*dim*2
    gr = np.frombuffer(blob, dtype=np.float16, count=rows, offset=p).astype(np.float32); p += rows*2
    nbit = max(1, (nc-1).bit_length()); nidx = rows*cols//dim
    if nbit == 8:
        idx = np.frombuffer(blob, dtype=np.uint8, count=nidx, offset=p).astype(np.int32)
    else:
        nby = (nidx*nbit+7)//8 + 1
        bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8, count=nby, offset=p), bitorder="little")
        pos = np.arange(nidx)*nbit
        idx = np.zeros(nidx, dtype=np.int32)
        for b in range(nbit): idx |= bits[pos+b].astype(np.int32) << b
    return (cb[idx] * gr.repeat(cols//dim)[:, None]).reshape(rows, cols)

def anchor_fin(ap, L, ntok):
    with open(ap, "rb") as f:
        hd = struct.unpack("<8I", f.read(32))
        S, HCM, DIM, NL = hd[1], hd[2], hd[3], hd[4]
        f.seek(40 + (L*S)*DIM*4)
        return np.frombuffer(f.read(ntok*DIM*4), dtype=np.float32).reshape(ntok, DIM), DIM

def cumspec(M, ks):
    s = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    c = np.cumsum(s**2)/ (s**2).sum()
    return {k: float(c[min(k, len(c))-1]) for k in ks}

def main():
    hf, ld, ap, L = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    nexp = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    ntok = int(sys.argv[6]) if len(sys.argv) > 6 else 512
    wmap = st_index(hf)
    X, DIM = anchor_fin(ap, L, ntok)
    blob = open(os.path.join(ld, f"dql_vq_L{L:02d}.bin"), "rb").read()
    KS = (1,2,4,8,16,32,64,128)
    print(f"L{L} 真实激活 X=[{ntok}×{DIM}] (锚 fin, 部署同源)")
    print(f"  X 自身谱: " + "  ".join(f"k={k}:{v*100:.1f}%" for k, v in cumspec(X, (8,16,32,64,128,256)).items()))
    print(f"\n{'专家':>5} {'矩阵':>4} {'ΔY相对':>8} " + "  ".join(f"k={k}" for k in KS))
    agg = []
    for e in range(0, 256, max(1, 256//nexp)):
        for wi, nm in ((0, "w1"), (1, "w3")):
            off = vq_slot(blob, e, wi)
            if not off: continue
            tn = f"layers.{L}.ffn.experts.{e}.{nm}.weight"
            if tn not in wmap: continue
            W = st_mxfp4(hf, wmap, tn); Wq = vq_dequant(blob, off)
            if W.shape != Wq.shape: W = W.T
            dY = X @ (W - Wq).T          # [ntok, 2048] 行为空间误差
            Yf = X @ W.T
            rel = np.linalg.norm(dY)/(np.linalg.norm(Yf)+1e-12)
            cs = cumspec(dY, KS)
            print(f"{e:>5} {nm:>4} {rel:>8.4f} " + "  ".join(f"{cs[k]*100:5.1f}%" for k in KS))
            agg.append([cs[k] for k in KS])
        if len(agg) >= 2*nexp: break
    if agg:
        m = np.mean(agg, axis=0)
        print("\n均值:      " + "  ".join(f"{v*100:5.1f}%" for v in m))
        print("对照 白噪声(k/512):" + "  ".join(f"{min(k/ntok,1)*100:5.1f}%" for k in KS))

if __name__ == "__main__":
    main()
