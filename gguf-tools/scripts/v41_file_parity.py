#!/usr/bin/env python3
"""v41_file_parity.py — 落盘量化模型的读回对拍(2026-09-12)。金标夹具, 不是数值链: numpy 那份
解包只用来当"笨参照", 生产解码在 libv41vq.so(与量化器同一个核)。

① VQ 三件(.vq.idx / .vq.cb / .vq.gain): numpy 逐位解包+查表  vs  C 核解码 → 必须逐元素相等
   (相等 = 位流布局/位宽/f16 口径三者都对; 差一位就整行错位, 而错位不报错只出假数)
② C 解码值 vs HF 出厂 FP4 dequant → 残差, 与量化器日志报的 cos 对账
③ 骨架: 重量化张量 dequant vs 出厂 FP8 dequant → 残差
   - FP4 档: scale 字节不得是 255(NaN 槽)
   - q4_K 档(2026-09-19): numpy 按 144 B 块布局笨解  vs  libv41vq.so 的 GPU 核 → 必须逐元素相等
     (那个核与 src/common/ds4_deq_q4_K 金标逐式同源; 块内 8 组 6-bit scale/min 的打包
      错一位不报错, 只让整组元素偏一个常数)

用法: v41_file_parity.py <hf-dir> <quant-dir> [层号=0] [对拍专家数=4]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v41_hf_io import FP4_TABLE, build_index, load_raw, dq_fp4, dq_fp8, load_vq, load_q4k


def np_q4k(blocks, rows, cols):
    """q4_K 笨参照(144 B / 256 元素): [d f16][dmin f16][12 B 打包的 8 组 6-bit scale/min][128 B qs]。
    qs 每 32 字节存相邻两个子块: 偶数子块取低 nibble, 奇数子块取高 nibble。"""
    nb = rows * (cols // 256)
    b = blocks.reshape(nb, 144)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(nb)
    sc, qs = b[:, 4:16], b[:, 16:144]
    out = np.empty((nb, 256), dtype=np.float32)
    for j in range(8):
        if j < 4:
            s, m = sc[:, j] & 63, sc[:, j + 4] & 63
        else:
            s = (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)
            m = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
        byte = qs[:, (j // 2) * 32:(j // 2) * 32 + 32]
        v = (byte >> 4) if (j & 1) else (byte & 0xF)
        out[:, j * 32:(j + 1) * 32] = (d[:, None] * s[:, None].astype(np.float32) * v.astype(np.float32)
                                       - dmin[:, None] * m[:, None].astype(np.float32))
    return out.reshape(rows, cols)


def np_decode(pk, cb, gain):
    rows, bytes_row = pk.shape
    nc, dim = cb.shape
    bits = int(np.ceil(np.log2(nc)))
    nidx = bytes_row * 8 // bits
    b = np.unpackbits(pk, axis=1, bitorder="little")[:, :nidx * bits].reshape(rows, nidx, bits)
    idx = (b.astype(np.int64) << np.arange(bits)).sum(-1)
    out = cb[idx].reshape(rows, nidx * dim).astype(np.float32) * gain[:, None].astype(np.float32)
    return out


def main():
    hf, qd = Path(sys.argv[1]), Path(sys.argv[2])
    L = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    NE = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    dev = "cuda"
    tbl = torch.tensor(FP4_TABLE, dtype=torch.float32, device=dev)
    hi, qi = build_index(hf), build_index(qd)
    print(f"[索引] hf {len(hi)} / 量化目录 {len(qi)} 张量")
    bad = 0
    # ---- ① ② VQ ----
    sse = en = 0.0
    for e in range(NE):
        for m in ("w1", "w3", "w2"):
            base = f"layers.{L}.ffn.experts.{e}.{m}"
            pk = load_raw(qi, base + ".vq.idx", "cpu").numpy()
            cb = load_raw(qi, f"layers.{L}.ffn.experts.{e}.vq.cb", "cpu").numpy()
            gain = load_raw(qi, base + ".vq.gain", "cpu").numpy()
            ref = torch.from_numpy(np_decode(pk, cb, gain)).to(dev)
            got = load_vq(qi, base + ".weight", dev)
            eq = torch.equal(ref, got)
            orig = dq_fp4(load_raw(hi, base + ".weight", dev), load_raw(hi, base + ".scale", dev), tbl)
            d = ((got - orig) ** 2).sum().item(); o = (orig ** 2).sum().item()
            sse += d; en += o
            print(f"  {base}: numpy==C {'✓' if eq else '★不等★'}  shape {tuple(got.shape)}  残差 {100*d/o:.2f}%  码字使用 {len(np.unique(np_idx(pk, cb)))}/{cb.shape[0]}")
            bad += not eq
    print(f"[VQ] 前 {NE} 专家合计残差 {100*sse/en:.2f}%  cos {np.sqrt(1-sse/en):.4f}")
    # ---- ③ 骨架(FP4 或 q4_K, 按盘上登记的 dtype 分流) ----
    for n in (f"layers.{L}.attn.wq_a.weight", f"layers.{L}.attn.wo_b.weight", f"layers.{L}.ffn.shared_experts.w2.weight"):
        if n not in qi:
            print(f"  {n}: 量化目录里没有(跳过)"); continue
        o = dq_fp8(load_raw(hi, n, dev), load_raw(hi, n[:-6] + "scale", dev))
        if qi[n][2] == "Q4_K":
            rows, cols = qi[n][3]
            ref = torch.from_numpy(np_q4k(load_raw(qi, n, "cpu").numpy(), rows, cols)).to(dev)
            q = load_q4k(qi, n, dev)
            eq = torch.equal(ref, q)
            bad += not eq
            extra = f"numpy==C {'✓' if eq else '★不等★'}"
        else:
            sq = load_raw(qi, n[:-6] + "scale", dev)
            q = dq_fp4(load_raw(qi, n, dev), sq, tbl)
            n255 = int((sq == 255).sum().item())
            bad += n255 > 0
            extra = f"scale=255 槽 {n255} 个"
        d = ((q - o) ** 2).sum().item(); oo = (o ** 2).sum().item()
        print(f"  {n}: dtype {qi[n][2]} {tuple(qi[n][3])} 残差 {100*d/oo:.3f}% cos {np.sqrt(1-d/oo):.4f}  {extra}")
    print("★对拍失败★" if bad else "对拍全绿")
    sys.exit(1 if bad else 0)


def np_idx(pk, cb):
    rows, bytes_row = pk.shape
    nc, dim = cb.shape
    bits = int(np.ceil(np.log2(nc)))
    nidx = bytes_row * 8 // bits
    b = np.unpackbits(pk, axis=1, bitorder="little")[:, :nidx * bits].reshape(rows, nidx, bits)
    return (b.astype(np.int64) << np.arange(bits)).sum(-1)


if __name__ == "__main__":
    main()
