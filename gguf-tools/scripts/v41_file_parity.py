#!/usr/bin/env python3
"""v41_file_parity.py — 落盘量化模型的读回对拍(2026-09-12)。金标夹具, 不是数值链: numpy 那份
解包只用来当"笨参照", 生产解码在 libv41vq.so(与量化器同一个核)。

① VQ 三件(.vq.idx / .vq.cb / .vq.gain): numpy 逐位解包+查表  vs  C 核解码 → 必须逐元素相等
   (相等 = 位流布局/位宽/f16 口径三者都对; 差一位就整行错位, 而错位不报错只出假数)
② C 解码值 vs HF 出厂 FP4 dequant → 残差, 与量化器日志报的 cos 对账
③ 骨架 FP4: 重量化张量 dequant vs 出厂 FP8 dequant → 残差; scale 字节不得是 255(NaN 槽)

用法: v41_file_parity.py <hf-dir> <quant-dir> [层号=0] [对拍专家数=4]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v41_hf_io import FP4_TABLE, build_index, load_raw, dq_fp4, dq_fp8, load_vq


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
    # ---- ③ 骨架 FP4 ----
    for n in (f"layers.{L}.attn.wq_a.weight", f"layers.{L}.attn.wo_b.weight", f"layers.{L}.ffn.shared_experts.w2.weight"):
        if n not in qi:
            print(f"  {n}: 量化目录里没有(跳过)"); continue
        sq = load_raw(qi, n[:-6] + "scale", dev)
        q = dq_fp4(load_raw(qi, n, dev), sq, tbl)
        o = dq_fp8(load_raw(hi, n, dev), load_raw(hi, n[:-6] + "scale", dev))
        d = ((q - o) ** 2).sum().item(); oo = (o ** 2).sum().item()
        n255 = int((sq == 255).sum().item())
        print(f"  {n}: dtype {qi[n][2]} {tuple(qi[n][3])} 残差 {100*d/oo:.3f}% cos {np.sqrt(1-d/oo):.4f}  scale=255 槽 {n255} 个")
        bad += n255 > 0
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
