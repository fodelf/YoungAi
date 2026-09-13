#!/usr/bin/env python3
"""v41_dequant_parity.py — torch 解权重 vs 本仓 C 读器解权重, 逐位对拍(2026-09-11)。

【定位, 别搞混】本文件是**金标夹具生成器**, 不是数值链的一部分(全仓零 Python 裁决:
算法一律 C 实现, Python 只做不参与数值的编排与金标)。它存在的唯一理由是: V4.1 的
权重格式(FP4 E2M1 两两打包 + ue8m0 scale, routed 专家 1×32 块 / 其余 32×32 块)如果
解错, 不会报错, 只会静默产出一整套好看的假数 —— 所以 C 读器(st_locate.h)必须有一个
独立实现来对拍, torch 的官方 dtype 就是这个独立实现。

【判据】逐元素完全相等(两边都是精确的查表 × 2 的幂缩放, 不存在浮点舍入差)。
差一个元素都算失败 —— 不接受"差不多"。

用法: v41_dequant_parity.py <hf-dir> <C端dump目录>
      C 端 dump 由 st_dump_tensor 产生(同名 .f32 裸浮点)
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def read_header(path: Path):
    """safetensors: [8B u64 LE header_len][JSON][data]。返回 (json, data_start)。"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def find(hf: Path, name: str):
    for p in sorted(hf.glob("*.safetensors")):
        hdr, start = read_header(p)
        if name in hdr:
            e = hdr[name]
            return p, start + e["data_offsets"][0], e["dtype"], e["shape"]
    raise KeyError(name)


def raw(path: Path, off: int, nbytes: int) -> torch.Tensor:
    with open(path, "rb") as f:
        f.seek(off)
        return torch.frombuffer(bytearray(f.read(nbytes)), dtype=torch.uint8)


def e8m0(b: torch.Tensor) -> torch.Tensor:
    """ue8m0: 无符号纯指数, 值 = 2^(b-127); 255 是 NaN。"""
    v = torch.exp2(b.float() - 127.0)
    return torch.where(b == 255, torch.zeros_like(v), v)


def dequant_fp8(hf: Path, name: str, r0: int, nr: int) -> torch.Tensor:
    """E4M3 权重 + ue8m0 块 scale。★只读要对拍的那几行★ —— engram 的 embed 是 98 GB,
    整张读进来会把 121 GB 的机器吃干(实撞过一次)。块高由 scale 行数反推: 专家/engram 是 1,
    attn 等是 32。"""
    p, off, dt, shp = find(hf, name)
    assert dt == "F8_E4M3", dt
    rows, cols = shp
    w = raw(p, off + r0 * cols, nr * cols).view(torch.float8_e4m3fn).view(nr, cols).float()
    ps, so, sdt, sshp = find(hf, name.rsplit(".", 1)[0] + ".scale")
    assert sdt == "F8_E8M0", sdt
    bh, bw = rows // sshp[0], cols // sshp[1]
    sr0, sr1 = r0 // bh, (r0 + nr - 1) // bh + 1
    s = e8m0(raw(ps, so + sr0 * sshp[1], (sr1 - sr0) * sshp[1]).view(sr1 - sr0, sshp[1]))
    s = s.repeat_interleave(bh, 0).repeat_interleave(bw, 1)
    return w * s[r0 - sr0 * bh : r0 - sr0 * bh + nr]


def dequant_fp4(hf: Path, name: str, r0: int, nr: int) -> torch.Tensor:
    """FP4 E2M1: [rows, cols/2] uint8, 低 nibble = 偶数列, 高 nibble = 奇数列。
    scale 形状 [rows, cols/32] —— routed 专家的块是 1 行 × 32 列, 官方 convert.py 的
    assert 写死了这一点。同样只读要的行。"""
    p, off, dt, shp = find(hf, name)
    assert dt == "I8", dt
    rows, packed = shp
    cols = packed * 2
    b = raw(p, off + r0 * packed, nr * packed).view(nr, packed)
    lo = FP4_TABLE[(b & 0x0F).long()]
    hi = FP4_TABLE[((b >> 4) & 0x0F).long()]
    w = torch.stack([lo, hi], dim=-1).flatten(1)
    ps, so, sdt, sshp = find(hf, name.rsplit(".", 1)[0] + ".scale")
    assert sshp[0] == rows, f"专家 scale 应为每行一份, 实为 {sshp}"
    s = e8m0(raw(ps, so + r0 * sshp[1], nr * sshp[1]).view(nr, sshp[1]))
    s = s.repeat_interleave(cols // sshp[1], 1)
    return w * s


def main():
    hf = Path(sys.argv[1])
    dumpdir = Path(sys.argv[2])
    cases = [
        ("layers.0.attn.wq_a.weight", dequant_fp8, "fp8 32x32块"),
        ("layers.1.engram.embed.weight", dequant_fp8, "fp8 每行8块(engram)"),
        ("layers.20.ffn.experts.0.w1.weight", dequant_fp4, "fp4 每行32列块(专家)"),
        ("layers.20.ffn.experts.0.w2.weight", dequant_fp4, "fp4 w2"),
    ]
    bad = 0
    for name, fn, tag in cases:
        cf = dumpdir / (name.replace("/", "_") + ".f32")
        if not cf.exists():
            print(f"  {name}: ★C 端 dump 缺 {cf}★")
            bad += 1
            continue
        meta = json.loads((dumpdir / (name.replace("/", "_") + ".json")).read_text())
        r0, nr, cols = meta["r0"], meta["nr"], meta["cols"]
        c = torch.from_numpy(np.fromfile(cf, dtype=np.float32)).view(nr, cols)
        t = fn(hf, name, r0, nr)[:, :cols].float()
        same = torch.equal(c, t)
        if same:
            print(f"  ✓ {name} ({tag}) {nr}x{cols} 逐位相等")
        else:
            d = (c - t).abs()
            nz = int((d > 0).sum())
            print(f"  ★不等★ {name} ({tag}): {nz}/{d.numel()} 个元素不同, 最大差 {d.max():.6g}")
            print(f"       C端前6: {c[0,:6].tolist()}")
            print(f"       torch前6: {t[0,:6].tolist()}")
            bad += 1
    print("★全部逐位一致★" if bad == 0 else f"★{bad} 项不一致★")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
