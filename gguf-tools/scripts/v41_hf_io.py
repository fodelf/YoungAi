#!/usr/bin/env python3
"""v41_hf_io.py — V4.1 HF 权重的索引/读取/解量化, 供教师前向复用(2026-09-11)。

【定位】金标/教师夹具的 IO 层, 不是数值链的一部分(全仓零 Python 数值链: 算法一律 C)。
这里只做三件事: 扫 safetensors 头建索引、按索引 memmap 零复制取张量、按 V4.1 的格式
解量化(E4M3 + ue8m0 块 scale / FP4 E2M1 两两打包)。

【为什么从 v41_teacher.py 拆出来】单文件 ≤500 行是仓规。拆点选在这里是因为 IO 层与
前向编排层本来就无耦合 —— 它只认"张量名 → 字节", 不认模型结构。

【正确性】本文件的解量化与 C 读器(gguf-tools/quantize/st_locate.h)逐位对拍过
(v41_dequant_parity.py, 四种块格式全绿)。★改这里必须重跑那个对拍★ —— V4.1 的格式
解错不会报错, 只会静默出一整套好看的假数。
"""
import json
import struct
from pathlib import Path

import numpy as np
import torch

FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ---------------- safetensors 索引 ----------------
def build_index(hf: Path):
    """扫一遍全部分片, 建 张量名 → (文件, 数据绝对偏移, dtype, shape)。
    只扫一次(96085 个张量), 之后所有加载都查这张表。
    ★有 model.safetensors.index.json 就以它的 weight_map 为准★: 量化目录(v41_quantize 产物)把
    engram 所在的出厂分片软链进来, 同目录下同名张量只认 map 指定的那个文件 —— 靠 glob 顺序
    "后扫的覆盖先扫的"是在赌文件名排序, 赌错不报错只出假数。"""
    idx = {}
    files = sorted(hf.glob("*.safetensors"))
    wm = None
    ij = hf / "model.safetensors.index.json"
    if ij.exists():
        wm = json.loads(ij.read_text())["weight_map"]
        files = sorted({hf / f for f in wm.values()})
    for p in files:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        base = 8 + n
        for name, e in hdr.items():
            if name == "__metadata__":
                continue
            if wm is not None and wm.get(name) != p.name:
                continue
            idx[name] = (p, base + e["data_offsets"][0], e["dtype"], tuple(e["shape"]))
    return idx


DT = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.uint8, "I8": torch.uint8, "U8": torch.uint8,
      "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


_MM = {}


def _mm(path):
    """每个分片一份 memmap, 复用。★不要每次 open+read★: 那样每个张量要多两次内存拷贝
    (read 出 bytes, 再 bytearray 复制一份), 一层 1152 个专家张量 × 40 层 = 46080 次,
    实撞的表现是 GPU 利用率 0% 而前向迟迟不动。"""
    if path not in _MM:
        _MM[path] = np.memmap(path, dtype=np.uint8, mode="r")
    return _MM[path]


_VIEW = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn}


def load_raw(idx, name, device):
    """按索引把一个张量读上来(不解量化)。E8M0/I8/U8 一律当 uint8 收 —— 解释权在用它的地方。"""
    p, off, dt, shp = idx[name]
    nbytes = int(np.prod(shp)) * (2 if dt in ("BF16", "F16") else 4 if dt == "F32" else 1)
    t = torch.from_numpy(_mm(p)[off:off + nbytes])
    if dt in _VIEW:
        t = t.view(_VIEW[dt])
    return t.view(*shp).to(device, non_blocking=True)


# ---------------- VQ 落盘产物(v41_quantize) ----------------
_VQLIB = None


def vq_lib():
    """libv41vq.so: 与量化器链的是同一个 v41_vq.o, 解码核只有一份(数值在 C, 这里只递指针)。"""
    global _VQLIB
    if _VQLIB is None:
        import ctypes
        so = Path(__file__).resolve().parent.parent / "quantize" / "libv41vq.so"
        _VQLIB = ctypes.CDLL(str(so))
        _VQLIB.v41_vq_decode_gpu.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        _VQLIB.v41_vq_decode_gpu.restype = ctypes.c_int
    return _VQLIB


def has_vq(idx, wname):
    return wname.endswith(".weight") and (wname[:-7] + ".vq.idx") in idx


def load_vq(idx, wname, device):
    """layers.N.ffn.experts.M.w1.weight → 用 .vq.idx / .vq.gain + 专家共享 .vq.cb 解成 f32 [rows, cols]。
    索引流逐行字节对齐 LSB 先, 位宽 = ceil(log2 nc); 列数从 (每行字节数, 位宽, dim) 反推。"""
    base = wname[:-7]
    ex = base.rsplit(".", 1)[0]
    pk = load_raw(idx, base + ".vq.idx", device).contiguous()
    gain = load_raw(idx, base + ".vq.gain", device).contiguous()
    cb = load_raw(idx, ex + ".vq.cb", device).contiguous()
    nc, dim = cb.shape
    bits = max(1, int(np.ceil(np.log2(nc))))
    rows, bytes_row = pk.shape
    cols = (bytes_row * 8 // bits) * dim
    out = torch.empty(rows, cols, dtype=torch.float32, device=device)
    rc = vq_lib().v41_vq_decode_gpu(pk.data_ptr(), cb.data_ptr(), gain.data_ptr(),
                                    rows, cols, dim, nc, out.data_ptr())
    if rc != 0:
        raise RuntimeError(f"VQ 解码失败 rc={rc} @{wname}")
    return out


# ---------------- dequant(与 C 读器逐位对拍过) ----------------
def e8m0_to_f32(b: torch.Tensor) -> torch.Tensor:
    v = torch.exp2(b.float() - 127.0)
    return torch.where(b == 255, torch.zeros_like(v), v)   # 255 = NaN 槽, 当 0 处理


def dq_fp8(w: torch.Tensor, s_u8: torch.Tensor) -> torch.Tensor:
    """E4M3 + ue8m0 块 scale。块高/宽由形状反推(专家与 engram 的块高是 1, 其余 32)。"""
    rows, cols = w.shape
    s = e8m0_to_f32(s_u8)
    bh, bw = rows // s.shape[0], cols // s.shape[1]
    s = s.repeat_interleave(bh, 0).repeat_interleave(bw, 1)
    return w.float() * s


def dq_fp4(w_u8: torch.Tensor, s_u8: torch.Tensor, tbl: torch.Tensor) -> torch.Tensor:
    """FP4 E2M1: [out, in/2] uint8, 低 nibble 偶数列 / 高 nibble 奇数列(官方 convert.py 口径)。"""
    lo = tbl[(w_u8 & 0x0F).long()]
    hi = tbl[((w_u8 >> 4) & 0x0F).long()]
    x = torch.stack([lo, hi], dim=-1).flatten(1)
    s = e8m0_to_f32(s_u8)
    return x * s.repeat_interleave(x.shape[1] // s.shape[1], 1)


