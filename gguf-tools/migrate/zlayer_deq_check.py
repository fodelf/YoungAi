#!/usr/bin/env python3
"""zlayer_deq_check.py — calib/zlayer 的 GGUF 标量 dequant 对 gguf-py 逐位对拍。

为什么单独一支: DS4_ZL_GGUF 模式下学生权重是从 GGUF 专家张量切片再反量化的, .py 走
gguf-py 的向量化路, C 走标量循环。这里造合法块喂给 `zlayer --selftest-deq`, 与
gguf.quants.dequantize 的结果比【逐位】—— 数值表抄错了这里立刻会炸(踩过一次: iq2_xxs
把编码器的搜索网格 {1,3,5,7} 当成了解码网格 {0x08,0x19,0x2b}, max|Δ| 直接 15)。

块里的量化位随机, 但 f16 尺度必须是合法数值 —— 纯随机 16 位会造出 NaN/Inf, 两边都是
nan 就什么也没验到。

用法(需要 gguf-py; 不想污染系统环境就装到别处再指 PYTHONPATH):
    pip3 install --target /tmp/pylib gguf
    PYTHONPATH=/tmp/pylib python3 zlayer_deq_check.py
退出码 0 = 七种类型全部逐位相同。
"""
import os, subprocess, sys
import numpy as np
from gguf.constants import GGMLQuantizationType as T
from gguf.quants import dequantize

ZL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib", "zlayer")
NB = 64
rng = np.random.default_rng(20260825)


def f16b(n):
    """合法的 f16 尺度(不造 NaN/Inf)"""
    return (rng.standard_normal(n).astype(np.float32) * 0.05).astype(np.float16).tobytes()


def u8(n):
    return rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()


def blocks(ty):
    out = b""
    for _ in range(NB):
        if ty == T.Q2_K:        out += u8(16) + u8(64) + f16b(1) + f16b(1)
        elif ty == T.Q4_K:      out += f16b(1) + f16b(1) + u8(12) + u8(128)
        elif ty == T.Q8_0:      out += f16b(1) + u8(32)
        elif ty == T.IQ2_XXS:   out += f16b(1) + u8(64)
        elif ty == T.F16:       out += f16b(1)
        elif ty == T.BF16:      out += (rng.standard_normal(1).astype(np.float32)
                                        .view(np.uint32) >> 16).astype(np.uint16).tobytes()
        elif ty == T.F32:       out += rng.standard_normal(1).astype(np.float32).tobytes()
    return np.frombuffer(out, dtype=np.uint8)


def main():
    bad = 0
    for ty in (T.Q2_K, T.Q4_K, T.Q8_0, T.IQ2_XXS, T.F16, T.BF16, T.F32):
        raw = blocks(ty)
        ref = dequantize(raw, ty).astype(np.float32).reshape(-1)
        got = subprocess.run([ZL, "--selftest-deq", str(int(ty)), str(NB)],
                             input=raw.tobytes(), capture_output=True, check=True).stdout
        got = np.frombuffer(got, dtype=np.float32)
        if got.shape != ref.shape:
            print(f"{ty.name}: 形状 {got.shape} vs {ref.shape}")
            bad += 1
            continue
        same = np.array_equal(got.view(np.uint32), ref.view(np.uint32))
        mx = float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))))
        print(f"{ty.name:9s} n={ref.size:6d} 逐位相同={same}  max|Δ|={mx:.3e}")
        if not same:
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
