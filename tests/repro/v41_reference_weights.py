"""CPU-only GGUF weight reader for the independent official V4.1 forward.

No Metal/CUDA kernels or engine dequantizers are called. Q4_K uses the existing
NumPy reference; DQV3 is unpacked with NumPy. Only one layer/expert is expanded.
"""
import re
import struct
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[2] / 'gguf-tools/scripts'
sys.path.insert(0, str(SCRIPTS))
from v41_gguf_parity import read_gguf
from v41_file_parity import np_q4k

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
MAPPING = {
    'embed.weight': 'token_embd.weight', 'norm.weight': 'output_norm.weight',
    'head.weight': 'output.weight',
    'attn.wq_a.weight': 'attn_q_a.weight', 'attn.q_norm.weight': 'attn_q_a_norm.weight',
    'attn.wq_b.weight': 'attn_q_b.weight', 'attn.wkv.weight': 'attn_kv.weight',
    'attn.kv_norm.weight': 'attn_kv_a_norm.weight', 'attn.attn_sink': 'attn_sinks.weight',
    'attn.wo_a.weight': 'attn_output_a.weight', 'attn.wo_b.weight': 'attn_output_b.weight',
    'attn.compressor.wkv.weight': 'attn_compressor_kv.weight',
    'attn.compressor.wgate.weight': 'attn_compressor_gate.weight',
    'attn.compressor.norm.weight': 'attn_compressor_norm.weight',
    'attn.indexer.wk.weight': 'indexer.wk.weight',
    'attn.indexer.k_norm.weight': 'indexer.k_norm.weight',
    'attn.indexer.wq_b.weight': 'indexer.attn_q_b.weight',
    'attn.indexer.weights_proj.weight': 'indexer.proj.weight',
    'ffn.gate.weight': 'ffn_gate_inp.weight', 'ffn.gate.bias': 'exp_probs_b.bias',
    'ffn.shared_experts.w1.weight': 'ffn_gate_shexp.weight',
    'ffn.shared_experts.w3.weight': 'ffn_up_shexp.weight',
    'ffn.shared_experts.w2.weight': 'ffn_down_shexp.weight',
    'engram.wkv.weight': 'engram_wkv.weight',
    'engram.q_weight': 'engram_q.weight', 'engram.k_weight': 'engram_k.weight',
}


def fp4_blocks(raw):
    b = np.asarray(raw).reshape(-1, 17)
    q = np.empty((len(b), 32), np.uint8)
    q[:, 0::2], q[:, 1::2] = b[:, :16] & 15, b[:, :16] >> 4
    assert np.all(b[:, 16] != 255), 'invalid FP4 scale'
    return FP4[q] * np.exp2(b[:, 16].astype(np.float32) - 127)[:, None]


class Weights:
    def __init__(self, path):
        self.path = Path(path)
        self.kv, self.tensors, self.data0, self.mm = read_gguf(path)

    def raw(self, off, count):
        if off < 0 or count < 0 or off + count > len(self.mm):
            raise ValueError('weight range outside file')
        return np.asarray(self.mm[off:off + count])

    def discard_pages(self):
        # Expanded parameters own their storage. macOS may retain RSS after
        # MADV_DONTNEED, so unmap between layers and open a fresh read-only view.
        # No returned tensor aliases this mapping; the file is never modified.
        self.mm._mmap.close()
        self.mm = np.memmap(self.path, dtype=np.uint8, mode='r')

    def name(self, full):
        if full in MAPPING:
            return MAPPING[full]
        m = re.fullmatch(r'layers\.(\d+)\.(.+)', full)
        if not m:
            raise KeyError(full)
        layer, tail = m.groups()
        if tail.startswith('hc_'):
            tail += '.weight'
        return f'blk.{layer}.' + MAPPING.get(tail, tail)

    def dense(self, name):
        ty, ne, off = self.tensors[name]
        shape = tuple(reversed(ne)); n = int(np.prod(shape)); off += self.data0
        if ty == 12:
            rows, cols = shape
            value = np_q4k(self.raw(off, n // 256 * 144), rows, cols)
        elif ty == 44:
            rows, cols = shape
            q = torch.from_numpy(self.raw(off, n).copy()).view(torch.float8_e4m3fn).float().numpy().reshape(rows, cols)
            sr, sc = (rows + 31) // 32, (cols + 31) // 32
            scale = np.exp2(self.raw(off + n, sr * sc).astype(np.float32) - 127).reshape(sr, sc)
            value = q * scale.repeat(32, 0).repeat(32, 1)[:rows, :cols]
        elif ty == 30:
            value = (self.raw(off, n * 2).view('<u2').astype(np.uint32) << 16).view(np.float32).reshape(shape)
        elif ty in (0, 1, 26, 27):
            dtype = {0: '<f4', 1: '<f2', 26: '<i4', 27: '<i8'}[ty]
            value = self.raw(off, n * np.dtype(dtype).itemsize).view(dtype).reshape(shape).copy()
        else:
            raise ValueError(f'unsupported type {ty}: {name}')
        return torch.from_numpy(value), ty in (12, 44)

    def expert(self, layer, expert, which):
        ty, ne, off = self.tensors[f'blk.{layer}.ffn_exps_vq.blob']
        assert ty == 42
        b = self.raw(self.data0 + off, ne[0])
        magic, ver, lid, count = struct.unpack_from('<4I', b)
        assert (magic, ver, lid) == (0x4C565144, 3, layer) and 0 <= expert < count
        pos = struct.unpack_from('<Q', b, 16 + (expert * 3 + which) * 8)[0]
        magic, dim, nc, rows, cols, flags, bits, cb_off = struct.unpack_from('<IHHIIIIQ', b, pos)
        assert magic == 0x33565144 and dim == 8 and bits == 12 and flags in (1, 3)
        assert nc == (8192 if flags & 2 else 4096) and cols % dim == 0
        cb = torch.from_numpy(b[cb_off:cb_off + nc * dim].copy()).view(torch.float8_e4m3fn).float().numpy().reshape(nc, dim)
        pos += 32
        gain = b[pos:pos + rows * 2].view('<f2').astype(np.float32); pos += rows * 2
        ni = cols // dim; main_bytes = rows * ni * 12 // 8
        packed = b[pos:pos + main_bytes].reshape(rows, -1, 3).astype(np.uint16)
        ids = np.empty((rows, ni), np.uint16)
        ids[:, 0::2] = packed[:, :, 0] | ((packed[:, :, 1] & 15) << 8)
        ids[:, 1::2] = (packed[:, :, 1] >> 4) | (packed[:, :, 2] << 4)
        if flags & 2:
            ext = b[pos + main_bytes:pos + main_bytes + rows * ((ni + 7) // 8)].reshape(rows, -1)
            ids |= np.unpackbits(ext, axis=1, bitorder='little')[:, :ni].astype(np.uint16) << 12
        return torch.from_numpy(cb[ids].reshape(rows, cols) * gain[:, None]), True

    def get(self, full):
        m = re.fullmatch(r'layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.weight', full)
        if m:
            layer, expert, w = map(int, m.groups())
            return self.expert(layer, expert, {1: 0, 3: 1, 2: 2}[w])
        return self.dense(self.name(full))


def sidecar(path):
    b = Path(path).read_bytes()
    rows, cols, ty = struct.unpack_from('<iii', b)
    n = rows * cols
    if ty == 43:
        value = fp4_blocks(np.frombuffer(b, np.uint8, offset=12)).reshape(-1)
    elif ty in (1, 2):
        value = np.frombuffer(b, '<f4' if ty == 1 else '<f2', offset=12).astype(np.float32)
    else:
        raise ValueError(f'unsupported sidecar type {ty}: {path}')
    assert len(value) % n == 0
    return torch.from_numpy(value.copy().reshape(-1, rows, cols)), ty
