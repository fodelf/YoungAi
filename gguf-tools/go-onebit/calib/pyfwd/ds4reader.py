"""Memory-safe streaming reader for DeepSeek-V4-Flash fp8 safetensors over NFS.
No torch. numpy only. Reads ONE tensor (or row-slice) at a time via byte offsets.
e4m3fn weights * f32 block-scale (128x128). RSS bounded.
"""
import json, struct, mmap, os
import numpy as np

HF = os.environ.get("DS4_HF", "/private/tmp/m1_ds4/hf/DeepSeek-V4-Flash-Base")
IDX = os.path.join(HF, "model.safetensors.index.json")

def e4m3_lut():
    out = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = (b >> 7) & 1; e = (b >> 3) & 0xF; m = b & 0x7
        sign = -1.0 if s else 1.0
        if e == 0:
            val = sign * (m / 8.0) * (2.0 ** -6)
        elif e == 0xF and m == 0x7:
            val = np.nan
        else:
            val = sign * (1.0 + m / 8.0) * (2.0 ** (e - 7))
        out[b] = val
    return out
LUT = e4m3_lut()

_weight_map = None
_hdr_cache = {}   # shard filename -> (header_dict, data_start)

def _wm():
    global _weight_map
    if _weight_map is None:
        _weight_map = json.load(open(IDX))["weight_map"]
    return _weight_map

def _shard_header(shard):
    if shard not in _hdr_cache:
        path = os.path.join(HF, shard)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        _hdr_cache[shard] = (hdr, 8 + n)
    return _hdr_cache[shard]

def raw_bytes(name, row0=None, row1=None):
    """Return raw fp8 bytes for tensor `name`, optionally rows [row0:row1)."""
    shard = _wm()[name]
    hdr, data_start = _shard_header(shard)
    meta = hdr[name]
    shape = meta["shape"]; dt = meta["dtype"]
    o0, o1 = meta["data_offsets"]
    path = os.path.join(HF, shard)
    if dt == "F8_E4M3":
        rowbytes = shape[1]
        if row0 is None: row0 = 0
        if row1 is None: row1 = shape[0]
        start = data_start + o0 + row0 * rowbytes
        nbytes = (row1 - row0) * rowbytes
        with open(path, "rb") as f:
            f.seek(start); buf = f.read(nbytes)
        a = np.frombuffer(buf, dtype=np.uint8).reshape(row1 - row0, rowbytes)
        return a, shape
    else:
        with open(path, "rb") as f:
            f.seek(data_start + o0); buf = f.read(o1 - o0)
        npdt = {"F32": np.float32, "BF16": np.uint16, "I64": np.int64}[dt]
        a = np.frombuffer(buf, dtype=npdt)
        if dt == "BF16":
            a = (a.astype(np.uint32) << 16).view(np.float32)
        return a.reshape(shape), shape

def get(name):
    """Full dequantized f32 numpy array for any tensor (F8_E4M3 with block-scale, BF16, F32, I64)."""
    shard = _wm()[name]
    hdr, _ = _shard_header(shard)
    if hdr[name]["dtype"] == "F8_E4M3":
        return read_weight(name)
    a, _ = raw_bytes(name)
    return a

def read_weight(name, scale_name=None, row0=None, row1=None):
    """Dequantized f32 weight (optionally row slice). scale_name defaults name->.scale."""
    a, shape = raw_bytes(name, row0, row1)             # uint8 [R,C]
    w = LUT[a]                                          # f32 [R,C]
    if scale_name is None:
        scale_name = name.replace(".weight", ".scale")
    sc, _ = raw_bytes(scale_name)                       # f32 [R/128, C/128]
    R, C = w.shape
    r_off = (row0 or 0) // 128
    nbr = (R + 127) // 128
    # expand scale block to per-element for the rows we read
    sc_rows = sc[r_off:r_off + nbr]                     # [nbr, C/128]
    sc_full = np.repeat(np.repeat(sc_rows, 128, axis=0), 128, axis=1)[:R, :C]
    return w * sc_full
