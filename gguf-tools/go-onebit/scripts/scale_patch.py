#!/usr/bin/env python3
"""scale_patch.py — 杠杆① 就地重标: mono GO1B_BLK 每行 scale ← 完整Gram激活感知。

在 M1 跑 (HF fp8 权重本地)。对每个 GO1B 层的每个专家 w1/w3/w2:
  s_row = Σ_x (w_row·x)(b_row·x) / Σ_x (b_row·x)²   (b=sign(w), x=该层捕获激活)
把 s_row 写进该行 16 个 256-块的 fp16 scale 字段 (全设同值=per-row); sign 位不动。
体积字节不变。w1/w3 的 x=层输入激活; w2 的 x=该量化下的中间 h=silu(gate)*up。

GO1B_BLK 行布局 (onebit_quant.h): 每行 = nblk × 34 字节; 块 = fp16 scale(2) + 32B sign。
张量 ne=(ncols, nrows, n_expert), 行主, 专家外层。

用法 (M1): DS4_HF=.../DeepSeek-V4-Flash-Base python3 scale_patch.py \
             --gguf mono.gguf --cap /tmp/capall --layers 0-22,35,37,39,41,42 [--ntok 1024]
"""
import argparse
import os
import struct
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "calib", "pyfwd"), os.path.join(_here, "pyfwd")):
    if os.path.isfile(os.path.join(_cand, "dsv4_fwd.py")):
        sys.path.insert(0, _cand)
        break
import dsv4_fwd as F   # noqa: E402

D = 4096
BLK_QK = 256
BLK_BYTES = 34
NACT = 6


def f32_to_f16_bits(x):
    return np.frombuffer(np.float32(x).astype("<f4").tobytes(), dtype="<u4")[0]  # placeholder


def gguf_tensor_table(path):
    """返回 {name: (ne, type, abs_offset)} + 数据段基址 (对齐后)。"""
    f = open(path, "rb")
    struct.unpack("<II", f.read(8))
    nt, nk = struct.unpack("<QQ", f.read(16))

    def rstr():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def rval(t):
        if t == 8: rstr()
        elif t in (0, 1, 7): f.read(1)
        elif t in (2, 3): f.read(2)
        elif t in (4, 5, 6): f.read(4)
        elif t in (10, 11, 12): f.read(8)
        elif t == 9:
            et, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            for _ in range(n): rval(et)

    align = 32
    kvpos = f.tell()
    # 读 alignment
    for _ in range(nk):
        k = rstr()
        t, = struct.unpack("<I", f.read(4))
        if k == "general.alignment" and t == 4:
            align, = struct.unpack("<I", f.read(4))
        else:
            rval(t)
    meta = {}
    for _ in range(nt):
        name = rstr()
        nd, = struct.unpack("<I", f.read(4))
        ne = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        typ, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        meta[name] = (ne, typ, off)
    base = f.tell()
    base = (base + align - 1) // align * align
    f.close()
    return meta, base


def act_scale_rows(W, X):
    """完整 Gram per-row scale, 返回 [nrows] float32。"""
    S = np.sign(W).astype(np.float32)
    S[S == 0] = 1.0
    Pt = X @ W.T
    Ps = X @ S.T
    num = (Pt * Ps).sum(axis=0)
    den = (Ps * Ps).sum(axis=0)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.abs(W).mean(axis=1)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--ntok", type=int, default=1024)
    a = ap.parse_args()

    layers = []
    for part in a.layers.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-")); layers += list(range(lo, hi + 1))
        else:
            layers.append(int(part))

    meta, base = gguf_tensor_table(a.gguf)
    nblk = D // BLK_QK
    row_bytes = nblk * BLK_BYTES
    gg = open(a.gguf, "r+b")

    for L in layers:
        gname = f"blk.{L}.ffn_gate_exps.weight"
        if gname not in meta or meta[gname][1] != 40:   # 只 GO1B (type 40)
            print(f"L{L}: 非 GO1B, 跳过", file=sys.stderr); continue
        xin = f"{a.cap}/raw_ffn_in_L{L}"
        rin = f"{a.cap}/raw_route_L{L}"
        if not (os.path.isfile(xin) and os.path.isfile(rin)):
            print(f"L{L}: 缺激活, 跳过", file=sys.stderr); continue
        X = np.fromfile(xin, dtype="<f2").reshape(-1, D).astype(np.float32)
        ids = np.fromfile(rin, dtype="<i2").reshape(-1, NACT)
        ok = np.abs(X).max(axis=1) > 0
        X, ids = X[ok], ids[ok]
        if len(X) > a.ntok:
            X, ids = X[:a.ntok], ids[:a.ntok]

        for mtag, tname in (("w1", f"blk.{L}.ffn_gate_exps.weight"),
                            ("w3", f"blk.{L}.ffn_up_exps.weight"),
                            ("w2", f"blk.{L}.ffn_down_exps.weight")):
            ne, typ, off = meta[tname]
            ncols, nrows, nexp = ne
            expert_bytes = nrows * row_bytes
            patched_rows = 0
            for e in range(nexp):
                sel = (ids == e).any(axis=1)
                Xe = X[sel]
                if len(Xe) < 8:
                    continue                       # 该专家未在校准里触发 → 保留原 scale
                w1 = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight")
                w3 = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight")
                if mtag == "w1":
                    W, Xu = w1, Xe
                elif mtag == "w3":
                    W, Xu = w3, Xe
                else:
                    s1 = np.sign(w1).astype(np.float32); s1[s1 == 0] = 1
                    s3 = np.sign(w3).astype(np.float32); s3[s3 == 0] = 1
                    a1 = act_scale_rows(w1, Xe); a3 = act_scale_rows(w3, Xe)
                    gate = Xe @ (s1 * a1[:, None]).T; up = Xe @ (s3 * a3[:, None]).T
                    Xu = (gate / (1.0 + np.exp(-gate))) * up
                    W = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight")
                s_row = act_scale_rows(W, Xu)                    # [nrows]
                hd = np.frombuffer(s_row.astype("<f2").tobytes(), dtype="<u2")  # fp16 每行
                # 写: 专家 e 的每行的 16 个块 scale ← hd[row]
                buf = bytearray(nblk * 2 * nrows)
                for r in range(nrows):
                    two = struct.pack("<H", int(hd[r]))
                    buf[r*nblk*2:(r+1)*nblk*2] = two * nblk
                # 逐行写块 scale (块内 scale 在偏移 0, sign 在 2..34)
                exp_off = base + off + e * expert_bytes
                # 高效: 读整个专家块, 改 scale, 写回
                gg.seek(exp_off)
                raw = bytearray(gg.read(expert_bytes))
                for r in range(nrows):
                    for b in range(nblk):
                        pos = r * row_bytes + b * BLK_BYTES
                        raw[pos:pos+2] = struct.pack("<H", int(hd[r]))
                gg.seek(exp_off)
                gg.write(raw)
                patched_rows += nrows
            print(f"L{L} {mtag}: patched {patched_rows} rows", file=sys.stderr, flush=True)
    gg.close()
    print("scale_patch done")


if __name__ == "__main__":
    main()
