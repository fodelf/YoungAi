#!/usr/bin/env python3
"""scale_apply.py — 杠杆① 第2段 (M4): 用 scale_compute 表就地 patch mono
GO1B (type40) 单 scale / GO2B (type41) 双 scale (d1,d2)。sign/码不动, 体积不变。

块布局:
  GO1B (34B): scale(2) + sign(32)。每行 16 块全写同一 per-row act scale。
  GO2B (68B): d1(2) + d2(2) + b1(32) + b2(32)。每行 16 块全写同一 (d1,d2)。
cov=False 的专家保留原值。

用法(M4): python3 scale_apply.py --gguf mono.gguf --scales /tmp/scales \
            --layers 0-42 [--dry-run]
"""
import argparse
import os
import struct
import sys

import numpy as np

D = 4096
BLK_QK = 256


def gguf_table(path):
    f = open(path, "rb")
    struct.unpack("<II", f.read(8))
    nt, nk = struct.unpack("<QQ", f.read(16))

    def rstr():
        n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")

    def rval(t):
        if t == 8: rstr()
        elif t in (0, 1, 7): f.read(1)
        elif t in (2, 3): f.read(2)
        elif t in (4, 5, 6): f.read(4)
        elif t in (10, 11, 12): f.read(8)
        elif t == 9:
            et, = struct.unpack("<I", f.read(4)); n, = struct.unpack("<Q", f.read(8))
            for _ in range(n): rval(et)

    align = 32
    for _ in range(nk):
        k = rstr(); t, = struct.unpack("<I", f.read(4))
        if k == "general.alignment" and t == 4:
            align, = struct.unpack("<I", f.read(4))
        else:
            rval(t)
    meta = {}
    for _ in range(nt):
        name = rstr(); nd, = struct.unpack("<I", f.read(4))
        ne = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        typ, = struct.unpack("<I", f.read(4)); off, = struct.unpack("<Q", f.read(8))
        meta[name] = (ne, typ, off)
    base = f.tell(); base = (base + align - 1) // align * align
    f.close()
    return meta, base


def fp16(v):
    return np.frombuffer(v.astype("<f2").tobytes(), dtype="<u2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--scales", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    layers = []
    for part in a.layers.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-")); layers += list(range(lo, hi + 1))
        else:
            layers.append(int(part))
    meta, base = gguf_table(a.gguf)
    nblk = D // BLK_QK
    gg = None if a.dry_run else open(a.gguf, "r+b")
    total = 0
    for L in layers:
        sp = f"{a.scales}/L{L}.npz"
        if not os.path.isfile(sp):
            continue
        z = np.load(sp)
        cov = z["cov"]
        gtype = int(z["type"]) if "type" in z else 40
        blkb = 34 if gtype == 40 else 68
        row_bytes = nblk * blkb
        for mtag, tname in (("gate", f"blk.{L}.ffn_gate_exps.weight"),
                            ("up", f"blk.{L}.ffn_up_exps.weight"),
                            ("down", f"blk.{L}.ffn_down_exps.weight")):
            if tname not in meta or meta[tname][1] != gtype:
                print(f"L{L} {mtag}: type 不符跳过", file=sys.stderr); continue
            ne, typ, off = meta[tname]
            ncols, nrows, nexp = ne
            expert_bytes = nrows * row_bytes
            npatch = 0
            for e in range(nexp):
                if not cov[e]:
                    continue
                if gtype == 40:
                    hd = fp16(z[mtag][e])                     # [nrows]
                else:
                    h1 = fp16(z[f"{mtag}_d1"][e]); h2 = fp16(z[f"{mtag}_d2"][e])
                if a.dry_run:
                    npatch += nrows; continue
                exp_off = base + off + e * expert_bytes
                gg.seek(exp_off)
                raw = bytearray(gg.read(expert_bytes))
                for r in range(nrows):
                    rb = r * row_bytes
                    for b in range(nblk):
                        bo = rb + b * blkb
                        if gtype == 40:
                            raw[bo:bo+2] = struct.pack("<H", int(hd[r]))
                        else:
                            raw[bo:bo+2] = struct.pack("<H", int(h1[r]))
                            raw[bo+2:bo+4] = struct.pack("<H", int(h2[r]))
                gg.seek(exp_off); gg.write(raw)
                npatch += nrows
            total += npatch
            print(f"L{L} {mtag} type{gtype}: {'DRY ' if a.dry_run else ''}{npatch} rows "
                  f"({int(cov.sum())}/{nexp} exp)", file=sys.stderr, flush=True)
    if gg: gg.close()
    print(f"scale_apply done: {total} rows")


if __name__ == "__main__":
    main()
