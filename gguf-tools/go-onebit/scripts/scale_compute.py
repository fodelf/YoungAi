#!/usr/bin/env python3
"""scale_compute.py — 杠杆① 第1段 (M1): 算全 GO1B 层 per-row 激活感知 scale 表。

★关键: b = mono 存的 EF sign (非 sign(fp8))。mono 是 error-feedback 量化, sign
翻转部分, 与 sign(fp8) 仅 88.94% 匹配。scale 必须对 mono 实际用的 sign 优化:
  s_row = Σ_x (w_row·x)(b_mono·x) / Σ_x (b_mono·x)²   (w=fp8, b_mono=mono存的sign)
这样对 mono 真实量化是严格改进 (L20 实测 0.9439→0.9953)。scale_apply 只改 scale
字节留 sign, 天然一致。

用法(M1): DS4_HF=.../DeepSeek-V4-Flash-Base python3 scale_compute.py \
  --gguf .../ds4-mono-mixed.gguf --cap /tmp/capall --layers 0-22,35,37,39,41,42 --out /tmp/scales
"""
import argparse
import os
import struct
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "calib", "pyfwd"), os.path.join(_here, "pyfwd")):
    if os.path.isfile(os.path.join(_cand, "dsv4_fwd.py")):
        sys.path.insert(0, _cand); break
import dsv4_fwd as F   # noqa: E402

D = 4096
NACT = 6
BLK = 256
BLKB = 34


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


GO2B_BLKB = 68


def mono_signs(mm, meta, base, tname, e):
    """GO1B (type 40): 每元素 1 sign 位, 块 34B (2 scale + 32 sign)。"""
    ne, typ, off = meta[tname]
    ncols, nrows, nexp = ne
    nblk = ncols // BLK
    rb = nblk * BLKB
    eoff = base + off + e * nrows * rb
    S = np.empty((nrows, ncols), np.float32)
    for r in range(nrows):
        ro = eoff + r * rb
        for b in range(nblk):
            sg = mm[ro + b * BLKB + 2: ro + b * BLKB + 34]
            bits = np.unpackbits(np.frombuffer(sg.tobytes(), dtype=np.uint8), bitorder="little")
            S[r, b * BLK:(b + 1) * BLK] = np.where(bits[:BLK] == 1, 1.0, -1.0)
    return S


def mono_go2b_codes(mm, meta, base, tname, e):
    """GO2B (type 41): 每元素 2 sign 位, 块 68B = d1(2)+d2(2)+b1(32)+b2(32)。
    返回 (B1, B2) ∈ {-1,+1}^[nrows,ncols]。"""
    ne, typ, off = meta[tname]
    ncols, nrows, nexp = ne
    nblk = ncols // BLK
    rb = nblk * GO2B_BLKB
    eoff = base + off + e * nrows * rb
    raw = np.frombuffer(mm[eoff:eoff + nrows * rb].tobytes(), dtype=np.uint8).reshape(nrows, nblk, GO2B_BLKB)
    b1 = np.unpackbits(raw[:, :, 4:36], axis=2, bitorder="little").reshape(nrows, nblk, BLK)
    b2 = np.unpackbits(raw[:, :, 36:68], axis=2, bitorder="little").reshape(nrows, nblk, BLK)
    B1 = np.where(b1 == 1, 1.0, -1.0).reshape(nrows, ncols).astype(np.float32)
    B2 = np.where(b2 == 1, 1.0, -1.0).reshape(nrows, ncols).astype(np.float32)
    return B1, B2


def act_b(W, S, X):
    """GO1B act scale for GIVEN signs S: s=(w·x)(b·x)/(b·x)² over samples."""
    Pt = X @ W.T; Ps = X @ S.T
    num = (Pt * Ps).sum(0); den = (Ps * Ps).sum(0)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.abs(W).mean(1)).astype(np.float32)


def act_d1d2(W, B1, B2, X):
    """GO2B per-row 2×2 解: [Σp1² Σp1p2; Σp1p2 Σp2²][d1;d2]=[Σt·p1;Σt·p2]."""
    P1 = X @ B1.T; P2 = X @ B2.T; T = X @ W.T
    a = (P1 * P1).sum(0); b = (P1 * P2).sum(0); c = (P2 * P2).sum(0)
    r1 = (T * P1).sum(0); r2 = (T * P2).sum(0)
    det = a * c - b * b
    det = np.where(np.abs(det) < 1e-9, 1e-9, det)
    d1 = (c * r1 - b * r2) / det
    d2 = (a * r2 - b * r1) / det
    return d1.astype(np.float32), d2.astype(np.float32)


def main():
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ntok", type=int, default=1024)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    layers = []
    for part in a.layers.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-")); layers += list(range(lo, hi + 1))
        else:
            layers.append(int(part))
    meta, base = gguf_table(a.gguf)
    mm = np.memmap(a.gguf, dtype=np.uint8, mode="r")

    for L in layers:
        gname = f"blk.{L}.ffn_gate_exps.weight"
        if gname not in meta or meta[gname][1] not in (40, 41):
            print(f"L{L}: 非GO1B/GO2B跳过", file=sys.stderr); continue
        gtype = meta[gname][1]                 # 40=GO1B, 41=GO2B
        xin, rin = f"{a.cap}/raw_ffn_in_L{L}", f"{a.cap}/raw_route_L{L}"
        if not (os.path.isfile(xin) and os.path.isfile(rin)):
            print(f"L{L}: 缺激活跳过", file=sys.stderr); continue
        X = np.fromfile(xin, dtype="<f2").reshape(-1, D).astype(np.float32)
        ids = np.fromfile(rin, dtype="<i2").reshape(-1, NACT)
        ok = np.abs(X).max(1) > 0
        X, ids = X[ok][:a.ntok], ids[ok][:a.ntok]
        nrows1 = meta[f"blk.{L}.ffn_gate_exps.weight"][0][1]
        nrows2 = meta[f"blk.{L}.ffn_down_exps.weight"][0][1]
        nexp = 256
        cov = np.zeros(nexp, bool)
        t0 = time.time()
        gn = ("gate", "up", "down"); tn = {"gate": f"blk.{L}.ffn_gate_exps.weight",
              "up": f"blk.{L}.ffn_up_exps.weight", "down": f"blk.{L}.ffn_down_exps.weight"}
        rows = {"gate": nrows1, "up": nrows1, "down": nrows2}
        if gtype == 40:
            out = {f"{k}": np.zeros((nexp, rows[k]), np.float32) for k in gn}
        else:
            out = {}
            for k in gn:
                out[f"{k}_d1"] = np.zeros((nexp, rows[k]), np.float32)
                out[f"{k}_d2"] = np.zeros((nexp, rows[k]), np.float32)
        for e in range(nexp):
            sel = (ids == e).any(1); Xe = X[sel]
            if len(Xe) < 8:
                continue
            w = {"gate": F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight"),
                 "up":   F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight"),
                 "down": F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight")}
            # 先算 gate/up 的量化重构 → 得 w2 输入 h
            if gtype == 40:
                sg = {k: mono_signs(mm, meta, base, tn[k], e) for k in gn}
                ag = {k: act_b(w[k], sg[k], Xe) for k in ("gate", "up")}
                gq = sg["gate"] * ag["gate"][:, None]; uq = sg["up"] * ag["up"][:, None]
                gg = Xe @ gq.T; uu = Xe @ uq.T; h = (gg / (1 + np.exp(-gg))) * uu
                ag["down"] = act_b(w["down"], sg["down"], h)
                for k in gn:
                    out[k][e] = ag[k]
            else:
                cd = {k: mono_go2b_codes(mm, meta, base, tn[k], e) for k in gn}
                dd = {}
                for k in ("gate", "up"):
                    dd[k] = act_d1d2(w[k], cd[k][0], cd[k][1], Xe)
                gq = cd["gate"][0] * dd["gate"][0][:, None] + cd["gate"][1] * dd["gate"][1][:, None]
                uq = cd["up"][0] * dd["up"][0][:, None] + cd["up"][1] * dd["up"][1][:, None]
                gg = Xe @ gq.T; uu = Xe @ uq.T; h = (gg / (1 + np.exp(-gg))) * uu
                dd["down"] = act_d1d2(w["down"], cd["down"][0], cd["down"][1], h)
                for k in gn:
                    out[f"{k}_d1"][e] = dd[k][0]; out[f"{k}_d2"][e] = dd[k][1]
            cov[e] = True
        np.savez(f"{a.out}/L{L}.npz", type=gtype, cov=cov, **out)
        print(f"L{L}: type{gtype} {int(cov.sum())}/{nexp} 覆盖, {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    print("scale_compute done")


if __name__ == "__main__":
    main()
