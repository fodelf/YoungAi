#!/usr/bin/env python3
"""l42_analysis.py — 出口层(L42)量化数据深挖 → 重设计依据(2026-07-27)。
①逐类相对误差(热/冷 w1/w3/w2, 含 base down 张量里的冷 w2 signref 解码)
②中心化层残差 E 的秩谱(rank 4..64 吸收曲线)
③误差的专家分布(定位误差预算住在哪)
用法(M1): DS4_HF=... python3 l42_analysis.py --gguf gguf/go-onebit/ds4-vq22.gguf \
          --layer 42 --cap /tmp/capV3 --oref /tmp/orefV3 --hot /tmp/hot64.txt
"""
import argparse, os, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "calib", "pyfwd"))
import ds4reader as R
from gr_refit_layer import gguf_blob, parse_segments, decode_seg, refit, D, MOEI, NEXP


def gguf_tensor(path, name):
    f = open(path, "rb")
    magic, ver, n_t, n_kv = struct.unpack("<IIQQ", f.read(24))
    def rstr():
        n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode()
    def skipv(t):
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8: rstr()
        elif t == 9:
            et, = struct.unpack("<I", f.read(4)); n, = struct.unpack("<Q", f.read(8))
            for _ in range(n): skipv(et)
        else: f.read(sz[t])
    for _ in range(n_kv):
        rstr(); t, = struct.unpack("<I", f.read(4)); skipv(t)
    tens = {}
    for _ in range(n_t):
        nm = rstr(); nd, = struct.unpack("<I", f.read(4))
        ne = struct.unpack("<%dQ" % nd, f.read(8*nd))
        ty, off = struct.unpack("<IQ", f.read(12))
        tens[nm] = (ne, ty, off)
    data0 = (f.tell() + 31)//32*32
    return f, tens, data0


def decode_go1b_expert(f, data0, off, e, rows, cols):
    """down_exps type40: 每行 cols/256 块 × 34B([2B f16 scale][32B 符号位])"""
    nblk = cols // 256
    row_b = nblk * 34
    f.seek(data0 + off + e * rows * row_b)
    raw = np.frombuffer(f.read(rows * row_b), np.uint8).reshape(rows, nblk, 34)
    sc = raw[:, :, :2].copy().view(np.float16).astype(np.float32)      # [rows, nblk, 1]
    bits = np.unpackbits(raw[:, :, 2:], axis=2, bitorder="little")     # [rows, nblk, 256]
    sign = bits.astype(np.float32) * 2.0 - 1.0
    return (sign * sc).reshape(rows, cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layer", type=int, default=42)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--oref", required=True)
    ap.add_argument("--hot", required=True)
    ap.add_argument("--nsample", type=int, default=12)
    ap.add_argument("--nrows", type=int, default=932)
    a = ap.parse_args()
    L = a.layer

    hot = None
    for line in open(a.hot):
        lab, rest = line.split(":")
        if int(lab[1:]) == L:
            hot = list(map(int, rest.split())); break
    hotset = set(hot)
    X = np.fromfile(f"{a.cap}/raw_ffn_in_L{L}", np.float16).reshape(-1, D)[:a.nrows].astype(np.float32)
    blob = gguf_blob(a.gguf, L)
    segs = parse_segments(blob)
    seg_map, si = {}, 0
    for e in range(NEXP):
        for k in (("w1", "w3", "w2") if e in hotset else ("w1", "w3")):
            seg_map[(e, k)] = segs[si]; si += 1

    gf, tens, data0 = gguf_tensor(a.gguf, f"blk.{L}.ffn_down_exps.weight")
    dne, dty, doff = tens[f"blk.{L}.ffn_down_exps.weight"]

    # ① 逐类误差(抽样) + 冷 w2 signref
    agg = {}
    hs, cs = hot[: a.nsample], [e for e in range(NEXP) if e not in hotset][: a.nsample]
    for e in hs + cs:
        cls = "hot" if e in hotset else "cold"
        w1 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight").astype(np.float32)
        w3 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight").astype(np.float32)
        w2 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight").astype(np.float32)
        q1, g1 = decode_seg(blob, seg_map[(e, "w1")])
        q3, g3 = decode_seg(blob, seg_map[(e, "w3")])
        for k, (Wt, Wq, gr) in {"w1": (w1, q1, g1), "w3": (w3, q3, g3)}.items():
            _, eo, _ = refit(X, Wq, Wt, gr)
            agg.setdefault(f"{cls}.{k}", []).append(eo)
        midq = (X @ q1.T); midq = midq/(1+np.exp(-np.clip(midq,-30,30))) * (X @ q3.T)
        if cls == "hot":
            q2, g2 = decode_seg(blob, seg_map[(e, "w2")])
            _, eo, _ = refit(midq, q2, w2, g2)
            agg.setdefault("hot.w2", []).append(eo)
        else:
            q2s = decode_go1b_expert(gf, data0, doff, e, D, MOEI)   # [4096,2048]
            Y = midq @ w2.T; P = midq @ q2s.T
            agg.setdefault("cold.w2(signref)", []).append(
                float(np.linalg.norm(P - Y) / max(np.linalg.norm(Y), 1e-9)))
        print(f"  e{e}({cls}) done", flush=True)
    print(f"\n{'类':<16} {'n':>3} {'相对误差':>8}")
    for k in sorted(agg):
        v = np.array(agg[k]); print(f"{k:<16} {len(v):>3} {v.mean():>8.4f}")

    # ② 层残差秩谱(中心化+95分位截断, 与 corr_fit v2 同口径)
    O = np.fromfile(f"{a.cap}/raw_ffn_out_L{L}", np.float16).reshape(-1, D)[:a.nrows].astype(np.float64)
    Rf = np.load(f"{a.oref}/oref_L{L}.npy").astype(np.float64)[:a.nrows]
    E = Rf - O
    rn = np.linalg.norm(E, axis=1); capn = np.percentile(rn, 95)
    E = E * np.minimum(1.0, capn / np.maximum(rn, 1e-9))[:, None]
    E = E - E.mean(0, keepdims=True)
    U_, S_, _ = np.linalg.svd(E, full_matrices=False)
    tot = (S_ ** 2).sum()
    print("\n[秩谱] 中心化层残差能量吸收:")
    for r in (4, 8, 16, 32, 64, 128):
        print(f"  rank {r:>3}: {100*(S_[:r]**2).sum()/tot:.1f}%")

    # ③ 误差的专家侧分布: 每 token 残差 vs 该 token 路由的 hot/cold 构成
    si6 = np.fromfile(f"{a.cap}/raw_route_L{L}", np.int16).reshape(-1, 6)[:a.nrows]
    nhot = np.array([[1 if p in hotset else 0 for p in row] for row in si6]).sum(1)
    rn2 = np.linalg.norm((Rf - O), axis=1)
    for k in range(7):
        m = nhot == k
        if m.sum() > 5:
            print(f"[路由构成] {k}热/{6-k}冷: n={m.sum():>4} 残差均值={rn2[m].mean():>9.1f}")


if __name__ == "__main__":
    main()
