#!/usr/bin/env python3
"""gr_refit_layer.py — g_r 饿死修复的单层干跑(2026-07-27 最小端到端)。

背景(fable5 审计): v2.2 行乘子 g_r≈1.000 全模型(每专家仅~9 校准行, <8 走回退,
过线的被 lam=sp2/(n+1) 收缩钉死) → 幅度校准从未生效 = teacher/student 幅度亏损
1.2-3.2× 的机制凶手。修法: 全集行拟合(932 行) + 温和正则(0.001·sp2)。
本脚本 = 指标干跑(不改字节): 抽样专家, 对比 g_r 修复前后 ‖X·Wq−X·W‖/‖X·W‖。
用法(M1): DS4_HF=... python3 gr_refit_layer.py --gguf ds4-vq22.gguf --layer 30 \
          --cap /tmp/capV3 --hot hot64.txt [--nhot 16 --ncold 16]
"""
import argparse, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib", "pyfwd"))
import ds4reader as R

D, MOEI, NEXP = 4096, 2048, 256


def gguf_blob(path, layer):
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
        tens[nm] = (ne, off)
    data0 = (f.tell() + 31)//32*32
    ne, off = tens[f"blk.{layer}.ffn_exps_vq.blob"]
    f.seek(data0 + off)
    return f.read(ne[0])


def parse_segments(blob):
    """按序扫 DQVQ 段 → [(seg_off, dim, nc, rows, cols, cb_off, gr_off, idx_off, idx_bytes)]"""
    segs, i = [], 0
    while True:
        j = blob.find(b"DQVQ", i)
        if j < 0: break
        dim, nc, rows, cols = struct.unpack_from("<HHII", blob, j + 4)
        if dim in (4, 8) and nc in (256, 512) and rows in (MOEI, D) and cols in (MOEI, D):
            cb_off = j + 16
            gr_off = cb_off + nc * dim * 2
            idx_off = gr_off + rows * 2
            nidx = rows * cols // dim
            nb = nidx if (dim == 8 and nc == 256) else (nidx * 9 + 7)//8 + 1
            segs.append((j, dim, nc, rows, cols, cb_off, gr_off, idx_off, nb))
            i = idx_off + nb
        else:
            i = j + 4
    return segs


def decode_seg(blob, seg):
    _, dim, nc, rows, cols, cb_off, gr_off, idx_off, nb = seg
    cb = np.frombuffer(blob, np.float16, nc * dim, cb_off).astype(np.float32).reshape(nc, dim)
    gr = np.frombuffer(blob, np.float16, rows, gr_off).astype(np.float32)
    nidx = rows * cols // dim
    if dim == 8:
        idx = np.frombuffer(blob, np.uint8, nidx, idx_off).astype(np.int64)
    else:  # 9bit LE 位流
        raw = np.frombuffer(blob, np.uint8, nb, idx_off).astype(np.uint64)
        bits = np.unpackbits(raw.astype(np.uint8), bitorder="little")
        idx = np.zeros(nidx, np.int64)
        for b in range(9):
            idx |= bits[b::9][:nidx].astype(np.int64) << b
    W = cb[idx].reshape(rows, cols) * gr[:, None]
    return W, gr


def refit(X, Wq, W_true, gr_old, lam_frac=1e-3):
    Y = X @ W_true.T
    P = X @ (Wq / np.maximum(gr_old[:, None], 1e-9)).T   # 去掉旧 g_r 的裸量化输出
    sp2 = (P * P).sum(0); spy = (P * Y).sum(0)
    lam = lam_frac * sp2 + 1e-9
    g = (spy + lam) / (sp2 + lam)
    g = np.clip(g, 0.25, 4.0)
    e_old = np.linalg.norm(gr_old[None, :] * P - Y) / max(np.linalg.norm(Y), 1e-9)
    e_new = np.linalg.norm(g[None, :] * P - Y) / max(np.linalg.norm(Y), 1e-9)
    return g, e_old, e_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--hot", required=True)
    ap.add_argument("--nhot", type=int, default=16)
    ap.add_argument("--ncold", type=int, default=16)
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
    print(f"L{L}: blob {len(blob)/2**20:.0f}MiB segs={len(segs)} X={X.shape}", flush=True)

    # 段→(e,which) 映射: 按专家序, 热=[w1,w3,w2] 冷=[w1,w3]
    seg_map, si = {}, 0
    for e in range(NEXP):
        ks = ("w1", "w3", "w2") if e in hotset else ("w1", "w3")
        for k in ks:
            seg_map[(e, k)] = segs[si]; si += 1
    assert si == len(segs), f"段数不匹配 {si} vs {len(segs)}"

    hs = hot[: a.nhot]
    cs = [e for e in range(NEXP) if e not in hotset][: a.ncold]
    agg = {}
    for e in hs + cs:
        cls = "hot" if e in hotset else "cold"
        w1 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight").astype(np.float32)
        w3 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight").astype(np.float32)
        q1, g1 = decode_seg(blob, seg_map[(e, "w1")])
        q3, g3 = decode_seg(blob, seg_map[(e, "w3")])
        for k, (Wt, Wq, gr) in {"w1": (w1, q1, g1), "w3": (w3, q3, g3)}.items():
            g, eo, en = refit(X, Wq, Wt, gr)
            agg.setdefault(f"{cls}.{k}", []).append((eo, en, np.median(g)))
        if cls == "hot":
            w2 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight").astype(np.float32)
            q2, g2 = decode_seg(blob, seg_map[(e, "w2")])
            # 量化 mid = silu(X·q1)·(X·q3), 与运行时/emitter 顺序补偿同口径(SWLIM 简化近似)
            midv = (X @ q1.T); midv = midv/(1+np.exp(-np.clip(midv,-30,30))) * (X @ q3.T)
            g, eo, en = refit(midv, q2, w2, g2)
            agg.setdefault("hot.w2", []).append((eo, en, np.median(g)))
        print(f"  e{e}({cls}) done", flush=True)

    print(f"\n{'类':<9} {'n':>3} {'相对误差 旧':>10} {'新':>8} {'改善':>7} {'g中位':>6}")
    for k in sorted(agg):
        v = np.array(agg[k])
        print(f"{k:<9} {len(v):>3} {v[:,0].mean():>10.4f} {v[:,1].mean():>8.4f} "
              f"{100*(1-v[:,1].mean()/v[:,0].mean()):>6.1f}% {v[:,2].mean():>6.3f}")


if __name__ == "__main__":
    main()
