#!/usr/bin/env python3
"""layer_truth.py — 单层真尺三判（逐层调试的标准工具）:
  ① 纯 HF 前向 vs 配对教师目标 (尺自检, 同代配对下应 ≈100%)
  ② 当前底座还原度
  ③ 底座+残差平面 (Q1(W−部署字节)) 还原度
还原度% = (1−||y*−g·ŷ||²/||y*||²)×100, g=train 窗逐 token 中位数增益, TE=尾部 17% chunk。
用法: python3 layer_truth.py --layer L --pack /tmp/pk_L{L} [--cap /tmp/cap_v3prep] [--gguf ...]
pack 由 M1 pack_layer.py 产 (w8/si/meta.json 三件即可)。目标文件名 routed_paired_L{L}.npy，
不存在则回落 routed_L{L}.npy（cap_l05 的 L0-5 与 hash 层本就配对语义）。
"""
import numpy as np, json, sys, os, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--cap", default="/tmp/cap_v3prep")
    ap.add_argument("--gguf", default="/Users/fodelf/git/ds4-main/gguf/ds4-go1b-v3.gguf")
    ap.add_argument("--swlim", type=float, default=10.0)
    ap.add_argument("--json", default="")  # 默认 gguf/v3-artifacts/layer_truth_L{L}.json (勿丢原则)
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib", "pyfwd"))
    from gptq1_rewrite import parse_gguf, read_expert_blocks, blocks_dequant, deq, swiglu
    from ds4reader import LUT
    L = args.layer; CAP = args.cap

    tgt = f"{CAP}/routed_paired_L{L}.npy"
    if not os.path.exists(tgt):
        tgt = f"{CAP}/routed_L{L}.npy"
    yt = np.load(tgt).astype(np.float32)
    X = np.load(f"{CAP}/ffn_in_L{L}.npy").astype(np.float32)
    route = np.load(f"{CAP}/route_L{L}.npy")
    rw = np.load(f"{CAP}/route_w_L{L}.npy").astype(np.float32)
    print(f"L{L} target={os.path.basename(tgt)} n={len(yt)}", flush=True)

    meta = json.load(open(f"{args.pack}.meta.json")); K = meta["kinds"]
    w8 = open(f"{args.pack}.w8", "rb"); si = open(f"{args.pack}.si", "rb")
    def hf_w(ki, kind, e):
        r, c = K[kind]["rows"], K[kind]["cols"]; sr, scn = K[kind]["si_shape"]
        off = sum(K[k]["rows"] * K[k]["cols"] * 256 for k in ("gate", "up", "down")[:ki])
        w8.seek(off + e * r * c)
        a = np.frombuffer(w8.read(r * c), dtype=np.uint8).reshape(r, c)
        soff = sum(int(np.prod(K[k]["si_shape"])) * 4 * 256 for k in ("gate", "up", "down")[:ki])
        si.seek(soff + e * sr * scn * 4)
        sc = np.frombuffer(si.read(sr * scn * 4), dtype=np.float32).reshape(sr, scn)
        return LUT[a] * np.repeat(np.repeat(sc, 128, axis=0), 128, axis=1)[:r, :c]
    g = open(args.gguf, "rb")
    tens, data0 = parse_gguf(g); T = {}
    for kind in ("gate", "up", "down"):
        ne, ty, off = tens[f"blk.{L}.ffn_{kind}_exps.weight"]
        T[kind] = (int(ne[0]), int(ne[1]), off)
    def base_w(kind, e):
        cols, rows, toff = T[kind]
        raw, _ = read_expert_blocks(g, data0, toff, e, rows, cols)
        s, B = blocks_dequant(raw, cols)
        return deq(s, B)
    def q1(W):
        s = np.abs(W).mean(axis=1, keepdims=True)
        return s * np.sign(W + 1e-30)

    n = min(len(yt), len(route))
    def fwd(wf):
        y = np.zeros((n, yt.shape[1]), dtype=np.float32)
        for e in range(256):
            sel = np.nonzero(route[:n] == e)
            if sel[0].size == 0:
                continue
            Wg, Wu, Wd = wf(e)
            h = swiglu(X[sel[0]] @ Wg.T, X[sel[0]] @ Wu.T, args.swlim)
            np.add.at(y, sel[0], (rw[:n][sel][:, None] * (h @ Wd.T)))
        return y
    def judge(name, y):
        nch = n // 512; nte = max(1, int(0.17 * nch + 0.5)); cut = (nch - nte) * 512
        num = np.sum(yt[:cut] * y[:cut], 1); den = np.sum(y[:cut] * y[:cut], 1) + 1e-20
        gr = float(np.median(num / den)); e2 = yt[cut:] - gr * y[cut:]
        cos = float(np.mean(np.sum(yt[cut:] * y[cut:], 1) /
                            (np.linalg.norm(yt[cut:], axis=1) * np.linalg.norm(y[cut:], axis=1) + 1e-20)))
        r2 = (1 - float(np.sum(e2 * e2)) / float(np.sum(yt[cut:] * yt[cut:]))) * 100
        print(f"L{L}[真尺] {name}: g={gr:.2f} cos={cos:.3f} restore={r2:.1f}%", flush=True)
        return dict(g=round(gr, 4), cos=round(cos, 4), restore=round(r2, 2))
    res = {}
    res["hf_selfcheck"] = judge("① 纯HF(自检)", fwd(lambda e: (hf_w(0, "gate", e), hf_w(1, "up", e), hf_w(2, "down", e))))
    res["base"] = judge("② 当前底座", fwd(lambda e: (base_w("gate", e), base_w("up", e), base_w("down", e))))
    res["base_plus_residual"] = judge("③ 底座+残差", fwd(lambda e: tuple(
        base_w(k, e) + q1(hf_w(i, k, e) - base_w(k, e)) for i, k in ((0, "gate"), (1, "up"), (2, "down")))))
    out = args.json or f"/Users/fodelf/git/ds4-main/gguf/v3-artifacts/layer_truth_L{L}.json"
    json.dump(dict(layer=L, target=os.path.basename(tgt), **res), open(out, "w"), indent=1)
    print(f"saved -> {out}", flush=True)

if __name__ == "__main__":
    main()
