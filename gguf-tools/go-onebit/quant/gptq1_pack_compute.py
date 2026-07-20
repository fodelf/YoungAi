#!/usr/bin/env python3
"""gptq1_pack_compute.py — M4 lane 计算端: 吃 pack_layer.py 的平面包, 跑与
gptq1_rewrite.py 完全相同的 block-256 GPTQ 符号重选 (import 同一实现), 产 sign blob.
blob 布局 = pack .b 同序 (kind-major expert-major rows×nblk×32), 先整体拷旧 sign,
量化到的专家区域再覆写 → skip 专家写回旧值, splice 端无需 validity 表.
解码奇偶校验: e0 gate 的 LUT×scale 重建必须与 M1 权威 .check.npy allclose.
"""
import os, sys, json, time, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--signs-out", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--ridge", type=float, default=0.01)
    ap.add_argument("--min-fired", type=int, default=48)
    ap.add_argument("--pyfwd", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib", "pyfwd"))
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.pyfwd))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ds4reader import LUT
    from gptq1_rewrite import gptq_signs, swiglu, tok_cos

    meta = json.load(open(f"{args.pack}.meta.json"))
    L = meta["layer"]; SWLIM = meta["swiglu_limit"]; NE = meta["experts"]
    KINDS = meta["order"]
    K = {k: meta["kinds"][k] for k in KINDS}

    off_w8, off_si, off_sb = {}, {}, {}
    o1 = o2 = o3 = 0
    for k in KINDS:
        r, c, nb = K[k]["rows"], K[k]["cols"], K[k]["nblk"]
        sr, scn = K[k]["si_shape"]
        off_w8[k] = o1; o1 += r * c * NE           # 单位=u8 元素(=字节)
        off_si[k] = o2; o2 += sr * scn * NE        # 单位=f32 元素 (memmap.size 是元素数不是字节!)
        off_sb[k] = o3; o3 += r * nb * NE          # 单位=块数; .s 每块2B, .b 每块32B
    class PR:  # pread 切片读取: memmap 触碰页顶爆 RSS 看门狗 (烟测: 52专家后 3.7G WDKILL)
        def __init__(self, path, dtype):
            self.f = open(path, "rb"); self.dt = np.dtype(dtype)
            self.size = os.path.getsize(path) // self.dt.itemsize
        def get(self, off, n):
            self.f.seek(off * self.dt.itemsize)
            return np.frombuffer(self.f.read(n * self.dt.itemsize), dtype=self.dt)
    w8 = PR(f"{args.pack}.w8", np.uint8)
    si = PR(f"{args.pack}.si", np.float32)
    sf = PR(f"{args.pack}.s", np.uint8)
    bf = PR(f"{args.pack}.b", np.uint8)
    sizes = dict(w8=(w8.size, o1), si=(si.size, o2), s=(sf.size, o3 * 2), b=(bf.size, o3 * 32))
    bad = {k: v for k, v in sizes.items() if v[0] != v[1]}
    assert not bad, f"pack 尺寸不符 (actual, expected): {bad}"

    out = np.memmap(args.signs_out, dtype=np.uint8, mode="w+", shape=(o3 * 32,))
    CH = 1 << 24
    for p in range(0, o3 * 32, CH):                 # 默认=旧 sign (skip 专家回填), 分块拷防峰值
        out[p:p+CH] = bf.get(p, min(CH, o3 * 32 - p))

    def dequant(k, e):
        r, c = K[k]["rows"], K[k]["cols"]
        sr, scn = K[k]["si_shape"]
        a = w8.get(off_w8[k] + e*r*c, r*c).reshape(r, c)
        sc = si.get(off_si[k] + e*sr*scn, sr*scn).reshape(sr, scn)
        w = LUT[a]
        scf = np.repeat(np.repeat(sc, 128, axis=0), 128, axis=1)[:r, :c]
        return (w * scf).astype(np.float32)

    def gguf_scales(k, e):
        r, nb = K[k]["rows"], K[k]["nblk"]
        raw = sf.get((off_sb[k] + e*r*nb)*2, r*nb*2)
        return raw.copy().view(np.float16).astype(np.float32).reshape(r, nb)

    def old_signs(k, e):
        r, c, nb = K[k]["rows"], K[k]["cols"], K[k]["nblk"]
        raw = bf.get((off_sb[k] + e*r*nb)*32, r*nb*32).reshape(r, nb, 32)
        bits = np.unpackbits(raw, axis=2, bitorder="little")
        return bits.reshape(r, -1)[:, :c].astype(np.float32) * 2.0 - 1.0

    def put_signs(k, e, Bn):
        r, nb = K[k]["rows"], K[k]["nblk"]
        bits = (Bn.reshape(r, nb, 256) >= 0).astype(np.uint8)
        out[(off_sb[k] + e*r*nb)*32 : (off_sb[k] + (e+1)*r*nb)*32] = \
            np.packbits(bits, axis=2, bitorder="little").reshape(-1)

    chk = np.load(f"{args.pack}.check.npy")
    assert np.allclose(dequant("gate", 0), chk, rtol=0, atol=0), "解码奇偶校验失败 (fp8/scale 不一致)"
    print(f"L{L} decode-parity OK", flush=True)

    Xall = np.load(f"{args.pack}.ffn_in.npy")
    route = np.load(f"{args.pack}.route.npy")
    fired_cnt = np.array([(route == e).sum() for e in range(NE)])
    order = np.argsort(-fired_cnt)
    todo = [int(e) for e in order if fired_cnt[e] >= args.min_fired]
    print(f"L{L}: {len(todo)} experts (fired>={args.min_fired}; skip {NE-len(todo)}), tok={Xall.shape[0]}, apply=True(m4)", flush=True)

    def deq(s, B):
        return B * np.repeat(s, 256, axis=1)

    res = []; t0 = time.time()
    for i, e in enumerate(todo):
        tok_idx = np.unique(np.where(route == e)[0])
        ho = tok_idx[::5]; tr = np.setdiff1d(tok_idx, ho, assume_unique=True)
        Xtr = Xall[tr].astype(np.float32); Xho = Xall[ho].astype(np.float32)
        w1, w3, w2 = dequant("gate", e), dequant("up", e), dequant("down", e)
        S = {k: gguf_scales(k, e) for k in KINDS}
        Bold = {"gate": old_signs("gate", e), "up": old_signs("up", e), "down": old_signs("down", e)}
        Bn = {}
        Bn["gate"] = gptq_signs(w1, S["gate"], Xtr, args.ridge)
        Bn["up"]   = gptq_signs(w3, S["up"],   Xtr, args.ridge)
        h_hat = swiglu(Xtr @ deq(S["gate"], Bn["gate"]).T, Xtr @ deq(S["up"], Bn["up"]).T, SWLIM)
        Bn["down"] = gptq_signs(w2, S["down"], h_hat, args.ridge)

        y_ref = swiglu(Xho @ w1.T, Xho @ w3.T, SWLIM) @ w2.T
        def student(Bs):
            g = Xho @ deq(S["gate"], Bs["gate"]).T
            u = Xho @ deq(S["up"], Bs["up"]).T
            return swiglu(g, u, SWLIM) @ deq(S["down"], Bs["down"]).T
        c_old = tok_cos(y_ref, student(Bold)); c_new = tok_cos(y_ref, student(Bn))
        flip = float(np.mean([np.mean(Bn[k] != Bold[k]) for k in KINDS]))
        for k in KINDS:
            put_signs(k, e, Bn[k])
        if i % 16 == 15:
            out.flush()                             # 泄掉脏页, RSS 有界
        res.append(dict(e=e, fired=int(fired_cnt[e]), cos_old=c_old, cos_new=c_new, flip=flip))
        el = time.time() - t0
        print(f"  [{i+1}/{len(todo)}] e{e} fired={fired_cnt[e]} cos {c_old:.4f}->{c_new:.4f} "
              f"(d{c_new-c_old:+.4f}) flip={flip*100:.1f}% {el:.0f}s eta={el/(i+1)*(len(todo)-i-1):.0f}s", flush=True)

    out.flush()
    mo = float(np.mean([r["cos_old"] for r in res])) if res else 0.0
    mn = float(np.mean([r["cos_new"] for r in res])) if res else 0.0
    print(f"L{L} SUMMARY experts={len(res)} mean_cos {mo:.4f} -> {mn:.4f} (d{mn-mo:+.4f}) applied=True", flush=True)
    if args.json:
        json.dump(dict(layer=L, applied=True, lane="m4",
                       summary=dict(cos_old=mo, cos_new=mn), experts=res),
                  open(args.json, "w"), indent=1)

if __name__ == "__main__":
    main()
