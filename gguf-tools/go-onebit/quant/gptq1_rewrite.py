#!/usr/bin/env python3
"""gptq1_rewrite.py — 新模型核心: 1-bit 符号平面的 Go 行为感知重选 (block-256 GPTQ 误差反馈).

底座格式不变 (go1b 34B 块; scale 字节一字不动), 只重写 sign 位:
sign 不再 = sign(W), 而是逐列量化 + 误差反馈到同 256-块内后续列
(H = XᵀX, X = 该专家在 Go 轨迹上 fired 的输入), 直接最小化 (W−Ŵ)X 行为误差.
表达内 EF 级联: gate/up 定型后, 用学生 ĥ(新 1-bit gate/up 前向) 重建 down 的 H —
runtime-faithful (部署时 down 看到的就是 1-bit ĥ, 不是 FP h).

块对角 H=256 的统计依据: 每专家平均 fired ≈ 6/256×N_tok, 撑得起 256×256 而撑不起 4096².

用法 (在有 HF shards + cap 的机器, 通常 M1):
  DS4_HF=/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base \
  python3 gptq1_rewrite.py --gguf gguf/ds4-go1b-v3.gguf --layer 20 \
    --ffn-in cap_v2r2/ffn_in_L20.npy --route cap_v2r2/route_L20.npy \
    [--experts 16] [--apply] [--json /tmp/l20.json] [--pyfwd path]

判决指标 (held-out fired 20%): 专家端到端输出 cos(FP专家, 1bit专家) old→new.
--apply 才写文件; 默认干跑只出表.
"""
import os, sys, json, time, argparse, struct
import numpy as np

QK, BLK = 256, 34

def parse_gguf(f):
    def rd(n):
        b = f.read(n); assert len(b) == n, "short read in header"; return b
    magic, ver, n_t, n_kv = struct.unpack("<IIQQ", rd(24))
    assert magic == 0x46554747, "not a GGUF"
    def rstr():
        n, = struct.unpack("<Q", rd(8)); return rd(n).decode()
    def skipval(t):
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8: rstr(); return
        if t == 9:
            et, = struct.unpack("<I", rd(4)); n, = struct.unpack("<Q", rd(8))
            for _ in range(n): skipval(et)
            return
        rd(sz[t])
    for _ in range(n_kv):
        rstr(); t, = struct.unpack("<I", rd(4)); skipval(t)
    tens = {}
    for _ in range(n_t):
        nm = rstr(); nd, = struct.unpack("<I", rd(4))
        ne = struct.unpack("<%dQ" % nd, rd(8*nd))
        ty, = struct.unpack("<I", rd(4)); off, = struct.unpack("<Q", rd(8))
        tens[nm] = (list(ne), ty, off)
    data0 = (f.tell() + 31) // 32 * 32
    return tens, data0

def read_expert_blocks(f, data0, toff, e, rows, cols):
    """-> raw [rows, nblk, 34] u8 (copy), file offset of the region"""
    nblk = cols // QK
    rb = nblk * BLK
    off = data0 + toff + e * rows * rb
    f.seek(off)
    raw = np.frombuffer(f.read(rows * rb), dtype=np.uint8).reshape(rows, nblk, BLK).copy()
    return raw, off

def blocks_dequant(raw, cols):
    """raw [rows,nblk,34] -> (scale[rows,nblk] f32, B[rows,cols] ±1 f32)"""
    s = raw[:, :, 0:2].copy().view(np.float16).astype(np.float32)[:, :, 0]
    bits = np.unpackbits(raw[:, :, 2:BLK], axis=2, bitorder="little")  # [rows,nblk,256]
    B = bits.reshape(raw.shape[0], -1)[:, :cols].astype(np.float32) * 2.0 - 1.0
    return s, B

def blocks_pack_signs(raw, Bnew):
    """写回 sign 位, scale 字节不动. Bnew [rows,cols] ±1"""
    rows, nblk = raw.shape[0], raw.shape[1]
    bits = (Bnew.reshape(rows, nblk, QK) >= 0).astype(np.uint8)
    raw[:, :, 2:BLK] = np.packbits(bits, axis=2, bitorder="little")

def deq(s, B):
    """(s[rows,nblk], B[rows,cols]) -> Ŵ [rows,cols] f32"""
    return B * np.repeat(s, QK, axis=1)

def gptq_signs(W, s, X, ridge):
    """块对角 GPTQ: 固定 scale, 逐列选 sign + 误差反馈同块后续列.
    W [rows,cols] f32 (FP 目标), s [rows,nblk], X [n,cols] fired 输入.
    -> Bnew [rows,cols] ±1"""
    rows, cols = W.shape
    nblk = cols // QK
    Bnew = np.empty((rows, cols), dtype=np.float32)
    for b in range(nblk):
        j0 = b * QK
        Xb = X[:, j0:j0+QK].astype(np.float32)
        H = Xb.T @ Xb
        lam = ridge * float(np.mean(np.diag(H)) + 1e-12)
        H[np.diag_indices(QK)] += lam
        Hinv = np.linalg.inv(H)
        Wk = W[:, j0:j0+QK].astype(np.float32).copy()
        sb = s[:, b]                                   # [rows] 本块 scale (不动)
        for j in range(QK):
            sign = np.where(Wk[:, j] >= 0.0, 1.0, -1.0).astype(np.float32)
            Bnew[:, j0+j] = sign
            q = sb * sign
            err = (Wk[:, j] - q) / Hinv[j, j]
            if j + 1 < QK:
                Wk[:, j+1:] -= np.outer(err, Hinv[j, j+1:])
    return Bnew

def swiglu(gate, up, swlim):
    if swlim > 0:
        up = np.clip(up, -swlim, swlim); gate = np.minimum(gate, swlim)
    return gate / (1.0 + np.exp(-gate)) * up          # silu(gate)*up

def tok_cos(A, Bm):
    num = np.sum(A * Bm, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(Bm, axis=1) + 1e-20
    return float(np.mean(num / den))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--ffn-in", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--experts", type=int, default=0, help="0=全 256; N=前 N 个(按 fired 数降序)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--ridge", type=float, default=0.01)
    ap.add_argument("--min-fired", type=int, default=48)
    ap.add_argument("--pyfwd", default=os.path.join(os.path.dirname(__file__), "..", "calib", "pyfwd"))
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.pyfwd))
    import ds4reader as R
    cfg = json.load(open(os.path.join(R.HF, "config.json")))
    SWLIM = cfg["swiglu_limit"]

    L = args.layer
    Xall = np.load(args.ffn_in)                        # [Nt,4096] f16
    route = np.load(args.route)                        # [Nt,6] int
    assert Xall.shape[0] == route.shape[0], "cap ffn_in/route token 数不一致"

    mode = "r+b" if args.apply else "rb"
    f = open(args.gguf, mode)
    tens, data0 = parse_gguf(f)
    T = {}
    for kind in ("gate", "up", "down"):
        nm = f"blk.{L}.ffn_{kind}_exps.weight"
        ne, ty, off = tens[nm]
        assert ty == 40, f"{nm} 不是 go1b (type {ty})"
        T[kind] = (int(ne[0]), int(ne[1]), off)        # cols, rows, off

    fired_cnt = np.zeros(256, dtype=np.int64)
    for e in range(256):
        fired_cnt[e] = int(np.sum(route == e))
    order = np.argsort(-fired_cnt)
    todo = [int(e) for e in order if fired_cnt[e] >= args.min_fired]
    if args.experts > 0:
        todo = todo[:args.experts]
    skipped = int(np.sum(fired_cnt < args.min_fired))

    print(f"L{L}: {len(todo)} experts (fired>={args.min_fired}; skip {skipped}), "
          f"tok={Xall.shape[0]}, apply={args.apply}", flush=True)

    res = []
    t0 = time.time()
    for i, e in enumerate(todo):
        tok_idx = np.unique(np.where(route == e)[0])
        ho = tok_idx[::5]
        tr = np.setdiff1d(tok_idx, ho, assume_unique=True)
        Xtr = Xall[tr].astype(np.float32); Xho = Xall[ho].astype(np.float32)

        w1 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight").astype(np.float32)
        w3 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight").astype(np.float32)
        w2 = R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight").astype(np.float32)

        raws, offs, S, Bold = {}, {}, {}, {}
        for kind, W in (("gate", w1), ("up", w3), ("down", w2)):
            cols, rows, toff = T[kind]
            assert (rows, cols) == W.shape, f"{kind} 形状不符 gguf({rows},{cols}) hf{W.shape}"
            raw, off = read_expert_blocks(f, data0, toff, e, rows, cols)
            s, B = blocks_dequant(raw, cols)
            raws[kind], offs[kind], S[kind], Bold[kind] = raw, off, s, B

        Bn = {}
        Bn["gate"] = gptq_signs(w1, S["gate"], Xtr, args.ridge)
        Bn["up"]   = gptq_signs(w3, S["up"],   Xtr, args.ridge)
        # down: H 用学生 ĥ (新 1-bit gate/up 的前向) — 部署时 down 看到的分布
        g_hat = Xtr @ deq(S["gate"], Bn["gate"]).T
        u_hat = Xtr @ deq(S["up"],   Bn["up"]).T
        h_hat = swiglu(g_hat, u_hat, SWLIM)
        Bn["down"] = gptq_signs(w2, S["down"], h_hat, args.ridge)

        # 判决: held-out 专家端到端输出 cos (FP vs 1bit), old vs new
        y_ref = swiglu(Xho @ w1.T, Xho @ w3.T, SWLIM) @ w2.T
        def student(Bset):
            g = Xho @ deq(S["gate"], Bset["gate"]).T
            u = Xho @ deq(S["up"],   Bset["up"]).T
            return swiglu(g, u, SWLIM) @ deq(S["down"], Bset["down"]).T
        c_old = tok_cos(y_ref, student(Bold))
        c_new = tok_cos(y_ref, student(Bn))
        flip = float(np.mean([np.mean(Bn[k] != Bold[k]) for k in ("gate", "up", "down")]))

        if args.apply:
            for kind in ("gate", "up", "down"):
                blocks_pack_signs(raws[kind], Bn[kind])
                f.seek(offs[kind]); f.write(raws[kind].tobytes())

        res.append(dict(e=e, fired=int(fired_cnt[e]), cos_old=c_old, cos_new=c_new, flip=flip))
        el = time.time() - t0
        print(f"  [{i+1}/{len(todo)}] e{e} fired={fired_cnt[e]} cos {c_old:.4f}->{c_new:.4f} "
              f"(d{c_new-c_old:+.4f}) flip={flip*100:.1f}% {el:.0f}s eta={el/(i+1)*(len(todo)-i-1):.0f}s",
              flush=True)

    mo = float(np.mean([r["cos_old"] for r in res])) if res else 0.0
    mn = float(np.mean([r["cos_new"] for r in res])) if res else 0.0
    print(f"L{L} SUMMARY experts={len(res)} mean_cos {mo:.4f} -> {mn:.4f} (d{mn-mo:+.4f}) "
          f"applied={args.apply}", flush=True)
    if args.json:
        json.dump(dict(layer=L, applied=bool(args.apply), summary=dict(cos_old=mo, cos_new=mn),
                       experts=res), open(args.json, "w"), indent=1)
    f.close()

if __name__ == "__main__":
    main()
