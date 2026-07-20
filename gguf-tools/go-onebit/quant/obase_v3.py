#!/usr/bin/env python3
"""obase_v3.py — 从部署的 v3 gguf 字节直接计算学生路由输出 ŷ (obase), 供 z 四损失解用.
比旧 precompute_obase (HF+量化模拟) 更忠实: 读的就是运行时 dequant 的同一份 sign/scale 字节.
语义对齐 cap 口径: ŷ_t = Σ_pick route_w[t,pick] × expert_out(x_t)  (routed-only, 含 1.5 缩放,
route_w 是引擎实际应用的 gate 权重), swiglu clamp 同运行时.

用法 (跑在有 v3 gguf 的机器, cap 三件套按层给):
  python3 obase_v3.py --gguf gguf/ds4-go1b-v3.gguf --layer 20 \
    --ffn-in cap/ffn_in_L20.npy --route cap/route_L20.npy --route-w cap/route_w_L20.npy \
    --out cap/obase_v3_L20.npy [--swlim 7.0]
"""
import os, sys, json, time, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--ffn-in", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--route-w", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--swlim", type=float, default=7.0)
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gptq1_rewrite import parse_gguf, read_expert_blocks, blocks_dequant, deq, swiglu

    L = args.layer
    X = np.load(args.ffn_in)                       # [n,4096] f16
    route = np.load(args.route)                    # [n,6] int16
    rw = np.load(args.route_w).astype(np.float32)  # [n,6] 引擎实际 gate 权重(含1.5)
    n = X.shape[0]
    assert route.shape[0] == n and rw.shape[0] == n

    f = open(args.gguf, "rb")
    tens, data0 = parse_gguf(f)
    T = {}
    for kind in ("gate", "up", "down"):
        ne, ty, off = tens[f"blk.{L}.ffn_{kind}_exps.weight"]
        assert ty == 40
        T[kind] = (int(ne[0]), int(ne[1]), off)    # cols, rows, off

    def expert_w(kind, e):
        cols, rows, toff = T[kind]
        raw, _ = read_expert_blocks(f, data0, toff, e, rows, cols)
        s, B = blocks_dequant(raw, cols)
        return deq(s, B)                           # [rows, cols] f32 = 部署字节的精确 dequant

    y = np.zeros((n, 4096), dtype=np.float32)
    t0 = time.time()
    done = 0
    for e in range(256):
        sel = np.nonzero(route == e)               # (tok_idx, pick_idx)
        if sel[0].size == 0:
            continue
        toks = sel[0]; w = rw[sel][:, None]        # [m,1]
        Xt = X[toks].astype(np.float32)
        w1 = expert_w("gate", e); w3 = expert_w("up", e); w2 = expert_w("down", e)
        h = swiglu(Xt @ w1.T, Xt @ w3.T, args.swlim)
        np.add.at(y, toks, (w * (h @ w2.T)).astype(np.float32))
        done += 1
        if done % 32 == 0:
            el = time.time() - t0
            print(f"L{L} obase {done} experts {el:.0f}s eta={el/done*(256-done):.0f}s", flush=True)
    np.save(args.out, y.astype(np.float16))
    print(f"OBASE-OK L{L} n={n} experts_touched={done} -> {args.out} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
