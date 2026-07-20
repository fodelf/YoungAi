#!/usr/bin/env python3
"""fix_l24_requant.py — shard-26 零洞修复后, 从(修好的) HF 整层重量化 L24 全 256 专家
写回 v3 gguf (scale+sign 全 34B 块, 语义=gen 默认: per-row mean|w| scale 复制进每块).
之后再跑 m4_lane L24 单层 GPTQ, 把 sign 升级成行为感知版.
用法 (M1): DS4_HF=... python3 fix_l24_requant.py --gguf gguf/ds4-go1b-v3.gguf [--layer 24]
"""
import os, sys, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--pyfwd", default="/Users/fodelf/ds4-main/gguf-tools/go-onebit/calib/pyfwd")
    args = ap.parse_args()
    L = args.layer
    sys.path.insert(0, args.pyfwd)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ds4reader as R
    from gptq1_rewrite import parse_gguf, read_expert_blocks

    f = open(args.gguf, "r+b")
    tens, data0 = parse_gguf(f)
    KINDS = (("gate", "w1"), ("up", "w3"), ("down", "w2"))
    fixed = 0
    for kind, hfw in KINDS:
        ne, ty, toff = tens[f"blk.{L}.ffn_{kind}_exps.weight"]
        assert ty == 40
        cols, rows = int(ne[0]), int(ne[1])
        nblk = cols // 256
        for e in range(256):
            w = R.read_weight(f"layers.{L}.ffn.experts.{e}.{hfw}.weight").astype(np.float32)
            assert w.shape == (rows, cols)
            s_row = np.abs(w).mean(axis=1).astype(np.float16)          # per-row scale (gen 默认)
            assert float((s_row != 0).mean()) > 0.999, f"e{e} {kind} 修复后仍有零 scale — shard 还是坏的?"
            raw, off = read_expert_blocks(f, data0, toff, e, rows, cols)
            raw[:, :, 0:2] = np.repeat(s_row.view(np.uint8).reshape(rows, 1, 2), nblk, axis=1)
            bits = (w >= 0).astype(np.uint8).reshape(rows, nblk, 256)
            raw[:, :, 2:34] = np.packbits(bits, axis=2, bitorder="little")
            f.seek(off); f.write(raw.tobytes())
            fixed += 1
        print(f"L{L} {kind}: 256 experts requantized", flush=True)
    f.close()
    print(f"REQUANT-OK L{L} blocks_written={fixed}", flush=True)

if __name__ == "__main__":
    main()
