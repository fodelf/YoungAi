#!/usr/bin/env python3
"""splice_signs.py — M4 lane 收尾 (跑在 M1): 把 sign blob 写回 gguf 对应层的专家块.
只覆写每 34B 块的 [2:34] sign 区, scale 字节不动. blob 布局与 pack .b 同序.
"""
import sys, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--signs", required=True)
    args = ap.parse_args()
    sys.path.insert(0, "/Users/fodelf/ds4-main/gguf-tools/go-onebit/quant")
    from gptq1_rewrite import parse_gguf, read_expert_blocks

    blob = np.memmap(args.signs, dtype=np.uint8, mode="r")
    f = open(args.gguf, "r+b")
    tens, data0 = parse_gguf(f)
    pos = 0; wrote = 0
    for kind in ("gate", "up", "down"):
        ne, ty, toff = tens[f"blk.{args.layer}.ffn_{kind}_exps.weight"]
        assert ty == 40
        cols, rows = int(ne[0]), int(ne[1])
        nblk = cols // 256
        for e in range(256):
            raw, off = read_expert_blocks(f, data0, toff, e, rows, cols)
            n = rows * nblk * 32
            raw[:, :, 2:34] = blob[pos:pos+n].reshape(rows, nblk, 32)
            f.seek(off); f.write(raw.tobytes())
            pos += n; wrote += n
    assert pos == blob.size, f"blob 尺寸不符: used {pos} != {blob.size}"
    f.close()
    print(f"SPLICE-OK L{args.layer} bytes={wrote}", flush=True)

if __name__ == "__main__":
    main()
