#!/usr/bin/env python3
"""pack_stream.py — 零磁盘层打包 (跑在 M1, stdout 流式): 顺序输出一层全部专家的
fp8 原始字节 (--part w8: kind-major expert-major) 或 128x128 block scale f32 (--part si)。
M1 /tmp 紧张时替代 pack_layer.py: `ssh M1 pack_stream.py --layer L --part w8 > local.w8`。
meta 由接收端按固定架构常量自建 (gate/up [2048,4096] si[16,32]; down [4096,2048] si[32,16])。
"""
import os, sys, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--part", choices=["w8", "si"], required=True)
    ap.add_argument("--pyfwd", default="/Users/fodelf/ds4-main/gguf-tools/go-onebit/calib/pyfwd")
    args = ap.parse_args()
    sys.path.insert(0, args.pyfwd)
    import ds4reader as R
    import numpy as np
    L = args.layer
    out = sys.stdout.buffer
    for hfw in ("w1", "w3", "w2"):
        for e in range(256):
            nm = f"layers.{L}.ffn.experts.{e}.{hfw}.weight"
            if args.part == "w8":
                a, _ = R.raw_bytes(nm)
                out.write(a.tobytes())
            else:
                sc, _ = R.raw_bytes(nm.replace(".weight", ".scale"))
                out.write(sc.astype(np.float32).tobytes())
        print(f"{hfw} streamed", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
