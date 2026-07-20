#!/usr/bin/env python3
"""scale_fold.py — 幅度塌缩修复: 每层闭式标量增益 g_L=⟨y*,ŷ⟩/⟨ŷ,ŷ⟩ 折进该层
down_exps 的每块 f16 scale (只动 down: 输出精确线性缩放; gate/up 过 swiglu 非线性不可折).
零侧车零运行时改动; cos 不变(尺度不变量), 修的是下游 43 层看得见的增量幅度 (实测塌缩 ~0.43×).
用法: python3 scale_fold.py --gguf F --cap DIR [--layers 0-42] [--dry]
  DIR 需含 routed_L{L}.npy(教师) 与 obase_v3_L{L}.npy(学生), 同 token 序.
"""
import os, sys, json, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--layers", default="0-42")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--json", default="/Users/fodelf/git/ds4-main/gguf/v3-artifacts/gains_fold.json")  # 入库勿丢
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gptq1_rewrite import parse_gguf, read_expert_blocks

    lo, hi = (int(t) for t in args.layers.split("-"))
    f = open(args.gguf, "rb" if args.dry else "r+b")
    tens, data0 = parse_gguf(f)
    out = {}
    for L in range(lo, hi + 1):
        pr = f"{args.cap}/routed_L{L}.npy"; po = f"{args.cap}/obase_v3_L{L}.npy"
        if not (os.path.exists(pr) and os.path.exists(po)):
            print(f"L{L}: cap missing, skip", flush=True); continue
        yt = np.load(pr).astype(np.float32); ys = np.load(po).astype(np.float32)
        n = min(yt.shape[0], ys.shape[0]); yt, ys = yt[:n], ys[:n]
        num = float(np.sum(yt * ys)); den = float(np.sum(ys * ys)) + 1e-20
        g = num / den
        out[L] = g
        print(f"L{L}: g={g:.4f}", flush=True)
        if args.dry or abs(g - 1.0) < 1e-3:
            continue
        ne, ty, toff = tens[f"blk.{L}.ffn_down_exps.weight"]
        assert ty == 40
        cols, rows = int(ne[0]), int(ne[1])
        for e in range(256):
            raw, off = read_expert_blocks(f, data0, toff, e, rows, cols)
            s = raw[:, :, 0:2].copy().view(np.float16)
            s *= np.float16(g)
            raw[:, :, 0:2] = s.view(np.uint8)
            f.seek(off); f.write(raw.tobytes())
    f.close()
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
    print(f"FOLD-{'DRY' if args.dry else 'OK'} layers={len(out)} mean_g={np.mean(list(out.values())):.4f}", flush=True)

if __name__ == "__main__":
    main()
