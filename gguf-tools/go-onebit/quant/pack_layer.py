#!/usr/bin/env python3
"""pack_layer.py — M4 lane 供料 (跑在 M1): 把一层的 GPTQ 重写所需全部数据打包成平面文件.
纯 IO (原始 fp8 字节直拷, 不做 dequant), 不占 M1 计算.

产物 (PREFIX.*):
  .w8   u8   kind-major expert-major 原始 fp8 权重字节 (gate,up,down × 256 experts)
  .si   f32  同序 128×128 block scale_inv
  .s    u8   同序 gguf go1b 每块 fp16 scale 原始字节 (rows×nblk×2)
  .b    u8   同序 gguf 旧 sign 字节 (rows×nblk×32) — M4 端 cos_old 对照 + skip 专家回填
  .ffn_in.npy / .route.npy   该层 cap (直拷)
  .check.npy  e0 gate 的 M1 权威 dequant f32 (M4 端逐层解码奇偶校验)
  .meta.json  形状/顺序/swiglu_limit
"""
import os, sys, json, shutil, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf", default="/Users/fodelf/ds4-main/gguf/ds4-go1b-v3.gguf")
    ap.add_argument("--pyfwd", default="/Users/fodelf/ds4-main/gguf-tools/go-onebit/calib/pyfwd")
    ap.add_argument("--capdirs", default="/Users/fodelf/ds4-main/cap_v2r2,/Users/fodelf/ds4-main/cap_v2r1_deep")
    args = ap.parse_args()
    L = args.layer
    import shutil as _sh
    free = _sh.disk_usage("/tmp").free
    assert free > 9 * 2**30, f"/tmp free {free/2**30:.1f}G < 9G, 拒绝打包 (防 ENOSPC 半截包)"

    sys.path.insert(0, args.pyfwd)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ds4reader as R
    from gptq1_rewrite import parse_gguf, read_expert_blocks

    cfg = json.load(open(os.path.join(R.HF, "config.json")))

    cap = None
    for d in args.capdirs.split(","):
        if os.path.exists(f"{d}/ffn_in_L{L}.npy"):
            cap = d; break
    assert cap, f"L{L}: no ffn_in cap"
    shutil.copyfile(f"{cap}/ffn_in_L{L}.npy", f"{args.out}.ffn_in.npy")
    shutil.copyfile(f"{cap}/route_L{L}.npy", f"{args.out}.route.npy")

    f = open(args.gguf, "rb")
    tens, data0 = parse_gguf(f)
    KINDS = (("gate", "w1"), ("up", "w3"), ("down", "w2"))
    meta = {"layer": L, "experts": 256, "order": [k for k, _ in KINDS],
            "swiglu_limit": cfg["swiglu_limit"], "kinds": {}}
    fw8 = open(f"{args.out}.w8", "wb"); fsi = open(f"{args.out}.si", "wb")
    fs = open(f"{args.out}.s", "wb");   fb = open(f"{args.out}.b", "wb")
    for kind, hfw in KINDS:
        ne, ty, toff = tens[f"blk.{L}.ffn_{kind}_exps.weight"]
        assert ty == 40
        cols, rows = int(ne[0]), int(ne[1])
        nblk = cols // 256
        sishape = None
        for e in range(256):
            nm = f"layers.{L}.ffn.experts.{e}.{hfw}.weight"
            a, shape = R.raw_bytes(nm)
            assert tuple(shape) == (rows, cols), f"{nm} {shape}"
            fw8.write(a.tobytes())
            sc, sshape = R.raw_bytes(nm.replace(".weight", ".scale"))
            sishape = list(sc.shape)
            sb = sc.astype(np.float32).tobytes()
            assert len(sb) == (rows // 128) * (cols // 128) * 4, \
                f"L{L} {kind} e{e} scale {sc.shape} {sc.dtype} -> {len(sb)}B (期望 {(rows//128)*(cols//128)*4}B)"
            fsi.write(sb)
            raw, _ = read_expert_blocks(f, data0, toff, e, rows, cols)
            fs.write(raw[:, :, 0:2].tobytes())
            fb.write(raw[:, :, 2:34].tobytes())
            if e % 64 == 0:
                print(f"pack L{L} {kind} {e}/256", flush=True)
        meta["kinds"][kind] = {"rows": rows, "cols": cols, "nblk": nblk, "si_shape": sishape}
    for h in (fw8, fsi, fs, fb): h.close()

    import ds4reader as R2
    np.save(f"{args.out}.check.npy",
            R2.read_weight(f"layers.{L}.ffn.experts.0.w1.weight").astype(np.float32))
    json.dump(meta, open(f"{args.out}.meta.json", "w"))
    print(f"PACK-OK L{L} -> {args.out}.*", flush=True)

if __name__ == "__main__":
    main()
