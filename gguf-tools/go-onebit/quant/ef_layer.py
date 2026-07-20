#!/usr/bin/env python3
"""ef_layer.py — 单层 go2b EF 重编 (双机可并行, 认领制): HF pack + ffn_in → encode_go2b(EF) → 写单层 go2b 张量。
输出 /shared 或本地 OUTDIR/go2b_ef_L{L}.bin (kind-major expert-major, 68B块)。
用法: ef_layer.py --layer L --pack PK --ffn-in FFN.npy --route RT.npy --out OUT.bin
pack 缺则自动从 M1 流式 (--stream-m1)。"""
import os, sys, json, argparse, subprocess
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")   # 每 worker BLAS 单线程: 6 进程 × 多线程 BLAS 会在 10 核上超订阅 thrash
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib", "pyfwd"))
from go2b_encode import encode_go2b
from ds4reader import LUT

M1 = "192.168.1.2"; M1ROOT = "/Users/fodelf/ds4-main"
DIMS = {"gate": (2048, 4096), "up": (2048, 4096), "down": (4096, 2048)}

def ensure_pack(L, pk):
    if os.path.exists(f"{pk}.w8"): return
    for part in ("w8", "si"):
        cmd = f"cd {M1ROOT} && DS4_HF={M1ROOT}/hf/DeepSeek-V4-Flash-Base python3 gguf-tools/go-onebit/quant/pack_stream.py --layer {L} --part {part}"
        with open(f"{pk}.{part}", "wb") as fo:
            subprocess.run(["ssh", "-o", "BatchMode=yes", M1, cmd], stdout=fo, stderr=subprocess.DEVNULL, check=True)
    K = {"gate": {"rows":2048,"cols":4096,"nblk":16,"si_shape":[16,32]},
         "up":{"rows":2048,"cols":4096,"nblk":16,"si_shape":[16,32]},
         "down":{"rows":4096,"cols":2048,"nblk":8,"si_shape":[32,16]}}
    json.dump({"layer":L,"kinds":K}, open(f"{pk}.meta.json","w"))

def hf(pk, ki, kind, e):
    K = json.load(open(f"{pk}.meta.json"))["kinds"]
    r,c = K[kind]["rows"], K[kind]["cols"]; sr,scn = K[kind]["si_shape"]
    with open(f"{pk}.w8","rb") as w8:
        off = sum(K[k]["rows"]*K[k]["cols"]*256 for k in ("gate","up","down")[:ki]); w8.seek(off+e*r*c)
        a = np.frombuffer(w8.read(r*c), dtype=np.uint8).reshape(r,c)
    with open(f"{pk}.si","rb") as si:
        soff = sum(int(np.prod(K[k]["si_shape"]))*4*256 for k in ("gate","up","down")[:ki]); si.seek(soff+e*sr*scn*4)
        sc = np.frombuffer(si.read(sr*scn*4), dtype=np.float32).reshape(sr,scn)
    return LUT[a]*np.repeat(np.repeat(sc,128,0),128,1)[:r,:c]

_G = {}   # fork 子进程经 COW 继承(X/route 零拷贝共享), 不走 pickle
def _enc(t):
    ki, kind, e = t
    X = _G["X"]; route = _G["route"]; n = _G["n"]
    tk = np.unique(np.nonzero(route[:n] == e)[0])
    fired = int(len(tk))                                          # 真点火数(封顶前取, 作加权)
    if len(tk) > 2048: tk = tk[::max(1, len(tk)//2048)][:2048]   # 采样封顶: 热专家 Xh 过大→swap 卡死
    Xh = X[tk] if len(tk) >= 32 else X[:256]
    W = _G["hfd"](ki, kind, e) if _G["hfd"] else hf(_G["pack"], ki, kind, e)
    blk, Wq = encode_go2b(W, Xh, mode='nf')
    err = -1.0                       # 输出空间 rel_L2(仅 gate/up 维度对齐时算; 反映 EF 真实效果, 非权重空间 Frobenius)
    Xs = Xh[:64]
    if Xs.shape[1] == W.shape[1]:
        d = np.linalg.norm((Wq - W) @ Xs.T); b = np.linalg.norm(W @ Xs.T) + 1e-12
        err = float(d / b)
    return blk.astype(np.uint8).tobytes(), err, fired

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ffn-in", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-direct", action="store_true", help="在有本地 HF 的机器(M1)直接读 HF fp8, 不流式 pack")
    args = ap.parse_args()
    L = args.layer
    hf_direct = None
    if args.hf_direct:
        sys.path.insert(0, "/Users/fodelf/ds4-main/gguf-tools/go-onebit/calib/pyfwd")
        sys.path.insert(0, "/Users/fodelf/git/ds4-main/gguf-tools/go-onebit/calib/pyfwd")
        import ds4reader as _R
        hfw = {0: "w1", 1: "w3", 2: "w2"}
        def hf_direct(ki, kind, e):   # noqa: F811
            return _R.read_weight(f"layers.{L}.ffn.experts.{e}.{hfw[ki]}.weight").astype(np.float32)
    else:
        ensure_pack(L, args.pack)
    X = np.load(args.ffn_in).astype(np.float32)
    route = np.load(args.route).astype(np.int64)
    n = min(len(X), len(route))
    _G.update(X=X, route=route, n=n, pack=args.pack, hfd=hf_direct)
    nw = max(1, min(6, (os.cpu_count() or 4) - 2))   # 256 专家 EF 各自独立, 单核100%是瓶颈→多进程 ~6×; 6核峰值~1.2G(16G安全)
    try: mp.set_start_method("fork", force=True)
    except RuntimeError: pass
    out = open(args.out, "wb")
    with mp.Pool(nw) as pool:
        se = 0.0; sf = 0            # fired-weighted rel_err 累计(仅 gate/up 有效 err)
        for ki, kind in ((0,"gate"),(1,"up"),(2,"down")):
            for b, err, fired in pool.imap(_enc, [(ki, kind, e) for e in range(256)], chunksize=4):
                out.write(b)   # imap 保序 → 按 expert 顺序落盘, 与 kernel 读法一致; 流式写省内存
                if err >= 0 and fired > 0: se += err * fired; sf += fired
            print(f"L{L} {kind} EF done", flush=True)
    out.close()
    fid = (se / sf) if sf else -1.0   # 每层输出空间保真度(越低越好); 逐层看哪层难还原
    print(f"EF-LAYER-OK L{L} -> {args.out} ({os.path.getsize(args.out)>>20}MiB, {nw}核) rel_err={fid:.4f}", flush=True)

if __name__ == "__main__":
    main()
