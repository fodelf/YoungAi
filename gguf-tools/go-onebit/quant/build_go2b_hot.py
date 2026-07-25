#!/usr/bin/env python3
"""build_go2b_hot.py — ★等体积 go2b 热专家侧车(2026-07-24, 用户漂移方案)★。
只对热 top-K 专家(prog_active 表)编码 go2b(2-bit 合并 base+残差), 冷专家不进(留 go1b base)。
★关键: encode_go2b 喂真实激活 X(cap_algo/raw_ffn_in) + DS4_GO2B_ACT_SCALE=1 → 输出最优,
+26% 输出误差改善(往返实测); 历史 Xh=None 版是"恒等"弱因。lut: 热专家→slot, 冷→-1。
用法: DS4_GO2B_ACT_SCALE=1 build_go2b_hot.py OUT.gguf L1,L2,... ACTIVE.txt CAPDIR
两遍: ①头部(全层维度已知, 无需数据) ②数据遍逐层实时从 M1 流式 HF pack → encode_go2b → 写 → 删pack。
峰值磁盘 = 输出(增至~26G) + 单层瞬态 pack(6.4G)。用法: build_go2b_combined.py OUT.gguf L1,L2,...
"""
import os, sys, json, struct, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib", "pyfwd"))
from go2b_encode import encode_go2b
from ds4reader import LUT

GO2B, F32 = 41, 0
DIMS = {"gate": (2048, 4096), "up": (2048, 4096), "down": (4096, 2048)}
M1 = "192.168.1.2"; M1ROOT = "/Users/fodelf/ds4-main"

def stream_pack(L):
    pk = f"/tmp/sc_L{L}"
    for part in ("w8", "si"):
        cmd = f"cd {M1ROOT} && DS4_HF={M1ROOT}/hf/DeepSeek-V4-Flash-Base python3 gguf-tools/go-onebit/quant/pack_stream.py --layer {L} --part {part}"
        with open(f"{pk}.{part}", "wb") as fo:
            subprocess.run(["ssh", "-o", "BatchMode=yes", M1, cmd], stdout=fo, stderr=subprocess.DEVNULL, check=True)
    K = {"gate": {"rows": 2048, "cols": 4096, "nblk": 16, "si_shape": [16, 32]},
         "up":   {"rows": 2048, "cols": 4096, "nblk": 16, "si_shape": [16, 32]},
         "down": {"rows": 4096, "cols": 2048, "nblk": 8,  "si_shape": [32, 16]}}
    json.dump({"layer": L, "experts": 256, "order": ["gate", "up", "down"], "swiglu_limit": 10.0, "kinds": K}, open(f"{pk}.meta.json", "w"))
    return pk

def hf_pack(pk, ki, kind, e):
    meta = json.load(open(f"{pk}.meta.json")); K = meta["kinds"]
    r, c = K[kind]["rows"], K[kind]["cols"]; sr, scn = K[kind]["si_shape"]
    with open(f"{pk}.w8", "rb") as w8:
        off = sum(K[k]["rows"]*K[k]["cols"]*256 for k in ("gate","up","down")[:ki])
        w8.seek(off+e*r*c); a = np.frombuffer(w8.read(r*c), dtype=np.uint8).reshape(r, c)
    with open(f"{pk}.si", "rb") as si:
        soff = sum(int(np.prod(K[k]["si_shape"]))*4*256 for k in ("gate","up","down")[:ki])
        si.seek(soff+e*sr*scn*4); sc = np.frombuffer(si.read(sr*scn*4), dtype=np.float32).reshape(sr, scn)
    return LUT[a]*np.repeat(np.repeat(sc, 128, axis=0), 128, axis=1)[:r, :c]

def main():
    out_p = sys.argv[1]; layers = [int(x) for x in sys.argv[2].split(",")]
    active_path = sys.argv[3]; cap_dir = sys.argv[4] if len(sys.argv) > 4 else None
    ACT = {}   # L -> hot expert list (top-K)
    for ln in open(active_path):
        if ln.startswith("L"):
            L = int(ln.split(":")[0][1:]); ACT[L] = [int(x) for x in ln.split(":")[1].split()]
    tens = []
    for L in layers:
        Khot = len(ACT[L])
        for kind in ("gate", "up", "down"):
            r, c = DIMS[kind]; tens.append((f"blk.{L}.ffn_{kind}_exps_res.weight", [c, r, Khot], GO2B))
        tens.append((f"blk.{L}.ffn_res_lut.weight", [256], F32))
    kvs = [("ds4.residual.present", 7, b"\x01")] + [(f"ds4.residual.layer.{L}", 7, b"\x01") for L in layers]
    out = open(out_p, "wb")
    def w(b): out.write(b)
    def wstr(s): b = s.encode(); w(struct.pack("<Q", len(b))); w(b)
    w(struct.pack("<IIQQ", 0x46554747, 3, len(tens), len(kvs)))
    for k, t, v in kvs: wstr(k); w(struct.pack("<I", t)); w(v)
    def sz_of(ne, ty): return ne[1]*(ne[0]//256)*68*ne[2] if ty == GO2B else int(np.prod(ne))*4
    align = 32; roff = 0
    for nm, ne, ty in tens:
        wstr(nm); w(struct.pack("<I", len(ne)))
        for d in ne: w(struct.pack("<Q", d))
        w(struct.pack("<IQ", ty, roff)); roff += (sz_of(ne, ty)+align-1)//align*align
    hdr = out.tell(); data0 = (hdr+align-1)//align*align; w(b"\0"*(data0-hdr))
    for L in layers:
        pk = stream_pack(L)
        hot = ACT[L]                       # 热专家真实 id 列表(top-K)
        # 该层激活 X(w1/w3 输入=ffn_in 4096; w2 输入=中间 2048, act_d1d2 内部切列)
        Xh = None
        if cap_dir:
            xp = f"{cap_dir}/raw_ffn_in_L{L}"
            if os.path.isfile(xp):
                Xh = np.fromfile(xp, dtype='<f2').reshape(-1, 4096).astype(np.float32)[:256]
        for ki, kind in ((0, "gate"), (1, "up"), (2, "down")):
            for e in hot:                  # ★只热 K 专家(稀疏), file-order★
                blk, _ = encode_go2b(hf_pack(pk, ki, kind, e), Xh=Xh, mode='nf')  # ★喂激活★
                out.write(blk.astype(np.uint8).tobytes())
            pad = (align - (out.tell()-data0) % align) % align; w(b"\0"*pad)
        lut = np.full(256, -1.0, dtype=np.float32)   # ★稀疏 lut: 热→slot, 冷→-1★
        for slot, e in enumerate(hot): lut[e] = float(slot)
        out.write(lut.tobytes())
        pad = (align - (out.tell()-data0) % align) % align; w(b"\0"*pad)
        for x in ("w8", "si", "meta.json"): os.remove(f"{pk}.{x}")
        out.flush()
        print(f"L{L} hot={len(hot)} written ({out.tell()>>20}MiB)", flush=True)
    out.close()
    print(f"COMBINED-OK {out_p} {os.path.getsize(out_p)>>20}MiB layers={layers}", flush=True)

if __name__ == "__main__":
    main()
