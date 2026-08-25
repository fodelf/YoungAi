#!/usr/bin/env python3
"""zlayer_fixture.py — 给 calib/zlayer.c 二期四个模式造【可本地复现的最小金标夹具】。

为什么要造夹具: 二期的四个模式(非XCAP / DS4_ZL_GGUF / XANCHOR / ADDON / ERF)全都要
真 HF 权重 + 真锚 + 真侧车才跑得起来, 而这些东西只在 spark 上。夹具把同一套输入缩到
D=4096(写死不能改)、MOEI=128、S=96、专家 3 个, 于是 zlayer.py 和 zlayer 都能在一台
Mac 上跑完, 直接对拍 —— 不用等 86G 模型。

生成的东西(默认落 migrate/zlayer_fixtures/, 不入库):
  hf/                      假 HF: model.safetensors.index.json + 一个 shard,
                           routed 专家按 0731 的真口径存(I8 容器 MXFP4 + F8_E8M0 scale)
  anchor.bin / anchor2.bin DQA2 锚(第二个给 DS4_ZL_XANCHOR 当链态锚)
  layers/dql_vq_L00.bin    VQ 侧车(真做一遍 dim=4/nc=256 的最近邻量化, 所以 Wq≈Wf,
                           dH 有真实结构 —— 纯随机的 dH 会让挽回率在 %.1f 上乱跳, 对不了拍)
  model.gguf               DS4_ZL_GGUF 用的标量底座(Q8_0; q2_K/iq2_xxs 的解码另有
                           逐位对拍, 见 zlayer_transcription_notes.md 金标节)
  layers/dql_L00.bin.tpl   ADDON/ERF 注入用的裸底座模板(含既有 bf.GE + zl.RRR 记录)

用法: python3 zlayer_fixture.py [输出目录]
"""
import json, os, struct, sys
import numpy as np

D = 4096          # zlayer.c 里写死, 不能改
MOEI = 128        # 专家隐层(要能被 32 整除: MXFP4 的 scale 是 1×32 微块)
NEXP = 3          # 只用前 3 个专家
S = 96            # 锚 token 数
NACT = 2          # 每 token 激活槽
L = 0

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "zlayer_fixtures")
FP4T = np.array([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=np.float32)


def mxfp4_encode(w):
    """f32 [R,C] → (I8 容器 [R,C/2], E8M0 scale [R,C/32])。与 st_read.c 的 I8 分支同几何。"""
    R, C = w.shape
    nblk = C // 32
    blk = w.reshape(R, nblk, 32)
    amax = np.abs(blk).max(axis=2)
    # E8M0: 值 = 2^(e-127)。取能把块内最大值压进 FP4 表上限 6.0 的最小指数。
    e = np.zeros((R, nblk), dtype=np.uint8)
    nz = amax > 0
    ex = np.zeros_like(amax)
    ex[nz] = np.ceil(np.log2(amax[nz] / 6.0))
    e[:] = np.clip(ex + 127, 1, 254).astype(np.uint8)
    scale = np.where(e == 0, np.float32(np.frombuffer(np.uint32(0x00400000).tobytes(), np.float32)[0]),
                     np.frombuffer((e.astype(np.uint32) << 23).tobytes(), np.float32).reshape(R, nblk))
    q = blk / scale[:, :, None]
    idx = np.abs(q[..., None] - FP4T.reshape(1, 1, 1, 16)).argmin(axis=3).astype(np.uint8)
    lo, hi = idx[:, :, 0::2], idx[:, :, 1::2]
    cont = (lo | (hi << 4)).reshape(R, C // 2).astype(np.uint8)
    return cont, e


def mxfp4_decode(cont, e):
    """给 fixture 自己算"FP 侧真值"用 —— 必须与 st_mxfp4 / st_read.c 完全一致"""
    R, Cc = cont.shape
    Cin = Cc * 2
    nblk = Cin // 32
    u = np.where(e.astype(np.uint32) == 0, np.uint32(0x00400000), e.astype(np.uint32) << 23)
    scale = np.frombuffer(u.astype(np.uint32).tobytes(), dtype=np.float32).reshape(R, nblk)
    b = cont.reshape(R, nblk, 16)
    w = np.empty((R, nblk, 32), dtype=np.float32)
    w[:, :, 0::2] = FP4T[b & 0x0F]
    w[:, :, 1::2] = FP4T[(b >> 4) & 0x0F]
    w *= scale[:, :, None]
    return w.reshape(R, Cin)


def write_safetensors(path, tensors):
    """tensors: {name: (dtype字符串, ndarray)}; 按 safetensors 的 8B 头长 + JSON + 数据"""
    hdr, blobs, off = {}, [], 0
    for nm, (dt, arr) in tensors.items():
        b = arr.tobytes()
        hdr[nm] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    # ★紧凑 JSON★ st_read.c 的 st_find 是 strstr("\"dtype\":\"") —— 冒号后有空格就找不到,
    # 真 safetensors 也是紧凑写的。这里跟着写紧凑, 否则 C 侧读不到权重(报"HF 权重读不到")。
    js = json.dumps(hdr, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(js)))
        f.write(js)
        for b in blobs:
            f.write(b)


def vq_pack(W, dim=4, nc=256, seed=0):
    """把 f32 [rows,cols] 打成 probe_behavior_spectrum.vq_dequant 认的载荷。
    nc=256 ⇒ nbit=8 ⇒ 索引是一字节一个(位流分支另有对拍, 这里图简单)。"""
    rows, cols = W.shape
    gr = np.abs(W).max(axis=1).astype(np.float32)
    gr[gr == 0] = 1.0
    gr16 = gr.astype(np.float16).astype(np.float32)
    Wn = (W / gr16[:, None]).reshape(-1, dim)
    r = np.random.default_rng(seed)
    cb = Wn[r.choice(Wn.shape[0], nc, replace=False)].astype(np.float16).astype(np.float32)
    idx = np.empty(Wn.shape[0], dtype=np.uint8)
    step = 4096
    for i in range(0, Wn.shape[0], step):
        chunk = Wn[i:i + step]
        d2 = ((chunk[:, None, :] - cb[None, :, :]) ** 2).sum(2)
        idx[i:i + step] = d2.argmin(1).astype(np.uint8)
    pay = struct.pack("<IHHII", 0x51565144, dim, nc, rows, cols)
    pay += cb.astype(np.float16).tobytes() + gr.astype(np.float16).tobytes() + idx.tobytes()
    return pay


def rec(nm, pay):
    h = bytearray(116)
    h[0:len(nm)] = nm.encode()
    struct.pack_into('<Q', h, 88, len(pay))
    struct.pack_into('<i', h, 112, 1)
    return bytes(h) + pay


def main():
    os.makedirs(os.path.join(OUT, "hf"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "layers"), exist_ok=True)
    rng = np.random.default_rng(20260825)

    # ---- HF 专家权重(MXFP4 存盘) ----
    tens, wmap, fp = {}, {}, {}
    for e in range(NEXP):
        for nm, shape in (("w1", (MOEI, D)), ("w3", (MOEI, D)), ("w2", (D, MOEI))):
            w = (rng.standard_normal(shape) * 0.05).astype(np.float32)
            cont, ex = mxfp4_encode(w)
            base = f"layers.{L}.ffn.experts.{e}.{nm}"
            tens[base + ".weight"] = ("I8", cont)
            tens[base + ".scale"] = ("F8_E8M0", ex)
            wmap[base + ".weight"] = "model-00001.safetensors"
            wmap[base + ".scale"] = "model-00001.safetensors"
            fp[(e, nm)] = mxfp4_decode(cont, ex)      # 落盘后的真值(= st_mxfp4 会读回来的东西)
    write_safetensors(os.path.join(OUT, "hf", "model-00001.safetensors"), tens)
    json.dump({"weight_map": wmap}, open(os.path.join(OUT, "hf", "model.safetensors.index.json"), "w"),
              separators=(",", ":"))

    # ---- 锚 ----
    def write_anchor(path, seed):
        r = np.random.default_rng(seed)
        fin = (r.standard_normal((S, D)) * 0.6).astype(np.float32)
        ridx = r.integers(0, NEXP, size=(S, NACT)).astype(np.int32)
        for t in range(S):                       # 同 token 的两个槽不能撞同一个专家
            if ridx[t, 0] == ridx[t, 1]:
                ridx[t, 1] = (ridx[t, 1] + 1) % NEXP
        rw = (0.3 + r.random((S, NACT)) * 0.7).astype(np.float32)
        hd = struct.pack("<8I", 0x32415144, S, 0, D, 1, 100, NACT, 0)
        with open(path, "wb") as f:
            f.write(hd)
            f.write(b"\0" * (40 - len(hd)) if len(hd) < 40 else b"")
            f.write(fin.tobytes())
            f.write(ridx.tobytes())
            f.write(rw.tobytes())
    write_anchor(os.path.join(OUT, "anchor.bin"), 7)
    write_anchor(os.path.join(OUT, "anchor2.bin"), 8)

    # ---- VQ 侧车 ----
    slots = np.zeros(256 * 3, dtype=np.uint64)
    body, off = b"", 16 + 256 * 3 * 8
    for e in range(NEXP):
        for wi, nm in ((0, "w1"), (1, "w3"), (2, "w2")):
            pay = vq_pack(fp[(e, nm)], seed=e * 3 + wi)
            slots[e * 3 + wi] = off
            body += pay
            off += len(pay)
    with open(os.path.join(OUT, "layers", f"dql_vq_L{L:02d}.bin"), "wb") as f:
        f.write(struct.pack("<IIII", 0x4C565144, 0, 0, 0))
        f.write(slots.tobytes())
        f.write(body)

    # ---- GGUF 标量底座(Q8_0) ----
    from gguf import GGUFWriter
    from gguf.constants import GGMLQuantizationType as T
    from gguf.quants import quantize
    gw = GGUFWriter(os.path.join(OUT, "model.gguf"), "ds4-fixture")
    for gname, key, shape in (("ffn_gate_exps", "w1", (NEXP, MOEI, D)),
                              ("ffn_up_exps", "w3", (NEXP, MOEI, D)),
                              ("ffn_down_exps", "w2", (NEXP, D, MOEI))):
        stack = np.stack([fp[(e, key)] for e in range(NEXP)], 0).astype(np.float32)
        q = quantize(stack.reshape(-1, shape[2]), T.Q8_0)     # [专家*行, 每行字节]
        # GGUFWriter 吃的是 numpy 序(外维在前)的【字节】形状, 内部再翻成 ne 并换算元素数
        gw.add_tensor(f"blk.{L}.{gname}.weight", q.reshape(shape[0], shape[1], -1),
                      raw_dtype=T.Q8_0)
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    # ---- ADDON/ERF 注入用的 dql 裸底座模板(带一条既有 GE + 一条既有 z) ----
    k0 = 8
    ge_old = (1.0 + rng.standard_normal(256).astype(np.float32) * 0.02).astype(np.float16)
    z0 = np.abs(rng.standard_normal(k0)).astype(np.float16)
    U0 = (rng.standard_normal((D, k0)) * 0.01).astype(np.float16)
    V0 = (rng.standard_normal((D, k0)) * 0.01).astype(np.float16)
    body = rec("bf.GE", ge_old.tobytes())
    body += rec("zl.RRR", struct.pack('<IfII', k0, 0.5, D, D)
                + z0.tobytes() + np.ascontiguousarray(U0).tobytes() + np.ascontiguousarray(V0).tobytes())
    with open(os.path.join(OUT, "layers", f"dql_L{L:02d}.bin.tpl"), "wb") as f:
        f.write(struct.pack("<III", 0x314C5144, 0, 2))     # [magic][?][nrec=2]
        f.write(body)

    print(f"夹具就绪: {OUT}")
    print(f"  D={D} MOEI={MOEI} 专家={NEXP} S={S} NACT={NACT}")


if __name__ == "__main__":
    main()
