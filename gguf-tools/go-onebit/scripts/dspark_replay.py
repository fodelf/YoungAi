#!/usr/bin/env python3
"""dspark_replay.py — drafter 首分歧二分针(2026-08-20 战役: 三方 avg_acc=1.05 判引擎语义 bug)。
官方语义(hf/inference/model.py DSparkAttention)逐级重放 × 引擎 DS4_DSPARK_DUMP 对拍。
针1: mx → wkv → kv_norm → rope(pos) vs winrow_b0_pos{P}.bin (drafter 入口段单点铁证)。
用法: dspark_replay.py <hf_dir> <dump_dir> <pos> [--stage winrow|q|logits]
"""
import json, struct, sys, os
import numpy as np

HF, DUMP, POS = sys.argv[1], sys.argv[2], int(sys.argv[3])
STAGE = sys.argv[4].split("=")[-1] if len(sys.argv) > 4 else "winrow"
WM = json.load(open(HF + "/model.safetensors.index.json"))["weight_map"]
CFG = json.load(open(HF + "/inference/config.json"))
EPS = CFG.get("norm_eps", 1e-6); RD = CFG.get("qk_rope_head_dim", 64)
THETA = CFG.get("rope_theta", 10000.0)

def st_load(k):
    shard = HF + "/" + WM[k]; f = open(shard, "rb")
    hl = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(hl))
    o = hdr[k]["data_offsets"]; f.seek(8 + hl + o[0]); raw = f.read(o[1] - o[0])
    return hdr[k]["dtype"], hdr[k]["shape"], raw

_E4M3 = None
def e4m3_lut():
    global _E4M3
    if _E4M3 is None:
        v = np.zeros(256, dtype=np.float32)
        for b in range(256):
            s = -1.0 if b & 0x80 else 1.0; e = (b >> 3) & 0xF; mfrac = b & 7
            if e == 0: v[b] = s * mfrac * 2.0 ** -9
            elif e == 0xF and mfrac == 7: v[b] = np.nan
            else: v[b] = s * (1 + mfrac / 8.0) * 2.0 ** (e - 7)
        _E4M3 = v
    return _E4M3

def load_fp8(k):
    dt, sh, raw = st_load(k)
    assert dt == "F8_E4M3", dt
    w = e4m3_lut()[np.frombuffer(raw, dtype=np.uint8)].reshape(sh)
    sdt, ssh, sraw = st_load(k.replace(".weight", ".scale"))
    se = np.frombuffer(sraw, dtype=np.uint8).astype(np.int32).reshape(ssh)
    scale = np.exp2(se - 127).astype(np.float32)
    br, bc = sh[0] // ssh[0], sh[1] // ssh[1]
    return (w.reshape(ssh[0], br, ssh[1], bc) * scale[:, None, :, None]).reshape(sh).astype(np.float32)

def load_bf16(k):
    dt, sh, raw = st_load(k)
    assert dt == "BF16", dt
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(sh).copy()

def rms_norm(x, w, eps=EPS):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

def rope_tail(x, pos):
    # 官方 apply_rotary_emb: 尾 RD 维视为复数对(相邻配对), freqs = theta^(-2i/RD)
    t = x[..., -RD:].reshape(*x.shape[:-1], RD // 2, 2).astype(np.float32)
    inv = THETA ** (-np.arange(0, RD, 2, dtype=np.float64) / RD)
    ang = pos * inv
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    out = np.empty_like(t)
    out[..., 0] = t[..., 0] * cos - t[..., 1] * sin
    out[..., 1] = t[..., 0] * sin + t[..., 1] * cos
    y = x.copy(); y[..., -RD:] = out.reshape(*x.shape[:-1], RD)
    return y

def rd_dump(name):
    p = os.path.join(DUMP, name)
    return np.fromfile(p, dtype=np.float32)

mx = rd_dump(f"mx_pos{POS}.bin")            # [4096] 引擎 main_x(已 main_norm)
if STAGE == "winrow":
    wkv = load_fp8("mtp.0.attn.wkv.weight")             # [512,4096]
    kvn = load_bf16("mtp.0.attn.kv_norm.weight")        # [512]
    ref = rope_tail(rms_norm(wkv @ mx, kvn), POS)       # 官方: rope@start_pos=POS
    eng = rd_dump(f"winrow_b0_pos{POS}.bin")            # [512]
    def cmp(a, b, tag):
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        r = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
        print(f"{tag}: cos={c:.6f} rel_l2={r:.4f} eng[:4]={b[:4]} ref[:4]={a[:4]}")
    cmp(ref, eng, f"winrow pos={POS}")
    # 位置敏感性: rope 位置差 1 的对照
    for dp in (-1, 1):
        cmp(rope_tail(rms_norm(wkv @ mx, kvn), POS + dp), eng, f"  rope@pos{POS + dp:+d}")

if STAGE in ("q", "blkkv", "heads"):
    B = 5
    attncur = rd_dump(f"attncur_b0_pos{POS}.bin").reshape(B, 4096)
    anw = load_bf16("mtp.0.attn_norm.weight")
    xn = rms_norm(attncur, anw)                          # [B,4096]
    def cmp2(a, b, tag):
        for i in range(a.shape[0]):
            c = float(np.dot(a[i].ravel(), b[i].ravel()) /
                      (np.linalg.norm(a[i]) * np.linalg.norm(b[i]) + 1e-12))
            r = float(np.linalg.norm(a[i] - b[i]) / (np.linalg.norm(b[i]) + 1e-12))
            print(f"{tag} row{i}: cos={c:.6f} rel_l2={r:.4f}")
    if STAGE == "blkkv":
        wkv = load_fp8("mtp.0.attn.wkv.weight"); kvn = load_bf16("mtp.0.attn.kv_norm.weight")
        ref = np.stack([rope_tail(rms_norm(xn[i] @ wkv.T, kvn), POS + 1 + i) for i in range(B)])
        eng = rd_dump(f"blkkv_b0_pos{POS}.bin").reshape(B, 512)
        cmp2(ref, eng, "blkkv")
    if STAGE == "q":
        wqa = load_fp8("mtp.0.attn.wq_a.weight"); qn = load_bf16("mtp.0.attn.q_norm.weight")
        wqb = load_fp8("mtp.0.attn.wq_b.weight")
        NH = 64; HD = 512
        qr = rms_norm(xn @ wqa.T, qn)                     # [B,1024]
        q = (qr @ wqb.T).reshape(B, NH, HD)
        q = q / np.sqrt((q * q).mean(-1, keepdims=True) + EPS)
        ref = np.stack([rope_tail(q[i], POS + 1 + i) for i in range(B)])
        eng = rd_dump(f"q_b0_pos{POS}.bin").reshape(B, NH, HD)
        cmp2(ref.reshape(B, -1), eng.reshape(B, -1), "q")
    if STAGE == "heads":
        # 用引擎自己的 q/winfull/blkkv 输入, 只对拍 attention 本体(sink softmax)
        NH = 64; HD = 512
        q = rd_dump(f"q_b0_pos{POS}.bin").reshape(B, NH, HD)
        winfull = rd_dump(f"winfull_b0_pos{POS}.bin").reshape(128, HD)
        blkkv = rd_dump(f"blkkv_b0_pos{POS}.bin").reshape(B, HD)
        dt, sh, raw = st_load("mtp.0.attn.attn_sink")
        sink = (np.frombuffer(raw, dtype=np.float32).copy() if dt == "F32"
                else (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).copy())
        n_win = min(128, POS + 1)
        keys = np.concatenate([winfull[:n_win], blkkv])   # [n_win+5, 512]
        scale = HD ** -0.5
        ref = np.zeros((B, NH, HD), dtype=np.float32)
        for t in range(B):
            for h in range(NH):
                s = keys @ q[t, h] * scale
                mx_ = max(float(s.max()), float(sink[h]))
                e = np.exp(s - mx_); den = e.sum() + np.exp(sink[h] - mx_)
                ref[t, h] = (e[:, None] * keys).sum(0) / den
        eng = rd_dump(f"heads_b0_pos{POS}.bin").reshape(B, NH, HD)
        cmp2(ref.reshape(B, -1), eng.reshape(B, -1), "heads")

if STAGE == "headswhy":
    # 归因: 哪个假设复现引擎 heads? (a)官方式 (b)无sink (c)n_win=128全窗 (d)scale带mscale?
    B = 5; NH = 64; HD = 512
    q = rd_dump(f"q_b0_pos{POS}.bin").reshape(B, NH, HD)
    winfull = rd_dump(f"winfull_b0_pos{POS}.bin").reshape(128, HD)
    blkkv = rd_dump(f"blkkv_b0_pos{POS}.bin").reshape(B, HD)
    dt, sh, raw = st_load("mtp.0.attn.attn_sink")
    sink = (np.frombuffer(raw, dtype=np.float32).copy() if dt == "F32"
            else (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).copy())
    print("sink dtype:", dt, "sink[:4]=", sink[:4])
    eng = rd_dump(f"heads_b0_pos{POS}.bin").reshape(B, NH, HD)
    def attn_ref(n_win, use_sink, scale):
        keys = np.concatenate([winfull[:n_win], blkkv])
        ref = np.zeros((B, NH, HD), dtype=np.float32)
        for t in range(B):
            for h in range(NH):
                s = keys @ q[t, h] * scale
                mx_ = max(float(s.max()), float(sink[h]) if use_sink else -1e30)
                e = np.exp(s - mx_)
                den = e.sum() + (np.exp(sink[h] - mx_) if use_sink else 0.0)
                ref[t, h] = (e[:, None] * keys).sum(0) / den
        return ref
    for tag, r in (("official", attn_ref(min(128, POS + 1), True, HD ** -0.5)),
                   ("nosink", attn_ref(min(128, POS + 1), False, HD ** -0.5)),
                   ("fullwin128", attn_ref(128, True, HD ** -0.5)),
                   ("scale2x", attn_ref(min(128, POS + 1), True, 2.0 * HD ** -0.5))):
        c = float(np.dot(r.ravel(), eng.ravel()) / (np.linalg.norm(r) * np.linalg.norm(eng) + 1e-12))
        rl = float(np.linalg.norm(r - eng) / (np.linalg.norm(eng) + 1e-12))
        print(f"{tag}: cos={c:.6f} rel_l2={rl:.4f}")

if STAGE == "attnout":
    B = 5; NH = 64; HD = 512; G = 8; RANK = 1024
    heads = rd_dump(f"heads_b0_pos{POS}.bin").reshape(B, NH, HD)
    # 官方: apply_rotary_emb(o[..., -rd:], freqs_cis, True) — 逆旋转 at 块位置
    def rope_tail_inv(x, pos):
        t = x[..., -RD:].reshape(*x.shape[:-1], RD // 2, 2).astype(np.float32)
        inv = THETA ** (-np.arange(0, RD, 2, dtype=np.float64) / RD)
        ang = pos * inv
        cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        out = np.empty_like(t)
        out[..., 0] = t[..., 0] * cos + t[..., 1] * sin
        out[..., 1] = -t[..., 0] * sin + t[..., 1] * cos
        y = x.copy(); y[..., -RD:] = out.reshape(*x.shape[:-1], RD)
        return y
    o = np.stack([rope_tail_inv(heads[i], POS + 1 + i) for i in range(B)])  # [B,NH,HD]
    wo_a = load_fp8("mtp.0.attn.wo_a.weight").reshape(G, RANK, NH // G * HD)  # [8,1024,4096]
    wo_b = load_fp8("mtp.0.attn.wo_b.weight")                                 # [4096,8192]
    og = o.reshape(B, G, NH // G * HD)
    r = np.einsum("bgd,grd->bgr", og, wo_a).reshape(B, G * RANK)
    ref = r @ wo_b.T                                                          # [B,4096]
    eng = rd_dump(f"attnout_b0_pos{POS}.bin").reshape(B, 4096)
    for i in range(B):
        c = float(np.dot(ref[i], eng[i]) / (np.linalg.norm(ref[i]) * np.linalg.norm(eng[i]) + 1e-12))
        rl = float(np.linalg.norm(ref[i] - eng[i]) / (np.linalg.norm(eng[i]) + 1e-12))
        print(f"attnout row{i}: cos={c:.6f} rel_l2={rl:.4f}")

if STAGE == "full":
    # 全链官方语义重放(HF 原始权重=FP 教师): mx+winfull(引擎已验) → 3×DSparkBlock → 出口 → draft ids
    # 判据: FP 权重在 q2 主干 mh 上的命中天花板 vs 引擎 draft vs 真实 next
    B = 5; NH = 64; HD = 512; HC = 4; DIM = 4096; MIX = (2 + HC) * HC   # 24
    TOPK = 6; RSCALE = 1.5; SWLIM = 10.0; ITERS = 20; HCEPS = CFG.get("hc_eps", 1e-6)
    NOISE = 128799
    ANCHOR = int(open(os.path.join(DUMP, f"draft_pos{POS}.txt")).read().split()[0])
    mx = rd_dump(f"mx_pos{POS}.bin")

    def st_rows(k, rows):
        """按行读大矩阵(bf16/f32), 只取 rows"""
        dt, sh, raw_off = None, None, None
        shard = HF + "/" + WM[k]; f = open(shard, "rb")
        hl = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(hl))
        o = hdr[k]["data_offsets"]; dt = hdr[k]["dtype"]; sh = hdr[k]["shape"]
        esz = {"BF16": 2, "F32": 4}[dt]; rowb = sh[1] * esz
        out = np.zeros((len(rows), sh[1]), dtype=np.float32)
        for i, r in enumerate(rows):
            f.seek(8 + hl + o[0] + r * rowb); b = f.read(rowb)
            if dt == "BF16":
                out[i] = ((np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32))
            else:
                out[i] = np.frombuffer(b, dtype=np.float32)
        return out

    _E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
    def load_mxfp4(k):
        dt, sh, raw = st_load(k)          # I8 [R, C/2] packed
        b = np.frombuffer(raw, dtype=np.uint8).reshape(sh)
        lo = _E2M1[b & 0xF]; hi = _E2M1[b >> 4]
        w = np.stack([lo, hi], axis=-1).reshape(sh[0], sh[1] * 2)
        sdt, ssh, sraw = st_load(k.replace(".weight", ".scale"))
        se = np.frombuffer(sraw, dtype=np.uint8).astype(np.int32).reshape(ssh)
        sc = np.exp2(se - 127).astype(np.float32)
        return (w.reshape(ssh[0], 1, ssh[1], 32) * sc[:, None, :, None]).reshape(w.shape)

    def sinkhorn_split(mixes, scale, base):
        pre = 1 / (1 + np.exp(-(mixes[:, :HC] * scale[0] + base[:HC]))) + HCEPS
        post = 2 / (1 + np.exp(-(mixes[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])))
        comb = (mixes[:, 2 * HC:] * scale[2] + base[2 * HC:]).reshape(-1, HC, HC)
        comb = np.exp(comb - comb.max(-1, keepdims=True))
        comb = comb / comb.sum(-1, keepdims=True) + HCEPS
        comb = comb / (comb.sum(-2, keepdims=True) + HCEPS)
        for _ in range(ITERS - 1):
            comb = comb / (comb.sum(-1, keepdims=True) + HCEPS)
            comb = comb / (comb.sum(-2, keepdims=True) + HCEPS)
        return pre, post, comb

    def hc_pre(x, fn, scale, base):        # x [B,HC,DIM]
        flat = x.reshape(B, HC * DIM)
        rs = 1 / np.sqrt((flat * flat).mean(-1, keepdims=True) + EPS)
        mixes = flat @ fn.T * rs
        pre, post, comb = sinkhorn_split(mixes, scale, base)
        return (pre[:, :, None] * x).sum(1), post, comb

    def hc_post(y, resid, post, comb):
        return post[:, :, None] * y[:, None, :] + np.einsum("bjk,bjd->bkd", comb, resid)

    def expert_fwd(prefix, x):             # x [n,DIM] -> [n,DIM]
        w1 = load_mxfp4(prefix + "w1.weight"); w3 = load_mxfp4(prefix + "w3.weight")
        w2 = load_mxfp4(prefix + "w2.weight")
        g = np.minimum(x @ w1.T, SWLIM)
        u = np.clip(x @ w3.T, -SWLIM, SWLIM)
        return (g / (1 + np.exp(-g)) * u) @ w2.T

    def moe(n, x):                          # 官方 Gate: sqrtsoftplus + bias-topk
        gw = load_bf16(f"mtp.{n}.ffn.gate.weight")
        gb = st_load(f"mtp.{n}.ffn.gate.bias")[2]
        gb = np.frombuffer(gb, dtype=np.float32)
        scores = np.sqrt(np.log1p(np.exp(x @ gw.T)))
        sel = np.argsort(-(scores + gb), axis=-1)[:, :TOPK]
        wts = np.take_along_axis(scores, sel, 1)
        wts = wts / wts.sum(-1, keepdims=True) * RSCALE
        y = np.zeros_like(x)
        for e in sorted(set(sel.ravel().tolist())):
            rows, slots = np.where(sel == e)
            y[rows] += wts[rows, slots, None] * expert_fwd(f"mtp.{n}.ffn.experts.{e}.", x[rows])
        # shared expert (fp8)
        s1 = load_fp8(f"mtp.{n}.ffn.shared_experts.w1.weight")
        s3 = load_fp8(f"mtp.{n}.ffn.shared_experts.w3.weight")
        s2 = load_fp8(f"mtp.{n}.ffn.shared_experts.w2.weight")
        g = np.minimum(x @ s1.T, SWLIM); u = np.clip(x @ s3.T, -SWLIM, SWLIM)
        return y + (g / (1 + np.exp(-g)) * u) @ s2.T

    def attn_blk(n, x, wins):               # x [B,DIM] (attn_norm 后)
        wqa = load_fp8(f"mtp.{n}.attn.wq_a.weight"); qn = load_bf16(f"mtp.{n}.attn.q_norm.weight")
        wqb = load_fp8(f"mtp.{n}.attn.wq_b.weight")
        wkv = load_fp8(f"mtp.{n}.attn.wkv.weight"); kvn = load_bf16(f"mtp.{n}.attn.kv_norm.weight")
        dt, sh, raw = st_load(f"mtp.{n}.attn.attn_sink")
        sink = np.frombuffer(raw, dtype=np.float32) if dt == "F32" else \
            (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        qr = rms_norm(x @ wqa.T, qn)
        q = (qr @ wqb.T).reshape(B, NH, HD)
        q = q / np.sqrt((q * q).mean(-1, keepdims=True) + EPS)
        q = np.stack([rope_tail(q[i], POS + 1 + i) for i in range(B)])
        kv = np.stack([rope_tail(rms_norm(x[i] @ wkv.T, kvn), POS + 1 + i) for i in range(B)])
        n_win = min(128, POS + 1)
        keys = np.concatenate([wins[:n_win], kv])
        scale = HD ** -0.5
        o = np.zeros((B, NH, HD), dtype=np.float32)
        for t in range(B):
            for h in range(NH):
                sc = keys @ q[t, h] * scale
                m = max(float(sc.max()), float(sink[h]))
                e = np.exp(sc - m); den = e.sum() + np.exp(sink[h] - m)
                o[t, h] = (e[:, None] * keys).sum(0) / den
        # 逆 rope + wo 分组
        def rope_inv(xx, pos):
            t = xx[..., -RD:].reshape(*xx.shape[:-1], RD // 2, 2)
            inv = THETA ** (-np.arange(0, RD, 2, dtype=np.float64) / RD)
            ang = pos * inv
            c, s2 = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
            out = np.empty_like(t)
            out[..., 0] = t[..., 0] * c + t[..., 1] * s2
            out[..., 1] = -t[..., 0] * s2 + t[..., 1] * c
            y = xx.copy(); y[..., -RD:] = out.reshape(*xx.shape[:-1], RD); return y
        o = np.stack([rope_inv(o[i], POS + 1 + i) for i in range(B)])
        G = 8; RANK = 1024
        wo_a = load_fp8(f"mtp.{n}.attn.wo_a.weight").reshape(G, RANK, NH // G * HD)
        wo_b = load_fp8(f"mtp.{n}.attn.wo_b.weight")
        og = o.reshape(B, G, NH // G * HD)
        return np.einsum("bgd,grd->bgr", og, wo_a).reshape(B, G * RANK) @ wo_b.T

    # ---- forward ----
    ids = [ANCHOR] + [NOISE] * 4
    x = st_rows("embed.weight", ids)                      # [B,DIM]
    h = np.repeat(x[:, None, :], HC, axis=1)              # [B,HC,DIM]
    wins = [rd_dump(f"winfull_b{n}_pos{POS}.bin").reshape(128, HD) for n in range(3)]
    for n in range(3):
        fa = load_bf16(f"mtp.{n}.hc_attn_fn") if False else st_rows(f"mtp.{n}.hc_attn_fn", list(range(MIX)))
        sa = st_rows(f"mtp.{n}.hc_attn_scale", [0]).ravel() if False else None
        # hc 参数是 f32 1D/2D: 直接 st_load
        def ld(name):
            dt, sh, raw = st_load(f"mtp.{n}.{name}")
            a = (np.frombuffer(raw, dtype=np.float32) if dt == "F32"
                 else (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32))
            return a.reshape(sh).astype(np.float32)
        resid = h
        y, post, comb = hc_pre(h, ld("hc_attn_fn"), ld("hc_attn_scale"), ld("hc_attn_base"))
        y = rms_norm(y, load_bf16(f"mtp.{n}.attn_norm.weight"))
        y = attn_blk(n, y, wins[n])
        h = hc_post(y, resid, post, comb)
        resid = h
        y, post, comb = hc_pre(h, ld("hc_ffn_fn"), ld("hc_ffn_scale"), ld("hc_ffn_base"))
        y = rms_norm(y, load_bf16(f"mtp.{n}.ffn_norm.weight"))
        y = moe(n, y)
        h = hc_post(y, resid, post, comb)
        print(f"block {n} done, h[0,0,:2]={h[0,0,:2]}", flush=True)
    # ---- 出口 ----
    def ld2(name):
        dt, sh, raw = st_load(name)
        a = (np.frombuffer(raw, dtype=np.float32) if dt == "F32"
             else (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32))
        return a.reshape(sh).astype(np.float32)
    fn = ld2("mtp.2.hc_head_fn"); sc = ld2("mtp.2.hc_head_scale"); ba = ld2("mtp.2.hc_head_base")
    flat = h.reshape(B, HC * DIM)
    rs = 1 / np.sqrt((flat * flat).mean(-1, keepdims=True) + EPS)
    pre = 1 / (1 + np.exp(-(flat @ fn.T * rs * sc + ba))) + HCEPS
    xh = (pre[:, :, None] * h).sum(1)                     # [B,DIM]
    xh = rms_norm(xh, ld2("mtp.2.norm.weight"))
    headw_rows = None
    # lm_head 全量 [129280,4096] bf16 ≈1GB — 直接读
    dt, sh, raw = st_load("head.weight")
    headw = ((np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)).reshape(sh)
    logits = xh @ headw.T                                  # [B,V]
    # markov 链
    mw1 = "mtp.2.markov_head.markov_w1.weight"; mw2 = "mtp.2.markov_head.markov_w2.weight"
    dt2, sh2, raw2 = st_load(mw2)
    w2 = ((np.frombuffer(raw2, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)).reshape(sh2)
    prev = ANCHOR; out = []
    for i in range(B):
        emb = st_rows(mw1, [prev]).ravel()
        lg = logits[i] + w2 @ emb
        prev = int(lg.argmax()); out.append(prev)
    eng_draft = [int(t) for t in open(os.path.join(DUMP, f"draft_pos{POS}.txt")).read().split()[1:]]
    print("python draft:", out)
    print("engine draft:", eng_draft)

if STAGE == "ffnbisect":
    # 锚捕获记录(blk0, pos=POS) vs python 参考: Fin / route / routed_out 三级二分
    B = 5; HC = 4; DIM = 4096; MIX = (2 + HC) * HC
    TOPK = 6; RSCALE = 1.5; SWLIM = 10.0; ITERS = 20; HCEPS = CFG.get("hc_eps", 1e-6)
    NOISE = 128799
    ANCHOR_F = sys.argv[5] if len(sys.argv) > 5 else "/tmp/dsanchor.bin"
    raw = open(ANCHOR_F, "rb").read()
    off = 0; rec = None
    while off < len(raw):
        blk, pos, b_, k_, d_ = struct.unpack_from("<5I", raw, off); off += 20
        fin = np.frombuffer(raw, np.float32, b_ * d_, off); off += b_ * d_ * 4
        sel = np.frombuffer(raw, np.int32, b_ * k_, off); off += b_ * k_ * 4
        rw = np.frombuffer(raw, np.float32, b_ * k_, off); off += b_ * k_ * 4
        oref = np.frombuffer(raw, np.float32, b_ * d_, off); off += b_ * d_ * 4
        if blk == 0 and pos == POS:
            rec = (fin.reshape(b_, d_), sel.reshape(b_, k_), rw.reshape(b_, k_), oref.reshape(b_, d_), k_)
    assert rec, "no blk0 record at POS"
    e_fin, e_sel, e_rw, e_oref, K = rec
    print(f"engine K={K}")
    # python 参考: embed → hc_pre(attn) → attn(用引擎已验各针) …重算到 ffn_norm 太长;
    # 捷径: 引擎 attn 段全对 ⇒ 用引擎 attnout dump + python hc_post/hc_pre/ffn_norm 接力
    ANCHOR_T = int(open(os.path.join(DUMP, f"draft_pos{POS}.txt")).read().split()[0])
    ids = [ANCHOR_T] + [NOISE] * 4
    def st_rows(k, rows):
        shard = HF + "/" + WM[k]; f = open(shard, "rb")
        hl = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(hl))
        o = hdr[k]["data_offsets"]; dt = hdr[k]["dtype"]; sh = hdr[k]["shape"]
        esz = {"BF16": 2, "F32": 4}[dt]; rowb = sh[1] * esz
        out = np.zeros((len(rows), sh[1]), dtype=np.float32)
        for i, r in enumerate(rows):
            f.seek(8 + hl + o[0] + r * rowb); b = f.read(rowb)
            out[i] = (((np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32))
                      if dt == "BF16" else np.frombuffer(b, np.float32))
        return out
    def ld(name):
        dt, sh, raw2 = st_load(f"mtp.0.{name}")
        a = (np.frombuffer(raw2, np.float32) if dt == "F32"
             else (np.frombuffer(raw2, np.uint16).astype(np.uint32) << 16).view(np.float32))
        return a.reshape(sh).astype(np.float32)
    def sinkhorn_split(mixes, scale, base):
        pre = 1 / (1 + np.exp(-(mixes[:, :HC] * scale[0] + base[:HC]))) + HCEPS
        post = 2 / (1 + np.exp(-(mixes[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])))
        comb = (mixes[:, 2 * HC:] * scale[2] + base[2 * HC:]).reshape(-1, HC, HC)
        comb = np.exp(comb - comb.max(-1, keepdims=True))
        comb = comb / comb.sum(-1, keepdims=True) + HCEPS
        comb = comb / (comb.sum(-2, keepdims=True) + HCEPS)
        for _ in range(ITERS - 1):
            comb = comb / (comb.sum(-1, keepdims=True) + HCEPS)
            comb = comb / (comb.sum(-2, keepdims=True) + HCEPS)
        return pre, post, comb
    x = st_rows("embed.weight", ids)
    h = np.repeat(x[:, None, :], HC, axis=1)
    flat = h.reshape(B, HC * DIM)
    rs = 1 / np.sqrt((flat * flat).mean(-1, keepdims=True) + EPS)
    mixes = flat @ ld("hc_attn_fn").T * rs
    pre, post, comb = sinkhorn_split(mixes, ld("hc_attn_scale"), ld("hc_attn_base"))
    attnout = rd_dump(f"attnout_b0_pos{POS}.bin").reshape(B, DIM)   # 引擎(已验 0.9999)
    h = post[:, :, None] * attnout[:, None, :] + np.einsum("bjk,bjd->bkd", comb, h)
    flat = h.reshape(B, HC * DIM)
    rs = 1 / np.sqrt((flat * flat).mean(-1, keepdims=True) + EPS)
    mixes = flat @ ld("hc_ffn_fn").T * rs
    pre2, post2, comb2 = sinkhorn_split(mixes, ld("hc_ffn_scale"), ld("hc_ffn_base"))
    y = (pre2[:, :, None] * h).sum(1)
    fin_ref = rms_norm(y, load_bf16("mtp.0.ffn_norm.weight"))       # = 官方 Fin
    for i in range(B):
        c = float(np.dot(fin_ref[i], e_fin[i]) / (np.linalg.norm(fin_ref[i]) * np.linalg.norm(e_fin[i]) + 1e-12))
        print(f"Fin row{i}: cos={c:.6f} ref[:3]={fin_ref[i][:3]} eng[:3]={e_fin[i][:3]}")
    # route: 官方 gate 在引擎 Fin 上(隔离 router 本身)
    gw = load_bf16("mtp.0.ffn.gate.weight")
    gb = np.frombuffer(st_load("mtp.0.ffn.gate.bias")[2], np.float32)
    scores = np.sqrt(np.log1p(np.exp(e_fin @ gw.T)))
    sel_ref = np.sort(np.argsort(-(scores + gb), axis=-1)[:, :TOPK], axis=-1)
    sel_eng = np.sort(e_sel[:, :TOPK], axis=-1)
    for i in range(B):
        inter = len(set(sel_ref[i].tolist()) & set(sel_eng[i].tolist()))
        print(f"route row{i}: overlap={inter}/6 ref={sel_ref[i]} eng={sel_eng[i]} rw={e_rw[i][:6]}")

if STAGE == "hcpost":
    # hc_post(attn) 出口对拍: python(embed→hc_pre→[引擎attnout]→hc_post) vs 引擎 hcpost dump
    B = 5; HC = 4; DIM = 4096; ITERS = 20; HCEPS = CFG.get("hc_eps", 1e-6); NOISE = 128799
    ANCHOR_T = int(open(os.path.join(DUMP, f"draft_pos{POS}.txt")).read().split()[0])
    def st_rows(k, rows):
        shard = HF + "/" + WM[k]; f = open(shard, "rb")
        hl = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(hl))
        o = hdr[k]["data_offsets"]; dt = hdr[k]["dtype"]; sh = hdr[k]["shape"]
        esz = {"BF16": 2, "F32": 4}[dt]; rowb = sh[1] * esz
        out = np.zeros((len(rows), sh[1]), dtype=np.float32)
        for i, r in enumerate(rows):
            f.seek(8 + hl + o[0] + r * rowb); b = f.read(rowb)
            out[i] = (((np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32))
                      if dt == "BF16" else np.frombuffer(b, np.float32))
        return out
    def ld(name):
        dt, sh, raw2 = st_load(f"mtp.0.{name}")
        a = (np.frombuffer(raw2, np.float32) if dt == "F32"
             else (np.frombuffer(raw2, np.uint16).astype(np.uint32) << 16).view(np.float32))
        return a.reshape(sh).astype(np.float32)
    def sinkhorn_split(mixes, scale, base):
        pre = 1 / (1 + np.exp(-(mixes[:, :HC] * scale[0] + base[:HC]))) + HCEPS
        post = 2 / (1 + np.exp(-(mixes[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])))
        comb = (mixes[:, 2 * HC:] * scale[2] + base[2 * HC:]).reshape(-1, HC, HC)
        comb = np.exp(comb - comb.max(-1, keepdims=True))
        comb = comb / comb.sum(-1, keepdims=True) + HCEPS
        comb = comb / (comb.sum(-2, keepdims=True) + HCEPS)
        for _ in range(ITERS - 1):
            comb = comb / (comb.sum(-1, keepdims=True) + HCEPS)
            comb = comb / (comb.sum(-2, keepdims=True) + HCEPS)
        return pre, post, comb
    x = st_rows("embed.weight", [ANCHOR_T] + [NOISE] * 4)
    h = np.repeat(x[:, None, :], HC, axis=1)
    flat = h.reshape(B, HC * DIM)
    rs = 1 / np.sqrt((flat * flat).mean(-1, keepdims=True) + EPS)
    mixes = flat @ ld("hc_attn_fn").T * rs
    pre, post, comb = sinkhorn_split(mixes, ld("hc_attn_scale"), ld("hc_attn_base"))
    attnout = rd_dump(f"attnout_b0_pos{POS}.bin").reshape(B, DIM)
    ref = post[:, :, None] * attnout[:, None, :] + np.einsum("bjk,bjd->bkd", comb, h)
    eng = rd_dump(f"hcpost_b0_pos{POS}.bin").reshape(B, HC, DIM)
    for i in range(B):
        c = float(np.dot(ref[i].ravel(), eng[i].ravel()) /
                  (np.linalg.norm(ref[i]) * np.linalg.norm(eng[i]) + 1e-12))
        print(f"hcpost row{i}: cos={c:.6f} ref[0,:3]={ref[i,0,:3]} eng[0,:3]={eng[i,0,:3]}")
    # 附: attn 段 hc_pre 的 attncur 对拍(python pre-sum vs 引擎 attncur dump)
    y = (pre[:, :, None] * h).sum(1)
    engc = rd_dump(f"attncur_b0_pos{POS}.bin").reshape(B, DIM)
    for i in range(B):
        c = float(np.dot(y[i], engc[i]) / (np.linalg.norm(y[i]) * np.linalg.norm(engc[i]) + 1e-12))
        print(f"attncur row{i}: cos={c:.6f}")
