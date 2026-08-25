#!/usr/bin/env python3
"""dspark_refcheck.py — DSpark drafter 分段对拍(引擎 dump vs HF 原始权重 numpy 参考)。

用法: python3 dspark_refcheck.py <HF_DIR> <DUMP_DIR> <pos>
分段: ①main_x = main_norm(main_proj(main_hidden))  ②窗行 = rope(kv_norm(wkv_b(main_x)))
第一个 cos<0.98 的段即 bug 段。权重从 HF fp8/bf16 直读(非量化), 量化差留 ~0.99 余量。
"""
import json, struct, sys
import numpy as np

HF, DUMP, POS = sys.argv[1], sys.argv[2], int(sys.argv[3])

_idx = json.load(open(f"{HF}/model.safetensors.index.json"))["weight_map"]
_shards = {}

def st_tensor(name):
    f = _shards.get(_idx[name])
    if f is None:
        fp = open(f"{HF}/{_idx[name]}", "rb")
        n, = struct.unpack("<Q", fp.read(8))
        hdr = json.loads(fp.read(n))
        base = 8 + n
        _shards[_idx[name]] = (fp, hdr, base)
        f = _shards[_idx[name]]
    fp, hdr, base = f
    info = hdr[name]
    off0, off1 = info["data_offsets"]
    fp.seek(base + off0)
    raw = fp.read(off1 - off0)
    shape = info["shape"]
    dt = info["dtype"]
    if dt == "BF16":
        a = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return a.view(np.float32).reshape(shape).astype(np.float32)
    if dt == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    if dt == "F8_E4M3":
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
    if dt == "F8_E8M0":
        b = np.frombuffer(raw, dtype=np.uint8).reshape(shape).astype(np.int32)
        return np.power(2.0, b - 127).astype(np.float32)
    raise SystemExit(f"dtype {dt} unsupported: {name}")

# e4m3 查表
_e4m3 = np.zeros(256, dtype=np.float32)
for b in range(256):
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 3) & 0xF
    m = b & 7
    if e == 0:
        v = (m / 8.0) * 2.0 ** (-6)
    elif e == 15 and m == 7:
        v = float("nan")
    else:
        v = (1 + m / 8.0) * 2.0 ** (e - 7)
    _e4m3[b] = s * v

def fp8_weight(name):
    w = _e4m3[st_tensor(name + ".weight")]           # [out, in]
    sc = st_tensor(name + ".scale").astype(np.float32)  # [out/128, in/128] (ue8m0→f32 已是数值)
    O, I = w.shape
    sb = np.repeat(np.repeat(sc, 128, axis=0)[:O], 128, axis=1)[:, :I]
    return w * sb

def rms(x, w, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w

def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

mh = np.fromfile(f"{DUMP}/mh_pos{POS}.bin", dtype=np.float32)      # [3*4096]
mx_eng = np.fromfile(f"{DUMP}/mx_pos{POS}.bin", dtype=np.float32)  # [4096]

# ① main_x
Wp = fp8_weight("mtp.0.main_proj")           # [4096, 12288]
mn = st_tensor("mtp.0.main_norm.weight")     # [4096]
mx_ref = rms(Wp @ mh, mn)
print(f"① main_x cos = {cos(mx_ref, mx_eng):.6f}  |ref|={np.linalg.norm(mx_ref):.4g} |eng|={np.linalg.norm(mx_eng):.4g}")

# ② 窗行(3 块层, rope 前后各对一次)
def rope_tail(v, pos, n_rot=64, base=10000.0):
    # 引擎 rope_tail_ext_inplace 语义: 尾部 n_rot, 相邻配对 (i, i+1),
    # theta 从 pos 起每对乘 theta_scale=base^(-2/n_rot)
    out = v.copy()
    d = v.shape[-1]
    t = out[d - n_rot:].copy()
    theta = float(pos)
    ts = base ** (-2.0 / n_rot)
    for i in range(0, n_rot, 2):
        c, s = np.cos(theta), np.sin(theta)
        x0, x1 = t[i], t[i + 1]
        t[i] = x0 * c - x1 * s
        t[i + 1] = x0 * s + x1 * c
        theta *= ts
    out[d - n_rot:] = t
    return out

# ②b kv_tmp(norm 前, 引擎循环最后一块=blk2)隔离 matmul
try:
    kvt_eng = np.fromfile(f"{DUMP}/kvtmp_b2_pos{POS}.bin", dtype=np.float32)
    Wkv2 = fp8_weight("mtp.2.attn.wkv")
    kvt_ref = Wkv2 @ mx_ref
    kvt_ref_engmx = Wkv2 @ mx_eng
    print(f"②b blk2 kv_tmp cos = {cos(kvt_ref, kvt_eng):.6f} (用 eng mx: {cos(kvt_ref_engmx, kvt_eng):.6f})")
    kn2 = st_tensor("mtp.2.attn.kv_norm.weight")
    win2_eng = np.fromfile(f"{DUMP}/winrow_b2_pos{POS}.bin", dtype=np.float32)
    ref_norm_engkvt = rms(kvt_eng, kn2)
    print(f"②c blk2 norm(引擎kvt)→vs 窗行: cos={cos(ref_norm_engkvt, win2_eng):.6f}"
          f"  cos(带rope)={cos(rope_tail(ref_norm_engkvt, POS), win2_eng):.6f}")
except FileNotFoundError:
    pass

for b in range(3):
    Wkv = fp8_weight(f"mtp.{b}.attn.wkv")            # [512, 4096]
    kn = st_tensor(f"mtp.{b}.attn.kv_norm.weight")   # [512]
    kv = rms(Wkv @ mx_ref, kn)
    win_eng = np.fromfile(f"{DUMP}/winrow_b{b}_pos{POS}.bin", dtype=np.float32)
    kv_rope = rope_tail(kv, POS)
    print(f"② blk{b} 窗行 cos(rope 后) = {cos(kv_rope, win_eng):.6f}  cos(rope 前) = {cos(kv, win_eng):.6f}")


# ================= ③ q 链对拍(块 embed→hc_pre→attn_norm→wq_a/q_norm/wq_b→head_rms→rope) ==
def hc4_split(mix, scale, base, iters=20, epsv=1e-6):
    out = np.zeros(24, dtype=np.float64)
    out[:4] = 1.0 / (1.0 + np.exp(-(mix[:4] * scale[0] + base[:4]))) + epsv
    out[4:8] = 2.0 / (1.0 + np.exp(-(mix[4:8] * scale[1] + base[4:8])))
    c = (mix[8:24] * scale[2] + base[8:24]).reshape(4, 4)
    c = np.exp(c - c.max(axis=1, keepdims=True))
    c = c / c.sum(axis=1, keepdims=True) + epsv
    c = c / (c.sum(axis=0, keepdims=True) + epsv)
    for _ in range(1, iters):
        c = c / (c.sum(axis=1, keepdims=True) + epsv)
        c = c / (c.sum(axis=0, keepdims=True) + epsv)
    out[8:24] = c.reshape(-1)
    return out.astype(np.float32)

def q_chain_ref(token_id, pos_q):
    emb_name = "embed.weight" if "embed.weight" in _idx else "model.embed_tokens.weight"
    # 只读 1 行 embed
    fp, hdr, base_off = None, None, None
    f = _shards.get(_idx[emb_name])
    if f is None:
        fp2 = open(f"{HF}/{_idx[emb_name]}", "rb")
        n, = struct.unpack("<Q", fp2.read(8))
        hdr2 = json.loads(fp2.read(n))
        _shards[_idx[emb_name]] = (fp2, hdr2, 8 + n)
    fp, hdr, base_off = _shards[_idx[emb_name]]
    info = hdr[emb_name]
    O, I = info["shape"]
    off0 = info["data_offsets"][0]
    fp.seek(base_off + off0 + token_id * I * 2)
    raw = fp.read(I * 2)
    a = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    e = a.view(np.float32).astype(np.float32)
    hc = np.tile(e, (4, 1))                       # [4, 4096]
    # hc_pre(attn)
    flat = hc.reshape(-1)
    flatn = flat / np.sqrt(np.mean(flat * flat) + RMS_EPS)
    fn = st_tensor("mtp.0.hc_attn_fn")            # [24, 16384]
    sc = st_tensor("mtp.0.hc_attn_scale")
    bs = st_tensor("mtp.0.hc_attn_base")
    mix = fn @ flatn
    sp = hc4_split(mix, sc, bs)
    attn_cur = (hc * sp[:4, None]).sum(axis=0)
    an = st_tensor("mtp.0.attn_norm.weight")
    x = rms(attn_cur, an, RMS_EPS)
    Wqa = fp8_weight("mtp.0.attn.wq_a")           # [1024, 4096]
    qn = st_tensor("mtp.0.attn.q_norm.weight")
    qr = rms(Wqa @ x, qn, RMS_EPS)
    Wqb = fp8_weight("mtp.0.attn.wq_b")           # [32768, 1024]
    q = (Wqb @ qr).reshape(64, 512)
    q = q / np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + RMS_EPS)
    q = np.stack([rope_tail(q[h], pos_q) for h in range(64)])
    return q.reshape(-1)

RMS_EPS = 1e-6
NOISE = 128799
try:
    q_eng = np.fromfile(f"{DUMP}/q_b0_pos{POS}.bin", dtype=np.float32).reshape(5, -1)
    import re as _re
    # anchor id 不易复原, 用 noise 位(位置 1, 输入恒 noise token)对拍
    q_ref1 = q_chain_ref(NOISE, POS + 2)          # 块位 1 的 rope pos = pos+2
    print(f"③ q(块位1=noise) cos = {cos(q_ref1, q_eng[1]):.6f}  |ref|={np.linalg.norm(q_ref1):.4g} |eng|={np.linalg.norm(q_eng[1]):.4g}")
except FileNotFoundError as e:
    print("③ skip:", e)

# ④ attention heads 对拍(blk0, 全 5 位): 参考重算 scores+softmax+V
try:
    heads_eng = np.fromfile(f"{DUMP}/heads_b0_pos{POS}.bin", dtype=np.float32).reshape(5, 64, 512)
    q_eng5 = np.fromfile(f"{DUMP}/q_b0_pos{POS}.bin", dtype=np.float32).reshape(5, 64, 512)
    blkkv = np.fromfile(f"{DUMP}/blkkv_b0_pos{POS}.bin", dtype=np.float32).reshape(5, 512)
    # 窗: 参考复算 0..POS 行? 引擎窗行历史多 pos 生成——只用引擎窗 dump 当前行不够。
    # 近似: 读引擎整窗(需 dump)——改用 sink+块内-only 对拍(n_win 贡献待查):
    sink = st_tensor("mtp.0.attn.attn_sink").astype(np.float32)
    n_win_r = min(POS + 1, 128)
    # 引擎整窗 dump
    win_eng = np.fromfile(f"{DUMP}/winfull_b0_pos{POS}.bin", dtype=np.float32).reshape(128, 512)
    scale = 1.0 / np.sqrt(512.0)
    ref = np.zeros_like(heads_eng)
    for t in range(5):
        for h in range(64):
            qv = q_eng5[t, h]
            sc_w = win_eng[:n_win_r] @ qv * scale
            sc_b = blkkv @ qv * scale
            sc = np.concatenate([sc_w, sc_b])
            m = max(sc.max(), sink[h])
            e = np.exp(sc - m)
            den = e.sum() + np.exp(sink[h] - m)
            w = e / den
            ref[t, h] = w[:n_win_r] @ win_eng[:n_win_r] + w[n_win_r:] @ blkkv
    print(f"④ heads(blk0) cos = {cos(ref, heads_eng):.6f}")
except FileNotFoundError as e:
    print("④ skip:", e)

# ⑤ markov 链参考: markov 前 logits(引擎) + HF bf16 markov 权重 → 模拟链 → draft_ref
try:
    lg5 = np.fromfile(f"{DUMP}/logits_premarkov_pos{POS}.bin", dtype=np.float32).reshape(5, -1)
    W1 = st_tensor("mtp.2.markov_head.markov_w1.weight")   # [129280, 256]
    W2 = st_tensor("mtp.2.markov_head.markov_w2.weight")   # [129280, 256]
    ANCHOR = int(sys.argv[4]) if len(sys.argv) > 4 else None
    if ANCHOR is not None:
        prev = ANCHOR
        draft_ref = []
        for i in range(5):
            bias = W2 @ W1[prev]
            nid = int(np.argmax(lg5[i] + bias))
            draft_ref.append(nid)
            prev = nid
        top_nb = int(np.argmax(lg5[0]))
        print(f"⑤ markov 链 draft_ref={draft_ref}  (纯 head top1 位0={top_nb})")
except FileNotFoundError as e:
    print("⑤ skip:", e)
