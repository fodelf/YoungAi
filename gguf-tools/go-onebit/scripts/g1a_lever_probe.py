#!/usr/bin/env python3
"""g1a_lever_probe.py — v2.1 杠杆 G1a 量化器侧分钟级 A/B(2026-07-25, 深研收割后首验)。
不动引擎; 真实 HF FP8 专家权重 + 真实校准激活 X(530tok 编程语料捕获)。

热路(2-bit, 68B/256=2.125bpw 等体积):
  A0 go2b 生产同款(GPTQ+2轮激活最优)   A1 +块对角 sequency-Walsh-256 旋转   A2 +自然序 Hadamard-256
冷路(1-bit, 34B/256=1.0625bpw 等体积, 统一拟合机器=sign+激活锚定 ridge scale):
  B0 行scale(生产 signref 布局)  B0b 真 per-block scale(槽位本就存在=免费)
  B1 Haar-1 分带(块scale=低带, 行ρ×高带)  B2 Haar-2 三带  B4 B1+1%显著列q8钉扎(bpw 单列)
判据: 层输出相对误差 relF=||Y-Ŷ||/||Y|| 与 cos(Y,Ŷ); Y=X·Wᵀ (w2 用真实中间激活)。
用法: g1a_lever_probe.py --layer 5 --x /tmp/xr_x_L05.npy [--hf DIR] [--nh 3 --nc 3] [--kinds w1,w2]
"""
import os, sys, json, struct, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "quant"))
from go2b_encode import encode_go2b, decode_go2b

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", file=sys.stderr, flush=True)

# ---------- HF fp8 safetensors 读取 ----------
_F8 = None
def f8lut():
    global _F8
    if _F8 is None:
        v = np.zeros(256, np.float32)
        for b in range(256):
            s = -1.0 if b & 0x80 else 1.0; e = (b >> 3) & 0xF; m = b & 7
            if e == 15 and m == 7: v[b] = np.nan
            elif e == 0:           v[b] = s * (m / 8.0) * 2.0**-6
            else:                  v[b] = s * (1 + m / 8.0) * 2.0**(e - 7)
        _F8 = v
    return _F8

class Shard:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = json.loads(self.f.read(n)); self.base = 8 + n
    def tensor(self, name):
        t = self.h[name]; o0, o1 = t["data_offsets"]
        self.f.seek(self.base + o0); raw = self.f.read(o1 - o0)
        if t["dtype"] == "F8_E4M3":
            return f8lut()[np.frombuffer(raw, np.uint8)].reshape(t["shape"])
        if t["dtype"] == "F32":
            return np.frombuffer(raw, "<f4").reshape(t["shape"]).copy()
        raise SystemExit(f"dtype {t['dtype']}?")
    def expert_w(self, L, e, kind):
        w = self.tensor(f"layers.{L}.ffn.experts.{e}.{kind}.weight")
        sc = self.tensor(f"layers.{L}.ffn.experts.{e}.{kind}.scale")
        r, c = w.shape
        return (w * np.repeat(np.repeat(sc, 128, 0), 128, 1)[:r, :c]).astype(np.float32)

def open_shard_for_layer(hf, L):
    """shard 号经验=L+2, 容错: 邻近±2 扫描找含该层专家键的 shard。"""
    probe = f"layers.{L}.ffn.experts.0.w1.weight"
    for s in [L + 2, L + 1, L + 3, L, L + 4]:
        p = os.path.join(hf, f"model-{s:05d}-of-00046.safetensors")
        if os.path.exists(p):
            sh = Shard(p)
            if probe in sh.h: return sh
    raise SystemExit(f"L{L}: 找不到含 {probe} 的 shard")

# ---------- 变换 ----------
def walsh(n, sequency=True):
    H = np.array([[1.0]])
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    if sequency:
        H = H[np.argsort((np.diff(H, axis=1) != 0).sum(1), kind="stable")]
    return (H / np.sqrt(n)).astype(np.float32)

def rot_groups(A, H):           # 输入维按 |H| 分组旋转: A[:, g] @ H.T
    G = H.shape[0]; out = np.empty_like(A)
    for j0 in range(0, A.shape[1], G):
        out[:, j0:j0+G] = A[:, j0:j0+G] @ H.T
    return out

def haar_pair(A):               # 一级 Haar(输入维相邻对): 返回 [低|高] 各半, 正交归一
    a = A.reshape(A.shape[0], -1, 2)
    lo = (a[:, :, 0] + a[:, :, 1]) / np.sqrt(2); hi = (a[:, :, 0] - a[:, :, 1]) / np.sqrt(2)
    return lo, hi

# ---------- 冷路统一拟合机器 ----------
def _batched_scales_fit(P, Y, s0):
    """批量行级多-scale ridge LS: P[n,rows,K], Y[n,rows], s0[rows,K] → s[rows,K]。"""
    A = np.einsum("nrk,nrl->rkl", P, P)                      # [rows,K,K]
    b = np.einsum("nrk,nr->rk", P, Y)                        # [rows,K]
    K = P.shape[2]
    lam = np.trace(A, axis1=1, axis2=2) / (P.shape[0] + 1.0) / K   # [rows]
    A[:, np.arange(K), np.arange(K)] += lam[:, None]
    try: s = np.linalg.solve(A, (b + lam[:, None] * s0)[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError: s = s0.copy()
    return np.clip(s, 0.0, 4.0 * np.maximum(s0, 1e-12))

def sign_quant(W, X, mode):
    """mode: row|blk|haar1|haar2 → (Ŵ dequant, bpw)"""
    rows, cols = W.shape; nb = cols // 256
    if mode in ("row", "blk"):
        S = np.where(W >= 0, 1.0, -1.0).astype(np.float32)
        if mode == "row":
            s0 = np.abs(W).mean(1)
            p = X @ S.T; y = (X @ W.T)                       # [n,rows]
            sp2 = (p * p).sum(0); spy = (p * y).sum(0)
            lam = sp2 / (len(X) + 1.0); lam = np.maximum(lam, 1e-3 * sp2 + 1e-9)
            s = np.clip((spy + lam * s0) / (sp2 + lam), 0, 4 * s0)
            Wq = S * s[:, None].astype(np.float32)
            return Wq, 1.0625
        # blk: 每 256 块一个 scale(f16 槽位本就存在), 行内 nb 元批量联合解
        Pb = np.stack([X[:, b*256:(b+1)*256] @ S[:, b*256:(b+1)*256].T for b in range(nb)], 2)  # [n,rows,nb]
        Y = X @ W.T
        s0 = np.stack([np.abs(W[:, b*256:(b+1)*256]).mean(1) for b in range(nb)], 1)            # [rows,nb]
        Sc = _batched_scales_fit(Pb, Y, s0)                  # [rows,nb]
        Wq = S * np.repeat(Sc, 256, axis=1).astype(np.float32)
        return Wq, 1.0625
    # haar 族: 系数空间 sign+scale, 输出域直接在系数空间评估(正交)
    lo, hi = haar_pair(W)                                    # 各 [rows, cols/2]
    Xl, Xh = haar_pair(X)
    if mode in ("haar1", "haar1b"):
        bands_W = [lo, hi]; bands_X = [Xl, Xh]; bpw = 1.0625 + 0.002   # 行ρ f32
    else:
        ll, lh = haar_pair(lo); Xll, Xlh = haar_pair(Xl)
        bands_W = [ll, lh, hi]; bands_X = [Xll, Xlh, Xh]; bpw = 1.0625 + 0.004
    if mode == "haar1b":
        # 等硬件公平版: 块粒度低带 scale s_b(占用现有 f16 槽) + 行级 ρ×高带; 交替 2 轮
        Sl = np.where(lo >= 0, 1.0, -1.0).astype(np.float32); Shi = np.where(hi >= 0, 1.0, -1.0).astype(np.float32)
        nb2 = lo.shape[1] // 128                              # 每 256 权重块 → 128 低+128 高系数
        Plo = np.stack([Xl[:, b*128:(b+1)*128] @ Sl[:, b*128:(b+1)*128].T for b in range(nb2)], 2)
        Phi = np.stack([Xh[:, b*128:(b+1)*128] @ Shi[:, b*128:(b+1)*128].T for b in range(nb2)], 2)
        Y = X @ W.T
        s0 = np.stack([np.abs(lo[:, b*128:(b+1)*128]).mean(1) for b in range(nb2)], 1)
        rho = (np.abs(hi).mean(1) / (np.abs(lo).mean(1) + 1e-12)).astype(np.float32)  # [rows]
        for _ in range(2):
            Peff = Plo + rho[None, :, None] * Phi
            Sc = _batched_scales_fit(Peff, Y, s0)             # [rows,nb2]
            num = np.einsum("nrb,rb->nr", Phi, Sc); den = (num * num).sum(0) + 1e-12
            resid = Y - np.einsum("nrb,rb->nr", Plo, Sc)
            rho = np.clip((num * resid).sum(0) / den, 0.0, 8.0).astype(np.float32)
        Ql = Sl * np.repeat(Sc, 128, 1).astype(np.float32)
        Qh = Shi * (np.repeat(Sc, 128, 1) * rho[:, None]).astype(np.float32)
        q = np.empty_like(W); q[:, 0::2] = (Ql + Qh) / np.sqrt(2); q[:, 1::2] = (Ql - Qh) / np.sqrt(2)
        return q, 1.0625 + 0.002
    Sb = [np.where(B >= 0, 1.0, -1.0).astype(np.float32) for B in bands_W]
    P = np.stack([bx @ sb.T for bx, sb in zip(bands_X, Sb)], 2)   # [n,rows,K]
    Y = X @ W.T
    s0 = np.stack([np.abs(B).mean(1) for B in bands_W], 1)   # [rows,K]
    # 行级 K 带 scale 联合 ridge(带0 存块槽行复制, 其余带=行ρ, 自由度与 B0 同量级公平)
    K = P.shape[2]
    Sc = _batched_scales_fit(P, Y, s0).astype(np.float32)    # [rows,K]
    Qb = [Sb[k] * Sc[:, k:k+1] for k in range(K)]
    # 逆 Haar 还原到权重域
    if mode == "haar1":
        q = np.empty_like(W); q[:, 0::2] = (Qb[0] + Qb[1]) / np.sqrt(2); q[:, 1::2] = (Qb[0] - Qb[1]) / np.sqrt(2)
    else:
        lo_r = np.empty_like(lo); lo_r[:, 0::2] = (Qb[0] + Qb[1]) / np.sqrt(2); lo_r[:, 1::2] = (Qb[0] - Qb[1]) / np.sqrt(2)
        q = np.empty_like(W); q[:, 0::2] = (lo_r + Qb[2]) / np.sqrt(2); q[:, 1::2] = (lo_r - Qb[2]) / np.sqrt(2)
    return q, bpw

def pin_cols(W, X, frac=0.01):
    imp = (X * X).mean(0)[:W.shape[1]] * (W * W).sum(0)
    k = max(1, int(W.shape[1] * frac))
    idx = np.argsort(imp)[-k:]
    Wp = W.copy(); Wp[:, idx] = 0.0
    q8 = np.zeros((W.shape[0], k), np.float32)
    for i, j in enumerate(idx):
        col = W[:, j]; s = np.abs(col).max() / 127.0 + 1e-20
        q8[:, i] = np.round(col / s) * s
    return Wp, idx, q8, 0.0002 + frac * 8

# ---------- 评估 ----------
def metrics(Y, Yq):
    d = Y - Yq
    rel = float(np.linalg.norm(d) / (np.linalg.norm(Y) + 1e-20))
    cos = float((Y * Yq).sum() / (np.linalg.norm(Y) * np.linalg.norm(Yq) + 1e-20))
    return rel, cos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--hf", default="/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base")
    ap.add_argument("--hot-table", default=os.path.join(HERE, "..", "corpus", "prog_active_top64.txt"))
    ap.add_argument("--nh", type=int, default=3); ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--kinds", default="w1,w2")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    L = a.layer
    X = np.load(a.x).astype(np.float32)                      # [530, 4096]
    hot = {}
    for ln in open(a.hot_table):
        p = ln.split(":"); hot[int(p[0][1:])] = [int(x) for x in p[1].split()]
    hids = hot[L][:a.nh]
    cold_all = [e for e in range(256) if e not in set(hot[L])]
    cids = [cold_all[0], cold_all[len(cold_all)//2], cold_all[-1]][:a.nc]
    sh = open_shard_for_layer(a.hf, L)
    Hseq = walsh(256, True); Hnat = walsh(256, False)
    out = []
    def emit(row):
        out.append(row); log("  " + " ".join(f"{x}" for x in row))

    log(f"G1a L{L}: hot={hids} cold={cids} X={X.shape} kinds={a.kinds}")
    os.environ["DS4_GO2B_ACT_SCALE"] = "1"                   # 生产同款联合迭代
    for e_set, lane in ((hids, "hot"), (cids, "cold")):
        for e in e_set:
            t0 = time.time()
            W1 = sh.expert_w(L, e, "w1"); W3 = sh.expert_w(L, e, "w3")
            Xin = {"w1": X, "w3": X}
            if "w2" in a.kinds:
                Zg = X @ W1.T; Zu = X @ W3.T
                Xin["w2"] = (Zg / (1 + np.exp(-np.clip(Zg, -30, 30))) * Zu).astype(np.float32)
            for kind in a.kinds.split(","):
                W = sh.expert_w(L, e, kind) if kind != "w1" else W1
                if kind == "w3": W = W3
                Xk = Xin[kind]; Y = Xk @ W.T
                if lane == "hot":
                    for tag, H in (("A0_go2b", None), ("A1_walshseq", Hseq), ("A2_hadnat", Hnat)):
                        Wf = rot_groups(W, H) if H is not None else W
                        Xf = rot_groups(Xk, H) if H is not None else Xk
                        blk, Wq = encode_go2b(Wf, Xh=Xf, mode="nf")
                        rel, cos = metrics(Y, Xf @ Wq.T)
                        emit((f"L{L}", kind, f"e{e}", lane, tag, "2.125", f"{rel:.4f}", f"{cos:.5f}"))
                else:
                    for tag, mode in (("B0_row", "row"), ("B0b_blk", "blk"), ("B1_haar1", "haar1"), ("B1b_haar1blk", "haar1b"), ("B2_haar2", "haar2")):
                        Wq, bpw = sign_quant(W, Xk, mode)
                        rel, cos = metrics(Y, Xk @ Wq.T)
                        emit((f"L{L}", kind, f"e{e}", lane, tag, f"{bpw:.4f}", f"{rel:.4f}", f"{cos:.5f}"))
                    Wp, idx, q8, extra = pin_cols(W, Xk)
                    Wq, bpw = sign_quant(Wp, Xk, "haar1")
                    Yq = Xk @ Wq.T + Xk[:, idx] @ q8.T
                    rel, cos = metrics(Y, Yq)
                    emit((f"L{L}", kind, f"e{e}", lane, "B4_haar1+pin1%", f"{bpw+extra:.4f}", f"{rel:.4f}", f"{cos:.5f}"))
            log(f"  e{e}({lane}) 完成 {time.time()-t0:.0f}s")
    # 汇总
    print("\nlayer kind expert lane variant bpw relF cos")
    for r in out: print(" ".join(map(str, r)))
    import collections
    agg = collections.defaultdict(list)
    for r in out: agg[(r[3], r[4])].append(float(r[6]))
    print("\n== 汇总(均值 relF, 越小越好) ==")
    for k in sorted(agg): print(f"{k[0]:4s} {k[1]:16s} relF={np.mean(agg[k]):.4f}  n={len(agg[k])}")
    if a.out:
        with open(a.out, "w") as f:
            f.write("layer kind expert lane variant bpw relF cos\n")
            for r in out: f.write(" ".join(map(str, r)) + "\n")
        log(f"报告 → {a.out}")

if __name__ == "__main__":
    main()
