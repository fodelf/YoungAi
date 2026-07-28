#!/usr/bin/env python3
"""g3_layer3_validate.py — 抽三层(L23/28/35)层级聚合验证(2026-07-25, 用户指令)。
口径: 真实路由加权聚合 Σ w_te·expert_e(x_t); w2 吃本 expert 量化 w1/w3 的中间(诚实链);
held-out(奇偶分行, fit 行再抽 530 匹配生产 S); 按路由质量取 top 专家(报覆盖率)。
配置: PROD = 热 go2b(GPTQ+ACT) + 冷 signref | SHIP = 热 vq4×512+GPTQ + 冷 w1/w3 vq8×256+GPTQ + 冷 w2 signref
输出: 每层 relF(PROD) vs relF(SHIP) + 分道明细。
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "quant"))
from g1a_lever_probe import open_shard_for_layer
from g1b_stack_probe import cold_C0, metrics
from g1c_vq_sweep import vq_gptq
from go2b_encode import encode_go2b
os.environ["DS4_GO2B_ACT_SCALE"] = "1"

# ★C 加速(磨刀 2026-07-25): DS4_VQ_SHIM=dylib 路径 → 三个编码器全走 C(15×提速)★
_SHIM = os.environ.get("DS4_VQ_SHIM")
if _SHIM:
    import ctypes
    _so = ctypes.CDLL(_SHIM)
    def _cf(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    def enc_vq(W, Xf, dim, nc):
        W = np.ascontiguousarray(W, np.float32); Xf = np.ascontiguousarray(Xf, np.float32)
        Wq = np.empty_like(W)
        _so.vq_encode_c(_cf(W), W.shape[0], W.shape[1], dim, nc, _cf(Xf), len(Xf), _cf(Wq))
        return Wq, 0.0
    def enc_go2b(W, Xf):
        W = np.ascontiguousarray(W, np.float32); Xf = np.ascontiguousarray(Xf, np.float32)
        Wq = np.empty_like(W)
        _so.go2b_encode_cext(_cf(W), W.shape[0], W.shape[1], _cf(Xf), len(Xf), _cf(Wq))
        return Wq
    def enc_sign(W, Xf):
        W = np.ascontiguousarray(W, np.float32); Xf = np.ascontiguousarray(Xf, np.float32)
        Wq = np.empty_like(W)
        _so.signref_encode_cext(_cf(W), W.shape[0], W.shape[1], _cf(Xf), len(Xf), _cf(Wq))
        return Wq, 0.0
else:
    def enc_vq(W, Xf, dim, nc): return vq_gptq(W, Xf, dim=dim, nc=nc)
    def enc_go2b(W, Xf): return encode_go2b(W, Xh=Xf, mode="nf")[1]
    def enc_sign(W, Xf): return cold_C0(W, Xf)

CAP = "/Users/fodelf/ds4-main/cap_ef2"
HF = "/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base"
NEXP_CAP = 96          # 每层按质量 top-N 专家(报覆盖率)
SWLIM = 7.0            # swiglu clamp(与引擎一致口径, 影响两配置相同)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", file=sys.stderr, flush=True)

def silu(z): return z / (1.0 + np.exp(-np.clip(z, -30, 30)))

def hot_ids(L):
    for ln in open(os.path.join(HERE, "..", "corpus", "prog_active_top64.txt")):
        p = ln.split(":")
        if int(p[0][1:]) == L: return set(int(x) for x in p[1].split())
    return set()

def expert_chain(W1q, W3q, W2q, X):
    H = silu(np.clip(X @ W1q.T, -SWLIM, SWLIM)) * np.clip(X @ W3q.T, -SWLIM, SWLIM)
    return H.astype(np.float32) @ W2q.T

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0); ap.add_argument("--shard", default="")
    A = ap.parse_args()
    layers = [A.layer] if A.layer else [23, 28, 35]
    si, sn = (int(x) for x in A.shard.split("/")) if A.shard else (0, 1)
    for L in layers:
        X = np.load(f"{CAP}/ffn_in_L{L}.npy").astype(np.float32)
        R = np.load(f"{CAP}/route_L{L}.npy").astype(np.int64)
        RW = np.load(f"{CAP}/route_w_L{L}.npy").astype(np.float32)
        n = len(X)
        ifit = np.arange(0, n, 2); ieval = np.arange(1, n, 2)
        rng = np.random.default_rng(7)
        fit530 = rng.choice(ifit, min(530, len(ifit)), replace=False)
        Xf, Xe = X[fit530], X[ieval]
        Re, RWe = R[ieval], RW[ieval]
        mass = np.zeros(256)
        for k in range(R.shape[1]): np.add.at(mass, R[:, k], RW[:, k])
        order = np.argsort(mass)[::-1]
        sel = [int(e) for e in order[:NEXP_CAP] if mass[e] > 0]
        cover = mass[sel].sum() / mass.sum()
        hids = hot_ids(L)
        sh = open_shard_for_layer(HF, L)
        log(f"L{L}: 采样 {len(sel)} 专家 覆盖 {cover*100:.1f}% 路由质量; 热 {sum(1 for e in sel if e in hids)}")
        Yfp = np.zeros((len(ieval), 4096), np.float32)
        Yp  = np.zeros_like(Yfp); Ys = np.zeros_like(Yfp)
        selset = set(sel)
        wsel = {}                                    # e -> [neval] 权重(不在路由的 token 为 0)
        sel = [e for k, e in enumerate(sel) if k % sn == si]
        for i, e in enumerate(sel):
            t0 = time.time()
            W1 = sh.expert_w(L, e, "w1"); W3 = sh.expert_w(L, e, "w3"); W2 = sh.expert_w(L, e, "w2")
            we = np.zeros(len(ieval), np.float32)
            for k in range(Re.shape[1]): we += np.where(Re[:, k] == e, RWe[:, k], 0.0)
            m = we > 0
            if not m.any(): continue
            Xem = Xe[m]
            yfp = expert_chain(W1, W3, W2, Xem)
            ishot = e in hids
            if ishot:
                W1p = enc_go2b(W1, Xf); W3p = enc_go2b(W3, Xf)
                Hf = silu(np.clip(Xf @ W1p.T, -SWLIM, SWLIM)) * np.clip(Xf @ W3p.T, -SWLIM, SWLIM)
                W2p = enc_go2b(W2, Hf.astype(np.float32))
            else:
                W1p, _ = enc_sign(W1, Xf); W3p, _ = enc_sign(W3, Xf)
                Hf = silu(np.clip(Xf @ W1p.T, -SWLIM, SWLIM)) * np.clip(Xf @ W3p.T, -SWLIM, SWLIM)
                W2p, _ = enc_sign(W2, Hf.astype(np.float32))
            yp = expert_chain(W1p, W3p, W2p, Xem)
            if ishot:
                W1s, _ = enc_vq(W1, Xf, 4, 512); W3s, _ = enc_vq(W3, Xf, 4, 512)
                Hfs = silu(np.clip(Xf @ W1s.T, -SWLIM, SWLIM)) * np.clip(Xf @ W3s.T, -SWLIM, SWLIM)
                W2s, _ = enc_vq(W2, Hfs.astype(np.float32), 4, 512)
            else:
                W1s, _ = enc_vq(W1, Xf, 8, 256); W3s, _ = enc_vq(W3, Xf, 8, 256)
                Hfs = silu(np.clip(Xf @ W1s.T, -SWLIM, SWLIM)) * np.clip(Xf @ W3s.T, -SWLIM, SWLIM)
                W2s, _ = enc_sign(W2, Hfs.astype(np.float32))
            ys = expert_chain(W1s, W3s, W2s, Xem)
            wm = we[m][:, None]
            Yfp[m] += wm * yfp; Yp[m] += wm * yp; Ys[m] += wm * ys
            if (i & 7) == 0 or i == len(sel) - 1:
                log(f"L{L} {i+1}/{len(sel)} e{e}({'热' if ishot else '冷'}) {time.time()-t0:.0f}s")
        if sn > 1:
            np.savez(f"/tmp/g3p_L{L}_s{si}.npz", Yfp=Yfp, Yp=Yp, Ys=Ys, cover=cover)
            log(f"L{L} 分片 {si}/{sn} 落盘")
        else:
            rp, cp = metrics(Yfp, Yp); rs, cs = metrics(Yfp, Ys)
            print(f"L{L} 层聚合(覆盖{cover*100:.1f}%): PROD relF={rp:.4f} cos={cp:.5f} | SHIP relF={rs:.4f} cos={cs:.5f} | Δ={(rs-rp)/rp*100:+.1f}%", flush=True)

if __name__ == "__main__":
    main()
