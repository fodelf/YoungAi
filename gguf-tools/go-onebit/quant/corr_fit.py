#!/usr/bin/env python3
"""corr_fit.py — 快速行为校准: corr 侧车闭式拟合(2026-07-27)。

数据: 学生轨迹 X=raw_ffn_in(f16,[n,4096]) / o_stu=raw_ffn_out; teacher 参考 o_ref(oref_on_student.py)。
拟合: E=o_ref−o_stu; 岭回归 W=argmin‖XWᵀ−E‖²+λ‖W‖² (闭式), 截断 SVD 到秩 r →
  引擎语义 out += Σ_{e∈sel} U(C_e⊙Vx)+b+β_e: 取 C_e≡1/k(k=6) → ΣC_e=1 → 等效 out += U(Vx);
  b=β=δ=0(δ=0 还跳过 router dispatch)。U=[d_model,d_l] V=[d_l,d_model] row-major, F32。
输出: corr GGUF (ds4.corr.present=true, blk.L.corr_{U,V,C,b,beta,delta})。
⚠ 历史雷(2026-07-23): 激活空间 corr 多层联合复利爆炸("单层可辨识/23层乱码") →
  部署必须 DS4_CORR_SCALE α 扫描 + 字节护栏把关, 本脚本只出 α=1 的因子。
用法: corr_fit.py --cap CAPDIR --oref OREFDIR --nrows N --rank 16 --out corr.gguf
"""
import argparse, struct, sys
import numpy as np

N_EXPERT, D, K_USED = 256, 4096, 6
ALIGN = 32


def fit_layer(cap, oref, L, nrows, rank, ridge, center=True, clip_pct=95.0):
    X = np.fromfile(f"{cap}/raw_ffn_in_L{L}", dtype=np.float16).reshape(-1, D)[:nrows].astype(np.float64)
    O = np.fromfile(f"{cap}/raw_ffn_out_L{L}", dtype=np.float16).reshape(-1, D)[:nrows].astype(np.float64)
    R = np.load(f"{oref}/oref_L{L}.npy").astype(np.float64)[:nrows]
    E = R - O
    # v2(α=1.0/0.25 双爆复盘): ①均值中心化 — mean(E) 是把一切推向文件头 token 的
    # 常量毒推力(43 层复利), fork 需要的是 x 依赖修正; 均值分量直接丢弃不进 b。
    # ②行范数分位截断 — '// ====' 标记/BOS 等野值行不得主导回归。
    if clip_pct:
        rn = np.linalg.norm(E, axis=1)
        cap_n = np.percentile(rn, clip_pct)
        sc = np.minimum(1.0, cap_n / np.maximum(rn, 1e-9))
        E = E * sc[:, None]
    if center:
        E = E - E.mean(0, keepdims=True)
    e0 = float(np.linalg.norm(E))
    G = X.T @ X
    lam = ridge * np.trace(G) / D
    W = np.linalg.solve(G + lam * np.eye(D), X.T @ E).T        # [D,D]: E ≈ X @ W.T
    U_, S_, Vt_ = np.linalg.svd(W, full_matrices=False)
    U = (U_[:, :rank] * S_[:rank]).astype(np.float32)          # [D, r]
    V = Vt_[:rank, :].astype(np.float32)                       # [r, D]
    e1 = float(np.linalg.norm(E - X @ (U @ V).T))
    return U, V, e0, e1


def w_str(b, s):
    b += struct.pack("<Q", len(s)) + s.encode()
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--oref", required=True)
    ap.add_argument("--nrows", type=int, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-42")
    a = ap.parse_args()
    lo, hi = map(int, a.layers.split("-"))

    tensors = []   # (name, ne, f32 array)
    for L in range(lo, hi + 1):
        U, V, e0, e1 = fit_layer(a.cap, a.oref, L, a.nrows, a.rank, a.ridge)
        r = U.shape[1]
        C = np.full((N_EXPERT, r), 1.0 / K_USED, dtype=np.float32)   # [n_exp][d_l] row-major
        tensors += [
            (f"blk.{L}.corr_U",     [r, D],        U),              # ne inner-first: [d_l, d_model]
            (f"blk.{L}.corr_V",     [D, r],        V),
            (f"blk.{L}.corr_C",     [r, N_EXPERT], C),
            (f"blk.{L}.corr_b",     [D],           np.zeros(D, np.float32)),
            (f"blk.{L}.corr_beta",  [N_EXPERT],    np.zeros(N_EXPERT, np.float32)),
            (f"blk.{L}.corr_delta", [N_EXPERT],    np.zeros(N_EXPERT, np.float32)),
        ]
        print(f"L{L:2d} rank={r} ‖E‖ {e0:.1f} → 残差 {e1:.1f} ({100*(1-e1/max(e0,1e-9)):.1f}% 吸收)", flush=True)

    # GGUF v3: header + 1 KV(bool present) + tensor table + data
    kv = w_str(b"", "ds4.corr.present") + struct.pack("<I", 7) + b"\x01"
    hdr = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), 1) + kv
    info, off = b"", 0
    offs = []
    for name, ne, arr in tensors:
        info = w_str(info, name)
        info += struct.pack("<I", len(ne)) + struct.pack("<%dQ" % len(ne), *ne)
        info += struct.pack("<IQ", 0, off)                     # type 0 = F32
        offs.append(off)
        off += (arr.nbytes + ALIGN - 1) // ALIGN * ALIGN
    data0 = (len(hdr) + len(info) + ALIGN - 1) // ALIGN * ALIGN
    with open(a.out, "wb") as f:
        f.write(hdr); f.write(info); f.write(b"\x00" * (data0 - len(hdr) - len(info)))
        for (name, ne, arr), o in zip(tensors, offs):
            f.seek(data0 + o); f.write(np.ascontiguousarray(arr).tobytes())
    import os
    print(f"[corr_fit] {a.out} {os.path.getsize(a.out)/2**20:.1f} MiB ({len(tensors)//6} 层, rank {a.rank})")


if __name__ == "__main__":
    main()
