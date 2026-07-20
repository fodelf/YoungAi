#!/usr/bin/env python3
"""go2b_encode.py — 把一个 fp32 权重矩阵编码成 block_go2b (type 41) 字节，供混合模型/侧车。
块 68B: [f16 d1][f16 d2][s1 32B][s2 32B]; 值 = (±d1)+(±d2), 小端位 g → 字节 g//8 位 g%8。
每行拟合对称 (d1,d2)(Lloyd) + GPTQ 误差反馈分配到 4 个 level {±d1±d2}。tern 模式强制 d1==d2。
自带往返自检: 用运行时同款 dequant 还原自身字节，须与分配 level 逐值一致。
API: encode_go2b(W, Xh=None, mode='nf'|'tern', grp=128, ridge=0.02) -> (blocks_u8[rows,nblk,68], Wq[rows,cols])
"""
import numpy as np

def _fit_d1d2(W, tern=False):
    a = np.abs(W)
    if tern:
        # 三值: level {-2d,0,0,2d} 需 d1==d2==d; d = 非零元均值/2 (TWN)
        thr = 0.7 * a.mean(1, keepdims=True); m = a >= thr
        s = (a * m).sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1)
        d = s / 2.0
        return d, d.copy()
    hi = np.quantile(a, 0.75, axis=1, keepdims=True)
    lo = np.quantile(a, 0.25, axis=1, keepdims=True)
    d1 = (hi + lo) / 2; d2 = (hi - lo) / 2
    for _ in range(3):
        L4 = np.concatenate([-(d1 + d2), -(d1 - d2), (d1 - d2), (d1 + d2)], 1)
        idx = np.abs(W[:, :, None] - L4[:, None, :]).argmin(2)
        m3 = idx == 3; m2 = idx == 2
        v3 = np.where(m3.sum(1, keepdims=True) > 0, (W * m3).sum(1, keepdims=True) / np.maximum(m3.sum(1, keepdims=True), 1), hi)
        v2 = np.where(m2.sum(1, keepdims=True) > 0, (W * m2).sum(1, keepdims=True) / np.maximum(m2.sum(1, keepdims=True), 1), lo)
        d1 = (v3 + v2) / 2; d2 = np.abs((v3 - v2) / 2) + 1e-6
    return d1.astype(np.float32), d2.astype(np.float32)

def _gptq_assign(W, L4, Xh, grp, ridge):
    """给定 4 level (每行), GPTQ 误差反馈分配码 → (Wq, sidx)。
    DS4_GO2B_NO_GPTQ: 强制闭式最近邻分配(无迭代误差反馈)——快(秒级)、闭式、无训练味。
    配 _act_d1d2 的输出最优 (d1,d2) 仍是 output-optimal, 只是分配走最近邻而非 GPTQ。"""
    import os as _os
    rows, cols = W.shape
    Wq = np.empty_like(W); sidx = np.empty((rows, cols), dtype=np.int8)
    if Xh is not None and len(Xh) >= 8 and not _os.environ.get("DS4_GO2B_NO_GPTQ"):
        for j0 in range(0, cols, grp):
            Xb = Xh[:, j0:j0+grp].astype(np.float32); g = Xb.shape[1]
            H = Xb.T @ Xb; H[np.diag_indices(g)] += ridge * (np.mean(np.diag(H)) + 1e-9)
            Hinv = np.linalg.inv(H); Wk = W[:, j0:j0+grp].copy()
            for j in range(g):
                col = Wk[:, j]; k = np.abs(col[:, None] - L4).argmin(1)
                q = L4[np.arange(rows), k]
                Wq[:, j0+j] = q; sidx[:, j0+j] = k
                err = (col - q) / Hinv[j, j]
                if j + 1 < g: Wk[:, j+1:] -= np.outer(err, Hinv[j, j+1:])
    else:
        k = np.abs(W[:, :, None] - L4[:, None, :]).argmin(2)
        Wq = L4[np.arange(rows)[:, None], k]; sidx = k.astype(np.int8)
    return Wq, sidx


def _act_d1d2(W, B1, B2, X):
    """输出最优 (d1,d2) per-row: 固定码 B1/B2∈{-1,+1}, 解 2×2
    [Σp1² Σp1p2; Σp1p2 Σp2²][d1;d2]=[Σt·p1;Σt·p2] (p1=B1·x,p2=B2·x,t=w·x)。
    X 切到 W 列宽 (w2 输入 2048 用层输入前 2048 维近似, 与现有 GPTQ 隐式切片一致)。"""
    Xc = X[:, :W.shape[1]]
    P1 = Xc @ B1.T; P2 = Xc @ B2.T; T = Xc @ W.T       # [n,rows]
    a = (P1 * P1).sum(0); b = (P1 * P2).sum(0); c = (P2 * P2).sum(0)
    r1 = (T * P1).sum(0); r2 = (T * P2).sum(0)
    det = a * c - b * b; det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    d1 = np.abs((c * r1 - b * r2) / det); d2 = np.abs((a * r2 - b * r1) / det)
    return d1.astype(np.float32)[:, None], d2.astype(np.float32)[:, None]


def encode_go2b(W, Xh=None, mode='nf', grp=128, ridge=0.02):
    import os
    rows, cols = W.shape; nblk = cols // 256
    d1, d2 = _fit_d1d2(W, tern=(mode == 'tern'))
    # f16 往返 (与运行时一致): d 存 f16
    d1h = d1.astype(np.float16).astype(np.float32); d2h = d2.astype(np.float16).astype(np.float32)
    # 4 level 与 (s1,s2) 位: level = (s1?+d1:-d1)+(s2?+d2:-d2)
    #  (1,1)=+d1+d2  (1,0)=+d1-d2  (0,1)=-d1+d2  (0,0)=-d1-d2
    L4 = np.stack([-(d1h + d2h)[:, 0], -(d1h - d2h)[:, 0], (d1h - d2h)[:, 0], (d1h + d2h)[:, 0]], 1)  # [rows,4]
    # level = (s1?+d1:-d1)+(s2?+d2:-d2): idx0=-d1-d2(0,0) idx1=-d1+d2(0,1) idx2=+d1-d2(1,0) idx3=+d1+d2(1,1)
    s1bit_of = np.array([0, 0, 1, 1]); s2bit_of = np.array([0, 1, 0, 1])
    Wq, sidx = _gptq_assign(W, L4, Xh, grp, ridge)
    # ★DS4_GO2B_ACT_SCALE: 联合迭代 —— 固定码解输出最优(d1,d2), 用新level重分配码, 2轮。
    # 产出自洽的 output-optimal GO2B (量化时施加, 非事后patch)。
    if os.environ.get("DS4_GO2B_ACT_SCALE") and Xh is not None and len(Xh) >= 8 and mode != 'tern':
        Xf = Xh.astype(np.float32)
        for _ in range(2):
            B1 = (s1bit_of[sidx].astype(np.float32) * 2 - 1)   # ±1 [rows,cols]
            B2 = (s2bit_of[sidx].astype(np.float32) * 2 - 1)
            d1, d2 = _act_d1d2(W, B1, B2, Xf)
            d1h = d1.astype(np.float16).astype(np.float32); d2h = d2.astype(np.float16).astype(np.float32)
            L4 = np.stack([-(d1h + d2h)[:, 0], -(d1h - d2h)[:, 0], (d1h - d2h)[:, 0], (d1h + d2h)[:, 0]], 1)
            Wq, sidx = _gptq_assign(W, L4, Xf, grp, ridge)
    # 打包 68B 块
    blk = np.zeros((rows, nblk, 68), dtype=np.uint8)
    blk[:, :, 0:2] = d1.astype(np.float16).view(np.uint8).reshape(rows, 1, 2)
    blk[:, :, 2:4] = d2.astype(np.float16).view(np.uint8).reshape(rows, 1, 2)
    s1 = s1bit_of[sidx].astype(np.uint8).reshape(rows, nblk, 256)
    s2 = s2bit_of[sidx].astype(np.uint8).reshape(rows, nblk, 256)
    blk[:, :, 4:36] = np.packbits(s1, axis=2, bitorder='little')
    blk[:, :, 36:68] = np.packbits(s2, axis=2, bitorder='little')
    return blk, Wq

def decode_go2b(blk, cols):
    """运行时同款 dequant: 从 68B 块还原 [rows,cols]。用于自检。"""
    rows, nblk = blk.shape[0], blk.shape[1]
    d1 = blk[:, :, 0:2].copy().view(np.float16).astype(np.float32)[:, :, 0]  # [rows,nblk]
    d2 = blk[:, :, 2:4].copy().view(np.float16).astype(np.float32)[:, :, 0]
    b1 = np.unpackbits(blk[:, :, 4:36], axis=2, bitorder='little').reshape(rows, nblk, 256)
    b2 = np.unpackbits(blk[:, :, 36:68], axis=2, bitorder='little').reshape(rows, nblk, 256)
    a = np.where(b1 == 1, d1[:, :, None], -d1[:, :, None])
    b = np.where(b2 == 1, d2[:, :, None], -d2[:, :, None])
    return (a + b).reshape(rows, nblk * 256)[:, :cols]

if __name__ == "__main__":   # 自检: L38 gate e0, 编码→解码 must == Wq, 且前向复现
    import json, sys
    sys.path.insert(0, "/Users/fodelf/git/ds4-main/gguf-tools/go-onebit/calib/pyfwd")
    from ds4reader import LUT
    meta = json.load(open("/tmp/rq_L38.meta.json")); K = meta["kinds"]
    w8 = open("/tmp/rq_L38.w8", "rb"); si = open("/tmp/rq_L38.si", "rb")
    def hf_w(ki, kind, e):
        r, c = K[kind]["rows"], K[kind]["cols"]; sr, scn = K[kind]["si_shape"]
        off = sum(K[k]["rows"]*K[k]["cols"]*256 for k in ("gate","up","down")[:ki])
        w8.seek(off+e*r*c); a = np.frombuffer(w8.read(r*c), dtype=np.uint8).reshape(r,c)
        soff = sum(int(np.prod(K[k]["si_shape"]))*4*256 for k in ("gate","up","down")[:ki])
        si.seek(soff+e*sr*scn*4); sc = np.frombuffer(si.read(sr*scn*4), dtype=np.float32).reshape(sr,scn)
        return LUT[a]*np.repeat(np.repeat(sc,128,axis=0),128,axis=1)[:r,:c]
    W = hf_w(0, "gate", 0)
    blk, Wq = encode_go2b(W, None, mode='nf')
    Wdec = decode_go2b(blk, W.shape[1])
    err = np.abs(Wdec - Wq).max()
    print(f"往返自检 max|decode-Wq|={err:.2e}  ({'PASS' if err < 1e-6 else 'FAIL'})")
    print(f"块字节 {blk.shape} = {blk.nbytes} bytes  (期望 {W.shape[0]}*{W.shape[1]//256}*68={W.shape[0]*W.shape[1]//256*68})")
    print(f"量化误差 cos(W,Wq)={float(np.sum(W*Wq)/(np.linalg.norm(W)*np.linalg.norm(Wq)+1e-20)):.4f}")
