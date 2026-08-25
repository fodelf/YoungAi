#!/usr/bin/env python3
# rgate.py — 四支柱③ route 预测 gate(x) 闭式落地(2026-08-15 用户令"跑")。
# 病灶: 路由漂移=通用域深带隐藏规律(神谕: KL −21.5%, 0.5186→0.4070 越过官方 0.4207)。
# solve: FP锚(ridx/rw=FP路由目标) × 链锚(XQ=部署链态) → 每层闭式 ridge W_L(x̃→稀疏门分数),
#        EV 行选 λ, 判据=FP top-NACT 命中率(对拍自路由基线)。
# patch: 判决链锚 XQ → topk(XQ·W_L) 预测路由+幅度对齐自路由权重和 → 写入判决FP锚副本的
#        ridx/rw 块(其余字节不动) → C 回放 DS4_ANCHOR_ROUTE=1 即为可部署信息判决(零FP偷看)。
import sys, os, struct, shutil
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from probe_layer_behavior import anchor_layer

NEXP = 256

def hdr(ap):
    return struct.unpack("<8I", open(ap, "rb").read(32))  # [?,S,HCM,DIM,NL,V,NACT,?]

def agree(pid, fid, NACT):
    a = 0
    for t in range(len(fid)):
        a += len(set(pid[t].tolist()) & set(fid[t].tolist()))
    return a / (len(fid) * NACT)

cmd = sys.argv[1]
if cmd == "solve":
    fpa, cha, outd = sys.argv[2], sys.argv[3], sys.argv[4]
    S, NF, EV0, EV1 = map(int, sys.argv[5:9])
    NL, NACT = hdr(fpa)[4], hdr(fpa)[6]
    os.makedirs(outd, exist_ok=True)
    for L in range(NL):
        X0, ridx, rw, _ = anchor_layer(fpa, L, S)
        XQ, ridxq, rwq, _ = anchor_layer(cha, L, S)
        # ★残差参数化(v2, 首版从零学门被机制闸拦: 收缩向0≠向自路由, 全层输给基线)★
        # 目标 = FP 稀疏门 − 自路由稀疏门; 部署门 = 自路由门 + x̃·W; λ→∞ 退化为自路由 = 基线保底
        G = np.zeros((S, NEXP), dtype=np.float32)
        m = (ridx >= 0) & (ridx < NEXP)
        np.put_along_axis(G, np.where(m, ridx, 0), np.where(m, rw, 0), axis=1)
        Gs = np.zeros((S, NEXP), dtype=np.float32)
        mq = (ridxq >= 0) & (ridxq < NEXP)
        np.put_along_axis(Gs, np.where(mq, ridxq, 0), np.where(mq, rwq, 0), axis=1)
        R = G - Gs
        tr, ev = slice(0, NF), slice(EV0, EV1)
        Xt = XQ[tr].astype(np.float64)
        XtX = Xt.T @ Xt; XtR = Xt.T @ R[tr].astype(np.float64)
        base = agree(ridxq[ev], ridx[ev], NACT)
        best = (base, None, None)   # 基线在场: 修正量打不过基线就不带修正(W=None=纯自路由)
        for lam in (1e-2, 1e-1, 1.0, 10.0):
            A = XtX + lam * np.trace(XtX) / Xt.shape[1] * np.eye(Xt.shape[1])
            W = np.linalg.solve(A, XtR)
            P = Gs[ev].astype(np.float64) + XQ[ev].astype(np.float64) @ W
            ag = agree(np.argsort(-P, axis=1)[:, :NACT], ridx[ev], NACT)
            if ag > best[0]: best = (ag, lam, W)
        ag, lam, W = best
        if W is None:
            np.save(os.path.join(outd, f"rgate_L{L:02d}.npy"), np.zeros((XQ.shape[1], NEXP), dtype=np.float32))
            print(f"L{L:02d} 命中 自路由={base*100:.1f}% → 修正无增益, 保基线", flush=True)
        else:
            np.save(os.path.join(outd, f"rgate_L{L:02d}.npy"), W.astype(np.float32))
            print(f"L{L:02d} 命中 自路由={base*100:.1f}% → rgate={ag*100:.1f}% (λ={lam})", flush=True)
elif cmd == "patch":
    fpa, cha, outd, out_ap = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    S = int(sys.argv[6])
    hd = hdr(fpa); Sa, DIM, NL, NACT = hd[1], hd[3], hd[4], hd[6]
    assert Sa == S, (Sa, S)
    shutil.copyfile(fpa, out_ap)
    f = open(out_ap, "r+b")
    fin_off = 40
    ridx_off = fin_off + NL * Sa * DIM * 4
    rw_off = ridx_off + NL * Sa * NACT * 4
    nfb = 0
    for L in range(NL):
        W = np.load(os.path.join(outd, f"rgate_L{L:02d}.npy"))
        XQ, ridxq, rwq, _ = anchor_layer(cha, L, S)
        Gs = np.zeros((S, NEXP), dtype=np.float32)
        mq = (ridxq >= 0) & (ridxq < NEXP)
        np.put_along_axis(Gs, np.where(mq, ridxq, 0), np.where(mq, rwq, 0), axis=1)
        P = Gs + XQ.astype(np.float32) @ W     # 残差式部署门 = 自路由门 + 修正
        pid = np.argsort(-P, axis=1)[:, :NACT].astype(np.int32)
        pv = np.take_along_axis(P, pid, axis=1)
        bad = (pv <= 0).all(1)                      # 预测失效 token 回退自路由
        pv = np.maximum(pv, 1e-6)
        pv = pv / pv.sum(1, keepdims=True) * np.maximum(rwq.sum(1, keepdims=True), 1e-6)
        pid[bad] = ridxq[bad]; pv[bad] = rwq[bad]
        nfb += int(bad.sum())
        f.seek(ridx_off + (L * Sa) * NACT * 4); f.write(np.ascontiguousarray(pid, dtype=np.int32).tobytes())
        f.seek(rw_off + (L * Sa) * NACT * 4);  f.write(np.ascontiguousarray(pv, dtype=np.float32).tobytes())
    f.close()
    print(f"patch 完: {NL}层 → {out_ap} (回退token计 {nfb})", flush=True)
else:
    print("用法: rgate.py solve <FP锚> <链锚> <outd> <S> <NF> <EV0> <EV1> | patch <FP锚> <链锚> <outd> <出锚> <S>")
    sys.exit(1)
