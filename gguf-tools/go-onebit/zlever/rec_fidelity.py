#!/usr/bin/env python3
"""zlever/rec_fidelity.py — 兑现链 S2 段: 注入载荷忠实度对拍(诊断, 2026-08-24 夜)。

问题: 层内自评正收益(zlayer held挽回)端到端不兑现。本探针把 dql 里"注入后的记录字节"
按 ds4quant_run.c bytes_moe type6/type5 的公式逐式重放在解算用的同一批 x 行上, 重评 held
挽回。若与解算自评(S1)一致 → 载荷/公式/fp16 无损, bug 在更下游(回放链态/稀释/链交互);
若塌 → bug 就在 因子→(z,U,V) 转换 / fp16 / 打包 / 解析 里。

用法: rec_fidelity.py <layers_dir> <L> [orig_len=855638144]
输入: <layers_dir>/dql_L{L:02d}.bin (注入态) + zcache_L{L:02d}.npz (dH/xcap/路由对)
"""
import sys, os, struct
import numpy as np

ld = sys.argv[1]; L = int(sys.argv[2])
orig = int(sys.argv[3]) if len(sys.argv) > 3 else 855638144
D = 4096

# ---- 解析注入记录(116B 头: name16 algo64 vol8 psz8 mean16 vd4) ----
p = os.path.join(ld, f"dql_L{L:02d}.bin")
sz = os.path.getsize(p)
recs = []
with open(p, "rb") as f:
    off = orig
    while off + 116 <= sz:
        f.seek(off)
        hdr = f.read(116)
        nm = hdr[:16].split(b"\0")[0].decode(errors="replace")
        psz = struct.unpack("<Q", hdr[88:96])[0]
        pay = f.read(psz)
        recs.append((nm, off, psz, pay))
        off += 116 + psz
print(f"L{L} 注入区 {sz-orig} B, 记录: " + " ".join(f"{n}[{ps}B]" for n, _, ps, _ in recs))

# ---- zcache: 解算现场(dH=教师-学生, xcap=学生x, 路由对) ----
zc = np.load(os.path.join(ld, f"zcache_L{L:02d}.npz"))
dH = np.asarray(zc["dH"], dtype=np.float64)
X = np.asarray(zc["xcap"], dtype=np.float32) if "xcap" in zc.files else None
prow = zc["prow"]; pe = zc["pe"]; pw = zc["pw"]; pYQ = zc["pYQ"]
NTOK = dH.shape[0]
print(f"zcache: dH{dH.shape} X={'有' if X is not None else '无(锚fin口径)'} 对数={len(prow)}")

# ---- 行集: 与发车 env 完全同式 (FR=块0-23剔前64, ER=块24-31剔前64) ----
tr = np.concatenate([np.arange(b*256+64, (b+1)*256) for b in range(24)])
ev = np.concatenate([np.arange(b*256+64, (b+1)*256) for b in range(24, 32)])
tr = tr[tr < NTOK]; ev = ev[ev < NTOK]

tok_pairs = [[] for _ in range(NTOK)]
for i, t in enumerate(prow): tok_pairs[int(t)].append(i)

def phi(M):  # 与 C 逐式: [x, x*x/rms, relu], rms=sqrt(mean(x^2))+1e-6
    n = np.sqrt((M.astype(np.float64)**2).mean(1)) + 1e-6
    return np.concatenate([M, (M*M)/n[:, None].astype(np.float32), np.maximum(M, 0)], 1)

def recov(rows, resid):
    e0 = float((dH[rows]**2).sum()); e1 = float((resid[rows]**2).sum())
    return (1 - e1/e0) * 100, e0

# ---- 按 C 公式重放记录 ----
resid = dH.copy()
for nm, off, psz, pay in recs:
    if "bf.GE" in nm and psz >= 512:
        ge = np.frombuffer(pay[:512], dtype=np.float16).astype(np.float64)
        n_eff = int((np.abs(ge - 1.0) > 1e-4).sum())
        for t in range(NTOK):
            for i in tok_pairs[t]:
                resid[t] -= (ge[int(pe[i])] - 1.0) * (pw[i] * pYQ[i]).astype(np.float64)
        print(f"  bf.GE: 活门={n_eff}")
    elif "zl.RRR" in nm and psz >= 16:
        k, tr_, din, dout = struct.unpack("<IfII", pay[:16])
        h = np.frombuffer(pay[16:16 + 2*(k + k*dout + k*din)], dtype=np.float16)
        z = h[:k].astype(np.float64)
        U = h[k:k + dout*k].astype(np.float64).reshape(dout, k)
        V = h[k + dout*k:].astype(np.float64).reshape(din, k)
        print(f"  zl.RRR: k={k} tr={tr_:.1e} din={din}({'ftA' if din == 3*D else 'lin'}) dout={dout}")
        Xf = phi(X) if din == 3*D else X
        # C: pv[c]=(Σ_d xin·V[d,c])·z[c]; zd[d]=Σ_c U[d,c]·pv[c]  (tr=1e6 → clip 不动)
        zd = ((Xf.astype(np.float64) @ V) * z) @ U.T
        resid = resid - zd
r_ev, e0_ev = recov(ev, resid)
r_tr, _ = recov(tr, resid)
print(f"★S2 载荷重放挽回: held(ev)={r_ev:.1f}%  fit(tr)={r_tr:.1f}%   [S1 解算自评对表: 日志值]")
