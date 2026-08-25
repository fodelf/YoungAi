#!/usr/bin/env python3
# amp_solve.py — v6.1 乘性放大器逐层解算(产品化, 2026-08-19)。零训练 ELM 闭式:
# z=tanh(x·V₀/s)(V₀=fit段PCA解析方向/固定seed随机, held择优), U=强收缩闭式ridge,
# λ/k/V₀ 全 held 网格自选。行为空间目标(比值域+|y_q|·感知列权+dither增广)。
# 产物: zrec 格式记录 "zl.AMP"(type7, 布局与 zl.RRR 同构: k,scale,din,dout + fp16 z(全1),U,V;
# 引擎语义 out⊙(1+U·tanh(Vᵀx/s)))。held≤0.5% → 空记录(层闸)。
# 用法: amp_solve.py <anchor> <zcache_LXX.npz> <out_amprec.bin> [NFIT=1638]
import sys, os, time, struct
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from probe_layer_behavior import anchor_layer

ap, zcp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
NFIT = int(sys.argv[4]) if len(sys.argv) > 4 else 0   # 0/缺省 → S*0.8(见下, S 此时还没读)
L = int(os.path.basename(zcp).split("_L")[1][:2])
zc = np.load(zcp)
dH, prow, pw, pYQ = zc["dH"], zc["prow"], zc["pw"], zc["pYQ"]
S, D = dH.shape
if NFIT <= 0: NFIT = S * 8 // 10   # 协议 80/20; 硬编码 1638 只在 S=2048 时才对(08-22 实锤)
# ★口径对齐(2026-08-19): tanh 特征的 X 必须是引擎在线喂 kernel 的同一向量。
# DS4_ZL_XANCHOR(量化链态锚, 引擎 DS4_CAP_DIR 捕获转制)给出时用链态 x_q;
# 否则退回 FP 锚(旧口径, 链态漂移大的底座上 held 挽回不迁移)。
# ★致命不一致修复(2026-08-22)★
# 这里原来无条件从 FP 锚重读 x —— 而 zlayer 早已把 x 换成引擎真值(只用在算 dH, 没存下来)。
# 于是解算器拟合 tanh(Vᵀ·x_FP锚/s), 引擎却算 tanh(Vᵀ·x_引擎/s), 两个 x 的逐行余弦只有 ~0.59。
# 特征算在完全不同的输入上 ⇒ 解出来的 U 到引擎里对不上。
# 实测(L32, 只开该层): 解算器自报挽回 8.79%, 引擎实际 −20.72%(反而更差),
# 引擎施加的 |g| 中位 0.276 / p99 5.92, 而解算器意图的量级远小于此。
# 修法: zcache 里带 x(zlayer 存的引擎真值)时一律用它, 锚只作没有 XCAP 时的退路。
if "xcap" in zc.files:
    X0 = zc["xcap"].astype(np.float32)
else:
    XAP = os.getenv("DS4_ZL_XANCHOR")
    X0, _, _, _ = anchor_layer(XAP if XAP else ap, L, S)
t0 = time.time()

# ★XCAP: yq = 引擎真实部署态 routed 输出(raw_ffn_out), 不再从 per-expert 片段重建。
# 放大器乘的就是这个张量, 口径必须逐字节同源(2026-08-22)。
if "yqe" in zc.files:
    yq = zc["yqe"].astype(np.float32)
else:
    yq = np.zeros((S, D), dtype=np.float32)
    np.add.at(yq, prow, pw[:, None] * pYQ)
yfp = yq + dH
tr, ev = np.arange(0, NFIT), np.arange(NFIT, S)
# ★φ 特征提升(2026-08-22 恢复)★ 引擎解析支持 din == 3*d_model 的 ftA 抬升,
# 语义 φ(x)=[x, x⊙x/rms, relu(x)](与引擎 ds4_zchain_zl_apply 里的 din==3d 分支逐式一致)。
# 早先被我从 amp86 抄来的 env 串(DS4_ZL_FTA=0)关掉了。实测 L32: 线性 x 挽回 23.59% →
# φ 提升 28.18%(+19% 相对)。V/A 的输入维随之变 3D, 载荷 A|U|V 三块按 din=3D 存。
def _phi(M):
    n = np.sqrt((M * M).mean(1, keepdims=True)) + 1e-6
    return np.concatenate([M, (M * M) / n, np.maximum(M, 0)], 1)
X0 = _phi(X0.astype(np.float32))
X = X0.astype(np.float64)
DIN = X.shape[1]        # 输入维: φ 提升后 = 3*D; 输出维恒为 D
eps = np.sqrt((yq[tr] ** 2).mean(0)) * 1e-2 + 1e-12
R = (dH * yq / (yq ** 2 + eps[None, :] ** 2)).astype(np.float64)
colw = np.sqrt(R[tr].var(0) + 1e-12) * np.sqrt((yq[tr] ** 2).mean(0) + 1e-12)
rms = np.sqrt((X[tr] ** 2).mean(1, keepdims=True))
r2 = np.random.RandomState(1)
Xa = np.vstack([X[tr], X[tr] + r2.randn(len(tr), DIN) * 0.04 * rms])
Ra = np.vstack([R[tr] * colw, R[tr] * colw])
e0 = float(((yfp[ev] - yq[ev]).astype(np.float64) ** 2).sum())

# k 上限 = 1024(引擎 ds4_zchain.c 解析的硬上限; 再往上要改 kernel 共享内存布局)。
# 2026-08-22: 原值 512 是我随手定的, 结果 43 层网格**全部顶到 512** —— 是撞天花板不是收敛。
# 放开后 L2 实测 5.65%(k=512) → 7.52%(k=1024)。k_L 仍逐层由 held 网格自选, 24MB 是上限非定值。
KMAX = 1024
# ★GPU 化(2026-08-20 铁律"spark 重计算必须 GPU"): SVD+网格矩阵乘上 cupy, 数学逐式同
# CPU 路(闭式解不动, 只换算存设备); held 网格对 ulp 级差异不敏感, 无逐位契约。
try:
    import cupy as _cp
    _GPU = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _cp = None; _GPU = False
xp = _cp if _GPU else np
_A = (lambda a: xp.asarray(a)) if _GPU else (lambda a: a)
_N = (lambda a: _cp.asnumpy(a)) if _GPU else (lambda a: np.asarray(a))
Xa_ = _A(Xa); Ra_ = _A(Ra); Xev_ = _A(X[ev]); colw_ = _A(colw)
yq_ev = _A(yq[ev]); yfp_ev = _A(yfp[ev])
_, _, Vt = xp.linalg.svd(Xa_ - Xa_.mean(0), full_matrices=False)
CANDS = {"PCA": Vt[:KMAX].T, "rand": _A(np.random.RandomState(7).randn(DIN, KMAX) / np.sqrt(DIN))}
scale = float(xp.sqrt((Xa_ ** 2).mean()))

# ★动态 z(方案B, 2026-08-22 用户裁决)★
# 旧式 z 是写死的全 1(z1=np.ones(k)) —— 引擎虽然乘了它, 数学上等于没解。静态 z 又能被 U
# 的列缩放吸收, 补上去是空动作。所以 z 必须是 x 的函数:
#     pv_c(x) = tanh(V_c·x/s) · tanh(A_c·x/s)
# 两个 tanh 特征的乘积 = 真正的二阶门, 且对 U 仍是线性 ⇒ U 保持闭式 ridge, 零训练不变。
# A 的候选与 V 同源(PCA 次 KMAX 段与 V 正交, 给门独立信息; 或另一颗种子的随机基),
# 由 held 网格自选。体积: 每层多一块 D×k fp16(+50%), 用户已明示接受。
# 动态 z 是定案(方案 B), 不留开关: 留开关等于还有一条未被裁决的路活着。
GATES = {"pca2": Vt[KMAX:2 * KMAX].T,
         "rnd2": _A(np.random.RandomState(11).randn(DIN, KMAX) / np.sqrt(DIN))}

best = (0.0, None)   # (held, (V0名, 门名, λ, k, U))
_diag = xp.arange(KMAX)
for nm, V0 in CANDS.items():
  for gn, A0 in GATES.items():
    Za = xp.tanh(Xa_ @ V0 / scale)
    Ze = xp.tanh(Xev_ @ V0 / scale)
    if A0 is not None:
        Za = Za * xp.tanh(Xa_ @ A0 / scale)
        Ze = Ze * xp.tanh(Xev_ @ A0 / scale)
    ZtZ = Za.T @ Za; ZtR = Za.T @ Ra_
    # λ 上界 1000(2026-08-20): cal12 底座全层 λ*=30 顶死旧上界30, 弱层全组合 held 负=假闸
    # (v6.1 判例"λ 入 held 网格后翻正"同款, 网格夹断复发)
    for lam in (1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 1.0):
        G = ZtZ.copy()
        G[_diag, _diag] += lam * float(xp.trace(ZtZ)) / KMAX + 1e-10
        U = xp.linalg.solve(G, ZtR)
        for k in (16, 32, 64, 128, 256, 384, 512, 768, 1024):
            g = (Ze[:, :k] @ (U[:k] / colw_[None, :])).astype(xp.float32)
            yh = yq_ev * (1.0 + g)
            rec = 1 - float(((yfp_ev - yh).astype(xp.float64) ** 2).sum()) / e0
            if rec > best[0]: best = (rec, (nm, gn, lam, k, _N(U[:k]).copy()))

def hdr(nm, psz):
    h = bytearray(116); h[0:len(nm)] = nm.encode()
    struct.pack_into('<Q', h, 88, psz); struct.pack_into('<i', h, 112, 1)
    return bytes(h)

if best[0] <= 0.005 or best[1] is None:
    # 116B 零载荷 zl.AMP 头=终态"层闸"标记(psz<16 合并器不入链); 空文件保留给"中断未跑",
    # zside 断点续跑据此区分重解
    open(outp, 'wb').write(hdr("zl.AMP", 0))
    print(f"★L{L} 放大器: held {best[0]*100:.2f}% ≤0.5% → 层闸(116B 终态标记) | {time.time()-t0:.0f}s", flush=True)
else:
    nm, gn, lam, k, U = best[1]
    Ueng = np.ascontiguousarray((U / colw[None, :]).T.astype(np.float16))       # D×k(引擎 hU[j*k+c])
    Veng = np.ascontiguousarray(_N(CANDS[nm][:, :k]).astype(np.float16))   # DIN×k        # D×k(引擎 hV[j*k+c])
    Aeng = np.ascontiguousarray(_N(GATES[gn][:, :k]).astype(np.float16))        # D×k 动态 z 的门投影
    pay = struct.pack('<IfII', k, scale, X0.shape[1], D) + Aeng.tobytes() + Ueng.tobytes() + Veng.tobytes()
    rec_nm = "zl.AMPD"                                                          # type9: 动态 z 乘性放大器
    open(outp, 'wb').write(hdr(rec_nm, len(pay)) + pay)
    print(f"★L{L} 放大器: held行为挽回 {best[0]*100:.2f}% @V₀={nm} 门={gn} λ={lam} k_L={k} "
          f"体积 {len(pay)/2**20:.1f}MB | {time.time()-t0:.0f}s", flush=True)
