#!/usr/bin/env python3
"""dspark_amp_fit.py — drafter 自己的反修放大器(用户四文件设计之第4件, 2026-08-21)。

背景: q2 drafter 被量化损伤 ⇒ 每位接受率 ~0.6, 而投机翻正需要 ~0.75。放大器把
drafter 的 routed 输出行为修回教师(HF 原始 mxfp4 专家)水平, 直接抬接受率。

口径与主模型放大器一致(乘性 ELM 闭式, zl.AMP/type7, out ⊙ (1+U·tanh(Vᵀx/s))):
  · 锚 = 引擎在部署态(q2 drafter + 真实投机流)捕获的 (Fin, 路由 idx/权重) —— 见
    ds4.c 的 DS4_DSPARK_ANCHOR; 教师/学生输出都在本脚本内按同一 Fin/路由离线重算,
    保证只解"权重量化误差", 不混入路由漂移。
  · 学生 = 量化 drafter gguf 的 mtp.N 专家; 教师 = HF mtp.N.ffn.experts.*(mxfp4)。
  · 每 mtp 块独立解, 产 zrec_L0{N}.bin, 再由 zrec_to_zchain.py 成 3 层链。

用法: dspark_amp_fit.py <hf_dir> <student_gguf> <anchor.bin> <out_dir> <block> [NFIT_FRAC=0.8]
"""
import json, os, struct, sys, time
import numpy as np

HF, GGUF, ANCHOR, OUTDIR, BLOCK = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
NFIT_FRAC = float(sys.argv[6]) if len(sys.argv) > 6 else 0.8
os.makedirs(OUTDIR, exist_ok=True)
SWLIM = 10.0
D = 4096

try:
    import cupy as cp
    xp = cp
    GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None; xp = np; GPU = False
A = (lambda a: cp.asarray(a)) if GPU else (lambda a: np.asarray(a))
N_ = (lambda a: cp.asnumpy(a)) if GPU else (lambda a: np.asarray(a))

# ---------- 锚读取 ----------
def load_anchor(path, blk):
    raw = open(path, "rb").read()
    off = 0
    fins, sels, rws = [], [], []
    while off < len(raw):
        b, pos, B, K, Dm = struct.unpack_from("<5I", raw, off); off += 20
        fin = np.frombuffer(raw, np.float32, B * Dm, off); off += B * Dm * 4
        sel = np.frombuffer(raw, np.int32, B * K, off); off += B * K * 4
        rw = np.frombuffer(raw, np.float32, B * K, off); off += B * K * 4
        off += B * Dm * 4          # o_ref: 只作校验, 教师/学生都在本脚本重算
        if b == blk:
            fins.append(fin.reshape(B, Dm)); sels.append(sel.reshape(B, K)); rws.append(rw.reshape(B, K))
    if not fins:
        raise SystemExit(f"anchor 无 block {blk} 记录")
    return np.concatenate(fins), np.concatenate(sels), np.concatenate(rws)

# ---------- HF 教师权重(mxfp4) ----------
WM = json.load(open(HF + "/model.safetensors.index.json"))["weight_map"]
_E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)

def st_raw(k):
    shard = HF + "/" + WM[k]; f = open(shard, "rb")
    hl = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(hl))
    o = hdr[k]["data_offsets"]; f.seek(8 + hl + o[0])
    return hdr[k]["dtype"], hdr[k]["shape"], f.read(o[1] - o[0])

def teacher_w(blk, e, nm):
    dt, sh, raw = st_raw(f"mtp.{blk}.ffn.experts.{e}.{nm}.weight")
    b = np.frombuffer(raw, np.uint8).reshape(sh)
    lo = _E2M1[b & 0xF]; hi = _E2M1[b >> 4]
    w = np.stack([lo, hi], -1).reshape(sh[0], sh[1] * 2)
    _, ssh, sraw = st_raw(f"mtp.{blk}.ffn.experts.{e}.{nm}.scale")
    se = np.frombuffer(sraw, np.uint8).astype(np.int32).reshape(ssh)
    sc = np.exp2(se - 127).astype(np.float32)
    return (w.reshape(ssh[0], 1, ssh[1], 32) * sc[:, None, :, None]).reshape(w.shape)

# ---------- 学生权重(量化 gguf) ----------
from gguf import GGUFReader
from gguf.quants import dequantize as gg_deq
_R = GGUFReader(GGUF)
_T = {t.name: t for t in _R.tensors}
_GG = {"w1": "ffn_gate_exps", "w3": "ffn_up_exps", "w2": "ffn_down_exps"}

def student_w(blk, e, nm):
    t = _T[f"mtp.{blk}.{_GG[nm]}.weight"]
    ne = [int(x) for x in t.shape]
    nexp, rows, cols = ne[-1], ne[-2], ne[0]
    db = t.data.reshape(-1); per = db.size // nexp
    w = gg_deq(db[e * per:(e + 1) * per], t.tensor_type)
    return np.ascontiguousarray(w.reshape(rows, cols).astype(np.float32))

def swiglu(g, u, lim):
    g = xp.minimum(g, lim); u = xp.clip(u, -lim, lim)
    return (g / (1.0 + xp.exp(-g))) * u

# ---------- 主流程 ----------
t0 = time.time()
X, SEL, RW = load_anchor(ANCHOR, BLOCK)
S = X.shape[0]
print(f"block {BLOCK}: 锚行 S={S} 唯一专家={len(set(SEL.ravel().tolist()))}", flush=True)
Xg = A(X.astype(np.float32))
dH = xp.zeros((S, D), dtype=xp.float32)
yq = xp.zeros((S, D), dtype=xp.float32)
# 引擎在空槽写 -1(路由权重 0, kernel 侧 clamp 到 0 号但权重为 0 无贡献): 直接跳过
need = sorted(set(int(e) for e in SEL.ravel() if e >= 0))
for i, e in enumerate(need):
    rows, slots = np.where(SEL == e)
    if len(rows) == 0: continue
    w = A(RW[rows, slots].astype(np.float32))[:, None]
    xs = Xg[A(rows)]
    Wt = {nm: A(teacher_w(BLOCK, e, nm)) for nm in ("w1", "w3", "w2")}
    Ws = {nm: A(student_w(BLOCK, e, nm)) for nm in ("w1", "w3", "w2")}
    yt = swiglu(xs @ Wt["w1"].T, xs @ Wt["w3"].T, SWLIM) @ Wt["w2"].T
    ys = swiglu(xs @ Ws["w1"].T, xs @ Ws["w3"].T, SWLIM) @ Ws["w2"].T
    ri = A(rows)
    dH[ri] += w * (yt - ys)
    yq[ri] += w * ys
    del Wt, Ws, yt, ys
    if (i + 1) % 8 == 0: print(f"  专家 {i+1}/{len(need)} (e={e})  {time.time()-t0:.0f}s", flush=True)

print(f"  专家循环收官 {time.time()-t0:.0f}s", flush=True)

# ---------- 数据体检(2026-08-21: block1 的 cuSOLVER SVD 因 NaN 卡死, CPU 0% 空转) ----------
def _bad(t):
    return int(N_(xp.isnan(t).sum())) + int(N_(xp.isinf(t).sum()))
nb_dh, nb_yq = _bad(dH), _bad(yq)
if nb_dh or nb_yq:
    print(f"  ★体检: dH 非有限 {nb_dh} / yq 非有限 {nb_yq} → 置零后继续", flush=True)
    dH = xp.nan_to_num(dH, nan=0.0, posinf=0.0, neginf=0.0)
    yq = xp.nan_to_num(yq, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  数据: |dH| mean={float(xp.abs(dH).mean()):.4g} |yq| mean={float(xp.abs(yq).mean()):.4g} "
      f"|X| mean={float(xp.abs(Xg).mean()):.4g}", flush=True)

# ---------- ELM 闭式解(与 amp_solve.py 同式) ----------
NFIT = int(S * NFIT_FRAC)
tr, ev = np.arange(0, NFIT), np.arange(NFIT, S)
yfp = yq + dH
eps = xp.sqrt((yq[A(tr)] ** 2).mean(0)) * 1e-2 + 1e-12
R = (dH * yq / (yq ** 2 + eps[None, :] ** 2))
colw = xp.sqrt(R[A(tr)].var(0) + 1e-12) * xp.sqrt((yq[A(tr)] ** 2).mean(0) + 1e-12)
Xtr = Xg[A(tr)]
rms = xp.sqrt((Xtr ** 2).mean(1, keepdims=True))
rs = np.random.RandomState(1)
Xa = xp.vstack([Xtr, Xtr + A(rs.randn(len(tr), D).astype(np.float32)) * 0.04 * rms])
Ra = xp.vstack([R[A(tr)] * colw, R[A(tr)] * colw])
e0 = float(((yfp[A(ev)] - yq[A(ev)]) ** 2).sum())
# k 上限随样本量放开(2026-08-21 扩锚): 885 行时 k=384 已顶格 ⇒ 欠拟合; 样本足时试到 768。
KMAX = min(768, max(256, int(len(tr) * 0.55)))
Xa = xp.nan_to_num(Xa, nan=0.0, posinf=0.0, neginf=0.0)
Ra = xp.nan_to_num(Ra, nan=0.0, posinf=0.0, neginf=0.0)
_t_svd = time.time()
try:   # cuSOLVER 对病态输入会长时间不收敛(CPU 0% 空转) — 超时回退 numpy
    _, _, Vt = np.linalg.svd(N_(Xa - Xa.mean(0)), full_matrices=False)
    Vt = A(Vt.astype(np.float32))
except Exception as ex:
    print("  SVD 失败, 用随机基:", ex, flush=True)
    Vt = A((np.random.RandomState(3).randn(KMAX, D) / np.sqrt(D)).astype(np.float32))
print(f"  SVD {time.time()-_t_svd:.0f}s", flush=True)
KMAX = min(KMAX, int(Vt.shape[0]))
CANDS = {"PCA": Vt[:KMAX].T, "rand": A(np.random.RandomState(7).randn(D, KMAX).astype(np.float32) / np.sqrt(D))}
scale = float(xp.sqrt((Xa ** 2).mean()))
Xev = Xg[A(ev)]; yq_ev = yq[A(ev)]; yfp_ev = yfp[A(ev)]
best = (0.0, None)
diag = xp.arange(KMAX)
for nm, V0 in CANDS.items():
    Za = xp.tanh(Xa @ V0 / scale); Ze = xp.tanh(Xev @ V0 / scale)
    ZtZ = Za.T @ Za; ZtR = Za.T @ Ra
    for lam in (1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 1.0):
        G = ZtZ.copy(); G[diag, diag] += lam * float(xp.trace(ZtZ)) / KMAX + 1e-10
        U = xp.linalg.solve(G, ZtR)
        for k in [x for x in (16, 32, 64, 128, 256, 384, 512, 640, 768) if x <= KMAX]:
            g = (Ze[:, :k] @ (U[:k] / colw[None, :]))
            rec = 1 - float(((yfp_ev - yq_ev * (1.0 + g)) ** 2).sum()) / e0
            if rec > best[0]: best = (rec, (nm, lam, k, N_(U[:k]).copy()))

def hdr(nm, psz):
    h = bytearray(116); h[0:len(nm)] = nm.encode()
    struct.pack_into('<Q', h, 88, psz); struct.pack_into('<i', h, 112, 1)
    return bytes(h)

outp = os.path.join(OUTDIR, f"zrec_L{BLOCK:02d}.bin")
if best[0] <= 0.005 or best[1] is None:
    open(outp, 'wb').write(hdr("zl.AMP", 0))
    print(f"★block {BLOCK}: held {best[0]*100:.2f}% ≤0.5% → 层闸 | {time.time()-t0:.0f}s", flush=True)
else:
    nm, lam, k, U = best[1]
    cw = N_(colw)
    Ueng = np.ascontiguousarray((U / cw[None, :]).T.astype(np.float16))
    Veng = np.ascontiguousarray(N_(CANDS[nm][:, :k]).astype(np.float16))
    z1 = np.ones(k, dtype=np.float16)
    pay = struct.pack('<IfII', k, scale, D, D) + z1.tobytes() + Ueng.tobytes() + Veng.tobytes()
    open(outp, 'wb').write(hdr("zl.AMP", len(pay)) + pay)
    print(f"★block {BLOCK}: held 行为挽回 {best[0]*100:.2f}% @V0={nm} λ={lam} k={k} "
          f"体积 {len(pay)/2**20:.1f}MB | {time.time()-t0:.0f}s", flush=True)
