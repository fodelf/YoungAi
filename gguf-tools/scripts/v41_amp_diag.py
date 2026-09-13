"""v41_amp_diag — 放大器的【逐行】诊断挂钩(2026-09-12)。只做编排与统计上屏, 数值全在 C 库。

要回答的问题: 逐层 held-out 能量增益全正(+9.1%), 端到端却从 PPL 比 1.453 崩到 2.380 —— 分叉
在哪? 假设: 能量口径被少数巨值行(massive 激活 token)垄断, 最小二乘伺候它们, 普通行被修坏。
判据: 同一层同一放大器, 【能量增益】与【逐行中位增益】并排; 顶 1% 行占的能量份额; 改善行比例。
"""
import ctypes

import numpy as np
import torch

import v41_amp_hooks as amph


def _rowdiag(lib, xf, yfp, qf, dA, dB, K):
    """返回 (行靶, 行残差, 行y_q, 列靶, 列残差, 列y_q), 全是范数(非平方)。"""
    n, d = xf.shape
    bufs = [(ctypes.c_float * n)() for _ in range(3)] + [(ctypes.c_float * d)() for _ in range(3)]
    rc = lib.v41_amp_rowdiag_gpu(ctypes.c_void_p(xf.data_ptr()), ctypes.c_void_p(yfp.data_ptr()),
                                 ctypes.c_void_p(qf.data_ptr()), ctypes.c_void_p(dA.data_ptr()),
                                 ctypes.c_void_p(dB.data_ptr()), n, d, K, *bufs)
    if rc != 0:
        return None
    return tuple(np.frombuffer(b, np.float32).copy() for b in bufs)


def _colstats(tgt, res, yq):
    """通道账: 靶能量是不是被少数通道(massive channel)垄断; 修正后残差留在哪些通道。
    列加权不改变满秩最小二乘解(各输出列独立解), 只改变 SVD 截断保留的方向 —— 所以
    通道垄断 + 低秩截断 = 秩全花在巨值通道上, 普通通道的误差原样留下。"""
    t2, r2, q2 = tgt ** 2, res ** 2, yq ** 2
    order = np.argsort(-t2)
    def share(k): return 100 * t2[order[:k]].sum() / t2.sum()
    top16 = order[:16]
    rest = np.ones(len(tgt), bool); rest[order[:64]] = False
    g_top = 100 * (1 - r2[order[:64]].sum() / t2[order[:64]].sum())
    g_rest = 100 * (1 - r2[rest].sum() / t2[rest].sum())
    print(f"    通道: 靶能量 顶1通道 {share(1):.0f}% 顶16 {share(16):.0f}% 顶64 {share(64):.0f}% (共{len(tgt)}) | "
          f"修正后 顶64通道能量增益 {g_top:+.1f}% vs 其余通道 {g_rest:+.1f}% | "
          f"顶16 通道号 {top16.tolist()} 其 y_q 能量份额 {100 * q2[top16].sum() / q2.sum():.0f}%", flush=True)


def _stats(tag, tgt, res, yq, rows):
    """一段(拟合/val)的账: 能量增益 vs 逐行增益; 顶 1% 行的能量份额; 改善行比例。"""
    t2, r2 = tgt ** 2, res ** 2
    e_gain = 100 * (1 - r2.sum() / t2.sum())
    row_gain = 100 * (1 - r2 / np.maximum(t2, 1e-30))          # 每行自己的增益
    k = max(1, len(tgt) // 100)
    top = np.argsort(-t2)[:k]
    share = 100 * t2[top].sum() / t2.sum()
    mask = np.ones(len(tgt), bool); mask[top] = False
    e_gain_rest = 100 * (1 - r2[mask].sum() / t2[mask].sum())
    rel = tgt / np.maximum(yq, 1e-30)
    print(f"    {tag}: 能量增益 {e_gain:+.1f}% | 逐行增益 中位 {np.median(row_gain):+.1f}% p10 {np.percentile(row_gain, 10):+.1f}% "
          f"改善行 {100 * (row_gain > 0).mean():.0f}% | 顶1%行({k}行)占能量 {share:.0f}%, 去掉它们后能量增益 {e_gain_rest:+.1f}% "
          f"| 靶/‖y_q‖ 逐行中位 {np.median(rel):.3f}(能量口径 {np.sqrt(t2.sum() / (yq ** 2).sum()):.3f})", flush=True)
    return top


def _seg(tag, t1, t2, tot, res):
    """一段行的 T1/T2 账: 能量口径。‖T1+T2‖² = ‖T1‖²+‖T2‖²+2·T1·T2 ⇒ 夹角余弦从三个范数反解。"""
    e1, e2, et, er = (t1 ** 2).sum(), (t2 ** 2).sum(), (tot ** 2).sum(), (res ** 2).sum()
    cos = (et - e1 - e2) / (2 * np.sqrt(e1 * e2) + 1e-30)
    print(f"    {tag}: ‖T1‖²={e1:.3g} ‖T2‖²={e2:.3g} ‖T1+T2‖²={et:.3g} (T2/T1 能量比 {e2 / e1:.2f}) "
          f"cos(T1,T2)={cos:+.3f} | 修掉 T1 后总误差 {100 * (1 - er / et):+.1f}% "
          f"(按行中位 {100 * np.median(1 - res ** 2 / np.maximum(tot ** 2, 1e-30)):+.1f}%, 改善行 {100 * (res < tot).mean():.0f}%)",
          flush=True)


def install_t2diag(net, dev, qmode, perm, K, lam, dump_dir):
    """★靶口径诊断★ 部署相关误差 = y_q(x_q) − y_fp(x_fp) = T1 + T2:
         T1 = y_q(x_q) − y_fp(x_q)   同 x 靶(放大器现在修的)
         T2 = y_fp(x_q) − y_fp(x_fp) FP ffn 对上游输入误差的响应(同 x 靶看不见它)
    y_fp(x_fp) 来自教师自然跑的 --dump-moe(同一串 token, 逐位置对齐)。每层用 (λ,K) 解同 x 放大器,
    报: T2/T1 能量比、cos(T1,T2)、修掉 T1 后【总误差】涨还是跌 —— 层内全绿但端到端负, 就看这一行。
    学生走裸量化态(不应用修正)。"""
    lib = amph._lib()
    lib.v41_amp_rowdiag_gpu.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.POINTER(ctypes.c_float)] * 6
    lib.v41_amp_rowdiag_gpu.restype = ctypes.c_int
    busy = {"v": False}
    nf = perm["nfit"]
    dd = Path(dump_dir)

    def mk(li):
        def h(mod, args, out):
            if busy["v"]:
                return None
            n, d, xf, qf, yfp = amph._same_x_target(mod, args, out, qmode, busy, li)
            f = dd / f"y_L{li:02d}.bin"
            if not f.exists():
                print(f"  [L{li:02d}] ★缺 {f.name}, 跳过★", flush=True); return None
            ydump = torch.from_numpy(np.fromfile(f, dtype=np.float32).reshape(n, d)).to(dev).contiguous()
            xp, qp, yp, ydp = amph._permute(perm, dev, xf, qf, yfp, ydump)
            dA = torch.zeros(K, d, device=dev, dtype=torch.float32)
            dB = torch.zeros(d, K, device=dev, dtype=torch.float32)
            r = ctypes.c_float(0)
            rc = lib.v41_amp_solve_layer_gpu(
                ctypes.c_void_p(xp[:nf].data_ptr()), ctypes.c_void_p(yp[:nf].data_ptr()),
                ctypes.c_void_p(qp[:nf].data_ptr()), nf, d, K, ctypes.c_float(lam),
                ctypes.c_void_p(dA.data_ptr()), ctypes.c_void_p(dB.data_ptr()), ctypes.byref(r), 0)
            if rc != 0:
                print(f"  [L{li:02d}] ★解算不过★", flush=True); return None
            zA = torch.zeros_like(dA); zB = torch.zeros_like(dB)
            t1, t1r = _rowdiag(lib, xp, yp, qp, dA, dB, K)[:2]          # ‖T1‖, ‖T1−Δ‖
            tot, totr = _rowdiag(lib, xp, ydp, qp, dA, dB, K)[:2]       # ‖T1+T2‖, ‖T1+T2−Δ‖
            t2 = _rowdiag(lib, xp, yp, ydp, zA, zB, K)[0]               # ‖y_fp(x_q) − y_fp(x_fp)‖ = ‖T2‖
            print(f"  [L{li:02d}] λ={lam:g} K={K}  同x靶 train {100 * (1 - r.value):+.1f}%(幅值比)", flush=True)
            _seg("拟合", t1[:nf], t2[:nf], tot[:nf], totr[:nf])
            _seg("val ", t1[nf:], t2[nf:], tot[nf:], totr[nf:])
            return None
        return h

    nl = 0
    for i, ffn in amph._moe_layers(net):
        ffn.register_forward_hook(mk(i)); nl += 1
    print(f"[靶口径诊断] {nl} 层挂钩, λ={lam:g} K={K}, 教师 dump ← {dd}; 不应用修正", flush=True)


def install_rowdiag(net, dev, qmode, perm, K, lam, whiten=0):
    """每层: 同 x 靶 → 按 layout 置换 → 用 (λ,K) 在拟合行解 → 逐行账(拟合段 / val 段分开)。
    不应用修正(纯诊断, 每层都在裸量化传播态上量, 与 scan 同基准)。"""
    lib = amph._lib()
    lib.v41_amp_rowdiag_gpu.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.POINTER(ctypes.c_float)] * 6
    lib.v41_amp_rowdiag_gpu.restype = ctypes.c_int
    busy = {"v": False}
    nf = perm["nfit"]
    pos = perm["perm"].numpy()                      # 置换后第 i 行 = 原始第 pos[i] 个 token

    def mk(li):
        def h(mod, args, out):
            if busy["v"]:
                return None
            n, d, xf, qf, yfp = amph._same_x_target(mod, args, out, qmode, busy, li)
            xp, qp, yp = amph._permute(perm, dev, xf, qf, yfp)
            dA = torch.zeros(K, d, device=dev, dtype=torch.float32)
            dB = torch.zeros(d, K, device=dev, dtype=torch.float32)
            r = ctypes.c_float(0)
            rc = lib.v41_amp_solve_layer_gpu(
                ctypes.c_void_p(xp[:nf].data_ptr()), ctypes.c_void_p(yp[:nf].data_ptr()),
                ctypes.c_void_p(qp[:nf].data_ptr()), nf, d, K, ctypes.c_float(lam),
                ctypes.c_void_p(dA.data_ptr()), ctypes.c_void_p(dB.data_ptr()), ctypes.byref(r), int(whiten))
            if rc != 0:
                print(f"  [L{li:02d}] ★解算不过★", flush=True)
                return None
            out3 = _rowdiag(lib, xp, yp, qp, dA, dB, K)
            if out3 is None:
                print(f"  [L{li:02d}] ★rowdiag 失败★", flush=True)
                return None
            tgt, res, yq, ctgt, cres, cyq = out3
            print(f"  [L{li:02d}] λ={lam:g} K={K} 白化={whiten}", flush=True)
            _stats("拟合", tgt[:nf], res[:nf], yq[:nf], pos[:nf])
            top = _stats("val ", tgt[nf:], res[nf:], yq[nf:], pos[nf:])
            _colstats(ctgt, cres, cyq)
            # 巨值行长什么样: 原始位置、窗内偏移(0 = 窗首 = 上下文断点)
            big = [(int(pos[nf + i]), float(tgt[nf + i]), float(yq[nf + i])) for i in top[:6]]
            print("    val 顶行: " + "  ".join(f"pos{p}(窗内{p % 128}) 靶{t:.0f}/yq{q:.0f}" for p, t, q in big), flush=True)
            return None
        return h

    nl = 0
    for i, ffn in amph._moe_layers(net):
        ffn.register_forward_hook(mk(i))
        nl += 1
    print(f"[逐行诊断] {nl} 层挂钩, λ={lam:g} K={K} 白化={whiten}, 分层 held-out {nf} 解 / {len(pos) - nf} 评; 不应用修正", flush=True)
