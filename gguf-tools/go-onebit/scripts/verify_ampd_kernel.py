#!/usr/bin/env python3
"""verify_ampd_kernel.py — type9(zl.AMPD 动态 z 放大器) 的引擎 kernel 端到端对拍。

【为什么需要】仓库里 zchain 一个测试都没有, type9 是新加的语义
    routed ⊙ (1 + U·[tanh(Aᵀx/s) ⊙ tanh(Vᵀx/s)])
宿主 apply / CUDA 通用核 两处各写了一遍, 载荷布局又和 type7 不同(A|U|V vs z|U|V)。
偏移写错不会崩、不会报错, 只会静默出垃圾 —— 判决数字就全废了。

【怎么验(用真实产物, 不造假数据)】引擎里 zchain 的应用点(ds4_gpu_zchain_scale_routed)
在捕获点(cap_batch_layer)**之前**, 所以带侧车跑一遍捕获, raw_ffn_out 就是 post-AMP 的:
    g_engine = raw_ffn_out(带侧车) / raw_ffn_out(裸) − 1
再用侧车里的 A/U/V 和裸捕的 x 在 numpy 里按公式算 g_numpy, 两者比。
★L0 最干净★: 上游没有东西, 两次跑的 x 逐位相同, g 可以精确对拍;
深层因为 AMP 改了上游出口, x 本身就变了, 只做量级/相关性参考。

用法: verify_ampd_kernel.py <zrec_dir> <cap_bare> <cap_amp> [层=0]
"""
import os, sys, struct
import numpy as np

zdir, cap0, cap1 = sys.argv[1], sys.argv[2], sys.argv[3]
L = int(sys.argv[4]) if len(sys.argv) > 4 else 0
D = 4096

def load_rec(path):
    b = open(path, "rb").read()
    nm = b[:16].split(b"\0")[0].decode()
    psz = struct.unpack_from("<Q", b, 88)[0]
    pay = b[116:116 + psz]
    k, scale, din, dout = struct.unpack_from("<IfII", pay, 0)
    h = np.frombuffer(pay[16:], dtype=np.float16).astype(np.float32)
    n = din * k
    A = h[:n].reshape(din, k)
    U = h[n:n + dout * k].reshape(dout, k)
    V = h[n + dout * k:n + dout * k + din * k].reshape(din, k)
    return nm, k, scale, A, U, V

def cap(d, nm, cols):
    a = np.fromfile(os.path.join(d, f"{nm}_L{L}"), dtype=np.float16).reshape(-1, cols)
    return a.astype(np.float32)

nm, k, scale, A, U, V = load_rec(os.path.join(zdir, f"zrec_L{L:02d}.bin"))
print(f"侧车 {nm}  L{L}  k={k} scale={scale:.4f}")
if nm != "zl.AMPD":
    print("★不是 AMPD 记录, 这个脚本不适用★"); sys.exit(2)

x   = cap(cap0, "raw_ffn_in", D)
y0  = cap(cap0, "raw_ffn_out", D)
y1  = cap(cap1, "raw_ffn_out", D)
n = min(len(x), len(y0), len(y1)); x, y0, y1 = x[:n], y0[:n], y1[:n]

# 引擎实际做了什么: g = y1/y0 − 1 (只在 |y0| 不接近 0 的位置可信)
m = np.abs(y0) > (np.abs(y0).mean() * 0.05)
g_eng = np.where(m, y1 / np.where(m, y0, 1.0) - 1.0, 0.0)

# numpy 参考: g = U·[tanh(Aᵀx/s) ⊙ tanh(Vᵀx/s)]
f = np.tanh(x @ V / scale) * np.tanh(x @ A / scale)
g_ref = f @ U.T

sel = m
d = np.abs(g_eng[sel] - g_ref[sel])
den = np.abs(g_ref[sel]) + 1e-6
print(f"  可比元素 {sel.sum()}/{sel.size} ({100*sel.mean():.1f}%)")
print(f"  g_ref  幅度: 中位 {np.median(np.abs(g_ref[sel])):.5f}  p99 {np.percentile(np.abs(g_ref[sel]),99):.5f}")
print(f"  g_eng  幅度: 中位 {np.median(np.abs(g_eng[sel])):.5f}  p99 {np.percentile(np.abs(g_eng[sel]),99):.5f}")
print(f"  绝对差: 中位 {np.median(d):.2e}  p99 {np.percentile(d,99):.2e}  max {d.max():.2e}")
print(f"  相对差: 中位 {np.median(d/den):.2e}  p99 {np.percentile(d/den,99):.2e}")
c = float(np.corrcoef(g_eng[sel].ravel(), g_ref[sel].ravel())[0, 1])
print(f"  相关系数: {c:.6f}")
print()
if c > 0.99 and np.median(d/den) < 0.05:
    print("判决: ✓ 引擎 kernel 与公式一致")
else:
    print("判决: ★不一致 — 偏移/布局/语义有错, 判决数字不可用★")
