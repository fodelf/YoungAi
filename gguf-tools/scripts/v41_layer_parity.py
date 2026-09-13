#!/usr/bin/env python3
"""v41_layer_parity.py — 引擎 vs Python 逐层 MoE 入/出对拍(2026-09-12, P2 定位夹具, 只读数不算模型)。

两边都落 f32 [n][5120]: Python `--dump-moe DIR` → DIR/x_Lnn.bin, y_Lnn.bin; 引擎(n ≤ 64 自动) → <out>.x_Lnn.bin。
每层打: x 的相对误差 ‖Δ‖/‖ref‖、cos; y 同; 误差在哪一层跳起来, 差就在那一层的那个块里。
用法: v41_layer_parity.py <python dump dir> <engine out prefix> <n_layers=40>
"""
import sys
from pathlib import Path

import numpy as np

pd, ep = Path(sys.argv[1]), sys.argv[2]
nl = int(sys.argv[3]) if len(sys.argv) > 3 else 40
# 第二个参数也可以是另一个 Python dump 目录(Python 口径 A vs B, 看两条 bf16 流水线自身的分叉地板)
if Path(ep).is_dir():
    ep = ep.rstrip("/") + "/"   # 目录前缀: "<dir>/" + "x_Lnn.bin"(Path(d)/"" 不会带尾斜杠, 别用)
    print(f"[口径] 目录 vs 目录: {pd} vs {ep}")
    _fmt = lambda kind, L: f"{ep}{kind}_L{L:02d}.bin"
else:
    _fmt = lambda kind, L: f"{ep}.{kind}_L{L:02d}.bin"


def rel(a, b):
    d = np.linalg.norm(a - b); r = np.linalg.norm(b)
    cos = float((a * b).sum() / (np.linalg.norm(a) * r + 1e-30))
    return d / (r + 1e-30), cos


# engram 对拍(有文件才做): hash id 逐个相等? engram 后 hc 相对误差?
hid = pd / "hash_ids.bin"
if hid.exists() and not Path(ep).is_dir():
    ids = np.fromfile(hid, np.int64)
    eng_layers = sorted(int(p.stem.split("_L")[1]) for p in Path(ep).parent.glob(Path(ep).name + ".erows_L*.txt"))
    if eng_layers:
        ne = len(eng_layers); ids = ids.reshape(-1, ne, 24)   # 官方 [B, L, n_engram_layers, n_hash_cols]
        for k, L in enumerate(eng_layers):
            er = np.loadtxt(f"{ep}.erows_L{L:02d}.txt", dtype=np.int64).reshape(-1, 24)
            n = min(er.shape[0], ids.shape[0]); same = int((er[:n] == ids[:n, k]).all(1).sum())
            bad = [t for t in range(n) if not (er[t] == ids[t, k]).all()]
            print(f"[engram L{L:02d}] hash 行号: {same}/{n} 位置全同; 不同的位置: {bad[:8]}")
            ph, eh = pd / f"hce_L{L:02d}.bin", Path(f"{ep}.hce_L{L:02d}.bin")
            if ph.exists() and eh.exists():
                a, b = np.fromfile(eh, np.float32), np.fromfile(ph, np.float32)
                m = min(a.size, b.size); a, b = a[:m].reshape(-1, 4 * 5120), b[:m].reshape(-1, 4 * 5120)
                r, c = rel(a, b); perrow = np.linalg.norm(a - b, axis=1) / (np.linalg.norm(b, axis=1) + 1e-30)
                print(f"[engram L{L:02d}] engram 后 hc: 相对误差 {r:.4e} cos {c:.5f}; 前 8 位置逐行: " + " ".join(f"{v:.3e}" for v in perrow[:8]))
print(f"{'层':>3} | {'x 相对误差':>10} {'x cos':>8} | {'y 相对误差':>10} {'y cos':>8} | 备注")
for L in range(nl):
    px, py = pd / f"x_L{L:02d}.bin", pd / f"y_L{L:02d}.bin"
    ex, ey = Path(_fmt("x", L)), Path(_fmt("y", L))
    if not (px.exists() and ex.exists()):
        print(f"{L:3d} | 缺文件"); continue
    a, b = np.fromfile(ex, np.float32), np.fromfile(px, np.float32)
    n = min(a.size, b.size); a, b = a[:n].reshape(-1, 5120), b[:n].reshape(-1, 5120)
    rx, cx = rel(a, b)
    note = ""
    if py.exists() and ey.exists():
        a2, b2 = np.fromfile(ey, np.float32), np.fromfile(py, np.float32)
        m = min(a2.size, b2.size); a2, b2 = a2[:m].reshape(-1, 5120), b2[:m].reshape(-1, 5120)
        ry, cy = rel(a2, b2)
        worst = int(np.argmax(np.linalg.norm(a2 - b2, axis=1) / (np.linalg.norm(b2, axis=1) + 1e-30)))
        note = f"y 最差行 t={worst}"
        print(f"{L:3d} | {rx:10.4e} {cx:8.5f} | {ry:10.4e} {cy:8.5f} | {note}")
    else:
        print(f"{L:3d} | {rx:10.4e} {cx:8.5f} | {'-':>10} {'-':>8} |")
