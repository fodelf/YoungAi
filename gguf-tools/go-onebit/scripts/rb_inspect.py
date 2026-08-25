#!/usr/bin/env python3
"""rb_inspect.py — 检查路由偏置侧车(RBIA)的有效性(2026-08-01 用户问"路由反修生效了吗")。

路由反修是三段式, 这个文件是第一段的产物:
  ① 量化阶段  收集 Δb(FP锚 top-k vs 学生 top-k 的 margin 统计)  ← 本文件
  ② 反修阶段  修权重/侧车(不碰路由; do_quant=0 时统计门控关闭)
  ③ 合并阶段  烘焙 gate_bias += α·Δb 进 exp_probs_b            ← 生效点
所以看到"反修中路由一致率还在掉"是正常的 —— Δb 尚未烘入。

用法: rb_inspect.py <route_bias.bin> [mincnt=8]
"""
import struct, sys, statistics as st

p = sys.argv[1]
MINCNT = int(sys.argv[2]) if len(sys.argv) > 2 else 8
f = open(p, 'rb')
hd = struct.unpack('<4I', f.read(16))
NL, NE = hd[1], hd[2]
ok = hd[0] == 0x41494252
print(f"头 magic={hd[0]:#x} NL={NL} NE={NE}  {'RBIA OK' if ok else '**HEADER BAD**'}")
if not ok:
    sys.exit(1)
acc = struct.unpack(f'<{NL*NE}f', f.read(NL * NE * 4))
cnt = struct.unpack(f'<{NL*NE}I', f.read(NL * NE * 4))

nz = sum(1 for x in acc if x != 0.0)
armed = [i for i in range(NL * NE) if cnt[i] >= MINCNT and acc[i] != 0.0]
print(f"非零槽 {nz}/{NL*NE}   武装槽(cnt>={MINCNT}) {len(armed)}   margin 事件总数 {sum(cnt):,}")
v = [acc[i] for i in armed]
if v:
    print(f"Delta-b 值域 [{min(v):+.5f}, {max(v):+.5f}]  均值 {st.mean(v):+.5f}  标准差 {st.pstdev(v):.5f}")
    pos = sum(1 for x in v if x > 0)
    print(f"正/负 {pos}/{len(v)-pos}  (正=该专家被 FP 锚选中但学生漏选, 需抬高 gate)")
print()
print("逐层武装槽:")
for L0 in range(0, NL, 6):
    row = []
    for L in range(L0, min(L0 + 6, NL)):
        a = sum(1 for e in range(NE) if cnt[L * NE + e] >= MINCNT and acc[L * NE + e] != 0.0)
        row.append(f"L{L:02d}:{a:3d}")
    print("  " + "  ".join(row))
print()
zero_layers = [L for L in range(NL)
               if not any(cnt[L * NE + e] >= MINCNT and acc[L * NE + e] != 0.0 for e in range(NE))]
print(f"零武装层 {len(zero_layers)}/{NL}" + (f": {zero_layers}" if zero_layers else " (L00-L02 是 hash 路由层, 无 gate 漂移属正常)"))
