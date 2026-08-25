#!/usr/bin/env python3
"""layer_detail.py — 提取【单层全机制】详细数据(2026-08-01 用户令"每层详细数据都要列出来
包括 z变量/四损失/感知/向后/当层质量, 不每层观测错了就浪费时间")。

量化阶段(DS4_TUNE=1)每层跑完整菜单, 日志里散落着:
  [R28] Lxx 档位          → 配置
  SEARCH L=x QT[...]      → 1bit 基座候选(调优前 held)
  TABREC L=x <算法> ...   → 每个机制的体积/指标/判定
  VQ_GATE L=xx            → 冷热重建 cos
  ★贪心选 / ★体积★       → 定稿 held(调优后)+ 落盘字节
本工具把它们按层聚合成一张表, 避免逐条翻日志漏看。

TABREC 的 m=[...] 四元组 = [val, held, 旋钮值, 最优参数], 判定: 1=落地 2=正向未落地 3=零接管。
用法: layer_detail.py <log> <层号> [--brief]
"""
import re, sys

log, L = sys.argv[1], int(sys.argv[2])
brief = '--brief' in sys.argv
txt = open(log, errors='ignore').read().splitlines()

tier = gate = greedy = vol = qt = rb = None
recs = []
for ln in txt:
    m = re.match(rf'\[R28\] L0*{L} 档位 (.+)', ln)
    if m: tier = m.group(1).strip()
    m = re.match(rf'VQ_GATE L=0*{L} (.+?) →', ln)
    if m: gate = m.group(1).strip()
    m = re.match(rf'L0*{L} ★贪心选 (.+)', ln)
    if m: greedy = m.group(1).strip()
    m = re.match(rf'L0*{L} ★体积★ (.+)', ln)
    if m: vol = m.group(1).strip()
    m = re.match(rf'SEARCH L={L} QT\[(.+?)\] val=([\d.]+) held=([\d.]+)', ln)
    if m: qt = (m.group(1), float(m.group(2)), float(m.group(3)))
    m = re.match(rf'L0*{L} \[路由\] Δb 就位\(武装槽=(\d+)\)', ln)
    if m: rb = int(m.group(1))
    m = re.match(rf'TABREC L={L} (\S+)\s+vol=(\d+) m=\[([\d.\-e ]+)\] 判定=(\d)\s*\|?\s*(.*)', ln)
    if m:
        vals = [float(x) for x in m.group(3).split()]
        recs.append((m.group(1), int(m.group(2)), vals, int(m.group(4)), m.group(5).strip()))

VERD = {1: '✓落地', 2: '正向未落地', 3: '零接管'}


def fam(n):
    if n.startswith('z.') or n.startswith('zl.'): return 'z变量'
    if n.startswith('loss.'): return '四损失'
    if n.startswith('pc.') or n.startswith('percept'): return '感知'
    if n.startswith('bwd.'): return '向后'
    if n.startswith('lf.') or n.startswith('ls.') or n.startswith('la.'): return '损失·动态'
    return '其他'


print(f"╔══ L{L:02d} 全机制详细数据 ══")
print(f"║ 配置   : {tier or '—'}")
if qt:  print(f"║ 1bit基座: {qt[0]}  val={qt[1]:.4f} held={qt[2]:.4f}(调优前)")
if gate: print(f"║ VQ门   : {gate}")
if rb is not None: print(f"║ 路由Δb : 武装槽 {rb}")
if greedy: print(f"║ ★定稿  : {greedy}(调优后)")
if vol: print(f"║ ★体积  : {vol}")
print("╠══ 机制明细 ══")
if not recs:
    print("║ (本层 TABREC 尚未产出)")
else:
    print(f"║ {'族':<8}{'算法':<14}{'体积':>9}  {'val':>7} {'held':>7}  判定   说明")
    order = {'z变量': 0, '损失·动态': 1, '四损失': 2, '感知': 3, '向后': 4, '其他': 5}
    for n, v, m, jd, note in sorted(recs, key=lambda r: (order.get(fam(r[0]), 9), r[0])):
        if brief and jd != 1: continue
        vs = f"{v}B" if v < 1024 else (f"{v/1024:.1f}KB" if v < 1048576 else f"{v/1048576:.2f}MiB")
        va = f"{m[0]:.4f}" if len(m) > 0 else '—'
        he = f"{m[1]:.4f}" if len(m) > 1 else '—'
        print(f"║ {fam(n):<8}{n:<14}{vs:>9}  {va:>7} {he:>7}  {VERD.get(jd,jd):<6} {note[:38]}")
    land = [r for r in recs if r[3] == 1]
    tot = sum(r[1] for r in land)
    ts = f"{tot}B" if tot < 1024 else (f"{tot/1024:.1f}KB" if tot < 1048576 else f"{tot/1048576:.2f}MiB")
    print(f"╠══ 落地 {len(land)}/{len(recs)} 项, 侧车合计 {ts} ══")
print("╚" + "═" * 30)
