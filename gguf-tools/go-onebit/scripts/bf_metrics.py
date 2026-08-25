#!/usr/bin/env python3
"""bf_metrics.py — 从反修日志提取【每层详细指标】(2026-08-01 用户令"反修每一层详细数据")。

输出两张表:
  ① 逐层收官表: held 基座→定稿 / 落地机制清单 / 新增侧车体积 / 路由一致 / 空专家 / 内存
  ② 逐层机制明细: 每个候选(z/四损失/感知/向后/路由α)的 val/held/提升/判定

用法: bf_metrics.py <log> [--detail Lxx]
"""
import re, sys, collections

log = sys.argv[1]
want = None
if '--detail' in sys.argv:
    want = int(sys.argv[sys.argv.index('--detail') + 1].lstrip('Lx'))

# 只取最后一次"反修起跑"之后的段
lines = open(log, errors='ignore').read().splitlines()
start = 0
for i, l in enumerate(lines):
    if '反修起跑' in l:
        start = i
lines = lines[start:]

items = collections.defaultdict(list)   # L -> [(类别, 算法, 体积, val, held, 提升, 判定)]
close = {}                               # L -> dict
base = {}                                # L -> 基座 held
alpha = {}                               # L -> (选中α, before, after)
ascan = collections.defaultdict(list)    # L -> [(α, relh)]
mem = {}

for l in lines:
    m = re.search(r'【L(\d+)】(\S+?)\s+算法=([^ ]+?)(?:\(|\s)', l)
    if m and '还原度=' in l:
        L = int(m.group(1)); cat = m.group(2); algo = m.group(3)
        v = re.search(r'val=([\d.]+)', l); h = re.search(r'held=([\d.]+)', l)
        vol = re.search(r'体积=(\S+?)\s', l); up = re.search(r'提升(-?[\d.]+)%', l)
        vd = re.search(r'研判=(\S+)', l)
        if v and h:
            items[L].append((cat, algo, vol.group(1) if vol else '-',
                             float(v.group(1)), float(h.group(1)),
                             float(up.group(1)) if up else None,
                             vd.group(1) if vd else ''))
            if cat == '量化' and '1bit' in algo:
                base[L] = float(h.group(1))
    m = re.search(r'\| 累积relL2 fit=([\d.]+) held=([\d.]+) \| 路由一致=\s*([\d.]+)% .*?空=(\d+)/(\d+)', l)
    if m:
        # 归属到当前最大层号
        L = max(items) if items else 0
        close[L] = dict(fit=float(m.group(1)), held=float(m.group(2)),
                        route=float(m.group(3)), empty=int(m.group(4)))
    m = re.search(r'\[mem\] L(\d+) footprint=([\d.]+)GB', l)
    if m: mem[int(m.group(1))] = float(m.group(2))
    m = re.search(r'L(\d+) \[α扫\] α=([\d.]+) relh=([\d.]+)', l)
    if m: ascan[int(m.group(1))].append((float(m.group(2)), float(m.group(3))))
    m = re.search(r'L(\d+) \[路由\] 选 α=([\d.]+) relh ([\d.]+)→([\d.]+)', l)
    if m: alpha[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))

if want is not None:
    print(f"===== L{want:02d} 全候选明细 =====")
    print(f"{'类别':<6} {'算法':<44} {'体积':>9} {'val':>8} {'held':>8} {'提升':>7}  判定")
    for cat, algo, vol, v, h, up, vd in items.get(want, []):
        u = f"{up:+.1f}%" if up is not None else "   —  "
        print(f"{cat:<6} {algo[:44]:<44} {vol:>9} {v:8.4f} {h:8.4f} {u:>7}  {vd}")
    if want in ascan:
        print("\n[路由 α 扫]")
        for a, r in ascan[want]:
            print(f"  α={a:<4} relh={r:.4f}")
    if want in alpha:
        a, b4, af = alpha[want]
        print(f"  ★选 α={a}  relh {b4:.4f} → {af:.4f}")
    sys.exit()

print(" 层 | 基座held | 定稿held |  Δ%   | 落地机制                          | 新增侧车 | 路由% | 空/256 | 内存GB | α")
print("----+----------+----------+-------+-----------------------------------+----------+-------+--------+--------+-----")
for L in sorted(items):
    land = [(c, a, vol) for c, a, vol, v, h, up, vd in items[L] if '✓正向落地' in vd and c != '量化']
    names = ','.join(a.split('(')[0] for c, a, vol in land) or '无'
    # 新增侧车体积(粗略求和, 只算落地项)
    tot = 0.0
    for c, a, vol in land:
        m = re.match(r'([\d.]+)(B|KB|MiB)', vol)
        if m:
            x = float(m.group(1)); u = m.group(2)
            tot += x * (1 if u == 'B' else 1024 if u == 'KB' else 1048576)
    sv = f"{tot:.0f}B" if tot < 1024 else (f"{tot/1024:.1f}KB" if tot < 1048576 else f"{tot/1048576:.2f}MiB")
    c = close.get(L, {})
    b = base.get(L)
    fin = c.get('held')
    d = f"{(fin-b)/b*100:+.2f}" if (b and fin) else "  —  "
    a = alpha.get(L)
    astr = f"{a[0]}" if a else "—"
    print(f" {L:2d} | {b if b else 0:8.4f} | {fin if fin else 0:8.4f} | {d:>5} | {names[:33]:<33} | {sv:>8} |"
          f" {c.get('route',0):5.1f} | {c.get('empty',0):3d}/256 | {mem.get(L,0):6.2f} | {astr}")
print(f"\n已收官 {len(close)} 层 / 有候选记录 {len(items)} 层")
