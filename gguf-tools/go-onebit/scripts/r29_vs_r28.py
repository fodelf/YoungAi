#!/usr/bin/env python3
"""r29_vs_r28.py — R29 跑到哪层、每层质量比上轮好还是差(2026-08-01, 盯盘用)。

对比口径说明(要紧):
  · held 是同一套 v5mini 校准集(S=1716/held=783)上的累积相对误差, 两轮**同语料**,
    所以可以直接比绝对值 —— 这与"跨语料只能比增量"的情况不同。
  · 真正要看的是【本层增量】d = held²[L] − held²[L−1]: R29 的设计假设就是把体积从
    深层撑着的层挪到 L05–L18, 那么这一段的增量必须比上轮小, 否则重分配没兑现。
  · 浅层 L00–L02 是 hash 路由层, 锚路由对它们无影响, 两轮该基本持平(可当对照组)。

用法: r29_vs_r28.py [r29日志=/tmp/r29_quant.log] [evidence.json]
"""
import json, re, subprocess, sys

r29_log = sys.argv[1] if len(sys.argv) > 1 else '/tmp/r29_quant.log'
ev_path = sys.argv[2] if len(sys.argv) > 2 else 'gguf/go-onebit/r29/evidence.json'

# R29 日志可能在远端
if r29_log.startswith('m1:'):
    txt = subprocess.run(['ssh', '192.168.1.2', f'cat {r29_log[3:]}'],
                         capture_output=True, text=True, errors='replace').stdout
else:
    txt = open(r29_log, encoding='utf-8', errors='replace').read()

held29, vol29, gate_hits = {}, {}, []
for ln in txt.splitlines():
    m = re.match(r'SEARCH L=(\d+).*?held=([\d.]+)', ln)
    if m:
        held29[int(m.group(1))] = float(m.group(2))
    m = re.match(r'L(\d+) ★体积★ 本层 ([\d.]+) MiB', ln)
    if m:
        vol29[int(m.group(1))] = float(m.group(2))
    if ln.startswith('SEARCH_GATE'):
        gate_hits.append(ln.strip())

ev = json.load(open(ev_path))
old = {r['L']: r for r in ev['layers']}

done = sorted(vol29)
if not done:
    sys.exit('R29 还没有落盘层')

print(f"R29 进度 {len(done)}/43 层   累计 {sum(vol29.values())/1024:.3f} GiB")
print(f"增益门拦下 {len(gate_hits)} 层(上一轮无此门, 负增益全部落地)")
print()
print(f"{'层':>3} {'R29体积':>8} {'上轮':>8} {'Δ体积':>7} | {'R29 held':>9} {'上轮held':>9} "
      f"| {'R29增量':>9} {'上轮增量':>9} {'判定':>6}")
prev29 = None
better = worse = 0
for L in done:
    h29 = held29.get(L)
    o = old.get(L, {})
    h28, d28, v28 = o.get('held'), o.get('d_energy'), o.get('vol_mib', 0)
    if h29 is None or h28 is None:
        continue
    d29 = h29 * h29 if prev29 is None else h29 * h29 - prev29 * prev29
    prev29 = h29
    # 增量更小 = 本层引入的误差能量更少 = 好
    if d29 < d28 - 1e-6:
        tag, better = "✓更好", better + 1
    elif d29 > d28 + 1e-6:
        tag, worse = "✗更差", worse + 1
    else:
        tag = "持平"
    print(f"L{L:02d} {vol29[L]:8.1f} {v28:8.1f} {vol29[L]-v28:+7.1f} | {h29:9.4f} {h28:9.4f} "
          f"| {d29:+9.5f} {d28:+9.5f} {tag:>8}")

print()
print(f"★逐层增量对比: 更好 {better} / 更差 {worse} / 持平 {len(done)-better-worse}")
seg = [L for L in done if 5 <= L <= 18]
if seg:
    s29 = sum(held29[L] ** 2 - (held29[L - 1] ** 2 if L - 1 in held29 else 0) for L in seg)
    s28 = sum(old[L]['d_energy'] for L in seg if L in old)
    print(f"★L05–L18(本轮加体积的重点段, 已跑 {len(seg)} 层): "
          f"累计能量增量 R29 {s29:+.5f} vs 上轮 {s28:+.5f} "
          f"⇒ {'重分配已兑现 ✓' if s29 < s28 else '尚未兑现'}")
if gate_hits:
    print("\n增益门拦下的层:")
    for g in gate_hits[-8:]:
        print("  " + g)
