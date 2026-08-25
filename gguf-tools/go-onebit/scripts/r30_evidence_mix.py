#!/usr/bin/env python3
"""r30_evidence_mix.py — v2 混合证据(2026-08-02 用户令"清除量化模型从头跑, 先确认配置")。

审计结论(r30 首跑 25 层): 证据加权有效(大配层 L08/L16/L18 增量砍到证据的 0.39-0.56×),
但 R29 证据在低配层系统性失配(L10/L20/L21/L23 实测 4.2-6.8×)—— 上轮深层的低增量/自愈
是"背上游债"条件下的表象, 本轮上游干净后失效。

v2 证据 = 分段取真:
  L00-L24  用 R30 首跑实测 d_energy(同配方同源, 最真)
  L25-L42  上轮读数不可信(同为"背债表象"), d = max(R29_d, DEEP_FLOOR)
           DEEP_FLOOR=0.015 ≈ R30 已见深层正增量(0.013-0.032)均值的 2/3 —— 保守但不失真

用法: r30_evidence_mix.py <r30日志> <evidence_r29run.json> <out.json>
"""
import json, re, sys

DEEP_FLOOR = 0.015

log_p, ev_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(log_p, encoding='utf-8', errors='replace').read()
held, vol = {}, {}
for ln in txt.splitlines():
    m = re.match(r'L(\d+) ★贪心选 v\d+x\d+ h\d+ held=([\d.]+)', ln)
    if m:
        held[int(m.group(1))] = float(m.group(2))
    m = re.match(r'L(\d+) ★体积★ 本层 ([\d.]+) MiB', ln)
    if m:
        vol[int(m.group(1))] = float(m.group(2))

ev = json.load(open(ev_p))
old = {r['L']: r for r in ev['layers']}
rows, prev, n_r30, n_floor = [], None, 0, 0
for L in range(43):
    if L in held and L in vol:
        h = held[L]
        d = h * h if prev is None else h * h - prev * prev
        prev = h
        rows.append({"L": L, "vol_mib": vol[L], "held": h, "d_energy": d, "src": "r30run1"})
        n_r30 += 1
    else:
        o = old[L]
        d = max(o['d_energy'], DEEP_FLOOR)
        if d != o['d_energy']:
            n_floor += 1
        rows.append({"L": L, "vol_mib": o['vol_mib'], "held": o['held'], "d_energy": d,
                     "src": f"r29+floor" if d != o['d_energy'] else "r29"})

tot = sum(r['vol_mib'] for r in rows)
for r in rows:
    r['eff'] = (max(r['d_energy'], 0.0) / r['vol_mib']) * 1000.0

json.dump({"total_gib": tot / 1024, "median_eff": sorted(r['eff'] for r in rows)[21],
           "deep_floor": DEEP_FLOOR, "layers": rows}, open(out_p, 'w'), ensure_ascii=False, indent=1)
print(f"混合证据: R30 实测 {n_r30} 层 + R29 抬地板 {n_floor} 层 + R29 原值 {43-n_r30-n_floor} 层 → {out_p}")
for r in rows:
    print(f"L{r['L']:02d} d={r['d_energy']:+.5f} [{r['src']}]")
