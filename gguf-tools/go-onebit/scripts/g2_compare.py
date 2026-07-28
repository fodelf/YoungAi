#!/usr/bin/env python3
"""g2_compare.py — G2 四侧逐层判决表(2026-07-25)。
从 /tmp/g2_<side>/all.out 提取: 每层 1bit 量化 m=[val held] · z.GL 后 m · INHERIT 下层漂移。
用法: g2_compare.py [side ...]  (默认 base blk blk_a05 blk_a10, 忽略缺席侧)
"""
import re, sys, os

ROOT = os.environ.get("G2AB_ROOT", "/Users/fodelf/ds4-main/gguf/go-onebit/g2ab")
sides = sys.argv[1:] or ["base", "blk", "blk_a05", "blk_a10"]
data = {}
for s in sides:
    p = f"{ROOT}/{s}/all.out"
    if not os.path.exists(p): continue
    txt = open(p, errors="ignore").read()
    d = {}
    for m in re.finditer(r"DQL2REC L=(\d+) 1bit\s+vol=\d+ paysz=\d+ m=\[([\d.]+) ([\d.]+)", txt):
        d[f"L{m.group(1)}_1bit_val"] = float(m.group(2)); d[f"L{m.group(1)}_1bit_held"] = float(m.group(3))
    for m in re.finditer(r"INHERIT L=(\d+) 上游累积漂移\(输入 vs FP\)=([\d.]+)", txt):
        d[f"INHERIT_L{m.group(1)}"] = float(m.group(2))
    # 每层最终(最后一条 z.GL/percept 记录的 m1) = 修正链后残余
    for L in range(6):
        recs = re.findall(rf"DQL2REC L={L} \S+\s+vol=\S+ paysz=\S+ m=\[([\d.]+) ([\d.]+)", txt)
        if recs: d[f"L{L}_final_val"] = float(recs[-1][0]); d[f"L{L}_final_held"] = float(recs[-1][1])
    data[s] = d

keys = sorted({k for d in data.values() for k in d})
w = max(len(k) for k in keys) if keys else 10
print(f"{'metric':<{w}} " + " ".join(f"{s:>10}" for s in data))
for k in keys:
    row = [data[s].get(k) for s in data]
    base = row[0] if row and row[0] else None
    cells = []
    for v in row:
        if v is None: cells.append(f"{'—':>10}")
        else:
            pct = f"({(v-base)/base*100:+.1f}%)" if base and v != base else ""
            cells.append(f"{v:>7.4f}{pct:<7}"[:14].rjust(10))
    print(f"{k:<{w}} " + " ".join(cells))
