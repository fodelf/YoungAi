#!/usr/bin/env python3
"""p4_kernel_ms.py — nsys sqlite → decode 每token逐kernel耗时表(2026-08-20)。
只统计 decode 段 kernel(实例数≥门限=生成token数), per-token=总时长/实例组数。
用法: p4_kernel_ms.py <profile.sqlite> <gen_tokens>
"""
import sqlite3, sys
db, ntok = sys.argv[1], int(sys.argv[2])
c = sqlite3.connect(db)
q = """SELECT s.value, COUNT(*), SUM(k.end-k.start)
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC"""
rows = c.execute(q).fetchall()
dec_total = 0.0
print(f"{'kernel':58s} {'inst':>6s} {'次/tok':>6s} {'µs/tok':>8s}")
for name, inst, tot in rows:
    per = inst / ntok
    if inst < ntok:           # prefill/一次性 kernel 略过
        continue
    us = tot / 1e3 / ntok
    dec_total += us
    nm = name.split('(')[0][:56]
    print(f"{nm:58s} {inst:6d} {per:6.1f} {us:8.1f}")
print(f"{'Σ decode kernel':58s} {'':6s} {'':6s} {dec_total:8.1f}  ({dec_total/1e3:.2f} ms/tok)")
