#!/usr/bin/env python3
"""p4_window.py — decode 纯窗口分析(2026-08-20): 取运行后 60% 的 kernel 时间窗,
按 attention_decode_mixed 计 token 数, 输出 每token GPU忙碌 / 间隙 / 逐kernel真账。
用法: p4_window.py <profile.sqlite>
"""
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
rows = c.execute("""SELECT s.value, k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start""").fetchall()
t0, t1 = rows[0][1], rows[-1][2]
W0 = t0 + int((t1 - t0) * 0.6)
win = [(n, s, e) for n, s, e in rows if s >= W0]
ntok = sum(1 for n, _, _ in win if "attention_decode_mixed" in n) / 43.0
span = (win[-1][2] - win[0][1]) / 1e6
busy_intervals = sorted((s, e) for _, s, e in win)
merged, cur_s, cur_e = [], busy_intervals[0][0], busy_intervals[0][1]
for s, e in busy_intervals[1:]:
    if s <= cur_e: cur_e = max(cur_e, e)
    else: merged.append((cur_s, cur_e)); cur_s, cur_e = s, e
merged.append((cur_s, cur_e))
busy = sum(e - s for s, e in merged) / 1e6
print(f"窗口 {span:.1f}ms, tokens={ntok:.1f}, 每token: 墙 {span/ntok:.2f}ms = GPU忙 {busy/ntok:.2f} + 间隙 {(span-busy)/ntok:.2f}")
from collections import defaultdict
agg = defaultdict(float)
for n, s, e in win:
    agg[n.split("(")[0][:40]] += (e - s)
print("逐kernel真账(µs/token):")
for n, t in sorted(agg.items(), key=lambda x: -x[1])[:14]:
    print(f"  {n:42s} {t/1e3/ntok:8.1f}")
