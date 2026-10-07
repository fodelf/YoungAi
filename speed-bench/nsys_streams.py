#!/usr/bin/env python3
"""nsys_streams.py — 看 nsys 的 cuda_gpu_trace CSV 里各流是否真的重叠(2026-09-30, 并发道判官)。

为什么: 合批解码把各路的注意力缓存段挂到各自的流上, 实测反而没变快(N=3 54.0 对串行 52.3 ms)。读码解释不了, 只能看时间线:
 若各道的核在时间上几乎不重叠 ⇒ 瓶颈是主机发射速率(核太小, 每发 ~5 µs 发射 + ~5 µs 执行), 多流没用, 要上 CUDA graph;
 若重叠了但总时长没变 ⇒ 核在抢 SM/带宽。
用法: nsys_streams.py <cuda_gpu_trace.csv> [窗口秒=0.5] [步数=0] [流列表, 逗号分隔]
  —— 只看轨迹最后这一段(解码步; 前面是预填的大核)。
输出: 每流 核数 / 忙时 / 平均核时长; 并集忙时; ≥2 流同时忙的时长(= 真正的并发); 空转(没有任何核在跑)。
第 3/4 个参数(2026-10-01, 合批走图的道): 给了步数与流列表, 就只对这几条流打逐核名表(发/步, 均 µs, ms/步) ——
走图重放时各道各占一条内部流(N=3 是三条 8400 核的流), 同一核在不同道上的均时长一比, 就知道三道是不是在抢 SM。
"""
import csv, sys

path = sys.argv[1]
win = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 0
only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
rows = []
with open(path, newline="") as f:
    rd = csv.reader(f)
    hdr = next(rd)
    ci = {h: i for i, h in enumerate(hdr)}
    st_i, du_i = ci["Start (ns)"], ci["Duration (ns)"]
    strm_i = ci.get("Strm", ci.get("Stream"))
    name_i = ci["Name"]
    for r in rd:
        try:
            s, d = int(r[st_i]), int(r[du_i])
        except ValueError:
            continue
        rows.append((s, s + d, r[strm_i], r[name_i]))
if not rows:
    print("空轨迹"); sys.exit(1)
t_end = max(e for _, e, _, _ in rows)
t0 = t_end - int(win * 1e9)
seg = [x for x in rows if x[0] >= t0]
per = {}
for s, e, strm, name in seg:
    p = per.setdefault(strm, [0, 0, 0.0])
    p[0] += 1; p[1] += e - s
# 扫描线: 同时在跑的流数
ev = []
for s, e, strm, _ in seg:
    ev.append((s, 1, strm)); ev.append((e, -1, strm))
ev.sort()
active = {}
last = t0; union = 0; multi = 0; idle = 0
for t, d, strm in ev:
    n_active = sum(1 for v in active.values() if v > 0)
    span = t - last
    if n_active == 0: idle += span
    else:
        union += span
        if n_active >= 2: multi += span
    active[strm] = active.get(strm, 0) + d
    last = t
print(f"窗口 {win:.2f}s: 核 {len(seg)} 个, 并集忙 {union/1e6:.1f} ms, ≥2 流同时忙 {multi/1e6:.1f} ms, 空转 {idle/1e6:.1f} ms")
for strm, (n, busy, _) in sorted(per.items(), key=lambda kv: -kv[1][1]):
    print(f"  流 {strm:>4}: 核 {n:6d}  忙 {busy/1e6:8.1f} ms  平均 {busy/n/1e3:6.1f} µs/核")
# 最短的那些核: 看看缓存段的核有多小
small = sorted(seg, key=lambda x: x[1] - x[0])[: max(1, len(seg) // 2)]
print(f"  最短一半的核平均 {sum(e - s for s, e, _, _ in small) / len(small) / 1e3:.1f} µs")
# 逐核名表(按流): 核名只取模板/括号之前的那一段
if steps and only:
    import re
    for strm in sorted(only, key=lambda x: int(x) if x.isdigit() else 0):
        tab = {}
        for s, e, st_, name in seg:
            if st_ != strm:
                continue
            nm = re.sub(r"^\s*void\s+", "", name.strip('"')); nm = re.split(r"[(<]", nm)[0] or "[memcpy/memset]"
            t = tab.setdefault(nm, [0, 0]); t[0] += 1; t[1] += e - s
        tot = sum(v[1] for v in tab.values())
        print(f"  -- 流 {strm}: {len(tab)} 种核, 合计 {tot / 1e6 / steps:.2f} ms/步")
        for nm, (n, b) in sorted(tab.items(), key=lambda kv: -kv[1][1])[:16]:
            print(f"     {n / steps:7.1f} 发/步 {b / n / 1e3:7.1f} µs/发 {b / 1e6 / steps:6.2f} ms/步  {nm[:56]}")
