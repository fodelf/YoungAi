#!/usr/bin/env python3
"""r29_evidence.py — 从 R28v7 实跑日志反推"体积分配到底合不合理"(2026-08-01 用户令
"关于28g模型体积不同层分布不合理, 根据日志重新修改脚本生成模型配置json")。

上一轮的分配是 rplan_solve_v4.py 用【预测量】做的: 难度取自 round1 held 增量、集中度取自
锚 hotcurve。本脚本改用【实跑量】——R28v7 自己的 43 层日志——做事后审计, 因为:

  · 难度是配方的函数: round1 的 d[L] 是"该层拿着 round1 配方"时测的, 换了配方就不成立
    (v5 让 L01 掉到 hot=0 后增量从 +0.0068 暴涨到 +0.0713, 10.5 倍, 就是踩这个)。
  · 只有实跑日志里的 (体积[L], 增量[L]) 才是同一配方下的真实边际点。

判据 = KKT 边际齐平: 分配最优 ⟺ 各层"每 MiB 买到的误差能量下降"齐平。
      显著高于均值的层 = 饿着(该加), 显著低于 = 撑着(该减)。

输出 evidence.json: 逐层 {体积, held, 能量增量 d, 边际效率 eff, 判定}
用法: r29_evidence.py <layers.txt> [out.json]
"""
import json, re, sys

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else None

vol, held = {}, {}
for ln in open(src, encoding='utf-8', errors='replace'):
    m = re.match(r'L(\d+) ★体积★ 本层 ([\d.]+) MiB', ln)
    if m:
        vol[int(m.group(1))] = float(m.group(2))
        continue
    # SEARCH L=n ... held=x —— 同层可能多行(菜单候选), 取最后一条 = 落地态
    m = re.match(r'SEARCH L=(\d+).*?held=([\d.]+)', ln)
    if m:
        held[int(m.group(1))] = float(m.group(2))

Ls = sorted(set(vol) & set(held))
if not Ls:
    sys.exit(f"没解析到层数据(vol={len(vol)} held={len(held)})")

rows = []
prev = None
for L in Ls:
    h = held[L]
    # 误差能量增量: held 是 L2 相对误差, 线性差对 L00 量纲不一致 → 用平方能量差
    d = h * h if prev is None else h * h - prev * prev
    prev = h
    rows.append({"L": L, "vol_mib": vol[L], "held": h, "d_energy": d})

# 边际效率: 每 MiB 压下多少能量。d<0(该层反而变好)记为 0 处理成"无需再加"
tot_v = sum(r["vol_mib"] for r in rows)
for r in rows:
    r["eff"] = (max(r["d_energy"], 0.0) / r["vol_mib"]) * 1000.0   # ×1000 便于读

effs = sorted(r["eff"] for r in rows)
med = effs[len(effs) // 2]
for r in rows:
    if r["eff"] > med * 2.0:
        r["verdict"] = "饿着(该加)"
    elif r["eff"] < med * 0.5:
        r["verdict"] = "撑着(该减)"
    else:
        r["verdict"] = "齐平"

print(f"总体积 {tot_v/1024:.3f} GiB / {len(rows)} 层   边际效率中位数 {med:.4f}")
print(f"{'层':>3} {'体积MiB':>9} {'held':>7} {'d能量':>10} {'边际eff':>9}  判定")
for r in rows:
    print(f"L{r['L']:02d} {r['vol_mib']:9.1f} {r['held']:7.4f} {r['d_energy']:+10.5f} "
          f"{r['eff']:9.4f}  {r['verdict']}")

hungry = [r["L"] for r in rows if r["verdict"].startswith("饿")]
full = [r["L"] for r in rows if r["verdict"].startswith("撑")]
print(f"\n饿着 {len(hungry)} 层: {hungry}")
print(f"撑着 {len(full)} 层: {full}")
print(f"齐平 {len(rows)-len(hungry)-len(full)} 层")
print(f"\n★分配是否合理★: 齐平占比 {(len(rows)-len(hungry)-len(full))/len(rows)*100:.0f}% "
      f"— 越高越接近最优; 饿/撑并存说明有可无损腾挪的体积")

if out:
    json.dump({"total_gib": tot_v / 1024, "median_eff": med, "layers": rows},
              open(out, 'w'), ensure_ascii=False, indent=1)
    print(f"\n已写 {out}")
