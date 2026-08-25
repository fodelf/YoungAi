#!/usr/bin/env python3
"""r30_layer_table.py — R30 逐层量化详表 + 自动异常检测(2026-08-02 用户令
"每一层的量化数据详细输出, 过程盯着可以提前发现问题")。

三方对账: 计划表(r30_rplan.json) × 实跑日志(/tmp/r30_all.log) × 上轮实跑(evidence_r29run.json)。
自动标出的问题(每条都是历史上真踩过的坑):
  ★路由    rrM≠1.0000 — 100% 锚路由破功(R28v2 败因之一)
  ★冷档    cold_cos 偏离档位设计值 ±0.05 — 量化器把该档做坏了(设计: v8x256=0.740, v12x256=0.646)
  ★热档    hot_cos < 0.90 — 热档质量塌(全程应 0.94+)
  ⚠体积    实跑 vs 计划偏差 >8% — 计划失配(贪心微调 hot 的小偏差正常)
  ⚠退步    本层误差能量增量 > 上轮同层 — 加了体积还更差
  ★突跳    增量 > max(前3层均值×3, 0.05) — v5 事故形态(L01 hot=0 增量暴涨 10.5 倍)

用法: r30_layer_table.py <r30日志> <r30_rplan.json> <evidence.json> [冠军ALL.md]
  第4参给冠军层表(layer-tables/ALL.md)时加两列: 冠军held / vs冠军, 并加
  ⚠冠军差距 判据(R30 held > 冠军同层 ×1.15 — 同档体积 32.4 vs 33.8 GiB, 直接可比;
  注意冠军对老 base 锚打分, 跨源读数留 15% 容差)。
"""
import json, re, sys

COS_DESIGN = {'v8x256': 0.740, 'v12x256': 0.646}
HOT_MIN, COLD_TOL, VOL_TOL = 0.90, 0.05, 0.08
CHAMP_TOL = 1.15

log_p, plan_p, ev_p = sys.argv[1], sys.argv[2], sys.argv[3]
champ = {}
if len(sys.argv) > 4:
    if sys.argv[4].endswith('.json'):
        # ★口径纪律(2026-08-02 用户裁决"量化跟量化比, 反修跟反修比, 不能交叉")★
        # champ_quant.json: {L: {q: 量化基座态 held, f: 侧车落地终态 held}}。
        # 本表是量化阶段盯盘 ⇒ 用 q 口径; 反修完成后另出 f 口径对比表。
        champ = {int(k): v['q'] for k, v in json.load(open(sys.argv[4])).items()}
    else:
        for ln in open(sys.argv[4], encoding='utf-8', errors='replace'):
            m = re.search(r'ELEMENTS L=(\d+) base=[\d.]+.*?终 val=([\d.]+) held=([\d.]+)', ln)
            if m:
                champ[int(m.group(1))] = float(m.group(3))
txt = open(log_p, encoding='utf-8', errors='replace').read()

pick, vol, gate, cos = {}, {}, {}, {}
for ln in txt.splitlines():
    m = re.match(r'L(\d+) ★贪心选 (v\d+x\d+) h(\d+) held=([\d.]+) w1w3bpw=([\d.]+) rrM=([\d.]+)', ln)
    if m:
        pick[int(m.group(1))] = dict(tier=m.group(2), hot=int(m.group(3)), held=float(m.group(4)),
                                     bpw=float(m.group(5)), rrM=float(m.group(6)))
    m = re.match(r'L(\d+) ★体积★ 本层 ([\d.]+) MiB \| 累计 ([\d.]+) GiB', ln)
    if m:
        vol[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    m = re.match(r'VQ_GATE L=(\d+) 热矩阵=(\d+) 冷w1w3=(\d+) 均值cos: hot=([\d.]+) cold=([\d.]+)', ln)
    if m:
        cos[int(m.group(1))] = dict(nh=int(m.group(2)), nc=int(m.group(3)),
                                    hot=float(m.group(4)), cold=float(m.group(5)))
    m = re.match(r'SEARCH_GATE L=(\d+) 增益 ([-\d.]+)% < 门', ln)
    if m:
        gate[int(m.group(1))] = float(m.group(2))

plan = {l['L']: l for l in json.load(open(plan_p))['layers']}
old = {r['L']: r for r in json.load(open(ev_p))['layers']}

done = sorted(set(pick) & set(vol))
if not done:
    sys.exit('还没有完成层')

hdr_ch = f"|{'冠军held':>8}|{'vs冠军':>7}" if champ else ""
print(f"{'层':>3}|{'档位':>8}|{'hot计/实':>9}|{'MiB计/实':>15}|{'held':>7}|{'增量':>9}|{'上轮增量':>9}"
      f"|{'上轮held':>8}{hdr_ch}|{'hot_cos':>7}|{'cold_cos':>8}|{'rrM':>6}|{'门':>7}| 异常")
prev_h, recent_d = None, []
n_warn = 0
for L in done:
    p, (v, cum), c = pick[L], vol[L], cos.get(L, {})
    pl, o = plan.get(L, {}), old.get(L, {})
    d = p['held'] ** 2 if prev_h is None else p['held'] ** 2 - prev_h ** 2
    prev_h = p['held']
    d28 = o.get('d_energy')
    issues = []
    if abs(p['rrM'] - 1.0) > 1e-6:
        issues.append('★路由')
    dz = COS_DESIGN.get(p['tier'])
    if dz and c and abs(c['cold'] - dz) > COLD_TOL:
        issues.append(f"★冷档({c['cold']:.3f}≠{dz})")
    if c and c['hot'] < HOT_MIN:
        issues.append('★热档')
    if pl and abs(v - pl['MiB']) / pl['MiB'] > VOL_TOL:
        issues.append(f"⚠体积({v:.0f}≠计划{pl['MiB']:.0f})")
    if d28 is not None and d > d28 + 1e-6:
        issues.append('⚠退步')
    base = sum(recent_d[-3:]) / len(recent_d[-3:]) if recent_d else None
    if base is not None and d > max(base * 3, 0.05):
        issues.append('★突跳')
    ch = champ.get(L)
    if ch and p['held'] > ch * CHAMP_TOL:
        issues.append(f"⚠冠军差距({p['held']/ch:.2f}×)")
    recent_d.append(d)
    n_warn += bool(issues)
    oh = o.get('held')
    oh_s = f'{oh:8.4f}' if isinstance(oh, float) else '       -'
    d28_s = f'{d28:+9.5f}' if d28 is not None else '        -'
    gate_s = f'{gate[L]:+6.2f}%' if L in gate else '      -'
    ch_s = (f"|{ch:8.4f}|{p['held']/ch:6.2f}×" if ch else "|       -|      -") if champ else ""
    print(f"L{L:02d}|{p['tier']:>8}|{pl.get('hot','-'):>4}/{p['hot']:<4}|"
          f"{pl.get('MiB',0):>7.1f}/{v:<7.1f}|{p['held']:7.4f}|{d:+9.5f}|{d28_s}"
          f"|{oh_s}{ch_s}|{c.get('hot',0):7.4f}|{c.get('cold',0):8.4f}|{p['rrM']:6.4f}|"
          f"{gate_s}| {' '.join(issues) if issues else '✓'}")

lastL = done[-1]
budget = sum(plan[L]['MiB'] for L in plan) / 1024
better = sum(1 for L in done if L in old and old[L].get('d_energy') is not None
             and (pick[L]['held']**2 - (pick[L-1]['held']**2 if L-1 in pick else 0)) < old[L]['d_energy'] - 1e-6)
ch_beat = sum(1 for L in done if champ.get(L) and pick[L]['held'] < champ[L])
ch_n = sum(1 for L in done if champ.get(L))
print(f"\n进度 {len(done)}/43   累计 {vol[lastL][1]:.2f}/{budget:.2f} GiB   增益门拦 {len(gate)} 层   "
      f"异常层 {n_warn}/{len(done)}   增量好于证据 {better}/{len([L for L in done if L in old])}"
      + (f"   ★好于冠军 {ch_beat}/{ch_n} 层★" if ch_n else ""))
ms = re.findall(r'前(\d+)层已调优\+后缀无损: Σmin=([\d.]+) KL=([\d.]+)', txt)
if ms:
    print('里程碑: ' + '  '.join(f"前{a}层 Σmin={b} KL={c}" for a, b, c in ms))
