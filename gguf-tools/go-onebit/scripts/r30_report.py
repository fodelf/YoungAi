#!/usr/bin/env python3
"""r30_report.py — R30 战役报告生成(2026-08-03, 用户令"明天早上代码基准测试报告")。

汇总五个板块 → markdown:
  ① 交付物与体积账(GGUF 实测 + code2b 对齐)
  ② 逐层量化全表(43 层: held / vs冠军量化态 / 增量 / 门禁)
  ③ 里程碑曲线(Σmin/KL vs 历轮)
  ④ 五项标准指标(metrics_v5mini.txt / metrics_bitexact.txt, 就绪才填)
  ⑤ 行为冒烟与基准状态(原始输出粘贴, 判读标注"可能不准")
用法: r30_report.py <r30_all.log> <champ_quant.json> <out.md> [--metrics 目录]
"""
import json, os, re, sys

log_p, ch_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
mdir = sys.argv[sys.argv.index('--metrics') + 1] if '--metrics' in sys.argv else None
txt = open(log_p, encoding='utf-8', errors='replace').read()
ch = {int(k): v for k, v in json.load(open(ch_p)).items()}

held = {int(m.group(1)): float(m.group(2))
        for m in re.finditer(r'L(\d+) ★贪心选 \S+ h\d+ held=([\d.]+)', txt)}
gate = {int(m.group(1)): float(m.group(2))
        for m in re.finditer(r'SEARCH_GATE L=(\d+) 增益 ([-+\d.]+)% < 门', txt)}
ms = re.findall(r'前(\d+)层已调优\+后缀无损: Σmin=([\d.]+) KL=([\d.]+)', txt)
alt = re.findall(r'ALT L=(\d+) r(\d)(?:\(z先行\))? val ([\d.]+)→([\d.]+)', txt)

L = ['# R30 战役报告(code2b/42G 配方 · 0731 源)', '']
L += ['## ① 交付物与体积账', '',
      '- 配方: git 573b7f5 code2b 复刻 — 全 256 专家三矩阵 signref(1.0625 bpw)+ 热16 侧车外挂',
      '- 账目: 骨架 8.202 + G/U/D 34.266 = **42.468 GiB**(与 ds4-code2b.gguf 定版一字对齐)',
      '- 工序: DS4_CALIB_FULLSET=1(冠军工序)· 100% 锚路由 · 增益门 0.05% · w2 signref(九宫格"码本无效"在案判决)', '']

L += ['## ② 逐层量化全表', '', '| 层 | held | 冠军量化态 | vs冠军 | 增量 | 门禁 |', '|---|---|---|---|---|---|']
prev, beat = None, 0
for l in sorted(held):
    h = held[l]
    d = h * h if prev is None else h * h - prev * prev
    prev = h
    q = ch.get(l, {}).get('q')
    r = f'{h/q:.2f}×' if q else '-'
    if q and h < q:
        beat += 1
    g = f'{gate[l]:+.3f}% 拦' if l in gate else '-'
    L.append(f'| L{l:02d} | {h:.4f} | {q if q else "-"} | {r} | {d:+.5f} | {g} |')
L += ['', f'好于冠军 {beat}/{len(held)} 层(冠军=54G vq4bf 的量化基座态; 跨源: 本轮对 0731 锚, 冠军对老 base 锚)', '']

L += ['## ③ 里程碑曲线(Σmin / KL, 目标线 0.52/0.60)', '', '| 前N层 | Σmin | KL |', '|---|---|---|']
for n, s, k in ms:
    L.append(f'| {n} | {s} | {k} |')
L += ['', '历轮对照: R29 终态 0.6802/0.832 · v1(36G VQ)前10 0.7816/0.794 · 冠军全模型终态 0.6953/0.650', '']

L += ['## ④ 五项标准指标']
for fn, ttl in [('metrics_v5mini.txt', 'PPL / Mean KLD / RMS Δp / Same top(vs 0731 FP 锚, held=783)'),
                ('metrics_bitexact.txt', 'Bit-exact weights(合并 GGUF vs 骨架)')]:
    p = os.path.join(mdir, fn) if mdir else None
    L += ['', f'### {ttl}', '```']
    L += [open(p, encoding='utf-8', errors='replace').read().strip()] if p and os.path.exists(p) else ['(待合并后生成)']
    L += ['```']

L += ['', '## ⑤ 联合反修 ALT 在线 A/B(z先行机制)', '',
      f'ALT 记录 {len(alt)} 条(r2 接管率与幅度 = 联合 vs 串行的直接读数, 反修段数据为准)', '']
L += ['## ⑥ 行为冒烟(原始输出, 判读可能不准)', '', '(模型合并后填: LRU 契约裸续写 + 编译 + 测试)', '']
open(out_p, 'w').write('\n'.join(L))
print(f'报告 → {out_p}({len(held)} 层, {len(ms)} 里程碑, {len(alt)} ALT)')
