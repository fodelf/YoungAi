#!/usr/bin/env python3
"""plan_json2rplan.py — 配置 JSON 即唯一真源, 派生量化器吃的计划表并校验(2026-08-01 用户令
"把量化脚本接入配置 json, 根据 json 配置进行量化")。

为什么不让量化器直接读 JSON: C 侧加 JSON 解析器是没必要的耦合。JSON 留给人审查/手改,
本工具做三件事 —— 派生 + 校验 + 对账:
  ① 派生: JSON.layers → "L=n dim= nc= hot= w2dim= w2nc=" 计划表(量化器既有格式, 零改动)
  ② 校验: 层号连续无缺、字段齐、档位合法、hot∈[0,256]、w2 档合法
  ③ 对账: 用 vq_qc.h 同一套布局公式【重算】每层字节, 与 JSON 里记录的 bytes 逐层比对。
     手改过 hot/档位却没更新 bytes 的 JSON 会在这里被抓住, 而不是跑完 5 小时才发现。
     重算值才是量化器真正会写出的字节数, 总量以重算为准。

用法: plan_json2rplan.py <plan.json> [out.rplan.txt] [--budget-gib N] [--force]
      --budget-gib 超预算即拒(不给则只警告)   --force 校验不通过也照写
"""
import json, sys

MOEI, D, NEXP = 2048, 4096, 256
HDR = 16 + NEXP * 3 * 8
VALID_DIM = {4, 8, 12, 16, 24, 32}
VALID_NC = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096}


def nbits(nc):
    b = 0
    while (1 << b) < nc:
        b += 1
    return max(b, 1)


def idxb(r, c, dim, nc):
    n = r * c // dim
    k = nbits(nc)
    return n if k == 8 else (n * k + 7) // 8 + 1


def pay(r, c, dim, nc):
    return 16 + nc * dim * 2 + r * 2 + idxb(r, c, dim, nc)


HW13, HW2 = pay(MOEI, D, 4, 512), pay(D, MOEI, 4, 512)


def layer_bytes(dim, nc, w2d, w2n, hot):
    """与 vq_qc.h vq_total_bytes 逐字对应 —— 量化器真正会写出的字节数。
    ★v3(2026-08-03 冠军复刻)★ w2d==0 = 冷 w2 走 signref D 段(落 dql, 不进 blob 账)。"""
    cold = (2 * pay(MOEI, D, dim, nc) if dim > 0 else 0) + (pay(D, MOEI, w2d, w2n) if w2d > 0 else 0)   # dim=0: 冷 w1w3 signref 在 dql, 不进 blob 账
    return HDR + hot * (2 * HW13 + HW2) + (NEXP - hot) * cold


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    force = '--force' in sys.argv
    budget = None
    if '--budget-gib' in sys.argv:
        budget = float(sys.argv[sys.argv.index('--budget-gib') + 1])
    src = args[0]
    out = args[1] if len(args) > 1 else src.rsplit('.', 1)[0] + '.rplan.txt'

    doc = json.load(open(src))
    layers = doc.get('layers')
    if not layers:
        sys.exit(f'★{src} 没有 layers 数组')
    layers = sorted(layers, key=lambda x: x['L'])

    errs, warns, lines, tot, mismatch = [], [], [], 0, 0
    seen = set()
    for l in layers:
        L = l['L']
        if L in seen:
            errs.append(f'L{L} 重复')
        seen.add(L)
        try:
            dim, nc = int(l['cold_dim']), int(l['cold_nc'])
            w2d, w2n = int(l['w2_dim']), int(l['w2_nc'])   # 0 = signref D 段(冠军配方)
            hot = int(l['hot'])
        except KeyError as e:
            errs.append(f'L{L} 缺字段 {e}')
            continue
        if dim != 0 and dim not in VALID_DIM:   # 0 = signref 非VQ分支(code2b, 2026-08-03)
            errs.append(f'L{L} cold_dim={dim} 非法(允许 {sorted(VALID_DIM)})')
        if w2d != 0 and w2d not in VALID_DIM:   # 0 = signref D 段(冠军配方, 2026-08-03)
            errs.append(f'L{L} w2_dim={w2d} 非法')
        if (nc != 0 and nc not in VALID_NC) or (w2n != 0 and w2n not in VALID_NC):
            errs.append(f'L{L} nc={nc}/w2_nc={w2n} 非法(允许 {sorted(VALID_NC)})')
        if not 0 <= hot <= NEXP:
            errs.append(f'L{L} hot={hot} 越界 [0,{NEXP}]')
        b = layer_bytes(dim, nc, w2d, w2n, hot)
        tot += b
        if 'bytes' in l and int(l['bytes']) != b:
            mismatch += 1
            warns.append(f'L{L} JSON 记录 {l["bytes"]} B ≠ 重算 {b} B '
                         f'(差 {b-int(l["bytes"]):+d}) — 以重算为准')
        lines.append(f'L={L} dim={dim} nc={nc} hot={hot} w2dim={w2d} w2nc={w2n}'
                     f'   # {b*8/(NEXP*3*MOEI*D):.4f}bpw {b} B\n')

    miss = [i for i in range(max(seen) + 1) if i not in seen] if seen else []
    if miss:
        errs.append(f'层号缺失: {miss}')

    gib = tot / (1 << 30)
    print(f'层数 {len(layers)}  载荷(重算) {gib:.3f} GiB  + backbone 8.202 = {gib+8.202:.3f} GiB')
    if mismatch:
        print(f'★{mismatch} 层的 JSON bytes 与重算不符(JSON 被手改过?)')
    for wmsg in warns[:10]:
        print('  警告:', wmsg)
    if len(warns) > 10:
        print(f'  ...另有 {len(warns)-10} 条')
    for e in errs:
        print('  ★错误:', e)
    if budget is not None:
        if gib > budget:
            errs.append(f'载荷 {gib:.3f} GiB 超预算 {budget:.3f} GiB')
            print(f'  ★错误: 载荷 {gib:.3f} > 预算 {budget:.3f} GiB')
        else:
            print(f'  预算校验 ✓ {gib:.3f} ≤ {budget:.3f} GiB(余 {budget-gib:.3f})')
    if errs and not force:
        sys.exit('★校验不通过, 未写出计划表(--force 可强制)')

    hdr = (f'# 由 {src} 派生(plan_json2rplan)—— JSON 是唯一真源, 本文件勿手改\n'
           f'# 层数 {len(layers)}; 载荷重算 {tot} B = {gib:.3f} GiB; + backbone 8.202 = {gib+8.202:.3f} GiB\n')
    with open(out, 'w') as f:
        f.write(hdr)
        f.writelines(lines)
    print(f'已写 {out}')


if __name__ == '__main__':
    main()
