#!/usr/bin/env python3
"""anchor_hotcurve.py — 从 FP 锚算【每层真实冷热结构】(2026-08-01 用户令"每层冷热专家动态")。

为什么必须用这个而不是"按难度给 hot":
  热专家走 4×512 高精档(约 6.8 MiB/专家), 冷专家走层档(约 1.4 MiB), 差 5 倍。
  hot 该给多少, 取决于【这一层的激活集中度】—— 激活集中的层, 少数专家就吃掉大部分
  路由权重, 多给热专家是浪费; 激活分散的层, 同样的 K 只能覆盖一半计算, 就该多给。
  这是每层客观存在的结构, 不是难度的函数。

锚文件布局(anchor_load, ds4quant_run.c:447):
  40B 头: magic 'DQA2' | S | HCM | DIM | NLAYERS | VOCAB | NACT | (pad) | idh(u64)
  然后 fin[NL][S][DIM]f32 | ridx[NL][S][NACT]i32 | rw[NL][S][NACT]f32 | H | logits

输出: 每层 覆盖 50/80/90/95% 路由权重所需专家数 + 前 K 名占比。
用法: anchor_hotcurve.py <anchor.bin> [目标覆盖率=0.80]
"""
import struct, sys, mmap

path = sys.argv[1]
TARGET = float(sys.argv[2]) if len(sys.argv) > 2 else 0.80

f = open(path, 'rb')
hd = struct.unpack('<8I', f.read(32))
idh = struct.unpack('<Q', f.read(8))[0]
magic, S, HCM, DIM, NL, VOCAB, NACT = hd[0], hd[1], hd[2], hd[3], hd[4], hd[5], hd[6]
assert magic == 0x32415144, f'锚 magic 不符: {magic:#x}'
print(f'锚: S={S} HCM={HCM} DIM={DIM} NL={NL} NACT={NACT}')

fin_b = NL * S * DIM * 4
ridx_b = NL * S * NACT * 4
ridx_off = 40 + fin_b
rw_off = ridx_off + ridx_b

mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


def cover_k(wsum, frac):
    """覆盖 frac 比例权重所需的专家数"""
    tot = sum(wsum)
    if tot <= 0:
        return 0
    acc, k = 0.0, 0
    for w in sorted(wsum, reverse=True):
        acc += w; k += 1
        if acc >= tot * frac:
            return k
    return len(wsum)


rows = []
print()
print(' 层 | K@50% | K@80% | K@90% | K@95% | top8占比 | top16占比 | top32占比 | 有效专家')
print('----+-------+-------+-------+-------+----------+-----------+-----------+---------')
for L in range(NL):
    wsum = [0.0] * 256
    base_i = ridx_off + L * S * NACT * 4
    base_w = rw_off + L * S * NACT * 4
    idx = struct.unpack_from(f'<{S*NACT}i', mm, base_i)
    wts = struct.unpack_from(f'<{S*NACT}f', mm, base_w)
    for e, w in zip(idx, wts):
        if 0 <= e < 256:
            wsum[e] += w
    tot = sum(wsum) or 1.0
    sw = sorted(wsum, reverse=True)
    k50, k80, k90, k95 = (cover_k(wsum, x) for x in (0.5, 0.8, 0.9, 0.95))
    t8 = sum(sw[:8]) / tot
    t16 = sum(sw[:16]) / tot
    t32 = sum(sw[:32]) / tot
    nz = sum(1 for w in wsum if w > 0)
    kt = cover_k(wsum, TARGET)
    rows.append((L, k50, k80, k90, k95, t8, t16, t32, nz, kt))
    print(f' {L:2d} | {k50:5d} | {k80:5d} | {k90:5d} | {k95:5d} |  {t8*100:5.1f}%  |   {t16*100:5.1f}%   |'
          f'   {t32*100:5.1f}%   |  {nz:3d}/256')

kts = [r[9] for r in rows]
print('----+-------+-------+-------+-------+----------+-----------+-----------+---------')
print(f'\n覆盖 {TARGET*100:.0f}% 权重所需热专家数: 最小 {min(kts)} 最大 {max(kts)} 均值 {sum(kts)/len(kts):.1f}')
print(f'各层差异 (max-min) = {max(kts)-min(kts)}  ⇒ ', end='')
print('层间冷热结构差异显著, hot 必须按本表逐层定' if max(kts) - min(kts) >= 6
      else '层间集中度接近, hot 可用统一值或按难度微调')
out = sys.argv[3] if len(sys.argv) > 3 else None
if out:
    # ★落 JSON 供计划表求解 + 人工审查(用户令: 体积与冷热分布必须可审)★
    # curve[L][k] = 前 k+1 名热专家累计覆盖的路由权重比例 —— hot 数的边际收益直接从这读。
    import json
    doc = {'S': S, 'NL': NL, 'NACT': NACT, 'layers': []}
    for L in range(NL):
        wsum = [0.0] * 256
        idx = struct.unpack_from(f'<{S*NACT}i', mm, ridx_off + L * S * NACT * 4)
        wts = struct.unpack_from(f'<{S*NACT}f', mm, rw_off + L * S * NACT * 4)
        for e, w in zip(idx, wts):
            if 0 <= e < 256:
                wsum[e] += w
        tot = sum(wsum) or 1.0
        sw = sorted(wsum, reverse=True)
        acc, curve = 0.0, []
        for w in sw:
            acc += w
            curve.append(round(acc / tot, 5))
        r = rows[L]
        doc['layers'].append({
            'L': L, 'k50': r[1], 'k80': r[2], 'k90': r[3], 'k95': r[4],
            'top8': round(r[5], 4), 'top16': round(r[6], 4), 'top32': round(r[7], 4),
            'nonzero_experts': r[8], 'cover_curve': curve,
        })
    with open(out, 'w') as g:
        json.dump(doc, g, ensure_ascii=False)
    print(f'\n已写 {out}(含每层 256 点累计覆盖曲线)')
