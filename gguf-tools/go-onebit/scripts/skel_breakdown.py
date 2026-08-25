#!/usr/bin/env python3
"""skel_breakdown.py — 拆解 backbone 骨架的体积构成(2026-08-01 用户问"一定需要 backbone 吗")。

骨架 = 合并时除路由专家之外的全部张量。它一直被当成不可动的 8.202 GiB 常数, 但它占
28 GiB 目标的 29.3% —— 先看清里面是什么、各占多少、量化类型是什么, 才能判断哪些能省。

用法: skel_breakdown.py <skeleton.gguf>
"""
import struct, sys, collections

TYPE_NAME = {0: 'f32', 1: 'f16', 8: 'q8_0', 12: 'q4_K', 10: 'q2_K', 19: 'iq2_xxs',
             40: 'go1b', 41: 'go2b', 42: 'vqblob'}
# 每类型每元素字节(块量化取块均值)
TYPE_BPE = {0: 4.0, 1: 2.0, 8: 34/32, 12: 144/256, 10: 84/256, 19: 66/256,
            40: 34/256, 41: 68/256}


def rd(f, n):
    b = f.read(n)
    assert len(b) == n
    return b


def parse(path):
    f = open(path, 'rb')
    magic = rd(f, 4)
    assert magic == b'GGUF', f'非 GGUF: {magic}'
    ver, = struct.unpack('<I', rd(f, 4))
    n_t, n_kv = struct.unpack('<QQ', rd(f, 16))

    def rstr():
        n, = struct.unpack('<Q', rd(f, 8))
        return rd(f, n)

    def skipv(t):
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in sz:
            f.seek(sz[t], 1); return
        if t == 8:
            rstr(); return
        if t == 9:
            et, n = struct.unpack('<IQ', rd(f, 12))
            for _ in range(n):
                skipv(et)

    for _ in range(n_kv):
        rstr()
        t, = struct.unpack('<I', rd(f, 4))
        skipv(t)

    tens = []
    for _ in range(n_t):
        nm = rstr().decode()
        nd, = struct.unpack('<I', rd(f, 4))
        ne = struct.unpack(f'<{nd}Q', rd(f, 8 * nd))
        ty, off = struct.unpack('<IQ', rd(f, 12))
        el = 1
        for x in ne:
            el *= x
        tens.append(dict(name=nm, ne=ne, type=ty, off=off, elements=el))
    return tens, ver


def classify(nm):
    if 'token_embd' in nm or 'embed' in nm:
        return '词表嵌入'
    if nm.startswith('output') or 'lm_head' in nm:
        return '输出头'
    if 'shexp' in nm:
        return '共享专家(每层3个)'
    if 'attn' in nm:
        return '注意力'
    if 'ffn_gate_inp' in nm or 'exp_probs_b' in nm:
        return '路由器+偏置'
    if 'hc_' in nm:
        return '超连接 hc'
    if 'norm' in nm:
        return 'norm'
    return '其他'


tens, ver = parse(sys.argv[1])
# 用相邻 offset 差推真实字节(最后一个用元素数×bpe 估)
tens.sort(key=lambda t: t['off'])
for i, t in enumerate(tens):
    if i + 1 < len(tens):
        t['bytes'] = tens[i + 1]['off'] - t['off']
    else:
        t['bytes'] = int(t['elements'] * TYPE_BPE.get(t['type'], 2.0))

g = collections.defaultdict(lambda: [0, 0, collections.Counter()])
for t in tens:
    k = classify(t['name'])
    g[k][0] += t['bytes']
    g[k][1] += 1
    g[k][2][TYPE_NAME.get(t['type'], f"t{t['type']}")] += 1

tot = sum(t['bytes'] for t in tens)
print(f"骨架张量 {len(tens)} 个  合计 {tot/2**30:.3f} GiB\n")
print("  分类               |   GiB   |  占比  | 张量数 | 量化类型")
print("---------------------+---------+--------+--------+----------------")
for k, (b, n, ty) in sorted(g.items(), key=lambda x: -x[1][0]):
    tys = ','.join(f'{a}×{c}' for a, c in ty.most_common(3))
    print(f"  {k:<18} | {b/2**30:7.3f} | {b/tot*100:5.1f}% | {n:6d} | {tys}")
print("---------------------+---------+--------+--------+----------------")
print(f"  {'合计':<18} | {tot/2**30:7.3f} | 100.0% | {len(tens):6d} |")

print("\n最大的 12 个张量:")
for t in sorted(tens, key=lambda x: -x['bytes'])[:12]:
    print(f"  {t['bytes']/2**20:8.1f} MiB  {TYPE_NAME.get(t['type'],t['type']):<8} "
          f"{'x'.join(map(str,t['ne'])):<22} {t['name']}")
