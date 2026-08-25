#!/usr/bin/env python3
# skel_bias_probe.py — 判定 skeleton 的 exp_probs_b 里有没有烘焙过路由偏置。
#
# 判据(2026-07-31): L00-L02 是 hash 路由层(tid2eid), 路由不漂移 ⇒ 任何 Δb 反修都不会
# 动它们的 bias; L03-L42 是 gate 路由层, 烘焙只发生在这里。所以:
#   · 未烘焙: 全 43 层 bias 的数值特征同族(同一套原始权重)
#   · 已烘焙: L03+ 相对 L00-L02 多出一层零碎扰动, 且扰动集中在 rb 文件里 cnt>=8 的槽
# 给定 rb 文件时进一步实证: 检查 L03+ 上"rb 有值的槽"与"rb 无值的槽"的 bias 分布是否分离。
#
# 用法: skel_bias_probe.py <skeleton.gguf> [rb.bin ...]
import struct, sys

NEXP, RB_MINCNT = 256, 8


def load_rb(path, n_layer):
    with open(path, 'rb') as f:
        hd = struct.unpack('<4I', f.read(16))
        if hd[0] != 0x41494252:
            return None
        nl, ne = hd[1], hd[2]
        acc = struct.unpack(f'<{nl*ne}f', f.read(nl*ne*4))
        cnt = struct.unpack(f'<{nl*ne}I', f.read(nl*ne*4))
    return nl, ne, acc, cnt


def read_bias(path):
    f = open(path, 'rb')
    f.read(8)
    n_t, n_kv = struct.unpack('<QQ', f.read(16))

    def rs():
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n)

    def sv(t):
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in sz:
            f.seek(sz[t], 1); return
        if t == 8:
            rs(); return
        if t == 9:
            et, n = struct.unpack('<IQ', f.read(12))
            for _ in range(n):
                sv(et)

    align = 32
    for _ in range(n_kv):
        k = rs(); t = struct.unpack('<I', f.read(4))[0]
        if k == b'general.alignment':
            align = struct.unpack('<I', f.read(4))[0]
        else:
            sv(t)
    tgt = {}
    for _ in range(n_t):
        name = rs().decode()
        nd = struct.unpack('<I', f.read(4))[0]
        f.seek(8 * nd, 1)
        ty, off = struct.unpack('<IQ', f.read(12))
        if name.endswith('.exp_probs_b.bias'):
            tgt[int(name.split('.')[1])] = (off, ty)
    data_start = (f.tell() + align - 1) // align * align
    out = {}
    for L, (off, ty) in sorted(tgt.items()):
        assert ty == 0, f'L{L} bias 非 f32'
        f.seek(data_start + off)
        out[L] = list(struct.unpack(f'<{NEXP}f', f.read(NEXP * 4)))
    f.close()
    return out


def frac_bits(v):
    """量化原始值通常落在少数几个尾数上; 烘焙加了任意小数 → 唯一值数量暴涨。"""
    return round(v, 7)


bias = read_bias(sys.argv[1])
print(f'层数 {len(bias)}  每层 {NEXP} 槽')
print()
print('  层 | 唯一值数 | 非零数 |     min     |     max     |   均值')
print('-----+----------+--------+-------------+-------------+---------')
for L in sorted(bias):
    v = bias[L]
    u = len(set(frac_bits(x) for x in v))
    nz = sum(1 for x in v if x != 0.0)
    print(f'  {L:2d} | {u:8d} | {nz:6d} | {min(v):+11.6f} | {max(v):+11.6f} | {sum(v)/len(v):+8.5f}')

shallow = [len(set(frac_bits(x) for x in bias[L])) for L in (0, 1, 2) if L in bias]
deep = [len(set(frac_bits(x) for x in bias[L])) for L in bias if L >= 3]
if shallow and deep:
    print()
    print(f'浅层(L00-L02, hash 路由) 唯一值均值 {sum(shallow)/len(shallow):.1f}')
    print(f'深层(L03+,   gate 路由) 唯一值均值 {sum(deep)/len(deep):.1f}')

# rb 实证(决定性判据): bias 绝对值全为正且量级 8-28(原始权重量级), 所以"同号率"这种
# 判据恒真、毫无信息。真正能分辨的是**层内残差**: 烘焙做的是 bias[e] += α·acc[e], 它只
# 改层内相对高低、不改层均值量级。故去掉层均值后对 acc 做最小二乘 —— 已烘焙则斜率 k≈α
# 且相关 r 显著; 未烘焙则 k≈0、r≈0(量化 bias 与本份 Δb 无因果关系)。
def lstsq_slope(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0, 0.0
    return sxy / sxx, sxy / (sxx * syy) ** 0.5


for rbp in sys.argv[2:]:
    r = load_rb(rbp, len(bias))
    if not r:
        print(f'\n{rbp}: 非 RBIA 头, 跳过'); continue
    nl, ne, acc, cnt = r
    X, Y = [], []
    for L in bias:
        if L >= nl:
            continue
        mean_L = sum(bias[L]) / len(bias[L])
        for e in range(min(NEXP, ne)):
            i = L * ne + e
            if cnt[i] >= RB_MINCNT and acc[i] != 0.0:
                X.append(acc[i])
                Y.append(bias[L][e] - mean_L)      # 层内残差
    k, corr = lstsq_slope(X, Y)
    verdict = ('★已烘焙(斜率≈α)★' if abs(corr) > 0.30 else '未烘焙(与本份 Δb 无关)')
    print(f'\n{rbp.split("/")[-1]}: 武装槽 {len(X)}  斜率 k={k:+.4f}  相关 r={corr:+.4f}  → {verdict}')
