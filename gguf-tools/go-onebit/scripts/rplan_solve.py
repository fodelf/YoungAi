#!/usr/bin/env python3
"""rplan_solve.py — 按【真实布局公式】反解 43 层计划表, 把总体积钉死在预算上。

为什么要这个(2026-07-31 用户令"多大就是多大, 重新设计量化脚本"):
  旧流程的体积是"跑完再说"——计划表写 hot=19 而实际 hot≈40, 实测 31.28 GiB vs 目标 28,
  超 11.7% 且过程中无人可见。本工具用 vq_qc.h 的 vq_payload_bytes 同式反解, 输出的每层
  字节数就是量化器将要写出的字节数, 求和即模型体积, 事前可验。

体积模型(与 vq_slot_off/vq_total_bytes 逐字对应):
  层字节 = hdr + hot·(2·hw13 + hw2) + (256-hot)·(2·cw13 + cw2)
  hw13 = pay(2048,4096,4,512)   热 w1/w3 固定 4×512
  hw2  = pay(4096,2048,4,512)   热 w2   固定 4×512
  cw13 = pay(2048,4096,dim,nc)  冷 w1/w3 按层档
  cw2  = pay(4096,2048,w2d,w2n) 冷 w2   按层档

分配策略: 浅层(靠近输入)质量优先给高密度档 + 多热专家; 深层递减。
  依据 fable5: 深半 L24-42 可映射/低秩性强(退化容忍高), 浅半 L0-23 高维不可约。

用法: rplan_solve.py [预算GiB=19.80] [出文件]
"""
import sys

MOEI, D, NEXP = 2048, 4096, 256
HDR = 16 + NEXP * 3 * 8


def nbits(nc):
    b = 0
    while (1 << b) < nc:
        b += 1
    return max(b, 1)


def idxb(rows, cols, dim, nc):
    nidx = rows * cols // dim
    nb = nbits(nc)
    return nidx if nb == 8 else (nidx * nb + 7) // 8 + 1


def pay(rows, cols, dim, nc):
    return 16 + nc * dim * 2 + rows * 2 + idxb(rows, cols, dim, nc)


HW13 = pay(MOEI, D, 4, 512)
HW2 = pay(D, MOEI, 4, 512)


def layer_bytes(dim, nc, w2d, w2n, hot):
    cw13 = pay(MOEI, D, dim, nc)
    cw2 = pay(D, MOEI, w2d, w2n)
    return HDR + hot * (2 * HW13 + HW2) + (NEXP - hot) * (2 * cw13 + cw2)


# 候选档位: 由密到疏。(dim,nc) 越大 = 每索引覆盖越多权重 = 越省。
LADDER = [
    (8, 256, 16, 256),    # 冠军密度
    (8, 512, 16, 256),
    (12, 256, 16, 256),
    (12, 512, 24, 256),
    (16, 256, 24, 256),
    (16, 512, 32, 256),
    (24, 256, 32, 256),
    (24, 512, 32, 512),
    (32, 256, 32, 512),
    (32, 512, 48, 512),
]


def solve(budget_gib, nl=43):
    budget = int(budget_gib * (1 << 30))
    # 热专家数: 浅层 32 → 深层 8 线性递减(热档是 4×512 高密度, 是体积大头)
    hot = [max(8, int(32 - 24 * l / (nl - 1))) for l in range(nl)]
    # 档位索引: 浅层 0(最密) → 深层递增; 先给一个梯度, 再按预算整体平移
    tier = [min(len(LADDER) - 1, int(l * len(LADDER) / nl)) for l in range(nl)]

    def total(tier, hot):
        return sum(layer_bytes(*LADDER[tier[l]], hot[l]) for l in range(nl))

    # 整体收紧: 从最深层开始逐层降档, 直到进预算
    guard = 0
    while total(tier, hot) > budget and guard < 4000:
        guard += 1
        # 找当前"降一档省得最多"的层, 优先深层(depth 权重)
        best, bl = -1, -1
        for l in range(nl):
            if tier[l] >= len(LADDER) - 1:
                continue
            cur = layer_bytes(*LADDER[tier[l]], hot[l])
            nxt = layer_bytes(*LADDER[tier[l] + 1], hot[l])
            gain = (cur - nxt) * (1.0 + 2.0 * l / nl)   # 深层降档代价低 → 加权
            if gain > best:
                best, bl = gain, l
        if bl < 0:
            # 档位到底, 再削热专家
            dl = max(range(nl), key=lambda l: hot[l] * (1.0 + 2.0 * l / nl))
            if hot[dl] <= 4:
                break
            hot[dl] -= 1
        else:
            tier[bl] += 1

    # 反向松弛: 还有余量就把浅层升回更密的档(质量优先)
    guard = 0
    while guard < 4000:
        guard += 1
        best, bl = -1, -1
        for l in range(nl):
            if tier[l] <= 0:
                continue
            cand = list(tier)
            cand[l] -= 1
            if total(cand, hot) > budget:
                continue
            gain = 1.0 / (1.0 + l)        # 越浅越优先升档
            if gain > best:
                best, bl = gain, l
        if bl < 0:
            break
        tier[bl] -= 1
    return tier, hot


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 19.80
    out = sys.argv[2] if len(sys.argv) > 2 else None
    tier, hot = solve(budget)
    lines, tot = [], 0
    print(" 层 | cold dim×nc | w2 dim×nc | hot |   层字节   |  bpw   | 累计 GiB")
    print("----+-------------+-----------+-----+------------+--------+---------")
    for l in range(43):
        dim, nc, w2d, w2n = LADDER[tier[l]]
        b = layer_bytes(dim, nc, w2d, w2n, hot[l])
        tot += b
        bpw = b * 8 / (NEXP * 3 * MOEI * D)
        print(f" {l:2d} |   {dim:2d}×{nc:<4d}   |  {w2d:2d}×{w2n:<4d}  | {hot[l]:3d} |"
              f" {b/2**20:7.1f} MiB | {bpw:.4f} | {tot/2**30:7.3f}")
        lines.append(f"L={l} dim={dim} nc={nc} hot={hot[l]} w2dim={w2d} w2nc={w2n}"
                     f"   # {bpw:.4f}bpw {b} B\n")
    print("----+-------------+-----------+-----+------------+--------+---------")
    print(f"\n★ 43 层载荷合计 = {tot/2**30:.3f} GiB (预算 {budget:.2f})")
    print(f"★ + backbone 8.202 GiB = {tot/2**30+8.202:.3f} GiB")
    if out:
        with open(out, 'w') as f:
            f.write(f"# R28 计划表 v2(rplan_solve 反解, 真实布局公式) 预算 {budget:.2f} GiB 载荷\n")
            f.write(f"# 43 层合计 {tot} B = {tot/2**30:.3f} GiB; + backbone 8.202 = {tot/2**30+8.202:.3f} GiB\n")
            f.write("# hot 列由量化器强制消费(hot_from_anchor), 实际热数必须==本表\n")
            f.writelines(lines)
        print(f"\n已写 {out}")


if __name__ == '__main__':
    main()
