#!/usr/bin/env python3
"""rplan_solve_v3.py — 按【实测每层难度】分配体积(2026-08-01 用户令"复杂层体积大一点")。

v2 的错误: 用"深度"当难度代理, 单调递减分配 ⇒ 高难度带 L05-L08 被降档、L41(全模型
第二难)拿到最稀的 v32, 而 L22-L40 十四个零/负增量层却占着中档。完全反了。

v3 的难度信号: 第一轮量化日志的逐层累积 held 【增量】= 本层自己引入的误差。
  d[L] = held[L] - held[L-1]  (L00: d = held[0], 从 0 起)
  d<=0 的层 = 误差不增反降(后层自愈), 压得再狠也不更差 ⇒ 优先降档
  d 大的层 = 本层是误差主产地 ⇒ 给密档

分配算法: 全层从最密档起步 → 反复挑"降一档的边际质量代价 / 省下的字节"最小的层降,
直到进预算。代价用 d[L] 加权(难度越高, 降档代价越大)。

用法: rplan_solve_v3.py <held.txt> [预算GiB=19.746] [出文件]
      held.txt 每行 "L held"(从第一轮日志 SEARCH 行提取)
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
    n = rows * cols // dim
    k = nbits(nc)
    return n if k == 8 else (n * k + 7) // 8 + 1


def pay(rows, cols, dim, nc):
    return 16 + nc * dim * 2 + rows * 2 + idxb(rows, cols, dim, nc)


HW13 = pay(MOEI, D, 4, 512)
HW2 = pay(D, MOEI, 4, 512)

# 档梯: 由密到疏(cold dim,nc, w2dim,w2nc)。索引越大越省。
LADDER = [
    (8, 256, 16, 256),
    (12, 256, 16, 256),
    (16, 256, 24, 256),
    (24, 256, 32, 256),
    (32, 256, 32, 512),
]
# 热专家数梯度: 高难度层多热, 低难度层少热(热档 4×512 是体积大头, 也是质量主力)
HOT_HI, HOT_LO = 32, 8


def layer_bytes(tier, hot):
    dim, nc, w2d, w2n = LADDER[tier]
    return (HDR + hot * (2 * HW13 + HW2)
            + (NEXP - hot) * (2 * pay(MOEI, D, dim, nc) + pay(D, MOEI, w2d, w2n)))


def main():
    held = {}
    for line in open(sys.argv[1]):
        p = line.split()
        if len(p) >= 2:
            held[int(p[0])] = float(p[1])
    nl = max(held) + 1
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 19.746
    out = sys.argv[3] if len(sys.argv) > 3 else None
    B = int(budget * (1 << 30))

    # 难度 = held 增量, 截断到 >=0(负增量层难度记 0 = 最该降档)
    d = {}
    for L in range(nl):
        d[L] = held[L] - (held[L - 1] if L > 0 else 0.0)
    dpos = {L: max(0.0, v) for L, v in d.items()}
    dmax = max(dpos.values()) or 1.0

    # 热数按难度线性映射
    hot = {L: int(round(HOT_LO + (HOT_HI - HOT_LO) * (dpos[L] / dmax) ** 0.5)) for L in range(nl)}
    tier = {L: 0 for L in range(nl)}   # 全部从最密档起步

    def total():
        return sum(layer_bytes(tier[L], hot[L]) for L in range(nl))

    # 贪心降档: 每次挑 代价/收益 最小者。代价 ∝ 难度(难度 0 的层降档零代价)
    guard = 0
    while total() > B and guard < 20000:
        guard += 1
        best, bl = None, -1
        for L in range(nl):
            if tier[L] >= len(LADDER) - 1:
                continue
            save = layer_bytes(tier[L], hot[L]) - layer_bytes(tier[L] + 1, hot[L])
            if save <= 0:
                continue
            cost = (dpos[L] / dmax + 0.02) / save * 1e9   # +0.02 防零代价层被无限降
            if best is None or cost < best:
                best, bl = cost, L
        if bl < 0:
            # 档到底, 削热(同样按难度反序)
            cand = [L for L in range(nl) if hot[L] > 4]
            if not cand:
                break
            dl = min(cand, key=lambda L: dpos[L])
            hot[dl] -= 1
        else:
            tier[bl] += 1

    lines, tot = [], 0
    print(" 层 |  难度Δ  | 冷档    | w2档    | 热 |  层字节  |  bpw   | 累计GiB")
    print("----+---------+---------+---------+----+----------+--------+--------")
    for L in range(nl):
        dim, nc, w2d, w2n = LADDER[tier[L]]
        b = layer_bytes(tier[L], hot[L])
        tot += b
        mark = '★' if dpos[L] > 0.04 else (' ' if dpos[L] > 0 else '·')
        print(f" {L:2d}{mark}| {d[L]:+.4f} | {dim:2d}x{nc:<4d} | {w2d:2d}x{w2n:<4d} | {hot[L]:2d} |"
              f" {b/2**20:7.1f}M | {b*8/(NEXP*3*MOEI*D):.4f} | {tot/2**30:6.3f}")
        lines.append(f"L={L} dim={dim} nc={nc} hot={hot[L]} w2dim={w2d} w2nc={w2n}"
                     f"   # d={d[L]:+.4f} {b*8/(NEXP*3*MOEI*D):.4f}bpw {b} B\n")
    print("----+---------+---------+---------+----+----------+--------+--------")
    print(f"\n★ 43 层载荷 = {tot/2**30:.3f} GiB (预算 {budget:.3f}) + backbone 8.202 = {tot/2**30+8.202:.3f} GiB")
    print("★=高难度层(Δ>0.04)  ·=零/负增量层(压狠不更差)")
    if out:
        with open(out, 'w') as f:
            f.write(f"# R28 计划表 v3(按实测每层难度 held 增量分配, 2026-08-01 用户令'复杂层体积大一点')\n")
            f.write(f"# 43 层合计 {tot} B = {tot/2**30:.3f} GiB; + backbone 8.202 = {tot/2**30+8.202:.3f} GiB\n")
            f.write("# d = 本层 held 增量(第一轮量化实测); d<=0 = 后层自愈, 优先降档\n")
            f.writelines(lines)
        print(f"\n已写 {out}")


if __name__ == '__main__':
    main()
