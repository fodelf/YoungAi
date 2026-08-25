#!/usr/bin/env python3
"""rplan_solve_r29.py — R29 体积分配求解器(2026-08-01 用户令"体积分布不合理, 根据日志重生成")。

与 v4 的唯一实质差别: **权重换成实跑量**。
  v4: 难度 d[L] 取自 round1 held 增量 —— 那是"上一版配方下"的预测量。
  r29: d[L] 取自 R28v7 自己跑完 43 层的日志(r29_evidence.py 反推的 d_energy)
       —— 同一配方下的真实边际点。

为什么必须换(实测): R28v7 跑完的边际效率审计显示分配方向是**反的** ——
  L00-L04 拿 664-788 MiB(全模型最高配)而边际效率只有 0.015-0.037;
  L05-L18 拿 439-747 MiB 却是 9/14 层饿着(效率 0.048-0.159, 误差正是在这段累积:
  held 0.37→0.86); L19-L42 有 16 层 d_energy≤0(给了体积不产生收益)。
  齐平仅 35%。首跑输出"前 10-15 token 连贯后塌缩"与 L05-L18 欠配吻合。

底线的处理(与 v4 的关键分歧): v4 强制"逐层不退步于 round1 的加权精度", 这正是把体积
锁死在深层的机制之一。r29 改为**只对饿着/齐平层保留不退步底线**, 对实测 d_energy≤0 的
"撑着"层放开 —— 让贪心按真实边际把这部分体积挪走。浅层底线一律保留(v5 事故: L01 掉到
hot=0 后本层误差增量暴涨 10.5 倍, 说明"被喂饱时的低难度"不能拿来论证"它可以饿着")。

用法: rplan_solve_r29.py <hotcurve.json> <evidence.json> [预算GiB] [out.txt] [out.json]
"""
import json, sys

MOEI, D, NEXP = 2048, 4096, 256
HDR = 16 + NEXP * 3 * 8


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
HOT_BYTES = 2 * HW13 + HW2
COS_HOT = 0.940          # 第一轮实测热档均值 0.936-0.951

# (cold dim, nc, w2dim, w2nc, 实测 cold cos) — cos 取第一轮 VQ_GATE 同档均值。
# ★v24×256 已剔除(2026-08-01 实测): cos 0.468 < v32 的 0.470 且字节更多 ⇒ 被 v32 严格支配,
#   是劣档。留着它会在贪心里制造一个负收益台阶(v32→v24), 把所有层的冷档卡死在最稀档
#   升不上去 —— 第一次跑 v4 就是这么全线停在 v32 的。档梯必须单调: 越贵越准。
LADDER = [
    (8, 256, 16, 256, 0.740),
    (12, 256, 16, 256, 0.646),
    (16, 256, 24, 256, 0.573),
    (32, 256, 32, 512, 0.470),
]


def cold_bytes(t):
    dim, nc, w2d, w2n, _ = LADDER[t]
    return 2 * pay(MOEI, D, dim, nc) + pay(D, MOEI, w2d, w2n)


def layer_bytes(t, hot):
    return HDR + hot * HOT_BYTES + (NEXP - hot) * cold_bytes(t)


def main():
    hc = json.load(open(sys.argv[1]))
    # ★权重源 = R28v7 实跑日志(r29_evidence.py 反推), 不再是 round1 的预测 held★
    ev = json.load(open(sys.argv[2]))
    ev_d = {r['L']: r['d_energy'] for r in ev['layers']}
    ev_vol = {r['L']: r['vol_mib'] for r in ev['layers']}
    ev_eff = {r['L']: r['eff'] for r in ev['layers']}
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 19.746
    out_txt = sys.argv[4] if len(sys.argv) > 4 else None
    out_json = sys.argv[5] if len(sys.argv) > 5 else None
    NL = hc['NL']
    B = int(budget * (1 << 30))
    curve = {l['L']: l['cover_curve'] for l in hc['layers']}
    meta = {l['L']: l for l in hc['layers']}

    # ★难度权重 = 误差【能量】增量 held²[L] - held²[L-1](2026-08-01 修正)★
    # 原先用线性增量 held[L]-held[L-1] 是错的, 两个理由:
    #  ① 量纲不一致: L00 没有上游, 它的"增量"是【从零开始的全部误差】(0.2020), 而其他层是
    #     【在已有误差之上的增量】。两者被放进同一个加权和 ⇒ L00 权重虚高近两倍, 优化器
    #     给它 255 热专家 / 1730.8 MiB(8.7% 预算)。改平方后 L00 从第1名掉到第10名。
    #  ② relL2 是相对误差, 线性增量不可加; 误差能量(平方)才是同一把尺。
    # 探针实测站在平方口径这边: 每 GiB 换到的误差改善 L41 0.6219 > L05 0.1361(4.6 倍),
    # 而线性口径把 L05/L00 排在 L41 前面 —— 正好排反。平方口径下 L41 升到第1名。
    # 另: 第一轮累积曲线 L00 0.202 → L18 0.886(见顶)→ L42 0.837, 误差在深层回落 ⇒ 浅层
    # 误差会被后续层部分吸收, 深层误差直接进最终输出无人可修。线性口径默认"每层误差等权
    # 进入输出", 该假设不成立, 平方口径天然压低浅层。
    # d_energy 在 r29_evidence.py 里已按 held²[L]-held²[L-1] 算好(同平方口径), 直接用。
    d = {L: ev_d.get(L, 0.0) for L in range(NL)}
    floor = 0.004      # 自愈层(能量负增量)的地板: 仍参与计算, 不能给 0 资源
    w = {L: max(d[L], floor) for L in range(NL)}

    def prec(L, t, hot):
        c = cover(L, hot)
        return c * COS_HOT + (1 - c) * LADDER[t][4]

    # ★热专家底线(2026-08-01 实测事故修)★
    # 事故: v5 让 L01 拿到 hot=0(256 专家全走最稀冷档), 实测本层误差增量从 +0.0068 暴涨到
    # +0.0713(10.5 倍), L00 的改善被它一层吃光还倒贴。
    # 根因: 难度权重是【配方的函数】不是层的固有属性 —— d[L01]=0.0028 是在上一轮"它拿着
    # hot=31 高配"时测的, 拿这个"被喂饱时的低难度"去论证"它可以饿着", 一饿难度立刻失效。
    # 这是自我实现的循环: 低难度→少给→难度暴涨, 而权重还停在旧值。
    # 修法: 把地板从【权重】挪到【结果】—— 不管难度信号说什么, 每层必须拿到
    #   ① 至少 HOT_MIN 个热专家, 且 ② 热覆盖率不低于 COVER_MIN。
    # 依据: 覆盖曲线在起点最陡(L01 的 top8 就吃掉 12.6% 权重), 前几个热专家是全模型
    # 性价比最高的字节, 不该让任何层拿不到。集中层几个热就过线, 分散层自然要多给 ——
    # 阈值同时起到"按集中度分配底线"的作用。
    # ★底线 = 逐层不退步(2026-08-01 证据驱动改版)★
    # 为什么不用统一阈值: 跨 43 层实测相关性 P vs 增量 r=+0.4265(正相关, 反直觉)——
    # 因为第一轮档位按深度递减, 浅层拿密档(P高)而浅层增量本来就大, P 与增量同被"层深度"
    # 驱动 = 混杂。统一阈值的前提"P 绝对值跨层可比"被证伪。
    # 真正有效的证据是【同层对照】(混杂消掉), 两组方向一致:
    #   L00  P 0.8094→0.8493(+0.040)  增量 0.2020→0.1485(−0.0535)
    #   L01  P 0.8060→0.6359(−0.170)  增量 0.0068→0.0573(+0.0505)  ← v5/v6 事故
    # ⇒ P 是有效的【层内】质量指标, 底线必须逐层: 每层 P 不低于第一轮同层的 P。
    # 第一轮 43 层完整跑完、误差曲线已知(L42 累积 0.8369), 以它为"不退步"基线,
    # 优化器只能在此之上加码。L01 那次 P 从 0.806 掉到 0.636 会被这条底线直接拦住。
    prec_ref = {}
    pref_path = sys.argv[6] if len(sys.argv) > 6 else None
    if pref_path:
        for line in open(pref_path):
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) >= 2:
                prec_ref[int(p[0])] = float(p[1])

    def cover(L, hot_):
        return curve[L][hot_ - 1] if hot_ > 0 else 0.0

    def floor_cfg(L):
        """满足 P>=prec_ref[L] 的最省 (tier,hot); 无参考则回到最省配置。

        ★r29 改动★: 实测 d_energy<=0 的"撑着"层不套底线 —— R28v7 有 16 层给了体积却
        不产生误差下降(d<=0), v4 的"逐层不退步"把这些体积焊死在深层, 正是分配反向的机制。
        浅层(L<=20)一律保留底线: v5 事故证明"被喂饱时的低难度"不能拿来论证它可以饿着。"""
        if ev_d.get(L, 0.0) <= 0.0 and L > 20:
            return len(LADDER) - 1, 0
        need = prec_ref.get(L)
        if need is None:
            return len(LADDER) - 1, 0
        best = None
        for t in range(len(LADDER)):
            for h in range(0, NEXP + 1):
                if prec(L, t, h) >= need:
                    b = layer_bytes(t, h)
                    if best is None or b < best[0]:
                        best = (b, t, h)
                    break          # hot 递增 P 单调↑, 首个达标即该档最省
        return (best[1], best[2]) if best else (len(LADDER) - 1, 0)

    tier, hot = {}, {}
    for L in range(NL):
        tier[L], hot[L] = floor_cfg(L)
    used = sum(layer_bytes(tier[L], hot[L]) for L in range(NL))
    nref = len(prec_ref)
    print(f"[底线] 逐层不退步(参考 {nref}/{NL} 层) ⇒ 热数 {min(hot.values())}-{max(hot.values())}"
          f"(均值 {sum(hot.values())/NL:.1f}), 底线占用 {used/2**30:.3f} GiB / 预算 {budget:.3f}")
    if used > B:
        sys.exit(f"★底线配置 {used/2**30:.3f} GiB 已超预算 {budget:.3f} GiB — 需加预算")
    last_ratio = {L: 0.0 for L in range(NL)}   # KKT 审计: 每层最后一次成交的边际收益
    lam = 0.0

    # ★不设人为热数上限(2026-08-01 用户令"动态设计, 结果最优; 都平均效果反而最差"):
    # 之前 HOT_CAP=64 顶满 13 层 = 我拍的常数在压制最优解。热专家 6.8 MiB/个很贵,
    # 边际收益 = Δcover×(cos_hot-cos_cold) 随 k 递减(覆盖曲线凹), 预算本身就是天然刹车,
    # 不需要人为封顶。让每层自己吃到边际收益低于冷档升级为止。
    HOT_CAP = NEXP
    guard = 0
    while guard < 200000:
        guard += 1
        best = None
        for L in range(NL):
            # 候选1: 加一个热专家(该专家从冷变热)
            if hot[L] < HOT_CAP:
                cost = HOT_BYTES - cold_bytes(tier[L])
                if cost > 0:
                    gain = w[L] * (prec(L, tier[L], hot[L] + 1) - prec(L, tier[L], hot[L]))
                    if gain > 0:
                        r = gain / cost
                        if best is None or r > best[0]:
                            best = (r, cost, L, 'hot')
            # 候选2: 冷档升一级
            if tier[L] > 0:
                nt = tier[L] - 1
                cost = layer_bytes(nt, hot[L]) - layer_bytes(tier[L], hot[L])
                if cost > 0:
                    gain = w[L] * (prec(L, nt, hot[L]) - prec(L, tier[L], hot[L]))
                    if gain > 0:
                        r = gain / cost
                        if best is None or r > best[0]:
                            best = (r, cost, L, 'tier')
        if best is None or used + best[1] > B:
            lam = best[0] if best else 0.0     # 停机门槛 λ = 下一步本该买、但预算不够的边际收益
            break
        _, cost, L, kind = best
        if kind == 'hot':
            hot[L] += 1
        else:
            tier[L] -= 1
        used += cost
        last_ratio[L] = best[0]                # 该层最后一次成交的边际收益

    rows, lines, tot = [], [], 0
    print(" 层 | 实测d能量| 上轮MiB | 上轮eff | top32 | 冷档    | w2档    | 热 | 热覆盖 | 加权精度 |  层字节  | Δ体积  | 累计GiB")
    print("----+---------+-------+---------+---------+----+--------+----------+----------+--------+--------")
    for L in range(NL):
        dim, nc, w2d, w2n, cc = LADDER[tier[L]]
        b = layer_bytes(tier[L], hot[L])
        tot += b
        cv, P = cover(L, hot[L]), prec(L, tier[L], hot[L])
        bpw = b * 8 / (NEXP * 3 * MOEI * D)
        dv = b/2**20 - ev_vol.get(L, 0.0)
        print(f" {L:2d} | {d[L]:+8.5f} | {ev_vol.get(L,0):7.1f} | {ev_eff.get(L,0):7.4f} |"
              f" {meta[L]['top32']*100:4.1f}% | {dim:2d}x{nc:<4d} | {w2d:2d}x{w2n:<4d} |"
              f" {hot[L]:3d} | {cv*100:5.1f}% |  {P:.4f}  | {b/2**20:7.1f}M | {dv:+7.1f} | {tot/2**30:6.3f}")
        rows.append(dict(L=L, difficulty=round(d[L], 5), top32=meta[L]['top32'],
                         k80=meta[L]['k80'], nonzero_experts=meta[L]['nonzero_experts'],
                         cold_dim=dim, cold_nc=nc, w2_dim=w2d, w2_nc=w2n, hot=hot[L],
                         hot_cover=round(cv, 4), weighted_prec=round(P, 4),
                         bytes=b, MiB=round(b / 2**20, 1), bpw=round(bpw, 4),
                         cum_GiB=round(tot / 2**30, 3)))
        lines.append(f"L={L} dim={dim} nc={nc} hot={hot[L]} w2dim={w2d} w2nc={w2n}"
                     f"   # d={d[L]:+.4f} cover={cv:.3f} P={P:.4f} {bpw:.4f}bpw {b} B\n")
    print("----+---------+-------+---------+---------+----+--------+----------+----------+--------+--------")
    obj = sum(w[L] * prec(L, tier[L], hot[L]) for L in range(NL))
    print(f"\n★ 43 层载荷 = {tot/2**30:.3f} GiB (预算 {budget:.3f}) + backbone 8.202 = {tot/2**30+8.202:.3f} GiB")
    print(f"★ 目标函数 Σ d·P = {obj:.4f}   热专家总数 {sum(hot.values())}  均值 {sum(hot.values())/NL:.1f}")

    # ================= 最优性审计(KKT 边际收益均衡) =================
    # 分配最优 ⟺ 所有层"最后一块字节"的边际收益都压在同一条门槛 λ 上。
    # 某层 last_ratio 远高于 λ ⇒ 它还欠体积(该多给); 远低于 λ ⇒ 它超配了(该收回)。
    # 这是纯数据判据, 不依赖任何人为设定的上限或偏好。
    # ★KKT 只对【竞价成交过】的层成立(2026-08-01 修): 被 HOT_MIN/COVER_MIN 底线托起来、
    # 之后一次都没竞价成功的层, last_ratio=0 —— 那是约束强制持有, 不是"超配"。有约束的
    # 优化里, 这些层的边际收益低于 λ 本就是 KKT 的正常形态(约束乘子非零)。把它们混进
    # 均衡检验会误报"分配不均衡"。
    bid = [L for L in range(NL) if last_ratio[L] > 0]
    floored = [L for L in range(NL) if last_ratio[L] == 0]
    import statistics as st
    md = st.median([last_ratio[L] for L in bid]) if bid else 0.0
    print(f"\n【最优性审计】停机门槛 λ={lam:.6e}  竞价层末次边际中位数={md:.6e}")
    print(f"  竞价层 {len(bid)}/{NL}   底线托底层 {len(floored)}/{NL}"
          f"{'(L' + ',L'.join(f'{L:02d}' for L in floored[:8]) + ('…' if len(floored) > 8 else '') + ')' if floored else ''}")
    dev = sorted(((abs(last_ratio[L] - md) / md if md else 0, L) for L in bid), reverse=True)
    print("  竞价层里偏离中位数最大的 5 个:")
    for x, L in dev[:5]:
        tag = '欠配(边际收益仍高)' if last_ratio[L] > md else '超配(边际收益已低)'
        print(f"    L{L:02d} 末次边际={last_ratio[L]:.4e} 偏离 {x*100:5.1f}%  {tag}")
    within = sum(1 for x, _ in dev if x <= 0.5)
    print(f"  {within}/{len(bid)} 竞价层在中位数 ±50% 内 ⇒ "
          f"{'竞价部分已边际均衡(约束内最优)' if bid and within >= len(bid)*0.8 else '★竞价部分不均衡★'}")

    # ================= 风险审计(误差贡献) =================
    # 单层风险 = 它对最终误差的贡献 = 难度权重 × 精度缺口。低精度本身不是风险,
    # 低精度 × 高难度才是。按这个排序看谁才是真正的薄弱环节。
    risk = sorted(((w[L] * (1 - prec(L, tier[L], hot[L])), L) for L in range(NL)), reverse=True)
    tot_risk = sum(r for r, _ in risk) or 1.0
    print(f"\n【风险审计】误差贡献 = 难度 × (1-加权精度); 全模型合计 {tot_risk:.4f}")
    print("   排名 | 层  | 难度Δ   | 加权精度 | 误差贡献 | 占比")
    for i, (r, L) in enumerate(risk[:6], 1):
        print(f"    {i:2d}  | L{L:02d} | {d[L]:+.4f} |  {prec(L,tier[L],hot[L]):.4f}  |  {r:.5f} | {r/tot_risk*100:5.1f}%")
    print("   ... 最低 3 层:")
    for r, L in risk[-3:]:
        print(f"        L{L:02d} | {d[L]:+.4f} |  {prec(L,tier[L],hot[L]):.4f}  |  {r:.5f} | {r/tot_risk*100:5.1f}%")
    if out_txt:
        with open(out_txt, 'w') as f:
            f.write("# R28 计划表 v4(双信号: 难度定体积 / 激活集中度定冷热, 2026-08-01)\n")
            f.write(f"# 43 层合计 {tot} B = {tot/2**30:.3f} GiB; + backbone 8.202 = {tot/2**30+8.202:.3f} GiB\n")
            f.writelines(lines)
        print(f"\n已写 {out_txt}")
    if out_json:
        json.dump({'budget_GiB': budget, 'payload_GiB': round(tot / 2**30, 3),
                   'model_GiB': round(tot / 2**30 + 8.202, 3), 'objective': round(obj, 4),
                   'hot_total': sum(hot.values()), 'cos_hot': COS_HOT,
                   'ladder': [{'dim': a, 'nc': b_, 'w2dim': c, 'w2nc': dd, 'cold_cos': e}
                              for a, b_, c, dd, e in LADDER],
                   'layers': rows}, open(out_json, 'w'), ensure_ascii=False, indent=1)
        print(f"已写 {out_json}")


if __name__ == '__main__':
    main()
