#!/usr/bin/env python3
"""route_concentration.py — 检验 Go 域路由是否"全覆盖", 判定 keyword-MTP + 驻留专家缓存能否提速。
输入: route_L{L}.npy [n,6] (top-6 专家/token), 层 L 范围。
输出: ①每层 support(用到几个专家/256) ②频率集中度(top-K mass) ③K-token 窗并集(=verify 驻留槽数→GB)。
用法: route_concentration.py DIR L0 L1 ...   (DIR 下有 route_L{L}.npy)
"""
import sys, os
import numpy as np

DIR = sys.argv[1]
LAYERS = [int(x) for x in sys.argv[2:]]
GO2B_MB = 2.2   # 每 go2b 专家(单 kind~2.2MB, 三 kind gate+up+down 合~6.6MB; 这里按整专家 6.6MB 更真实)
EXPERT_MB = 6.6

routes = {}
for L in LAYERS:
    p = os.path.join(DIR, f"route_L{L}.npy")
    if not os.path.exists(p): continue
    routes[L] = np.load(p).astype(np.int32)   # [n,6]
Ls = sorted(routes)
n = min(len(routes[L]) for L in Ls)
print(f"层 {Ls}  token={n}  (每 token 每层 6 专家)")

# ① 每层 support: 全语料用到多少专家
print("\n[①每层 support / 256]")
supports = []
for L in Ls:
    u = np.unique(routes[L][:n])
    supports.append(len(u))
    print(f"  L{L}: {len(u)}/256 ({100*len(u)/256:.0f}%)")
print(f"  → 均值 {np.mean(supports):.0f}/256  ({100*np.mean(supports)/256:.0f}%)  [全覆盖判据: 接近256=全覆盖]")

# ② 频率集中度: top-K 专家吃掉多少 token-activation (对照 x9 K=16=95.6%)
print("\n[②频率集中度: top-K 专家占全部 activation 的%]")
for L in Ls[:3] + Ls[-1:]:
    flat = routes[L][:n].flatten()
    cnt = np.bincount(flat, minlength=256)
    order = np.sort(cnt)[::-1]
    tot = order.sum()
    for K in (8, 16, 32, 64):
        print(f"  L{L} top-{K}: {100*order[:K].sum()/tot:.1f}%", end="  ")
    print()

# ③ K-token 窗并集: 一段 K 连续 token, verify 要取的 (layer,expert) 槽并集 → 驻留 GB
#    这才是 keyword-MTP 能否提速的要害: 窗并集小=可驻留=提速; 接近满=撞IO墙
print(f"\n[③ K-token 连续窗的专家槽并集 (跨 {len(Ls)} 层; 满槽={256*len(Ls)})]")
tot_slots = 256 * len(Ls)
# 预构造每 token 的槽集合: slot = L_index*256 + expert
slot_of = np.stack([routes[L][:n] for L in Ls], axis=1)  # [n, nL, 6]
slot_of = slot_of + (np.arange(len(Ls))[None,:,None] * 256)  # 全局槽 id
slot_of = slot_of.reshape(n, -1)  # [n, nL*6]
for K in (1, 4, 8, 16, 32, 64):
    if K > n: break
    us = []
    step = max(1, (n-K)//200)   # 采样 ~200 个窗
    for s in range(0, n-K+1, step):
        us.append(len(np.unique(slot_of[s:s+K])))
    avg = np.mean(us); mx = np.max(us)
    print(f"  K={K:3d}: 并集均值 {avg:6.0f} 槽 ({100*avg/tot_slots:4.1f}%满)  最大 {mx:5.0f}  "
          f"→ 驻留 ~{avg*EXPERT_MB/len(Ls):.1f}×{len(Ls)}层? 全模型43层外推 ~{avg*EXPERT_MB*43/len(Ls)/1000:.1f}GB")

# 整语料并集 (所有 token 一起用了多少槽) = 一次完整 Go 生成的专家足迹
allu = len(np.unique(slot_of))
print(f"\n[④全语料专家足迹] {allu}/{tot_slots} 槽 ({100*allu/tot_slots:.1f}%满)  "
      f"→ 43层外推 ~{allu*EXPERT_MB*43/len(Ls)/1000:.1f}GB (=想全驻留需的RAM)")
print("\n判据: ③窗并集%小 & ④足迹%小 → 可驻留→keyword-MTP能提速; 都接近满 → 全覆盖→撞IO墙(用户猜想成立)")
