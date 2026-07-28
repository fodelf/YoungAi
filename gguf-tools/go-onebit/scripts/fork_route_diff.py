#!/usr/bin/env python3
# fork_route_diff.py — 行为校准 Step A 归因: fork 位逐层路由对比(teacher 原始 vs 学生量化)。
# 学生: DS4_CAP_DIR 采集 raw_route_L{L}.i16 [n×6] + raw_route_logits_L{L}.f16 [n×256];
#       fork 行 = 倒数第二行(67 prefill + decode0(fork 位) + decode1)。
# teacher: fork_teacher.py FORK_ROUTE_OUT npz, idx_L{L}[1]/raw_L{L}[1] = fork 位。
# 判决: top-6 专家集合交集逐层表 → 分歧集中 = corr 路由 δ 杠杆; 全一致 = 专家输出值漂移(z 杠杆)。
# 用法: fork_route_diff.py <teacher.npz> <cap_dir_coord(L0-19)> <cap_dir_worker(L20-42)>
import sys, os
import numpy as np

tz = np.load(sys.argv[1])
dirs = {**{L: sys.argv[2] for L in range(0, 20)}, **{L: sys.argv[3] for L in range(20, 43)}}

print(f"{'L':>3} {'交集':>4} {'teacher top6':<28} {'student top6':<28} {'raw相关':>7}")
mism = []
for L in range(43):
    d = dirs[L]
    pi = os.path.join(d, f"raw_route_L{L}")
    pl = os.path.join(d, f"raw_route_logits_L{L}")
    if not os.path.exists(pi):
        print(f"{L:>3} 缺学生文件"); continue
    si = np.fromfile(pi, dtype=np.int16).reshape(-1, 6)
    sl = np.fromfile(pl, dtype=np.float16).reshape(-1, 256).astype(np.float32)
    srow, slog = si[-2], sl[-2]                      # fork 位 = 倒数第二行
    t_idx = tz[f"idx_L{L}"][1]
    t_raw = tz[f"raw_L{L}"][1]
    inter = len(set(srow.tolist()) & set(t_idx.tolist()))
    corr = float(np.corrcoef(t_raw, slog)[0, 1])
    mark = "" if inter == 6 else "  ★分歧"
    if inter < 6:
        mism.append((L, inter))
    print(f"{L:>3} {inter:>3}/6 {str(sorted(t_idx.tolist())):<28} {str(sorted(srow.tolist())):<28} {corr:>7.4f}{mark}")

print(f"\n== 归因判决: 分歧层 {len(mism)}/43 → {mism if mism else '全层路由一致 → 漂移在专家输出值(z/系数杠杆), 非路由(δ杠杆)'} ==")
