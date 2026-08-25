#!/usr/bin/env python3
"""r30_plan_champ.py — 冠军 vq4bf 配方固定计划(2026-08-03 用户令"看冠军 git 配置修复")。

git 573b7f5 在案判决(九宫格全测 + HumanEval 18/20 定版):
  热 64  全三矩阵 → vq4×512+GPTQ(2.25 bpw)
  冷 192 w1/w3   → vq8×256+GPTQ(1.00 bpw)
  冷 w2          → signref 保持(1.0625; ★码本无效★ — R29/R30 的 vq16 w2 违背此判决)
全 43 层同配置, 无竞价(冠军是固定表)。w2dim=0 触发量化器 dq_signref_export 原生分支,
D 段落 dql(GO1B_BLK 34B/256el, 1,114,112 B/专家), 合并带 --down-offsets 不带 --no-down。

体积账: blob(热3矩阵+冷w1w3) 860.3M/层×43=36.1G + down 11.42G + 骨架 8.2G ≈ 55G(vq4bf 54.09 同档)。
用法: r30_plan_champ.py <out.json>
"""
import json, sys

MOEI, D, NEXP, NL = 2048, 4096, 256, 43
HDR = 16 + NEXP * 3 * 8
SZ_D = D * (MOEI // 256) * 34          # 1,114,112 — dql D 段 signref 块格式(dql_down_offset 同款)


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


import os
HOT = int(os.environ.get('PLAN_HOT', 47))
CDIM = int(os.environ.get('PLAN_COLD_DIM', 0))
CNC = int(os.environ.get('PLAN_COLD_NC', 0))
W2D = int(os.environ.get('PLAN_W2_DIM', 0))   # w2 档位=实验参数(0=signref; 九宫格判决是当时实测非永久定理)
W2N = int(os.environ.get('PLAN_W2_NC', 0))   # ★55GB 落地终案(2026-08-05 用户令: 体积=落地十进制GB)★
# 热 16→64(top64 表满配): blob≈860M/层×43=36.1G(全内嵌) + down 11.42 + 骨架 8.2 ≈ 55.7G ≤ 60,
# 余 ~4.3G 安全垫(L42 大层+账实差)。冷=signref 1bit 同前(G/U/D 全 256 落 dql)。
# 口径铁律: 体积=单文件全部字节, 外挂/内嵌一律计入, 永不摘项。
HW13, HW2 = pay(MOEI, D, 4, 512), pay(D, MOEI, 4, 512)
hot_b = 2 * HW13 + HW2
cold_blob = 2 * pay(MOEI, D, CDIM, CNC) if CDIM > 0 else 0   # code2b: 冷 w1w3 signref 在 dql G/U 段
blob = HDR + HOT * hot_b + (NEXP - HOT) * cold_blob           # blob=热侧车(外挂, 不进 GGUF 体积账)
down = NEXP * SZ_D * 3                                        # G/U/D 三段全 256 signref → 全进 GGUF

rows = []
for L in range(NL):
    rows.append(dict(L=L, cold_dim=CDIM, cold_nc=CNC, w2_dim=W2D, w2_nc=W2N, hot=HOT,
                     bytes=blob, MiB=round(blob / 2**20, 1),
                     bpw=round((blob + down) * 8 / (NEXP * 3 * MOEI * D), 4),
                     cum_GiB=round(blob * (L + 1) / 2**30, 3)))
tot_blob, tot_down = blob * NL, down * NL
# ★全口径账(2026-08-04 用户铁律: 体积=单文件全部字节, 永不摘项)★
# 合并形态实测: blob 内嵌 = 热 vq + 冷 w1w3 1bit 副本(引擎侧车布局, 槽表实测 2×1.008MiB/冷专家);
# down_exps = w2 1bit(272M/层, = D段的 1/3); 骨架 8.202。
COLD_BLOB_1BIT = 0 if CDIM > 0 else (NEXP - HOT) * 2 * 1.008 * 2**20   # 冷 vq(CDIM>0)已在 blob, 无 1bit 副本
gguf_blob = blob + COLD_BLOB_1BIT                             # 内嵌 blob 实际字节/层
gguf_down = down / 3                                          # 仅 w2 进 down_exps
model_gib = (gguf_blob * NL + gguf_down * NL) / 2**30 + 8.202
doc = dict(budget_GiB=round(tot_blob / 2**30, 3), payload_GiB=round(tot_blob / 2**30, 3),
           down_GiB=round(tot_down / 2**30, 3),
           model_GiB=round(model_gib, 3), model_GB=round(model_gib*2**30/1e9, 2),   # 落地口径=十进制GB(2026-08-05 用户终裁)
           recipe=f'hot{HOT}@vq4x512 cold_w13@vq{CDIM}x{CNC} w2@vq{W2D}x{W2N}(0=signref)',
           ladder=[dict(dim=CDIM, nc=CNC, w2dim=0, w2nc=0, cold_cos=0.70)], layers=rows)
json.dump(doc, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print(f"全口径: blob内嵌 {gguf_blob*NL/2**30:.3f} GiB(热vq {tot_blob/2**30:.3f} + 冷1bit副本 {COLD_BLOB_1BIT*NL/2**30:.3f}) "
      f"+ down(w2) {gguf_down*NL/2**30:.3f} + 骨架 8.202 = {doc['model_GiB']} GiB")
print(f"每层: blob {gguf_blob/2**20:.1f}M (热{HOT}×{hot_b/2**20:.2f}M + 冷{NEXP-HOT}×{cold_blob/2**20:.2f}M) + w2 {gguf_down/2**20:.1f}M")
