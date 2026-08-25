# zlever — 动态 z + 四损失 + 感知 侧车战役基建(2026-08-08)

设计(用户定案): 基础 = 标准 q2 平权量化 86G 本体(全 256 专家, w1/w3 vq4x512 + w2 vq4x256,
无 1bit 无冷热); 侧车 = 每专家动态 z(四损失闭式求解, 感知列权), 全模型 ≤10G。
rank/λ 都是活的(用户 08-08 定案, 对齐三段式第三份"每层 k 可调"):
- 活 rank: 每专家精确 SVD 出完整 rank-收益曲线(KGRID=8..96), 层预算(avg_rank×专家数
  总 rank)内按边际能量收益贪心分配 — 误差大的专家多吃 rank;
- 活 λ: 每专家在 {3,10,30} 用内部验证集(训练集内切 1029/258)自选, 选择端用隐式 W 的
  随机截断 SVD(不成形 4096², 快); held-out 25% token 只作终判, 不参与任何选择。

工具:
- solve.py  单层活rank/活λ求解+held-out 评估(数据源: HF MXFP4 原始 / dql_vq 侧车 / 锚 fin)
  数学与量化器 z_solve_dual 同源: 对偶 ridge G=XaXa^T+λ·tr/d → α → W=Xa^Tα → SVD;
  四损失: classify=per-dim 方差列权(解前乘 sqrt(colw), 解后反缩放) /
          smooth+fixed=固定种子 dither 增广行(x+δ→同目标, 行权 0.25) / align=数据项本体 /
          fixed(ridge λ)。
- sweep.sh  驱动: sweep.sh <层> <专家数> ["预算avg列表"], 结果落 reports/L%02d_alloc.txt
判据: held-out 误差能量挽回率(层级=能量加权, 兼报专家均值/最差)。
历史: reports/L00_sweep_r1.txt(钉死 rank 网格首轮)/L00_sweep_r2_lam.txt(λ 补扫) —
钉死版 λ 甜点=10, rank48 均值 23.1%; 活版以此为对照基线。禁 /tmp 散件。

## 2026-08-08 口径纠正(用户裁决: per-expert z 根本不对)
- 正典 z = 引擎 ds4_z/ds4_zchain 三段式产物③: **每层一份** z^L, routed += clip·U·diag(z)·Vᵀ·x,
  k_L 可调, 全秩 4096 时 43 层也只 2.9GB。solve.py 的 per-expert 对象是错误自造物,
  其数据(L00 均值 23%@10G, reports/L00_expertz_partial.txt 136 专家)只作专家级误差结构诊断, 不进方案对比。
- 正典测试件 = layerz.py(教师路由 ΔH + 四损失纪律 + rank 全曲线)。
- 先验(probe_layer_z_validate 门脸): 整层 ΔH 极低维, 训练集内 k=1 抓 49.3% / k=32 抓 70.4%;
  此前 held-out 判负是在无 λ 纪律下测的, 本轮重测。
