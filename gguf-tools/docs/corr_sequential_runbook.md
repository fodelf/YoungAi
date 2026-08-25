# corr 顺序拟合 runbook（2026-07-23 立项；corr/z 侧车路线复活的权威入口）

## ⚠ 0.5 先跑重建门再谈顺序拟合（2026-07-23 修正，最重要）

**启动顺序拟合前, 必须先确认 corr 的低秩拟合本身够格**。诊断工具 `corr_reconstruct_check.py`
实测 algo.gguf: **64-token 均值 cos=0.18 / 幅度比=0.18 → FAIL(门=0.85)**。
(前一轮"cos 0.93"是 token0 离群大值误判, 均值才是真相。)

含义: rank-32 只捕获 teacher-student 残差 R 的 ~18%, **R 大部分是高维噪声, 低秩抓不住** ——
与历史在案 **[[go_onebit_activation_space_exhausted]] NO-GO**(严格1-bit专家误差=高维噪声,
激活空间修正失败)完全吻合。弱拟合(0.18) × 23层复利 = 乱码, 两因素叠加。

**判决前置**: 顺序拟合能治"复利", 但治不了"低秩拟合本身弱"。启动前先用重建门验证:
- 若单层新鲜拟合 cos 仍 <0.85(如现状 0.18) → corr **激活空间受限是根本墙**, 顺序拟合白费,
  转向: 提 rank(64/128, 但历史 rank@95% 数据显示专家近满秩=低秩死路) 或 **放弃 corr 走别的杠杆**。
- 若单层 cos ≥0.85 但多层爆 → 纯复利, 顺序拟合是正解, 走下文 §1。
先跑 `corr_reconstruct_check.py <corr.gguf> L<n> x.npy r.npy`, PASS 才投顺序拟合。

## 0. 为什么需要它（结案根因）

corr(激活空间低秩修正)当前 bug = **逐层独立拟合的联合复利爆炸**。证据链(fable5 2026-07-23):
- numpy 重建单层数学正确(cos 0.93 / 幅度 0.95) —— 求解与格式都对;
- corr 单层(1/43)可辨识不爆 / corr 23 层缅甸文完全乱码 —— 复利实锤;
- 对照: 权重空间残差(Q1(W−Q1(W)))联合应用不爆(与激活无关)。

**根因**: 每层 corr 在"其他层未修正(学生态)"的捕获 X 上拟合; 23 层联合应用后, 上游修正使下游层输入分布漂移 → 独立拟合的前提被破坏 → 误差逐层复利 → 指数爆炸。

## 1. 修复原理: 顺序(贪心)拟合

逐层从浅到深, 每层在**已修正的上游态**上重新捕获输入再拟合:

```
corr_stack = {}                       # 已定版的 corr 层
for L in 深半层序 (如 20→42):
    mount(base + residual + corr_stack)   # 上游已修正态
    X_L = capture ffn_in @ L              # ★关键: 在已修正态下重捕获(输入已漂移)★
    O_BASE_L = capture ffn_out @ L        # 同上, 学生输出在修正态下
    O_REF_L  = HF 教师 FFN(x=X_L)         # 教师目标(注意 x 也是修正态输入)
    R_L = O_REF_L − O_BASE_L
    z_L = ds4_z_solve(X_L, R_L, rank)     # 在真实漂移分布上拟合
    corr_stack[L] = z_L                   # 定版, 进入下一层的上游态
```

每层 fit 时上游已是最终修正态 → 该层看到的 X 分布 = 部署时的真实分布 → 拟合有效 → 无复利。

## 2. 复用资产(已在库, 不重造)

- 捕获: `capture_alllayers.sh`(单层可用 DS4_CAP_LAYERS=L-L; drain 修复已在, off-by-one 已治)
- 教师: `dsml_oref.py`(error-feedback: 引擎 x̂+路由 + HF fp8 重算 FFN; M1 本机 shard)
- 求解: `zsolve`(闭式 ridge RRR; CLI: --out --rank --layer L x.npy r.npy)
- 挂载缩放: `DS4_CORR_SCALE=α`(已落码 ds4_corr.c; 顺序拟合应≈1.0, α 仅诊断)
- 单层侧车合并: 需新增 `corr_merge`(把逐层 corr_L.gguf 合成一个多层侧车) —— 或 zsolve 支持增量 --layer 追加已可(现成)

## 3. 判决门(每层 + 整体)

- 每层落地前: numpy 重建 cos(corr_L, R_L) ≥ 0.85(单层数学门, 复用本次诊断脚本);
- 每加 5 层: 12 针算法面板 A/B, 输出不得从可辨识退化成乱码(复利早期预警);
- 整体收官: 全 23 层 corr_stack + 残差 挂载 → algo 12针 + prog 15针 vs v2res 基线, 赢线 = 算法真体 10→11+/12 且 prog 不回退 + 行为门(192口径)不退化(corr 不许伤工具帧, v2 反修教训同族)。

## 4. 铁律(硬约束)

- **盘**: 逐层捕获+教师中间件走 /tmp, 单层 ~1GB, 每层用完即清(capture-OOM 铁律); emit 已流式化(峰值受控)。
- **内存**: 双机 12G 红线看门狗; capture 关 prefetch + 低预算。
- **体积**: corr 侧车 25MB 例外已批(用户 2026-07-23); 不动 base/原始 q2/HF。
- **交接**: 每层 corr 定版前不删上一版; 整体侧车过行为门才换现役。
- **顺序拟合的代价**: 23 层 × (capture+oref+solve) 串行 ≈ 数小时~1天(每层 mount 需重启栈)。二阶改进, 排在交付栈可用之后; 启动前先算盘+时间账(07-15 emit 教训)。

## 5. 最小验证先行(10分钟铁律)

全量 23 层跑之前, 先 3 层(L20-22)顺序拟合小样:
- L20 独立拟合(=现状) → L21 在 L20-修正态重捕获拟合 → L22 同理;
- 判据: 3 层顺序 corr 挂载不爆(vs 3 层独立 corr 应已开始退化) → 顺序拟合有效性首证;
- 通过才投全量。不通过 = 顺序拟合也救不了(可能 rank/lambda 或更深问题), 转阻尼 sweep 或放弃 corr。

## 6. 入口命令(待实现 corr_pipeline.sh seq 模式)

```
# 3层小样(先行判决)
LAYERS=20-22 SEQ=1 gguf-tools/go-onebit/scripts/corr_pipeline.sh seq
# 全量(小样过后)
LAYERS=20-42 SEQ=1 gguf-tools/go-onebit/scripts/corr_pipeline.sh seq
```
corr_pipeline.sh(seq 模式)= 本 runbook §1 循环的脚本化, 未实现; algo_pipeline.sh 是并行(非顺序)版, 可作骨架改造。
