# go2b 等体积重量化 runbook（2026-07-24；漂移方案的权威执行入口）

## 0. 为什么（已验证的前提）

现流程割裂: 量化(base-alone拟合z)→反修(base-alone)→残差独立emit叠加 → 从残差起漂移。
用户诊断: 除后训练外合并、在合并态拟合，消漂移。★已验证★:
- drift_output_check_all.py: 合并 go2b(激活最优) vs stacked(1+1bit) 全23层降 **26.4%** 专家输出误差(L38-42 尾段 +28~34%, 离logits近漂移代价最大)。
- 往返(实际bytes decode)确认 +26~29%; ★激活是唯一关键★(go2b None=历史恒等弱因, DS4_GO2B_ACT_SCALE用cap raw_ffn_in)。
- 引擎 go2b(type41) overlay 路径实测可用(6层加载+应用无崩)。

## 1. 等体积设计（体积账已确认）

- 现役: base go1b 全256(1单位) + 残差 go1b 热64(1) = 320单位 = 54.8G。
- go2b: 冷192 go1b(1) + 热64 go2b(2, 合并base+残差) = 320单位 ≈ 54.15G。★真等体积★。
- 热表 = prog_active_top64(路由点火, 43层)。冷专家保 go1b, 热专家 go2b(无独立残差)。

## 2. 集成点（已定位, 分钟级门贯穿）

### A. 量化器 emit (ds4quant_run.c)
- 现: `dq_quant_expert_signref` 每专家 go1b, 层文件 dql_L 存 1bit字节+修正记录(1007/1845行)。
- 改: 热专家(top-64)→ go2b encode(激活最优, 复用层内 calib 激活 X); 冷专家保 go1b。
  层文件加 go2b 块 + 热/冷标记。★门: 每层 emit 后 corr_reconstruct 同款单层 cos≥0.85(现状stacked已0.93+, go2b应更高)★。
- ⚠ 精密文件, 逐元素核对 signref→go2b 的字节布局(68B块, 与 go2b_encode/block_go2b 对齐)。

### B. 反修在 go2b 前向 (ds4quant_run.c BF_ONLY/replay)
- 现: replay dql(go1b) 回放推进(2060/2099行), z 系数在 go1b-alone 前向拟合。
- 改: replay 热专家 go2b dequant, z 在 go2b(合并态)前向拟合 → ★消 backfit-z 漂移★(这是用户方案核心)。

### C. rr_verdict 吃 go2b (rr_verdict.sh / ds4quant_run replay)
- 现: BF_ONLY 回放 go1b dql 打 smin vs FP锚。
- 改: 回放热go2b → ★分钟级端到端判决★(teacher-forced smin, 非生成)。这是每阶段的判据。

### D. 引擎稀疏 base 路由 (ds4.c / ds4_metal.m) —— 仅部署等体积需要
- 现: go2b 热/冷 split 已实现(g_moe_hot_*/g_hot_pick_slot/kernel_mul_mm_id_go2b), 但 base 存全256 mask热(+9.2G冗余)。
- 改: base pass 走稀疏冷192(cold_lut 256→192 或跳热) → 真等体积。量化器 emit 稀疏 cold base。
- ★可延后★: 验证阶段用 base全256+go2b overlay(rr_verdict只看质量不看体积), 质量过了再做稀疏 base 落等体积。

## 3. 分钟级门序（每阶段, 无一小时盲跑）

1. 每层 go2b emit → 单层重建 cos≥0.85 (corr_reconstruct_check 同款, 秒级)。
2. 全层 emit → rr_verdict smin vs 现役 0.3610 (teacher-forced, 分钟级)。★赢线: smin 显著 > 0.3610★。
3. 反修在go2b前向 → rr_verdict 复测 (应再升, 漂移消除)。
4. merge → 43针 + drift 复测。
5. 后训练 corr 插件不动(知识环 knowledge-primer 保留)。

## 4. 铁律
- 体积不增(用户 2026-07-24 硬令): 冷192+热64=320单位≤现役。稀疏 base 是等体积的必要件。
- 盘: go2b build 逐层 stream(pack 6.4G transient 即删); 全层overlay 17.5G(数据卷19-27G, 删旧模型腾)。
- 交接: 新模型 rr_verdict+43针过前不删现役 v3+prog残差。
- 原始 q2/hf 只读永不删。
- 精密 C 改(量化器 emit/replay)= 逐元素字节核对, 小样(3层)先跑通再全量。

## 5. 入口（待实现）
- `build_go2b_hot.py`(已落): 激活最优 go2b 热overlay(验证用)。
- `go2b_validate.sh`(已落): build|probe|baseline|compare(overlay 生成验证)。
- `corr_reconstruct_check.py`/`drift_output_check_all.py`(已落): 单层/全层重建门。
- 待实现: ds4quant_run.c go2b emit+replay(A/B/C) + 稀疏 base(D)。这是主工程, 需谨慎逐块+小样门。

## 6. 已否决的捷径（防重走）
- go2b Xh=None(历史恒等): 漏激活, 只比stacked略好。必须喂激活。
- corr/z 侧车: 激活空间高维噪声死路(cos 0.18), 与 go2b(权重空间合并)无关, 不复活。
- 生成质量当判据: 1h+foggy, 用 rr_verdict teacher-forced smin(分钟级确定性)替代。
