# Task 07 — Stage 4：attention 重量化 Q8→Q5/Q4（攻地板，质量门把关）

**状态**：未开始
**前置依赖**：#03（地板已标定，确认 attention backbone 是地板）；可与 #06 并行（离线量化）
**类型**：离线 · 量化（`gguf-tools`）+ 质量门验证

## 目标

把 attention backbone 从 Q8 重量化到 Q5_K（必要时 Q4_K），降每 token 激活字节，攻「Q8 attention 地板」：
backbone ~8.85GB → ~6GB，每 token 激活 ~7GB → ~5.5GB。**铁律 correctness-before-speed：质量门不过即回滚。**

## 实施步骤

1. **离线重量化**（`gguf-tools`，流式不 mmap，内存安全）：把 attention 张量
   （`q_a` / `q_b` / `kv` / `output_a` / `output_b`）与输出头从 Q8 → Q5_K。
2. **逐张量灰度**：一次只降一组张量，每步记录 Δt/s 与质量,定位哪组张量伤质量。
3. 与专家并行模型（#06 的 k33 等效）合并产出重量化版 GGUF。

## 质量门（铁律，任一不过即回滚该张量到 Q8）

- `./ds4_test --logprob-vectors`（对官方 DeepSeek 向量，容差内）。
- `./ds4-eval -m <requant-model> --plain --questions 4 --tokens 2048 --temp 0 --seed 1`
  对照 README 期望 token 数。
- 顺带 `./ds4_test --metal-kernels` 确认 Q5_K kernel 数值正确。

## ≥20 t/s 物理推算（动手前）

- 用 #03 实测带宽：每 token 激活 7GB→5.5GB，TP 聚合 188GB/s → ~29ms ≈ 34 t/s 纯算 / ~26 实测。
- 若推算达不到 20，说明仅靠 Q5 不够，需叠 #08 MTP；记录在 execution-log。

## 内存安全

- 离线量化峰值 RSS 应 ≤ 数百 MB（参考 k4 生成 198MB）；过看门狗。
- 重量化降低常驻字节 → 反而放松 #06 的 L1 预算（可多塞 1-2 层/专家），重算预算闸。

## 验证命令

```sh
# 离线重量化（gguf-tools，逐张量灰度）
make -C gguf-tools
# (生成 requant GGUF)
./ds4_test --logprob-vectors
./ds4_test --metal-kernels
./ds4-eval -m gguf/<requant>.gguf --plain --questions 4 --tokens 2048 --temp 0 --seed 1
```

## 产出

- 通过质量门的 Q5（或部分 Q4）attention 模型 + 逐张量 Δt/s/质量 表。
- 更新后的常驻预算（放松量）+ 更新的 ≥20 t/s 推算 → 喂给 #08。
