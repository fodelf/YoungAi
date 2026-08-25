# Task 03 — Stage 1b：单机标定真实地板 + MTP 收益 + 200K 内存外推

**状态**：✅ 测量完成（decode/prefill/KV 三表 + 200K 可行性 + 20t/s 推算齐；MTP 子项推迟到 #01 修好）。详见 `notes/execution-log.md` → Task 03 条目。
**前置依赖**：#01（k4 输出正确）、#02（DS4_PROFILE + 看门狗）

## 实测结论（2026-05-31，k4 单机，DS4_MEM_BUDGET_MB=13000 看门狗护栏，三跑全 exit=0 无 OOM）

- **decode 地板**：gen t/s 平坦 9.42~9.56 跨 ctx 2K/8K/32K → backbone 带宽限制（非 KV）。有效带宽 ~90 GB/s（M4 ~120 的 75%）。单机 20 t/s **物理不可能**（绝对 ceiling ~13.6 t/s）。
- **prefill**：71.85 → 108.46 → 104.34 t/s（chunk 2048→4096 后平台）。
- **KV 闭式**（ratio 分布 43 层=0×2/4×21/128×20）：ctx 2048=0.184 / 8192=0.420 / 32768=0.608 / **200000=1.889** / 1000000=8.016 GiB。raw 项 ctx≥8192 恒定 0.357 GiB，增长全在 ratio-4 indexer comp 项。证实「MLA KV ~1.x GB @200K」。
- **TP 下 KV**：layer-slice 按层切分不复制，两机各 ~0.95 GiB @200K。
- **MTP**：暂缺 — #01 k4 输出仍垃圾，argmax 接受率无意义，**推迟到 #01 修好**（机制在位，草稿 3.54GB）。
- **≥20 t/s 推算**：20 t/s 需 189 GB/s 持续；TP 聚合 188 GB/s → ceiling 19.87 t/s（零开销）。⇒ **#07 Q5 缩 backbone 是过 20 的硬前置（非可选）；#05/#06 TP 闸在 E0(#4) RTT≤50µs；MTP 提供乘子**。
- **200K 内存**：单机 KV 1.89+模型 9.33+scratch ≈ 13.5 GiB 贴 16GB 红线（违反平稳铁律）→ 走 TP 两机分担更稳。
- **顺带暴露 Task 02 缺陷**：DS4_PROFILE CSV 全 0（钩子没接进实际 GPU graph 路径）；phys_footprint=0.181 GiB 不含 mmap 驻留 → 真正护内存的是 L1 静态闸，footprint 看门狗对 mmap 失效，待修。
**类型**：单机 · 测量（数据驱动决策，不写新推理代码）

## 目标

用 #02 的计时把单机真实物理上限标定清楚，得到：理论上限地板、MTP 真实收益、200K 内存可行性。
这是 TP（#05/#06）与重量化（#07）值不值得做的数据依据。

## 实施步骤（全部为测量，逐档记录到 execution-log）

1. **decode 地板**：`--ctx` 2K→8K→32K 各测 per-token decode 时长 → 反推激活字节 & 有效带宽。
   - 验证 plan 论点：~7GB/token（k4 backbone 主导）、单机 ~16 t/s 带宽地板、k4 实测 ~10.6 t/s。
2. **prefill t/s**：同档位记 `prefill_chunk` per-chunk ms 与 prefill t/s。
3. **KV 曲线**：测 KV 字节随 ctx 增长曲线（`ds4.c:14847` 闭式核对），外推 200K，确认 MLA 下 KV ~1.x GB。
   - 记录 **TP 下 KV 是否两机各留一份**的判断（MLA latent KV 跨 head 共享 → 近似各留一份 ~1.4GB）。
4. **MTP 真实收益**：`--mtp <draft> --mtp-draft N --mtp-margin F` 测接受率 / 实测加速
   （`ds4_session_eval_speculative_argmax` `ds4.c:20026`；timing `ds4.c:20087`）。
   - 在 k4 正确输出基础上跑，记 N=2/4/6 的接受率与净 t/s。

## 闸门 / 验收

- 管线 + 内存护栏在单机验证通过（无 OOM，峰值 footprint 在预算内）。
- 产出三张表：①decode/prefill t/s vs ctx ②KV 字节 vs ctx（含 200K 外推） ③MTP 接受率/加速 vs N。
- 给出 **单机理论上限结论** + **200K 内存可行性结论**（明确够不够、缺多少）。

## ≥20 t/s 铁律对接

- 用本 task 的实测带宽/激活字节，给出「TP 聚合 188GB/s + Q5 + MTP」到 20 t/s 的物理推算更新；
  若推算达不到 20，先在此暴露，再决定 #05 是否启动。

## 验证命令

```sh
for c in 2048 8192 32768; do
  DS4_PROFILE=1 DS4_PROFILE_FILE=/tmp/cal-$c.csv \
    ./ds4 -m gguf/ds4flash-k4.gguf --ctx $c --temp 0 -n 64 -p "$(head -c 200 speed-bench/promessi_sposi.txt)"
done
# MTP
DS4_PROFILE=1 DS4_PROFILE_FILE=/tmp/mtp.csv DS4_MTP_TIMING=1 \
  ./ds4 -m gguf/ds4flash-k4.gguf --mtp gguf/DeepSeek-V4-Flash-MTP-*.gguf --mtp-draft 4 --temp 0 -n 128 -p "..."
# 也可用 ds4-bench 取前沿 sustained t/s
```

## 产出

- 单机标定数据三件套 + 200K 内存可行性结论 → 决定 #05（TP）与 #07（重量化）的优先级与目标。
