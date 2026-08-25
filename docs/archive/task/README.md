# DS4 双机 200K / 20 t/s — Task 清单（按顺序执行）

> 来源：`plan.md`（已批准）。本目录把 plan 拆成有序、可独立验收的 task。
> 每个 task 完成后：把测量/补丁/范围决策/阻塞追加到 `notes/execution-log.md`（铁律）。
> 所有需要加载模型的 task 必须先过内存安全闸（L1 静态预算 + L3 phys_footprint 看门狗），改完代码等用户确认再运行。

## 执行顺序与依赖

状态图例：⬜ 未开始 · 🔄 进行中 · 🟡 代码完成·待确认运行 · ✅ 完成 · ⛔ 闸门未过/阻塞

| # | Task | 文件 | 前置依赖 | 类型 | 完成状态 |
|---|------|------|---------|------|---------|
| 1 | GPU route_translate（让 k4 输出正确） | [01-gpu-route-translate.md](01-gpu-route-translate.md) | Stage 1a 已完成 | 单机·代码 | ⚠️ 部分修复·**输出仍垃圾**。代码 make 全绿 + 顺带修了 n_total_expert 越界 bug，但 k4 实跑仍输出 `packfully`/垃圾 → 专家计算链路还有错，**未通过输出正确性验证** |
| A1 | 装载架构：选择性热集驻留（专家 offload，省内存） | [A1-expert-offload-residency.md](A1-expert-offload-residency.md) | Task 01 进行中插入 | 单机·Metal 代码 | 🟡 机制跑通·**待内存硬数据**。`DS4_METAL_EXPERT_OFFLOAD=1`：91 view（90 resident+1 non-resident），not-covered 已灭，residency 请求 2372→322ms（只 wire backbone）；速度无回归。**但 physical footprint 降幅尚无可信硬数据**（vmmap 待补测） |
| 2 | Stage 0：DS4_PROFILE 计时 + phys_footprint 看门狗 | [02-stage0-profiling-watchdog.md](02-stage0-profiling-watchdog.md) | — | 单机·基础设施 | 🟡 代码完成·make 全绿(exit0/0warn)。Stage 0 模块(footprint 看门狗 200ms 采样+`DS4_MEM_BUDGET_MB` 90% abort+DS4_PROFILE load/prefill/decode CSV+L1 起飞前 85% 预算闸)落地, 8 处 Edit, 接线 eval/sync/engine_create。零回归(env 未设走旧路径)。**待运行验证**(用户确认+看门狗) |
| 3 | Stage 1b：单机标定真实地板 + MTP 收益 + 200K 内存外推 | [03-stage1b-calibration.md](03-stage1b-calibration.md) | #1, #2 | 单机·测量 | ✅ 测量完成。三跑(ctx 2048/8192/32768)无OOM。gen 平坦 9.42~9.56 t/s(backbone带宽限→单机20t/s物理不可能);KV闭式 200K=1.89GiB/1M=8.02GiB;**20t/s需189GB/s vs TP聚合188→ceiling 19.87 ⇒ #07 Q5 缩backbone是硬前置, E0 RTT≤50µs 才行, MTP当乘子**。MTP子项推迟到#01修好。顺带暴露 #02 DS4_PROFILE钩子没接进GPU graph路径+footprint不含mmap驻留→真正护内存的是L1静态闸 |
| 4 | E0：雷电 all-reduce 延迟基准（TP go/no-go 闸门） | [04-e0-thunderbolt-allreduce-bench.md](04-e0-thunderbolt-allreduce-bench.md) | #2 | 双机·测量 | ✅ 完成·**TP NO-GO**。`tools/e0_pingpong.c`(`make e0`)双机实测。真雷电桥=`bridge0`/192.168.1.x(en5=100baseTX 非雷电，已排除)：median RTT 68µs(floor)/86µs(8KB)/116µs(32KB)，落 50-150µs MARGINAL 带。86 all-reduce/token × ~7ms 同步 + Task3 零同步 ceiling 19.87 → TP+Q5 推算 **17-18 t/s < 20**（铁律不过）。**不启动 #05/#06**，回退 **#08 单 M4+Q5+MTP**。唯一翻盘项：直配 IP 绕 bridge0 压 RTT≤50µs(需用户同意改网络，概率低) |
| 5 | Stage 2：张量并行最小骨架（只切 down_proj） | [05-stage2-tp-skeleton.md](05-stage2-tp-skeleton.md) | 用户裁定链路=雷电桥40Gb/s，覆盖 #4 | 双机·代码 | 🟡 代码完成·待双机验证。TP all-reduce 传输原语(`DS4_DIST_MSG_ALLREDUCE`+`ds4_dist_tp_allreduce_f32`)+`--tp/--tp-layers`+图集成(routed_out element-split→all-reduce, 单元测试`--tp-allreduce`绿)。非-TP 路径字节不变, `make`/`--server`全绿。骨架用 element-range 切(零 kernel 改, bit-exact), 真实算力切分推迟 #06。**待 #4 双机对拍+per-layer ms 实测** |
| 6 | Stage 3：全层 TP + 专家并行（装载策略定夺） | [06-stage3-full-tp-expert-parallel.md](06-stage3-full-tp-expert-parallel.md) | #5 | 双机·代码 | 🔄 部分。已落地 **MTLSharedEvent 局部同步**(`ds4_gpu_tp_signal_after_batch`/`_host_wait` 替代 full-drain barrier，快路径)。仍缺 shared/all-reduce overlap、真实算力切分、attn 列切、专家并行池化、64/36 owner 切分 |
| 7 | Stage 4：attention 重量化 Q8→Q5/Q4（质量门把关） | [07-stage4-attn-requant-q5.md](07-stage4-attn-requant-q5.md) | #3（地板已标定） | 离线·量化 | ⬜ 未开始 |
| 8 | Stage 5：投机解码叠加 + 拉到 200K + 逼近 20 t/s | [08-stage5-speculative-200k.md](08-stage5-speculative-200k.md) | #6, #7 | 双机·集成 | ⬜ 未开始。**因 #4 TP NO-GO，#08 备选路线（单 M4 + Q5 + MTP，不依赖 #6 双机）升级为主路径** |

## 关键闸门（任何一处不过即停）

- **E0（#4）**：RTT ≤ ~50µs → TP 继续；≥ ~150µs → TP no-go，回退「单 M4 + Q5 + MTP」备选（见 #8）。
- **TP 对拍（#5/#6）**：双机 greedy 与单机 `--dump-logprobs` 逐 logit 一致。
- **质量门（#7，铁律 correctness-before-speed）**：`ds4_test --logprob-vectors` + `ds4-eval q1..q4 --temp 0 --seed 1` 必须过，不过即逐张量回滚到 Q8。
- **内存闸（全程）**：峰值 phys_footprint M4 ≤12GB / M1 ≤8GB，全程无 OOM/panic；保内存/保平稳 > 保速度。
- **≥20 t/s 铁律**：每个会动速度的 task 动手前给出 ≥20 t/s 物理推算，达不到不执行渐进式 1-2 t/s 提升。

## 模型实数（`ds4.c:155` `DS4_SHAPE_FLASH`）

`n_layer=43`, `n_embd=4096`, `n_vocab=129280`, `n_head=64`, `n_head_kv=1`(MLA), `n_head_dim=512`,
`n_expert=256`, `n_expert_used=6`, `n_expert_shared=1`, `n_ff_exp=2048`, `n_swa=128`, indexer `top_k=512`。

- 全量 86.7GB；k4 = `gguf/ds4flash-k4.gguf` 10.02GB（单 M4 驻留 9.3GB 实证）；MTP 草稿 3.54GB。
- 实测地板：backbone（非专家 Q8 attention）= 8.81GB/token；基线单 M4 k4 = prefill 17.43 / gen 10.58 t/s。
