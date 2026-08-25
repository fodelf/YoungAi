# Task 06 — Stage 3：全层 TP + 专家并行（装载策略定夺）

**状态**：🔄 部分进行。**已落地 MTLSharedEvent 局部同步**（`ds4_gpu_tp_signal_after_batch`/`ds4_gpu_tp_host_wait`，替代 #05 骨架的 `end_commands` full-drain barrier，走 [[metal_wait_anukari_precedent]] 快路径；`make` 全绿、非-TP 字节不变、CUDA stub 保 Linux 链接）。**仍未做**：shared-expert/all-reduce overlap、真实算力切分（mask 专家/ff-row）、attention 列切+第 2 个 all-reduce、专家并行池化、64/36 owner 切分。详见 `notes/execution-log.md`。
**前置依赖**：#05（最小 TP 骨架 + 对拍通过）
**类型**：双机 · 代码（主要新建工作量）

## 目标

把 TP 推广到每层标准 2 切，并用专家并行把 backbone + 热专家在两机间按带宽比并行算（聚合 ~188GB/s），
峰值内存守住 M4≤12 / M1≤8，质量过 `ds4-eval q1..q4`。

## 实施步骤

1. **每层标准 2 切**：
   - **attention**：`q_b` 按 head 列切 → output proj 行切 → all-reduce#1。
   - **MoE**：router 复制；专家按 owner 切到两机 → 加权专家输出 all-reduce#2；shared expert 切分。
2. **专家并行池化**：两机各存**不相交**的专家子集，合起来常驻 ≈ k33 等效（已优于单机 k16）。
   按各机 RAM 预算分配 owner。
3. **冷专家策略（用数据二选一）**：
   - **a 静态剪（mask）**：丢弃尾部专家 → 等效 k33 静态剪枝，零运行时 I/O，简单。**先做 a 拿基线。**
   - **b SSD 流式 + 预取**（LLM-in-a-flash / ExpertFlow）：保全量 256，router 先出选择 → 预取冷专家；
     受「投机×offload」张力约束，须热缓存覆盖投机窗口专家并集。**a 质量不足再上 b。**
4. **带宽比 owner 切分**：按「使 `M4_share/120 ≈ M1_share/68`」的 **64/36** 分配各机权重份额，实测微调。

## 闸门 / 验收

- 两机出字**正确**（与单机 greedy `--dump-logprobs` 对拍一致，重量化前应精确一致）。
- 峰值 footprint **M4 ≤12 / M1 ≤8**（看门狗两机各一，全程无 OOM/panic）。
- `./ds4-eval -m <model> --plain --questions 4 --tokens 2048 --temp 0 --seed 1` 质量可接受（对照 README 期望 token 数）。
- `DS4_PROFILE` 量出全层 all-reduce 总开销，更新 t/s 物理推算。

## 内存安全（重点：这是装载关）

- **L1 静态预算闸**：扩成全层后绑 buffer 前重算 resident（backbone份额 + owner专家×每专家字节 + KV份额 + scratch + 固定开销），`>预算×0.85` abort。
- **L2 构造上有界**：懒 mmap、关 WILLNEED 预读；Metal 只 wire 本机 owner 逐张量小视图，**绝不绑整个 86GB buffer**（IOGPU 整-buffer wire 陷阱，#68 Path1 v1 曾因此爆）。
- 绝不单机双载 86GB；冷专家 SSD（方案 b）走懒加载 expert_server_mode，避免 open 对整文件 WILLNEED 预读亚秒灌爆。
- 改完等用户确认再跑;两机二进制同步;不删另一台机器文件。

## 验证命令

```sh
make    # 两机；M1 同步重编 + 传二进制
# 全层 TP + 专家并行（方案 a 静态剪基线）
# mini(coordinator) + macbook(worker)，按 64/36 切 owner
./ds4-eval -m gguf/ds4flash-k4.gguf --plain --questions 4 --tokens 2048 --temp 0 --seed 1
```

## 产出

- 全层 TP + 专家并行可工作基线（方案 a）+ 质量分 + 峰值 footprint + sustained t/s。
- 决定是否需要方案 b（SSD 流式）或转 #07 重量化攻地板。
