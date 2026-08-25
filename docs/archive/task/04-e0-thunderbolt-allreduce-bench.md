# Task 04 — E0：雷电 all-reduce 延迟基准（TP go/no-go 闸门）

**状态**：✅ 完成 · **裁决 TP NO-GO**。双机实测 bridge0/192.168.1.x（真雷电桥，en5=100baseTX 非雷电已排除）：median RTT 68µs(floor)~116µs(32KB)，落 MARGINAL 带；TP+Q5 推算 17-18 t/s < 20（铁律），不启动 #05/#06，回退 #08。详见 `notes/execution-log.md`。
**前置依赖**：#02（计时基础设施，复用 `bench_now_sec` / CLOCK_MONOTONIC）
**类型**：双机 · 测量（plan 定义为「整个方案的第一个动作」，是 TP 总闸门）

## 目标

在雷电直连链路上实测往返延迟与抖动，决定张量并行（#05/#06）是否值得做。
TP 每 token 约 **86 次逐层同步**（2 all-reduce × 43 层），延迟直接决定 TP 性价比。

## 链路

- Mac mini = M4，mini 侧 `en5`（link-local `169.254.188.38`），雷电直连，≤40 Gb/s（≈5 GB/s）。
- MacBook = M1（base，~68 GB/s 理论带宽，TB3/USB4，无 macOS RDMA）。

## 实施步骤

1. 写最小 ping-pong（**独立小程序或 ds4-bench 子命令**，不动核心推理代码）：
   - TCP over `en5`，`TCP_NODELAY`，交换 **32KB** payload，往返 **1 万次**。
   - 两端各用 `clock_gettime(CLOCK_MONOTONIC)` 记 RTT；统计 min/median/p99/抖动。
2. 顺带测有效吞吐（确认 ≈5 GB/s），用于 #06 的带宽比 64/36 owner 切分校核。
3. 产出一张 RTT / 抖动表，写入 `notes/execution-log.md`。

## 闸门 / 验收（TP go/no-go）

- **RTT ≤ ~50µs** → 同步开销 ~2.6–4.3ms/token → **TP 可行，继续 #05**。
- **RTT ≥ ~150µs** → ~13ms/token+ → **TP 性价比崩**，回退「单 M4 + Q5 + MTP」备选（见 #08 备选路线）。
- 中间区间（50–150µs）：记录实测，按「TP+Q5 是否仍能物理推算到 ≥20 t/s」决定，不达标不启动 TP。

## 内存安全

- ping-pong 只分配 KB 级缓冲，不加载模型 → 无内存风险（仍走看门狗确认）。
- 不删另一台机器任何文件；"清理"= 关进程不删文件（铁律）。

## 验证命令

```sh
# 形态示意（具体落点：ds4-bench 子命令或独立 tools/ 小程序）
# mini:    ./e0-pingpong --listen 169.254.188.38:5599 --size 32768 --iters 10000
# macbook: ./e0-pingpong --connect 169.254.188.38:5599 --size 32768 --iters 10000
```

## 产出

- RTT/抖动/吞吐基准表 → TP go/no-go 决断；若 go，给出每层 all-reduce 预测 ms（供 #05 核对实测）。
