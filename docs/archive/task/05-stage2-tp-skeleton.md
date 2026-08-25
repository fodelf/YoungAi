# Task 05 — Stage 2：张量并行最小骨架（只切一处）

**状态**：🟡 代码完成·待双机验证。TP all-reduce 传输原语 + `--tp/--tp-layers` + 图集成（routed_out element-range 切分 → all-reduce，单元测试 `ds4_test --tp-allreduce` 绿，`make`/`--server` 全绿，非-TP 路径字节不变）。骨架用零-kernel-改的 element-split（bit-exact），真实算力切分（mask 专家/ff-row）+ MTLSharedEvent 局部同步推迟到 #06。待 #4 双机 top1 对拍 + per-layer all-reduce/barrier 实测 ms。详见 `notes/execution-log.md`。
**前置依赖**：用户裁定链路=雷电桥 40Gb/s（覆盖 #04 RTT 闸）
**类型**：双机 · 代码（先只切一处，最小可验证）

## 目标

实现最小可验证的 TP：**仅对 MoE 的 `down_proj`（row-parallel）做两机切分 + 1 次 all-reduce**，其余仍单机，
只跑 2–3 层。目的是把「图编排切分点 + all-reduce 帧 + 两机对拍」整条链路打通，再谈推广。

## 新代码落点

- `ds4_distributed.c` / `.h`：增 TP 模式。复用现有 TCP 帧 `WORK`/`RESULT`，或加紧凑 `ALLREDUCE` 帧
  （活动 token 的局部和 → 求和 → 广播）。沿用现有 `*_usec` 计时透出（`ds4_distributed.c`）。
- `ds4_metal.m`：加「部分 matmul（只算本机 owner 的行）+ 把局部和交给 host all-reduce」钩子。
- `ds4.c`：图编排在 `down_proj` 切分点插同步（local matmul → host all-reduce → 继续）。
- 沿用 `--role coordinator|worker`；coordinator 负责 tokenization/sampling。

## Apple 特性降同步开销

- 用 `MTLSharedEvent.waitUntilSignaledValue` 替代 `waitUntilCompleted`（TP 小 CB 多，每-CB 调度开销是大头；
  Anukari/Apple 工程师确认可把 ~150ms 降到 <50µs）。
- 统一内存零拷贝；`MTLResidencySet` 预算；all-reduce 缓冲 KB 级预分配（不进 per-token 分配热路径）。

## 闸门 / 验收

- **逐 logit 对拍**：TP 两机 greedy 输出与单机 greedy `--dump-logprobs` **逐 logit 一致**（容差 ~0，row-parallel 是精确切分）。
- `DS4_PROFILE`（#02 的 all-reduce 桩接上）量出每层 all-reduce 实测 ms，**核对 E0（#04）预测**，偏差解释清楚。
- 内存：两机峰值 footprint 在预算内（只切 down_proj，权重份额几乎不变）。

## 内存安全

- 不改装载策略，仍 k4；all-reduce 缓冲预分配封顶；看门狗两机各一。
- 本机编译后考虑 M1 是否需同步重编 + 传二进制（两机共享 CORE_OBJS；漏传 replica 会 abort）。

## 验证命令

```sh
make                         # mini
# (M1 上同步 make + 传二进制)
# 单机基准
./ds4 --dump-logprobs /tmp/single.json --temp 0 -m gguf/ds4flash-k4.gguf -p "1+1=?"
# 双机 TP（只切 down_proj，2-3 层）
# mini:    ./ds4 --role coordinator --layers ... --dump-logprobs /tmp/tp.json --temp 0 -p "1+1=?"
# macbook: ./ds4 --role worker ...
diff <(jq .top1 /tmp/single.json) <(jq .top1 /tmp/tp.json)   # 应一致
```

## 产出

- 可工作的 TP 切分点 + all-reduce + 对拍闸门 → 解锁 #06 推广到全层。
- 每层 all-reduce 实测 ms（对照 E0），更新 ≥20 t/s 物理推算。
