# DS4 双机 MTP 投机解码 + 层切分 技术方案

> 目标:用第二台机器(M1)对**单流 decode** 真正提速,而不是把第二台机器塞进同一个
> token 的确定计算里(物理上没位置)。核心手段 = **MTP 投机造"宽度" + 层切分管理容量/腾挪**。
> 测量用**真实代码问题**(有 prefill、有多 token),不是单条 `hi`。

---

## 1. 当前代码现状(已在 `mac` 分支验证,不靠记忆)

| 能力 | 状态 | 位置 |
|---|---|---|
| 分布式**层切分** (A 跑 0-32, B 跑 33-42) | ✅ 就绪 | `ds4_distributed.c`: `--role coordinator/worker`, `--layers A:B` (如 `0:32` / `33:output`) |
| 分布式 prefill 流水线 (分 chunk、可并行) | ✅ 就绪 | `--dist-prefill-chunk` / `--dist-prefill-window` |
| 张量并行 TP (element-split) | ✅ 就绪(对 k4 是纯屏障税,本方案不用) | `--tp` / `--tp-layers` |
| **单机** MTP 投机 (draft/verify, **greedy-only**) | ✅ 就绪 | `ds4.c`: `metal_graph_eval_mtp_draft` (14264), 批量验证器 (11248, 14971); CLI `--mtp/--mtp-draft/--mtp-margin` |
| **MTP 接进分布式** (drafter 在 B、跨机批量验证) | ❌ **不存在,本方案要新建** | 分布式 eval 循环里零 MTP 调用 |
| off-host MTP | ❌ 当前树搜不到任何符号 | — |

**结论:层切分和单机 MTP 各自就绪,但"MTP 在另一台机器 + 跨机验证"是净新代码。**

---

## 2. 物理背景与诚实预期(决定哪些会赢、哪些不会)

decode 一个 token 的前向是**深 43、宽 1** 的串行链:层 i+1 的输入 = 层 i 的输出,任意时刻
只有一个向量在流动。**两台机器对单 token 没有并行位置**(交棒只是换执行地点,一台总在空等)。
能让第二台机器真正干活的,只有两条:

- **造宽度**:MTP 一次猜 K 个未来 token → 把 K 个候选**打包成一个 batch 一次过 43 层**。
  decode 是带宽绑定,读一遍权重喂 1 个还是 K 个 token 墙钟几乎一样 → 命中就 ~K× 吞吐。
- **降地板**:量化 / 跳层(本方案不涉及)。

**逐项预期(k4, 能进单机 16G):**

| 配置 | prefill t/s | decode t/s | 说明 |
|---|---|---|---|
| 单机 无 MTP | 基线 | 基线 (~10+ ) | — |
| 单机 + MTP | 基线 | **↑** (×acceptance) | MTP 摊薄,**单机就能拿**;greedy-only,DS4 MTP 现为实验级、增益"slight" |
| 双机 层切分(33/10) 无 MTP | **↑ 可能超单机** | **↓ 比单机慢** | prefill 可流水并行;decode 单流多一跳、无重叠 |
| 双机 层切分 + MTP | ↑ | ≈ 或 略低于 单机+MTP | k4 不缺内存 → 层切分那一跳是纯成本 |
| 双机 **A 全模型验证 + B 纯 drafter 异步** | ≈单机 | **可能 略超 单机+MTP** | drafter 不抢 A 的 GPU;见 §4.3 |

**诚实主结论:**
- **prefill 上双机能赢**(代码问题有真实 prompt)→ 这是你会直接看到的双机红利。
- **decode 单流地板**靠 MTP 摊薄,**主要是单机本事**;双机层切分对 k4 decode 不加分(反而多一跳)。
- 双机 decode 唯一可能略超单机+MTP 的,是 §4.3 的**drafter 离线异步**(B 抢跑不占 A 算力)。
- **层切分的真正价值在容量**(模型装不下单机时),不在 k4 单流延迟。

---

## 3. 架构设计

### 3.1 方案 A — 用户原案:层切分(A:0-32 / B:33-42) + MTP drafter 在 B

```
机器 A (coordinator, 本机)        机器 B (worker, M1)
  layers 0..32                      layers 33..42 + MTP draft head + draft 模型
  tokenize / sample / 编排           出最终 hidden state → MTP 猜 K 个未来 token
        │   h_32                          │
        ├──────────────  forward  ───────►│  (验证批 K 个候选: A 的 0-32 → B 的 33-42)
        │◄────────────  logits/draft ─────┤
   接受/拒绝 + KV 回滚                  draft 在管道尾端天然就位(B 持有末层)
```

- **draft 放 B 的理由**:MTP 头吃的是主模型**最终** hidden state,而 B 持有末段层(33-42),
  draft 头跟末层同机,出 hidden 后立刻 draft,不必把 hidden 拉回 A。
- **验证批**:K 个候选 token 是一次 mini-batch 前向,等价于一个小 prefill chunk →
  **复用现有分布式 WORK frame 的批量/chunk 路径**(`--dist-prefill-chunk` 已有批通道)。
- **k4 评价**:层切分对 k4 无内存收益,每次验证前向多 A→B 一跳 → decode 大概率 ≤ 单机+MTP。
  **此方案的价值是"大模型 + MTP"通用骨架**,不是 k4 提速。

### 3.2 关键子问题(无论 A/B 方案都要解决)

1. **draft 位置与 hidden 流向**:draft 头必须在持有主模型**末层**的机器上;否则要把最终
   hidden 跨机拉回,白付一跳。
2. **跨机批量验证**:K 候选打包,走分布式 batch 前向(A 段 → B 段),末端出 K 组 logits。
3. **KV 回滚**:接受 j 个、拒绝其余时,**两台机器各自的层切片 KV** 都要回滚到 j。需要在
   WORK/RESULT 协议里带"接受长度",worker 据此截断自己那段 KV。现有协议有 token-prefix
   滚动哈希防陈旧 KV,可在此之上加 accept_len 字段。
4. **greedy 约束**:沿用单机 MTP 的 temp=0 限制(投机语法期强制贪婪),先不动采样语义。
5. **滚动哈希一致**:每个 work item 的 64-bit token-prefix 哈希要把"已接受前缀"算进去,
   保证 worker KV 与 leader 一致(分布式已有此机制,扩展即可)。

### 3.3 方案 B(k4 专用,更可能真超单机)— A 全模型验证 + B 纯 drafter 异步抢跑

```
机器 A: 完整 k4 (全 43 层, 验证器/verifier)      机器 B: 仅 draft 模型 (MTP, 体积小)
  验证第 N 轮的 K 候选 (单机速度, 无层切分跳)        同时在猜第 N+1 轮的 K 候选
        │◄──── 上一轮接受的末 token ─────┤
   两台真正重叠: A 验证 round N 时, B 已在 draft round N+1
```

- **为什么可能赢单机+MTP**:单机上 draft 和 verify **抢同一块 GPU**;把 draft 挪到 B,
  A 的关键路径只剩 verify → 省掉 draft 占用。增益 = draft 成本 / 总成本,有上限但为正。
- A 需装全 k4(16G 够);B 只装小 draft 模型(轻)。**不切主模型层、不付层切分那一跳。**
- 代价:draft↔verify 跨机同步(每轮 1 次,传 K 个 token id + 末 hidden,极小)。
- **这是 k4 单流 decode 想超单机的最优落点;层切分(方案 A)留给装不下单机的大模型。**

### 3.4 动态层级配置

- 启动期可配:`--layers 0:32` / `33:output`,改数字即可重切(已支持)。
- **运行期动态再平衡**(跑着改 33/10→30/13)= 额外工程(要在线迁移层权重 + KV),
  **v1 不做**,先做启动期可配 + 测出最优静态切点。

---

## 4. 需要新增/改动的代码(grounded)

1. **分布式 eval 接 MTP draft**(主工作量):
   - worker(持末层)出最终 hidden 后调用现有 `metal_graph_eval_mtp_draft` 产出 K 候选;
   - 候选回传 coordinator;coordinator 编排 K 候选的批量验证前向(复用 chunk 批通道)。
2. **协议扩展**:WORK/RESULT 加 `draft_tokens[K]` 与 `accept_len` 字段;滚动哈希纳入已接受前缀。
3. **跨机 KV 回滚**:worker 收到 `accept_len` 后截断自身层切片 KV 到该长度。
4. **CLI**:`--mtp FILE` 在分布式下需指明"draft 在哪个 role/哪台机";加 `--mtp-role worker`
   (或自动绑定到持末层的 worker)。方案 B 另需"B 只载 draft 模型、不载主模型层"的模式。
5. **方案 B 的异步抢跑**:drafter 与 verifier 解耦成生产者/消费者,B 提前 draft 下一轮。

> 不动:TP all-reduce(本方案不用 TP);单机 MTP 既有路径(复用不改语义);mmap 加载策略。

---

## 5. 分阶段实施(每阶段可独立测量、可回退)

- **Phase 0(零改动,先拿基线)**:用现成功能测 4 个点 ——
  ① 单机无 MTP ② 单机+MTP ③ 双机层切分无 MTP ④(若有 draft 模型)单机+MTP 的 acceptance。
  **先知道起点,再决定值不值得建。**
- **Phase 1**:方案 A —— MTP draft 接进分布式 + 跨机批量验证 + KV 回滚(greedy)。先正确性
  (双机+MTP 输出 == 单机+MTP 输出,逐 token),再看 t/s。
- **Phase 2**:方案 B —— B 纯 drafter 异步抢跑(k4 单流提速的真正落点)。
- **Phase 3(可选)**:运行期动态层级再平衡。

---

## 6. 测量协议(真实代码问题)

- **Prompt**:一个简单代码问题(如"写一个 Python 函数判断字符串是否回文,并解释"),
  保证有真实 prefill + 几十~上百 decode token。`--temp 0 --seed 1` 可复现。
- **指标**:分别记 **prefill t/s** 与 **decode t/s**(用 ds4 自带计时行;decode 看瞬时,不混 prefill)。
- **对照矩阵**(同 prompt、同 seed):
  | # | 配置 | 预期看点 |
  |---|---|---|
  | 1 | 单机 无 MTP | decode 基线 |
  | 2 | 单机 + MTP | decode 摊薄增益(单机本事) |
  | 3 | 双机 层切分 33/10 无 MTP | prefill 是否超单机;decode 是否如预期变慢 |
  | 4 | 双机 层切分 + MTP (方案 A) | 是否 ≤ #2 |
  | 5 | 双机 drafter 离线 (方案 B) | 是否 略超 #2 |
- **判据**:prefill 看 #3/#4 是否超 #1(双机 prefill 红利);decode 看 #5 能否超 #2。
  **数据说话,不再推演估算。**

---

## 7. 内存安全闸(硬约束)

- k4 backbone ~8-10G,A 单机/各机层切片均 < 16G,沿用 `tools/tp_k4_speed.sh` 的看门狗
  (RSS 超阈两边同杀、只杀进程不删文件)与 `DS4_MEM_BUDGET_MB` 预算闸。
- 方案 B:A 载全 k4(已知安全足迹),B 只载小 draft 模型(更轻)。
- **绝不单机双载主模型;改完编译通过后等用户在终端手动跑(Ctrl+C 在手),不自动执行。**
- 两机共享 `CORE_OBJS`(含 `ds4_distributed.o`),任何改动两机都要同步重编 + 传二进制。

---

## 8. 风险与未决

- **MTP acceptance 率是总闸**:DS4 MTP 当前实验级、增益"slight"(见 `CLAUDE.md`)。draft 模型
  与目标分布越贴,K 有效接受越高;贴不上则 §2 的增益全打折。Phase 0 ④ 先测真实 acceptance。
- **跨机 KV 回滚的正确性**:接受/拒绝时两机 KV 必须一致,是 Phase 1 的主要正确性风险点。
- **greedy-only**:投机沿用 temp=0;带温度采样的投机不在本方案范围。
- **层切分对 k4 decode 是负担**(无内存收益 + 多一跳)→ k4 想超单机靠方案 B,不靠方案 A 的切层。
