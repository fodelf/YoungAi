# tensor-parallel-design.md — 张量并行(TP)改造调研与方案（双机 16GB Mac）

> ⚠️ **2026-06-20 对抗性复核更正（必读)：本文 §3 的定量预估已被 `tensor-parallel-feasibility.md` 证伪。**
> 关键翻盘：① TP 与 layer-slicing 在 `ds4_distributed.c:10354-10364` **硬互斥**,「EP 叠在 layer-slicing 骨架上」不成立,
> 当前 5.36 是纯 layer-pipeline 无 TP;② TP 模式每台**常驻全 backbone ≈8.4GiB**(layer-slice 两倍),顶 12GiB 看门狗红线;
> ③ EP 真实成绩是**冷态 1.02→1.67 t/s**(非 4.6-5.4),暖态净负;④ 「300µs RTT / AR drain 占 40%」实测为 84-140µs / AR 占 decode<1%。
> **裁决=no-go(经典 TP 主路,禁改码前提)。** 读本文 §3 前先读 `tensor-parallel-feasibility.md`。
> **2026-06-20 更新:用户决定允许改码** ⇒ no-go 翻转为有条件可行,落地方案见 **`tensor-parallel-ep-plan.md`**(EP-replaces-PP + backbone Q4_K + KV Lever B/C,冷态封顶 ~1.3 t/s)。本文方案部分已被该文档取代。
>
> 状态：调研稿（2026-06-20）。本文承接 `plan.md`（TP/投机总路线）与 `project.md`
> （q2 全量 SSD 流式总规划），聚焦"**TP 改造能否、以何种机制把当前 5.36 t/s 往上推**"。
> 结论先行：在本项目的物理约束下，**经典 TP（切 compute）不是提速主路；真正有价值的是
> 把现有 TP 重新定位为 Expert-Parallel（EP）——用两机 RAM 池化降低每台 SSD 流量**。
> 执行纪律不变：每个补丁/测量/决策追加 `notes/execution-log.md`。

---

## 0. TL;DR（先认账）

- **瓶颈是 routed-expert SSD IO（~1.70 GiB/token），不是 compute，也不是 KV。** TP 的经典价值
  （切 compute 让算力翻倍）在 compute-bound 时才直接兑现；本项目 **batch=1 decode 是 IO/带宽-bound**，
  切 compute 不解瓶颈。这一点是整篇评估的地基。
- **业界最强证据（MLX TB5 RDMA 实测）：** 4 节点 M3 Ultra 跑 1T MoE（Kimi-K2），**TP 只比 PP 快 2.3%**
  （14.82 vs 14.49 t/s），而且这还是在 **RDMA 3µs 延迟**下；本项目是 **TCP-over-Thunderbolt ~300µs 延迟**，
  TP 的 86 次/token 逐层同步会被延迟吃掉。**经典 TP 在本项目无收益甚至负收益。**
- **现有 TP skeleton 已落地三相**（`DS4_TP_EXPERT_SPLIT` / `_SHARED_SPLIT` / `_EXPERT_SPLIT_BATCH`），
  其中 **`DS4_TP_EXPERT_SPLIT` 本质就是 EP**——每个 peer 只 gather/compute 自己那 k 个专家，
  **每台 SSD 流量减半**，这才是 TP 框架在本项目唯一能直接提速的机制。
- **现实裁决**：在 layer-slicing 之上把 EP（expert-split）打磨稳，**现实档 1.3–1.5× ≈ 4.6–5.4 t/s 区间稳态**，
  乐观档（EP + 池化命中叠加）**~6–8 t/s**，悲观档（同步链 drain 吃掉收益）**回退到 ≈ 5.36 持平**。
  **30 t/s 不靠 TP 这条线达成**（撞 W3 backbone 带宽墙 10–18 t/s），它属于 §3.5 编程域有效 t/s 范畴。

---

## 1. 调研综述

### 1.1 经典 TP 理论（Megatron-LM）

Megatron 的两种切法（[Megatron-LM 原论文](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf)）：

- **Column-parallel（按输出维切）**：第一个线性层（MLP 的 `gate/up`，attention 的 `q/k/v` proj）
  按列切，每 peer 算输出的一段，**无需同步**即可喂给下一层。
- **Row-parallel（按输入维切）**：第二个线性层（MLP 的 `down`，attention 的 `output` proj）按行切，
  每 peer 算一个 **partial out[hidden]**，**末尾一次 all-reduce SUM** 重组完整结果。

**每层每 token 的同步次数**：标准 Transformer block = **2 次 all-reduce**（attention 输出后 1 次 +
MLP 输出后 1 次），forward+backward 推理只算 forward = 2 次。通信量公式（forward，单 token decode）：

```
per-token all-reduce 字节 = n_layer × 2(次/层) × hidden_dim × dtype_bytes × 2(all-reduce 两阶段折算)
```

对本项目（见 §3 硬数字），**通信量小（KB 级），瓶颈在延迟而非带宽**——这是 batch=1 decode 的本质：
每次同步只交换一个 `[hidden]` 向量，TCP 往返延迟（~300µs）× 同步次数（86）才是开销大头。
（[AWS Neuron TP 概述](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tensor_parallelism_overview.html)、
[LLM Scaling Hierarchy](https://akashsahani2001.medium.com/the-llm-scaling-hierarchy-mastering-every-dimension-of-parallelism-67937ae1d78e)）

### 1.2 MoE 并行：EP vs TP（关键区分）

- **Tensor Parallel for MoE**：把**每个专家内部**的 gate/up/down 按 Megatron 列/行切，所有专家都被两机
  各算一半，末尾 all-reduce。**每台仍要 touch 全部被选专家的权重** → 不降 SSD 流量。
- **Expert Parallel（EP）**：把**专家整体**分到不同设备（每设备持有不相交的专家子集），token 经
  all-to-all 路由到拥有该专家的设备。**每台只 touch 自己那份专家** → **降权重带宽/SSD 流量**。
  （[TensorRT-LLM EP](https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html)、
  [LMSYS 大规模 EP 部署](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)）

**vLLM/SGLang 的现代做法**（对本项目最有启发）：对 DeepSeek 这类 **MLA + MoE** 模型，
**attention 走 data-parallel（DP），MoE 走 EP**——因为 MLA 的 KV 头很少（本模型 `n_head_kv=1`），
TP 切 attention 收益小且引入同步，DP 复制 attention 反而省同步；省下的通信预算全给 MoE 的 EP。
（[vLLM Data Parallel 部署](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)、
[Red Hat Wide-EP](https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep)）

> **对本项目的直接映射**：本项目的 `DS4_TP_EXPERT_SPLIT` = 两机各 gather/compute 自己那
> k=n_used/2 个专家 → **正是 2-way EP**。它降的就是 SSD 流量（W2 墙），不是降 compute。
> 这是现有 skeleton 里**唯一对 IO-bound 瓶颈直接有效**的相位。

### 1.3 开源实现对比（重点：消费级/Mac/Thunderbolt 低带宽）

| 项目 | 互联 | 并行策略 | 关键数字/结论 | 对本项目适用性 |
|---|---|---|---|---|
| **EXO 1.0 + JACCL** | TB5 **RDMA** | PP（Ring）+ TP（JACCL，需 RDMA） | RDMA 把 Mac 间延迟 **300µs→3µs**；**TP 明确要求 RDMA**，TCP 下 TP "理论可行但实际不可用" | ⚠️ 本项目是 **TCP-over-TB（~300µs）非 RDMA** → 经典 TP 落在"不可用"区；EXO 自己也只在 RDMA 上开 TP |
| **MLX distributed** | TB5 RDMA / Ring | PP（Ring backend）+ TP（JACCL backend） | **实测 PP4=14.49 / TP4=14.82 t/s（1T MoE，4×M3 Ultra）→ TP 仅快 2.3%**；PP 加载快 4.5×、省 21% 内存 | ✅ 黄金对照：即便 RDMA 把 all-reduce "几乎免费"，TP 相对 PP 也几乎无增益 → **PP/layer-slicing 是对的主路** |
| **llama.cpp `--rpc`** | TCP（10GbE/TB-IP） | tensor-split（layer/tensor 切）+ AllReduce | **10Gbps 下 RPC ≈ 0.2ms/层/token，80 层 = 16ms/token 网络开销** | ⚠️ 量级直接套用：43 层 × 0.2ms ≈ 8.6ms/token 纯同步（仅当带宽是瓶颈；本项目延迟更主导） |
| **prima.cpp** | Wi-Fi/有线 | **Pipelined-Ring Parallelism + mmap offload** | 70B@4 家用设备 674ms/token；**mmap 懒加载 + 预取-释放冲突处理 + 投机 32B 达 26 t/s** | ✅✅ 与本项目同形态（mmap/慢盘/异构）：**PP + IO/compute overlap + 投机** 是验证过的消费级路线，非 TP |
| **vLLM / SGLang** | NVLink/RDMA 数据中心 | DP attention + EP MoE | SGLang PD 分离 + 大规模 EP 22.3k tok/s/node | △ 拓扑思路（DP-attn + EP-MoE）可借，绝对数字不可比（NVLink 800GB/s vs TB ~5GB/s） |
| **TensorRT-LLM** | NVLink | TP / EP / 混合 | EP "每设备一专家，all-to-all 通信" | △ EP 概念来源，硬件假设不成立 |

### 1.4 低带宽互联下的通信优化（结论性）

- **Ring vs Tree all-reduce**：两机 = 退化为**单次有序全交换**（本项目 `dist_tp_exchange` 已用全双工
  pump，~1× 单向延迟，已是 2-peer 最优）。Ring/Tree 拓扑优化只在 ≥3 节点才有意义，本项目 N=2 无空间。
- **通信压缩（int8/fp8 all-reduce）**：训练侧用 int8 ring-all-reduce（fp32 累加）省 4× 带宽
  （[INTELLECT-1](https://arxiv.org/pdf/2412.01152)、[FP8-LM](https://arxiv.org/pdf/2310.18313)）。
  **但本项目通信量是 KB 级、瓶颈是延迟不是带宽 → 压缩无用**（压 16KB→4KB 不改 300µs RTT）。
- **计算-通信重叠**：prima.cpp 的 PRP 把 disk IO 与 compute/通信重叠，是本项目该借的核心技巧
  （已部分落地：A3 gather overlap N=2）。
- **Thunderbolt 实测**：TB3/4 IP-networking 标称 ~10GbE，实测有效 ~5–20Gbps；**关键不是峰值带宽而是
  TCP 往返延迟（~100–300µs）**，这正是 batch=1 decode 每层小同步的杀手。
  （[macperformanceguide TB3 带宽](https://macperformanceguide.com/blog/2019/20190128_1352-understanding-Thunderbolt3-bandwidth.html)）

### 1.5 TP vs 当前 layer-slicing（pipeline）取舍——业界明确结论

> "Pipeline parallelism reduces inter-stage communication to **point-to-point** operations rather than
> the **O(n²) all-reduce per layer** required by tensor parallelism." —— exo/MLX 文档

- **PP（本项目主路）**：decode 每 token **一次**跨机 hop（coord→worker），自回归串行。
- **TP**：decode 每 token **每层 2 次** all-reduce = 43×2 = **86 次同步**，每次卡 ~RTT。
- **MLX 实测裁决**：连 RDMA（3µs）下 TP 都只赢 PP 2.3%；本项目 TCP（300µs）下 TP 的 86 次同步
  ≈ 86×300µs = **25.8ms/token 纯延迟**，**远比 PP 的单 hop 贵**。

**=> 经典 TP（切 compute）在本项目劣于现有 layer-slicing。** 但这不否定 TP 框架——见 §2 重定位为 EP。

---

## 2. 本项目 TP/EP 改造方案（基于现有 skeleton 的增量设计）

### 2.0 现有 skeleton 现状盘点（先读代码再改）

| 组件 | 落点 | 状态 |
|---|---|---|
| TP all-reduce 传输 | `ds4_distributed.c:1760-1884`（`struct ds4_dist_tp` / `dist_tp_exchange` 全双工 pump / `ds4_dist_tp_allreduce_f32`） | ✅ 已落地，2-peer 最优全交换，KB 级 scratch，内存安全 |
| 连接建立 + reverse-connect | `ds4_distributed.c:20024-20041`（`DS4_TP_REVERSE_CONNECT` 绕 M1 出站 EHOSTUNREACH） | ✅ 已落地 |
| 非阻塞控制 socket poll-wait | `ds4_distributed.c:1473-1511`（wave-72 decode bug 修复） | ✅ 已落地 |
| row-parallel Q8_0 matvec kernel | `ds4_gpu.h:187-200`（`ds4_gpu_matmul_q8_0_rowslice_tensor`） | ✅ kernel 已实现 |
| **Phase 3 EP：routed-expert split** | `ds4.c:11141-11260`（`DS4_TP_EXPERT_SPLIT` + `DS4_TP_SPLIT_LOW` 非对称 k） | ✅ 已落地，**每台 gather k 专家 → SSD 流量减半**，AR-SUM 重组 bit-exact |
| **Phase 1：shared-expert dense FFN split** | `ds4.c:11261-11344`（`DS4_TP_SHARED_SPLIT`，gate/up 列切 + down 行切 + AR） | ✅ 已落地 |
| **Phase 3 batch：prefill expert split** | `ds4.c:14146-14211`（`DS4_TP_EXPERT_SPLIT_BATCH`） | ✅ 已落地，冷批实验 opt-in |
| TP 同步用 MTLSharedEvent 快路径 | `ds4.c:11222-11253`（signal+flush+host_wait 替代 waitUntilCompleted） | ✅ 已落地 |
| 状态字段 | `ds4.c:9305-9309`（`tp` / `tp_layers` / `tp_owns_low` / `tp_vec`） | ✅ |

**结论：基础设施完备，不需要从零写 TP。** 改造 = **把重心从"切 compute"挪到"EP 池化降 SSD 流量"+ 调同步链**。

### 2.1 方案 A（推荐主路）：EP 池化降 SSD 流量 + 同步链优化

**核心论点**：`DS4_TP_EXPERT_SPLIT` 已经让两机各 gather 一半专家。下一步把它从"对称 50/50"升级为
**带宽感知 + RAM 池化感知**的 EP，让两机的专家**驻留集不相交且互补**，从而：
1. 每台 SSD 每 token 只拉 ~k/2×6.75MiB（已减半）；
2. 两机 12+12=24GiB RAM 池化后，热专家命中率叠加 → 进一步减少落盘的专家数。

**改哪些文件/函数（增量）：**

1. **`ds4.c:11148-11172`（expert-split owner 选择）**：当前按 slot 序 `[0,k)/[k,n_used)` 切。
   改为**按 expert-id 的固定哈希/owner-map 切**（每个 expert-id 静态归属某台），让"谁拥有哪些专家"
   在跨 token 间稳定 → 各机的专家缓存命中可累积（当前 slot 序切，owner 随路由变化，缓存抖动）。
   新 env：`DS4_TP_EP_OWNER_MAP`（`hash`/`slot`，默认 `slot` 保持现状可 A/B）。
2. **`ds4.c:11148`（k 不平衡调度）**：`DS4_TP_SPLIT_LOW` 已支持非对称 k；与 §1.1 的盘速不对称
   （mini 盘 1.9GB/s，MacBook 盘 8-21GB/s 突发）联动——让快盘 worker 多拿专家。无需新代码，
   是调参 + 在 `notes/execution-log.md` 记最优 k。
3. **同步链 drain 优化**（`ds4.c:11222-11253`）：当前 AR 前要 `signal+flush+host_wait` 把
   routed_out 读回 host。`DS4_TP_AR_LOG` 实测显示 drain 占 AR 总时 40-45%（与 project.md §1.5 一致）。
   优化方向：**把 attention-output AR 与 routed-MoE AR 合并成一次同步**（一个 `[2×n_embd]` exchange），
   把 43×2 次同步降到 43 次。新 env：`DS4_TP_FUSE_AR`（默认 off）。

### 2.2 方案 B（探索）：DP-attention + EP-MoE 拓扑（借 vLLM/SGLang）

本模型 `n_head_kv=1`（MLA），attention 切 TP 收益极小却引入同步。借 vLLM 思路：
- **attention 两机各算完整一份（DP，KV 各留一份 ~1.4GB/机）**，无 attention 同步；
- **省下的同步预算全给 MoE 的 EP**（方案 A）。

这与当前 layer-slicing 是**冲突**的（layer-slicing 是把层切开两机各算一段；DP-attn 是两机都算全部
attention 层）。**不建议在 layer-slicing 拓扑里硬塞**，列为 P2 探索：仅当未来切换到"两机都驻全
backbone + 各驻一半专家"的对称拓扑时才有意义（届时 backbone 需重量化降到能两机各驻一份）。

### 2.3 不做的事（明确排除）

- ❌ **经典 TP 切 attention/dense compute 求算力翻倍** —— §1.5 实测裁决：TCP 延迟下负收益。
- ❌ **通信量压缩（int8/fp8 all-reduce）** —— 通信量 KB 级、延迟主导，压缩不解。
- ❌ **Ring/Tree all-reduce 拓扑** —— N=2，已是单次全交换最优。

---

## 3. 定量性能评估（硬数字，乐观/现实/悲观三档）

### 3.1 模型实数（`ds4.c` `DS4_SHAPE_FLASH` 实证）

```
n_layer = 43,  n_embd(hidden) = 4096,  n_head = 64,  n_head_kv = 1(MLA),  n_head_dim = 512
n_expert = 256,  n_expert_used(top-k) = 6,  n_expert_shared = 1,  n_ff_exp = 2048
每专家 ≈ 6.75 MiB (gate 2.06 IQ2_XXS + up 2.06 + down 2.625 Q2_K)
每 token routed 激活 = 43 层 × 6 专家 × 6.75 MiB ≈ 1.70 GiB
```

### 3.2 经典 TP 跨机通信量（每 token decode）

每层 2 次 all-reduce，每次交换 `[n_embd]` f32：

```
每次 all-reduce payload = 4096 × 4 B = 16 KiB（单向，全双工 pump 后 ≈ 1× 单向延迟）
每 token 同步次数 = 43 层 × 2 = 86 次
每 token 通信量 = 86 × 16 KiB ≈ 1.34 MiB   ← 带宽上微不足道（÷5GB/s = 0.27ms）
每 token 同步延迟 = 86 × RTT
```

**延迟才是杀手**（这是 batch=1 decode 的本质）：

| 互联/延迟 | 每 token 纯同步延迟 (86×RTT) | 裁决 |
|---|---|---|
| RDMA（EXO/JACCL，~3µs） | 86 × 3µs ≈ **0.26ms** | TP 可行（但 MLX 实测也只赢 PP 2.3%） |
| **本项目 TCP-over-TB（~300µs，实测以 E0 为准）** | 86 × 300µs ≈ **25.8ms** ≈ 39 t/s 的纯同步天花板 | ⚠️ 与 backbone 计算时间同量级，**经典 TP 吞掉收益** |
| 悲观 TCP（抖动 ~1ms） | 86 × 1ms ≈ **86ms** ≈ 11.6 t/s 天花板 | ❌ TP 直接拖垮 |

> **对比当前 layer-slicing 单 hop**：decode 每 token **1 次** coord→worker activation 传输（~几十 KB），
> 单 RTT ≈ 0.3–1ms。**PP 的同步成本是 TP 的 1/86。** 这就是 MLX 实测 PP≈TP 的根因（PP 同步少到忽略，
> TP 同步多但单次便宜，两者抵消；TCP 下 TP 单次不便宜 → PP 胜）。

### 3.3 EP（方案 A）的提速机制——这才是有效路径

EP **不靠同步省时间，靠减少每台 SSD 流量**：

```
当前（无 expert-split）：每台 touch 全部 6 专家 → 每层 6×6.75MiB
EP split k=3/3：       每台 touch 3 专家 → 每层 3×6.75MiB = SSD 流量减半
```

W2 SSD 墙重算（project.md §1）：1.70GiB/token ÷ 双机并行 ~6GB/s ≈ 280ms/token ≈ 3.5 t/s 全冷。
**EP 把每台流量减半后**，若两盘真并行，等效聚合带宽翻倍 → 理论 ~7 t/s 全冷；叠 RAM 池化命中
（24GiB 给专家缓存，路由偏斜下 40-60% 命中）→ 现实 ~4.5–6 t/s。**但要扣同步链 drain 开销**
（AR 的 signal+flush+host_wait 占 40-45%）。

### 3.4 三档预估（基于 5.36 t/s 现状基线）

| 档位 | t/s | 机制 | 假设（写清） |
|---|---:|---|---|
| **乐观** | **6–8** | EP owner-map 稳定缓存 + 两盘完美并行 + AR 融合（86→43 次同步）+ 池化命中 50-60% | 同步 drain 砍半、worker 快盘满载、路由偏斜稳定、E0 实测 RTT≤300µs 不抖 |
| **现实** | **4.6–5.4** | EP 减半 SSD 流量 + 现有同步链 + 命中 40% | drain 仍占 40%、盘速不对称（mini 慢盘拖后腿）、k 调到带宽最优、与 layer-slicing 协同稳态 |
| **悲观** | **≈5.36（持平）/ 略降** | EP 收益被同步链 drain + AR 抖动吃掉 | TCP RTT 抖到 ~1ms、owner-map 缓存抖动、worker 掉链、双盘 staging 在 mini 侧净亏（project.md §1 已实证 worker←mini staging -18%） |

**30 t/s 裁决（诚实）：TP/EP 这条线达不到 30。** 撞 **W3 backbone 带宽墙**：即便专家全免费，
coord 4.07GiB÷120GB/s + worker 4.4GiB÷68GB/s 串行 ≈ 56–99ms/token ⇒ **10–18 t/s 单前向天花板**。
30 t/s 只属于 §3.5 编程域有效 t/s（copy-spec 多 token/前向 + 前缀复用），**不是 TP 能给的**。

### 3.5 与编程域有效 t/s 的关系

TP/EP 提的是**单前向 decode 的 t/s**；30 t/s 目标走的是 `project.md` §3.5 / P-Code 的
copy-speculation（n-gram drafter 复用 VERIFY 协议，多 token/前向）。**两者正交可叠加**：
EP 把单前向 5.36→现实 ~5 区间稳住，copy-spec 在 echo 重的编辑回合把有效 t/s 乘上去。
**EP 的价值是抬高 copy-spec 的基线乘数，不是自己冲 30。**

---

## 4. 与现有 layer-slicing / expert-streaming 的关系

| 维度 | layer-slicing（PP，现主路） | TP（经典，切 compute） | EP（方案 A，本文推荐） |
|---|---|---|---|
| 同步频率/token | 1 hop | 86 次 all-reduce | 同 PP（EP 在层内做，复用 PP 拓扑） |
| 对 W2 SSD 墙 | 不减流量 | 不减流量 | **每台减半** ✅ |
| 对 W3 backbone 墙 | 受制（串行 hop） | 切 backbone 算力但同步抵消 | 不动 backbone |
| TCP-over-TB 适配 | ✅ 单 hop 友好 | ❌ 延迟杀手 | ✅ 复用 PP 拓扑 |
| 与 A3 expert-streaming | ✅ 已协同 | 冲突（重排图） | **✅ 正交协同**（EP 减 gather 量，A3 管 IO/overlap） |

**协同结论**：**保留 layer-slicing 为骨架（PP），在其上叠 EP（expert-split）降每台 SSD 流量，
A3 streaming 管 IO/compute overlap，copy-spec 管有效 t/s。** 四者正交。
**不引入经典 TP 切 attention/dense**（与 layer-slicing 冲突且 TCP 延迟下负收益）。

EP 与 expert-streaming 的精确关系：EP 决定"这台该 gather 哪 k 个专家"，A3 streaming 决定"这 k 个
专家的字节怎么从 SSD/远端最快拿到"。EP 让 A3 的工作量减半，A3 让 EP 的减半真正变成时间收益。

---

## 5. 风险与物理墙（诚实说明）

| 风险/墙 | 性质 | 说明 |
|---|---|---|
| **W3 backbone 带宽墙（10–18 t/s）** | 物理墙 | 单前向 decode 的硬顶，TP/EP 都越不过；30 t/s 必须走有效 t/s |
| **W2 SSD 带宽墙** | 物理墙（可缓解） | EP 减半流量 + 池化命中是唯一缓解手段，但受盘速不对称约束 |
| **TCP-over-TB ~300µs 延迟** | 可改代码/硬件墙 | 经典 TP 的死穴；除非上 RDMA（EXO/JACCL 路线，需 TB5 + 自研 RDMA，超出当前范围），否则经典 TP 无解 |
| **AR drain 开销（占 40-45%）** | 可改代码 | signal+flush+host_wait 链；AR 融合（86→43）是主要优化 |
| **盘速不对称 staging 净亏** | 实测约束 | worker←mini staging 已实证 -18%（project.md §1）；EP owner-map 要避免让慢盘多拿 |
| **owner-map 缓存抖动** | 可改代码 | 当前 slot 序切 owner 随路由变 → 缓存命中不累积；hash owner-map 修复 |
| **bit-exact 上限** | 正确性闸 | project.md §1.5 已实证 q2 双机触及 bit-exact 上限；EP 的 AR-SUM 重组必须保持 ~1e-6 drift（row-parallel 固有），过 `--dump-logprobs` parity |

**一句话诚实结论**：在本项目（IO-bound、TCP-over-TB、MoE-on-SSD），**经典 TP 是错的工具**
（MLX 实测连 RDMA 下都只赢 2.3%，TCP 下负收益）；**正确的是把已落地的 TP skeleton 当 2-way EP 用**，
靠"两机各 gather 一半专家 + RAM 池化"降 SSD 流量，现实把 5.36 稳在 ~5 并为 copy-spec 抬基线。
**30 t/s 不由这条线交付，由编程域有效 t/s 交付。**

---

## 来源链接（调研引用）

- Megatron-LM（TP 理论）：https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf
- AWS Neuron TP 概述：https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tensor_parallelism_overview.html
- **MLX TB5 RDMA 实测 PP vs TP（核心证据）**：https://github.com/ml-explore/mlx/discussions/2990
- EXO（RDMA/JACCL，TP 需 RDMA）：https://github.com/exo-explore/exo
- TensorRT-LLM EP：https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html
- vLLM DP-attention + EP：https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/
- LMSYS 大规模 EP（96×H100 DeepSeek）：https://www.lmsys.org/blog/2025-05-05-large-scale-ep/
- Red Hat Wide-EP（DeepSeek-style MoE）：https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep
- prima.cpp（消费级 PP + mmap offload + 投机）：https://arxiv.org/abs/2504.08791
- llama.cpp RPC 基准（10GbE/层/token）：https://github.com/kjaiswal/llama-cpp-distributed-benchmarks
- int8/fp8 all-reduce 通信压缩：https://arxiv.org/pdf/2412.01152 ・ https://arxiv.org/pdf/2310.18313
- Thunderbolt 带宽：https://macperformanceguide.com/blog/2019/20190128_1352-understanding-Thunderbolt3-bandwidth.html

> 任务拆分见 `tensor-parallel-tasks.md`。
