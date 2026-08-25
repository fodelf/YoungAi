# tensor-parallel-feasibility.md — TP/EP 改造可行性深度复核(对抗性裁决)

> 本文档是对 `tensor-parallel-design.md` / `tensor-parallel-tasks.md` 的**对抗性可行性复核**。
> 方法:5 个互不通气的维度(D1 数字核验 / D2 延迟墙真伪 / D3 减半-IO 机制 / D4 拓扑与内存墙 / D5 历史实测约束)
> 各自尝试证伪前一轮裁决 → 综合 → 红队复核。所有结论带 `file:line` 证据,均为读码 + 读 `notes/execution-log.md` 实测 + 联网,
> **未跑任何加载模型脚本**(内存安全铁律)。

---

## 0. TL;DR — 裁决

**经典 TP(切 attention/MLP compute)主路:no-go。** 定性方向(经典 compute-split 在双机 Mac 非主路、EP 才打 W2 SSD 墙)**对**;
但 `tensor-parallel-design.md` 的**全部三档 t/s 预估不可信**,因为它们建在两个被代码与实测直接证伪的假设上:

1. **「EP 叠在 layer-slicing 骨架上、正交协同」——代码证伪。** TP(`g->tp`/expert-split)与 layer-slicing(`active_layer_slice`)是
   `ds4_distributed.c:10354-10364` 的**硬互斥 if/else**:`tp_enabled ⇒ load_slice=false`(全模型加载),`layers.set` 走另一分支。
   当前 5.36 t/s 跑的是**纯 layer-pipeline(PP)+MTP,完全没有 TP**(`tools/mtp_pipe_q2_speed.sh:2,51,55`)。EP 代码全部门控在
   `g->tp` 上,layer-slicing 路径下 `g->tp==NULL` ⇒ EP 全短路。两者当前不可共存。
2. **「每台 backbone 不变、只减半专家流量」——内存墙翻盘。** TP 模式每台**必须常驻全 43 层 backbone ≈8.4GiB**
   (实测 backbone=8.81GB,`notes/execution-log.md:35-37,133`;注释 `ds4_distributed.c:10355-10358`「each machine must load the full model」),
   是 layer-slicing 每台 4.07/4.4GiB 的**约两倍**。加 KV(200K~0.9-1.4GiB)+ 运行时(~2GiB)≈ **11-12GiB,顶满 12/12GiB 看门狗红线**,
   专家缓存空间反被挤小 ⇒ 名义减半的专家 SSD 流量被 backbone 常驻翻倍**净亏抵消甚至倒挂**。

**EP(`DS4_TP_EXPERT_SPLIT`)本身的真实成绩单**(已被真跑,wave-72/73,2026-06-18):
**冷态 smoke/long-context 域 1.02 → 1.67 t/s(1.64×)**,不是文档的「现实 4.6-5.4」。把这个**冷态 IO 域**机制的收益挂到
**暖态 compute 域 5.36 基线**上,是全文最大范畴错位(EP 进暖态实测**腰斩到 ~2**,copy-spec-in-TP code-edit=2.10 << 单机 3.77,`:3814`)。

**红队补的两条更优路径(文档完全没提):**
- **纯速度目标下 k16 全驻留(13GiB,零 SSD IO,理论 10-18 t/s,受 W3 不受 W2)物理上碾压整个 TP/EP 分析**(EP 冷态封顶 1.67)。
  代价是**质量**(16/256 专家剪枝,路由命中 95.6%),是 quality tradeoff 不是 speed 墙。EP 只在「**必须保 256 专家质量**」约束下才相关。
- W2 墙在 **layer-slicing 真拓扑**下早被 remote expert-fetch(P2.2)/worker staging 攻击过且**物理证伪**(盘速不对称,wave-33/34,
  `notes/execution-log.md:2599-2613`)——这比「TP backbone OOM」是更直接相关的 no-go 论据。

**一句话:** TP 这条线对「双机 30 t/s」不仅无贡献,连维持 5.36 都有内存倒退风险。**判 no-go(主路)**,仅留一条需同时过
**内存墙 + 慢盘墙 + 质量墙三关**的有条件探索(backbone 重量化后的对称拓扑)。

---

## 1. 被证实的部分(文档对的地基)

| 项 | 文档主张 | 复核 | 证据 |
|---|---|---|---|
| 模型维度 | 43 层 / n_embd=4096 / n_expert=256 / top_k=6 / shared=1 / n_ff_exp=2048 / MLA(n_head_kv=1) | ✅ 逐字正确 | `ds4.c:158-172` |
| 每专家字节 | 6.75 MiB(gate 2.06 IQ2_XXS + up 2.06 + down 2.625 Q2_K) | ✅ 精确 | `ds4.c:322-348, 2855-2868, 11107-11110` |
| 每 token routed IO | 43×6×6.75MiB ≈ 1.70 GiB | ✅ 精确(=1.7007 GiB) | 同上;总 routed 72.56 GiB 与 W1 墙自洽 |
| AR 单次通信量 | 4096 float = 16 KiB | ✅ | `ds4.c:11250/11343`, `DS4_N_EMBD=4096` |
| expert-split 减半字节 | 每台只 gather k=3 专家 → 冷读字节减半,bit-exact | ✅ **硬事实非估计**(cold_mib 40.5→20.2 实测) | `ds4.c:11159-11192`; `ds4_metal.m:19834,16493,15176`; `:3790` |
| 经典 TP 非主路 | layer-slicing/PP 优于跨机 compute-split | ✅ 方向正确 | wave-74 代码级否定跨机 compute-split |
| W3 backbone 墙 | 单前向 10-18 t/s,30 t/s 须靠 copy-spec 有效 t/s | ✅ 与历史一致 | `:3573,3732,3734` |

**减半机制的对抗性证伪全部失败(即减半是真的):** ① 同 token 同层 top-6 做 slot 二分 `[0,k)/[k,6)`,两段映射的 expert-id
**结构上不相交**(top-k 无放回,`metal/argsort.metal`),不存在「两台 stream 相同专家致总 IO 不降」;② shared expert 走独立 Q8 dense 路径
不污染 routed 减半账;③ 减半来自**静态 slot view 无命中依赖**,owner-map/池化只是叠加的二阶增益,不成立也只是回到「每台冷读 k 专家」仍减半。

---

## 2. 被证伪 / 修正的部分(逐条,带证据)

### 2.1【fatal·拓扑】「EP 叠在 layer-slicing 骨架上,正交协同」
- **真相:** `ds4_distributed.c:10354-10364` 硬互斥——`if (tp_enabled){load_slice=false; load_output=true;} else if (dist_enabled){load_slice=true; ...}`。
  注释 `:10258-10260`「Tensor-parallel mode loads the whole model on every peer … does not take a --layers range」。
  5.36 跑的是纯 layer-pipeline(`tools/mtp_pipe_q2_speed.sh:2` 标题「层切分(layer-pipeline)+MTP」,`:51,55` `SPLIT_COORD=0:19/SPLIT_WORKER=20:42`,从不传 `--tp`)。
- **冲击:** 文档 §4「EP 同 PP(EP 在层内做,复用 PP 拓扑)、正交协同」不成立。要兑现需改 C 源码让 `g->tp` 与 `active_layer_slice` 共存(2D 并行),铁律禁改码。
- (红队精确化:`--tp + --layers` 组合实际**未被 validation 禁止**,`ds4.c:19514` 会让 `layers.set` 把 `load_slice` 改回 true,但 TP run loop
  `ds4_distributed.c:10393-10399` 整体假设「全模型 eval over full layer range」⇒ 该组合是**语义不 coherent 的 latent bug,不是被拒**。结论不变,论据应这样表述。)

### 2.2【fatal·内存】TP 模式每台常驻全 backbone,顶 12GiB 红线
- **真相:** TP `load_slice=false` ⇒ 每台全模型加载。backbone(非 routed-expert)实测 **8.81GB**(`:35-37,133`:attention Q8 5.7GB + shared Q8 1.1GB + output Q8 0.56GB ≈7.4-8.4GiB;`project.md:25` 标 8.4GiB)。
- **冲击:** layer-slicing 每台只 4.07/4.4GiB(`project.md:34,46`);TP 翻倍到 8.4GiB,+KV+运行时 ≈11-12GiB 顶满看门狗。专家缓存被挤 ⇒ **更多专家落 SSD,内存上比 layer-slicing 更差**。文档三档 t/s 完全没做这项核算。
- **红队加重:** 看门狗对 mmap RSS 可能失效(MEMORY task3「footprint 看门狗对 mmap 失效」)⇒ TP 全模型 mmap 的 OOM「12/12 红线兜底」可能是**纸面保护**。

### 2.3【major·W2】「减半字节 → 等效带宽翻倍 → ~7 t/s」
- **真相:** 减半字节为真,但**翻倍被 per-tp-layer AR 强同步屏障证伪**。`ds4.c:11222-11253` 每个 tp-layer 都要 `signal+flush+host_wait`(drain)
  + 阻塞 `ds4_dist_tp_allreduce_f32`(`ds4_distributed.c:1827-1884` blocking exchange)⇒ **43 层 lockstep**,临界路径 = `max(peerA, peerB) + net`,**不是两盘字节之和/聚合带宽**。
  盘速不对称(mini ~1.9GB/s vs MacBook 8-21GB/s,**~4-10×**)使慢盘封顶。实测 43 层 AR ~291ms/token 吃掉约一半 IO 收益(`:3811`),**EP 冷态封顶 1.67 t/s**,理想 2× 也只 ~2.0。

### 2.4【major·W3】暖态 EP 是净负,5.36 与 TP/EP 无关
- **真相:** 5.40/5.36 = **单机暖态** layer-pipeline + copy-spec + spec-pipe + MoE-thin(topk=4)(`:3830,3699-3706`),暖态前向 compute 天花板 ~118ms/token ≈ 8.5 t/s(`:3639`)。
  暖态 compute-bound 下 TP 只加 AR 不省 compute ⇒ copy-spec-in-TP code-edit **2.10/0.88 << 单机 3.77**(`:3814`)。
- **冲击:** 文档把「现实 4.6-5.4」并列在暖态 5.36 旁会被读成「EP 维持 5.36」,与实测**直接冲突**——EP 进暖态是腰斩。

### 2.5【minor·延迟】「TCP ~300µs RTT、86×300µs=25.8ms、AR drain 占 40-45% 可优化」
- **真相:** 300µs 是**二手业界数字**(Apple TN3205 RDMA-vs-TCP,被博客转引;RDMA 压到 <50µs),非本项目实测。本项目 wave-71
  `tools/tp_rtt_probe.c` 雷电直连 20000 iters **实测 p50:8B=84µs / 16KiB=96µs / 64KiB=140µs**(`:3721-3727`)。用 140µs 重算 86 syncs ≈ **12ms**,不是 25.8ms。
- **真瓶颈不是 TCP RTT:** AR 实测拆解 drain=1.9ms/层 + net=7.4ms/层(net 大头是 **lockstep skew** 不是 TCP,典型 allreduce 仅 0.1ms),
  **AR total 占 decode <1%**(`:3782-3783`)。「drain 占 40-45%」是占 **AR 内部**非占 decode ⇒ 砍 AR 拿不到 40% 提速。
- **相位归因错:** 「86 = attn-AR + MLP-AR」错——本代码 **attention 无任何 TP all-reduce**,decode 真实 AR = **EP-only 43** 或 **EP+SHARED_SPLIT 86**,
  构成是 MoE-AR + sharedFFN-AR(`ds4.c:11250,11343`;全文仅 3 处 allreduce);86 仅两 split 全开时数值巧合。
- **DS4_TP_FUSE_AR(86→43 融合,P1.3):** grep 全文**无此 env 实现**,纯纸面;且只砍 net 次数对真大头 drain 无效。

### 2.6【证伪·乐观叠乘】「RAM 池化命中 40-60% 再降落盘」
- wave-76(`:3856`)「pool 救不动 / code-edit 无 IO 可省」+ MEMORY `feedback_stability_over_limits`(显式 RAM 专家缓存
  hard_copy/mlock **饿死 OS page cache 净变慢 3.84→1.84**)**直接证伪**。乐观档把已证伪的池化收益乘了上去。

### 2.7【降级】DP-attention + EP-MoE(§2.2 / P2.1)从「P2 可探索」→「已被代码级否定」
- wave-74(`:3836-3844`)已**代码级否定**跨机 compute-split(attention head-split):层间顺序依赖 + per-layer 跨机 stall 失去 pipelining =
  **结构性 GPU 流水中断硬件互联墙**(`host_wait` 已是快路径 <50µs,故 overhead 是流水中断不是 wait 延迟)。照文档做会重复撞已知死墙,除非换 RDMA。

---

## 3. 物理墙清单(本方案要过的关)

| 墙 | 内容 | 对 TP/EP 主路 | 对「重量化后对称拓扑」唯一保留路径 |
|---|---|---|---|
| **拓扑墙** | TP 与 layer-slicing 硬互斥,EP 门控在 `g->tp` | fatal,需改码做 2D | 改码后可解 |
| **内存墙** | TP 每台全 backbone 8.4GiB,顶 12GiB(看门狗对 mmap 可能失效) | fatal | 需 Q8→Q5 重量化把每台压回 ≤6GiB |
| **W2 慢盘墙** | 43 层 lockstep,mini 1.9GB/s 恒定短板,减半的那半仍受慢盘限速 | major,封顶 ~2 t/s | **同样致命——重量化解内存不解慢盘** |
| **W3 backbone 墙** | 单前向 10-18 t/s,暖态 8.5 | 30 t/s 不由此线交付(一致) | 同 |
| **质量墙** | Q8→Q5 backbone 与 AProjQ8 高精度意图冲突,1M 长程召回退化未知 | — | **未评估,可能使该路径质量上也不可行** |

---

## 4. 修正后的三档 t/s(拆冷态/暖态,基于实测)

> 前提:三档全部**先要解开 TP/PP 互斥(改码)且每台 backbone 能塞进 12GiB(当前撞墙,需重量化)**——否则连启动都过不了内存闸。

- **乐观:冷态 smoke/long-context ~2.0 t/s。** 非对称 `SPLIT_LOW` 压掉 lockstep skew 到理想 2× IO 减半(490ms/token),AR 开销忽略,
  **不含任何池化/copy-spec 叠乘**(已证伪)。这是 wave-73 实测 1.67 的工程上界。绝非文档的 6-8。
- **现实:冷态 1.3-1.7 t/s**(EP/单机冷 1.02 的 1.25-1.64×,实测);**暖态 code-edit:EP 净负,维持单机 5.36 不上 TP**。
  文档把两域混成一个「4.6-5.4」是范畴错位。
- **悲观:撞内存墙直接 no-go / 倒退 <1 t/s。** 每台 backbone 8.4GiB 顶红线,专家缓存被挤,decode 比 layer-slicing 倒退甚至 L1 预算闸 abort。
  即维持 5.36 都有内存倒退风险。
- (红队保留意见:8.4GiB 是 mmap 名义值,MTLResidencySet 预算后实际 wired 可能更低;「<1 t/s」硬结论应先由 **P0 静态内存闸**核算坐实,而非直接断言。)

---

## 5. 红队补的更优路径(文档完全没提)

1. **k16 全驻留(纯速度目标的真答案):** `ds4flash-k16-v2.gguf` ≈13GiB,同 q2 量化族,双机全驻留 ⇒ **零 SSD IO**,理论 **10-18 t/s**(受 W3 backbone 墙而非 W2)。
   `project.md:43,292` 自标 P4 决策门备选,`notes/execution-log.md:21,31-32` 实证「内存非约束、第二台只为容量」。**对纯速度,这条 physically dominates 整个 TP/EP 分析。**
   代价是**质量**:16/256 专家剪枝(路由命中 95.6%),是**已知量**的 quality tradeoff(相比之下 Q8→Q5 重量化的质量影响是未知量)。
2. **W2 在真拓扑(layer-slicing)下已被攻击且证伪:** remote expert-fetch(`DS4_DIST_EXPERT_FETCH_SERVE`,`ds4.c:19601`,`ds4_distributed.c:10760+`)
   + worker staging 在 wave-33/34 实测净负(code-edit 2.83 降 / smoke 1.75 大降),根因 = **盘速不对称(mini 顺序盘喂不饱 worker)是物理结论非参数问题**(`:2599-2613`)。
   即 EP-under-TP 即便能跑也撞同一面慢盘墙。

---

## 6. 裁决与 P0 重排(替换 `tensor-parallel-tasks.md` 的 P0)

**go/no-go = no-go(经典 TP 主路)。** EP 仅在「必须保 256 专家质量」约束下作为冷态域(smoke/long-context)有限杠杆,且须先过下列死结门:

1. **P0-内存闸(新增·最高优先,替代原 P0.1):** 静态核算 TP 模式每台峰值 = 全 backbone(MTLResidencySet 预算后**实际 wired** 量,非 mmap 名义 8.4GiB)
   + KV(200K~0.9GiB)+ 运行时,并验证**看门狗对 TP 全模型 mmap RSS 是否真能拦截**(MEMORY task3 记其对 mmap 失效)。`>12GiB` ⇒ TP 主路直接 no-go,后续全废。**纯预算闸,不跑模型。**
2. **P0-拓扑可行性(新增):** 确认不改 C 源码能否让 EP 脱离 `g->tp` 全模型加载而存在。当前 EP 门控在 `g->tp ⇒ load_slice=false`,即「layer-slicing 上叠 EP」**必须改码**;若铁律禁改码,EP 主路本轮不可执行,原 P1.* 全建在错误前提。
3. **P0.2(降级):** EXPERT_SPLIT on/off A/B **必须新增「暖态 code-edit 域」列**(历史只冷态 smoke 验过)。先证暖态 EP 不是净负(唯一数据点 2.10<<3.77)再谈 P1。依赖前两门通过 + 跑脚本授权 + 内存安全。
4. **P0.1(降级为次要,基本已闭合):** E0 RTT 应引 `tp_rtt_probe.c`(wave-71 已实测 p50 84-140µs)而非 `tp_bw_probe.c`(测吞吐)。实测已证 RTT **非瓶颈**,不应作为整方案闸门。

---

## 7. 对前一轮两份文档的具体修正清单

- `design §2.0/§4/§0`:删除「EP 在 layer-slicing 骨架上叠加、正交协同」——代码证伪(硬互斥)。注明 5.36 是纯 layer-pipeline 无 TP。
- `design §3.4` 三档表:**全部数字作废重写**。乐观 6-8→冷态 ~2.0;现实 4.6-5.4→拆冷态(1.3-1.7)/暖态(净负、不上 TP)两列;悲观→内存墙 no-go/<1;新增「每台 backbone 8.4GiB 顶 12GiB」翻盘前提。
- `design §1.5/§3.1-3.3`:「86 次 = attn-AR+MLP-AR」改为「attention 无 TP AR,真实 EP-only=43 或 +SHARED_SPLIT=86(MoE-AR+sharedFFN-AR)」;「25.8ms 同步天花板」用错的 300µs+86,实测 RTT 140µs 且 AR 占 decode<1%。
- `design §1.3/§3.2/§5`:「TCP ~300µs」注明非本项目实测(实测 84-140µs);延迟杀手论删除,真大头是 drain+lockstep skew;来源 `tp_bw_probe.c`(吞吐)改引 `tp_rtt_probe.c`(延迟)。
- `design §2.1/§5`:「AR drain 占 40-45%」注明是占 AR 内部(AR 仅占 decode<1%),非 40% 整体提速。
- `design §3.3/§3.4 乐观档`:删除「RAM 池化命中 40-60% 叠乘」(已证伪,显式缓存净变慢)。
- `design §2.2 / tasks P2.1`:DP-attn+EP-MoE 从「P2 可探索」降级为「wave-74 已代码级否定,除非换 RDMA 否则死路」。
- `design §2.1 / tasks P1.3`:`DS4_TP_FUSE_AR` 标「纸面未落地(grep 无实现),且对真大头 drain 无效」,优先级低于非对称切分(已实证 1.64×)。
- **新增章节**:① k16 全驻留作为纯速度目标的 dominant 备选(质量 tradeoff 已知);② W2 在 layer-slicing 真拓扑下已被 remote-fetch 证伪(慢盘墙)。

---

## 8. P0 静态门坐实结果(2026-06-20,不跑模型 / 不改码)

> 应「先跑 P0 门坐实再定」要求,把两道**纯静态就能给 go/no-go**的门(P0-拓扑、P0-内存闸)读码坐实。
> 需跑脚本的 P0.1(RTT)/P0.2(EXPERT_SPLIT A/B)留待授权 + 内存安全闸。

### P0-拓扑门:**坐实 = 互斥,不改码不可叠加(NO-GO 前提)**
- `ds4_distributed.c:10354-10364`:`if(tp_enabled){load_slice=false; load_output=true;} else if(dist_enabled){load_slice=true; ...}` 硬互斥。
- backbone/expert span 在 `ds4.c:3619-3640` 区分(shared expert 算 backbone,routed experts reclaimable);`load_slice` 决定 backbone span 是「本台那段层」(PP)还是「全 43 层」(TP)。
- EP/AR 全门控在 `g->tp && il<g->tp_layers`,而 `g->tp` 仅 `tp_enabled` 分支赋值 ⇒ layer-slicing 路径下 `g->tp==NULL`,EP 全短路。
- **结论:** 「layer-slicing 骨架上叠 EP」必须改 C 源码做 2D 并行;铁律禁改码 ⇒ **EP 主路本轮不可执行**。

### P0-内存闸:**坐实 = L1 启动放行,但运行时看门狗在长上下文必杀**
- **看门狗对 mmap 有效(反驳 `task3` 旧记忆)**:`ds4.c:724-727` 看门狗采样 Mach **`phys_footprint`** 非 ps/rss,注释明写「ps/rss is blind to Metal no-copy mmap residency: 9.3GiB resident reads as ~38MiB」⇒ phys_footprint **能看见实际 wired 工作集**。90% 红线触发 `_exit(137)`(`ds4.c:834-841`)。即 TP 冲红线是**硬 abort 杀进程**,不是悄悄 page-thrash。
- **L1 启动闸只查 backbone、不含 KV**:`ds4.c:19680` 调用 `ds4_l1_budget_gate(base_l1_resident_bytes + mtp_l1_resident_bytes, 0)`(kv/scratch 传 0)。TP 模式 backbone ≈8.4GiB < 85%×12GiB=10.2GiB ⇒ **启动放行**(不会 refuse)。
- **运行时峰值(phys_footprint)核算 @ DS4_MEM_BUDGET_MB=12288**:
  - @ **1M ctx**(项目铁律要保):backbone 8.4 + KV 4.56(FP8 attn lever-A landed,`dsv4_fp8_kv_quantize` 在码;memory `lever_a_landed_state`)+ scratch/prefill ~1.5 ≈ **14.5GiB ≫ 10.8GiB(90% 红线)⇒ 运行时必被看门狗杀**。
  - @ **200K ctx**:8.4 + KV ~0.9-1.4 + scratch ~1.5 + expert gather ~0.3 ≈ **11.1-11.6GiB**,贴/越 90% 红线 ⇒ **高危**,且 backbone 8.4GiB 把专家 page cache 从 PP 的 ~5-6GiB 挤到 ~1-2GiB ⇒ 专家冷读命中暴跌(同 `feedback_stability` 实测「显式缓存饿死 page cache 净变慢 3.84→1.84」),EP 减半收益被吃掉。
- (诚实保留:8.4GiB 是 backbone span 名义,实际 wired 受 `DS4_METAL_BACKBONE_MLOCK_BUDGET_MB` 默认 4608MiB cap(`ds4_metal.m:5266`)+ MTLResidencySet reclaimable hint 调控;但 decode 持续 touch 全 backbone ⇒ faulted resident 仍趋近 8.4GiB。精确峰值需 P0.2 用 `DS4_PROFILE` 读 phys_footprint 实测印证,属需授权的跑脚本门。)

### P0 静态裁决
- **TP 主路 @ 1M ctx = no-go**(运行时看门狗必杀)。**@ 200K ctx = 贴红线 + 专家缓存饿死,EP 减半收益倒挂。**
- 叠加 P0-拓扑门(不改码不可叠加)⇒ **EP 主路在「保 1M ctx + 禁改码」双约束下,本轮静态判 no-go。**
- 唯一能松动的前提:backbone Q8→Q5 重量化把每台压回 ≤6GiB(解内存墙),但仍需另过 **慢盘墙 + 质量墙**(见 §3),且重量化与 AProjQ8 高精度意图冲突。
- **剩余门(需授权跑脚本):** P0.2 用 `DS4_PROFILE` 实测 TP 模式 phys_footprint 峰值印证上述核算;P0.1 用 `tp_rtt_probe.c` 复确 RTT(已知 84-140µs 非瓶颈)。两者均须过内存安全闸 + 单独授权。

---

## 9. 与 `react-go-opus46-design.md` 量化模型对齐(2026-06-20,读码坐实 / 不跑模型 / 不改码)

> §0-§8 的全部裁决建在**旧 81GiB 全模型 + 全程 SSD 专家流式**假设上。`react-go-opus46-design.md` 换了模型(§3.6 精度倒置 hot-resident 2-bit / §3.7 Mode P/G 双模式 / §3.8 hybrid GGUF+HF)。**模型一换,墙就换**——本节把 TP/EP 裁决在新模型下重判。§0-§8 不作废,只加限定域:其 no-go 对「旧 81GiB SSD-bound」成立;Mode P(hot-resident compute-bound)是新域,适用本节。

### 9.1 两个被 react-go 直接改写的前提
- **「铁律禁改码」放宽。** react-go 自己就规划改码:`deepseek4-quantize --experts-hot-mask`(§3.6.6)、keep-map 屏蔽内核修复(§3.7.3)、safetensors 冷档 loader(§3.8.3)。⇒ §0/§6/§8 用「禁改码 ⇒ EP/2D 不可执行」做的 no-go,前提在 react-go 工程里**不再成立**(它本就是改码项目)。
- **W2 慢盘墙在 Mode P 消失。** hot experts 常驻、热路径零 SSD IO(§3.5.4 / §3.5.1)。⇒ §2.3/§3/§5 赖以判 EP「冷态封顶 1.67」的那面 W2 墙,在**优化目标域(Mode P)根本不存在**。

### 9.2 角色反转(本次对齐的核心结论)
decode 瓶颈换了,TP 与 EP 的价值**互换**:

| | 旧世界(81GiB,SSD-bound) | 新世界(Mode P,hot-resident,compute-bound) |
|---|---|---|
| decode 瓶颈 | W2 专家 SSD IO(1.70 GiB/token) | W3 单前向 compute(~118ms/token≈8.5 t/s,§2.4 / `:3639`) |
| **TP**(并行 compute) | 无用——IO 限,并行算力不省 IO ⇒ §0 判 no-go**对** | **唯一攻击新瓶颈的拓扑杠杆**(并行算力直接砍 compute) |
| **EP**(减半冷读字节) | 唯一相关杠杆(冷态 1.02→1.67) | **收益归零**——无 SSD IO 可减半 |

⇒ feasibility 全文的「surviving lever = EP / dead main path = TP」结构,**在 Mode P 整个反转**:EP 退场,TP 上场。

### 9.3 react-go §3.5.1「常驻 → 30 t/s」的算账缺口
§3.5.1 只扣了 SSD IO(33ms 预算 − compute = 专家须 ≤50MiB),**没核单前向 compute 本身**。暖态单前向 compute 天花板 = ~118ms/token ≈ 8.5 t/s(§2.4)。⇒ **零 SSD IO 也只到 ~8.5 t/s,离 30 还差 ~3.5×**。缺口必须靠 **compute 加速**补:TP 并行(~2×)+ MTP/copy-spec 有效 t/s(§3.5),不是单靠常驻。**这正是 TP 在新模型下重新相关的根因。**

### 9.4 但 TP 在 Mode P 撞的是内存布局墙(不是慢盘墙)
- 现 `DS4_TP_EXPERT_SPLIT` 是 **slot-split 非 storage-split**:`ds4.c:11162-11169` view `router_selected[slot_start:cnt]`,两台都 mmap **全**专家文件,只各 compute/touch 自己 slot 的专家。任一专家都可能落进本台 slot ⇒ 多 token 后两台 page-cache **各自填满全部 hot experts 9.1 GiB**,**slot-split 不缩驻留工作集**。
- TP `load_slice=false`(`ds4_distributed.c:10358`「each machine must load the full model」)⇒ 每台 = 全 backbone 8.4 + 全 hot experts 9.1 ≈ **17.5 GiB/台 ≫ 12**。
- PP(layer-slicing)层切 **backbone 4.2 + 本层 hot 4.5 = 8.7 GiB/台**——**正是 react-go §3.8.2 自己选的布局**。
- ⇒ **hot-resident 布局结构上要求 PP;TP 在 Mode P 直接装不下。** §2.2 内存墙在新模型下不但成立,而且更紧(多了 9.1 GiB 常驻热专家压在全 backbone 上)。

### 9.5 解锁 TP-for-Mode-P 的充要改造(命名,排序靠后,不投机先做)
要让 TP 的并行算力(唯一能动 8.5→30 缺口的拓扑杠杆)在 Mode P 落地,须**同时**满足三件:
1. **backbone Q8→Q5**(8.4→~5.3;§3「重量化」条件路径,带 1M 长程质量墙未测风险);
2. **真 expert-STORAGE-split**(每台存**不同**半数 hot experts 4.5,router 按 owner 路由)——区别于现 slot-split 全复制;附带 router-to-owner 映射 + 负载不均(token top-6 全落一台则另一台空转)新码;
3. **AR drain 优化**(MTLSharedEvent 快路,memory `metal_wait_anukari_precedent`;`ds4.c:11229-11233` 已是 signal+flush+host_wait 快路雏形)。
- 凑齐 ⇒ 5.3 + 4.5 = **9.8 GiB/台装下**;Mode P 无 disk-skew(§9.1),AR 仅剩 compute-imbalance skew(M4/M1 算力差,可用 `DS4_TP_SPLIT_LOW` 非对称切平衡,`ds4.c:11155-11161`)⇒ 理论 **~14-17 t/s**(单前向 8.5 的 1.6-2×)。**仍 <30**,须再叠 MTP/copy-spec。

### 9.6 对齐后裁决
1. **Mode P 主路 = PP(layer-slicing),与 react-go §3.8 一致、已实现(5.36 跑在它上)、装得下 ⇒ 先交付。**
2. **TP 改判:不再是「双机零贡献」**,而是 Mode P compute 天花板的**唯一拓扑杠杆**;但解锁前置 = storage-EP + backbone-Q5 + AR 优化三件套,封顶 ~17 t/s。**排序:在「PP-Mode-P 跑通并实测 Mode P 真实 compute 天花板」之后再评**,不先建(投机违 `feedback_evidence_driven`)。
3. **EP(`DS4_TP_EXPERT_SPLIT`)降级为 Mode G 限定**:Mode P 收益归零;仅 Mode G(冷 FP16 流式)还原其冷读减半价值,但 Mode G 是 by-design 次要慢档(~2 t/s,§3.7),非优化目标。
