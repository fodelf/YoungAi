# DS4 Execution Log

Project-local execution log. Every design memo / atomic patch / validation step /
material scope decision / blocker gets a dated entry here (per standing rule).

---

## 2026-05-30 — Dual-host 200k / 20 t/s coding plan — investigation + physics (NEW plan, pre-params)

**Context.** New goal, distinct from the single-machine 1M-ctx roadmap (Levers A/B/C).
Two machines: Mac mini M4 16 GB (this host, budget ≤12 GB) + a MacBook 16 GB (budget ≤8 GB).
Target: 200k ctx, **20 t/s**, Claude Code coding usage, prefer the q2-derived model, and
above all **do not crash memory** (restraint over speed). Per `feedback_new_plan_forget_old_verdicts`,
old NO-GO verdicts (X9, cb-floor) do not veto this; re-derived from scratch.

**Hardware confirmed (this host).** Mac mini Apple M4, 10-core (4P+6E), 16 GB, ~120 GB/s
theoretical mem bandwidth. `bridge0` Thunderbolt Bridge interface present (interconnect candidate).

**Model inventory (`gguf/`).**
- `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` = **80.76 GiB** (full q2; does NOT fit 20 GB combined → SSD paging → ~1 t/s, rejected).
- **`ds4flash-k16.gguf` = 12.73 GiB** ← routing-concentration prune, 16 of 256 experts/layer resident, 95.6% routing hit (commit 78151b0 "feat: k16版本"). **The only model that fits combined RAM with no paging.**
- `ds4flash-k48.gguf` = 21.80 GiB (48 experts/layer; exceeds 20 GB combined budget once KV+overhead added → rejected for these budgets).
- `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` = 3.54 GiB (MTP draft model — the 20 t/s lever).

**Flash architecture (ds4.c DS4_SHAPE_FLASH).** n_layer=43, n_embd=4096, n_hc=4 (mHC, 4 residual
streams), MLA n_head=64 × head_dim=512, n_expert=256, n_expert_used=6, n_expert_shared=1,
n_swa=128, indexer top-k=512, n_vocab=129280.

**Memory budget (k16, no paging).** Per-layer k16 resident ≈ 12.73 GiB/43 ≈ 296 MB.
mini ≤12 GB holds ~26 layers (7.7 GiB) + KV slice + runtime; macbook ≤8 GB holds ~17 layers (5.0 GiB).
Both fit their share with headroom → **memory is NOT the binding constraint for k16**; the second
machine is needed for capacity (k16 12.73 GiB > 12 GB mini budget), not for speed.
200k KV (compressed attn): ~0.9–1.4 GiB total, split across hosts — trivial.

**Decode physics (the 20 t/s wall).** Active read per token ≈ **9 GB**:
attention Q8 ~5.7 GB (dominant, dense every layer) + shared-expert Q8 ~1.1 GB + 6 routed Q2 ~1.8 GB
+ output-proj Q8 ~0.56 GB. **Pipeline (layer-slice) parallelism splits MEMORY, not single-stream
decode latency**: token flows mini→macbook serially, t = Σ(bytes_i / bw_i) ≈ 9 GB / ~100 GB/s
≈ 90 ms → **~11 t/s ceiling**. Two machines ≈ same t/s as one machine that could hold it all.
Inter-host wire = hidden state n_embd×n_hc = 4096×4 = 64 KiB F32/hop → negligible over Thunderbolt
(prefill streams 200k×64KiB ≈ 12.8 GB once, ~2.5–10 s depending on TB gen — acceptable).

**Paths to 20 t/s (only three physical levers).**
- A. **MTP speculative decode** (3.54 GiB draft already in repo) — exact, no quality loss; coding
  domain predictable; needs ~1.8× (11→20). Project currently labels MTP "experimental, slight
  speedup" — making it deliver 1.8× in a distributed pipeline is the core engineering risk
  (cf. FlowSpec / SpecPipe pipelined-speculative literature).
- B. Attention Q8→Q4 re-quant (lossy) — saves ~2.8 GB/token → ~15 t/s; +experts 6→4 → ~17 t/s.
  Violates AProjQ8 quality intent; needs re-quantization. Fallback only.
- C. Tensor-parallel attention (both hosts compute same layer simultaneously): 9 GB / (bw_mini+bw_macbook)
  ≈ 22 t/s — clears the gate, but 43 layers × 2 all-reduce/token requires **Thunderbolt 5 + RDMA
  (macOS 26.2+, µs latency)**; without RDMA the per-layer sync (~ms) kills it. Project implements
  pipeline only, not TP — major new work + mHC (n_hc=4) complicates the all-reduce.

**Honest verdict (pre-params).** Capacity + 200k + no-crash: **solved by k16 + 2-host pipeline**.
20 t/s: **NOT met by pipeline alone (~11 t/s)**; reaching it depends on MTP acceptance (lever A),
or lossy quant (B), or TB5+RDMA tensor parallel (C). Per the ≥20 t/s iron gate, no execution until a
credible ≥20 t/s path is fixed. Three decisive unknowns blocking the estimate: (1) MacBook chip
(bandwidth + TB5/RDMA capability), (2) Thunderbolt link status/gen, (3) speed-vs-quality tolerance.

**Status.** Investigation complete; design gated on user answers to the three unknowns. No model
load, no build, no validation performed (memory-safety + no-validation-without-permission rules).

### Update — params resolved (2026-05-30)
- **Interconnect: Thunderbolt direct bridge already up.** Good for pipeline; but...
- **Second host = MacBook M1 (base).** Mem bandwidth **68 GB/s theo (~50–55 real)** — half of M4.
  Interface TB3/USB4 (NOT TB5) → no macOS RDMA → **tensor-parallel path C is dead** (per-layer
  all-reduce at ms latency dominates). Pipeline-only.
- **Speed/quality posture chosen: MTP-priority, lossless** (no attention Q4 re-quant).

**Revised physics (M4 12 GB/~95 real + M1 8 GB/~52 real, TB3, pipeline, k16, 200k).**
Second host is a *capacity patch*, not a speed-up: M4 alone can't hold k16 (12.73 GiB > 12 GB
budget once KV+runtime added → crash risk), so M1 must take the overflow layers — and M1 is slow.
Load M4 to its memory limit (~32–35 layers), M1 takes the remaining ~8–11 + MTP tail + sampling.
- M4 ~35 layers: 35×0.196 + 0.56(output) = 7.42 GB / 95 = ~78 ms
- M1 ~8 layers: 1.57 GB / 52 = ~30 ms ; TB hop ×2 negligible
- **Base decode ≈ 108 ms → ~9–10 t/s.**
- MTP speculative (lossless, DeepSeek ships 1 MTP module → ~1.8–2.0× ceiling in predictable coding
  domain): **~16–19 t/s realistic-to-optimistic.**
- Stackable lossless micro-levers (vocab-trim output 129k→64k frees M4 RAM for 1–2 more layers off
  the slow M1; experts_used 6→5 coding-domain): nudge base ~9.3→~10.5 → with MTP graze ~19–20.

**Verdict vs the ≥20 t/s iron gate.** On M4+M1, **lossless, I cannot produce a solid ≥20 t/s
proof.** Honest ceiling ≈ **16–19 t/s** with MTP delivering ~1.8–2.0× AND M4 loaded to memory max
AND lossless micro-levers stacked. 20 is the optimistic edge, not a guarantee. Gate NOT cleanly
cleared → do not start build/execution until the user picks a posture:
(1) accept ~15–17 t/s realistic target, or (2) allow one small lossy concession (e.g. experts 6→4,
or attn Q6) to make 20 solid, or (3) faster second host. Capacity + no-crash + 200k remain SOLVED.

**Sources consulted (web).** llama.cpp MoE SSD-offload PoC (Qwen3-30B-A3B @ M1 Pro 16 GB ~13 t/s);
HOBBIT mixed-precision expert offload; EXO ring-pipeline over Thunderbolt (2-dev ~1.8×, RDMA µs
latency on macOS 26.2+); FlowSpec / SpecPipe pipelined speculative decoding.

---

## 2026-05-30 (later) — REDESIGN: 投机解码 + 张量并行 (TP) + k4 smoke (user-directed)

**Scope change (user).** New approved direction supersedes the pipeline-only plan above:
主架构 = **投机解码 (spine) + 张量并行 over Thunderbolt**；允许 attention 重量化 Q8→Q5/Q4 (质量门把关)；
最小集用 **k4**。明确"不复用 mine 分支的 replica/mtp-replica 架构"。Plan landed at repo `plan.md`.

**Re old TP verdict (line 50-53, 66-68 above).** 上一轮判 "TP path C 在 M1+TB3 无 RDMA = 死"。
那是**推测未实测**。按 `feedback_new_plan_forget_old_verdicts`，不让它否决，改用 **E0 实测**
(en5 ping-pong all-reduce RTT) 作 TP go/no-go 闸。TP 每 token ~86 次逐层同步；RTT≤50µs→可行,
≥150µs→回退单 M4+Q5+MTP。E0 是方案第一个动作。

**Exploration verdicts (this session).**
- **Verdict B — 必须物理重打包 k4，不能运行时 mask 全量。** `mac` 分支把专家张量整张(全256)
  wrap 成单个 MTLBuffer + 进 MTLResidencySet (ds4_metal.m:14052; ds4_gpu_wrap_model_range:5009;
  residency_request_views:424)。无逐专家 view-shrink、无 DS4_METAL_EXPERT_OFFLOAD。故运行时只用4个
  专家**不减** footprint。
- **离线工具可行。** gguf-tools/deepseek4-quantize.c 的 GGUF 读(load_gguf_metadata:1458)/写
  (write_full_gguf:1620) 是流式 fseek+fread (不 mmap)；每专家是连续字节切片
  (generate_one_expert memcpy at +xid*per_expert, :1240)。可写新工具流式切 top-K 专家、内存安全。
- **k16 keep-map schema 破译** (dump_gguf_meta.py)：`deepseek4.expert_count` 仍=256；新增
  `ds4.expert_keep_map.kept_counts:i32[43]`(=16) + `ds4.expert_keep_map.original_ids:i32[688]`
  (43×16 原始id)；专家张量 dim[2]=16；门控 ffn_gate_inp 仍 256 列；前3层(hash) 另有
  `ffn_gate_tid2eid.weight[6,129280]` 需一并处理。
- **硬约束。** k4: expert_used 6→必须钳到 ≤4；门控可能选到未保留专家 → loader 须安全处理。

**Blocker / decision needed.** 任何物理裁剪模型 (k4/k16/k48) 要**加载**都需 reduced-expert loader
(`load_expert_keep_map` + 路由 remap)，该代码=mine 分支那套，与"不用 mine"冲突。已就此向用户提问
(port-minimal-loader vs base-on-mine vs clean-room-rewrite)。已完成: plan.md 落地; gguf-tools 与
runtime 残留度调研; k16 schema dump。未做: 任何模型加载/build/86GB 读取 (内存安全 + 待决策)。

### 决策 + k4 文件已生成 (2026-05-30)
- **用户定**: (1) loader = 移植最小 reduced-expert loader 到 mac (顺带让 k16/k48 可用); (2) k4 专家选 = 前4个最简 (smoke 质量非重点)。
- **数据工具移植**: 从 mine 取 shrink_gguf.py + make_expert_mask.py 进 gguf-tools/ (纯数据工具,
  流式 64MB, 不 mmap/不推理)。新写 make_firstk_mask.py (每层留专家 0..K-1)。
- **k4 已生成并验证**: mask-k4 (每层 keep 0..3) → `gguf/ds4flash-k4.gguf` = **10.02 GB**,
  10.8s, **峰值 RSS 198 MB** (内存安全实证)。元数据: expert_count=256(不变), expert_used=6(待 loader 钳到4),
  ds4.expert_keep_map.kept_counts=[4×43] + original_ids=[0,1,2,3×43], 专家张量 dim[2]=4, tid2eid 逐字节拷贝(原始id)。
- **物理验证**: 可行性报告实测 backbone(非专家)=86.72−77.91=**8.81 GB** (= 带宽地板, 吻合 ~8.85 估算);
  专家 77.91GB→1.22GB。
- **下一步**: 移植 loader (load_expert_keep_map + 路由 remap + expert_used 钳位 + GPU route_translate LUT),
  然后 build + 单机 smoke 加载 (内存看门狗)。

### ds4.c loader 移植完成 + 编译通过 (2026-05-30)
- **决定**: 不钳 expert_used (tid2eid 张量是 [6,vocab], 钳了会和它 validation 冲突)。靠 keep-mask + route_translate(被丢→slot0) 优雅处理 kept(4)<used(6)。
- **ds4.c 改动 (4 处, 全部编译通过, 0 error)**:
  1. ds4_model 结构体加 expert_shrunken / expert_layer_count / expert_kept_count[] / expert_orig_to_compact[]。
  2. 加 model_expert_kept_count() + model_expert_compact_slot() + load_expert_keep_map() (适配 mac 现成的 model_get_array/cursor API)。model_close 释放新分配。
  3. model_open 在 parse_tensors 后调 load_expert_keep_map(m)。
  4. weights_validate_layout(m, w): 专家张量检查改用 model_expert_kept_count(m,il); 调用点传 m。
- **build**: `make` 全绿, ds4/ds4-server/... 全部链接成功。
- **Metal 侧调研**: 路由 top-k 在 GPU (g_router_selection_buffer); 专家 matvec 用 _id_ kernel (g_moe_mul_mv_id_*) 按 id-map 索引; n_total_expert 来自专家张量 dim[2]=4 → **wrap=4×expert_bytes, 加载内存安全**; 但 id-map 用原始 id(0..255) 索引 4-专家张量会 GPU 越界 → 需 route_translate (metal/dsv4_misc 加 kernel_dsv4_route_translate + ds4_gpu_set_expert_keep_lut + 在 router 选择后/专家 matvec 前 wiring + load 时 set LUT)。GPU 越界被驱动沙箱化, 不爆 RAM/不 panic, 仅输出垃圾。
- **未跑**: 任何 k4 模型加载 (遵守"改完等确认再运行"铁律, 待用户确认 + 13GB 看门狗)。
- **已交付产物**: gguf/ds4flash-k4.gguf(10GB), plan.md, gguf-tools/{shrink_gguf,make_expert_mask,make_firstk_mask,dump_gguf_meta}.py, ds4.c loader 移植。

### Stage 1a smoke 结果 — k4 单机加载 (2026-05-30, 用户确认后跑, 13GB RSS 看门狗)
命令: `./ds4 -m gguf/ds4flash-k4.gguf --ctx 2048 --temp 0 -n 8 -p "1+1=?"`
- ✅ **loader 工作**: 打印 "reduced-expert model: keep-map over 43 layers (min kept 4 of 256)", 无 exit(1)。
- ✅ **内存安全**: 模型驻留 mapped **9554.67 MiB (~9.3GB)** (2 overlapping shared buffers); residency 请求 5930ms (页入 9.3GB); 机器无 OOM/无崩溃。
- ✅ **管线跑通**: 43 层 GPU prefill + 生成全程正常; context buffers 197 MiB (ctx=2048)。
- ✅ **基线速度**: **prefill 17.43 t/s, generation 10.58 t/s** (单 M4, k4)。
- ⚠️ **输出垃圾** (重复 <bos>): route_translate 未补, 专家 id 索引错乱 → 预期内, 不崩不爆内存 (GPU 越界被驱动沙箱化)。
- **关键发现 1 — 度量修正**: `ps -o rss` 只报 **38MB** (严重失真!), 因 Metal no-copy mmap (MAP_SHARED) 驻留不计入进程 anonymous RSS。**真实内存看门狗必须用 mach task_vm_info 的 phys_footprint, 不能用 ps rss**。这修正 plan Stage 0 看门狗的实现。
- **关键发现 2 — 物理论点证实**: k4 只 4 专家 (专家计算极轻) 仍只 ~10.6 t/s → **瓶颈是 Q8 attention backbone 不是专家**, 与 plan 地基物理一致。TP 带宽聚合 + 投机 + Q5 重量化是攻地板的手段。
- **下一步**: 补 GPU route_translate (metal kernel + LUT + wiring + load 时 set), 让输出正确; 然后才有意义跑 ds4-eval 质量 / 多 token 速度。

---

## 2026-05-30 — plan.md 拆分为有序 task 清单 (scope/planning)

**动作.** 按用户要求把已批准的 `plan.md` 拆成有序、可独立验收的 task,落地到 `task/` 目录(纯文档,
不动代码/不加载模型)。

**拆分结果 (8 task + 索引).**
- `task/README.md` — 索引: 执行顺序、依赖表、关键闸门 (E0 / TP 对拍 / 质量门 / 内存闸 / ≥20t/s 铁律)、模型实数。
- `01-gpu-route-translate.md` — 让 k4 输出正确 (metal kernel + LUT + wiring + load 时 set)。前置: Stage 1a 已完成。
- `02-stage0-profiling-watchdog.md` — DS4_PROFILE 计时 + phys_footprint 看门狗 + L1 预算闸。基础设施。
- `03-stage1b-calibration.md` — 单机标定地板 + MTP 收益 + 200K 内存外推。前置: #01,#02。
- `04-e0-thunderbolt-allreduce-bench.md` — 雷电 all-reduce RTT 基准 = TP go/no-go 闸。前置: #02。
- `05-stage2-tp-skeleton.md` — 最小 TP (只切 down_proj + 1 all-reduce)。前置: #04 通过。
- `06-stage3-full-tp-expert-parallel.md` — 全层 TP + 专家并行 + 装载策略。前置: #05。
- `07-stage4-attn-requant-q5.md` — attention Q8→Q5/Q4 离线重量化 + 质量门。前置: #03。
- `08-stage5-speculative-200k.md` — MTP 叠加 + 200K + 逼近 20 t/s + 备选路线。前置: #06,#07。

**顺序依据.** 单机线 (#01→#02→#03) 不依赖雷电,先打通+标定; E0 (#04) 是 TP 总闸,gate 住 #05–#06;
#07 离线量化可与 #06 并行; #08 集成收尾。每个 task 含: 目标/前置/落点(file:line)/步骤/闸门/内存安全/
验证命令/产出,与 plan 各 Stage 一一对应。

**状态.** 仅写文档。未编译、未加载模型、未做任何验证 (遵守内存安全 + 改完等确认铁律)。
plan.md 进度 checklist 不变,仍为单一事实源; task/ 是其可执行展开。

---

## 2026-05-30 — 开始 Task 01 (GPU route_translate): 摸清落点 + 工具损坏 BLOCKER

**动作.** 用户指示: 给 task 索引加「完成状态」列(已加, 图例 ⬜/🔄/🟡/✅/⛔), 然后按序从 #1 开始干活,
干完更新状态并等指令。开始 Task 01。

**已确认代码事实(grep 多次交叉一致, 可信).**
- `selected`(原始 id 0..255) 由 `ds4_gpu_router_select_tensor`(ds4_metal.m:13798)/`_batch_tensor`(:13893)
  写入 `g_router_selection_buffer`(:122)。
- 消费方 `ds4_gpu_routed_moe_one_tensor`(:13989) 与 `_batch_tensor`(:14296) **均带 `layer_index` 形参**;
  用 `g_moe_mul_mv_id_*`(:72-80) 按 id 索引专家张量(k4 仅 4 行)→ 越界根因。
- 实现锚点: 两函数取出 `selected` 后、构造 mul_mv_id 前, 插一次新 kernel `kernel_dsv4_route_translate`
  原地把 `selected` 改写为 compact slot(gate/up/down 复用同一 selected, :14524 注释佐证)。
- LUT 源 = host `m->expert_orig_to_compact`(ds4.c:1159, `[layer*256+orig]→slot/-1`), `load_expert_keep_map`
  (ds4.c:1377) 填; 新增 `ds4_gpu_set_expert_keep_lut` 在 model_open(ds4.c:1616 后)上传; kernel `slot<0→0`。
- kernel 模板 = `kernel_dsv4_router_weights_one`/`_finalize_one`(metal/dsv4_misc.metal:105/115)。

**BLOCKER (本次会话工具层间歇性损坏 Read 输出).** 证据: (1) 读 ds4_metal.m 返回**乱序行号**
(14015 早于 14012)——Read 是 cat -n 顺序, 不可能乱序; (2) 结果尾部被追加伪造的 `<result name=System>`
块+畸形 `<parameter name=3>` 标签, 非 harness 真实行为; (3) 出现"凑巧已存在"的伪造注释
(// see route translate below); (4) 早前 /tmp 文件一度返回 `?? out of memory` 等垃圾。复读小范围
(14008-14019)干净且与 grep 一致, 但尾部(14020-14022)又引用未声明变量——**间歇性拼接, 不稳定**。

**裁决.** 不在工具输出损坏期间盲改 Metal 推理核心: (a) Edit 需精确匹配真实字节, 且即便匹配成功也会基于
被拼接/伪造的上下文心智模型放错位置, 违反 correctness-before-speed 铁律; (b) 改完按铁律也无法运行验证,
本轮无论如何无法真正"完成" #1; (c) 盲改 14k 行文件有损坏真实源码风险。故: 未改码、未编译、未运行。
Task 01 置 🔄 进行中(已记录全部落点, 恢复后可直接据 task/01「进展」逐处即时重读→Edit→异常即停)。等用户指令。

### 续 — 工具短暂恢复, 落地一半 wiring; 工具再次崩(返回空)后停手 (2026-05-30)

**用户确认.** LUT 传递方式 = 常驻 GPU buffer + 全局 kernel(AskUserQuestion 选项 1)。

**已落地(7 处 Edit, 每处精确唯一匹配 = 可信, Edit 安全-by-construction):**
1. metal/dsv4_misc.metal: `struct ds4_metal_args_dsv4_route_translate` + `kernel_dsv4_route_translate`(orig id→compact slot, 丢弃/越界→slot0)。
2. ds4_gpu.h: 声明 `ds4_gpu_set_expert_keep_lut` + `ds4_gpu_translate_expert_ids`。
3-5. ds4_metal.m: 全局 `g_dsv4_route_translate_pipeline`+`g_expert_keep_lut_buffer`+`g_expert_keep_lut_layers`;
   init 获取 pipeline + 加入校验链; teardown(一处) pipeline=nil。

**未做(工具崩):** ds4_metal.m 两函数体(需读 CB/residency 约定); ds4.c set LUT + 两处插 translate
(decode :10739 / prefill :13585); g_expert_keep_lut_buffer teardown nil; make; run。详见 task/01「当前代码状态」。

**⚠️ 树处于半 wiring 态, 可能不 build 干净**(未用 static 变量 + 两个无定义声明; 若 -Werror -Wunused 则断)。
明确不宣称可编译。补完 #6-#9 才干净。

**正确性已核验:** translate 必须在 router_select 后(weights 已用原始 id 算完); 前3层 hash 与 topk 同写
一个 selected, 统一翻译覆盖。设计安全。

**工具故障升级证据:** Read 从乱序/拼接 → 返回空; `sed -n` 空; 连 `echo probe-$(date)` 都空输出。
非内容问题, 是工具层传输崩。无法读=无法拿 CB 约定锚点, 无法 build 验证 → 按 correctness-before-speed
铁律停手, 不盲改 14k 行推理核心。等工具恢复的会话/轮次继续(据 task/01 清单逐处即时重读→Edit→make 兜底)。

### 续 2 — 工具恢复, Metal 侧全部完成; ds4.c wiring 时工具再崩, 停手 (2026-05-30)

**纠正上一条的过度悲观.** 工具恢复后复读干净且自洽, 之前的"空输出"多为回显竞争, Edit 其实都成功了。
拿到完整 CB/encoder/buffer 样板(ds4_gpu_command_buffer:249 / compute_encoder / end_compute_encoder /
finish_command_buffer:297 / ensure_scratch_buffer:325 / newBufferWithBytes / tensor_offset:743)。

**Metal 侧 100% 完成(6 处 Edit, 全部精确唯一匹配):**
- 在 ds4_gpu_router_select_tensor(:13807) 前实现 `ds4_gpu_set_expert_keep_lut`(newBufferWithBytes 建 Shared
  常驻 LUT, n_layer×256 int16 ≈21KB, 替换旧) + `ds4_gpu_translate_expert_ids`(无LUT→return1 no-op;
  越界layer报错; 否则 CB→encoder→setBytes args(0)/setBuffer lut(1)/selected(2)→dispatchThreads(n_tokens×
  n_expert_used)→end→finish)。
- teardown 补 g_expert_keep_lut_buffer=nil + layers=0。
- helper 全验证存在(tensor_offset/_bytes/_hot_pipeline/command_buffer/finish_command_buffer)。
- 设计: LUT 是独立 21KB 小常驻 buffer, 不进 model residency set(那是给 mmap 模型视图的); setBuffer 绑定的
  Shared buffer Metal 自动 wire。host LUT 格式 m->expert_orig_to_compact(int16[layer*256]) = kernel 期望格式, 直传。

**仅剩 ds4.c 3 处(被工具二次崩阻塞):**
7. model_open(:1614 load_expert_keep_map 后) 若 expert_shrunken 调 set_expert_keep_lut 上传 LUT。
   **未决点: 必须先查 ds4.c 怎么用 #ifdef DS4_NO_GPU 守 GPU 调用(CPU build 不能调)**——这正是工具崩时
   想 grep 的(grep DS4_NO_GPU 返回 `?? out of memory` 伪输出)。
8. decode metal_graph_ffn_layer(:10739) 插 translate_expert_ids(g->router_selected, il, USED, 1, kept_count)。
9. prefill(:13585) 插 translate_expert_ids(g->batch_router_selected, il, USED, n_tokens, kept_count)。

**裁决.** Metal 侧自闭合、可信。ds4.c 三处需先读清 #ifdef 守卫 + 两个 graph 函数局部(ok 变量/model 变量名)
才能写精确 Edit; 工具正返回 `?? out of memory`/`...`省略/伪造行/行号乱序, 读不可信。**不在读损坏期盲改
16k 行推理核心**(误放 GPU 调用进 CPU build, 或放错 ok 链都会坏)。未 make、未运行。Task 01 仍 🔄(Metal 完成,
ds4.c 待)。等工具稳定继续。当前树: k4 行为不变(仍输出垃圾, 因 translate 还没被调用)——非回归。

### 续 3 — 关键架构发现: 需补 CPU 参考实现; 工具三次崩, 收尾 (2026-05-30)

**关键发现(grep 短暂可用时拿到, 可信).** ds4.c 全文仅 3 行提 DS4_NO_GPU(:20-22, 仅 backend 名)。
调用点**无 ifdef 守卫** → `ds4_gpu_*` 符号在 GPU build 来自 ds4_metal.m/cuda, **CPU build 来自 ds4.c 内 CPU
参考实现**(CLAUDE.md 印证)。**推论: 必须在 ds4.c CPU 参考段补 set_expert_keep_lut + translate_expert_ids
的 CPU 版, 否则 CPU build 链接缺符号。** translate CPU 版大概率 no-op(CPU routed_moe 参考可能已用
model_expert_compact_slot 翻译)——动手前需读 ds4.c CPU 参考段确认。这把 ds4.c 待办从 3 处增到 4 处(+CPU stub)。

**工具状态.** 三次崩(grep -c / true;echo 均返回空)。这是本会话工具层硬阻塞, 重试无效, 不可预测。

**本会话产出小结(已落盘, 可信):**
- task/ 8 个 task + README 索引(带完成状态列) ✓
- Metal 侧 route_translate 完整(kernel + 2 函数体 + 全局/init/teardown wiring), 6 处精确 Edit ✓
- ds4_gpu.h 2 声明 ✓
- 全部落点 + CPU 参考实现待办 + 验证步骤记录在 task/01「当前代码状态」, 恢复即可续
- **未做: ds4.c 4 处(set LUT + decode/prefill translate + CPU 参考 stub)、make、run**
- 当前树非回归(translate 未被调用, k4 行为同 Stage 1a)

**恢复后第一步(给下一个会话/轮次):** grep `ds4_gpu_router_select_tensor` 的**定义**(在 ds4.c CPU 参考块)
定位 CPU stub 落点 → 读两个 graph 函数(:10561 metal_graph_ffn_layer / prefill :13556 区)局部 ok/model 变量
→ 逐处即时重读→Edit→make 兜底→等用户确认跑 k4。

### 续 4 — 工具恢复, Task 01 代码完成 + make 全绿 (2026-05-30) ✅代码

**纠正"需 CPU stub"判断(续3过度推断).** python 预处理器分析(权威): 4 个 graph 调用点(decode :10719/10740,
prefill :13556/13587)**全在 `#ifndef DS4_NO_GPU`(8761 大块)内** → CPU build 不编译 → **无需 CPU stub**。
`grep -c ds4_gpu_router_select_tensor ds4.c`=1(仅调用点) 证实 CPU 走独立 forward_token_*_cpu 路径, 不用同名
ds4_gpu_* 符号。只有 set-LUT 在 model_open(全 build 编译) 故加 #ifndef DS4_NO_GPU 守卫。

**ds4.c 3 处落地(精确 Edit, 已 Read 复核位置):**
- :1617 model_open load_expert_keep_map 后, #ifndef DS4_NO_GPU 内, expert_shrunken → ds4_gpu_set_expert_keep_lut
  (m->expert_orig_to_compact, m->expert_layer_count), 失败 ds4_die。
- :10746 decode metal_graph_ffn_layer router 后/moe_one 前: if(ok&&model->expert_shrunken) translate
  (g->router_selected, il, USED, 1, model_expert_kept_count(model,il))。
- :13595 prefill router 后/moe_batch 前: 同上 (g->batch_router_selected, ..., n_tokens, ...)。model 变量名确认
  (:13374 const ds4_model *model)。

**编译验证(ground truth, 已纠正不诚实表述):** `make` 链接成功(ds4/server/bench/eval/agent 全部重链), 但
**有 1 个警告**(我先前误写"0 warning", 错): `ds4.c:1365 unused-function 'model_expert_compact_slot' [-Wunused-function]`。
GPU 路径改用 GPU 内 LUT 翻译 → host model_expert_compact_slot() 没人调了(Stage 1a loader 移植时加的)。
不影响链接(5 binary 都生成), 但违反干净-build。处理选项待用户定。
(另: ds4.c:5385 clang-tidy integer-division 是会话前已存在的未提交改动, 非本 task 引入。)

**完整实现清单(8 处 Edit, 全编译通过):** kernel+args(misc.metal) / 2 声明(gpu.h) / pipeline+buffer+layers 全局
(metal.m) / init 获取+校验 / teardown 清零 / set_expert_keep_lut+translate_expert_ids 函数体(metal.m) /
set-LUT(ds4.c model_open) / decode+prefill translate(ds4.c)。

**设计正确性:** translate 在 router_select 后(weights 已用原始 id 算完); 前3层 hash 与 topk 同写一个 selected
统一翻译; 全模型双保险 no-op(LUT 未设 + expert_shrunken 守卫) 零开销; LUT 21KB 独立小常驻 buffer 不进 model
residency set, setBuffer 自动 wire。

**裁决: Task 01 → 🟡 代码完成·编译全绿。仅剩运行验证**(./ds4 -m k4 -p "1+1=?" 看是否输出连贯),
**需用户确认 + 13GB phys_footprint 看门狗**(内存安全 + no-validation-without-permission 铁律, 改完等确认再跑)。
当前树: 全模型路径零影响(双 no-op 守卫); k4 路径加了 id 翻译(待运行验证是否修复垃圾输出)。等用户指令跑验证。

**关于工具:** 本会话 Bash stdout echo 间歇性丢失/串字(grep 输出残缺/伪造行), 但 Read 文件稳定可靠, Edit
精确匹配安全。策略已切到 "命令重定向到 /tmp 文件 → Read 读回", 绕过 echo 故障, 验证可信。

---

## 2026-05-30 — 装载架构重构 A1: 选择性热集驻留 (用户驳回"全量驻留", 要求懒加载/分批)

**用户裁决.** "不管内存是否够, 一次加载所有模型的方式毫无扩展性, 设计有问题, 必须重新改, 分批/懒加载等机制。"
中止 Task 01 运行验证, 转做装载架构。plan mode 出方案, 用户批准: 路线 A(per-tensor view-shrink+懒驻留)先做
A1(最小改动) 验证物理效果, 不行再 A2; 验证先 k4 验逻辑再大模型验价值。计划存
~/.claude/plans/memoized-frolicking-blum.md。

**根因(已读清 file:line).** ds4.c:19031 全量分支把 [tensor_data_pos, size) 整段传 set_model_map_range →
ds4_metal.m:472 add_model_view_range 切巨型覆盖视图(backbone+专家交错同盖) → :435 request_views 全量
requestResidency → 整模型 wire(k4 实测 9.3GB)。CUDA 侧 ds4.c:1754 已用 "_exps." 跳过冷专家只缓存 backbone,
Metal 无等价物。

**关键发现.** (1) 现成机制可复用: weights_model_map_spans(ds4.c:3303)+model_map_span_vec_*(:3229)+
set_model_map_spans(ds4_metal.m:4955, layer-slice 分布式在用)。(2) wrap_model_range(ds4_metal.m:5040)找不到
覆盖 view 就 nil→崩, 所以冷专家必须仍 wrap(只是不 request residency)。(3) 装载分支在 #ifndef DS4_NO_GPU
(ds4.c:18863)内, CPU build 不碰 → 无需 CPU stub。本地有 k4/k16/k48/全量, 大模型验证有料。

**A1 实现(8 处 Edit, make 0 warning/0 error, 5 binary 22:17 重链):**
- ds4_metal.m: g_model_views 加 resident_hint(:208); add_model_view_range 加 bool resident 参数(2 调用点传 true);
  request_views 只 add resident_hint 的 view + 统计驻留数; set_model_map_spans 重构为 _impl(带可选 resident_flags[],
  NULL=全驻留=旧行为) + 旧接口转调 NULL + 新 set_model_map_spans_split(要求非 NULL flags)。
- ds4_gpu.h: set_model_map_spans_split 声明 + 语义注释。
- ds4_cuda.cu: split 等价实现(忽略 flags, 转调普通 spans; CUDA 冷专家已走 UVA fallback)。
- ds4.c: 新增 model_map_span_vec_split_layer(专家张量→expert 组, 其余含 shared expert→backbone 组, 按字段不靠名)
  + model_map_span_vec_finalize + weights_model_map_spans_split; ds4_engine_create 全量分支加
  `else if DS4_METAL_EXPERT_OFFLOAD` 路径: split→拼 offsets/sizes/resident_flags→调 _split, 打印
  "X GiB backbone resident, Y GiB experts reclaimable"; 默认/split 失败回退原 range 调用 = 零回归。

**设计要点.** 默认行为 byte-for-byte 不变(env 不设走旧 range)。专家仍被 wrap(热路径不 nil), 只是不进
residency set → file-backed 干净页可被内核 VM 压力回收。shared expert 归 backbone(每 token 必用)。

**A1 是"软"控制(诚实).** 靠内核回收非驻留页, 不保证物理页一定换出; 巨型 view(其实 split 后是 backbone/expert
分开的 view)仍占 VM 地址空间。够不够要 k4 实测 footprint 才知道 → 不够再上 A2 逐张小 view 硬控制。

**状态: A1 代码完成·make 全绿。待运行验证**(对比默认 vs DS4_METAL_EXPERT_OFFLOAD=1 的 k4 加载 phys_footprint),
需用户确认 + 13GB 看门狗。这次验证可同时确认 Task 01 route_translate 输出是否连贯(两件事一次跑)。

### A1 验证 (2026-05-30, 用户确认后跑, 13GB 看门狗) — 装载机制通, 暴露 2 个遗留 bug

**第一跑 (DS4_METAL_EXPERT_OFFLOAD=1) 报错 not-covered.** prefill 卡 layer 19:
`Metal model range 0.50..1.15 GiB is not covered by mapped model views`, out 0 字节。

**逐层诊断 (加诊断打印, 解析 GGUF 张量目录, 3 次加载实测) 锁定根因:**
- GGUF 布局其实很干净(非交错!): token_embd[0..8.7MiB] → 全部专家[8.7MiB..1.14GiB] → 全部 backbone[1.14GiB..9.33GiB]。
  backbone/expert merged span 零重叠。host split 分组**正确**。
- 真因 = **wrap 请求 [532901664, 1237544736) 跨度 704MB 越界**。routed_moe 内 gate_tensor_bytes =
  n_total_expert × gate_expert_bytes(ds4_metal.m:14215)。host 两调用点(decode ds4.c:10855 / prefill :13713)
  传 `DS4_N_EXPERT`(=256), 但 k4 张量实际只 4 专家 → wrap 范围比真张量大 64×, 冲进 token_embd 区。
- **全模型路径下此 bug 不暴露**(整 tensor-data 是一个连续 view, 越界 wrap 仍落在大 view 里; GPU 按 route_translate 限定的 id∈[0,4) 索引不真读越界)。**只有 A1 split 成多 view 后才暴露成 not-covered。** 这是
  Task 01/reduced-expert loader 的遗留正确性 bug。

**修复 (2 处 Edit):** ds4.c:10855 + :13713 的 `DS4_N_EXPERT` → `model_expert_kept_count(model, il)`。
全模型零回归(无 keep-map 时该函数返回 DS4_N_EXPERT=256, ds4.c:1357)。make 0 warning。

**第二跑 (修复后):** ✅ not-covered 消失; 全 43 层 prefill + 16 token 跑通 exit 0; 91 view(90 res+1 non-res);
**residency 请求 2372ms→322ms(只 wire backbone, 快 7×, 间接证明只驻留热集)**; prefill 17.60 / gen 9.36 t/s。

**⚠️ 两个未了结:**
1. **footprint 硬数据没拿到** — 看门狗用 footprint -p 采样但没记峰值数字; mapped 9556MiB 是标称 view 映射不是
   物理驻留。A1 核心目标(证明物理内存真降)**尚无硬数据**, 需补测对比默认 vs offload。
2. **输出仍垃圾**(`packfully`+空白) — Task 01 route_translate 没真正修好 k4 输出。n_total_expert 修复让范围对了
   不崩了, 但专家计算链路仍有错。

**诚实裁决: A1 装载机制通了(not-covered 灭, residency 只请热集), 但 (a) 没用 footprint 硬证物理内存降幅,
(b) route_translate 输出正确性仍坏。两者都要继续。不宣称 A1 成功, 只说"机制跑通, 待硬数据 + 输出修复"。**

### 续 5 — 清死代码, make 0 warning (2026-05-30) ✅代码完成

**用户定:** 删 model_expert_compact_slot(选项1)。GPU 路径用 GPU 内 LUT 翻译, 该 host 函数已死代码。
**改动:** 删 ds4.c:1362-1371 model_expert_compact_slot()(连注释)。expert_orig_to_compact 表仍由
load_expert_keep_map 填(将来 CPU 参考翻译可重建)。
**重 make:** 0 warning / 0 error, ds4.o 重编 + 5 binary 重链, grep 确认 model_expert_compact_slot=0 残留,
route-translate 符号在 ds4.c/metal.m/gpu.h 各 3 处完整。(ds4.c:5374 integer-division 是会话前已存在改动, 行号
随删函数从 5385 漂到 5374, 与本 task 无关。)

**Task 01 状态: 🟡 代码完成·make 全绿。唯一剩余 = 运行验证**(./ds4 -m gguf/ds4flash-k4.gguf --ctx 2048
--temp 0 -n 32 -p "1+1=?" 看输出是否连贯/不再重复 BOS), **需用户确认 + 13GB phys_footprint 看门狗后才跑**
(no-validation-without-permission + 内存安全铁律)。等用户指令。

---

## 2026-05-30 — 开始 Task 02 (Stage 0 计时+看门狗): 读取层主动拼接, 停手 (BLOCKER)

**动作.** 用户指示跳到 Task 02。建 4 个子任务跟踪 (DS4_PROFILE 计时 / phys_footprint 看门狗 /
L1 静态预算闸 / make+文档)。开始摸锚点。

**已拿到的可信事实 (grep 与小文件 Read 多次一致).**
- `now_seconds()` @ds4.c:715 (clock_gettime CLOCK_MONOTONIC); `DS4_MTP_TIMING` @20258; `DS4_TIMING`
  打印 model load 时间的先例在 ds4_engine_create 内。
- **全代码库无 task_vm_info/phys_footprint/task_info 使用** → 看门狗要新写 (+`#include <mach/mach.h>` __APPLE__ 守卫)。
- `#include <pthread.h>` @:24, pthread_create @:930 (看门狗线程有基建)。
- 锚点: model_open @2391, model_close @2589, ds4_engine_create @2600(t0/t1 已在), 调用
  `ds4_engine_create_gpu_model` @2614, 其定义 grep 报 @2841。

**BLOCKER — 读取层主动拼接伪造函数体 (本会话再现, 决定性证据).**
- ds4_engine_create(@2614) 把 `ds4_engine_create_gpu_model(...)` 返回值赋给 `ds4_model *m`, 但
  @2841 读到的定义却 `int ds4_engine_create_gpu_model(...)` + 可疑 `_finish` helper → 类型矛盾 (C 不可能编译)。重试稳定复现。
- 决定性: offset-700 读把 `ds4_thread_count()` 体显示成 now_seconds 的 clock_gettime, 但干净 grep
  明确 ds4_thread_count 真实体是 `getenv("DS4_THREADS")`(:725)。**读取层把 A 函数体拼进 B 函数 → 连"自洽"的读取都不可信。**
- 另: grep→/tmp→cat 大输出截断; Read 同一 /tmp 文件多匹配截断; 源码 offset 读间歇空/截断/拼接。

**裁决 (correctness-before-speed + 执行日志续2/续3 既定规则).** 不在读取损坏期盲改 16k 行推理核心:
即便 Edit 精确匹配或失败 (安全), 但围绕上下文的心智模型建立在被拼接的读取上 → 会把增量代码放错作用域。
**本会话零 Edit, 磁盘源码完好 (未改=不可能回归), 仅创建 4 个跟踪 task + 2 个 /tmp 探针文件。**

**Task 02 诚实切分 (恢复后直接据此落地).**
- 本轮推迟全部代码改动 (读不可信)。
- 读稳后**先落安全增量** (锚点已交叉验证): (a) phys_footprint helper + 200ms 看门狗线程 +
  DS4_MEM_BUDGET_MB env + 峰值记录 + 超 budget×0.9 abort (头号诉求"绝不爆内存"); 落点 now_seconds(:715)
  后建自包含模块, 看门狗在 ds4_engine_create(:2600) 顶部 start, atexit flush。(b) DS4_PROFILE 加载墙钟+
  footprint 增量 (engine_create 已有 t0/t1) + CSV 汇总行。
- **依赖 GPU residency 内部 (当前读不可信) 的推迟**: L1 起飞前预算闸 (需 create_gpu_model 内 requestResidency
  前落点) + 逐 chunk/逐 token CSV 行 (需 prefill/decode 热循环锚点)。读稳能交叉验证 create_gpu_model 真实签名后再补。

**make/运行.** 本轮未 make、未运行、未加载模型。运行验证仍需用户确认 + 看门狗 (内存安全铁律)。

### 续 — 读取损坏经实测确认 (grep/Read/小窗口全中招), 持久化可落地设计, 停手 (2026-05-30)

**纠正上一条措辞.** 上条"主动拼接"判断方向对, 但本会话经多轮实测细化为: **读取层对任何稍大输出
间歇性重复/拼接行** —— 决定性证据: (1) Read offset18900 把 model_open 同时显示在 18924 与 18929
(权威 grep 只 18929/18945), 且 18921 的 g_engine_active_backend 在 18930 重复; (2) 小至 10 行的
Read(716-725) 把 now_sec 唯一的 return 复制成两行(721+722); (3) grep stdout 把 prefill 块整段重复、
丢失 eval/model_open 段。结论: **小范围 grep -n 行号可信, 但行内容与任何 Read 都可能被复制 → 构造
不出可信的 Edit old_string, 且 make stdout 同样损坏无法验证 → 不改源码 (铁律)。** 注: Bash 文件写
(heredoc >>) 可靠, 仅 stdout 回读损坏。

**本会话零源码改动. 已交付: 4 个跟踪 task + 本日志设计. 树干净 (git status 仅原有 5 M + 未跟踪), 无回归.**

**Task 02 锚点 (grep 多次交叉一致, 可信):**
- `now_sec()` @ds4.c:718-722 (clock_gettime CLOCK_MONOTONIC, 单 return) ← 模块插入点 (其后)。
- 全库无 task_vm_info/phys_footprint → 新写。`DS4_MAYBE_UNUSED` 定义在 ds4.h:21 (ds4.c:39 引入, 722 处可用)。
- 接线点 (下轮读稳后逐处小窗口重读再 Edit): model_open def@1561 (load 计时+看门狗已由构造函数覆盖, 可不改);
  `ds4_session_eval`@20186 (3 行, 调 _internal → 包计时 add_decode); `ds4_session_prefill`@19726 /
  `_internal`@19737 (包计时 add_prefill)。L1 静态闸需 engine_create residency 区 (@~18990-19113, 正读坏) → 推迟。

**可直接落地的自包含模块 (插在 ds4.c now_sec() 之后, 零热路径依赖; 构造函数自启覆盖加载期 OOM):**

```c
/* ---- Stage 0: phys_footprint 看门狗 + DS4_PROFILE 计时 (见 task/02) ----
 * 看门狗: 200ms 采样 Mach phys_footprint(非 ps/rss, 后者对 Metal no-copy mmap 失真,
 *   9.3GiB 驻留只报 ~38MiB), 记峰值; DS4_MEM_BUDGET_MB 设了则越 90% 预算前 _exit 防 thrash。
 *   构造函数自启 → 覆盖整进程(含模型加载大 mmap+residency), 不碰任何热路径。
 *   DS4_MEM_BUDGET_MB / DS4_PROFILE 都没设 → 一次 getenv 后返回, 不起线程 = 零开销。
 * DS4_PROFILE: load/prefill/decode 墙钟, atexit 输出单行 CSV 到 DS4_PROFILE_FILE(或 stderr)。 */
#if defined(__APPLE__)
#include <mach/mach.h>
static uint64_t ds4_phys_footprint_bytes(void) {
    task_vm_info_data_t info; mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) return 0;
    return (uint64_t)info.phys_footprint;
}
#else
static uint64_t ds4_phys_footprint_bytes(void) { return 0; }
#endif
static const double DS4_GIB = 1024.0 * 1024.0 * 1024.0;
static pthread_t         g_mem_watch_thread;
static volatile int      g_mem_watch_run = 0;
static int               g_mem_watch_started = 0;
static uint64_t          g_mem_budget_bytes = 0;
static volatile uint64_t g_mem_peak_footprint = 0;
typedef struct { int inited, enabled; const char *csv_path;
    double load_sec; uint64_t load_footprint_delta;
    uint64_t prefill_tokens; uint32_t prefill_chunks; double prefill_sec;
    uint64_t decode_tokens; double decode_sec; } ds4_profile_state;
static ds4_profile_state g_prof;
static DS4_MAYBE_UNUSED double   g_prof_load_begin_sec;
static DS4_MAYBE_UNUSED uint64_t g_prof_load_footprint_begin;
static void ds4_profile_init(void) {
    if (g_prof.inited) return; g_prof.inited = 1;
    g_prof.enabled = getenv("DS4_PROFILE") != NULL;
    g_prof.csv_path = getenv("DS4_PROFILE_FILE");
}
static void ds4_profile_flush(void) {
    if (!g_prof.enabled && !g_mem_watch_started) return;
    uint64_t peak = g_mem_peak_footprint, now = ds4_phys_footprint_bytes();
    if (now > peak) peak = now;
    double ptps = g_prof.prefill_sec > 0.0 ? (double)g_prof.prefill_tokens / g_prof.prefill_sec : 0.0;
    double dtps = g_prof.decode_sec  > 0.0 ? (double)g_prof.decode_tokens  / g_prof.decode_sec  : 0.0;
    FILE *f = stderr; int close_f = 0;
    if (g_prof.csv_path && g_prof.csv_path[0]) { FILE *cf = fopen(g_prof.csv_path, "a"); if (cf) { f = cf; close_f = 1; } }
    fprintf(f, "ds4_profile load_sec=%.3f load_footprint_gib=%.3f "
        "prefill_tokens=%" PRIu64 " prefill_chunks=%u prefill_sec=%.3f prefill_tps=%.2f "
        "decode_tokens=%" PRIu64 " decode_sec=%.3f decode_tps=%.2f "
        "peak_footprint_gib=%.3f budget_gib=%.3f\n",
        g_prof.load_sec, (double)g_prof.load_footprint_delta / DS4_GIB,
        g_prof.prefill_tokens, g_prof.prefill_chunks, g_prof.prefill_sec, ptps,
        g_prof.decode_tokens, g_prof.decode_sec, dtps,
        (double)peak / DS4_GIB, (double)g_mem_budget_bytes / DS4_GIB);
    if (close_f) fclose(f);
}
static DS4_MAYBE_UNUSED void ds4_profile_load_begin(void) {
    ds4_profile_init(); if (!g_prof.enabled) return;
    g_prof_load_begin_sec = now_sec(); g_prof_load_footprint_begin = ds4_phys_footprint_bytes();
}
static DS4_MAYBE_UNUSED void ds4_profile_load_end(void) {
    if (!g_prof.enabled) return;
    g_prof.load_sec += now_sec() - g_prof_load_begin_sec;
    uint64_t fp = ds4_phys_footprint_bytes();
    if (fp > g_prof_load_footprint_begin) g_prof.load_footprint_delta += fp - g_prof_load_footprint_begin;
}
static DS4_MAYBE_UNUSED void ds4_profile_add_prefill(uint64_t tokens, double sec) {
    if (!g_prof.enabled) return; g_prof.prefill_tokens += tokens; g_prof.prefill_sec += sec; g_prof.prefill_chunks++;
}
static DS4_MAYBE_UNUSED void ds4_profile_add_decode(uint64_t tokens, double sec) {
    if (!g_prof.enabled) return; g_prof.decode_tokens += tokens; g_prof.decode_sec += sec;
}
static void *ds4_mem_watchdog_main(void *arg) {
    (void)arg;
    while (g_mem_watch_run) {
        uint64_t fp = ds4_phys_footprint_bytes();
        if (fp > g_mem_peak_footprint) g_mem_peak_footprint = fp;
        if (g_mem_budget_bytes != 0 && fp > (uint64_t)((double)g_mem_budget_bytes * 0.9)) {
            fprintf(stderr, "\n[ds4-watchdog] phys_footprint %.2f GiB crossed 90%% of the "
                "%.2f GiB budget -- aborting before page thrash.\n",
                (double)fp / DS4_GIB, (double)g_mem_budget_bytes / DS4_GIB);
            fflush(stderr); _exit(137);
        }
        usleep(200000);
    }
    return NULL;
}
static void ds4_mem_watchdog_start(void) {
    if (g_mem_watch_started) return;
    ds4_profile_init();
    const char *bud = getenv("DS4_MEM_BUDGET_MB");
    if (bud && bud[0]) { long mb = strtol(bud, NULL, 10); if (mb > 0) g_mem_budget_bytes = (uint64_t)mb * 1024ull * 1024ull; }
    if (!g_prof.enabled && g_mem_budget_bytes == 0) return;
    g_mem_watch_run = 1;
    if (pthread_create(&g_mem_watch_thread, NULL, ds4_mem_watchdog_main, NULL) != 0) { g_mem_watch_run = 0; return; }
    g_mem_watch_started = 1;
    atexit(ds4_profile_flush);
}
__attribute__((constructor)) static void ds4_stage0_autostart(void) { ds4_mem_watchdog_start(); }
```

**编译注意 (恢复后核对):** atexit/strtol←stdlib.h(:29); pthread←pthread.h(:24); usleep/_exit←unistd.h(:37);
PRIu64←inttypes.h(:20); 全已 include。已引用: init/flush/watchdog/构造函数链 → 仅 4 个 hook + 2 个 global 未引用,
故标 DS4_MAYBE_UNUSED, build 干净。

**下轮执行序 (读稳后):** (1) 单 Edit 插模块于 now_sec() 后 → make 验证 (compiler=ground truth)。
(2) 接 decode/prefill 计时 (ds4_session_eval@20186 + ds4_session_prefill@19726, 小窗口重读再 Edit) → make。
(3) L1 静态闸 (engine_create residency 区) 待该区读得干净再补。(4) 更新 task/02 + README 状态。
(5) 运行验证 (DS4_PROFILE CSV + 预算 1GB 自测 abort) 需用户确认 + 看门狗。本轮全部未跑、未 make。

### 续 2 — 工具实为高延迟批量返回(非损坏); Task 02 代码完成 + make exit0/0warn (2026-05-30) ✅代码

**纠正前两条"读取损坏/拼接"判断(过度悲观).** 真因 = 工具**高延迟批量返回**: 同批里个别 Read 瞬时
重复行, 但**重读即干净且与 grep 交叉一致**。拿到权威干净读: now_sec()@718-722(单 return)、
engine_create residency 区@18981-19139 完整、KV 闭式 metal_graph_kv_cache_bytes_for_context、
ds4_session_eval 真实@20331(非损坏读的 20186)。策略: 命令重定向 /tmp + 小窗口 Read, 重读核对 → 可信落地。

**Task 02 全部落地(ds4.c 8 处 Edit, make exit 0 / grep -cE 'warning:|error:' = 0):**
1. Stage 0 自包含模块插 now_sec() 后: `ds4_phys_footprint_bytes`(Mach TASK_VM_INFO.phys_footprint,
   __APPLE__ 守卫) + `ds4_mem_watchdog_main`(200ms 采样+峰值+`DS4_MEM_BUDGET_MB` 90% _exit(137)) +
   `__attribute__((constructor)) ds4_stage0_autostart`(自启覆盖加载期) + DS4_PROFILE state +
   `ds4_profile_{init,flush,load_begin,load_end,add_prefill,add_decode}`(atexit flush 单行 CSV)。
2. `ds4_l1_budget_gate(resident_model_bytes, kv_scratch)`: >85% 预算 _exit(137); 三加载分支各调
   (slice=span_bytes / offload=resident_bytes / 全量=tensor-data 字节; KV 暂传 0, 看门狗兜)。
3. decode 计时: 包 `ds4_session_eval`(薄包装, !enabled 直接转调内层=零回归)。
4. prefill 计时: `ds4_session_sync`→`_internal` 改名 + 新薄包装(捕获 prefill token=prompt.len−匹配前缀长度;
   公共签名 ds4.h:228 不变)。
5. load 计时: engine_create model_open(@18929) 前 load_begin + success-return(@19305) 前 load_end
   (覆盖 mmap+residency+accelerator_cache 全部驻留 → footprint 增量真实)。

**零回归核验.** DS4_PROFILE/DS4_MEM_BUDGET_MB 都没设 → 构造函数一次 getenv 即返回(不起线程); eval/sync
!g_prof.enabled 直接转调内层 = byte-for-byte 旧路径; L1 闸无预算 = no-op。include 全已在
(mach/mach.h 新加 __APPLE__ 块; pthread/stdlib/unistd/ininttypes 都在)。未引用的 4 hook+2 global 标
DS4_MAYBE_UNUSED(ds4.c:306) → 干净 build。

**编译 ground truth.** `make` 两次: 仅 ds4.o 重编 + 5 binary 22:57 重链; `grep -cE 'warning:|error:'`=0;
exit 0。注: clang-tidy integer-division note 行号 5610→5645 随我插模块下移 = 会话前已存在的未提交改动, 非本 task。

**裁决: Task 02 → 🟡 代码完成·make 全绿。仅剩运行验证**(DS4_PROFILE CSV 跑一遍 + DS4_MEM_BUDGET_MB=1024
自测 L1 闸/看门狗 abort), 需用户确认 + 13GB 看门狗(no-validation-without-permission + 内存安全铁律)。
当前树: 全模型/k4 路径零影响(env 未设走旧路径)。等用户指令跑验证。

---

## Task 03 — Stage 1b 单机标定 (2026-05-31) ✅ 测量完成

**安全闸**: k4 (10GB, 驻留 9.33GiB 实证) + `DS4_MEM_BUDGET_MB=13000` 看门狗。三跑 (ctx 2048/8192/32768) **全 exit=0, 无 OOM/panic, L1 静态闸均报 "planned resident 9.33 GiB within 12.70 GiB"**。满足内存安全铁律。

### 表① decode/prefill t/s vs ctx (ds4 原生计时为 ground truth)
| ctx | prefill t/s | gen t/s | context buffers | residency req |
|-----|------------|---------|-----------------|---------------|
| 2048 | 71.85 | 9.56 | 197.20 MiB | 4412 ms |
| 8192 | 108.46 | 9.48 | 495.92 MiB | 108 ms |
| 32768 | 104.34 | 9.42 | 880.67 MiB | 135 ms |

- **gen t/s 平坦 9.42~9.56** 跨 16x ctx → decode **不是 KV 带宽限制, 是 backbone 权重带宽限制**。
- 反推有效带宽: 9.5 t/s × 8.81 GiB/tok backbone = **83.7 GiB/s ≈ 90 GB/s** (M4 ~120GB/s 理论的 ~75%)。
- prefill 72→108 t/s 后平台 (prefill_chunk 2048→4096)。
- 实测 gen 9.5 vs plan 论点 10.58/10.6 t/s: 同量级, 本轮略低 (k4 reduced-expert min-keep-4 + 不同 prompt)。**单机 ~16 t/s 带宽地板论点成立, 实跑落在 9.5~10.6**。

### 表② KV 字节 vs ctx (闭式 `metal_graph_kv_cache_bytes_for_context` 纯算; ratio 分布: 43 层 = 0×2, 4×21, 128×20; raw_cap=4352 大ctx)
| ctx | KV (闭式) | raw 项 | comp 项 |
|-----|----------|--------|---------|
| 2048 | 0.184 GiB | 0.168 | 0.016 |
| 8192 | 0.420 GiB | 0.357 | 0.063 |
| 32768 | 0.608 GiB | 0.357 | 0.251 |
| **200000** | **1.889 GiB** | 0.357 | 1.532 |
| 1000000 | 8.016 GiB | 0.357 | 7.659 |

- raw 项 (滑窗 f32) 在 ctx≥8192 后**恒定 0.357 GiB** (raw_cap 封顶 4352); 增长全在 comp 项 (ratio-4 indexer 主导)。
- **200K KV ≈ 1.89 GiB** — 证实 plan「MLA 下 KV ~1.x GB」。闭式与实测 context-buffers 一致 (context-buffers 额外含 prefill scratch 2·comp_cap·prefill_cap·f32)。
- **TP 下 KV 判断**: layer-slice 流水线 → 每机只存自己层片的 KV, **总量按层切分不复制**。200K 时两机各 ~0.95 GiB。MLA latent KV 跨 head 共享不影响该结论 (按层切已隔离)。

### 表③ MTP 接受率/加速 — **暂缺, 阻塞于 #01**
- #01 k4 输出仍为垃圾 (`packfully`/`tempo tempo`)。MTP argmax 接受率比的是垃圾 logits, 测了无意义。
- **裁决: MTP 收益测量推迟到 #01 k4 输出正确后再补**。机制 (`ds4_session_eval_speculative_argmax`) 在位, 草稿 3.54GB 可加载。

### ≥20 t/s 物理推算更新 (关键决策数据)
- **20 t/s 需 backbone 8.81 GiB/tok × 20 = 176 GiB/s = 189 GB/s 持续带宽**。
- **TP 聚合 188 GB/s → 天花板 19.87 t/s (零 allreduce 开销)**。即: **真·tensor-parallel 也刚好够不到 20, backbone 不缩根本过不了**。
- 结论链:
  1. **单机理论上限**: backbone 8.81 GiB/tok 不可压时, 满 M4 120GB/s ⇒ 绝对 ceiling ~13.6 t/s, 实跑 9.5~10.6。**单机 20 t/s 物理不可能**。
  2. **TP 必须配 Q5 (#07)**: 仅 TP 顶到 ~19.9 ceiling 且被 allreduce 吃掉 → **#07 attention Q8→Q5 缩 backbone 是过 20 的硬前置, 不是可选**。
  3. **E0 (#4) 是闸**: per-layer allreduce ×~86。RTT 50µs ⇒ +4.3ms ⇒ ~18 t/s; RTT 150µs ⇒ +12.9ms ⇒ ~15 t/s (失败)。**RTT≤50µs 才有戏**。
  4. **MTP (#08) 提供乘子** 叠在 Q5+TP 之上才稳过 20。
- **200K 内存可行性**: KV 1.89 GiB + k4 9.33 GiB + context scratch ≈ 2~3 GiB ⇒ 单机 ~13.5 GiB, **16GB 机踩 13GB 预算边缘** (本测最大只跑到 32768=9.33+~0.9)。200K 单机需 offload(A1) 或 TP 分担。**够不够: 单机勉强但贴红线, 违反平稳铁律 → 走 TP 两机各 ~0.95GiB KV 更稳**。

### Task 02 缺陷暴露 (顺带, 需修)
- `DS4_PROFILE` CSV 全 0: `prefill_tokens=0 decode_tokens=0 peak_footprint_gib=0.181`。计时薄包装 **没接进实际跑的 GPU layer-major graph 路径** (该路径有自己的 prefill/gen t/s 打印)。
- `phys_footprint=0.181 GiB` **没算到 mmap 驻留的 9.5GB** (macOS phys_footprint 不含可回收 file-backed 页)。**真正护住内存的是 L1 静态闸 (planned resident 9.33 GiB), 不是 footprint 看门狗**。
- 影响: 看门狗 budget 基于 phys_footprint 会**漏判模型本体内存**。Task 02 验收需降级为「L1 静态闸有效, footprint 看门狗对 mmap 失效待修」。建议看门狗改用 resident/RSS 或在 L1 已知 resident 上叠加。

---

## Task 04 — E0 雷电 all-reduce 延迟基准 (代码完成·待双机实测) — 2026/05/31

**类型**: 双机·测量 (TP 总闸门)。前置 #02 计时基础设施已就绪 (复用 CLOCK_MONOTONIC)。

### 落地
- 新增独立小程序 `tools/e0_pingpong.c` (~280 行, 标准库 only, **不链接任何 ds4 核心 obj, 不加载模型**, 只分配 KB~32KB 级缓冲 → 内存安全 by construction, 无需看门狗)。
- Makefile 新增 `make e0` 目标 (→ `e0-pingpong`), 已接入 clean。
- 形态: TCP + `TCP_NODELAY`, server echo / client 计时。client 先做 64 次 warmup (TCP ramp + page-in) 再计 `iters` 次往返; server 改为 **echo until EOF** (避免 warmup 让固定计数提前关闭 → 修了首版 "Connection reset by peer")。
- 统计: RTT min/median/p99/max + stddev 抖动 + 有效吞吐 (双向 2×size/round-trip) + TP 预测 (86 syncs/token × RTT/2)。内置 GO/NO-GO 裁决印在输出末行。

### 验证 (本机 loopback 烟测, 非真实链路)
```
make e0  → exit 0, 0 warn
./e0-pingpong --listen 127.0.0.1:5599 --size 32768 --iters 2000 &
./e0-pingpong --connect 127.0.0.1:5599 --size 32768 --iters 2000
→ server: done, 2064 round-trips echoed (=2000+64 warmup) ✓
→ client: RTT median=19us p99=76 max=136, throughput 2.48 GB/s, VERDICT GO
```
逻辑链路全通 (server/client/stats/verdict)。**loopback 数字无物理意义** — 真实闸门数据必须在两机雷电直连 `en5` (169.254.188.38) 上跑。

### 待办 (双机实测, 需 MacBook + 用户操作)
```
# mini:    ./e0-pingpong --listen 169.254.188.38:5599 --size 32768 --iters 10000
# macbook: ./e0-pingpong --connect 169.254.188.38:5599 --size 32768 --iters 10000
```
- 闸门: median RTT ≤~50µs → TP GO 继续 #05; ≥~150µs → NO-GO 回退 #08 (单 M4 + Q5 + MTP); 50–150µs → 看 TP+Q5 能否物理推算到 ≥20 t/s。
- 顺带读吞吐, 校核 ≈5 GB/s (供 #06 带宽比 64/36 owner 切分)。
- 二进制只在本机编了; 双机跑需把 `e0-pingpong` 传 MacBook 或在 MacBook `make e0` (纯标准库, 任意 mac 可编)。

### Task 04 — E0 双机实测完成 · TP NO-GO 裁决 — 2026/05/31

**链路真相 (推翻 task doc 假设)**:
- task doc 假设 `en5`/`169.254.188.38` = 雷电直连。**错**。`ifconfig en5` media=**100baseTX**(100Mbit 以太网 dongle)→ 实测 median 2206µs / 0.03 GB/s，丢弃。
- 真正雷电是 **`bridge0`/`192.168.1.x`**（members en2/en3/en4 = Thunderbolt Bridge）。用户给的 192.168.1.3↔192.168.1.2 是对的。

**实测 (mini listen 192.168.1.3, M1 peer connect, iters=10000)**:
| payload | min | median | p99 | max | jitter σ |
|---|---|---|---|---|---|
| 64 B | 48 | 72 | 132 | 238 | — |
| 1 KB | 41 | 68 | 114 | 402 | — |
| 8 KB | 52 | 86 | 159 | 276 | — |
| 32 KB | 70 | **116** | 190 | 295 | 24.85 |

- RTT 单位 µs。延迟地板 ~68µs median（min 偶触 41-48 但不持续 <50）。throughput 全是 ping-pong latency-bound（非链路带宽，真带宽需 streaming 测）。
- bridge0 是 macOS L2 bridge，含 learning/forwarding 开销；直配 IP 到 TB 成员口（绕过 bridge）可能压向 41µs min，但 median 难破 50µs。

**≥20 t/s 物理推算 (铁律)**:
- 86 all-reduce/token。乐观 floor 68µs → 5.85ms/token 同步；现实 8KB 86µs → 7.4ms/token。
- Task3 标定 TP 零同步理想上限 = 19.87 t/s（50.3ms/token，带宽闸）。
- 加同步：50.3+5.85=56ms → **17.9 t/s**(最乐观)；现实 57.7ms → **17.3 t/s**；p99 抖动更差。
- 每种情形 <20 t/s。≤50µs GO 门从未触及。

**裁决: TP NO-GO**（落入 50-150µs MARGINAL 带，但 TP+Q5 推算 17-18 t/s < 20，过不了铁律）。
- **不启动 #05/#06**。回退 **#08 单 M4 + Q5 + MTP** 路线（与 Task3 预测一致：单机 ceiling + MTP 乘子）。
- 唯一可能翻盘项：直配静态 IP 到 TB 成员口绕过 bridge0，压 RTT 进 ≤50µs GO 带。但需用户同意改网络配置（会动 bridge0），且 median 地板 68µs 难破 50µs → 翻盘概率低，列为可选后续。
- peer 上留有 /tmp/e0_pingpong.c + /tmp/e0-pingpong（我建的，非预存）；未删（铁律：不删他机文件）。

### Task 05 — Stage 2 TP 最小骨架 (代码完成·待双机验证) — 2026/05/31

用户裁定 E0 链路定性=雷电桥 40Gb/s, 推翻 #04 NO-GO, 命「基于此进行代码开发」。落地最小可验证 TP 骨架。

**关键架构事实**: 现有 distributed 是**层流水并行**(layer-pipeline, 激活 worker→worker), 不是 TP。TP 需两机锁步同层 + all-reduce。故 TP 是新子系统, 与 pipeline 并存。

**已落地 (全部 make 全绿, 非-TP 路径字节不变, g->tp==NULL 守卫)**:
1. **TP 传输原语** (`ds4_distributed.c`): `DS4_DIST_MSG_ALLREDUCE` 帧 + 持久 TP socket (listener=coordinator send-first, connector=worker recv-first, 有序交换防死锁) + `ds4_dist_tp_allreduce_f32()` 两机求和 all-reduce + lazy 增长 scratch (封顶 64MiB)。floats 走主机字节序 (两机同为 arm64 LE)。
   - **单元测试**: `ds4_dist_tp_selftest()` (socketpair+pthread 环回) → `ds4_test --tp-allreduce` **绿** (求和正确+帧格式+无死锁; 无模型, KB 缓冲, 内存安全)。
2. **选项/CLI** (`ds4.h`+`ds4_distributed.c`): `tp_enabled`/`tp_layers` + `--tp`/`--tp-layers N`; TP 模式强制全模型加载 (load_slice=false, 两机各载全层)。
3. **图集成** (`ds4.c` `metal_graph_encode_decode_layer`, 单 token decode 路径): routed_moe 产出 `g->routed_out` 后插入 TP 重组 —
   - **最小骨架用 element-range 切分** (非 kernel 改): 每机将 routed_out 不属于自己的那半清零, sum-all-reduce → **逐元素 bit-exact 复现单机值** (各元素两机都全算, 一方清零, 存活值即单机全值, 无跨界浮点重结合)。证明「切分点+all-reduce 帧+跨机同步+对拍」全链路, **零 Metal kernel 风险**。
   - barrier: `ds4_gpu_end_commands()`(commit+wait 排空 routed_moe 使 host 可见) → host 读/清零/all-reduce/写回 → `ds4_gpu_begin_commands()` 重开 batch 续 shared+combine。
   - 引擎持 `ds4_dist_tp *tp` (首会话建链, coordinator listen / worker connect), session-create 拷进 `s->graph`; `tp_vec` 堆分配 (DS4_N_EMBD 运行时值, 不能做结构体定长数组)。
   - **#02 任务原计划的 Metal partial-down_proj kernel 改 (ff-row 切): 骨架刻意不做, element-split 已达闸; 真实算力切分 (mask 专家/ff-row) 推迟到 #06。**

**回归**: `make` 全绿(exit0,0warn); `ds4_test --server` OK; `--tp-allreduce` OK。非-TP 路径未触碰。

**待验证 (#4, 闸: 内存安全+用户确认+双机)**:
- 双机 greedy vs 单机 `--dump-logprobs` top1 逐 logit 对拍 (element-split 应 bit-exact)。
- `DS4_PROFILE` 量每层 all-reduce+barrier 实测 ms, 核对 E0 (16KB routed_out ≈ 95µs 网络; **但每层一次 full GPU drain barrier 是已知低效**, 真实 per-layer 成本=骨架要测的关键数, 决定全层 TP go/no-go)。
- 命令形态: `./ds4 --role coordinator --listen <mini> 5599 --tp --tp-layers 3 --dump-logprobs /tmp/tp.json --temp 0 -p "1+1=?"` + 另机 `--role worker --coordinator <mini> 5599 --tp --tp-layers 3 ...`。
- **未验证前不得声称工作**; 需同步 M1 重编+传二进制 (两机共享 CORE_OBJS)。

**≥20t/s 物理推算 (铁律, 骨架阶段)**: 全层 TP 网络 43×~95µs=4.1ms/token; 但 element-split+full-barrier 每层一次 waitUntilCompleted 排空, 失 batched-CB overlap → 若每 barrier 加 ~0.5ms 则 43×0.5=21.5ms/token 直接压垮。⇒ **骨架是测量仪器**: 先 --tp-layers 2-3 量实测 per-layer 成本, 再决策。#06 必须用 MTLSharedEvent 局部同步 ([[metal_wait_anukari_precedent]]) 替代 full drain, 否则全层 TP 不可能过 20t/s 闸。

### Task 06 (部分) — MTLSharedEvent 局部同步替代 full-drain barrier — 2026/05/31

用户指定先做这块 (全层 TP 能否过 20t/s 闸的关键)。骨架 (#05) 的 TP barrier 用 `ds4_gpu_end_commands()` = `waitUntilCompleted` 全排空 (慢路径, per-CB 调度开销大)。换成 [[metal_wait_anukari_precedent]] 的 MTLSharedEvent 快路径。

**落地 (make 全绿 exit0/0warn, 非-TP 路径字节不变)**:
- `ds4_metal.m`: 新增 `g_tp_event` (MTLSharedEvent, lazy 建) + `g_tp_event_value` 单调递增。
  - `ds4_gpu_tp_signal_after_batch()`: 关编码器 → 在当前 batch_cb 尾 `encodeSignalEvent:value:` → 返回该 value (0=错)。
  - `ds4_gpu_tp_host_wait(value)`: `[event waitUntilSignaledValue:value timeoutMS:]` 快路径等待 (默认 60s, `DS4_TP_EVENT_TIMEOUT_MS` 可调)。
  - cleanup 清 g_tp_event。
- `ds4_gpu.h`: 声明两函数。`ds4_cuda.cu`: 加 stub (flush 已 cudaDeviceSynchronize, host_wait no-op 成功 → CUDA TP 正确但无快路径)。
- `ds4.c` TP 插入点改写: `signal_after_batch()` → `flush_commands()` (commit 不全等待, 重开 batch) → `host_wait(ev)` (快等 routed_moe 完成) → 读/清零/all-reduce/写回。**去掉 end_commands/begin_commands 全排空**。
  - 正确性: host 写回落入统一内存, 在 (已重开的) batch combine 被 commit 前完成 (调用方 end_commands 在后) → 无需 GPU wait-back。event value 单调, 跨 token/layer 永增不溢。

**机理收益**: routed_moe 那个 CB commit 后异步跑, host 用 event 精确等它 (而非 waitUntilCompleted 排空所有 pending + 完成回调调度)。Anukari/Apple 实测该机制 ~150ms→<50µs/sync。**理论上把每层 TP 同步从 full-drain 降到 event-wait**。

**仍未做 (用户只点了 MTLSharedEvent 这块; #06 其余推迟)**:
- shared-expert (与 routed_out 无关) 与 host all-reduce 的 GPU/网络 overlap (需重排编码顺序: shared 在 wait 前, 仅 combine 等)。当前是 signal→flush→等→读, shared 仍在等待后编码, 未 overlap。
- 真实算力切分 (mask 专家 / ff-row row-parallel) 替代 element-split。
- attention 列切 + 第 2 个 all-reduce/层; 专家并行池化 (不相交专家集省内存); 64/36 带宽比 owner 切分。
- **per-layer 实测 ms 仍需 #4 双机跑量化** (event-wait 真实开销 vs full-drain 的对比, 决定全层 TP go/no-go)。

**回归**: `make` 全绿; `ds4_test --tp-allreduce` OK; `ds4.c` 在 `-DDS4_NO_GPU` 下编译通过 (TP 图代码 GPU-guarded)。CUDA stub 保 Linux 链接。

### 同步重编 (M1) + 修 Makefile default-goal 回归 — 2026/05/31

用户命「同步重编」(铁律 [[feedback_rebuild_check_other_machine]]: 两机共享 CORE_OBJS)。
- M1 仓库在 `/Users/fodelf/ds4-main` (**非 git, 是源码副本**), 整体陈旧 (TP 符号 0, `dsv4_misc.metal` 校验和不符)。
- rsync 全部构建源码 (顶层 *.c/*.h/*.m/*.cu + Makefile + metal/ + tests/ds4_test.c + tools/) 到 M1, **不带 --delete, 不传模型/产物, 不删任何文件**。
- **发现并修复我引入的 Makefile 回归**: 把 `e0:` 目标放在了 `all:` 之前 → e0 成了 default goal → 两机 `make` 只建 e0-pingpong, 不建套件 (本机此前靠显式 `make ds4`/`make ds4_test` 才有新二进制, 套件其余陈旧)。修复: e0 规则移到 clean 之后; `.DEFAULT_GOAL := all` 恢复。Makefile 重新 rsync 到 M1。
- **M1 `make clean && make` 全量重编**: exit 0, ds4/server/bench/eval/agent 齐全, **ds4 含 TP 符号** (2 个 pre-existing 未用符号 warning, 非我引入)。
- **本机修好 Makefile 后全量 `make`**: exit 0, ds4 含 TP 符号, 5 二进制最新。

两机源码一致 + 二进制均含 TP 代码 (element-split + MTLSharedEvent 同步), 可做 #4 双机验证。**仍未跑模型** (待内存安全闸 + 用户确认)。

### #4 受阻: TP 两机锁步编排缺失 (跑前发现, 未加载模型) — 2026/05/31

k4 已传 M1 (29.5s/~316MB/s 走雷电桥, 大小+头部md5 校验一致)。两机均 16GB, 均有 k4 + 含 TP 符号二进制。

**但 #4 现在跑会死锁** (代码审查发现, 故未加载模型):
- `ds4_dist_run`: coordinator→`dist_run_coordinator`(流水线生成, 给 worker 发 WORK 帧让算层切片); worker→`dist_run_worker`→`dist_worker_read_loop` **等 WORK 帧**。**无 TP 专用编排**。
- TP 模式 (load_slice=false 全层本地) 下 coordinator decode 到第 0 个 TP 层调 `ds4_dist_tp_allreduce_f32` 阻塞等 worker; worker 在 read loop 等永不到来的 WORK 帧 → **死锁**。
- 另: TP all-reduce hook 只在单token decode (`metal_graph_encode_decode_layer`), **不在 prefill batch 路径** (`metal_graph_encode_layer_batch`)。骨架 OK (prefill 两机全算=复制, KV 同; 仅 decode 切分+all-reduce, element-split 仍 bit-exact)。

**缺的编排 (TP leader/follower 锁步, 才能跑 #4)**:
- **follower(worker)**: 收 prompt → 本地 prefill(复制全算)→ 每 decode 步: 收 coordinator 采样的 token → 跑全层 decode forward(命中 all-reduce 与 coordinator 会合)→ 弃 logits → 循环。
- **leader(coordinator)**: 发 prompt → prefill → 每步: decode forward(命中 all-reduce)→ 采样 → 广播 token 给 worker → 循环。
- token 广播走 TP socket 加控制消息或单独通道。
- 零-stub 替身不可行(worker 必须真算它那半 routed_out, 否则 all-reduce 结果错→logits 错)。

**结论**: TP 零件齐 + 单元测试绿, 但端到端两机锁步需补 leader/follower run loop。下一步=实现该编排, 再跑 #4。

### #5 编排实现完成 + 验证受阻于 M1 实例锁 — 2026/05/31

**#5 代码完成 (两机 make EXIT=0, TP 符号在位)**:
- `ds4_distributed.c`: TP 控制协议 (`DS4_DIST_MSG_TP_PROMPT/STEP`) + `dist_tp_send/recv_prompt/step` (直接用 `tp->fd`) + `dist_run_tp_leader` (tokenize→send_prompt→prefill via eval_layer_slice 全层→greedy argmax 循环: send_step+打印+eval, 末尾 send_step(-1)) + `dist_run_tp_follower` (recv_prompt→prefill→循环 recv_step→eval, token<0 停)。`ds4_dist_run` 在 tp_enabled 时分流。
- `ds4.c`: `ds4_engine_tp()` 访问器; session_create 在 tp_enabled 时跳过 pipeline 分布式会话 (`&& !tp_enabled`), 保持纯本地 + tp handle。
- `ds4_cli.c`: coordinator+tp 也走 ds4_dist_run (原只 worker)。
- 两机 rsync + clean make 全绿。

**单机 k4 基线 (mini, 内存安全 peak 0.42GiB footprint/budget 12.7)**: prefill 17.41 / gen 10.04 t/s (健康), **但输出垃圾** (" pushing —— pointedfully...") = **#01 k4 专家链路 pre-existing bug, 非 TP 问题**。⇒ #5 验证判据改为 "TP 输出 == 单机输出 (element-split bit-exact)", 不是 "正确文本"。

**验证受阻**: M1 实例锁被 **`ds4-expert-replica` (pid 83221, 已跑 16h, RSS 0.14GB)** 占用 (用户早期双机 expert-remote 遗留) → TP follower "refusing to start" → leader 挂 accept (已杀)。**铁律: 不擅自杀另一机进程**, 需用户确认是否停 83221 释放 M1 实例锁。mini 已清理无残留。

### ✅ #4/#5 双机 TP 对拍 PASS (bit-exact) — 2026/05/31

用户确认停掉 M1 expert-replica(pid 83221)释放实例锁。双机 TP 跑通并验证。

**踩坑→修复**:
1. M1 实例锁被 16h expert-replica 占 → 用户确认后 `kill 83221`。
2. M1 (M1 Pro 16GB) 全模型 k4 prefill GPU OOM(`kIOGPUCommandBufferCallbackErrorOutOfMemory`)— 单机 M1 也复现 ⇒ M1 GPU 容量临界, **非 TP bug**(mini M4 不 OOM)。修复:`DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_NO_RESIDENCY=1`(8.20GiB backbone resident + 1.13GiB experts reclaimable)→ M1 跑通。
3. `DS4_METAL_PREFILL_CHUNK=4` 致 "chunk 13 exceeds prefill cap 4"(我的 TP prefill 一次喂 13 token)→ 去掉该 env,默认 cap 够。

**最终验证(两机 16GB,env=EXPERT_OFFLOAD+NO_RESIDENCY,-c 2048,--tp --tp-layers 3,-n 16 --temp 0 --nothink,"1+1=?")**:
- 单机基线: ` pushing      —— pointedfully pointing  fullyand standing pointingthe—`
- 双机 TP   : ` pushing      —— pointedfully pointing  fullyand standing pointingthe—`
- **逐字 bit-exact 一致**(仅文件尾换行差异)。md5 文本内容相同。
- follower "prefilled 13 prompt tokens, following leader" 全程无 OOM;两机 exit 0,无残留进程。

**结论**: #5 TP leader/follower 锁步编排 + #1 all-reduce 传输 + #3 element-split 图集成 + #6 MTLSharedEvent 同步 **端到端正确**(16 token 锁步,decode all-reduce 逐字复现单机)。输出"垃圾"文本=#01 k4 专家链路 pre-existing bug(两机同样垃圾→TP 证明正确,k4 正确性是 #01 独立问题)。

**仍 TODO**: per-layer all-reduce 实测 ms 未量(DS4_PROFILE 钩子不在 slice 路径=#02 已知问题;TP eval_layer_slice 路径也无 t/s 打印)。需在 TP 路径加计时才能核对 E0 预测 / 更新 ≥20t/s 推算。M1 expert-replica 已停(用户确认),未自动重启。

---

## 2026-05-31 TP decode 测速尝试 (per-layer all-reduce 计时 / ≥20t/s 核对)

**目标**: 给 TP 路径加计时, 量真实 prefill/decode t/s, 核对 E0 / ≥20t/s 闸。

**本次代码改动 (都已两机重编 ds4_distributed.o)**:
1. `dist_run_tp_leader` decode 循环加 wall-clock 计时, 末尾打印
   `prefill N tok in Xs (Y tok/s) | decode N tok in Xs (Y tok/s)`。
2. CLI 校验 `--role x requires --layers` 在 `tp_enabled` 时豁免 (TP 全模型加载,
   不做层切分; 传 `--layers 0:output` 反而触发 slice 映射 + L1 gate 80.76GiB 拒绝)。
3. TP prefill 改分块喂: `dist_tp_prefill_chunk()` (env `DS4_TP_PREFILL_CHUNK`, 默认 4)。
   leader/follower 原本一次喂全 prompt 13 token → prefill GPU OOM。

**踩坑**:
- `--layers 0:output` → "restricting metal model map ... 80.76 GiB" → L1 gate refuse。
  TP 不能传 --layers。
- `-c 4096` / `-c 2048`: M1 Pro (并 M4) prefill `kIOGPUCommandBufferCallbackErrorOutOfMemory`。
  OOM 主因是 **ctx-size 决定的 attention scratch GPU buffer**, 不是 token batch 宽度。
- `-c 1024` + 分块: 两机 prefill 均过关 (follower "prefilled 13 prompt tokens, following leader")。

**关键实测结果 (`-c 1024 --tp-layers 3 -n 64`, env=EXPERT_OFFLOAD+NO_RESIDENCY)**:
- prefill 过关, decode 启动正常, 输出首 token。
- **decode 数分钟才出 2 个 token (`1+`) ⇒ <0.05 t/s**。
- 根因: expert-offload + NO_RESIDENCY 下 82GiB 模型在 16GB 机器, decode 每 token
  命中不同 expert (K=16/layer × 43 layer) → 持续 **SSD page-in (I/O-bound)**。
  印证 [[cb_floor_truth]]: working set 进不了 RAM, decode = SSD I/O bound。
- **这不是 TP bug**: 双机 TP 骨架已验证 bit-exact 正确; 但 TP (tp-layers 3 + 全模型加载)
  没改变"每机都要全模型"的事实 → 单机内存墙照样卡 decode, 双机一样慢。
- 用户中途叫停 -n 8 精确复测 (太慢无意义)。两机进程已全停, 无残留, 未动文件。

**结论 / 下一步**: 当前双机 TP 配置 decode 远不到 20 t/s, 瓶颈是 16GB 物理内存墙
(SSD I/O), 非网络/非 all-reduce。要让 decode 提速必须让 working set 进 RAM ——
唯一路径是 **专家并行池化 (EP): 每机只存不相交的 expert 子集** (#06 待办), 使两机
合计 RAM 能容纳活跃专家工作集, 绕过单机 SSD I/O。tp-layers element-split 本身不省内存。

---

## 2026-05-31 双机 TP k4 测速脚本 `tools/tp_k4_speed.sh`

用户要求的一键脚本: 同步源码→M1 → 两边 `make clean && make ds4` → k4 模型
(`gguf/ds4flash-k4.gguf`, 10GB, 能进 16G RAM 不 SSD-bound) + `-c 32` + `-p "hi"`
跑双机 TP → 打印生成文本 + prefill/decode tok/s。
安全闸(任一触发"两边同杀, 只杀进程不删文件"): ① 终端 Ctrl+C; ② 本机 ds4 RSS>12G;
③ M1 ds4 RSS>8G。看门狗每秒 `ps -o rss=` 读两边 RSS。rsync 不带 --delete。
参数可 env 覆盖 (CTX/NPRED/PROMPT/LOCAL_MAX_GB/REMOTE_MAX_GB/MODEL/...)。
已知风险: k4 10GB, M1 8G 阈值可能在加载阶段触发误杀; 用户裁定"保持 8G 先跑看实际 RSS",
且由用户自己在终端运行(掌控 Ctrl+C), 未自动执行(内存安全闸铁律)。

---

## 2026-05-31 雷电网桥休眠导致脚本 follower connect EHOSTUNREACH

**现象**: `tools/tp_k4_speed.sh` 跑 follower 报 `unable to connect to 192.168.1.3:5599:
No route to host`, 但手动跑能连、且 ping 通。

**诊断 (不加载模型)**: python bind 192.168.1.3:5599 + M1 `nc -z` 连续 6 次全 OK ⇒
网络层/bridge0(雷电)/bind 全正常, ds4 用标准 socket。`dist_connect_endpoint`
已内置 200×25ms=5s 重试且 EHOSTUNREACH 在 retryable 列表 — 不是 ds4 bug。

**真因**: 雷电网桥(Thunderbolt Bridge=bridge0)空闲进低功耗, 首次流量需重协商,
有 20+s 的 down 窗口。脚本前置 rsync + 两边 make clean&&build 占数十秒无 192.168.1.x
流量 → 雷电休眠 → follower connect 正撞重协商窗口 → EHOSTUNREACH 熬不过 5s 重试。
手动跑"行"是链路被频繁操作喂着、是醒的。诊断用的 ping/nc 把链路唤醒了。

**修复**: 脚本启动 follower 前改为"持续 ping 预热, 要求 M1→leader 连续 3 次通才放行"
(env WARM_STREAK/WARM_TIMEOUT 可调), 唤醒并确认雷电稳定后再 connect。

---

## 2026-05-31 (续) M1 ds4 出站 connect 怪象 → reverse-connect workaround + k4 双机首个 decode 数字

**怪象 (未查明内核根因, 已 workaround)**: M1 那台机器, ds4 进程加载完模型后
"出站 connect 192.168.1.3:5599" 稳定返回 EHOSTUNREACH(No route), 但:
- 同机 nc / 最小C(照搬 ds4 connect 序列) / Metal 程序 connect 同地址 全部 OK
- 强制 bind 正确源(192.168.1.2)仍 EHOSTUNREACH → 排除源地址选择
- Metal 初始化前后 connect 都 OK → 排除 Metal
- 反向(本机 ds4 加载模型后 connect M1 192.168.1.2)→ CONNECT OK
排除了: 网络层/雷电休眠(物理直连无此事)/静默/getaddrinfo(解析正确 IPv4)/源地址/Metal。
唯一确定: M1 这台机器 ds4 进程出站方向坏, 本机→M1 方向好。根因存疑(M1 内核/进程态)。

**Workaround**: `DS4_TP_REVERSE_CONNECT=1` (ds4.c session_create) 翻转 TP 网络角色 —
coordinator 主动 connect, worker listen。TP all-reduce 对称求和, 方向不影响结果;
tp_owns_low 仍按 role 不按谁 listen。CLI 校验放宽(reverse 时 coordinator 用
--coordinator / worker 用 --listen)。两机重编。

**k4 双机 TP 反转跑通 (c=32, tp-layers=3, "hi")**:
- prefill 10 tok 0.671s = 14.9 t/s ✓
- **decode 7 tok 32.700s = 0.21 t/s** — 极慢, 远不到 20 t/s。
- worker "prefilled 10 prompt tokens, following leader" ✓ 全程无 OOM。
- 慢因(待查): 每 decode token 在 tp-layers 3 层各做一次 GPU drain(flush_commands)
  + 跨机 all-reduce + MTLSharedEvent 同步; 叠加 expert-offload page-in。

**坑**: 5599 端口被我调试用的 python listener 残留占用 → worker "Address already in use"
(SO_REUSEADDR 对活进程无效)。脚本启动前加 lsof -t :5599 | kill 清端口。

**脚本**: tools/tp_k4_speed.sh 已改 reverse 版(worker listen 先起, coordinator connect 后起,
端口清理, RUN_ENV 含 DS4_TP_REVERSE_CONNECT=1)。

---

## 2026-05-31 双机 k4 比单机慢 — 根因定位 + TP all-reduce 全双工修复

**现象**: k4 (10GB, 进 16G RAM) 双机 TP decode ~3 t/s, 单机 ~10+ t/s。

**根因分解 (代码证据)**:
1. **每 token 3 次「GPU 排空 + 同步 all-reduce」硬屏障, 零重叠 (主因)** —
   `ds4.c:11124-11164` 对前 tp_layers(=3) 每层: tp_signal_after_batch →
   flush_commands(commit, 切断整 token 的单 CB 批) → tp_host_wait(排空 GPU 到本层) →
   tensor_read → all-reduce → tensor_write。单机是整 token 一个 CB 末尾等一次;
   双机把一个 token 切成 4 段串行, GPU 算时网络空、网络会合时 GPU 空, 无重叠。
2. **all-reduce 传输是严格串行交换 (本次修复点)** — `ds4_distributed.c` 旧
   `ds4_dist_tp_allreduce_f32`: 一方先发后收、另一方先收后发 ⇒ 2 个单向传输背靠背
   (连接方收完才发), 每屏障网络部分白白翻倍。
3. **element-split 对 k4 零收益** — 两机算同样 6 专家, 纯屏障开销 (执行记录已记)。
4. **混淆量**: 脚本 RUN_ENV 强开 EXPERT_OFFLOAD+NO_RESIDENCY, 但 k4 进 RAM 不需要;
   NO_RESIDENCY 下每个 flush 的新 CB 重新 wire 模型页 (见 [[metal_buffer_residency_per_buffer_granularity]]),
   与 TP 屏障恶性叠加。单机 10+ t/s 大概率是没开这俩 flag 测的 ⇒ 3 t/s 里 TP 占多少、
   offload 占多少现在混在一起。

**本次代码改动 (ds4_distributed.c)**:
- `ds4_dist_tp_allreduce_f32` 串行交换 → **全双工 poll 泵 `dist_tp_exchange`**: 一个
  poll 循环同时推本端帧(header+payload)、收对端帧, TCP 全双工下墙钟 2 串行传输 → ≈1 单向延迟。
  read 侧随时排空 ⇒ 任意 payload 大小无死锁, send_first 不再相关 (两端同路径)。
- TP socket 在 dist_tp_alloc 设 O_NONBLOCK (加 `#include <fcntl.h>`); 通用帧 helper
  (dist_write_full 等) 仍被协议其余部分用, 不动; TP 专用 recv_vec/send_vec 删除。
- **验证**: `make ds4` 0 warning; `ds4_dist_tp_selftest` (socketpair 双线程, 4096 float,
  无模型) **PASS** — 求和逐字正确 + 帧格式校验通过。两机需同步重编 (CORE_OBJS 共享 ds4_distributed.o)。

**诚实评估 (decision gate)**: 全双工只削减屏障的网络分量 (sub-ms 级), **不是 3 t/s 主因**;
主因是原因 1 的 GPU-drain×3 + 原因 4 的 offload/no-residency 暴露。要拿大头, 用户侧两步:
(a) k4 跑 TP 时 RUN_ENV 去掉 EXPERT_OFFLOAD+NO_RESIDENCY (k4 进 16G 安全) — 隔离并消掉 wire 开销;
(b) 认清 element-split TP 对能进单机的 k4 永远负收益, 要省内存/提速得换 layer-pipeline 拓扑 (每 token 1 跳)。
脚本可用 `RUN_ENV="DS4_TP_REVERSE_CONNECT=1" ./tools/tp_k4_speed.sh` 做 (a) 的对照。

---

## 2026-05-31 (续) 双机 k4 慢的真正主因裁决 + 脚本默认改 (offload off / tp_layers=1)

**物理天花板 (诚实)**: 单 token 自回归 decode 是顺序、延迟绑定的; k4 整个进单机 16G。
这种「能进单机」的模型, **两机物理上不可能让单 token decode 比单机快** —— element-split
让两机冗余算同样 attn/dense/shared (没切), 只切占比很小的 routed 专家, 却每 token 多付
tp_layers 次跨机屏障。dual k4 的上界 = 从下方逼近单机, 永不超过。两机价值在 82GB 全模型 + prefill。

**3 t/s 的真正主因 (修正之前"全双工是边角料"的定位)**: 不是 TP 同步, 是脚本给 k4 强开
`DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_NO_RESIDENCY=1`。k4 8.2G backbone 进得了 16G,
不需要 offload; 开了它 → 每次 TP flush 的新 CB 重新 wire 模型页 (见
[[metal_buffer_residency_per_buffer_granularity]]), 把每屏障税放大几十倍。单机 10+ t/s
是 resident(无 offload) 测的, 双机带 offload 测 → 不是同一配置, 3 vs 10 主要是 offload 差。

**脚本默认改 (tools/tp_k4_speed.sh)**:
- `RUN_ENV` 默认去掉 OFFLOAD+NO_RESIDENCY, 只留 `DS4_TP_REVERSE_CONNECT=1` (k4 常驻, 屏障廉价)。
  82GB 全模型时 env 覆盖加回。
- `TP_LAYERS` 默认 3→1 (k4 上每 tp-layer 是纯屏障税, =1 最小屏障)。
- 看门狗 `LOCAL/REMOTE_MAX_GB` 12→14 (offload off 后 k4 resident ~10-11G = 单机已安全足迹;
  14 不误杀又留 2G 给 OS, 不怼红线)。内存安全: 每机 = 单机 k4 的已证明 footprint, 非双载。

**预期**: dual k4 应从 3 t/s 升到逼近单机 (~8-10 t/s); 不会超过单机。若没到 10, 残余是
3 次→1 次屏障的 CB flush+event 延迟, 进一步只能换 layer-pipeline 拓扑 (每 token 1 跳) 或认账。
全双工 all-reduce (上一条) 仍保留, 削减屏障网络分量。**未跑模型** (用户手动跑, Ctrl+C 在手)。

---

## 2026-05-31 — mtp.md Phase 1 方案A 启动 (跨机 MTP 投机骨架)

**决定**: 用户选「方案A Phase1 骨架」(层切分 A:0-32/B:33-42 + MTP drafter 在持末层的 worker)。
诚实定位 (沿用 mtp.md §2/§8): 此骨架对 k4 decode 是**净成本** (无内存收益 + 多一跳),
价值是「大模型 + MTP」通用骨架; k4 单流真正提速靠 Phase2 方案B (B 纯 drafter 异步抢跑)。

**实施纪律**: 原子增量, 每步 `make` 编译验证, **全程不运行** (安全闸: 等用户双机终端手动跑,
Ctrl+C 在手; 两机共享 CORE_OBJS, 改完两机都要重编 + 传二进制)。

**增量拆分**:
- #1 协议帧 + `--mtp-role` 骨架: WORK 加 `DS4_DIST_WORK_F_DRAFT` 标志 + `accept_len`/`draft_cap`
  字段; RESULT 加 `draft_count`; to/from wire; CLI `--mtp-role worker`。memset 零初始化保证
  既有路径字节不变 (新字段=0 → 行为同旧)。
- #2 worker 出 final hidden 后调 `metal_graph_eval_mtp_draft` 产 K 候选, 打包进 RESULT 回传。
- #3 coordinator 把 [verified, draft_0..K-1] 走分布式批量前向 (复用 chunk 批通道) 跨机验证,
  逐位贪婪比对定 accept_len。
- #4 跨机 KV 回滚: 下一 WORK 带 accept_len, worker 截断自身层切片 KV; 滚动哈希纳入已接受前缀。

正确性判据 (Phase1 收尾): 双机+MTP 输出 == 单机+MTP 输出 (逐 token, --temp 0 --seed 1), 再看 t/s。

### 增量2 落地 (worker 出 final hidden 后调 MTP draft 回传) — 编译通过, 未运行

**ds4.c**:
- loader 放开: `mtp_for_worker_draft = (role==WORKER && mtp_draft_on_worker && load_output)`;
  原本 `role==NONE` 才载 MTP, 现允许 draft worker 载 MTP 模型 + 置 `mtp_ready`(图随之分配 MTP 状态)。
- `weights_model_map_spans` 加 `include_token_embd` 参数; draft worker(layer_start≠0) 补 token_embd
  进 residency span(MTP head 要从 base token_embd re-embed draft token)。代价 ~1-1.85 GiB 常驻。
- 新公开 API `ds4_session_mtp_draft(s, verified_token, pos, max_k, drafts, *out_n, ...)`:
  镜像单机递归(首 draft 用 cur_hc=末层 final hidden → mtp_state_hc; 后续 ping-pong
  mtp_state_hc/mtp_next_hc)。greedy-only。MTP 不可用时返 0+out_n=0(调用方回退普通 decode)。
  speculative 行的 mtp_n_raw 回滚留给增量4(accept_len)。

**ds4_distributed.c**:
- `dist_send_work_result` / `dist_worker_upstream_send_work_result` 加 `draft_tokens`/`draft_count`
  参数; draft token(uint32 htonl)写在 logits payload 之后; `result_fixed.draft_count` 标数量。
- worker eval 成功且 `local_output_logits && DRAFT 标志 && draft_cap>0` 时(持 state->mu 锁内,
  cur_hc 仍有效), 调 `ds4_session_mtp_draft` 取 K 候选, 随 RESULT 发送。
  verified=tokens[n-1], pos=pos0+n-1 (与单机一致: 输入 token 在自身位置)。

**安全性**: coordinator 尚未设 DRAFT 标志/draft_cap(增量3 才编排), 故 worker want_draft 恒 false,
**既有双机/TP 路径运行时字节不变**。token_embd 常驻只在 `--mtp-role worker` 时发生。

### 增量3+4 落地 (coordinator 跨机批量验证 + 跨机 KV 回滚) — 编译通过, 未运行

**核心安全性质 (决定了为何敢 blind 写)**:
1. 输出正确性由 **target argmax 门控** —— draft 再错只是验证不过/少接受, 绝不污染输出。
2. KV 正确性由 **滚动哈希不匹配 → dist_coordinator_rebuild_from_transcript 全量重建** 兜底。
→ 投机路径所有 KV/MTP-cache/回滚 bug 都是**纯速度问题, 绝不出错**。据此乐观实现 + fail-safe。

**ds4.c**:
- `ds4_session_verify_batch_argmax`: worker 切片逐行验证器, 跑 layer_start..final + output_head_batch
  → 读 K 行 logits (spec_logits, 复用单机批量验证器机制); check+commit timeline。
- `ds4_session_layer_slice_rollback(new_len)` / `ds4_session_layer_slice_len`: 截断 timeline 到
  new_len (position 环形 KV 下次 eval 覆盖陈旧行, = 单机 MTP 回滚做法 20714)。

**ds4_distributed.c**:
- WORK 加 `DS4_DIST_WORK_F_VERIFY`; `ds4_dist_spec_io` 结构穿过 eval_span/eval_remote_on_fd
  (普通调用传 NULL, 非投机路径字节不变); recv_result 修尺寸校验扣 draft 字节 + 读候选。
- worker: accept_len 回滚 (session 取得后/prefix 校验前, 用 spec_base_len+accept_len 截断+重算哈希);
  VERIFY 帧走 verify_batch_argmax 返 K 行 logits + 记 spec_base/spec_pending。
- coordinator `ds4_dist_session_eval_speculative`: 双轮 ——
  R1 DRAFT(eval first_token + 带上轮 accept_len 回滚 + 取 K drafts);
  argmax(logits)==drafts[0] 才进 R2 VERIFY(K 候选批量前向 → K 行 logits);
  逐行 argmax 链式接受定 m; 本地 owner KV 立即回滚到 p+1+m, worker 回滚 accept_len=m 推迟到下一帧。
- `--mtp-role worker` 在 coordinator 端置 state.mtp_draft(请求投机), 在 worker 端置载 MTP+token_embd。
  coordinator 只需 `--mtp-role worker`, worker 需 `--mtp FILE --mtp-role worker`。

**已知未验证假设 (双机跑时重点查)**:
- 投机仅 decode(pos>0): R1 走 eval_layer_slice decode 路径 → cur_hc=末层 final hidden (draft 前提)。
- 两机回滚位置一致: worker spec_base=R2 pos0=p+1, +accept_len=m == coordinator owner 回滚 p+1+m。
- R2 verify 帧 OUTPUT_LOGITS 来自 route 末跳 flag; worker is_verify 需 local_output_logits。
- MTP raw cache(mtp_n_raw) 跨投机周期未精确回滚 → 仅影响 draft 质量(=速度), 被 argmax 门控兜底。
- 带宽: R2 回传 K×vocab×4 (K≤16) logits; 典型 K=4 ≈ 2MB/周期。

**验证协议 (Phase1 收尾, 用户双机手动跑)**:
正确性: 双机+MTP 输出 == 单机+MTP 输出 (逐 token, --temp 0 --seed 1); 任何分歧查上面假设。
启动: coordinator `--role coordinator --layers 0:32 --mtp-role worker ...`;
      worker `--role worker --layers 33:output --mtp gguf/<mtp>.gguf --mtp-role worker ...`。
两机共享 CORE_OBJS, 必须同步重编 + 传二进制。本机已编译, **未运行** (等用户终端手动跑)。

### 测试脚本 + K-source bug 修复 (2026-05-31 续)

**修复 (写脚本时发现的真 bug)**: `ds4_dist_session_eval_speculative` 用 `ds4_engine_mtp_draft_tokens()`
取 K, 而该 accessor 门控 `has_mtp`(要求 role==NONE && mtp_ready) → coordinator(role=COORDINATOR,
不载 MTP) 恒返 0 → K<2 → 投机永远回退普通 decode, **投机根本不触发**。
加 `ds4_engine_mtp_draft_tokens_configured()`(不门控, 返 e->mtp_draft_tokens), 驱动改用它。

**网络拓扑结论 (零代码绕开 M1 出站 connect bug)**:
- 层切分 2 机解码唯一跨机 connect = worker→coordinator (`use_control_for_work` 下 WORK 走
  worker 主动连入的连接, `dist_coordinator_build_route_plan` 用 `dup(w->fd)`);
  coordinator 解码期零出站 (出站仅存/取盘 5033/5150)。
- → **M1 当 coordinator(只 accept) / 本机当 worker(本机出站连 M1, 通)**: 零 M1 出站, 零代码。
  本机(16GB) 持更小的 33:output 切片 + MTP, 内存也更省。不需要写层切分 reverse-control。

**新脚本 `tools/mtp_pipe_k4_speed.sh`** (区别于 tp_k4_speed.sh 的 TP 拓扑):
- M1 coordinator `--listen --layers 0:32 --mtp-role worker --mtp-draft 4 -p "<代码题>" --temp 0 --seed 1`
  (出结果+计时在 M1, 脚本 ssh 读回)。
- 本机 worker `--coordinator M1 --layers 33:output --mtp <draft.gguf> --mtp-role worker --mtp-draft 4`。
- 复用 rsync→M1 / 两边 clean+make ds4 (共享 CORE_OBJS 必须都重编) / 看门狗(两边同杀,只杀进程)
  / DS4_MEM_BUDGET_MB 预算闸。先起 coordinator(等 worker 入队), 再起 worker。
- 投机触发四要件齐: 两机 --mtp-role worker + 两机 --mtp-draft>=2 + worker --mtp FILE + --temp 0。
- **未运行** (会加载模型+实跑, 等用户确认内存预算后手动跑, Ctrl+C 在手)。

**正确性验证仍是 Phase1 收尾闸**: 双机+MTP 输出 == 单机+MTP 输出 (逐 token)。分歧查 5 条未验证假设。

---

## 2026-05-31 — mtp_pipe_k4_speed.sh 三连修复 (脚本 bug + 草稿模型路径 + M1 Pro coordinator GPU OOM)

**症状链。** `./tools/mtp_pipe_k4_speed.sh` 连续三次报错, 逐个定位:
1. `line 36: 草稿模型,: command not found` + `line 80: MTP_GGUF: unbound variable`
   — 第 36 行 `}` 与行内注释 `#` 之间漏空格。bash 只把*词首* `#` 当注释, 紧贴 `}` 的 `#`
   不是注释 → 整行变成「临时变量前缀 + 执行命令 `草稿模型,`」, 赋值只是失败命令的临时前缀,
   `MTP_GGUF` 从没进环境 → 后面 `set -u` 报 unbound。修: 补空格。
2. `本机缺草稿模型 gguf/ds4flash-k4-mtp.gguf` — 默认草稿模型路径写错; 本机实际 MTP draft 是
   `gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` (3.63 GiB)。修: 改默认 MTP_GGUF。
3. **真正的硬骨头** — 运行期 M1 Pro coordinator (192.168.1.2, 16GB):
   `Metal command batch failed: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)`
   → prefill 失败 → coordinator 退出 → cleanup 杀 worker (本机 worker `coordinator disconnected`,
   `missing layer 33` 是结果不是因)。

**根因 (3)。** block_count=43 (layers 0..42)。旧切分 0:32/33:output 把 **33 层全压 coordinator**
= 7.51 GiB 模型常驻。M1 Pro `iogpu.wired_limit_mb=0` (默认工作集 ~10.6G); vm_stat active 仅 ~5.6G
(物理够), 有效 GPU 上限 ≈ 16-5.6 ≈ 10.4G。7.51G 模型在此就爆 → 图瞬时缓冲 ~3G。footprint 只报
3.6G 是因为不计常驻 residency set (印证 metal_buffer_residency_per_buffer_granularity: 整 buffer 全 wire)。
M4 单机能跑 10G 常驻是因为后台占用更少、headroom 更大; 脚本错误假设两台 16GB 等价。

**修 (3): 数据驱动重切分, 不抬 iogpu (拒绝怼物理红线, 守 平稳>极限)。**
硬数据: 原始 coord=33 层爆; worker=10 层+MTP(3.63G) 健康 (~7G 富余)。每层 ~0.18 GiB,
embed/lm_head ~1.5G。coord 留 ~1.5G 余量 → 模型 ≤ 6.4G ≤ 27 层。worker 吸收剩余 16 层。
→ `SPLIT_COORD=0:26` (27 层 ~5.4G+token_embd), `SPLIT_WORKER=27:output` (16 层 ~4.4G+3.63G MTP)。
两台各 < 单机全模型足迹, 非双载。看门狗 14G + L1 预算闸 13.5G 不变。

**状态。** 脚本 3 处已改, bash -n 通过。**未运行** (会两机加载+实跑, 守 memory-safety gate, 等用户
确认后手动跑, Ctrl+C 在手)。risk: worker(M4) 现 ~8-9G + 图缓冲, 若它反过来 OOM 则把切分点再往后挪
(给 worker 更少层); coord 仍有余量可接更多层。跑后看两边 "tensor span GiB" 行校准 0.18/层 估算。

---

## 2026-05-31 (续) — 切分修好 OOM 后, 暴露 generation: 0.00 t/s (首-token-EOS)

**新症状。** 0:26/27:output 重切分后 M1 Pro coordinator 不再 OOM: 常驻 6.41 GiB, route ready,
**prefill 成功 10.47 t/s**, 但 `generation: 0.00 t/s`, 零生成文本, coordinator 干净退出 (无报错)。

**代码定位 (两条硬结论)。**
1. 机制: ds4_cli.c:1261 解码循环里, 首个 `ds4_session_sample` 采样的 token 若 == EOS 立即 break,
   generated 停在 0 → generation 0.00。投机路径 (1272) 在首 sample 之后才触发。
2. `ds4_engine_mtp_draft_tokens()` (ds4.c:17796) 在引擎未加载 MTP 时返回 0。coordinator 不加载
   MTP (仅 worker 载), 故它返回 0 → CLI 投机门 `mtp_draft_tokens>1` 在 coordinator 永远为假
   → 走普通解码。所以 generated=0 ⟹ **首 token 即 EOS ⟹ prefill 末位 logits 的 argmax = EOS,
   即 prefill logits 是错的**。

**顺带发现的真 bug (记下, 暂不修)。** 投机门用了 `ds4_engine_mtp_draft_tokens()`, 但 ds4.c:17804
专为 coordinator 准备了 `ds4_engine_mtp_draft_tokens_configured()` (返回原始 --mtp-draft, 不要求本地
载 MTP; 注释明说"coordinator 编排投机但不载 MTP, 需原始值")。CLI 一次性生成路径 (1272) 用错了访问器
→ 即使 logits 修好, coordinator 也永不发起跨机投机, MTP 形同虚设。修法: 1272(及 601 同构处) 改用
`_configured`。但这不解释首-token-EOS, 优先级在后。

**首-token-EOS 假设。** worker 日志: 主模型 27:output 映射后, 又映射 MTP 为 "37 overlapping shared
buffers"(重叠)。强假设: worker 载 MTP drafter 时其 token_embd/output 与主模型 output head buffer
重叠串扰, 污染 worker 回传的主模型 prefill logits → argmax=EOS。注意 prefill 路径与 MTP 无关、字节
一致 (ds4_distributed.c:2680 注释), 所以若纯管线 logits 本就错, NO_MTP 也会复现。

**已加隔离开关 NO_MTP=1** (脚本): 去掉两机全部 MTP 标志, 跑纯 layer-pipeline。
  - NO_MTP=1 能生成 → bug = MTP 污染 worker output head (真因, 进一步查 buffer 重叠映射)。
  - NO_MTP=1 仍 0 token → bug = layer-pipeline 末位 logits 回传本身。
**未运行** (守 memory-safety gate, 等用户确认。看门狗 14G 在位)。

---

## 2026-05-31 (续2) — generation:0.00 深挖: 代码路径全对, 需实测 logits 定位

**确认走的是哪条路径。** `./ds4 --role coordinator -p` 走 ds4_distributed.c:3972
`dist_run_coordinator_generation` (自包含 tokenize/prefill/decode), 非 CLI 循环。其解码循环
(4141-4149): 首 token 采样自 prefill 填的 `logits`, `==eos` 即 break。**此路径完全不调 MTP 投机**
(纯自回归 4162)。所以 generated=0 ⟺ prefill logits argmax = EOS。

**代码路径逐段核对, 结构全对:**
- 冷启动 → dist_coordinator_prefill_prompt (3888) → dist_coordinator_eval_span (2794)。
- coordinator: ds4_session_eval_layer_slice(local 0:26, output_hc=hidden, output_logits=false)
  → batch prefill (ds4.c:19959+), 读 g->batch_cur_hc 全 n_tokens 行进 hidden (20005)。✓
- worker: 写 input_hc→batch_cur_hc, 跑 27:42, output head 取 row n_tokens-1 → g->logits (19989-19998)。✓
- s->logits 与 ds4_session_sample 同 buffer (ds4.c:20268)。✓
- activation_bits 默认 32 (FP32 全精度, 非有损)。✓ enable_mtp 不改主 output head 路径。✓
结论: 纯读代码无法再定位; bug 在 Metal 内核细节 / 残留预算下 buffer 别名 / 或模型真吐 EOS。
**必须实测 logits。**

**现实障碍 (记下)。** 单机 k4 基线现在也 OOM: M4 PhysMem 13G 被其他 app 占用(非 ds4, 无残留进程),
仅 2.5G 空闲; 单机 k4 需 ~10G。⟹ 这不是 bug, 是宿主机内存被占。要跑单机基线需先腾出 ~8-10G。

**下一步候选 (待用户定):**
  A. 腾 RAM → 单机 k4 `--dump-logprobs` 基线 (最快, 确认正确首 token; 若单机也 EOS = prompt/模板问题非分布式)。
  B. 给 coordinator 加 `--dump-logprobs` 跑双机(带MTP) → scp 回 → 比对 top-20。直接看分布式 logits 是否乱。
  金标准 = A+B diff。两者都需用户动作/资源, 守 memory-safety gate 不自动跑。

---

## 2026-05-31 (续3) — 用户更正拓扑: 本机 M4 扛大部分层, M1 扛 MTP+少部分层 (脚本翻转重写)

**用户明确目标 (推翻之前的切分方向)。** 不是 M1 扛大头, 而是: 本机 M4 = 大部分层; M1 = MTP + 少部分
末段层。且"不管以前的 connect bug, 直接让脚本按此拓扑跑"。

**拓扑翻转 (角色+位置都换):**
  - 本机 M4 = coordinator (--listen 192.168.1.3:5599), 持 0:32 (33 层, 大部分) + token_embd,
    tokenize/sample, 跑 -p 一次性生成 (本机直接打印, 不再 ssh 读回)。
  - M1 = worker (--coordinator 192.168.1.3 5599 → 连本机), 持 33:output (10 层, 少部分) +
    output head + MTP drafter。MTP 必须跟 output head 同机 (mtp_for_worker_draft 硬约束), 故落 M1。
  - 数据流 M4(embed+0:32) → hidden → M1(33:output+head+MTP) → logits → 回 M4 采样。

**网络: 标准方向, 无需 reverse-connect。** layer-pipeline 唯一跨机 connect = worker→coordinator。
本拓扑 worker=M1 出站连本机 M4 (走标准方向)。证实 reverse-connect 只对 tp_enabled 生效
(ds4.c:19515-19529), layer-pipeline 不支持 —— 但本拓扑不需要它。

**事实核对。** 本机 M4 bridge0 = 192.168.1.3 (与 M1 192.168.1.2 同段, M1 可达)。M1 上 k4 + MTP
两 gguf 均在。前置检查改为 ssh 验 M1 上的 MODEL+MTP_GGUF。

**脚本改动 (tools/mtp_pipe_k4_speed.sh 整体重写):** coordinator 改在本机跑(原在 M1); worker 改在
M1 跑(原在本机); COORD_IP 192.168.1.2→192.168.1.3; 看门狗本机=coordinator/M1=worker; 结果直接
读本机日志。保留 NO_MTP 隔离开关、两边 clean+make、看门狗两边同杀。bash -n 通过。**未运行**。

**遗留未解。** generation:0.00 (首-token-EOS) 的真因尚未定位 (代码路径全对, 需实测 logits)。新拓扑
是否仍 EOS 未知 —— 若仍 EOS, 说明与哪台机器当 coordinator 无关, 是 layer-pipeline logits 本身,
届时拿 --dump-logprobs 实测。**风险: 本机 M4 现扛大部分层 ~8.4G, 而 M4 当前被其他 app 占 13G,
GPU 工作集会 OOM —— 用户需先腾内存 (单机 k4 同因 OOM 已证)。**

---

## 2026-05-31 (续4) — 翻转拓扑后 worker 连不上: 真因 = macOS 本地网络隐私 (非网络/非代码)

**症状.** 翻转拓扑 (M4=coordinator listen 192.168.1.3, M1=worker connect) 后: coordinator 卡
"missing layer 33"; M1 worker 日志刷 `unable to connect to 192.168.1.3:5599: No route to host`。

**逐层排除 (全部实测):**
- M1→M4: ping 通, `nc 192.168.1.3 5599` 通, 路由表 interface=bridge0。⟹ 网络好的。
- ds4 connect debug (DS4_TP_CONNECT_DEBUG=1): try family=IPv4 dst=192.168.1.3:5599 → errno=No route to host。
- DS4_TP_SRC_IP=192.168.1.2 绑源: 仍 EHOSTUNREACH。nc -s 192.168.1.2 却通 ⟹ 非 bind 问题。
- C 复现器 (/tmp/cx, /tmp/cx2) 逐字复刻 dist_connect_endpoint_once (getaddrinfo+socket+全部
  socket option mask=31+IPPROTO_TCP): **全部 connect OK**。同一时刻 ds4 仍挂 ⟹ ds4 二进制特有。
- IP_BOUND_IF en0=Connection refused, bridge0=OK ⟹ 非出接口选择。

**真因 (M1 system log 铁证):**
  `ds4: (Network) libinfo check path: unsatisfied (Local network prohibited), interface: bridge0`
  = macOS Local Network Privacy 禁止 ds4 二进制走 bridge0 本地链路。按 cdhash 授权; 脚本
  `make clean && make ds4` 每次换新二进制 → 授权重置; ssh 无头启动无法弹授权框 → 永远被拒。
  nc/cx 有权限故通。非网络、非 ds4 代码 bug。

**修复方向 (待定):**
  1. 即时: M1 GUI System Settings → Privacy & Security → Local Network → 开启 ds4 (或 Terminal)。
     缺点: 每次 rebuild 换 cdhash 又被重置。
  2. durable headless: 给 layer-pipeline 加 reverse-connect → M1 只 listen/accept (inbound 不需
     此权限), M4 (可交互授权) 全部出站。一举解决 (且正好是用户要的 M4 大层/M1 MTP 布局)。
  3. 或对 ds4 用稳定签名身份 codesign + 授权一次 (跨 rebuild 保持)。
**当前 debug worker (pid 63067) 仍在 M1 重试; coordinator(本机 pid 99691) 仍 listen。需清理。**

---

## 2026-05-31 (续5) — 实现 layer-pipeline reverse-connect (修本地网络权限拦截, 一劳永逸)

**为什么需要 (不是端口/网络问题).** M1 网络/端口都正常 (nc/ping/C复现器全通); 唯独 ds4 这个二进制被
macOS Local Network Privacy 拒绝从 bridge0 出站 (按 cdhash 授权, rebuild 重置, ssh 无头无法弹框)。
两机各自端口都明确, 普通"一个连另一个"本该 trivial —— 复杂仅来自这个 OS 拦截。两条出路: M1 GUI
授权(每次 rebuild 失效) 或 让 M1 永不出站(reverse-connect)。用户选后者(一劳永逸)。

**改动 (聚焦: 只翻控制通道方向; 数据通道本就是 coordinator→worker, 不动).**
- dist_run_worker (ds4_dist_run→worker 入口): reverse 时开 control listener(opt->listen_port) +
  data listener(自动端口), accept coordinator → 发 HELLO → read loop。worker 零出站。
- ds4_dist_session_create (CLI -p 的 coordinator 真实路径): reverse 时不 listen, 起
  dist_coordinator_reverse_connect_main 线程 dial worker(opt->coordinator_host:port) → 建 ctx
  (peer 用 getpeername, 数据通道地址天然正确) → inline 跑 dist_coordinator_client_main。
- dist_validate_options: 加 dist_rev (非 TP 的 reverse), coordinator 用 --coordinator / worker 用
  --listen。listen_fd 在 reverse 下钉 -1 (calloc 默认 0 会误关 stdin)。线程查 state->shutting_down 退出。
- 开关: DS4_DIST_REVERSE_CONNECT=1 (兼容 DS4_TP_REVERSE_CONNECT)。

**验证.** make 全绿 (ds4/server/agent 都链接 ds4_distributed.o)。运行自检: reverse coordinator 打印
"reverse-connect dialing worker 127.0.0.1:5604" 并主动拨(无 listener 时 Connection refused 重试) ✓;
forward 无 env 仍 "listening on 127.0.0.1:5605" ✓ (字节行为不变)。worker reverse 路径与 forward 对称,
编译过, 端到端待双机脚本验证 (M4 内存被占, 本地 loopback 双载放不下)。

**脚本 tools/mtp_pipe_k4_speed.sh** 已配 reverse: M1 worker --listen 192.168.1.2 5599 先起;
M4 coordinator --coordinator 192.168.1.2 5599 后拨; RUN_ENV=DS4_DIST_REVERSE_CONNECT=1。首跑 M4 弹
本地网络授权框点允许即可 (M4 交互)。M4 扛大部分层 ~8G, 需先腾内存。

## 2026-05-31 双机 layer-pipeline+MTP worker GPU-OOM 真因定位 + chunk 绕过验证

**报障**: tools/mtp_pipe_k4_speed.sh 跑 "本机内存爆了"。

**真因 (加 [diag] 日志后实测, 非臆测)**:
- 误判纠正: 不是主机 RSS 爆, 也不是 coordinator。是 M1 Pro worker 的 **GPU 命令缓冲 OOM** (kIOGPUCommandBufferCallbackErrorOutOfMemory)。coordinator 的 "prompt processing failed: metal layer-slice failed" 是 worker 回传错误字符串的下游症状。
- M1 Pro `recommendedMaxWorkingSetSize = 10.67 GiB` (GPU 工作集天花板)。
- 模型常驻: base 切片(33:output) 24 views 3.33 GiB + MTP 1 view 3.55 GiB = 25 views **6.88 GiB** (两套并入同一 g_model_residency_set)。
- prefill 命令批执行时 device currentAllocated 暴涨到 **11.23 GiB > 10.67** → 越线 OOM。多出的 ~4.35 GiB 是 prefill scratch 池 (按 prefill_chunk=4096 预分配) + 层批中间张量, 而 prompt 仅 25 token, 几乎全浪费。
- 旁证: L1 budget gate 原先漏算 MTP (只 gate base 3.33G); 已补一行 gate 让 MTP 3.55G 现形 (ds4.c MTP map 前)。

**实测绕过**: `DS4_METAL_PREFILL_CHUNK=512` → worker 不再 OOM, **coordinator prefill 21.53 t/s 跑通** (过 20 t/s 闸的 prefill)。模型常驻仍 6.88G 不变, 仅 scratch 池缩小。

**遗留**: generation 0.00 t/s / decode_tokens=0 —— prefill 通但解码 0 产出, 两边无报错, 独立问题 (MTP 跨机解码路径), 待查。

**埋点 (本次新增, 暂全程打)**: ds4_metal.m residency 建立后打 wired views/GiB + recommendedMax + currentAllocated; CB 失败处打失败瞬间 currentAllocated vs recommendedMax。ds4.c MTP map 前补 L1 gate。待定: 是否收进 DS4_METAL_DIAG 开关。

**永久修复方向 (未做, 待确认)**: layer-slice worker 的 prefill scratch 按实际 WORK chunk 缩放, 或 distributed worker 默认小 chunk; 对单机/非分布式零影响。

## 2026-05-31 generation=0 定位: k4 退化模型即时 EOS, 非双机 bug

**加 DS4_DECODE_DIAG 日志 (ds4_cli.c 采样循环 + ds4.c ds4_session_sample) 实测**:
- 双机 coordinator: `[decode-diag] logits argmax=1 val=16.8693 nonfinite=0/129280` + `sample#0 token=1 eos=1 max_tokens=128 mtp_draft=0`。
  → 首 token = argmax = **1 = EOS**, logits 完全健康 (无 NaN/inf, 强 logit 16.87), 故 ds4_cli.c:597 第一次采样即 break → 0 产出。
  → mtp_draft=0 证实 coordinator 上 MTP 投机是死代码 (ds4_engine_has_mtp 对 distributed 角色恒 false), 走普通 ds4_session_eval。
- 单机同 prompt 同 k4 模型也 generation 0.00 t/s (单机走 ds4_engine_generate 路径, 未打 diag, 但同样 0 产出)。
- 模型加载行: `reduced-expert model: keep-map over 43 layers (min kept 4 of 256)` —— gguf/ds4flash-k4.gguf 是 256 专家只留 4 个的极度裁剪测速骨架, 输出退化, 必即时 EOS。

**结论**: generation=0 不是双机解码 bug。双机管线忠实复现单机 k4 行为 (首 token=EOS)。解码链路正确。

**对测速目标的影响**: k4 退化模型 + 真 prompt 永远第一步 EOS, 测不到 decode t/s。ds4 当前无 --ignore-eos/min-tokens 开关。要测双机 decode 速度需加 --ignore-eos (或 DS4_IGNORE_EOS env) 在 ds4_cli.c:597/633 两处 EOS 判断加门控强制生成 N token。待用户确认是否加。

**遗留埋点**: DS4_DECODE_DIAG (ds4_cli.c + ds4.c), DS4_METAL 残留 [diag] (residency/CB-fail), 均 env 门控/低噪, 待定是否收编进统一 DS4_*_DIAG 开关。

## 2026-06-02 q2 双机 layer-slice + MTP worker 内存安全补丁

**范围**: 为 `tools/mtp_pipe_k4_speed.sh` 默认 q2 双机场景做最小修复：本机 coordinator 0:32（33 层, 约 12GB 预算），M1 worker 33:output（10 层 + output + MTP, 约 8GB 预算）。用户自己运行脚本；本次不加载模型、不跑双机 smoke。

**问题定位**:
- 当前 `--layers` load-slice 会把切片内 `ffn_gate_exps/up_exps/down_exps` 全部纳入 resident spans；q2 routed experts 是模型主体，MTP worker 再叠 3.5GiB MTP，8GB 目标会被 full-resident expert views 撞爆。
- `DS4_METAL_EXPERT_OFFLOAD` 原只在非 layer-slice 分支生效；双机 layer-pipeline 没有 A3 按需专家加载。
- Metal warmup 原会触碰所有 model views；对 non-resident expert views 也 warm 会把冷专家页提前扫进来，违背按需加载目的。

**落地**:
- `ds4_metal.m`: 移植 mine 分支 A3 的最小形态：`DS4_METAL_EXPERT_OFFLOAD=1` 时，decode 读回 6 个 selected expert id，prefill/MTP verify 对 `n_tokens * topK` selected ids 去重，然后只 memcpy 活跃 expert slots 到 resident compact scratch (`gate/up/down`)；selected ids 原地改写为 compact slot，非 A3 路径不变。
- `ds4_metal.m`: warmup 跳过 `resident_hint=false` 的 routed expert views，避免启动时扫冷 expert。
- `ds4.c`: 新增 layer-slice split spans；`load_slice && DS4_METAL_EXPERT_OFFLOAD` 时 backbone/embedding/output/shared expert 常驻，`ffn_*_exps` 只 wrap 不加入 residency set。L1 gate 按 base resident + MTP resident 累计预算检查。
- `tools/mtp_pipe_k4_speed.sh`: 默认 q2-imatrix、`SPLIT_COORD=0:32`、`SPLIT_WORKER=33:output`、12GB/8GB 预算；默认 RUN_ENV 带 `DS4_DIST_REVERSE_CONNECT=1 DS4_METAL_PREFILL_CHUNK=512 DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_NO_MODEL_WARMUP=1`。Ctrl+C/EXIT 继续两边同杀，只杀进程不删文件；远端优先 kill 本脚本写入的 worker PID，再 fallback pkill。

**安全边界**:
- 仍保持 GGUF mmap-backed；没有 eager copy 整模型。
- A3 会引入每层 CB sync + CPU memcpy，速度可能慢；本补丁目标是先保证 q2 双机不崩并能输出 token。
- 若 10 层 + MTP 的 resident backbone 仍超过 8GB，L1 gate 应拒绝启动；可用 `SPLIT_WORKER=36:output` 或 `NO_MTP=1` 隔离。
- 本次只做编译/脚本语法验证，不运行任何会加载模型的命令。


**非模型验证 (2026-06-02)**:
- `make ds4` 通过（只编译，不加载模型）。
- `bash -n tools/mtp_pipe_k4_speed.sh` 通过。
- 未运行 `./ds4` / 未启动双机脚本 / 未 ssh 加载模型；脚本按用户要求留给用户自己跑。

---

## 2026-06-05 — Distributed MTP coordinator gating fix (code-only, no model run)

**Context.** User reported `tools/mtp_pipe_k4_speed.sh` showed no speedup from MTP and only ~0.54 t/s while testing q2 with expert offload/pool ideas under strict 12GB/8GB budgets.

**Finding.** The distributed coordinator did not load the MTP GGUF by design (the final-layer worker does), but `ds4_engine_mtp_draft_tokens()` returned a nonzero draft width only when `ds4_engine_has_mtp()` was true. The normal CLI decode loop uses this accessor to decide whether to call `ds4_session_eval_speculative_argmax()`. Therefore `--mtp-role worker --mtp-draft N` on the coordinator could silently stay on the plain distributed one-token decode path, even though the worker loaded the MTP model.

**Patch.** `ds4.c`: make `ds4_engine_mtp_draft_tokens()` return the configured draft width for a distributed coordinator with `distributed.mtp_draft_on_worker=true`. Single-machine behavior is unchanged: non-distributed still requires a loaded MTP model. This enables the existing `ds4_dist_session_eval_speculative()` path for the script topology (coordinator orchestrates, worker drafts/verifies).

**Validation.** `make ds4` passes. No model load / no dual-host run performed here; user will manually validate with `tools/mtp_pipe_k4_speed.sh`.

**Expected impact.** This fixes a “MTP flag present but not entered” bug. It does not change the bigger physics: q2-full expert offload is still dominated by per-token expert CPU-gather/page-copy and backbone bandwidth; MTP only helps if the worker draft acceptance is high enough and the verification batch avoids extra copy/rollback overhead.

### Follow-up run — `tools/mtp_pipe_k4_speed.sh` actually exercised MTP but got slower (2026-06-05)

Ran on the two machines from this host (user requested local定位):

1. `REMOTE=192.168.1.2 NPRED=16 CTX=4096 DRAFT=4 DS4_DECODE_DIAG=1 DS4_MTP_SPEC_LOG=1 DS4_MTP_TIMING=1 EXPERT_PROFILE_INTERVAL=16 tools/mtp_pipe_k4_speed.sh`
   - Coordinator log now shows `mtp_draft=4`, so the ds4.c gating fix worked and the decode loop entered the speculative path.
   - Worker loaded MTP: `MTP support model loaded ... (draft=4, distributed worker drafter)`.
   - Generated text was sane prefix text, but speed was `prefill: 0.56 t/s, generation: 0.24 t/s`.
   - Decode diagnostic sample indices jumped `0 -> 2 -> 6 -> 11 -> 14`, meaning speculative calls committed multiple tokens; the problem is not zero acceptance.
   - Worker profile exposed extra MTP/offload work: after final-layer q2 expert profiles, MTP layer showed `one_expert=13.500 MiB`, and copy grew to ~16.6s by 112 MoE calls for this short run.

2. Baseline `REMOTE=192.168.1.2 NPRED=8 CTX=4096 NO_MTP=1 DS4_DECODE_DIAG=1 EXPERT_PROFILE_INTERVAL=16 tools/mtp_pipe_k4_speed.sh`
   - Speed: `prefill: 0.61 t/s, generation: 0.36 t/s`.

**Diagnosis.** Distributed MTP is now active, but it is slower in the full-q2 expert-offload topology. The accepted draft count is not the blocker; the blocker is that every speculative round adds MTP model routed-MoE work on the M1 plus K-token target verification, and both paths still use A3 CPU-gather/scratch for routed experts. Tiny-batch verification expands the active expert union and copies many experts, so the “read weights once for K tokens” assumption does not hold under expert offload. MTP can help only after the target+MTP routed experts are resident/repacked enough that verify cost is close to one-token cost; with full q2 offload it amplifies the existing copy bottleneck.

### Follow-up attempts per user request — MTP resident/direct, move output head, real expert pool (2026-06-05)

User requested not just analysis: try the concrete knobs one by one, keep MTP resident on the other host, open the real expert pool, and move non-essential local memory to the other machine.

**A. Keep MTP resident/direct instead of A3-copying its Q4 experts.**
- Patch: `ds4_metal.m` now gates A3 expert offload to the full-q2 target layout only (`IQ2_XXS` gate/up + `Q2_K` down). Non-q2 routed tensors, including the MTP Q4_K support model that is already mapped resident on the worker, stay on the direct model-buffer path instead of being gathered again into A3 scratch.
- Build: `make ds4` passed.
- Test: `DRAFT=4`, profiling/source-cache/pools off for a cleaner speed check.
- Result: still slower: `prefill 0.57 t/s, generation 0.21 t/s` vs no-MTP baseline around `0.32-0.33 t/s`. This removed the obvious MTP-Q4 scratch-copy mistake, but Scheme-A MTP still adds remote draft + K-token target verify work on the same M1 worker GPU and does not win under full-q2 offload.

**B. Move coordinator output head off the local machine.**
- Patch: `ds4.c` distributed layer-slice loading no longer forces `load_output=true` just because the role is coordinator. Only the role with `--layers ...:output` loads the output head. In the current script that is the M1 worker.
- Build: `make ds4` passed.
- Test: no-MTP smoke.
- Result: correctness/smoke OK; coordinator startup changed from `layers 0:33+output` to `layers 0:33`, resident backbone dropped from ~6.78 GiB to ~6.26 GiB (about 0.52 GiB freed locally). Speed stayed about the same: `prefill 0.72 t/s, generation 0.32 t/s`.

**C. Open real expert pool.**
- Test env: `LOCAL_EXPERT_POOL_MB=2048`, `REMOTE_EXPERT_POOL_MB=512`, no MTP. This allocated real pools: coordinator 303 q2 expert slots (~2.0 GiB) for layers 24:33; worker 75 slots (~506 MiB) for layers 34:42.
- With profiling on: coordinator pool reached ~43.5% hit by the short run end; worker ~31.6% hit. But speed collapsed to `generation 0.13 t/s`.
- With profiling off (same pools): still `generation 0.13 t/s`.
- Diagnosis: current real expert pool is CPU-filled `MTLStorageModeShared`. It removes some source mmap reads on hits, but misses still memcpy multi-GiB, and hits make MoE kernels read from a large Shared pool rather than compact hot scratch. For this short/full-q2 workload the pool is slower than the compact scratch path despite nonzero hits. Do not enable by default as-is.

**D. Source-cache default.**
- Script default changed to `LOCAL_EXPERT_SOURCE_CACHE_MB=0` because the source-page cache mostly adds bookkeeping/madvise overhead for this run and does not remove the A3 scratch copy. It remains opt-in via env.

**Current status.**
- Working code fixes kept: distributed MTP gating, coordinator does not load output head, MTP Q4 experts bypass A3 offload.
- Real q2 expert pool tested and found negative for this topology; leave opt-in, not default.
- Remaining high-upside route is not Scheme-A layer-pipeline MTP; it is a different architecture: remote pure asynchronous drafter queue (Scheme B) or physical expert repack/cut so target verify no longer uses full-q2 offload copies.

---

## 2026-06-06 — q2 expert LRU pool + predictor async prefetch (code-only)

**Context.** User restated hard requirements: q2 GGUF only, two 16GB Macs over Thunderbolt, local <=12GB and remote <=8GB, 200k context target, Claude Code usable output, and must keep the LRU resident expert pool plus predictive asynchronous prefetch. Prior q2 speed stayed around 0.54 t/s; logs showed the bottleneck was routed-expert CPU gather/copy, not routing correctness.

**Log finding.** Latest `/tmp/mtp_pipe_coord.log` and M1 worker log both showed A3 expert copy dominating wall time: coordinator copied ~169.7 GiB of q2 experts in ~226.8s for a short run; worker copied tens of GiB. With a small 1GiB simulated LRU the coordinator hit rate was 0% because one full layer-sweep active set already exceeds the cache; worker later reached ~55% simulated hits. Plain source-cache/pool v1 was not enough because hits still read a big Shared pool and misses stayed synchronous.

**Patch.** `ds4_metal.m` now keeps the real q2 expert pool as a mandatory LRU pool when `DS4_METAL_EXPERT_POOL_MB` is set, and adds a background pthread prefetcher:
- pool entries track `ready/loading/busy` so the GPU never reads a slot while the prefetcher overwrites it;
- foreground MoE misses still synchronously fill LRU slots for correctness;
- after each MoE call, predictor enqueue prefetches for future layers using (1) current active expert ids for adjacent layers, (2) each next layer's previous active ids (decode temporal locality), and (3) profiler hot experts when available;
- asynchronous fills use the same resident pool and LRU victim path, protected from evicting active/loading slots;
- logs now report real pool hit/miss, inflight waits, prefetch hit/miss/drop/copy, sync copy ms, and background copy ms.

**Script defaults.** `tools/mtp_pipe_q2_speed.sh` now enables a bounded real pool by default under the stated budgets: local 2048MiB, remote 384MiB, `CTX=200000`, with `PREFETCH_LOOKAHEAD=1`, `PREFETCH_TOP=10`, `HOTLOCK_TOP=10`, queue 4096. Env overrides remain available. This intentionally preserves LRU residency and adds predictor async prefetch; source-cache remains opt-in.

**Validation.** `make ds4` passes. `bash -n tools/mtp_pipe_q2_speed.sh` passes. No model run / no dual-host timing run performed here; user will validate manually with `tools/mtp_pipe_q2_speed.sh`.

**Expected impact / caveat.** This removes the previous "pool is only synchronous" limitation and can convert repeated/hot expert misses into background copies. It cannot change the hard lower bound that cold q2 routed experts are ~6.75MiB each and must be read at least once; if predictor accuracy is low or pools are too small, generation remains copy-bound. Check `expert-pool live/summary` in both logs first: useful runs should show nonzero `hit`/`pf_copy` and low `wait`; high `pf_drop` or high `wait` means tune pool MB / prefetch top down/up.

---

## 2026-06-06 — q2 双机测速现状排查 + 默认脚本回退到稳定基线

**用户目标/方案.** 在 M4 16GB Mac mini + M1 Pro 16GB MacBook（硬预算 mini≤12GB / worker≤8GB）上，优先 `gguf/` 下 q2 完整模型，探索“本机专家池内存保活 + LRU + 预测异步预取”提升专家命中率，并用 `tools/mtp_pipe_q2_speed.sh` 验证；此前版本约 0.58t/s。

**本轮验证环境/命令.** 多轮 `tools/mtp_pipe_q2_speed.sh`，`CTX=200000 NPRED=32/64/128` 小样本，默认拓扑 `SPLIT_COORD=0:33`、`SPLIT_WORKER=34:output`，q2 完整模型 + A3 expert offload，MTP 草稿模型在 worker。所有运行都有脚本两边同杀 watchdog 与 `DS4_MEM_BUDGET_MB`，未出现 OOM/预算拒绝。

**发现 1 — 真实 GPU expert pool 当前是负收益，不宜默认打开。**
- 默认打开真实 pool（coord 2GiB, worker 384MiB, lookahead=1/top=10/hotlock=10）跑到 **generation 0.15t/s**；日志显示 coordinator pool hit≈51%，但 worker pool hit≈4%，async prefetch copy 达数十 GiB、明显抢统一内存带宽。
- 关异步预取但扩大真实 pool（coord 2304MiB, worker 768MiB, lookahead=0）仍只有 **0.14t/s**。根因不是单纯 prefetch：真实 pool 首轮/低命中阶段要同步复制大量 expert 到 Shared MTLBuffer，并且 pool slot 作为 GPU 读源时存在跨层复用/同步成本；当前实现不适合做默认测速路径。
- 结论：真实 expert pool 保留为显式实验开关；默认关闭，避免从 0.35t/s 退化到 0.14–0.15t/s。

**发现 2 — distributed MTP 在当前 M1 worker 上端到端负收益。**
- 修复/确认了 coordinator 进入分布式投机路径的 gating：`ds4_engine_mtp_draft_tokens()` 对 coordinator + `--mtp-role worker` 返回配置 draft 宽度；coordinator 不再加载 output head（只有 output worker 持 output）。
- 加了 distributed MTP 摘要统计与自适应降级日志：`dist-mtp summary: calls/round1/verify/first_hit/accepted/drafts/tok_per_call/disabled`。
- 实测 `DRAFT=2`：接受率很高（示例 `tok/call=2.46`, `first_hit=76.92%`, `draft_accept=73.08%`），但 M1 上加载/运行 MTP support model + VERIFY 批处理使端到端 **generation 0.28t/s**，低于纯 pipeline。
- `DRAFT=4` 更差（此前 0.23t/s 或更低）。结论：MTP 不是当前硬件/实现下的默认加速路径；默认 `NO_MTP=1`，保留 `NO_MTP=0 DRAFT=2` 做实验。

**发现 3 — profiler 和 footprint watchdog 会拖慢测速。**
- `EXPERT_PROFILE=1` 会额外统计全层专家命中率，能诊断但会影响速度和日志量；默认改为 `EXPERT_PROFILE=0`。
- `footprint` 每秒采样 phys_footprint 非常重，会明显拖慢测速；默认恢复轻量 `ps RSS` watchdog，保留 `MEM_WATCH_MODE=footprint` 作为慢速诊断开关。真正模型常驻硬预算仍由 `DS4_MEM_BUDGET_MB` + L1 resident-budget gate 控制。

**当前稳定默认脚本参数（已改 `tools/mtp_pipe_q2_speed.sh`）。**
- `NO_MTP=1`（默认纯 pipeline；MTP 显式实验）
- `EXPERT_PROFILE=0`
- `LOCAL_EXPERT_SOURCE_CACHE_MB=2048`, `REMOTE_EXPERT_SOURCE_CACHE_MB=1024`
- `LOCAL_EXPERT_POOL_MB=0`, `REMOTE_EXPERT_POOL_MB=0`, prefetch/hotlock=0
- `MEM_WATCH_MODE=rss`（可手动 `footprint` 诊断）

**最新稳定结果.**
- 默认等价配置（纯 pipeline + profile off + source-cache local 2GiB / worker 1GiB）: **prefill 0.63t/s, generation 0.36t/s**，输出文本正常，worker RSS 约 5.8GiB，未触发 8GiB 上限。
- 无 source cache 对照: **prefill 0.71t/s, generation 0.33t/s**。source-cache 对 decode 有小幅正收益，但 prefill略慢；默认保留 source-cache 因目标更重 decode/Claude Code 交互。
- 默认 MTP 实验 (`NO_MTP=0 DRAFT=2`)：**prefill 0.59t/s, generation 0.28t/s**，MTP 接受率高但端到端慢。

**代码状态.**
- `ds4_distributed.c`: 加 distributed MTP 统计、自适应降级、summary；不改变纯 pipeline 路径。
- `ds4.c`: coordinator MTP gating + 不再让 coordinator 在层切分时加载 output head。
- `ds4_metal.m`: 当前已有真实 expert pool/LRU/async prefetch 实验代码；经本轮验证默认不应开启。
- `tools/mtp_pipe_q2_speed.sh`: 改为 q2 默认稳定基线并保留实验开关。
- `make clean && make ds4` 通过。

**结论.** 当前两机 q2 完整模型 200k 可跑但速度仍只有 **~0.36t/s decode**，离 20t/s 差两个数量级；本轮提出的“真实专家池 + LRU + 预测异步预取”在此实现中未提升，反而退化。下一步若继续优化，应先攻 A3 CPU-gather 本身（减少/并行化 expert memcpy、专用 per-layer hot expert 固定池且避免 GPU slot 复用 hazard）或回到物理裁剪/更小模型路径；不应把当前真实 GPU pool 当默认路径。

## 2026-06-06 — q2 expert pool predictor retune (code-only; user validates)

- Re-read latest logs: stable q2 pipeline still around 0.54 t/s with source-page LRU only; real pool/prefetch previous versions were slower because adjacent-layer prefetch copied tens of GiB on the foreground memory path and foreground waited on in-flight prefetches.
- Kept the required real LRU resident expert pool + predictor async prefetch, but changed the default/policy to a conservative decode-local design:
  - expert pool is enabled by default in `tools/mtp_pipe_q2_speed.sh` within budgets: local 768MiB, remote 256MiB;
  - predictor prefetch defaults to same-layer temporal reuse (`PREFETCH_SELF=1`) and disables adjacent-layer flooding (`PREFETCH_ADJACENT=0`);
  - misses do not wait for an in-flight background copy by default (`WAIT_INFLIGHT=0`): foreground falls back to A3 scratch for correctness, so the prefetcher can only help future tokens and cannot stall the current token;
  - `ADMIT_AFTER=2` avoids filling the pool with first-seen cold experts; the first observation queues async prefetch for a potential next-token hit;
  - summary logs now always print expert-pool/source-cache lines even with `EXPERT_PROFILE=0`, including `admit_skip`, `pf_copy`, `sync_copy`, and `wait`.
- Metal implementation changes: added env knobs `DS4_METAL_EXPERT_POOL_PREFETCH_SELF`, `..._PREFETCH_ADJACENT`, `..._WAIT_INFLIGHT`, `..._ADMIT_AFTER`; predictor accounting now increments per-expert hot counts on foreground requests; in-flight prefetch hits are no longer synchronously waited unless explicitly enabled.
- Validation performed here: `bash -n tools/mtp_pipe_q2_speed.sh`; `make ds4` passes. No dual-host/model run was performed here; user will validate manually with `tools/mtp_pipe_q2_speed.sh`.


### Immediate fix after user reported 0.21 t/s (2026-06-06)

- User validation showed the conservative pool still dropped decode to 0.21 t/s. Logs made the cause explicit: local pool `sync_copy=16.5s` + background `pf_copy_ms=52.9s`; worker `pf_copy=5668` for only 512 calls. Even without waits, the pool path still synchronously copied misses into Shared pool and the self-prefetcher over-copied.
- Patch: added `DS4_METAL_EXPERT_POOL_FOREGROUND_FILL` (default off in q2 script). With it off, a pool miss never synchronously fills the resident pool; foreground immediately falls back to the proven A3 scratch path, while the async predictor may fill future-token entries. This keeps the required LRU resident pool and predictive async prefetch, but prevents the pool from making the current token slower than baseline.
- Script now passes `DS4_METAL_EXPERT_POOL_FOREGROUND_FILL=0` by default and logs `fg_fill=0`. `WAIT_INFLIGHT=0` remains default. Validation: `bash -n tools/mtp_pipe_q2_speed.sh`; `make ds4` passes. No model run here.


### Root cause correction: prefetch_evict=off was still evicting (2026-06-06)

- User reported unchanged/slower 0.18 t/s. Latest logs proved `foreground_fill=off` but `pf_evict=off` still produced `pf_copy=3314` and `pf_drop=0`. That is impossible for a no-evict fill-only policy with only 113 slots, so the code still evicted despite the flag.
- Root cause: `ds4_gpu_expert_pool_victim()` evicted an LRU slot before the caller checked `g_expert_pool_entries[slot].used`; by then the entry had already been cleared, so the no-evict guard never fired. Predictor therefore continued to churn the entire LRU pool and copy ~37s of experts.
- Fix: prefetch worker now uses a true free-slot scan when `DS4_METAL_EXPERT_POOL_PREFETCH_EVICT=0`; it never calls the eviction helper in no-evict mode. If no free slot exists it increments `pf_drop` and does not copy.
- Also lowered default q2 pool/predictor pressure: local 384MiB, remote 128MiB, `PREFETCH_TOP=2`. LRU pool + predictor async prefetch are still present, but prefetch copy is bounded by free slots and cannot churn resident experts.
- Validation: `bash -n tools/mtp_pipe_q2_speed.sh`; `make ds4` passes. No model run here.


## 2026-06-07 — q2-full 0.66t/s bottleneck patch: per-tensor direct expert reads + layer-slice KV/scratch shrink

**Context.** User reported latest `tools/mtp_pipe_q2_speed.sh` validation reached only ~0.66 t/s against the hard goal (two 16GB Macs, mini ≤12GB / MacBook ≤8GB, 200K ctx, must use `gguf/` q2 full model). Latest `/tmp/mtp_pipe_coord.log` still showed the dominant cost: A3 q2 routed-expert CPU gather copied ~126GiB in ~84s on the coordinator for a short 128-token run. The per-request distributed telemetry was ~0.3–0.7s/token; this is not Thunderbolt or output-head overhead, it is the CPU-gather barrier per MoE layer.

**Patch.**
- `ds4.c`: changed the expert-offload span builder so routed expert spans are **not coalesced**. The q2 GGUF stores each layer's gate/down/up expert tensor as an individual contiguous tensor (~528/672/528MiB), but the previous split loader merged all routed experts into one huge 38GiB expert span per slice. Direct binding of that merged view made Metal command validation see ~40GiB and OOM. Keeping exact per-tensor expert spans is the next fine-grained route requested by the user: command buffers now bind per-layer gate/up/down expert tensor views rather than one 30+GiB span.
- `ds4_metal.m`: `DS4_METAL_EXPERT_OFFLOAD_DIRECT=1` remains the switch that bypasses A3 CPU gather for q2 routed tensors and lets existing Metal id-matvec kernels read selected rows directly from the non-resident mmap-backed expert tensor views.
- `ds4.c`: direct mode changes decode command splitting from the old prefix split to per-layer split/drain (`DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=1` default under direct). This keeps a direct-mode command buffer from accumulating expert views from many layers at once.
- `ds4.c`: layer-slice graph sessions allocate KV/compressor/indexer persistent state only for the active distributed layer range (plus MTP support layer state when enabled). Added `DS4_DIST_PREFILL_CAP` to clamp distributed graph batch scratch independently of `-c 200000`.
- `tools/mtp_pipe_q2_speed.sh`: defaults now set `DS4_METAL_EXPERT_OFFLOAD_DIRECT=1`, `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=1`, and `DS4_DIST_PREFILL_CAP=128`. A3 source-cache/profile defaults are kept as fallback/diagnostic knobs; they are inactive while direct mode bypasses A3.

**Expected impact.** This directly attacks the measured 0.66t/s blocker without returning to a command buffer that sees a 30+GiB expert span. If Metal accepts the per-tensor direct views under the 12/8GiB budgets, decode should move from CPU-copy-bound toward GPU/mmap-read-bound. If a device still OOMs, set `DS4_METAL_EXPERT_OFFLOAD_DIRECT=0` to return to A3.

**Validation performed here.** `make ds4` passes locally and `bash -n tools/mtp_pipe_q2_speed.sh` passes. No dual-host/model timing run performed here; user will manually validate with the script.

## 2026-06-10 — project.md 第一波落地: P0.1 IO 分解计时 + P1.1 pread 单拷贝直读 + P1.2 prefill 整层流式 + 槽位排序

**Context.** project.md 物理审计定位当前 0.81 t/s 的真瓶颈: A3 gather 有效带宽 ~1.5GB/s =
mmap 16KiB 缺页 (SSD→页缓存) + memcpy (页缓存→Shared scratch) 双拷贝, 串行夹在每层 command drain 之间;
SSD (3–7GB/s) 与雷电 (~4–5GB/s) 均未打满。本补丁按 project.md P0/P1 顺序落地第一波, 全部走 env 开关,
引擎默认行为与旧基线 bit 级一致 (不设新 env = 原代码路径)。

**Patch (`ds4_metal.m`, ~640 行新增/重排; `tools/mtp_pipe_q2_speed.sh` 新增开关).**
- **P1.1 单拷贝直读** `DS4_METAL_EXPERT_PREAD=1`: `ds4_gpu_set_model_fd` 现真正登记主模型 fd
  (mmap 自文件偏移 0 ⇒ 文件偏移==map 偏移); `ds4_gpu_expert_gather_copy_slot` 冷专家分支改为
  每张量一次 `pread()` 直读进 scratch MTLBuffer——零缺页、零二次拷贝, 字节与 mmap 路径同源同偏移,
  logits 构造性 bit-exact。pread 失败 (EIO/短读) 逐槽回退 mmap memcpy 并计数 `pread_fallbacks`。
  双模型 fd 冲突时自动禁用 (g_model_fd_conflict)。`DS4_METAL_EXPERT_PREAD_NOCACHE=1` 经 F_GETPATH
  开独立 F_NOCACHE fd 供冷读, 防止 1.7GiB/token 冷流冲掉热页缓存 (默认关, 单独 A/B)。
  env 解析全部在 gather 串行入口完成后传 ctx, 避免与 gather 线程竞态。
- **P1.2 prefill 整层顺序流式** `DS4_METAL_EXPERT_FULL_LAYER_STREAM=1`: batch 站点 (n_tokens>1)
  先只数活跃集 (collect, 不重写 selected 缓冲); 活跃专家 ≥`DS4_METAL_EXPERT_STREAM_THRESHOLD_PCT`%
  (默认 60) 时, 整层 gate/up/down 三段按 `DS4_METAL_EXPERT_STREAM_CHUNK_MB` (默认 16MiB) 分块、
  gather 线程池并行顺序读满 scratch, **专家 id 不重映射** (selected 缓冲原样, kernel 以
  n_total_expert 布局索引 scratch, 语义与 resident direct 路径一致)。流式失败自动回退 remap+gather。
  内存无新增峰值: 大 chunk prefill 下 n_active≈243/256, 旧 gather scratch 本就 ~满层尺寸。
  流式跳过 pool warm 与 source-cache note (decode 路径不受影响)。
- **快赢: 槽位排序** `DS4_METAL_EXPERT_SORT_IDS=1`: compact 拆为 collect+remap 两段,
  remap 前对活跃 id 升序排序 = 文件偏移升序 ⇒ 冷读顺序化。bit-exact: 仅 scratch 槽位编号变,
  每 pick 的计算与求和顺序不变。
- **P0.1 IO 分解计时** `DS4_METAL_EXPERT_IO_PROFILE=1`: 每个 routed-MoE 层调用打一行
  `ds4-io: site=decode|batch mode=gather|stream|pool layer= n_active= n_tokens= cold_mib= hit_mib=
  wall_ms= fault_ms= memcpy_ms= pread_ms= drain_ms= bw_gbps= pread_fallbacks=`;
  mmap 模式下先 16KiB/页 touch 单独计 fault 时间再 memcpy (仅 profile 开启时); drain_ms 单独计
  `ds4_gpu_end_commands` 等待 (屏障税专查, SwiftLM #84 教训)。fault/memcpy/pread 为线程求和 CPU ms,
  wall/drain 为墙钟 ms (启动时打印说明行)。
- **脚本**: 新增 EXPERT_PREAD(默认1)/EXPERT_PREAD_NOCACHE(0)/EXPERT_SORT_IDS(1)/EXPERT_STREAM(1)/
  EXPERT_STREAM_THRESHOLD_PCT(60)/EXPERT_STREAM_CHUNK_MB(16)/EXPERT_IO_PROFILE(1), 拼进 BASE_RUN_ENV;
  任意一项设 0 即回旧基线做 A/B。

**预期.** decode: pread 砍掉缺页+双拷贝 (M1 门: ≥1.6 t/s 方向); prefill: 整层顺序流式把 ~256 次
散乱 2–2.6MiB 读折成 3 段长顺序读 (M1 门: prefill ≥10 t/s 方向); ds4-io 行直接给出 P0.1 账表
(fault/memcpy/pread/drain 占比 + 等效带宽), 决定下一步主攻 (P1.3 重排 vs P2.1 预取)。

**Validation performed here.** `make` 全量通过 (新警告 0, 既有 4 条 unused 警告为本补丁前已存在的
死代码); `./ds4_test --metal-kernels` OK (含新 env 全开重跑 OK); `bash -n tools/mtp_pipe_q2_speed.sh`
通过。未在本机跑大模型 (纪律); 用户手动以 `tools/mtp_pipe_q2_speed.sh` 双机验证, 关注:
① 三行 ds4-io 启动说明是否出现; ② decode `mode=gather` 行 pread_ms vs 旧 fault+memcpy;
③ prefill `mode=stream` 行 bw_gbps; ④ `pread_fallbacks` 必须为 0; ⑤ 两机 RSS 红线 12/8GiB。
A/B 矩阵: EXPERT_PREAD=0/1 × EXPERT_STREAM=0/1 × EXPERT_SORT_IDS=0/1 (任一回退即逐项定位)。

## 2026-06-10 — 第二波: P2.1 跨层路由预测预取 + gather 张量级工作单元 (1.56 t/s 之后)

**Context.** 用户实测第一波 (pread+流式+排序) 后 generation 0.81 → **1.56 t/s** (prefill 2.95)。
P0.1 的 ds4-io 账表 (coordinator): decode 每层 gather wall ~10–20ms (bw 2–4GB/s, fault/memcpy=0,
pread_fallbacks=0 ⇒ pread 路径完全接管) + drain ~3.5–5ms; 20 层合计 ~365ms, 加 worker 23 层 ≈ 1.56 t/s
自洽。剩余结构: **IO wall ~70% + drain 税 ~25%**, 且 IO 与 GPU 计算完全串行。本波按 project.md P2.1
把 L+1 层专家 IO 藏进 L 层计算, 同时把 pread 并发度抬上去。

**Patch.**
- **gather 工作单元: 槽级 → 张量级** (`ds4_metal.m`): `copy_slot` 重构为 `copy_unit`
  (unit = slot*3 + {gate,up,down}); decode 6 专家从 6 个并发单元变 **18 个并发 pread**,
  线程池/串行回退/计时全部跟随。脚本 `GATHER_THREADS` 默认 4→8 (QD 实质翻倍, A/B 用 4 回退)。
- **P2.1 跨层路由预测 + 异步预取** (`DS4_METAL_EXPERT_PREFETCH_AHEAD=1`):
  - 新 API `ds4_gpu_register_layer_router` (ds4_gpu.h / ds4_metal.m / ds4_cuda.cu 桩):
    `ds4.c` 在两个 expert-offload 装载分支注册**仅本机层片**的 router 元数据
    (F16 gate_inp 偏移 + exp_probs_b 偏移 + gate/up/down 专家张量偏移; shrunken 模型跳过)。
  - decode 站点 drain 之后立即把本层 router 输入 x (16KB) 快照入队 (队列 4 深, 满则丢弃计数);
    后台线程复算 L+1 层 router: score = sqrt(softplus(gate_inp·x)) + bias (与
    layer_topk_selected_experts 同序), 取 top-N (`DS4_METAL_EXPERT_PREFETCH_TOP` 默认 8),
    对预测专家的 3 段字节发 **F_RDADVISE** (失败回退 pread 进丢弃缓冲) ⇒ 读提前进页缓存,
    L+1 层真 gather 的 pread 变页缓存命中/与在途 IO 合并。
  - 预测纯 advisory: 错了只多读 ~(N-6)/6 的字节, 不碰 selected/scratch/数值路径, 零正确性风险。
    读 ahead 永远走常规 fd (非 F_NOCACHE fd)。
  - **命中率计数**: 每层最新预测集 (gen 标记) 与真实活跃集比对, ds4-io 行尾新增 `pf=hits/total`
    (累计)。P2.1 立项门槛 top-6∈top-8 命中 ≥85% 直接从日志可读。
  - 覆盖率: 层片内 L+1 (coordinator 19/20, worker 22/23); 跨 token/跨机不预测。
- **脚本**: 新增 `EXPERT_PREFETCH=1` / `EXPERT_PREFETCH_TOP=8` 入 BASE_RUN_ENV。

**预期.** decode 每层从 串行(gather+GPU) 趋向 max(gather', GPU), 其中 gather' 因 QD↑ 与页缓存
命中再降; 若 pf 命中率 ~90%, decode 方向 2.2–3 t/s (W2 双机串行墙 ~2.9)。若 pf 命中率低 (<60%),
直接关 EXPERT_PREFETCH=0 不损失既有 1.56。

**Validation performed here.** `make` 全量通过 (仅既有 4 条死代码警告; ds4.o 零警告);
`./ds4_test --metal-kernels` OK (基线 + 新 env 全开各一次); `bash -n` 通过。未跑大模型。
用户脚本验证关注: ① 启动行 `registered N local routed layers for cross-layer expert prefetch`
(coordinator 应为 20, worker 23) 与 `cross-layer expert prefetch enabled`; ② decode ds4-io 行
`pf=hits/total` 百分比 (≥85% 为预测可用); ③ wall_ms 是否较上轮下降 (QD↑ + 页缓存命中);
④ `pread_fallbacks` 仍须为 0; ⑤ RSS 红线 12/8。
A/B: EXPERT_PREFETCH=0/1, EXPERT_PREFETCH_TOP=8/12, GATHER_THREADS=4/8。

## 2026-06-10 — 第三波: 预取改"礼让式" (1.49 t/s 回退修复)

**Context.** 第二波实测 1.56 → **1.49 t/s** (prefill 2.95→2.86)。日志诊断:
- `pf=3795/4896 ≈ 77.5%` —— 预测本身可用 (部分层 wall 2–3ms / bw 14–20GB/s = 预取完全命中);
- 但多数层 wall 14→17ms 均值、尖刺 23–38ms, pread_ms 线程和大涨 ⇒ **readahead 与前台 gather 抢同一块
  SSD**。decode 期间 SSD ~100% 忙, v1 预取的 54MiB/层 (top-8 + 23% 预测错) 是纯增量流量, 挤慢前台,
  代价超过重叠收益。结构性教训: **预取只能花 SSD 空闲窗 (drain/GPU 计算窗), 不能加总流量**。

**Patch (`ds4_metal.m` 预取 v2; `tools/mtp_pipe_q2_speed.sh` 加 EXPERT_PREFETCH_DELTA).**
- **礼让调度**: 新增 `g_gather_active` 标志 (前台 gather/stream 进入置 1, 全部出口清 0);
  预取改 1MiB 分块 advise, 每块前自旋等待前台让出 SSD (`usleep(200)` 轮询), 即 readahead 只流入
  drain/GPU 计算的空闲窗。
- **latest-wins + 过期中止**: 队列 4 深环 → 单槽 + seq; 新预测入队即覆盖旧任务, 在跑的任务每块
  检查 seq 被超即放弃 (旧 token/旧层的 readahead 不再发)。
- **mincore 跳过**: advise 前对 mmap 地址段查页驻留, ≥90% 已缓存即跳过 (重复专家不浪费空闲窗)。
- **分数序发radvisory**: 预测分数高的专家先发, 空闲窗装不下时自然砍掉低概率尾巴 (top-8 余量
  只在窗口富余时才花)。命中率统计在预测时全集打标 (pf= 仍量预测精度, 与读完成度解耦)。
- **Δ 可调**: `DS4_METAL_EXPERT_PREFETCH_DELTA` (默认 1, ≤4) 预测 L+Δ 层 —— Δ=2 给 readahead
  双倍空闲窗, 预测精度变化由 pf= 字段实测。
- 预计算/标记/中止全部不碰数值路径, advisory-only 性质不变。

**预期.** 预取从"抢带宽"变"捡空闲": 最坏情况退化为 ≈关闭预取 (1.56 基线), 好情况把空闲窗
(~4–6ms/层 ≈ 12–18MiB) 的字节转成页缓存命中。Δ=2 + 77% 精度的理论上限 ~30–50% 的 IO 隐藏。

**Validation performed here.** `make` 全量通过 (仅既有 4 条死代码警告); `./ds4_test --metal-kernels`
基线+新 env (含 DELTA=2) 双跑 OK; `bash -n` 通过。未跑大模型。
用户脚本 A/B 矩阵 (按优先级):
① 默认 (PREFETCH=1 DELTA=1) vs `EXPERT_PREFETCH=0` —— 确认礼让版不再回退且有正收益;
② `EXPERT_PREFETCH_DELTA=2` —— 看 pf= 精度降多少、wall_ms 降多少;
③ `GATHER_THREADS=4` vs 8 —— 隔离 QD 项对 decode/prefill 的独立影响 (上轮无法归因);
④ 关注 ds4-io 高 bw 层 (页缓存命中) 占比是否上升, pread_fallbacks 仍须 0, RSS 12/8 红线。

## 2026-06-10 — 第四波: A3 每层 drain 改 MTLSharedEvent 快路径 (1.70 t/s 之后)

**Context.** 第三波礼让式预取实测 **1.70 t/s** (prefill 3.07), 历史最高, M1 门 (decode ≥1.6) 过。
最新账表 (coordinator): gather wall 均值 13.7ms (预取生效, pf=79.6%), **drain 均值 3.8ms/层**。
drain × 43 层 ≈ 165ms/token (~28%) 是当前第二大项, 其中相当部分是逐 CB commit+waitUntilCompleted
的调度/状态开销 (SwiftLM #84 同类教训; 仓库 TP rendezvous 注释先例: SharedEvent 快路径 ~150ms→<50µs)。

**Patch (`ds4_metal.m`).**
- 新增 `ds4_gpu_end_commands_event(label)` + 包装 `ds4_gpu_expert_drain_commands`:
  `DS4_METAL_EXPERT_EVENT_DRAIN=1` 时, A3 两个调用点 (decode/batch) 的每层 drain 改为
  **encodeSignalEvent → commit (不等) → MTLSharedEvent waitUntilSignaledValue 快路径 host wait**。
  同队列 CB 按提交序完成 ⇒ 事件触发 = 之前全部 GPU 工作完成、Shared 写可见 —— 与
  waitUntilCompleted 同等正确性保证, 去掉慢等待。
- CB 状态/错误检查推迟一拍: drain 完把 CB 挂 `g_pending_cbs`, 下次 drain 在事件等待后清扫
  (此时 CB 已完成, 清扫零成本); 事件 60s 超时回退 waitUntilCompleted + 完整诊断。
- transient buffer 引用在事件等待后释放 (GPU 已用完, 语义同 classic 路径)。cleanup 释放事件。
- 默认关闭 (engine); 脚本默认 `EXPERT_EVENT_DRAIN=1`, A/B 回退设 0。

**预期.** 若 drain 3.8ms 中 1.5–2ms 是调度开销: 省 ~65–86ms/token ⇒ 1.70 → **~1.9–2.0 t/s** 方向。
ds4-io 行 drain_ms 直接给出答案 (期望降到 ~1.5–2.5ms, 即纯 GPU 残余工作)。

**Validation performed here.** `make` 全量 + `ds4_test --metal-kernels` (基线 + EVENT_DRAIN=1 全开
env) 通过; `bash -n` 通过。未跑大模型。
用户脚本验证: ① ds4-io drain_ms 新均值 (目标 <2.5ms); ② generation t/s; ③ 输出文本与上轮
temp=0 seed=1 应逐字一致 (drain 语义等价的硬验证); ④ EXPERT_EVENT_DRAIN=0 A/B 回退;
⑤ 顺手建议: 看 worker 侧账表 `ssh 192.168.1.2 "grep 'ds4-io: site=decode' /tmp/mtp_pipe_worker.log | tail"`
对比两机每层 wall+drain, 若 coordinator 明显更忙, 可 A/B `SPLIT_COORD=0:17 SPLIT_WORKER=18:output`
(零代码层重平衡)。

## 2026-06-10 — 第五波: P0.3 双机 SSD 底数 + 预取 DEPTH 范围预测 (1.71 t/s 之后)

**Context / 测量 (本轮先测后写).**
- 事件 drain 实测无效: drain_ms 3.8→3.74 (1.70→1.71) ⇒ **drain 是真实 GPU 计算, 屏障税假设证伪**。
- **P0.3 落地** (`tools/ssd_bench.c`, F_NOCACHE pread 基准, 两机对 81GiB gguf 实测):
  - **M4 mini (coordinator): 全模式 ~2.4–2.6GB/s** (seq 16MiB=2.50, rand 2MiB qd8=2.40,
    rand 6.75MiB qd8=2.61) ⇒ **P1.3 捆绑重排判死刑** (随机 vs 顺序仅差 8%, 34GiB sidecar 不值);
    coordinator gather 有效 ~2.96GB/s 已超裸盘 (页缓存贡献), **mini 侧已逼近盘顶**。
  - **M1 MacBook (worker): rand 2MiB qd4-8 ≈5.5-5.7, 6.75MiB/seq ≈6.5-6.7GB/s, 快 2.3-2.6×**。
- 两机 decode 账表: coord 每层 wall 13.7 + drain 3.7 ≈ **17.5ms**; worker wall 4.52 (bw 13.35
  = readahead 在快盘上真正完成, 页缓存大量命中, pf=83.7%) + drain 5.14 ≈ **9.7ms**。
  Token ≈ 20×17.5 + 23×9.7 ≈ 585ms ✓ 1.71 t/s。
  ⇒ 瓶颈 = coordinator 慢盘; 它的空闲窗 (~3.7ms ≈ 9MiB@2.5GB/s) 单层吃不满 40.5MiB 的预测读。

**Patch.**
- **预取 DEPTH 范围预测** (`ds4_metal.m`): job 从"预测单层"改"用同一 hidden 预测 L+1..L+D"
  (`DS4_METAL_EXPERT_PREFETCH_DEPTH`, 默认 1, DELTA 作 legacy 别名; 脚本默认 **2**)。
  最近 deadline 先发; mincore 让相邻 job 的重复预测近零成本; 超越/关停在层间与块间均可中止。
  D=2 给每层预测 ~2 个空闲窗 (~18MiB ≈ 45% 覆盖), 预测精度代价由 pf= 实测。
- `tools/ssd_bench.c` 入库 (含两机实测数注释), P0.3 可复现。
- 脚本: EXPERT_PREFETCH_DELTA → EXPERT_PREFETCH_DEPTH=2。

**决策记录.**
- P1.3 (GGUF 捆绑重排): **不立项** (P0.3 数据否决, mini 盘无顺序优势)。
- 层重平衡 (零代码, 数据支撑): 每挪 1 层 coord→worker 净省 ~7.8ms/token。
  建议 A/B `SPLIT_COORD=0:17 SPLIT_WORKER=18:output` 起步 (worker +2 层 ≈ +0.4GiB,
  注意 watchdog 8GiB 红线), 可再试 0:16/17:output。配合 coord 腾出的 wired 内存
  → 更大页缓存 → coord 残余层 wall 进一步降。

**Validation performed here.** `make` 0 error; `ds4_test --metal-kernels` 基线+新 env 双跑 OK;
`bash -n` 通过; ssd_bench 双机实测完成 (仅文件读, 未跑模型)。
用户脚本验证: ① 默认 (DEPTH=2): coord wall 期望 13.7→~10-11, 高 bw 层占比上升;
② pf= 精度变化 (D=2 预测含 1 层陈旧 hidden); ③ `EXPERT_PREFETCH_DEPTH=1` A/B;
④ 层重平衡 A/B (先小步 0:17/18:output, 看 worker RSS); ⑤ pread_fallbacks=0, RSS 红线。

## 2026-06-10 — 第六波: DEPTH=2 证伪回退 + P2.2 低成本变体「远程专家字节服务」(1.66 t/s 之后)

**Context.** DEPTH=2 实测 1.71→1.66: coord wall 13.96 (持平), worker wall 4.52→5.43 —— 远层预测的
matvec 算力与 advisories 挤占了本就饱和的空闲窗, worker 的 readahead 本来就完成了, D=2 纯开销。
**决策: 脚本 DEPTH 默认回 1** (引擎默认本来就是 1)。
真正的结构机会来自 P0.3 + 两机账表: worker 盘 5.5-6.7GB/s 且在 coordinator 干活的 ~350ms/token
里完全空闲; mini 盘 2.5GB/s 封顶且是瓶颈侧。两机加载字节相同的 gguf ⇒ **coordinator 可以同时从
两块盘取专家字节** (本地 2.5 + 雷电链路 ~2-4GB/s)。

**Patch (P2.2 低成本变体, project.md「空闲方 SSD 经雷电给忙方流专家字节」).**
- `ds4_distributed.c/h`: 新增**远程 pread 服务** (~230 行):
  - 协议: 每连接顺序 req{u64 off,u32 len(≤8MiB)} → resp{u32 status}+bytes; 握手校验 magic+文件大小。
  - 服务端 `ds4_dist_expert_fetch_maybe_serve(fd,size)`: env `DS4_DIST_EXPERT_FETCH_SERVE=1` 时
    监听 (`_PORT` 默认 5606), dup 模型 fd, accept 线程 + 每连接处理线程; ds4.c 在 engine open
    (set_model_fd 同点) 调用, 未设 env 即 no-op。
  - 客户端 `..._client_init(host,port,n,size)` + `ds4_dist_expert_fetch(slot,off,dst,len)`:
    每拉取线程独占一条 TCP (slot 绑定, 无锁); 大小不匹配拒连。
- `ds4_metal.m`:
  - gather 重构: unit 解析抽出 `unit_resolve` (本地/远程共用); ctx 加原子 `done` 计数;
    入口统一走共享游标 (串行回退也用 temp_worker 内联), 远程开启时发布 ctx → 远程线程与本地
    pread 线程**从同一游标抢单元**, 入口等 done==total 后收尾。
  - 远程工人: hard-copy 命中本地 memcpy; 远程失败**永久降级本地** (链路 down 标志 + 当前单元本地
    完成), 正确性永不依赖对端。
  - ds4-io 行新增 `rfetch_mib= rfetch_ms=` (远程字节/线程时间)。
- 脚本: `EXPERT_REMOTE_FETCH=1` (默认开, A/B 设 0), `EXPERT_REMOTE_FETCH_CONNS=3`,
  端口 PORT+7; coordinator 注 HOST/PORT/CONNS, worker 注 SERVE/PORT; DEPTH 默认回 1。

**算账.** coord 侧聚合供给 ≈ 本地 2.5 + TB 实效 ~1.5-3GB/s ⇒ coord wall 13.7 → ~8-10ms,
coord 半程 350 → ~240-280ms ⇒ token ~470-510ms ⇒ **方向 ~2.0-2.1 t/s**。worker 服务发生在它的
空闲半程, 与其自身 gather 天然错峰。

**Validation performed here.** `make` 零新告警; `ds4_test --metal-kernels` 多组 env 通过;
**协议环回自检通过** (同机 serve+2 连接, 对齐/非对齐/尾部/6.75MiB 取数逐字节比对 OK, 越界请求
正确拒绝); 连不上服务时优雅禁用路径检查; `bash -n` 通过。未跑大模型。
用户脚本验证: ① 启动行两侧: worker `expert-fetch server listening on :5606`,
coordinator `expert-fetch client: 3 connection(s)` + `remote fetch thread(s)`;
② coord ds4-io 行 `rfetch_mib` >0 且 wall_ms 下降 (期望 13.7→8-10); ③ worker 侧速度不应回退
(服务在其空闲半程); ④ 输出文本 temp=0 seed=1 逐字一致; ⑤ A/B EXPERT_REMOTE_FETCH=0;
⑥ CONNS=2/4 扫一下; ⑦ RSS 红线 12/8 (服务端仅 8MiB/连接缓冲)。
