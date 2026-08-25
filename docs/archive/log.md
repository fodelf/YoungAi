# log.md — 双机 q2 全量模型逐波实测记录

> 从 `project.md` 迁出（2026-06-17）。本文件只放**波次实测进展**（M0.5 起的逐波 decode/prefill 实测与结论）。
> 前瞻计划（目标阶梯 M0–M4C、物理墙审计 §1/§1.5、P0–P4 方案、预算、测试协议）仍在 `project.md`。
> 更细的逐补丁/测量/决策流水账见 `notes/execution-log.md`。

## 逐波实测阶梯（decode / prefill t/s；起点 M0 0.81/0.78 见 project.md 目标阶梯表）

| 里程碑 | decode | prefill | 依赖 |
|---|---|---|---|
| M0.5 实测（2026-06-10 第一波后） | 1.56 | 2.95 | P1.1 pread + 排序（短 prompt 未触发 P1.2 流式） |
| M1 达成实测（第三波后） | 1.70 | 3.07 | + P2.1 礼让式预取（pf 79.6%）；M1 门 decode ≥1.6 ✅ |
| 第六波实测（P2.2 低成本变体） | 1.81 | 4.80 | + 远程专家字节服务（rfetch 8.4MiB/层，coord wall 13.7→11.5） |
| 第九波实测（预测驱动远程暂存） | 1.99 | 4.81 | staging hit 21.8MiB/层（54% 走 RAM），coord wall 8.91 |
| 第十一波实测（在途等待+TOP 预算） | 2.00 | 4.99 | coord/worker 半程 249/248ms 完美平衡 |
| **第十四波实测（哈希层精确暂存）** | **2.03** | 4.65 | 层 0–2 为 token 哈希路由（tid2eid 查表=100% 可知）；L2 完美形态 cold=0/wall=1.3ms |
| 第十六/十七波（负结果已回滚） | 1.79/1.80 | — | keepalive 与就绪度门控均 -12%（门控引发 racing 级联、饿死 lookahead）；代码已恢复 2.03 基线 |
| **单 token 地板判定（2026-06-10）** | ~2.1 | — | 剩余=两机 drain GPU 计算 ~175ms + worker 本地读 + L0 结构窗口；**M2 ≥3 须走 PC.1 多 token 路线（§3.5）** |
| **第二十波实测（PC.1 v2 上线）** | **2.05（smoke 档新高）** | 4.71 | 复制投机最长后缀锚+自适应抄长；smoke 档 fire 率 0%=纯不赔钱，收益待 code-edit 档（PC.5） |
| 第二十一—二十五波（code-edit 档攻坚） | **code-edit 1.95（二十五波新高）**；smoke 2.09（二十四波新高）/2.02 | — | mm_id GEMM 修 GPU 侧（批 drain 69→38ms）；NOCACHE 判决=prefill 冷读赢/verify 批输 → 收窄 ≥24 token；**backbone mlock 4.08GiB 根治 r1 驱逐爬升（548→2361ms 消失，稳 578-874）**；账面闭合：97tok≈Σr1(14.2s)+Σr2(35.7s)，r2=73% 是唯一大头 |
| **第二十六波实测** | **code-edit 2.24（新高）/ smoke 2.09** | 7.33 | DRAFT 16 即刻生效（4 轮 16/16 全中）；反向 efetch 因 TB 桥启动期 ARP 瞬态未起（首连失败被永久缓存） |
| 第二十七波（**未被真正测到**） | 报 2.23=wave26 复测 | — | 用户开跑时编辑只落了一半（脚本 DRAFT=32 未落 → K 被钳 16）；时间线复盘见 log 第二十八波 |
| **第二十八波实测** | **code-edit 2.30 / smoke 2.13（双新高）** | 7.60 | 但 K=32 仍未生效（漏改 eval 体内第二处 `K>16` 钳位，两波白测）；反向 efetch 31 连败——decode 期 staging 也把 TB 打满，ARP 全程饿死 |
| **第二十九波实测** | **code-edit 2.90（新高）/ smoke 2.13（持平）** | — | K=32 终于生效（sent=4/4/7/13/25/32 接受 2/4/7/13/25/29、draft_accept 94%、tok/call 5.71）；反向 efetch 静默窗拨号仍瞬败（进程内 worker→mini 出连必败之谜未解） |
| **第三十波实测** | code-edit 2.54（回落）/ smoke 2.08 | 8.39 | **accept 模式反向 efetch 终于连通**（worker 6 连、racing 生效、prefill gather 4.4→6.3-6.8GB/s）；回落=48 注赔钱（sent=49 只中 29，抄源分叉点=29，多付 3.4s）+新发现：流式层无 racing（mini L0-2 单盘 1.9GB/s×950ms/层） |
| **第三十一波实测** | **code-edit 2.90（追平最高）/ smoke 2.08** | 10.41 | 两修都生效（next_len 顶 32、4 份日志 0 条 mode=stream）；smoke 没回 2.13 的元凶定位：**worker decode gather 在 racing**（426/1104 带 rfetch，`\|\| !stage_enabled` 臂在 accept 模式下永真，本地 2-6ms 的活去跨 TB 拉 6ms+ 纯加尾）——时间线与 wave30 accept 连通完全吻合；另：worker decode **hit_mib=0.0 全程零命中**（source cache 仅 128MiB） |
| **第三十二波实测（设备限制更新：两台都 12G/总 24G）** | **code-edit 3.00（新高）/ smoke 2.14（新高）** | 9.84 | racing units 下限生效（worker decode rfetch 0/1104）；worker mlock wired 4.14GiB、verify drain 74→~50ms/层；r2(33)=6493 闭合=IO walls 60%+drain 38%。**两个真相**：① source cache async 模式从不 admit（两侧 summary hits=0 admit=0，2048MiB 配置是死字段，只做 madvise 加热）；② coordinator decode hit_mib 全来自 **staging**，worker hit=0 只因 stage_enabled() 门硬要 FETCH_HOST |
| 第三十三波实测（**负结果，已回滚**） | code-edit 2.83 / smoke 1.75（双降） | 10.00 | worker staging 净亏：completion 64%（coordinator 同机制 97%）、dropped 33%、decode 命中仅 26%，wall 反升 2-6→4-9.6ms/层，未暂存层 racing 复活 125/1104，r1 +20%。根因=盘速不对称：mini 盘慢（1.9GB/s）+兼任 efetch server，喂不饱 worker 暂存窗口。`WORKER_EXPERT_STAGE` 默认回 0 |
| **第三十四波实测** | **code-edit 3.32（新高）/ smoke 2.16（新高）** | 9.91 | PC.1 融合单轮全生效（6 轮全 r1_ms=0）；意外之喜：+1 对齐偏移让末轮 33/33 全中（旧形态 29 分叉）；EOS 自然结束于 97；账：fire 轮 84 tok/Σr2 21.3s + miss 13 tok/7.8s |
| **第三十五波实测** | **code-edit 3.39（新高）/ smoke 2.15** | 10.23 | NOCACHE 阈值修正近似打平（r2(25) -310 / r2(33) +190）：轮间复用被容量否决——每轮 union ~7GiB 装不进 ~3-4G 空闲页缓存；改动无害保留且是 36 波前置条件。排除项：MoE 分半 dispatch（路由依赖+位精确风险）、MTP-on-miss（小批 r2 成本曲线下只值 +5-10%）、批 token 分半流水（union 稀释吃掉重叠收益） |
| **第三十六波实测** | **code-edit 3.54（新高）/ smoke 2.17（新高）** | — | 批 union 预测预取生效（与 35 波 cached fd 协同）：六轮 r2 全降至 1354/1460/2286/3497/4821/5819，Σr2 19.2s；kc=33 新账目 walls 53%（5.2GB/s 贴双盘饱和）+ drain 45%（真 GEMM）；结构：fire 70% + miss 30%（13 tok×630ms 贴单轮地板） |
| **第三十七波实测** | **code-edit 3.77（新高）/ smoke 2.05（噪声带内回落）** | 10.19 | 注梯 ×4 生效（next_len 3→3→12→32→32，5 轮 acc 2/4/13/33/32=84 fire +13 miss=97 EOS）；smoke 2.05 = 运行噪声（wave-37 仅动融合 copy-spec 分支，smoke fire 率 0% 结构上不可能受影响；历史带 2.02–2.17）|
| **第三十八波 = 上限审计（无补丁）** | 维持 3.77 | — | §1.5：IO work-stealing 已带宽最优、walls 贴物理盘+TB 顶、drain 是 GEMM/M1-GPU 墙；token 流水稀释净亏、跨层重叠 worker 侧慢盘预测净亏 |
| **第三十九波 = 重叠可行性修正** | 维持 3.77 | — | **修正第三十八波误判**：层内专家分半重叠**是 bit-exact 的**（按 slot 区间分两组，每 (token,pick) down 值逐位不变、sum_experts 6-pick 顺序不变），不需放宽契约。这是唯一有效大杠杆(~15-17%)。下一步实现 `DS4_METAL_MOE_OVERLAP`（slot-range mm_id + 两趟 gather/GEMM 重叠），默认 OFF、--dump-logprobs 验逐位、metal-kernels 验 slot-range kernel |
| **第四十波 = P-OVL step 1（落地）** | 维持 3.77 | — | `kernel_mul_mm_id_map0` 加 `[slot_lo,slot_hi)` 区间门（moe.metal + host struct），loop-invariant `in_range` 不发散；全区间=逐字节 no-op。metal-kernels 绿。bit-exact 基石 |
| **第四十一波 = P-OVL step 2（落地，默认 OFF）** | 待 A/B（默认维持 3.77） | — | host 两趟重叠编排。**单 scratch 不相交 slot 区间**（pass0 填[0,half)/pass1 填[half,n)，写不相交⇒GPU读 pass0 时 CPU 填 pass1 无别名，内存中性，省第二套 scratch）。gather 抽 `ensure_moe_scratch`+`gather_experts_run`（dst 参数化，range-gather 不改 worker）。`flush_commands` 异步提交 pass0 让 GPU 与 pass1 gather 并行。自包含分支（提前 return，linear 字节不动）。`make`+metal-kernels+server 绿。开 `DS4_METAL_MOE_OVERLAP=1` A/B + `--dump-logprobs` 验逐位 |
| **第四十一波 = P-OVL step 2 双机实测 A/B（验证通过）** | code-edit gen **3.47→3.72（+7.2%）**、prefill 10.02→11.96（+19%） | — | 亲跑同会话 A/B。**bit-exact 实证**：两跑生成文本逐字节相同（md5 一致，temp 0 贪心）⇒ 无需 dump-logprobs。verify 批 r2_ms 普降 ~9-10%（结构性增益非噪声）。看门狗未触（≤6.6G/12G）。裁决：杠杆成立，暂默认 OFF（建议干净机复跑后再 default ON）|
| **第四十二波 = P-OVL N 趟泛化 + 定论（N=2 触顶）** | code-edit gen baseline 3.47 / **N=2 3.84（+10.7%）** / N=4 3.60（+3.7%） | — | `DS4_METAL_MOE_OVERLAP_PASSES` 默认 2。同会话 A/B：**N=2 > N=4**，撞 remote-racing 96 单元地板（N>2 子块 gather <96 单元→racing 关→丢 worker 快盘→gather 变慢）。三者 md5 全等 ⇒ 任意 N **bit-exact**。**重叠杠杆 N=2 终点**。再快需降 gather 字节（pool/staging/source-cache，注意 source-cache hit=0%/admit=0 待查）|
| **第四十三波 = source-cache 0% hit 查清** | 维持 3.79-3.87 | — | 非 bug：async 模式下 source cache 只是 madvise 页预读提示（不持副本/不追踪 ⇒ hit/admit/used 恒 0）。A/B OFF≈ON ⇒ 对 decode 零增益 |
| **第四十四波 = 动态内存基建 + RAM 专家缓存 A/B（定论负面）** | hard_copy 超时 / mlock **1.84（半速）** vs baseline 3.84 | — | 规则更新：内存应动态用满 12G。建 dynamic 定容基建（默认 OFF）。**关键纠正：~5.4G "headroom" 不闲置=OS page cache**；显式 RAM 专家缓存(hard_copy/mlock)偷 page cache RAM→多数 miss 变冷盘→净大幅变慢(3.84→1.84)+admit 20-56s。**page cache 已最优用满内存，无可榨取闲置；decode 确属 gather 物理地板**。这条线关闭，再快只剩 X9/路由级降激活 |
| **第四十五波 = 当前配置复验 + gather 线程审计** | code-edit **gen 4.13（本会话新高）**, bit-exact, peak 8.93G | 11.75 | 当前生产配置(overlap N=2)复验稳定正确。固定限制 `GATHER_THREADS` A/B: 16→3.80 < 8→4.13 ⇒ 8 已最优(慢盘 QD 甜点)，无可榨取 |
| **第四十六波 = expert pool(X9 机制)探针 → 定论: 热集装不下** | pool-alone **3.42 ≈ baseline（无提升）** | 10.76 | pool 默认仅覆盖 7/64 层、hit 53% 但只省 ~6% 总 gather。全层 top-16 需 ~6.75GiB GPU 驻留→撞天花板+饿死 page cache。**热集本质>可用RAM，所有驻留缓存打不过物理地板**。decode in-budget 优化空间**完全穷尽**，overlap N=2(+10~19%)是最终增益；再快只能降工作集(量化/路由)或换硬件，或回 P-Code 域提"有效 t/s" |
| **第五十八波 = 本机 MTP（coordinator drafter）落地+实测** | smoke gen **0.65（净亏）** / prefill 2.58；同档基线 NO_MTP=1 **gen 2.12** / prefill 3.45 | — | **功能全绿**：`--mtp-role coordinator` 落地，drafter 搬到本机 M4（output head + token_embd + 2.14G 草稿都在 coord，worker 只持 backbone `20:42` 返 hidden、coord 本地 output head + MTP draft + 跨机 verify）。dist-mtp tok/call=2.18 first_hit=68% accepted=48 disabled=0，生成文本正确。**内存解耦达成**：M1 worker resident **3.60G（从 ~6.3G 卸掉草稿+output）**、M4 3.99G peak、两边≤12G。**但 smoke 速度净亏 3.3×**（0.65 vs 2.12）：verify batch K=2 的 2× backbone expert 冷读 + 草稿 experts non-resident 冷读，SSD-bound 下额外 I/O ≫ 投机 2.18 tok/call 产出。与历史"MTP 本硬件净亏"一致 |

### 第五十八波结论：本机 MTP = 功能/内存成功，smoke 速度净亏（推进方向）

**做成了什么**：把 MTP drafter 从 worker 搬到 coordinator（`--mtp-role coordinator`，源码里写的
“Phase 2 / Scheme B”本机 drafter，此前未实现）。worker 只持 backbone `20:42` 返 hidden，coordinator
持 output head + token_embd + 草稿，本地 draft + 跨机 verify（复用 PC.1 copy-spec 的本地-draft/远程-verify
骨架 + `ds4_session_mtp_draft` 与 output head 共享 `g->cur_hc`）。**端到端实测功能全绿**（tok/call=2.18、
first_hit=68%、accepted=48、disabled=0、生成文本正确），**内存解耦达成**（M1 worker 3.60G，从 ~6.3G）。

**为什么 smoke 净亏（gen 0.65 vs 同档 NO_MTP=1 基线 2.12）**：SSD-bound 下投机的代价是放大 expert I/O——
每个投机 call = Round1(1× backbone gather) + MTP draft(草稿 experts non-resident 冷读) + verify batch
(K=2 → 2× backbone gather)，约 3–4× 单 token 冷 gather，只换 2.18 tok/call。decode 已贴 gather 物理地板
（第四十四/四十六波定论：page cache 已最优、热集 > 可用 RAM），多读必然净亏。**换 drafter 落点不改变 SSD I/O
物理墙**，与 worker-draft 时代“MTP 本硬件净亏”结论一致。

**推进方向（优先级从高到低）**：
1. **smoke 默认保持 NO_MTP=1**：本机 MTP 作为可选能力保留，不进默认路径。投机在 in-budget decode 已穷尽的
   硬件上不是速度杠杆。
2. **把“内存解耦”变现（最有价值的副产品）**：本机 MTP 让 M1 卸出 ~2.7G。这 ~2.7G 应投到能降 gather 的
   地方——重切 SPLIT 让 coord 卸层、worker 用解耦预算多驻留 1–2 层 backbone / 扩 source-cache 命中区 /
   留给更长 KV。需先把 **coordinator output-head 驻留从 `mtp_for_coord_draft` 解耦**（当前关 MTP 就不驻留
   output head，新拓扑 NO_MTP=1 预计起不来），让“worker 不持 output”成为独立可用的拓扑，再 A/B。
3. **只在 code-edit 档验证 MTP 是否净正**：copy-spec 已是本地 draft 且 code-edit 有收益（§3.5）；MTP head
   draft 在回显型负载能否压过 verify I/O 成本，需 `PROMPT_PROFILE=code-edit NO_MTP=0` 单独测一轮再裁决。

**未做/限制**：单机 logprob parity（避免擅自单载 81GiB）；新拓扑 NO_MTP=1 未测（基线用旧拓扑 worker `20:output`）。

