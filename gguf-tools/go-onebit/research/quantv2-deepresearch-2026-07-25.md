# 等体积还原率提升 · 深度研究报告（2026-07-25 收割）

**方法论**: deep-research 工作流（5 路并行检索 → 30 源 → 110 条论断全部带原文引文 → 每条 3 票对抗核实）。用户于 verify 后期指令收割：**19 条论断走完对抗核实（4 击毙 / 15 幸存），91 条带原文引文但未走完核实（下文标 ⚠未核实）**。原始论断+票据全文见同目录 `quantv2-deepresearch-appendix-raw.md`。

研究问题：严格等体积（~1.29 bpw 专家均位，任何边信息必须从别处扣位）、零训练（无 QAT/backprop）、自研 kernel 可改，在 1–2.5 bpw 区间还有哪些实测杠杆能把还原率从 Σmin 0.696 / top1 72.4% 再往上推。

---

## 1. 击毙表（对抗核实 ≥2/3 票否决——防止再踩）

| 被毙论断 | 击毙理由（核实者读原文全文） |
|---|---|
| PTQ1.61 "1.61 bpw 全含成本 SOTA, 胜 BiLLM/2-bit OmniQuant" | 头条数字**含 20K 步 LoRA restorative 预处理（backprop 训练全精权重）**；量化阶段本身也是 AdamW 20 epoch 梯度优化 scale（OmniQuant 类），非闭式。零训练判据下头条数字不可引用 |
| BTC-LLM "~1.11 bit 近乎追平 FP16, 数量级碾压 ARB/BiLLM" | 6.06 vs 5.47 = +10.8% 不是"近平"；vs 最强基线 ARB-RC/STBLLM 只有 2×非 10×；且头条依赖 **Adam 学的可逆变换（STE sign flips）** = backprop，零训练判据排除 |
| HBLLM@1.06b 胜 FrameQuant@2.2b ⇒ "结构胜位宽"泛化 | 原始数字对，但 11.36 GB 是**量化过程显存/未打包 checkpoint**，不是部署体积；FrameQuant 真实部署 ~2.2 bpw 打包；"全局变换边信息抹掉存储优势"论断原文无出处 |
| BBT-spectral "零训练旋转+能量缩放 −15~58% ppl" | 基线与本法都跑在 **Intel auto-round（SignSGD 梯度优化 rounding, 200 步）之后**——增量不是零 backprop 测得；零训练框架不成立 |

**教训**：sub-2bit 文献头条常把"训练组件"藏在 pipeline 里（LoRA 预处理 / 学习变换 / auto-round），引用前必须查训练成分；体积对比必须用部署字节不是显存。

## 2. 幸存核心结论（0/3 或 0/2 反驳票，可直接采信）

### 2a. 旋转该用"块对角 + sequency 排序"，不是全局 Hadamard ★最大单杠杆
GSR（ACL 2025 SRW, arXiv 2505.03810; LLaMA-2-7B W2A16 GPTQ g128, 三票全存活）：
- **局部（块对角, 块=量化组 128）Hadamard 单独就把全局 Hadamard 的 ppl 20.29 → 12.11**（离群影响被限制在组内）
- 仅把 Hadamard 行重排成 sequency（Walsh）序（零成本置换）：20.29 → 15.38
- 两者合并（GSR）：**20.29 → 11.59（−43%），零训练、零存储字节**（旋转折进权重）；W2A4 下同样成立（31.33 → 15.23）
- 免训练 GSR ≈ 训练法 OSTQuant（10.97），且可作训练法初始化再提升
- 边界：证据仅 dense LLaMA-2-7B、2-bit 组量化区间；作者自述 4-bit 以上无效果

**对我们**：v2 设计里"256-块对角 Hadamard 是妥协 MVP"的假设被推翻——**块对角本就是更优解，且 sequency 排序再免费加一档**。kernel 代价 O(d·G) 远小于全维 O(d log d) 全局变换。

### 2b. 冷专家 1-bit 的正确形态 = Haar 小波 + 结构化分组（HBLLM, NeurIPS 2025 spotlight, arXiv 2512.00862）
- **全零训练**（BiLLM 式 GPTQ 校准, 无 backprop）：LLaMA2-13B **1.08 bpw ppl 6.71**（FP 4.88, ×1.38）；匹配位宽下比 ARB-RC/BiLLM 砍 ppl 33–66%（L1-7B 8.82 vs 13.45/44.85 @1.09b）
- 机制：两个固定 size-2 卷积核 [½,½]/[½,−½]（O(d) 滑窗，非 O(d²) 全局变换），低/高频分带 + 行内自适应分组 → **每行可辨识反量化值 CIQ 从 8–10 提到 1024**；解码 GEMV 实测 ~31.8% FP16 延迟
- ⚠未核实补充（引文在案）：行自适应分组是最大单成分（16.32→11.08）；**同频带 2 组共享均值省 0.25 bpw 且 ppl 反而略降**（省出的字节可再分配）
- 边界：dense-only，作者明说不支持 MoE——需我们自移植到 per-expert 256-block 布局

**对我们**：我们的 signref 冷专家 = 行 scale × sign，**CIQ=2**——文献同位宽最强做法的可辨识值密度是我们的 ~500 倍。这是冷 tier 最大的表示能力缺口。

### 2c. 显著通道用"结构化 1-D mask"，成本 0.0002 bpw（PTQ1.61 存活部分）
- BiLLM/PB-LLM 的逐元素 unstructured mask 诚实核算要 ~1 bpw（BiLLM 真实 ~2.1 bpw）；**按输入激活选显著通道的 1-D 结构化 mask 只要 0.0002 bpw**，通道走 4-bit
- PTQ1.61 的头条被毙（含训练），但这个 mask 结构本身是零训练可移植的
**对我们**：冷专家离群列钉扎的正确实现——列粒度索引，不是元素 mask；体积几乎免费。

### 2d. 顺序补偿的文献定式 = GPTAQ "asymmetric calibration"（ICML 2025, arXiv 2504.02692, 0/2 存活）
- 闭式、比 GPTQ 多 ~20 行、零新增字节零解码成本：在**漂移输入 X̂ 上回归 FP 模型输出目标 W·X̃**，多出残差项 r·Xᵀ·H⁻¹
- ⚠未核实补充：W2A4+QuaRot 下 vs QuaRot+GPTQ 砍 ppl 20–90%（L3-8B 102→17.9）；越低位收益越大；可叠 SpinQuant 之上
**对我们**：**这正是我们合并态/顺序补偿纪律的同构文献版**——方向被独立证实。但要审计我们 quant_apply 内环的回归目标是否严格 = "漂移输入→FP 参考输出"（GPTAQ 形式），而不是"漂移输入→漂移目标"。1-bit 侧另有存活谱系（§3f）警告：拟合目标错了会比不拟合更糟。

### 2e. 自适应变换的边际收益会被 GPTQ 吃掉大半（WUSH, ICML 2026, IST-DASLab, 三票存活）
- 数据感知非正交变换胜最强 Hadamard 基线 +2.8 分（RTN）——但 **加上 GPTQ 误差反馈后只剩 +0.7 分**（4×缩水）；证据在 W4A4
**对我们**：降温剂——我们有 GPTQ 级误差反馈，别指望数据感知变换在我们栈上复现头条增量；先吃 GSR（其证据正是 GPTQ 栈上测的），WUSH 类后置。

### 2f. 其余存活
- PTQTP 双 trit-plane 质量数字真（vs ARB/BiLLM 大幅领先）**但真实体积 ~3.17 bpw**（"1.58"是对标 BitNet 品牌不是存储）——超预算，不适用；其闭式 ridge scale + 第二平面当残差补偿器的结构思想可借鉴
- BTC-LLM 二值码本组件（Hamming K-means, XNOR+POPCNT, 0.8b 码本开销 3.4%）零训练可移植——但近-FP16 头条依赖训练变换，预期要按 ablation 打折
- BBT 存活部分：旋转 −10.3% 与能量缩放 −11.5% 的分解比例（在 auto-round 栈上测得，仅作机制参考）

## 3. 未走完对抗核实的高价值结论（⚠引文在案、按对我们的可操作性排序）

a) **MoE 逐专家位分配的量级**（PMQ/MC#, arXiv 2510.10962）：Mixtral 2.05 bpw 下 ILP {1,2,3}-bit 按重要度分配 = **63.25% vs uniform 2-bit 42.67%（+20.6 分）**；1.57 bpw 仍 54.49% 非悬崖。重要度信号：激活频率 > 路由权重。
b) **分配粒度**：linear-block（gate/up/down 分开）> 整专家粒度（MxMoE 6.11 vs 6.32; AlphaQ Table 5 同结论）；**专家数越多增益越大**（60+ 专家模型 0.4–2.4 ppl vs Mixtral-8 专家 ~0）→ 我们 256 专家应更受益。
c) **校准域是命门**（EAC-MoE, ACL 2025）：跨域校准让 Mixtral Code 任务 **73.86% → 3.80% 崩塌**——我们的 code-domain 校准语料是 load-bearing 的独立文献印证；同文：量化致路由翻转单独贡献 ~0.44 ppl（我们 router 全精 + FP 路由复用，此风险天然规避）。
d) **重要度信号配比**（QuantMoE-Bench, arXiv 2406.08155）：权重离群分 80% + 激活频率 20% 是实测最优混合；**首层组高位 >> 尾层组高位**（+10.8 分 @Mixtral）；attention/共享专家升位性价比远高于 routed 专家；**热 tier 加宽 > 热 tier 加深**（top-10@8bit ≈ top-10@4bit, 省下的位给 top-25@4bit 更好）。
e) **QEP 阻尼警告**（NeurIPS 2025, arXiv 2504.09629）：跨层误差补偿要带 per-layer 强度 α∈[0,1]（∝ proximal 正则）——**全强度补偿在 2-bit 有翻车实例**（L2-13B GPTQ 1301→2782 变差）；QEP×QuIP 叠乘（65.6→11.97）证明补偿与旋转正交。我们的反修目前无阻尼旋钮。
f) **1-bit 拟合目标警告**（arXiv 2512.21651）：1-bit 下朝"量化流自身漂移目标"拟合比纯权重驱动**更糟**（ARB-X 21.61 vs ARB-RC 16.36）；必须朝 **FP 参考输出**拟合 + Gram/注意力几何保持项；等位零字节 ~6% ppl 增益。
g) **低秩边因子该不该买**（CALDERA/FLRQ/ODLRI）：固定字节下 4-bit 因子×4×rank > fp16 因子；**但 2-bit 纯 PTQ 下旋转+格码本(QuIP#)仍胜 标量+低秩**（L3-8B 12.74 vs 14.12）→ 低秩排旋转/码本之后；若买，初始化要对准激活离群通道（ODLRI top-k=16, 角色被初始化永久锁定），且 rank 收益前 20 秩占 84%（小 rank 就够）。与用户"加体积=偷懒"铁律一致：默认不买。
h) **2-bit 表示排序**（Q-Palette NeurIPS 2025 / GLVQ NeurIPS 2025）：高斯化权重上 TCQ 失真 0.071 vs 标量 0.117（−40%）；GLVQ 每组学格基 3.36 < QTIP 3.78 < QuIP# 3.91（L2-70B W2）——热 tier 标量→trellis/格码本的升档证据链完整；Q-Palette：只沿输入维旋转+共享旋转把在线 Hadamard 从 14 次/block 砍到 4 次。
i) **分数位分配闭式**（Q-Palette）：固定预算下最优 per-layer 位宽 ∝ log(灵敏度)，连续分数位 > 少数整数档；knapsack ILP 秒解。

## 4. 对 v1 栈的三重印证（我们已做对的）

1. 合并态/顺序补偿 ≈ GPTAQ asymmetric calibration（ICML 2025 才发表，我们已在跑）——方向正确且是低位区最大补偿杠杆
2. code-domain 校准语料 = EAC-MoE 实证的命门（跨域校准 Code 崩到 3.80%）
3. 热/冷频率分层方向对，但文献说粒度（linear-block）、信号（离群分主导）、深度感知（首层优先）都还有免费增量

## 5. v2.1 架构修订（等体积，全零训练）

```
┌ 预处理(热+冷): 块对角 sequency-Walsh 旋转, 块=量化组(128/256)
│   折进权重, 激活侧 per-group O(d·G) 前变换; 证据 2a(全局H的ppl一半)
├ 热64:  2-bit + GSR 旋转 + GPTAQ 式内环(漂移输入→FP参考目标+残差项)
│   二阶段: TCQ/格码本升档(证据 3h, 失真再−40%)
├ 冷192: HBLLM 式 Haar 小波分带 + 行自适应分组 sign 量化(CIQ 2→~10³)
│   + 结构化显著列 4-bit 钉扎(0.0002 bpw 索引) + 同带共享均值(−0.25 bpw 回血)
├ 分配:  ILP {1,2,3}-bit, linear-block 粒度(gate/up/down 分开),
│   重要度=离群分0.8+频率0.2, 深度感知(首层优先), 等体积约束
└ 链条:  合并态顺序补偿(保留) + per-layer 阻尼 α(3e 翻车警告)
         + 拟合目标审计: 必须=FP参考输出(3f)
```

**预期（文献锚定）**：2a 单项在 2-bit GPTQ 栈实测 −43% ppl；2b 在 1-bit 匹配位宽实测 −33~66% ppl；3a/3b 在 MoE 等预算实测 +5~20 分。叠加后 Σmin 0.696 → 0.78+ 带是有依据的目标；但全部证据是 dense 或 ≤64 专家 MoE，**我们 256 专家 1.29 bpw 是文献外推区，一切以 G1 实测为准**。

## 6. 验证门序（10 分钟总纲 + 可用性铁律不变）

- **G1a**（最先、量化器侧、不动引擎）：单层输出 cos A/B——块对角 sequency-Walsh + 现 go2b vs 裸 go2b；同层 HBLLM 式冷 1-bit vs signref。分钟级判决。
- **G1b**：GPTAQ 内环目标形式审计 + 阻尼 α 扫 {1.0, 0.7, 0.5}
- **G2**：赢家配置 3 层小样 → rr_code smin vs 0.6959
- **G3**：只给 G2 赢家写 Metal/CPU-gather kernel（块对角变换 + 小波 dequant）
- **G4**：gen_coding_probe 三真实任务 + LRU 翻转点归因复测
- 全程盘/内存账前置、fable5 记录、原始输出必贴。

## 7. 来源索引

存活证据主源：GSR arXiv:2505.03810（ACL25 SRW）· HBLLM arXiv:2512.00862（NeurIPS25 spotlight）· GPTAQ arXiv:2504.02692（ICML25）· WUSH arXiv:2512.00956（ICML26）· PTQTP arXiv:2509.16989 · BTC-LLM arXiv:2506.12040 · PTQ1.61 arXiv:2502.13179（ACL25）
⚠未核实层主源：MC#/PMQ arXiv:2510.10962 · MxMoE arXiv:2505.05799（ICML25）· AlphaQ arXiv:2606.04980 · EAC-MoE arXiv:2508.01625（ACL25）· QuantMoE-Bench arXiv:2406.08155 · QEP arXiv:2504.09629（NeurIPS25）· 输出对齐 arXiv:2512.21651 · CALDERA arXiv:2405.18886 · FLRQ arXiv:2601.05684 · ODLRI arXiv:2506.02077 · Q-Palette arXiv:2509.20214（NeurIPS25）· GLVQ arXiv:2510.20984（NeurIPS25）· FrameQuant arXiv:2403.06082 · D²-MoE arXiv:2502.17298 · 2605.25203（BBT）
