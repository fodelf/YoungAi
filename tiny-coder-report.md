# tiny-coder 项目报告 —— 技术方案 · 质量/速度结论 · 规划 · 价值评估

> 2026-07-07 落稿。全部数字为当日实测（证据链在 `fable5.md` 对应段落）。
> 姊妹文档：`tiny-coder-plan.md`（路线权威版）、`p6-speed-plan.md`（速度分析，agent worktree）。

## 1. 当前技术方案

### 1.1 运行拓扑

```
Claude Code CLI ──ANTHROPIC_BASE_URL──▶ ds4-server (M4, /v1/messages Anthropic 兼容)
                                          │  coordinator, 层 0:19 + tokenizer/采样/编排
                                          │  KV: live 会话 + 磁盘断点 (跨轮增量 15-113 tok)
                                          ▼
                                        M1 worker (层 20:output, 常驻复用, 反连)
                                          │
                                          ▼
                                        SSD 专家流式 (A3, 59GB mono 双机各 ≤12G 红线)
```

- 模型：`ds4-mono-mixed.gguf`（59GB，浅层 GO1B 严格 1-bit / 深层 GO2B 2-bit 混合）
- 常驻服务：`tools/svc.sh`（up/probe/claude/down，看门狗 12G 两边同杀）

### 1.2 关键机制（今日新增★）

| 机制 | 作用 |
|---|---|
| ★guided tool-call generation（`--tool-primer`） | **结构由 server 注入（懂语法+schema），内容由模型生成**——弱指令跟随模型的工具调用构造性保证可解析。四级定位阶梯（复读→few-shot 无效→primer 语义解锁→guided 闭环）全实测 |
| ★数值安全模型自适应 | 引擎检出 GO1B/GO2B 张量自动 armed MATH_SAFE/KV_RAW_F32/ROPE/REPEAT_FREQ；thinning 对脆弱类型硬拒——错误配置无法打破正确性 |
| ★服务端护栏 | `--max-output-tokens`（cap 截断答 end_turn 防会话中止）、`--nothink`（渲染前拦截）、UTF-8 残尾裁剪、分布式下 continued 断点自动禁用 |
| copy-spec / knowledge-MTP | 引擎天然默认投机（惩罚重放 gate=lossless）；内置 Go 惯用语料 + `DS4_REF_CORPUS` 数据扩展 |
| KV 缓存体系 | live 前缀 + 磁盘断点 + DSML 精确重放——无状态 API 的 86% 重复前缀只付一次 |
| ★模块族 | `ds4_mtp/z/loss/posttrain/multimodal` 单文件零耦合 + 合成单测；接缝=tokenizer 回调 |

### 1.3 P2 后训练管线（六步已贯通）

`dsml_pipeline.sh`: corpus(✓已跑 200 条) → capture(脚本✓) → ref(脚本✓) → solve(✓zsolve 烟测过,
产物与 corr_load 字段级吻合) → mount → verify。产物=corr 侧车 GGUF，`corr_switch` 热插拔不动 base。

## 2. 质量与速度结论（全实测）

### 2.1 质量

| 层 | 判决 | 证据 |
|---|---|---|
| 协议/机制层 | **生产级** | Gate v2/v3 共 15 轮全部合法工具调用、CC 回路自转、内存零告警 |
| 模型内容层 | **长上下文检索缺失**（唯一瓶颈） | 三次独立复现：23.4k 上下文参数值=占位符/工具漂移；同版本 452 tok 探针真路径全对 |
| 数值敏感性 | batch/single kernel 差异可翻转贪心 | 批量注入实测把真路径翻成占位符→按质量>速度铁律回退 |
| 纯代码续写域 | 强 | twoSum/threeSum 逐字正确（proven 基线） |

### 2.2 速度

| 指标 | 双机 | 单机 |
|---|---|---|
| prefill | 70-83 t/s（峰 251） | 16.3 t/s |
| decode @24k | 1.5-1.8 t/s | 1.0 t/s |
| CC 首请求 23.4k | ~5 分钟（全量） | ~24 分钟 |
| **稳态工具轮** | **38-45s** = inject 26-29s(65-70%) + gen 4-6s + 增量 prefill ~8s | — |

- 小 span 增量 prefill 有 ~5.5s 固定截距 + 0.07s/tok 边际（agent 实测）
- 跨会话首请求复用受限：CC 上下文自漂（v2/v3 差 74 tok，时间/环境项）
- 使用形态：**挂机式 agent 工作流可用；交互式对话不可用**

## 3. 后续规划建议

### 近期（判决点）
1. **P2 侧车 = 质量唯一解**（本文档落稿后即开执行窗口）：capture(1-2h)→ref(HF 双机)→solve→mount→verify；
   判决=长上下文参数命中率 A/B + twoSum 回归门。
2. P4 收尾：image 编码器 C 接线（本次一并落地）+ server 图片 content block（后续）。
3. #5 回滚复用（prompt⊂live 方向）：需 dist 协议级 worker KV 截断，正确性向修复，非速度。

### 中期（冻结令解除后）
- P6 速度按 `p6-speed-plan.md` Top-5：copy-spec 接管自由 decode（最大块）、批量注入（条件：kernel 等价）、closer-skip、截距归因、前缀复用。
- P3 灵魂：数据+评测门定义行为（智能路由/诚实/专注），引擎只供机制。
- P1 体积（59→36→≤24GB）：消除双机依赖的战略项。

### 治理建议（工程债）
- 四大单体文件（11-23k 行）逐步模块化（今日模块族是模板）；CI/编译门；
- 多 lane 并发需最低限度协议（分支/worktree 或文件属主约定——今日 q2 模型失踪、脚本互踩为证）。

## 4. 客观项目价值评估

### 独特价值（公开生态几乎无等价物）
1. **16GB 消费级双机跑 59-81GB MoE 并接入真实 Claude Code 的完整闭环**——llama.cpp 系不具备层切分双机 + 压缩 KV 磁盘断点 + DSML 精确重放 + guided 生成的组合。
2. **guided generation 范式**（结构/内容分工）对一切弱指令模型通用——本项目最可迁移的单点创新。
3. **闭式后训练插件体系**（零训练、侧车热插拔、四损失/RRR 工具链）是可复用研究资产。
4. **工程护栏文化**（质量门先于默认、引擎级硬拒、预算门+看门狗）——今日 MOE_THIN 事故→引擎硬拒的闭环是范例。

### 硬局限（诚实清单）
1. 模型内容质量是**未闭环的赌注**：P2 未证之前，"极小体积还原大模型能力"的核心主张停在机制层。
2. 速度只够挂机 agent（38-45s/轮）；交互场景不可用。
3. 平台绑定 macOS/Metal（CUDA 路径在但非主战场）；单用户单流。
4. 无 CI、多 lane 治理缺失，工程债累积中。

### 价值判决与务实退路
- **作为"本地私有 Claude Code 后端"的 PoC：机制闭环已达成**（这是今日的里程碑）。
- 产品化价值取决于 P2：判决点明确（长上下文参数命中率）、管线就绪、成本可控（几小时级）。
- **务实退路（若 P2 失败）**：同一栈换装 7B-32B 开源 coder 模型——引擎/服务/guided/KV 体系全部复用，
  放弃"恢复 671B 能力"的上限赌注，保留全栈自研可控性。本项目的基建投资在两条路上都不作废。
