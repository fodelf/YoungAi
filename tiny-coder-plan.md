# 北极星计划：极小体积 · 极好质量 · 工程化灵魂 · 接入 Claude Code

> 2026-07-07 立项（用户口述目标的落地计划）。铁律：最小端到端前置、质量门先于速度门、
> 后训练产物=插件（侧车热插拔，永不改 base）、q2/hf 永不删。
> **冻结令（用户 2026-07-07）：体积优化（P1）与速度优化（P6）冻结——模型已可用，
> 先走通功能链路（P0 缺口→P2 域侧车→P3 灵魂→P4 多模态→P5 集成），速度排最后一项。**
> 主记录 fable5.md；本文件是路线的唯一权威版本，进度打勾就地更新。

## 0. 一句话目标

把 DeepSeek V4 Flash 量化到**极小体积**并用闭式后训练**还原到极好质量**（Go/计算机经典领域），
训练出**工程化灵魂**（诚实、长期专注、智能路由、容错、高效输出），多模态与后训练全部**插件化**，
最终以 **ds4-server 接入 Claude Code** 完成日常开发。

## 1. 现有资产盘点（全部已验证/已编译）

| 支柱 | 已有资产 | 状态 |
|---|---|---|
| 极小体积 | mono-mixed 59GB（浅层 GO1B/深层 GO2B 混合，81GB q2 → 59GB） | 双机逐字正确 twoSum，gen 1.6-1.7 t/s ✅ |
| 质量还原 | `ds4_z.c`（闭式 ridge+秩-k 隐变量，k_L 可调=产物③）、`ds4_posttrain.c`（Go-aware scale，实测 cos 0.55→0.70）、`ds4_loss.c`（四损失） | 合成单测全绿；未接真激活管线 |
| 数值安全 | 引擎按模型自适应（GO 类型→MATH_SAFE/KV_RAW_F32/ROPE/REPEAT_FREQ 自动 armed；thinning 对 GO 硬拒） | 2026-07-07 落地+验证 ✅ |
| 后训练=插件 | z-corr 侧车 + `ds4_engine_corr_switch`（多域热插拔，秒级，不动 base） | 运行时机制已在 |
| 领域知识 | `ds4_mtp.c` knowledge-MTP（内置 Go 惯用语料 + `DS4_REF_CORPUS` 文本扩展 + gotrie 统计扩展） | 已落地，命中即投机、miss 零成本 |
| 多模态=插件 | `ds4_multimodal.c`（模态→token 注册表，text 内置，未注册干净拒绝） | 接缝已在，编码器待接 |
| Claude Code 接口 | ds4-server `/v1/messages`（Anthropic 兼容）+ DSML 精确重放映射 + 流式 | 已在，端到端未对 Claude Code 冒烟 |
| 校准基建 | go-onebit calib（hf_read/layer_probe/npy/pyfwd 双机分 shard 校准） | 已在 |

## 2. 阶段与数字门

### P0（先行，最小端到端铁律）：Claude Code 冒烟闭环 ✅ 2026-07-07 协议层打通（fable5 P0 判决: 多轮循环+34.4k 双机 prefill 70 t/s+跨轮前缀复用 44tok；余 3 缺口: stop_reason 语义/MAX_THINKING_TOKENS/continued-checkpoint bug）
愿景的终点先打通，之后每个阶段的产物都能在真实工作流里判优劣。
- 起 ds4-server（mono 双机后端）→ `ANTHROPIC_BASE_URL=http://localhost:PORT` 指给 Claude Code。
- **门**：Claude Code 里完成一次真实小任务（读文件→改一个 Go 函数→跑测试）不需人工修补。
- 已知缺口：mono 是 BASE 裸续写模型，chat/DSML 模板路径需验证（tool-call 质量门 `ds4_test --tool-call-quality`）。
- ✅ 2026-07-12 **真实 MCP 调用场景支持**：①`tool_result.content` 块数组完整解析——text 块
  与旧字符串形字节等价（disk-KV 前缀兼容），**image 块（browser/screenshot 类 MCP 工具）
  走多模态注册表**产 `<image>` 草图（含 spatial/css 增强节），像素送不到=fail-closed 400
  （原实现静默吞图）；②`tool_result.is_error` → 信封内显式 `[tool_error] ` 标记（原被丢弃，
  空/短错误载荷时 flag 本身就是信号）；③`POST /v1/messages/count_tokens`——同模板同重放
  attach 真渲染+真 tokenizer 计数 `{"input_tokens":N}`，不推理（Claude Code 上下文预算用）；
  ④`mcp__server__tool` 名字经 schema 行/guided-primer/DSML invoke 全程原样往返（单测钉死，
  含 $schema/additionalProperties/cache_control 噪声）。`ds4_test --server` 全绿。

### P1：体积→质量还原主循环（59GB → 36GB → ≤24GB）
每一刀都走同一闭环：改类型图 → 闭式后训练还原 → 质量门 → 记录。
- 手段（全部零训练闭式）：
  1. 深半层 GO2B→GO1B+z 侧车（mapnet 判决：深半可映射近无损；单层 A/B 先行）；
  2. Go-aware per-row scale 重标定（`ds4_posttrain.c`，已验证 +0.10 cos）；
  3. z 隐变量按层配秩（`ds4_z_solve` + `ds4_z_set_rank`，四损失选 k_L——体积/质量的连续旋钮）。
- **质量门（逐级，先小后大）**：twoSum/threeSum 逐字判据（分钟级）→ go-bench 5 题 → `ds4-eval q1..q4 --temp 0 --seed 1`；层级 held-out cos ≥0.85（ALGORITHM.md 判据）。
- **体积门**：≤24GB = 16G 单机可跑（消除双机依赖，速度和部署双赢）。

### P2：后训练域扩展（插件工厂）
- 语料域：①Go 真项目（stdlib/gin/etcd 级别）②算法经典 ③重构 ④计算机底层。
- 每域产物 = 一个 z-corr 侧车 gguf（`X 激活捕获 → ds4_z_solve → 侧车`），运行时 `corr_switch` 热插拔。
- knowledge-MTP 同步扩域：每域一个 `DS4_REF_CORPUS` 文本片段库（gin 样板等，加数据不加码）。
- **门**：每域一套 12-16 token 快判集（快速质量验证铁律）+ 域内基准（如算法题 5 题）。

### P3：工程化灵魂（行为层，杠杆各就各位）
| 灵魂 | 杠杆 | 落点 |
|---|---|---|
| 智能路由（一次性 vs 深度） | think 模式二级路由 + 置信门 | 引擎已有 think/think-max；加"简单问题不进深思"路由门 |
| 容错（大小写/错别字） | 校准数据字符级增广（`ds4_loss_dither` 思路扩到文本层）——2026-07-07 裁决：matcher 层大小写折叠证据弱（token 级折叠≠改比较符，收益边际），不做；容错归 P2 校准数据路径 | 校准数据（P2 耦合） |
| 诚实 | 校准语料只收"承认不知道/引用出处"样本 + eval 门加幻觉判题 | 数据+评测门 |
| 长期专注 | 磁盘 KV 会话（ds4_agent 已有）+ 长上下文事实召回门（`--long-context` 已有） | 已有基建，加门 |
| 高效输出减 debug | agent 生成后自验回路（编译/测试作为 verify 批的行为级对应物） | ds4_agent 工具位 |
- 原则：**灵魂先由数据与评测门定义，引擎只提供机制**——不许拿引擎 hack 冒充行为训练。

### P4：多模态插件（2026-07-11 用户裁决：主攻**前端开发域**）
- 定位：图片→文本的编码不再是"裸 OCR 行"，而是**前端可推理的 UI 结构草图**——
  尺寸/调色板/匀色块几何/文字(含前景背景色)，CSS 同款左上原点像素坐标，
  足以支撑"按钮什么颜色/为什么没对齐/照截图写 HTML+CSS/贴报错截图修 bug"。
  - ✅ 2026-07-11 前端域编码器 `tools/mm_ui.swift`（`make mm-ui`）：flood-fill 匀色块矩形
    (tol=6 分得开 #ffffff 卡片 vs #f5f6f8 页底)+5bit 众数桶颜色(design token 级真值)+
    sRGB 钉死(hex 直接进 CSS、跨机确定)+Vision OCR 带几何/fg-bg 色+同形折叠(西里尔 с→c
    防毒化 KV 键)。`--selftest` 五门(逐字/按钮/顶栏/白卡分离/字色关联)全绿，输出跨进程
    字节一致；`--render-ui` 供端到端测试。mm-ocr 降级为诊断工具。
  - ✅ 2026-07-11 C 侧接线：`ds4_multimodal.c` 加 TEXT 级编码(`ds4_mm_encode_as_text`，
    server 管线全文本哈希所以必须 text 级)+严格 base64；引擎持有 registry
    (`ds4_engine_mm`)，自动绑 `./mm-ui`（`DS4_MM_IMAGE_CMD` 覆盖）；multimodal.o 进 CORE_OBJS。
  - ✅ 2026-07-11 server 落地：`/v1/messages` image content block（base64）→ 编码为
    `<image>…</image>` 文本进 content——KV 前缀键/精确重放字节稳定，同图重贴命中缓存；
    无编码器/URL源/坏base64/不支持类型一律 fail-closed 400（替换原静默丢弃）。
    单测 `ds4_test --server` 新增 registry/b64/image-block 三路 fail-closed，全绿。
  - ✅ 2026-07-11 **物理方位插件 `ds4_spatial.c`**（enricher 链首个）：草图→空间关系闭式
    推导——`[ids]` 标签系统 / `[where]` 九宫格方位 / `[in]` 包含树 / `[align]` 对齐组+
    居中判定 / `[stack]` 列/行流+间距。模型不用再从裸坐标算"按钮居中吗/右上角哪个"。
  - ✅ 2026-07-11 **CSS 理解插件 `ds4_css.c`**：草图→CSS 布局事实——`[css page]` 底色、
    容器 padding 四向、`display:flex; flex-direction; gap; align-items` 推导。只输出
    派生属性（宽高颜色草图已有不重复）；font-size/border-radius 诚实不猜。
  - ✅ enricher 链机制（`ds4_mm_register_enricher`）：编码器出文本后按模态追加增强节，
    每个 enricher 只见编码器原文（无解析链耦合）；text/token 两形同源（token 形对增强
    文本 tokenize）；非草图输入两节诚实缺席。引擎 open 自动挂 image 族。
  - ⬜ OpenAI chat-completions `image_url` 块（Claude Code 不需要，低优先）。
  - ⬜ ds4-agent 进程内贴图入口（`/img <path>`，token 级走 `ds4_mm_encode`）。
- **门（前端域 v2）**：Claude Code 里贴一张 UI 截图 → 模型说对按钮颜色/布局并产出对应
  HTML/CSS；贴一张 JS 报错截图 → 逐字读出 `TypeError…` 并修复。（真模型端到端待跑）

### P5：集成收敛
- ds4-agent（进程内）与 Claude Code（API）双通道并存；会话=磁盘 KV（重启续）。

### P6（最后一项，此前全程冻结）：速度优化
> 用户裁决 2026-07-07：体积、质量、速度三者的优化**全部冻结**，先把功能链路走完；
> 速度是整个清单的最后一项。此前任何阶段只记录速度数字，不为速度改任何东西。
- 届时的杠杆清单（按已有判决排序）：copy-spec/knowledge-MTP 在编辑型工作流吃满（有效 t/s，§3.5）、
  prefill 前沿（chunk/cap 重标定）、request-batching（agent 多流聚合，notes/request-batching-design.md）、
  MTP 草稿净收益复评、双机 spec-pipe 链深。
- 基线锚点（冻结时实测，回归判据）：双机 mono prefill ~70 t/s / decode 1.6-1.8 t/s / 跨轮增量 prefill 44 tok 级。

## 2.5 量化还原度里程碑（2026-07-09）

严格 1-bit 动态量化(每层动态标量增益 D0/D1 + 动态四损失 + per-block 输出最优 scale + 向前/向后/感知，`gguf-tools/go-onebit/scripts/full_optimize.py`)诚实还原度(分布还原率 Σmin/KL/PPL)：

| 域 | 分布还原率 | PPL 比 |
|---|---|---|
| **编程域(真实多样代码)** | **0.7902** | **×1.24** |
| 通用 hard-text(对照) | 0.5450 | ×2.71 |

- 编程窄域远好 → 79%/×1.24 = 接近可用，印证「model 已可用」。
- 已证边界(见 fable5 2026-07-09)：①逐层 relL2 是中间层投影伪影(只认末层/分布)；②低秩方向 z(RRR)43/43 层 held-out 全拒 = 严格 1-bit 方向到顶，**79% 是 1-bit 天花板**，再抬只有 2-bit(混精 L38-42 均值 1.18bit/w)——用户要 1-bit 则 79% 即上限。
- **结论**：到"正常业务开发"的 gap **不在量化数(已 79%)，在 agent/解码层长上下文名字检索** → 走 P0/P2。

## 3. 立即下一步（按序，各带判决；P1/P6 冻结不在此列）

1. **P0 收尾三缺口**：①server cap 截断改答 `end_turn`（小改）②冒烟加 `MAX_THINKING_TOKENS=0`
   ③修 mid-prefill continued KV 在双机流水线下的 snapshot mismatch（真 bug）。
2. **P0 真任务门**：⚙ Gate v2 判决(2026-07-07)——机制层 8/8 全胜(guided 调用+CC 回路自转)；内容层输给长上下文检索(填模板占位符而非 23.5k 前部的用户路径；短上下文同模型填对)。few-shot 污染源已删。**门收敛为 P2 依赖**：长上下文+工具域校准侧车。
3. **P2 首个域侧车**：Go 真项目激活捕获 → `ds4_z_solve` → corr 侧车 → 快判集。
4. 容错 matcher 归一化（纯代码）：ref-corpus 锚匹配大小写折叠 + 单测。

## 4. 不做清单（防跑偏，历史判决）

- 不做删容量的压缩（数量裁专家/瘦-F 已判死；引擎已对 GO 类型硬拒 thinning）。
- 不自训练替代模型当产物（恢复原始能力铁律；训练只允许出现在侧车/行为语料层）。
- 质量门未过的任何杠杆不进默认（MOE_THIN 教训）。
- q2/hf 原始模型永不删（当前 q2 两机缺失待用户裁决恢复）。
