# 代码生成基准测试方案设计（Go + React）

> 本文档是 `benchmarks/` 下 Go / React 两套大模型代码生成基准的**正式方案设计**。
> 设计以主流学术/工业基准的公开方法论为依据（见文末"引用来源"），并针对本仓库
> "只填 API 地址 + key 即可跑、接受 Claude/Anthropic 消息体格式"的目标做了工程化收敛。
>
> 适用对象：要给某个大模型（含本仓库的 DeepSeek V4 `ds4-server`）的 Go / React 编码能力打分。

---

## 0. 为什么要重做（旧设计的问题）

第一版基准已具备：HumanEval 风格题面、隐藏单元测试、`go test` / `vitest` 执行式评测、
pass@1 / pass@k（Codex 无偏估计）、0–100 得分。但对照专业基准的标准做法，它**过于简单**，
存在以下会导致"分数虚高、不可信、易被污染"的缺口：

| 缺口 | 后果 | 标准做法（依据） |
|------|------|------------------|
| 每题只有**一套**隐藏测试，未度量充分性 | 测试不足会**系统性高估** pass@k，甚至让模型**排名错乱** | EvalPlus：加测后 pass@1 最多 −19~29%；BigCodeBench 要求 ~99% 分支覆盖 |
| 无**覆盖率/充分性门禁** | 无法证明"通过=真的对" | BigCodeBench 用分支覆盖率作为充分性标准 |
| 无**置信区间 / 采样协议** | 单点分数无法判显著性；pass@k 的 n、温度未规范 | Codex pass@k 需 n≥k；贪心测 pass@1、采样测 pass@k |
| 沙箱**隔离不足**（共享构建缓存、并发题无 `-race`、未记录工具链版本） | 不可复现、并发 bug 漏检、跨题串扰 | SWE-bench / BigCodeBench：隔离沙箱、断网、固定环境 |
| 无**数据污染防护/检测** | 题目可能已在训练集中，分数不可信 | LiveCodeBench 时间分段；n-gram 重叠检测 |
| prompt 协议未标准化（无 stop、无 top-p、few/zero-shot 未声明） | 结果不可复现、跨模型不可比 | bigcode-evaluation-harness 规范 prompt/停止符/抽取 |

本方案把以上每条缺口都落到**可执行的 harness 能力 + 题库规范**上。

---

## 1. 设计原则（提炼自主流基准，逐条标注依据）

### P1. 功能正确性 = 执行通过（不靠文本相似度）
模型产物在**隔离环境里真实编译运行**，仅当**全部隐藏测试通过**才算 pass。
这是 HumanEval / MBPP / EvalPlus / BigCodeBench 的共同根基〔1〕〔2〕〔3〕〔5〕。

### P2. 测试充分性是第一质量杠杆 —— base/plus 双层 + 覆盖率门禁
- **现象**：HumanEval 平均仅 **7.7** 测试/题、MBPP 仅 **3** 断言/题；测试不足 →
  错误解被误判为通过 → pass@k **系统性高估**，并导致模型**误排名**（EvalPlus 实证：
  WizardCoder 等排名在加测后变化）〔3〕。
- **量级**：EvalPlus 用自动化手段把测试规模放大约 **80×**，使被测 pass@k **最多下降 19.3~28.9%**〔3〕〔4〕。
- **标准**：BigCodeBench 强制高充分性，每题接近 **99% 分支覆盖率**〔5〕〔6〕。
- **本方案落地**：
  1. **base 层**：原始隐藏测试（基本正确性）。
  2. **plus 层**：边界/压力/对抗用例（空、极值、并发竞态、错误路径、属性/不变量）。
     报告同时给出 `pass@1(base)` 与 `pass@1(plus)`，以及**高估缺口 = base − plus**。
     **最终严格得分以 plus 层为准**（无 plus 时退回 base）。
  3. **覆盖率充分性门禁**：自检时用 canonical 解跑覆盖率，
     **要求每题语句/分支覆盖率 ≥ 阈值（默认 90%）**，否则判定"该题测试不充分"，
     提示补题——把"测试是否够"从主观变成可度量（对齐 BigCodeBench 的充分性理念）。

### P3. 指标：pass@k 无偏估计 + 采样协议 + 置信区间
- **估计量**：`pass@k = 1 − C(n−c, k) / C(n, k)`（n 个样本中 c 个通过），要求 **n ≥ k**〔1〕。
- **采样协议**（可复现、跨模型可比）：
  - `pass@1`：**贪心解码 temperature=0**（BigCodeBench 主指标即贪心 Pass@1〔5〕）。
  - `pass@k (k>1)`：温度 **0.8**、top-p **0.95**、`n ≥ k`（沿用 Codex 采样设置〔1〕）。
- **置信区间**：对"每题 pass 率的题集均值"给出 **95% 置信区间**（题间方差的正态近似/自助法），
  使两次评测/两个模型的差异可判显著性。

### P4. 沙箱隔离 + 可复现
执行不可信代码必须隔离〔5〕〔6〕〔14〕：
- **断网**（Go `GOPROXY=off`；React 离线、mock fetch），
- **每题超时**、临时目录**用完即删、互不串扰**，
- **隔离构建缓存**（独立 `GOCACHE/GOMODCACHE/GOTMPDIR`，避免跨题/跨机污染），
- **并发题强制 `go test -race`**（竞态检测，对 goroutine/channel 题是硬要求），
- **确定性**：固定随机种子、React 用假定时器；
- **记录环境**：报告写入 `go version` / `node` / `vitest` 等**工具链版本**，保证可复现。

### P5. Prompt 协议标准化（可复现、可比）
- **zero-shot instruct**：统一 system 提示 + 题面，声明"只输出一个代码块"。
- **停止符**：可配置 `stop_sequences`，避免模型续写测试/解释污染抽取。
- **采样参数**：`temperature` / `top_p` 显式可配。
- **稳健代码抽取**：优先带语言标注的 fenced block → 任意 fence → 裸代码；多块取最长。
- **消息体**：严格 **Anthropic Messages 格式**（`POST /v1/messages`，`x-api-key` +
  `anthropic-version`，`messages:[{role,content}]`），与 Claude 完全一致，便于直连任意兼容端点〔17〕。

### P6. 数据污染防护与检测
- **手写原创题**，不从公开题库照搬（HumanEval 即强调手写以防训练集重叠〔1〕）。
- **可检测**：提供 `--contamination-check <语料>`，计算每题 prompt(+canonical) 与给定语料的
  **n-gram（默认 13-gram）最大重叠比例**，对齐 n-gram 重叠/时间分段的污染识别方法〔7〕〔12〕。
- **可时间分段**：每题元数据带 `created` 日期与 `held_out` 标记，支持"只评模型训练 cutoff
  之后产生的题"（LiveCodeBench 的 release-date 防污染思路〔7〕〔18〕）。

### P7. 语言专属测试策略
> 说明：以下 Go/React 具体测法是基于通用执行式评测原则的**工程推荐**（研究覆盖偏 Python；
> 见"局限"）。多语言可行性参考 MultiPL-E（把 HumanEval/MBPP 翻译到含 Go 在内的 18+ 语言）〔19〕。
- **Go**：表驱动单元测试 + 边界用例；并发题加 `-race`；适用处加**属性/不变量测试**
  （与朴素串行实现对比、随机大规模一致性）；隐藏测试与题面强契约（固定函数签名、`package solution`）。
- **React**：以 **@testing-library** 做**行为/用户视角断言**（按 `data-testid`/role/文本、`userEvent` 交互），
  避免脆弱的快照测试；定时器/副作用题用**假定时器**；异步题**注入可控 mock**；强契约（组件名、props、`export default`）。

---

## 2. 目标基准的具体方法论速查（被本方案借鉴的点）

| 基准 | 关键做法 | 本方案采纳 |
|------|----------|-----------|
| **HumanEval**〔1〕 | 164 手写题，平均 7.7 测试；pass@k 无偏估计；手写防污染 | pass@k 估计量、手写原创题 |
| **MBPP**〔2〕 | 974 众包题，每题 3 断言 | 反例：印证"测试太少"的风险 |
| **EvalPlus / HumanEval+**〔3〕〔4〕 | 自动扩测 ~80×，pass@k 掉 19~29%，纠正误排名 | **base/plus 双层**、充分性优先 |
| **BigCodeBench**〔5〕〔6〕 | 1140 题，~99% 分支覆盖，贪心 Pass@1 主指标 | **覆盖率充分性门禁**、贪心 pass@1 |
| **LiveCodeBench**〔7〕〔18〕 | 持续收题 + release-date 防污染 | `created`/`held_out`、时间分段 |
| **SWE-bench harness**〔14〕 | 隔离沙箱、固定环境、可复现 | 沙箱隔离 + 工具链版本记录 |
| **MultiPL-E**〔19〕 | 把基准翻译到 18+ 语言（含 Go） | Go 多语言评测可行性背书 |
| **bigcode-evaluation-harness**〔20〕 | 规范 prompt/停止符/抽取 | Prompt 协议标准化 |
| **污染研究**〔12〕〔17〕 | n-gram 重叠 + 时间分段检测 | `--contamination-check` |

---

## 3. 本基准的目标规格（Go + React 统一）

```
benchmarks/
  go-bench/    Go：go test 执行；并发题 -race；覆盖率门禁
  react-bench/ React：vitest + @testing-library；行为断言；覆盖率(best-effort)
```

**统一约定（两套一致，可横向对比）**
- 消息体：Anthropic Messages（见 P5）。
- 配置优先级：CLI flag > 环境变量 > 默认值；"只填 `--api-base` + `--api-key` 即可跑"。
- 题库结构（每题一目录）：
  - `meta.json` / `problem.json`：`id, title, difficulty, tags, created, source, held_out`。
  - `prompt.md` / `problem.json.prompt`：题面（强契约，零样本）。
  - 隐藏 base 测试 + 可选 **plus 测试**（`*_plus_test.go` / `*.plus.test.jsx`）。
  - `canonical`：参考解（仅自检/覆盖率/污染检测用，绝不进 prompt）。
- 指标与评分：
  - `pass@1(base)`、`pass@1(plus)`、高估缺口、`pass@k`（无偏估计）、95% CI；
  - 覆盖率（canonical 在隐藏测试下的语句/分支覆盖）；
  - **得分 = pass@1(plus) × 100**（严格层），并给难度加权得分与 easy/medium/hard 分档、等级。
- 报告 `report.json`：含 model、时间戳、**完整协议与工具链环境**、每题 base/plus/覆盖率/样本明细。

**评分口径（0–100）**
- 主得分 = 严格层 pass@1 × 100；难度权重 easy=1 / medium=2 / hard=3；
- 等级 ≥90 A · ≥75 B · ≥60 C · ≥40 D · 否则 F。

---

## 4. Harness 能力清单（本次落地）

1. **base/plus 双层执行**：自动发现 plus 测试；分别报告，给出高估缺口。
2. **覆盖率充分性门禁**：`--self-test` 用 canonical 跑覆盖率，低于阈值（默认 90%）告警。
3. **沙箱硬化**：隔离构建缓存、断网、并发题 `-race`、超时、用完即删、记录工具链版本。
4. **采样协议 + 统计**：`--temperature/--top-p`，pass@k 无偏估计，题集均值 95% CI。
5. **Prompt 协议**：统一 system、可配 `stop`、稳健抽取。
6. **污染工具**：`--contamination-check <FILE>` 输出每题最大 13-gram 重叠比例；元数据 `created/held_out/source`。
7. **报告与评分**：base/plus/缺口/覆盖率/CI/环境全部进 `report.json` 与控制台得分框。

---

## 5. 运行手册（最简）

```sh
# Go：只填 api-base + key
cd benchmarks/go-bench
python3 run_bench.py --self-test                         # 离线：跑 canonical + 覆盖率门禁
python3 run_bench.py --api-base http://localhost:8080 --api-key YOUR_KEY        # 贪心 pass@1(base/plus)
python3 run_bench.py --api-base ... --api-key ... -n 5 --k 5 --temperature 0.8 --top-p 0.95   # pass@5
python3 run_bench.py --contamination-check /path/to/corpus.txt                  # 污染自检

# React：先装一次依赖
cd benchmarks/react-bench
npm install
python3 run_bench.py --self-test
python3 run_bench.py --api-base http://localhost:8080 --api-key YOUR_KEY
```

---

## 6. 局限与后续路线（诚实声明）

- 研究语料以 Python 基准为主；**Go/React 的具体测法属工程推荐**，非逐条文献结论。
- pass@k 置信区间用题集均值的正态近似/自助法（实务常用），非某一篇论文的钦定口径。
- **plus 层是渐进扩充的**：本次先建机制 + 覆盖率门禁（充分性可度量），并对高价值题补 plus；
  目标是逐步把每题推到 BigCodeBench 级（≈99% 分支覆盖）。
- LiveCodeBench 类"持续收题"需要长期运营；本仓库以 `created/held_out` + 污染检测做轻量替代。

---

## 引用来源（deep-research 对抗核验 25/25 通过，0 推翻）

1. Chen et al., *Evaluating LLMs Trained on Code*（Codex / HumanEval）— arxiv.org/abs/2107.03374
2. Austin et al., *Program Synthesis with LLMs*（MBPP）— arxiv.org/pdf/2108.07732
3. Liu et al., *Is Your Code Generated by ChatGPT Really Correct?*（EvalPlus）— arxiv.org/abs/2305.01210
4. EvalPlus 项目主页 — evalplus.github.io ；代码 — github.com/evalplus/evalplus
5. Zhuo et al., *BigCodeBench* — arxiv.org/html/2406.15877v4
6. BigCodeBench 代码 — github.com/bigcode-project/bigcodebench
7. Jain et al., *LiveCodeBench* — arxiv.org/abs/2403.07974 ；主页 — livecodebench.github.io
12. *Data contamination via n-gram overlap / 时间分段分析* — arxiv.org/html/2406.04244v1
14. SWE-bench 评测 harness 文档 — swebench.com/SWE-bench/reference/harness/
17. *污染防护与 prompt 协议* — arxiv.org/abs/2311.04850
18. LiveCodeBench（prompt 协议 / 污染防护角度）— arxiv.org/abs/2403.07974
19. Cassano et al., *MultiPL-E*（多语言含 Go）— nuprl.github.io/MultiPL-E/
20. bigcode-evaluation-harness 文档 — github.com/bigcode-project/bigcode-evaluation-harness

> 另含 metrics & 语言专属来源：openreview 1qvx610Cu7、arxiv 2501.01054、2510.04265、
> 2506.13832、2511.10868、2506.18315、2509.26111、2506.06251（详见 deep-research 报告）。
