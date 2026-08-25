# P6 速度计划 — guided 工具调用每轮延迟分解 + 杠杆排序

> 2026-07-07，worktree `agent-aefddd3f7c34793a0`（并行分析线，未合并、未运行模型）。
> 证据源：`/tmp/ds4-svc.log`（07:38–07:49 真实 Claude Code guided 会话，mono 双机
> M4(0:19)+M1(20:output)，ctx≈23.5k–24.7k）。基线锚点见 `tiny-coder-plan.md` P6 段。

## 1. 每轮延迟分解（实测，逐轮）

注入 token 数推导：`下一轮 continuation 的 cached − (本轮 prompt 末位 + gen)`
（例：轮2 cached=23672，prompt 末 23599 + gen 33 = 23632 → 注入 40）。
每 eval 秒 = decode 段秒 ÷ (自由 token + 注入 token)。

| 轮 | 工具 | 增量 prefill tok / s | 自由 gen tok | 注入 tok | decode 段 s | s/eval | 轮总 s |
|---|---|---|---|---|---|---|---|
| 1 | Read       | 23482(全量)/298.3 | 6  | 40 | 30.2 | 0.657 | 328.5 |
| 2 | Write      | 71 / 10.31  | 33 | 40 | 46.8 | 0.641 | 57.1 |
| 3 | Write      | 42 / 8.45   | 97 | 40 | 87.8 | 0.641 | 96.2 |
| 4 | Edit       | 42 / 9.66   | 81 | 40 | 80.3 | 0.664 | 90.0 |
| 5 | Edit       | 53 / 10.04  | 81 | 40 | 80.1 | 0.662 | 90.2 |
| 6 | TaskCreate | 53 / 11.50  | 98 | 39 | 91.6 | 0.668 | 103.1 |
| 7 | TaskUpdate | 113 / 12.95 | 98 | 40 | 89.5 | 0.648 | 102.4 |
| 8 | Skill      | 15 / 6.04   | 96 | ~40 | 91.0 | ~0.669 | 97.0 |

（另：06:50 的 452-token 独立探针 = "46s 轮"下界锚点：prefill 17.6s + decode 28.7s
= 8 自由 + 40 注入 = 48 eval × 0.597s。）

**结论（稳态轮 90–103s 的构成）**：

| 成分 | 秒 | 占比 | 机制 |
|---|---|---|---|
| 自由 token decode | 53–64 (81–98 tok × 0.65s) | 55–62% | 每 token 一次双机往返，1.5 t/s |
| **结构 token 逐个注入** | **25–27 (≈40 tok × 0.65s)** | **26–46%**（短轮占大头） | PRIMER_INJECT 每 token 调一次 `ds4_session_eval` = 一次完整双机 forward，成本与 decode 完全等价（0.64–0.67 s/eval 与自由 decode 无差） |
| 增量 prefill | 6–13 (15–113 tok) | 7–13% | 单个 dist span；**~5.5s 固定截距 + 0.07 s/tok 边际**（15tok→6.0s，42→8.4s，113→13.0s） |
| 其它（解析/渲染/KV 记账） | <0.1 | ~0% | 轮总−prefill−decode 段 ≈ 0.002s（轮5：90.188−10.044−80.142） |

疑点证实：**PRIMER_INJECT 逐 token eval 确认为 ~26s/轮纯注入开销**（任务书怀疑成立，
且注入 eval 与 decode eval 单价严格相同——结构 token 并不比自由 token 便宜）。

## 2. 已落地 patch：结构 token 批量注入（本 worktree，未合并未运行）

- 位置：`ds4_server.c` guided 块 `PRIMER_INJECT` 宏（≈L10356）。
- 改法：每个字面量先 `ds4_tokenize_rendered_chat`（与旧路径同一 tokenizer 调用，
  token 流逐字节一致），然后拷贝 `ds4_session_tokens()`（live checkpoint）+ 追加注入
  token，调 **`ds4_session_sync`** 一次性批量前向——这正是 server prompt prefill 的
  同一入口：分布式路由变成单个 `dist_coordinator_eval_span`（一个 WORK 帧、
  layer-major 批），单机 Metal 走 `metal_graph_prefill_chunked_range` resume
  （<4 token 字面量自动落回 sync 内部的逐 token 环，行为同旧）。
- 行为不变依据：批量 span 与逐 token eval 的 KV/位置等价是 copy-spec 提交已验证
  draft 批所依赖的既有不变式；sync 后 `s->logits` = 最后一个注入 token 的 logits，
  紧随的 `PRIMER_GEN` argmax 语义与旧路径一致；`gok=false` 后宏保持旧的 no-op；
  ctx 越界改为整段前置检查（更严格、同为 gok=false）。
- 编译：`make ds4-server` 零 error（ds4_server.o 零 warning）。
- 预期：26s/轮 → 3 个小 span（opener/param-head/closer）。收益区间的两个界：
  - 悲观界（注入 span 成本 = 实测冷增量 prefill 小 span 曲线 5.5+0.07n）：3×~6s=18s，
    省 ~8s/轮；
  - 乐观界（DSML 结构 token 每轮重复 → 专家页缓存热，span 成本→RTT+kernel ~1–2s）：
    3–6s，**省 ~20–23s/轮**。
  - 哪个界成立由 A/B 判决（这同时回答 §3-杠杆4 的固定截距归因）。

### A/B 验证步骤（主线常驻服务执行；本 worktree 不跑）

1. `tools/svc.sh up`（已常驻则幂等）。记 `git diff` 只含本 patch 后在主线重编:
   `make ds4-server`，重启 server 端（worker 无需动，M1 不涉及本改动——改动只在
   coordinator 侧 server 代码；仍按铁律确认 CORE_OBJS 未变即免同步）。
2. 发一条**带 tools** 的探针（stock `probe` 不带 tools，guided 不触发）：
   ```sh
   curl -s --noproxy '*' http://127.0.0.1:8013/v1/messages -H 'content-type: application/json' -d '{
     "model":"deepseek-chat","max_tokens":200,"temperature":0,
     "tools":[{"name":"Read","description":"Read a file",
               "input_schema":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}],
     "messages":[{"role":"user","content":"Read the Go file /tmp/gotask/main.go"}]}'
   ```
3. 判据（`/tmp/ds4-svc.log`，与 07:38–07:49 基线行对齐）：
   - 正确性：`tool calls ... names=[Read]`、`finish=tool_calls`、探针输出
     `TOOL_USE: Read {"file_path": "/tmp/gotask/main.go"}` 与基线逐字节一致
     （temp0 + argmax → 应完全复现）。
   - 速度：`decoding ... chunk=X t/s ... Ns` 的 N 与基线同 gen 数对比；
     以及 (下一轮 cached − prompt − gen)=注入数不变（≈40），
     但 decode 段秒 ≈ 自由 tok×0.65 + 3×span 成本。452-tok 探针基线 46.3s 轮总
     → 预期 ~28–38s。
   - 回归门：CC 真流量一轮（`tools/svc.sh claude ...`）确认 continuation
     `match=tool-output-ids` 仍命中（cached 长度语义未变）。
4. 若行为漂移（tool call 字节不一致）→ 直接判 patch 有 bug（不是采样噪声，
   全链路 temp0），回退重查——不留兜底开关。

## 3. P6 杠杆 Top-5（按预期收益排序）

基准：稳态 guided 轮 90–103s；短轮（少自由 token）46–57s；CC 首请求 23.5k≈5 分钟。

| # | 杠杆 | 预期收益 | 验证方法 |
|---|---|---|---|
| 1 | **结构 token 批量注入**（本 patch，已写码） | **-8 ~ -23s/轮**（26s→3–18s；短轮相对收益最大：46s 轮→~25–38s） | §2 A/B；判决同时给出小 span 固定成本归因 |
| 2 | **copy-spec 接管自由 decode**（`DS4_DIST_COPY_SPEC` n-gram drafter 上 dist 路由；PRIMER_GEN 的 value 生成与自由轮 decode 都是 argmax→兼容验收协议） | 自由 decode 53–64s 是最大块（55–62%）。编辑型值（路径/代码行）逐字引用上下文 → 历史判决 code-edit 有效 t/s 2–5×；**-15 ~ -40s/轮**（Write/Edit 轮最大） | svc 常驻 + `DS4_DIST_COPY_SPEC=1` 重启 server A/B 同一 CC 任务；coord log 看 accept_len 分布；温度非 0 的普通轮不受影响（greedy-only 自门控） |
| 3 | **closer-skip**：终结段 `</parameter>...</tool_calls>` 之后无任何采样，其 eval 可整段跳过——文本仍进响应/重放映射，KV 由下一轮增量 prefill 以 0.07s/tok 边际价补上 | 在 #1 之上再 **-4 ~ -6s/轮**（3 span→2 span） | 小 patch（去掉第三个 PRIMER_INJECT 的 eval、保留 buf_append）；A/B 判据加一条：下一轮 continuation cached 减少 ~12 tok 但仍 `match=tool-output-ids` 命中 |
| 4 | **小 span 固定截距（~5.5s）归因与消除**：15tok span 6.0s vs 单 token 0.65s，9× 不成比例；嫌疑=批内新增 unique 专家冷 IO vs 管线/gather-barrier 固定开销 | 增量 prefill 6–13s → ~2–6s，**-3 ~ -7s/轮**；若归因=冷专家 IO，同时抬高 #1 的实际收益 | `DS4_DIST_PIPE_PROFILE=1` + `DS4_METAL_EXPERT_IO_PROFILE=1` 起 svc，跑一条 42-tok 级增量探针，读 t_local/t_remote 与 gather 分解；按归因选杠杆（专家预取 staging vs span 路径削固定开销） |
| 5 | **首请求 23.5k 全量 prefill 的跨会话复用**：298s 一次性成本；07:37:54 已落盘 cold 23364-tok checkpoint，但 07:33 出现 `live kv cache miss ... common=1 reason=token-mismatch`（前缀首 token 即不匹配 = 系统提示/工具集漂移，disk 也救不了） | 再次 CC 会话 23.5k → 若 disk 命中只补尾部：**~5min → <1min/会话** | 主线重启一次 CC 同任务，看 `kv cache hit`/`PC.4` 行；若仍 token-mismatch，diff 两次渲染 prompt 首 2048 tok 定位漂移源（工具集顺序/日期头），再决定 anchor 化 |

不做/暂缓（历史判决防重走）：MTP 草稿净收益复评（off-host MTP 判决=受延迟/退化封顶）、
request-batching（单 CC 流无聚合对象，agent 多流才启用）、任何削专家容量的"提速"。

回归门（任一杠杆落地时，按 CLAUDE.md）：guided 探针字节一致（temp0）+
`ds4_test --server`；触路由/量化的另加 `--logprob-vectors` + q1..q4。

---

## 后记(2026-07-20 合并入主线时补记)

本文档从 worktree `agent-aefddd3f7c34793a0` 归档合并(快照存档于该分支提交 d8939af)。
worktree 内的批量注入 patch **未取代码**——主线已独立走完整个弧线,判决以主线为准:

- 杠杆1(批量注入): 07-07 主线合并→实测质量回归→回退;07-14 以
  `DS4_PRIMER_BATCH_INJECT` env 杠杆复活,首次 A/B=胜(轮延迟 15min→~5min);
  现行实现在 `ds4_server.c` `PRIMER_INJECT_KV`。
- 更优后继: `DS4_PRIMER_COMPACT`(07-14 紧凑 KV 注入,text/KV 解耦,68→~20 tok),
  超越本计划的 span 化思路。
- 杠杆2(copy-spec): 已成天然默认 drafter(env 全删,惩罚重放 gate)。
- 杠杆3(closer-skip)/杠杆4(小 span ~5.5s 固定截距归因)/杠杆5(首请求 23.5k
  前缀跨会话复用): 截至本后记未见主线落地记录,仍是开放杠杆(P6 冻结令约束下排队)。
