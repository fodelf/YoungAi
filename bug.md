# bug.md —— 温 0 复读问题的引擎侧审计: 写死参数逐项对官方 + 一处确认的提示模板 bug(2026-09-21 夜)

用户令: "现在项目要设置 DRY 才能不重复, 明显是 bug, 分析这个问题给出解决方案" → "我猜肯定是引擎哪里写死参数的 bug, 为了速度或者等等其他原因, 乱写一通" → "思考得出解决方案后把方案落地 bug.md, 不要直接改代码"。
本文只做两件事: ①把引擎生成路里每一个"写死的量"跟官方 `config.json` / `inference/model.py` / `encoding/encoding.py`(都在 spark `~/ds4-main/hf/DeepSeek-V4.1-Flash/`)逐项对账; ②给改码清单、门、验证顺序。**没改一行代码。** DRY 判死的实测与死循环形态见 fable5.md 09-21 夜末两节, 这里不重复。

## 0 结论(先说)

1. **数值路的写死参数逐项对过, 没有对不上的**(§2 表): 窗口 128 / 索引 topk 512 / 候选块 2048×8 / 压缩比表 / 源层表 / RoPE θ 10000·160000 / YaRN 16·32·1·65536 / sinkhorn 20 / eps 1e-20 / route_scale 1.5 / swiglu 10 / 头数·维度·秩, 全部从 GGUF 元数据读(转换器 `v41_to_gguf.c` 逐项落, 引擎 `core_validate.c` 逐项核), 与官方相同。生成路独有的核(radix topk / 候选块 / 解码整步 graph / 窗口环 / 压缩步)读码是逐位复刻官方语义, 两级 topk 在 16384 之后才生效的那一支也对得上。
2. **找到一处确认的 bug, 不在数值路, 在提示模板, 而且产品路每一条请求都中**: 带 system 消息的对话, 官方渲染是
   `<｜begin▁of▁sentence｜><｜System｜>{system}<｜User｜>{user}<｜Assistant｜></think>`,
   引擎渲染是 `<｜begin▁of▁sentence｜>{system}<｜User｜>…` —— **漏了 `<｜System｜>`**。它是 V4.1 新增的特殊 token(id 128799; V4 的 tokenizer 里没有), 引擎的模板是 V4 的, 没随 V4.1 更新; 引擎 tokenizer 的特殊 token 表也没登记它(就算渲染出来也会被当普通文本切碎)。qtf 的每条请求都带 system(CFO 人设 + 原则) ⇒ **产品一直在一个模型没训练过的提示格式上跑**。同根还有三处(§1.3): thinking 模式的 effort 前缀、会话中途的 system 消息、工具块的位置与措辞。
3. **它不能直接给死循环定罪**: 模型在错格式上照样写出了正文和数字, 错格式只是把整条分布往"没见过的样子"推了一把, 推多少没量过。但它是产品路上唯一"每条请求必错、修了零风险(只是对齐官方字节)"的东西, 所以**第一刀是它**, 验证 = 同两只股票、同二进制、只差模板, 温 0 前后成对跑 qtf replay(§5)。
4. 生成路独有、与判决路不同的三处偏差(§3): CED 解码器回放长度 = 提示长 mod 512(官方固定 128) / 解码器 20 层从没写过的窗口环槽不屏蔽照读(官方标 -1 屏蔽; 读到的是 cudaMalloc 回收的上一条请求的 KV) / 整个 CUDA 后端 `--use_fast_math`(sin·cos·exp·pow·div 全走近似, RoPE 角度误差随位置线性涨)。都是"小噪声"量级, 修法与门在 §6, 排在 §4 之后。

## 1 提示模板 bug: 证据(全部可复现)

### 1.1 官方编码器实跑(spark, `python3` 直接 import `encoding.py`, 不需要模型)

```
chat     + system : '<｜begin▁of▁sentence｜><｜System｜>S<｜User｜>U<｜Assistant｜></think>'
chat     无 system: '<｜begin▁of▁sentence｜><｜User｜>U<｜Assistant｜></think>'
thinking + system : '<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\nS<｜User｜>U<｜Assistant｜><think>'
thinking 无 system: '<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>U<｜Assistant｜><think>'
多轮 + 中途 system: '<｜begin▁of▁sentence｜><｜System｜>S<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜><｜System｜>mid<｜User｜>q2<｜Assistant｜></think>'
带 tools         : '<｜begin▁of▁sentence｜><｜System｜>S\n\n## Tools\n\nYou have access …'   (工具块在 system 内容之后)
```
官方自己的单测 `encoding/test_encoding.py::test_v41_leading_system_message_uses_system_token` 断言的就是第一行。
(那份 test 文件里另有两个用例期望没有 `<｜System｜>`, 与代码实跑矛盾 —— 以实跑的 `encode_messages` 为准, 它就是官方 `generate.py` 用的那份。)

### 1.2 引擎实际渲染(spark `/tmp/ds4-trace-real-v3.txt` 请求 2, qtf 真实请求, messages = 1 条 system + 1 条 user)

```
<｜begin▁of▁sentence｜>You are 量化交易首席策略官（风险回报比优先策略）… 8.输出报告标题格式… Thought:<｜Assistant｜></think>
```
BOS 之后直接是 system 正文, 没有 `<｜System｜>`。渲染代码 `src/server/server_dsml_render.c:375-377`:
`buf_puts(&out, "<｜begin▁of▁sentence｜>"); if (think_mode == DS4_THINK_MAX) buf_puts(&out, ds4_think_max_prefix()); buf_puts(&out, system.ptr)`。
CLI 的 `--system` 同病(`src/core/core_bpe.c:351` encode_chat_prompt), agent/REPL 的 `ds4_chat_append_message(…, "system", …)` 同病(core_bpe.c:449), completion 端点注入的 "You are a helpful assistant" 同病(`server_parse_completion.c:187-189`)。

### 1.3 token 表

| token | V4.1 tokenizer | V4 tokenizer | 引擎特殊表 `core_bpe.c:372 special_token_at` |
|---|---|---|---|
| `<｜System｜>` | **128799** | 没有 | **没登记**(登记的只有 BOS/EOS/User/Assistant/think/DSML) |
| `<｜latest_reminder｜>` | 128828 | 128828 | 没登记(官方 latest_reminder 角色, 引擎不用, 不算错) |

⇒ 引擎的聊天模板是 V4 时代写的; V4.1 官方 `encoding/README.md` 第 25 行明写 "Mid-conversation system messages are supported via the `<｜System｜>` token", 第 152 行写 thinking 模式格式 `<｜begin▁of▁sentence｜><｜System｜>{reasoning_effort_prefix}{system_prompt}`。

### 1.4 同根的另外三处(都是"模板没随 V4.1 更新")

| 项 | 官方 V4.1 | 引擎 | 影响面 |
|---|---|---|---|
| thinking 模式 effort 前缀 | 每条 thinking 请求都有: `<｜System｜>Reasoning Effort: N (range 1-100, the higher the value, the more thorough the reasoning)\n\n`, N 默认 high=75, max=100(low=50) | HIGH 档**什么都不加**; MAX 档加一段 V4 时代的英文长段落(`core_globals.c:10 DS4_REASONING_EFFORT_MAX_PREFIX`), 且无 `<｜System｜>` | 所有 thinking 请求(agent/REPL/服务 reasoning_effort) |
| 会话中途的 system 消息 | 原位渲染 `<｜System｜>mid`, 并像 user 一样触发 `<｜Assistant｜>` 头 | 全部 system 消息合并进开头(server_dsml_render.c:363-368) | agent compact(`agent_compact.c:227` 把摘要当 system 追加) |
| 工具块 | system 内容 + `\n\n## Tools…`(工具在后); 措辞 "help answer the user's question" | 工具块在 system 内容**之前**(为了 `--kv-cache-boundary-trim-tokens` 裁剪, 有单测 `test_render_chat_prompt_text_renders_tools_before_system` 钉住); 措辞多两句(`server_knowledge.c:105`) | 带 tools 的请求(agent); qtf 不带 tools, 与本次无关 |

产品路(qtf, 非思考, 无 tools)只中第一处(1.2)。

## 2 写死参数逐项对账(数值路)

读法: "来源"列写引擎从哪拿这个数; "判"列 ✓ = 与官方相同。

| 参数 | 官方(config.json / model.py) | 引擎 | 来源 | 判 |
|---|---|---|---|---|
| sliding_window | 128 | DS4_N_SWA 128 | GGUF `attention.sliding_window` + shape 默认 | ✓ |
| index_topk | 512 | DS4_N_INDEXER_TOP_K 512 | GGUF `attention.indexer.top_k` | ✓ |
| index_n_heads / head_dim | 32 / 128 | 32 / 128 | GGUF | ✓ |
| candidate: source_layer / topk_blocks / block_size | 20 / 2048 / 8 | 同 | GGUF `attention.candidate.*` | ✓ |
| compress_ratios(40 层 + 3 塔) | [0,0,2×18,1×20,0,0,0] | 同表 | GGUF `attention.compress_ratios` | ✓ |
| kv_source / index_source | [2,8,14,20] / [2,8,14,20,24,28,32,36] | 同 | GGUF | ✓ |
| CED 分界层 | 解码器全局 KV 读最后一个源层 L20 | ced_edge = 最后一个 is_kv_source = 20 | 元数据推导 | ✓ |
| rope_theta / compress_rope_theta | 10000 / 160000 | 窗口层 10000 无 YaRN; 压缩层 160000 + YaRN | GGUF + core_v41_attn.c:11 | ✓(与 model.py Attention.__init__ 同支) |
| YaRN factor / beta_fast / beta_slow / original | 16 / 32 / 1 / 65536 | 同 | GGUF `rope.scaling.*`, core_validate.c 逐项核 | ✓ |
| YaRN 公式 | corrected_dim + 线性 ramp, **无 mscale** | cuda_v41_2.inc.cu:8-18 同式, 无 mscale | — | ✓ |
| softmax_scale(注意力 / indexer) | head_dim^-0.5 / index_head_dim^-0.5·n_heads^-0.5 | 1/√512 / 1/√128/√32 | core_v41_attn.c:136,276 | ✓ |
| attn_sink | 只进分母 exp(sink−max) | "sink 只进分母" | ds4_gpu_v41.h | ✓ |
| 激活量化 | 窗口 KV fp8/32; 压缩 KV fp4 e4m3/16; 索引键与索引 q fp4 ue8m0/32 | 同(act_quant_fp8 32; ckv_pack e4m3/16; idxk_pack ue8m0/32; iq fp4 32 e4m3=false) | core_v41_attn.c:255,110-114,132 | ✓ |
| 索引分数 | bf16(Σ_h bf16(relu(bf16(q·k))·w)) | 同舍点 | cuda_v41_indexer.inc.cu:41-50 | ✓ |
| 候选块 | 块内 amax; 含最新位置的块钉 +inf; 不可达 −inf 不选 | 同 | cuda_v41_indexer.inc.cu:116-125 | ✓ |
| 候选块 kk<nb 那一支(16384 位置之后才走) | topk 精确 | 4 轮 8 位 radix select = 精确; 并列取小块号 | 同文件 126-153 | ✓(读码; 但**没有任何尺跑过这一支**, 见 §3.4) |
| topk 并列规则 | torch.topk 未定义 | 取小下标 | — | 不可比(教师同样任意) |
| 压缩器 | 组内逐维 softmax 池化, 尾巴留到下块; 组位置 = g·ratio | 同; posg = (g0+g)·ratio | core_v41_attn.c:60-100 | ✓ |
| hc sinkhorn / eps | 20 / 1e-6 | 20 / 1e-6 | GGUF `hyper_connection.*` | ✓ |
| norm_eps | 1e-20 | 1e-20 | GGUF + shape | ✓ |
| route_scale / sqrtsoftplus / noaux_tc / norm_topk | 1.5 / 是 / 是 / 是 | 同 | GGUF | ✓ |
| swiglu_limit | 10.0 | 10.0 逐层 | GGUF | ✓ |
| EOS / BOS | 1 / 0 | 1 / 0(按字符串查) | tokenizer | ✓ |
| 非思考 assistant 头 | `<｜Assistant｜></think>` | 同 | server_dsml_render.c:421 | ✓ |
| 历史 assistant 轮 | `<｜Assistant｜></think>{content}<｜end▁of▁sentence｜>`; 带 tools 保留 reasoning | 同 | server_dsml_render.c:397-414 | ✓ |
| tool 结果 | `<tool_result>{content}</tool_result>` 并进 user | 同 | server_dsml_render.c:390-394 | ✓ |
| 采样 | temperature 0 = argmax; 默认 1.0 纯采样(无 top-p); 官方评测 1.0 / top-p 0.95 | 温 0 argmax(设备核, 与主机顺扫自检); 采样默认关 | model.py:1285 / core_v41_api.c | ✓ |

引擎独有、官方没有对应物的量(不是"对不上", 是"官方没这个"):

| 量 | 值 | 作用 | 数值影响 |
|---|---|---|---|
| DS4_V41_CHUNK | 512 | 预填分块; 也决定 CED 回放长度 = np mod 512 | 见 §3.1 |
| DGRAPH_BUCKET | 1024 | 解码整步 graph 每 1024 位置重捕获, ng/topk 上限按桶算, 核里按设备位置取真值 | 逐字节门(64 步)过 |
| DS4_V41_GEMV_MAX_TOK | 8 | ≤8 token 走融合 GEMV/VQ 核, 否则 GEMM | 09-20 解码路 vs 预填路 PPL 差 0.010(噪声地板) |
| DS4_V41_MAX_CTX_P2C | 524288 | 状态缓冲上限 | 只影响内存 |
| `--use_fast_math`(Makefile:93) | 全局 | 见 §3.3 | 见 §3.3 |

## 3 生成路独有的偏差(判决路碰不到, 所以五指标看不见)

### 3.1 CED 解码器回放长度 = 提示长 mod 512, 官方固定 n_win = 128
官方报告 §2.2/§3.2.2: "Decoder SWA Bounded Replay, which only prefills the last n_win tokens of the prompt for the SWA computation"。引擎(`core_v41_api.c:238-242`)把提示按 512 切块, 除最后一块外只跑 20 层编码器 + L20 的全局 KV 投影, 最后一块跑满 40 层 ⇒ 解码器 20 层的窗口 KV 只有最后 (np mod 512) ∈ [1, 512] 个位置。
本次两条提示 13710 / 14102 ⇒ 398 / 278, 比官方的 128 **多**回放(更接近精确), 这两条上不是它的锅; 提示长 mod 512 < 128 的请求回放不足官方, 首批生成 token 的窗口直接缺位。

### 3.2 从没写过的窗口环槽不屏蔽, 照读
官方 `get_window_topk_idxs`: "-1 marks a slot holding nothing"(环没填满的槽标 -1, sparse_attn 里 -1 ⇒ −inf)。引擎注意力核按绝对位置取 [p−127, p](`cuda_v41_attn_split.inc.cu:66`, `cuda_v41_attn_mma_decode.inc.cu:54`), 环行号 = a mod 128, **没有"这一槽写没写过"的概念**; 而 CED 让 L20~L39 在中间块里从不写环 ⇒ 最后一块的前 127 个位置在这 20 层里读到的历史键 = `cudaMalloc` 回来的旧内容(`cuda_lifecycle.inc.cu:119` 无 memset; 状态每请求新建, 大概率就是上一条请求同层同槽的 KV)。
本次两条提示: 污染最后一块前 127 位(位置 13312~13438 / 13824~13950), 生成 token 的窗口不直接碰到(np mod 512 ≥ 127), 二阶影响未量。mod 512 < 127 的提示: 生成 token 直接吃脏槽。

### 3.3 `--use_fast_math` 全局
`Makefile:93 NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math` ⇒ sinf/cosf/expf/logf/powf/除法/开方全部换成 SFU 近似, 且 denormal 清零。RoPE 核(`cuda_v41_2.inc.cu:29-30`)算 `ang = pos × freq; cosf(ang), sinf(ang)`: 快速 sin/cos 的范围归约在 fp32 做 frac(x/2π), 位置 p 的角度误差 ≈ 2^-11 圈 × 高频维 ⇒ 位置 30000 处约 3e-3 rad; 官方是 torch fp32 `outer` 再 `polar`(自身舍入 ≈ 1e-3 rad)。同量级、图样不同, 单点不致命, 但它随位置线性涨、遍布每个核, 是引擎 vs Python 学生 **KLD 0.013 / top-1 2.34% 不一致**(wt2 512, 09-21)的候选来源之一。
在 10k token 的贪心轨迹上, 2.34% 的 top-1 不一致 ≈ 230 个位置引擎选的与权重本身"想选"的不同 —— 每一个都是潜在的分岔点。

### 3.4 从没被尺覆盖的区间
| 尺 | 覆盖 | 产品运行区间 |
|---|---|---|
| 金融 j / wt2 五指标 | 8192 窗, `--score-ids` 预填路(每块 40 层, 无 CED) | 位置 13.7k~30k, CED 生成路 |
| 引擎 vs Python 学生 | wt2 512 | 同上 |
| 直发==走图==复跑(d1_kv_ring_gate) | 2K/12k 各 64 步 | 走图几千步 |
| 解码路 vs 预填路(09-20) | 1.4k 提示 | 同上 |
| 候选块 kk<nb 支 | 无(nb > 2048 要 ng > 16384) | 16384 之后每一步 |
| CED vs `--decoder-full` | 2K 意大利文冒烟(文本逐字同) | 13.7k 提示 |

## 4 修法(改码清单, 按官方字节对齐; 不写码)

### 4.1 `<｜System｜>` 与 effort 前缀(第一刀, 产品路)

| 文件 | 改什么 | 为什么这样 |
|---|---|---|
| `src/core/core_types.h` ds4_vocab | 加 `int system_id`(没有 = −1) | 版本差异按 tokenizer 内容判(铁律: 不写版本名), V4 的 tokenizer 没这个 token |
| `src/core/core_bpe.c` vocab 装载 | `system_id = 可选查找("<｜System｜>")`(现有 `vocab_lookup` 缺 token 直接 exit, 要加一个不退出的查找); 特殊 token 表 `special_token_at` 加 `{"<｜System｜>", system_id}`, id<0 的项跳过 | 渲染出来的 token 文本必须被切成单个 id 128799, 否则等于没改 |
| 同文件 `encode_chat_prompt`(CLI `--system`) / `ds4_chat_append_message("system")`(agent/REPL) | BOS 之后: 有 system 或 thinking 开 ⇒ 先推 system_id(≥0 时), 再 effort 前缀, 再 system 文本; 中途 system 消息 = system_id + 文本 | 与官方 render_message 同序 |
| `src/core/core_globals.c` / `core_internal.h` / `core_engine_api.c` / `ds4.h` | 删 V4 长段落 `DS4_REASONING_EFFORT_MAX_PREFIX`, 换官方模板 `"Reasoning Effort: %d (range 1-100, the higher the value, the more thorough the reasoning)\n\n"`: HIGH=75, MAX=100; `ds4_think_max_prefix()` 保留(返回 100 那条), 新增 `ds4_think_effort_prefix(mode)`; 新增 `ds4_chat_system_token()`(vocab 有 ⇒ "<｜System｜>", 否则 "") 与单测用的设置入口 | 一件事只写一处: 模板字符串只在 core 一份, 服务端/CLI/agent 都引它 |
| `src/server/server_dsml_render.c:374-377` | `BOS; effort = ds4_think_effort_prefix(think_mode); if (effort[0] \|\| system.len) 写 ds4_chat_system_token(); 写 effort; 写 system` | 与 §1.1 四行逐字节同 |
| `src/server/server_parse_completion.c:186-192` | 同一套(注入的 "You are a helpful assistant" 前加 token) | 同上 |
| `src/server/server_gen_log.c:89-100` rendered_chat_system_region | 跳过 BOS 后的 `<｜System｜>` 与新旧 effort 前缀 | 工具报错回填的 "System prompt reminder" 别把 token 抄进去 |
| `tests/server_tests_render.c`(+ `_run.c`/`_internal.h` 登记) | 改 `test_render_think_max_prompt_prefix` 的期望; **新增金标用例**: 把 §1.1 的四行字符串原样写进断言(chat+system / chat 无 system / thinking+system / 多轮+中途 system) | 官方 encoding.py 就是金标, 以后模板再漂一个字节单测就红 |
| `README.md` Runtime/服务一节 | 一句话: V4.1 提示按官方 encoding.py 渲染, 金标在单测 | 读者知道去哪核 |

不在这一刀里: 工具块的位置(引擎故意放前面, 有 KV 裁剪的理由与单测钉住, 换顺序要用户定)与措辞; 会话中途 system 的"触发 assistant 头"语义(agent 路, 另立)。

### 4.2 门(改完必须全过)
1. Mac `make test` 全绿(server 单测离线, 含新金标用例; linecount)。
2. spark `make cuda-spark` + `serve_1m_spark.sh smoke`(温 0 48 token 中文金融问答, 原样贴回答)。
3. **温 0 逐字节回归只对"无 system、非 thinking"的请求成立**(那条渲染没变): `d1_kv_ring_gate.sh` 2K 64 步与旧二进制逐字节同; 带 system 的请求**本来就该变**, 不能拿逐字节门。
4. 渲染字节 = 官方: 拿 qtf 真实请求的 messages JSON 喂官方 `encode_messages(thinking_mode="chat")`, 与服务端 `--trace` 里的 rendered prompt 逐字节 cmp(这是产品路的终门, 不是单测能替的)。

## 5 验证死循环(顺序即优先级; 都是分钟级, 只有 5.4 的教师锚要几十分钟)

| 步 | 做什么 | 判 |
|---|---|---|
| 5.1 成对重跑 | 同一二进制、同两只股票(002490 / 601975 @20260903)、温 0、grrb 对: 旧渲染(现役服务已有读数: 002490 16384 顶格 / 601975 6195 正常) vs 新渲染, `qtf_replay_probe.sh` 三率 + 数字齐 | **已跑(09-21 22:29~22:55, tag sysfix1): 002490 仍 16384 顶格; 601975 从 6195 正常翻成 16384 顶格。两只都把 JSON 写全了(53%/63% 处), 然后不停, 继续编"九、完整交易计划"直到逐字复读。⇒ 模板无罪于此, 进 5.2** |
| 5.2 fast-math A/B | 去掉 `--use_fast_math` 重编一份(只改 NVCCFLAGS), `v41_engine_parity_spark.sh wt2 512` 引擎 vs Python 学生 KLD/Same top; 再 d1 2K 64 步看速度 | KLD 0.013 → ≤ 0.003 且 top-1 ≥ 99% ⇒ 快速数学是引擎自己的主噪声, 默认改精确(速度账另量); 没变化 ⇒ 排除, 恢复 |
| 5.3 CED 两偏差 | 按 §6 修后 `--decoder-full` vs 默认在 002490 提示上末位 logits KL + 首 64 token 同轨 | KL ≤ 0.01 ⇒ CED 对这类提示无损; 否则 CED 本身要重新设计回放 |
| 5.4 三方对拍(fable5 09-21 夜 E0/E1) | 服务 trace 落 token id; 002490 失败序列: 解码路 vs `--score-ids` / 引擎预填 vs Python 学生(入口附近 512 位) / FP 教师入口 argmax | 5.1~5.3 全过仍循环时才走; 结论三分: 引擎分歧修核 / 结构 token 量化损失 / FP 也抄 = 温 0 固有 |

## 6 CED 与 fast-math 的修法(排 §4 之后)

| 项 | 改什么 | 门 |
|---|---|---|
| 回放长度 | 最后一块不足 128 位时把上一块尾巴并进来(最后一块最短 128, 最长 639), 或者按官方固定 128: 中间块全 CED, 再补一块 128 跑满 | `--decoder-full` 输出不变(它不走 CED); 提示长 mod 512 ∈ {1,127,128,511} 四条扫描, 首 64 token 逐字节同 |
| 环槽屏蔽 | 状态里记每层"环里最早有效位置"(CED 跳过的块不推进它), 注意力核窗口下界取 max(p−127, 最早有效位置); 新建状态 `cudaMemset` 环缓冲(防读旧请求) | 同上 + 一条 mod 512 = 1 的提示前后对比 |
| fast-math | NVCCFLAGS 去 `--use_fast_math`, 只保留 `-fmad=true`; 需要快的核(专家 VQ 即乘、GEMV)里显式用 `__expf` 等内建 | §5.2 的 KLD 门 + 12k 稳态 ms/步 ≤ 基线 + 0.5 |

## 7 不做 / 不改

- DRY / 频率 / 出现惩罚不进任何默认与产品口径(09-21 夜实测: 真实请求上照样 16384 顶格且把数字写坏; fable5 同日夜)。
- 不拿单提示的"约？"或复读次数当判决; 只认 qtf replay 固定股票集的三率。
- 不加 env; 不写版本名; 模板改动只在 core 一处。
- 工具块顺序与措辞、agent 的中途 system 语义: 记账, 等用户定。

## 8 依据

- 官方: `hf/DeepSeek-V4.1-Flash/encoding/encoding.py`(render_message 561/564/674-682; REASONING_EFFORT_TEMPLATE 393-403; TOOLS_TEMPLATE 431), `encoding/README.md` 25/136/152-160, `inference/model.py`(ModelArgs 45; get_window_topk_idxs 410; Compressor 429; Indexer 488; select_candidate_blocks 583; Attention 613; sample 1285), `inference/kernel.py`(sparse_attn 311; fp4_act_quant 184), `config.json`, `DeepSeek_V41_Tech_Report.pdf` §2.2 / §3.2.2(Decoder SWA Bounded Replay = 最后 n_win 个 token)。
- 引擎: `src/server/server_dsml_render.c` 345-427, `src/core/core_bpe.c` 322-464, `src/core/core_globals.c` 10, `src/server/server_parse_completion.c` 186-192, `src/server/server_knowledge.c` 105, `src/core/core_v41_api.c` 196-245(分块与 CED), `src/core/core_v41_forward.c` 337-349, `src/core/core_v41_attn.c`, `src/cuda/cuda_v41_indexer.inc.cu`, `src/cuda/cuda_v41_attn_split.inc.cu` 66, `src/cuda/cuda_v41_2.inc.cu` 8-30, `src/cuda/cuda_lifecycle.inc.cu` 119, `Makefile` 93, `gguf-tools/quantize/v41_to_gguf.c` 395-435, `src/core/core_shape_select.c` 100-115, `src/core/core_validate.c` 198-203。
- 真实请求 trace: spark `/tmp/ds4-trace-real-v3.txt`(17:09 v3 裸) / `-v3dry.txt`(17:44 DRY 0.8) / `-grrb.txt`(20:53 grrb); serve 日志 `/tmp/serve_real_*.log`。

## 9 进度(2026-09-21 23:05, 用户令"开始改"之后)

| 项 | 状态 |
|---|---|
| §4.1 改码清单 | **全部落地**(core_types/core_bpe/新文件 core_chat_frame.c/core_globals/core_internal/core_engine_api/ds4.h/server_dsml_render/server_parse_completion/server_gen_log/cli_repl+cli_internal/agent_sysprompt/tests server_tests_render_v41.c/README); 声明"不在这一刀"的两项(工具块顺序与措辞、中途 system 原位语义)按计划没动, 等用户定 |
| §4.2 门 1 Mac make test | 过(含 linecount; core_bpe.c 拆出角色帧后 431 行, ds4.h 499 行) |
| §4.2 门 2 spark 重编 + 冒烟 | 过(22:25 编, 22:28 服务; 23:01 又部署了带 E0 的版本) |
| §4.2 门 3 无 system 请求逐字节回归 | 过: 同一对(grrb)冒烟输出, 旧二进制 21:22 服务 vs 新二进制 22:28 / 23:01, md5 三者相同 |
| §4.2 门 4 真实请求渲染 = 官方 | 过: qtf 002490 请求的 messages 喂官方 encode_messages(chat) vs 服务 trace 的 rendered prompt, 22800 字节逐字节同 |
| §5.1 成对重跑 | 跑了: 两只都 16384 顶格(601975 从 6195 正常翻成顶格), JSON 都写全然后不停 ⇒ 模板无罪于此 |
| §5.2 fast-math A/B | **跑了**: wt2 512 引擎 vs 文件态学生, --use_fast_math KLD 0.01319 / 97.66% vs 去 fast-math KLD 0.01480 / 96.09% ⇒ 去掉反而更远, fast-math 无罪保留 |
| §5.3 CED 修后对拍 | **跑了**: 002490 真实提示 256 步对精确路(--decoder-full): 修前 CED 第 33 位分叉, 修后第 174 位分叉; 短尾提示(mod 512 = 63) 修前 52 位 / 修后 148 位分叉 ⇒ 脏环槽是真失真, 修复有效; G1 精确路逐字节不变 |
| §5.4 三方对拍 | **全部跑了**(601975 cedfix1 序列 24400 位, fable5 09-22 02:15 节): (a) 精确预填路 vs 解码路 生成段 96.5~99.5%/千位; (d) FP 教师: 入口 23096 "M2评分10…" FP argmax 同为 "M", 复读区逐位 KLD 0.000, 10297 位里 FP 从没想 EOS ⇒ 复读是 FP 自己的贪心; 入口前的分歧全是 KLD 0.3~1.2 的近平局; (c) 文件态学生 跑了: 引擎 vs 学生 KLD 0.0294 / 生成段 same top 98~100%/千位, 学生对 FP 比引擎还差一点(PPL 比 1.087 vs 1.073) ⇒ 引擎长位置无病 |
| §6 CED 两偏差 / fast-math | CED 两处已落地(win_from 钳位 + 环清零 + 最后一块 ≥128); fast-math 按 §5.2 读数保留 |
| 提交 | 未提交(commit 要验证 + 用户批; 验证已过) |
