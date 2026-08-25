# Task 01 — GPU route_translate（让 k4 输出正确）

**状态**：⚠️ 部分修复 · **运行验证未通过（k4 输出仍是垃圾）**。代码 `make` 全绿；运行验证（2026-05-30，用户确认后跑）暴露并修了一个真实 bug，但 k4 输出仍不正确。

## ⚠️ 运行验证结果（2026-05-30）：输出仍垃圾，route_translate 未真正修好

实跑 `./ds4 -m gguf/ds4flash-k4.gguf -p "1+1=?"` 生成的是 `packfully` + 空白/换行，**不是连贯文本**。

**验证中发现并修复的真实 bug（n_total_expert 越界）**：
- `routed_moe` 内 `gate_tensor_bytes = n_total_expert × gate_expert_bytes`（`ds4_metal.m:14215`），用来 wrap 专家张量。
- host 两调用点（decode `ds4.c:10855` / prefill `ds4.c:13713`）传的是 `DS4_N_EXPERT`（=256），但 k4 张量实际只 4 专家 → wrap 范围比真张量大 64×。
- **全模型路径下此 bug 不暴露**（整 tensor-data 是一个连续 view，越界 wrap 仍落在大 view 里；GPU 按 route_translate 限定的 id∈[0,4) 索引不真读越界）。但在 A1 split 多 view 后暴露成 `not covered by mapped model views`。
- **修复**：两处 `DS4_N_EXPERT` → `model_expert_kept_count(model, il)`。全模型零回归（无 keep-map 时该函数返回 256，`ds4.c:1357`）。

**但修完 wrap 范围后输出仍垃圾** → route_translate / 专家计算链路里**还有别的错**（LUT 翻译时机、weights 归一化、或 keep-map 与 hash 路由前 3 层的交互）。Task 01 的核心目标（k4 输出连贯）**尚未达成**。

**下一步**（恢复时）：在正确 wrap 范围基础上，逐层 dump（`--dump-logprobs` / `metal_graph_debug_dump_*`）对比 k4 vs 全模型的 router_selected / routed_out，定位翻译后专家计算为何仍错。

---

### 旧状态行（存档）：🟡 代码完成 + make 全绿，仅剩运行验证

## ✅ 警告已清

之前 GPU 路径改用 GPU 内 LUT 翻译后，host 端 `model_expert_compact_slot()`（`ds4.c:1365`）成为死代码 → unused-function 警告。**已按用户决定删除该函数**，重新 `make` 确认 0 warning（`expert_orig_to_compact` 表仍由 `load_expert_keep_map` 填，将来若需 CPU 参考翻译可重建）。
（`ds4.c:5374` clang-tidy integer-division 是会话前已存在的未提交改动，非本 task 引入。）

## ✅ 实现完成（Metal + ds4.c 全部落地，编译通过）

| 处 | 文件:行 | 内容 |
|---|---|---|
| kernel | `metal/dsv4_misc.metal:231/238` | `struct ds4_metal_args_dsv4_route_translate` + `kernel_dsv4_route_translate`（orig id→compact slot，丢弃/越界→slot0；一维 grid = n_tokens×n_expert_used） |
| 声明 | `ds4_gpu.h:616` | `ds4_gpu_set_expert_keep_lut` + `ds4_gpu_translate_expert_ids` |
| 全局 | `ds4_metal.m:103/127/128` | pipeline + 常驻 LUT buffer + layers |
| init/teardown | `ds4_metal.m:4178/4190/4559` | 获取 pipeline + 校验链 + 释放清零 |
| 函数体 | `ds4_metal.m`（`:13807` 前） | `set_expert_keep_lut`（newBufferWithBytes 建 21KB Shared 常驻 LUT）+ `translate_expert_ids`（无 LUT→no-op；否则 CB→encoder→dispatchThreads→finish） |
| set LUT | `ds4.c:1617`（`#ifndef DS4_NO_GPU` 守卫） | model_open 内 load_expert_keep_map 后，`expert_shrunken` 则上传 `expert_orig_to_compact` |
| decode | `ds4.c:10746` | router_select 后、routed_moe_one 前：`translate(g->router_selected, il, USED, 1, kept_count)` |
| prefill | `ds4.c:13595` | router_select 后、routed_moe_batch 前：`translate(g->batch_router_selected, il, USED, n_tokens, kept_count)` |

**架构确认（python 预处理器分析）**：4 个 graph 调用点都在 `#ifndef DS4_NO_GPU` 大块内 → CPU build 不编译 → **无需 CPU stub**；只有 set-LUT 在 model_open（全 build 编译）所以加了 `#ifndef DS4_NO_GPU` 守卫。`grep -c ds4_gpu_router_select_tensor ds4.c` = 1（仅调用点，无 CPU 同名定义，证实 CPU 走独立路径）。

**正确性保证**：translate 在 router_select 之后（weights 已用原始 id 算完，见 `kernel_dsv4_router_weights_one`）；前 3 层 hash 与 topk 同写一个 selected，统一翻译覆盖；全模型 LUT 未设 → 函数内 no-op + 调用点 `expert_shrunken` 守卫双保险，零运行时开销。

## ⏳ 唯一剩余：运行验证（待用户确认）

```sh
./ds4 -m gguf/ds4flash-k4.gguf --ctx 2048 --temp 0 -n 32 -p "1+1=?"   # 期望: 连贯文本, 不再重复 BOS
```
过 13GB phys_footprint 看门狗（参考 Stage 1a 兜底）。通过后 #1 → ✅，解锁 #03 的 ds4-eval 质量 / 多 token 速度测量。

---

（以下为原始 task 设计 + 中途落地记录，存档对照）

### 旧状态行（存档）：🔄 Metal 侧全部完成；只剩 ds4.c 3 处 wiring

## ⚠️ 当前代码状态（半成品，可恢复，2026-05-30）

### ✅ 已落地（Edit 精确唯一匹配，可信）
1. `metal/dsv4_misc.metal:231/238`：`struct ds4_metal_args_dsv4_route_translate` + `kernel void kernel_dsv4_route_translate`。原始 id→compact slot，越界/丢弃→slot 0；grid 一维 = n_tokens×n_expert_used。
2. `ds4_gpu.h:616`：`int ds4_gpu_set_expert_keep_lut(const int16_t *lut, uint32_t n_layer);` +
   `int ds4_gpu_translate_expert_ids(ds4_gpu_tensor *selected, uint32_t layer, uint32_t n_expert_used, uint32_t n_tokens, uint32_t n_total_expert);`
3. `ds4_metal.m:103/127/128`：`g_dsv4_route_translate_pipeline` + `g_expert_keep_lut_buffer` + `g_expert_keep_lut_layers`。
4. `ds4_metal.m:4178`：init 获取 pipeline + 加入失败校验链（`:4190`）。
5. `ds4_metal.m:4559`：teardown pipeline=nil；buffer/layers 清零（`g_router_weight_sum_buffer = nil` 后）。
6. `ds4_metal.m`（`ds4_gpu_router_select_tensor` 前，`:13807` 上方）：**两个函数体已实现**——
   `ds4_gpu_set_expert_keep_lut`（`newBufferWithBytes` 建 Shared 常驻 LUT，存 n_layer×256 int16，替换旧 LUT）；
   `ds4_gpu_translate_expert_ids`（无 LUT→`return 1` no-op；越界 layer 报错；否则 `ds4_gpu_command_buffer`→`compute_encoder`→setBytes(args,0)/setBuffer(lut,1)/setBuffer(selected,2)→`dispatchThreads(total)`→`end_compute_encoder`→`finish_command_buffer`）。用的 helper 全部已验证存在（`ds4_gpu_tensor_offset:743`、`_bytes:738`、`_hot_pipeline`、`command_buffer:249`、`finish_command_buffer:297`）。

### ⏳ 未做（仅剩 ds4.c 3 处 + 验证）
7. `ds4.c`：`load_expert_keep_map(m)` 后（`ds4.c:1614`，model_open 内 `return m` 前）若 `m->expert_shrunken`，
   把 `m->expert_orig_to_compact`（int16，正好 `expert_layer_count*256`）经 `ds4_gpu_set_expert_keep_lut(m->expert_orig_to_compact, m->expert_layer_count)` 上传。
   - **✅ 已查清 GPU 守卫方式**：ds4.c 全文只有 3 行提 `DS4_NO_GPU`（`:20-22`，仅 backend 名字符串），
     **调用点无 ifdef**。即 `ds4_gpu_*` 符号在两种 build 都存在：GPU build 来自 `ds4_metal.m`/`ds4_cuda.cu`，
     CPU build 来自 **ds4.c 内的 CPU 参考实现**（CLAUDE.md 证实）。
   - **⚠️ 因此新增的 #6（CPU 参考实现）**：必须在 ds4.c 的 CPU 参考段（`#ifdef DS4_NO_GPU` 那套 `ds4_gpu_*` stub
     所在处）补 `ds4_gpu_set_expert_keep_lut` + `ds4_gpu_translate_expert_ids` 的 CPU 版，**否则 CPU build 链接缺符号**。
     - `set_expert_keep_lut`：CPU 版可只存指针/或 no-op（CPU 路由翻译由现有 `model_expert_compact_slot` 在 host 算）。
     - `translate_expert_ids`：查 CPU build 里 routed_moe 参考实现是否已用 `model_expert_compact_slot` 翻译——
       若已翻译则此函数 CPU 版 = no-op 返回 1；若没翻译则需在 CPU 版里就地翻译 selected。**动手前先读 ds4.c CPU 参考段确认。**
   - 落点：找 ds4.c 里 `ds4_gpu_router_select_tensor` 的**定义**（不是声明）所在的 CPU 参考块。本轮 grep
     被工具故障打断（`grep -c` 都返回空），未定位到行号——恢复后第一件事就是 grep 这个定义。
8. `ds4.c` decode `metal_graph_ffn_layer`（`:10739`，router_select 后、`ds4_gpu_routed_moe_one_tensor` 前）插
   `if (ok) ok = ds4_gpu_translate_expert_ids(g->router_selected, il, DS4_N_EXPERT_USED, 1, model_expert_kept_count(model, il)) != 0;`
9. `ds4.c` prefill（`:13585`，router_select 后、`ds4_gpu_routed_moe_batch_tensor` 前）插
   `ok = ds4_gpu_translate_expert_ids(g->batch_router_selected, il, DS4_N_EXPERT_USED, n_tokens, model_expert_kept_count(model, il)) != 0;`
   （全模型 LUT 未设 → 函数内 no-op，双保险，无需额外 expert_shrunken 守卫，但加上更省一次 dispatch）
10. `make` 编译验证（compiler = ground truth）。
11. 运行验证（需用户确认 + 13GB 看门狗，遵守铁律）。

### ⚠️ 当前树编译状态
Metal 侧三处函数（set/translate/kernel）现在**互相闭合**（声明有定义、全局有用、pipeline 有获取/释放），
Metal 侧单独应能编译干净。但 `ds4_gpu.h` 的两个声明在 **ds4.c 编译单元**里尚无调用——`-Wunused` 不针对声明，
故大概率不影响 build；真正的未完成是功能（k4 仍不会翻译 id → 仍输出垃圾）。**完成判定 = #7-#11 全过 + k4 输出连贯。**

### 正确性已核验的关键点
- 翻译必须在 `router_select` **之后**：weights 在 router_select 内部用**原始 id** 索引 256-wide probs 算完才返回（`kernel_dsv4_router_weights_one` `metal/dsv4_misc.metal`），之后翻译 `selected` 不影响 weights，routed_moe 只用 `selected` 索引 compact 专家张量——顺序安全。
- 前 3 层 hash 路由把原始 id 直接写进 `selected`（`kernel_dsv4_router_finalize_one` hash 分支），与 topk 同写一个 `selected`，统一翻译一并覆盖。

---

（以下为原始 task 设计，落地时对照）


## 进展 (2026-05-30)（已确认的代码事实，恢复后可直接据此实现）

- 路由选择 `selected`（int32，原始 id 0..255）由 `ds4_gpu_router_select_tensor`（`ds4_metal.m:13798`）/
  `_batch_tensor`（`:13893`）产出，写入 scratch `g_router_selection_buffer`（`:122`）。
- 专家 matvec 消费方两处，**均已带 `layer_index` 形参**（可拿到层号取 per-layer LUT）：
  - `ds4_gpu_routed_moe_one_tensor`（`ds4_metal.m:13989`，签名末 `n_total_expert, n_expert, layer_index`）
  - `ds4_gpu_routed_moe_batch_tensor`（`ds4_metal.m:14296`，同上；错误日志在 `:14487` 已用 layer_index）
- 它们用 `_id_` kernel（`g_moe_mul_mv_id_*` `:72-80`）按 `selected` 的 id 索引专家张量
  （`n_total_expert` 行；k4 时=4）→ 原始 id 越界根因。
- **实现锚点（确认）**：在这两个函数取出 `selected` 之后、构造 `mul_mv_id` 派发之前，插入一次
  `kernel_dsv4_route_translate`（新）把 `selected` **原地**改写为 compact slot；gate/up/down 复用同一
  `selected`（`:14524` 注释佐证），一次翻译即覆盖全部。
- **LUT 源**：host 侧 `m->expert_orig_to_compact`（`ds4.c:1159`，`[layer*256+orig] → slot 或 -1`），
  `load_expert_keep_map`（`ds4.c:1377`）填充；新增 `ds4_gpu_set_expert_keep_lut(...)` 在 `model_open`
  （`ds4.c:1616` 之后）上传 per-layer LUT；kernel 里 `slot = lut[layer*256+orig]; if(slot<0) slot=0`。
- kernel 模板参考 `kernel_dsv4_router_weights_one` / `_finalize_one`（`metal/dsv4_misc.metal:105/115`，
  逐 gid 处理 `selected`/`weights` 的一维 tiny dispatch）。
- 前 3 层 hash 路由的 `ffn_gate_tid2eid.weight[6,vocab]`（原始 id）同样需经 LUT 翻译后索引——待实现时核对。

**阻塞**：工具 Read 对 `ds4_metal.m` 大函数体返回乱序行号/拼接内容（详见 `notes/execution-log.md`
2026-05-30 blocker 条目）。需在工具输出稳定的会话/轮次里逐处「即时重读 → 立即 Edit → 任何异常即停」地落地。
**前置依赖**：Stage 1a 已完成（reduced-expert loader 已移植、k4 可加载、内存安全 9.3GB 实证）
**类型**：单机 · GPU 代码

## 目标

补齐 GPU 侧专家路由 id 翻译，让 k4（每层只留专家 0..3）输出正确文本，解除「输出垃圾（重复 BOS）」。
这是后续一切有意义的质量/多 token 速度测量的前置。

## 问题根因（Stage 1a 实测）

- 专家 matvec 用 `_id_` kernel（`g_moe_mul_mv_id_*`）按 router 选出的**原始 id（0..255）**索引专家张量。
- k4 专家张量 `dim[2]=4`，用原始 id 索引 4-专家张量 → GPU 越界（被驱动沙箱化，不崩不爆内存，但输出垃圾）。
- host 侧已有映射：`expert_orig_to_compact[]`（`ds4.c:1159`）+ `model_expert_compact_slot()`（`ds4.c:1365`）+ `load_expert_keep_map()`（`ds4.c:1377`，由 `model_open` 在 `ds4.c:1616` 调用），但**未下发到 GPU**。

## 实施步骤

1. **metal kernel**：`metal/dsv4_misc.metal` 加 `kernel_dsv4_route_translate`——读 router 选择缓冲（原始 id），按 LUT 映射成 compact slot；被丢弃专家（不在 keep-map）→ slot 0（安全兜底，权重轻误差可接受，smoke 阶段质量非重点）。
2. **GPU 接口**：`ds4_gpu.h` + `ds4_metal.m` 加 `ds4_gpu_set_expert_keep_lut(layer_or_global, lut, n)`；LUT 为 256→compact 的 i32 表。
   - k4-first-K 的 LUT **逐层相同** → 单层（全局）LUT 即可，省内存。
3. **wiring**：在 router top-k 选择之后、专家 matvec（`g_moe_mul_mv_id_*`）之前插入 route_translate；把翻译后的 slot id 喂给 matvec。
4. **load 时 set**：`model_open` 加载 keep-map 后调用 `ds4_gpu_set_expert_keep_lut` 下发 LUT。
5. **前 3 层 hash 路由**：注意前 3 层另有 `ffn_gate_tid2eid.weight[6,vocab]`（原始 id），同样需经 LUT 翻译后再索引。

## 闸门 / 验收

- `make`（Metal）全绿。
- `./ds4 -m gguf/ds4flash-k4.gguf --ctx 2048 --temp 0 -n 32 -p "1+1=?"` 输出**连贯文本**（不再重复 BOS）。
- `--dump-tokens` / `--dump-logprobs` 抽查：router 选出的 slot ∈ [0,4)，无越界。
- 内存：phys_footprint 不超 Stage 1a 的 ~9.3GB（LUT 仅 KB 级）。

## 内存安全

- LUT 是 256×i32 ≈ 1KB，无新增大 buffer；不改 mmap/residency 策略。
- 运行前过 13GB phys_footprint 看门狗（用 #02 的看门狗，或临时沿用 Stage 1a 的兜底）。

## 验证命令

```sh
make
./ds4 -m gguf/ds4flash-k4.gguf --ctx 2048 --temp 0 -n 32 -p "1+1=?"
./ds4 --dump-logprobs /tmp/k4.json --temp 0 -m gguf/ds4flash-k4.gguf -p "1+1=?"
```

## 产出

- 正确输出的 k4 → 可跑 `ds4-eval` 质量 / 有意义的多 token 速度。
- 解锁 #03（Stage 1b 标定）的 MTP 接受率与真实地板测量。
