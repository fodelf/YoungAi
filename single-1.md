# single-1.md — 复审 single.md: 54.2 ms 为什么还离 25 ms 这么远(2026-09-16)

> 读者假设第一次接触。本文只做一件事: 拿 single.md 自己给的实测数, 对着代码逐项核, 回答"是方案错了,
> 还是代码里藏着 bug"。**没有改任何代码, 没有新跑任何尺**; 凡是我算出来的数都标【估】, 凡是要上机验的
> 都写清楚用什么尺、看什么指标。对应代码全部给到文件:行。

## 0. 三句话

1. **single.md 的核心判决"核内到头, 墙是盘上格式"不成立。** 它靠两个实验: A 路平面副本只值 4%,
   S2 三刀全负。复核后, A 路实验根本没用上平面格式的好处(核里仍按 4 B 一次读, §2.2), S2 三刀没有一刀碰到
   真瓶颈(§2.1)。而且本仓 09-05 的 fused2 VQ 核在同一块板上跑到 **196 GB/s**(memory `spark_gb10_decode_byte_wall`:
   gateup 140 + down 70 µs/层), 现在的 v41 VQ 核只有 **110**。同一台机器、同一类格式, 差 1.8 倍, 这不是格式的墙。
2. **两个大核都有能算出来的形态病, 不是玄学**: VQ 核的 L1/LSU 波前算术正好算出 115 GB/s(实测 110~115, §2.1);
   fp4 GEMV 每条访存指令只搬 4 B, 而 214 GB/s 的 fp8blk 核每条搬 16 B(§2.2)。此外三处"逐位同"的纯 bug:
   注意力 128 键阈值让短上下文全走 8-block 串行核(5.2 ms/步, §2.3), 每步 7 次隐式排空(§2.4),
   S3 说消失的 rsqrt/scale_rows 还在发(§2.5)。
3. **最大的坑在尺上**: 所有账都在 3 token 提示上算; 12k 上下文一步 **211 ms**, 比短上下文多 157 ms,
   这 157 ms **没有分账**(fable5 只有总数)。indexer 打分核在解码时 grid 只有 1 个 block(§2.3), 就是 S4 前注意力核那种病。
   用户要 40 t/s 是在真对话里要, 那里注意力/indexer 才是主菜, single.md "长上下文已治"是没证据的话。

## 1. 用 single.md 自己的数重算 54.2 ms

| 项 | ms | 墙(字节÷240) | 差 | single.md 的解释 | 复审 |
|---|---:|---:|---:|---|---|
| 骨架 GEMV | 22.2 | 15.8 | 6.4 | "大矩阵已贴墙, 差在小矩阵固定开销" | 它自己的逐形状表反驳它: wq_b 178 / wo_a 170 / wo_b+down 158 / head 189, **没有一个到 240**。按形状算差距: 大矩阵 ≈4 ms, 小矩阵 ≈2 ms【估】 |
| 专家 VQ | 14.7 | 6.7 | 8.0 | "VQ 固有代价, 换格式=换量化" | 本仓 fused2 同类核 196 GB/s; 现核瓶颈是 LSU 波前(§2.1) |
| 注意力(短) | 5.2 | ~0 | 5.2 | "长上下文已治" | 短上下文走的是 8-block 原核(阈值 128 挡住 split), 130 µs/层算 ~70 个键 |
| mHC+engram+杂 | 6.8 | ~3 | 3.8 | — | 每步仍 ≈1400 发(47443 发/33 步, fable5 S6 自己的数), 不是 §3 写的"≈400" |
| GPU 空转 | 3.2 | 0 | 3.2 | "主机循环 + 读盘" | 每步 2 次 DeviceSynchronize + 5 次同步 cudaMemcpy(§2.4) |
| **合计** | **54.2** | **26.3** | **27.9** | | |

★注意★: §0 说"读 6.0 GB 要 25 ms", §1 表合计是 **6.3 GB = 26.3 ms**。40 t/s = 25 ms 在 6.3 GB/步不变的前提下等于
**105% 带宽**。所以"40"这个数在本文件的字节账下本来就不是核形态能兑现的, 要 40 就得同时降字节或一步多 token;
但 54.2 → 30 出头这一段(§4 的账)全是核形态和主机侧的活, 与格式无关。

## 2. 逐项深挖

### 2.1 专家 VQ 核(14.7 ms): 瓶颈是 L1/LSU 波前数, 一道算术就对上实测

核: `src/cuda/cuda_v41_4.inc.cu:211` `v41_vq_row_dot`。一 warp 一行, lane j 拿第 j、j+32、… 个 12 bit 索引,
每个索引对应 8 个权重。**一轮(warp 处理 256 个权重)LSU 要付的波前**(波前 = L1 一个时钟能处理的一个访存单元):

| 读什么 | 代码 | 每 warp 一轮的波前数【估】 |
|---|---|---:|
| 位流 | `memcpy(&wv, m.ix + by, 4)`, by 是任意字节偏移 | 编译器不能假设对齐 ⇒ 大概率发 **4 条 LDG.U8**(硬件不支持非对齐 32 位读, 会报 misaligned address —— 09-15 20 B 跨距那次撞的就是这个错), 每条跨 48 B 段 ≈ 2 扇区 ⇒ ~8 |
| 激活 x | 两条 `float4`, lane 步长 32 B(`xs + j*8`) | 一条指令跨 1 KB ⇒ 8 波前, 两条 ⇒ **16** |
| 码本 | shared 随机 16 B/lane | 4~8 |
| **合计** | | **≈30 波前 / 256 权重** |

LSU 每 SM 每时钟 1 个波前 ⇒ 8.5 权重/clk = 1.6 B 位流/clk ≈ 2.4 GB/s/SM, ×48 SM = **115 GB/s**。实测 110~115。
这台机器上 mem_ceiling 满占用能到 238, 所以不是 DRAM, 不是占用率, 是 **L1 数据管道被激活重读和字节读塞满了**。

这也解释了 S2 三刀为什么"全持平"(fable5 09-15 表): 码本改全局 gather ⇒ 波前更多(慢 21%, 对); 激活进 shared ⇒
lane 步长没变, 波前数一个没少(持平, 对); 20 B 跨距 ⇒ 退成 4 次读, 波前更多(持平偏慢, 对)。
**三刀都没减波前, 所以三刀都不动。** 而 fused2(`src/cuda/cuda_vq_fused2_0.inc.cu` 头注 ①~④)的四刀正好全是减波前:
激活转置装 shared 让 32 个 lane 读连续 float4(4 波前), 码本 uint2 一次, 位流每 lane 一段连续字节整读进寄存器后
funnelshift 抽位(全对齐 32 位读, 偏移编译期常量), 两行/四行一趟先发全部读再算。09-05 消融"去点积只留访存 144 µs =
196 GB/s"就是这个形态的上限。

**怎么验**(ncu, `--replay-mode application`, 一次只问这三个): `l1tex__data_pipe_lsu_wavefronts.sum`、
`smsp__warp_issue_stalled_mio_throttle` / `lg_throttle` / `long_scoreboard` 占比、`dram__throughput`。
如果 mio/lg throttle 占大头而 long_scoreboard 小, 就是上面这笔账。另外 `cuobjdump -sass` 看那条 memcpy 到底发了几条 LD。

**改法方向**(按 fused2 形态, 数值逐位同: 位流怎么读、激活放哪、都是"数据在哪", 乘加配对与 f32 累加序不动):
位流按 lane 连续段整读(12 bit × 20 索引 = 30 B/lane/行, 补齐到 32 B 走两条 LDG.128); 激活转置进 shared 让
lane 连续; 码本 uint4 一次(已是)。预期 gateup 235 → ~130 µs/层(fused2 同形态 140), 整步 **−5~6 ms【估】**。

### 2.2 骨架 GEMV(22.2 ms): 每 lane 4 B 的读法, 把平面副本的好处挡在门外

核: `src/cuda/cuda_v41_4.inc.cu:33` `v41_fp4x32_gemv_kernel`。现在 4 个 lane 分一个 16 B 块, 每 lane
`memcpy(&w4, p + q*4, 4)`(非对齐, 同 §2.1 的 LD.U8 疑问) + 1 B scale; 一 warp 一轮读 8 块 = 136 B, 两组展开 = 272 B 在飞。

对照 214 GB/s 的 `v41_fp8blk_gemv_kernel`(`src/cuda/cuda_v41_gemv_highprec.inc.cu:184`): **同一个 8 warp/ksplit
形态**, 但每 lane 一条 `uint4` = 16 B, 一 warp 一轮 512 B, 一组就 512 B 在飞。两个核唯一的结构差别就是
**每条访存指令搬多少字节**。fp4 核每字节多付 4 倍指令、每 warp 在飞字节只有一半。

★A 路实验为什么不算数★: 平面副本(`cuda_v41_fp4_planar.inc.cu`)装好后, 核里 PLANAR=1 那条路仍是
`p + q*4` 的 4 B memcpy(`cuda_v41_4.inc.cu:80-84`)。平面格式的全部意义是"16 B 对齐 ⇒ 一条 LDG.128", 这一条一次都没用上,
所以只量到对齐带来的 4%。**拿这个实验去否定"格式/读法是主因", 是实验没测到要测的东西。** 副本已经在库里, 不用重转 GGUF 就能补测。

正确形态: PLANAR 时 lane 拿整块(LDG.128 16 B + scale 1 B, 32 个 lane 的 scale 连续 32 B = 1 波前), 一 warp 一轮 32 块 =
1024 权重。代价在 x 侧: lane 要 32 个 f32 = 128 B, 一 warp 4 KB = 32 波前 —— **FP4 GEMV 的结构性负担是激活字节是
权重字节的 8 倍**(4 B 激活 vs 0.5 B 权重)。x 已在 bf16 格点(核注释 `cuda_v41_4.inc.cu:93` 自己写死的前提),
以 bf16 存一份 x 给 GEMV 读, 字节减半且**逐位同**(bf16→f32 是补 16 个 0, 值不变)。
数值门: lane 分工变 ⇒ warp 规约序变 ⇒ NLL 尺(与 09-15 段 2 那次同)。预期大矩阵 165~189 → 200+, GEMV 22.2 → ~18 ms【估】。

小矩阵(wkv 1.4 MB / wq_a 3.5 MB / gate·up 6.3 MB)的 2 ms 是另一回事: 一层 8 发串行, 每发独立启动排空。
合发试过持平是因为合发后每发还是各自排空; 真正省的是 PDL(§2.5)让下一发的序幕叠在上一发的尾巴上。

### 2.3 注意力: 短上下文 5.2 ms 是阈值挡的; 长上下文 157 ms 没人分账

**短上下文**: `src/cuda/cuda_v41_attn_split.inc.cu:21` `V41_ATTN_SPLIT_MIN_KEYS 128`。32 步解码可见键 ~70 < 128,
于是走原核 `v41_sparse_attn_kernel`(grid 8 block, 48 SM 用 8 个), 实测 130 µs/层 × 40 = 5.2 ms。
split 核切 8~16 键一段就能铺满 SM, 段核 + merge 估 ~15 µs/层 ⇒ **省 ~4.5 ms/步【估】**。
阈值的理由是"段太短合并开销占大头", 但合并核只有 64 block × 512 维, ~5 µs; 130 vs 15+5 不用争。门: NLL 尺(分段改变在线 softmax 分组, 与 S4 同)。

**长上下文**: 12464 token 提示 211.3 ms, 减短上下文 54.2 = **157 ms 随上下文涨**。字节账里随上下文涨的只有 KV 读
(f32 行 2 KB × 3116 组 × 40 层 ≈ 250 MB ≈ 1 ms), 所以 157 ms 几乎全是核形态。fable5 09-15 S4 只给了总数, 没有逐核表。
从代码能指出的嫌疑, 按大小排:

1. **indexer 打分核 grid = n_tok**: `src/cuda/cuda_v41_2.inc.cu:140` `v41_indexer_score_kernel<<<n_tok, 256>>>`。
   解码 n=1 ⇒ **整层只有 1 个 block, 8 个 warp 串行啃全部组**, 每组 64 头各做 4 FMA + 5 级 shfl + 两次 bf16 舍。
   12k/ratio 4 = 3116 组 ⇒ 每 warp 390 组 × 64 头 × ~150 clk ≈ **2.5 ms/层【估】**, 8 个 indexer 源层(§1 表 "indexer q_b 8 发")
   ≈ 20 ms/步。预填时 n 大 grid 够宽, 解码就剩 1 个 SM —— **与 S4 修之前的注意力核一模一样的病**, 32k 时翻倍。
2. **topk 核**(`cuda_v41_2.inc.cu:198`): 1 block, radix select 并行, 但写出段线程 0 串行扫 ng(3116 次)。~20 µs/层, 小。
3. split 注意力 2176 键(128 窗口 + 2048 topk) ⇒ 34 段 × 8 头组 = 272 block, 每块 64 键 8 tile, 内层还是每键每头 5 级 shfl
   串行 ⇒ ~1 ms/步【估】。
4. **压缩 KV 缓存存 f32**: `core_v41_attn.c:75-81` act_quant_fp4/fp8 之后仍以 f32 [ng][512] 落缓存(2 KB/行), 官方是 e4m3
   512 B/行。读 4 倍字节, 12k 处 ~1 ms/步, 32k ~2.6 ms。不改值(格点上的数以 f32 存或以 fp8 存是同一个数), 逐位同。

这四条加起来解释不了 157(能指到的 ≈25 ms), **剩下的必须上尺**: 同一把 `d0a_decode_profile.sh` 的 nsys 分账,
PROMPT 换成 12k 文件, 一次就出逐核表。这张表不出来, 长上下文的任何"已治/未治"都是猜。

### 2.4 主机侧: 每步 7 次隐式排空, 全是逐位同能去掉的

按一步的时间顺序(解码 n=1):

| 在哪 | 代码 | 干了什么 |
|---|---|---|
| 上一步出口 | `core_v41_forward.c:327-328` | `ds4_gpu_end_commands`(= cudaDeviceSynchronize, `cuda_graphcap.inc.cu:434`) 紧接着 `ds4_gpu_synchronize` **又一次** |
| 取 token | `core_v41_api.c:197-198` | argmax 核 → synchronize → 同步 D2H 4 B |
| 本步开头 | `core_v41_forward.c:280,286` | tok / pos / pre_mix **三次同步 `cudaMemcpy` H2D**(`ds4_gpu_tensor_write` 是同步版, `cuda_lifecycle.inc.cu:209`); pos = pos0+i 是常量算术, pre_mix 是每步一样的 one-hot |
| engram 层 L01/L14 | `core_v41_engram.c:242` | 层循环**中间**一次同步 H2D(48 行 raw), 队列排空后主机再重灌 |
| kv 源层每 4 步 | `core_v41_attn.c:65` | posg 同步 H2D(1 个 int, 值 = (g0+g)·ratio, 核里两条乘法就能算) |

每次同步 = GPU 排空 + 主机往返 + 重灌队列, 50~150 µs 空转; S6 量到的 0.87 + 0.47 ms 就是这一串。
逐位同的修法: argmax 核直接写 `st->tok`(token 留在设备上, emit 用晚一步的异步 D2H 值打印); pos / pre_mix / posg 由核在设备上生成;
engram raw 走 pinned 缓冲 + `cudaMemcpyAsync` + 事件。预期 **−1.5 ms【估】**, 且是 CUDA graph 的前提(graph 里不能有同步 memcpy)。

### 2.5 发射数 ≈1400/步, 不是 ≈400; S3 说消失的核还在

- fable5 S6: `cudaLaunchKernel` 47443 发 / 33 步 ≈ **1437 发/步**。single.md §3 S6 行写"S3/S5 之后发射数 ≈400", 与自己的数对不上。
- S3 表里"row_rsqrt / scale_rows / round_bf16 / add 随之消失": 代码里 `ds4_gpu_v41_hc_mix_tensor`(`cuda_v41_hc.inc.cu:30-34`)
  仍每半层发 rsqrt + f32 gemv + scale_rows 三发, 80 半层 = 240 发/步, §1 表实测 rsqrt 1.13 + scale 在"杂"里。
- 窗口平移每层 2 发 D2D memcpy(`core_v41_attn.c:209-210`), 80 发/步; y = copy + add + round 三发/层(`core_v41_forward.c:155-157`)。
- GPU 侧核间空档 4.4 ms ÷ 1437 ≈ **3 µs/发**。这是 GPU 前端每个核的启动/排空延迟, 不是 CPU 发射慢。
  S6 用"CPU 发射只花 3.4 ms"否定 graph, 是把两件事混了: graph 与 PDL(programmatic dependent launch)减的正是 GPU 侧这 3 µs。
  仓里 PDL 基础设施已有(`cuda_internal.cuh:88-103` `ds4_launch_pdl` / `DS4_PDL_WAIT`, fused2 核在用), **v41 核零使用**(grep 结果 0)。
- 逐位同的融合候选: rms_norm 并进后续 GEMV 的序幕(每 block 用同一棵 256 线程树形归约重算, 结果逐位同)−163 发;
  rsqrt + scale_rows 并进 f32 gemv 尾巴 −160 发; 窗口改环形缓冲 −80 发 memcpy; copy+add+round 一发 −80 发。
  1437 → ~900【估】, 空档 4.4 → ~2.5 ms; 再上 PDL 或 graph 收剩下的。

### 2.6 尺本身的四个问题

1. **3 token 提示不代表使用形态**(§2.3)。真对话里每步是 100~200 ms, 不是 54。
2. `d0a_decode_profile.sh` 带 `--zchain gguf/v41/gr-fin-40-fp4` 跑速度尺。若目录里挂了 amp_Lnn.bin, 每层多 2 发 cuBLAS Sgemm + 1 发 round
   (`core_v41_amp.c:257-260`); 只挂 gr 文件则只多每行一次 gov 读。铁律"跑分只认裸路径"下, 速度尺至少要裸跑一次对照并写明差多少。
3. **DSpark 默认是开的**: `src/cli/cli_diag.c:44` `set_dspark(!no_dspark)`, `core_v41_api.c:14` 初值 1; single.md §5 写"`--no-dspark` 默认关", 说反了。
   这次 nsys 表(GEMV<1> 329 发 = 8×40 + head + indexer 8)证明量的是纯解码 —— 要么现役文件没带运行参数(`core_v41_draft.c:54` ready=0),
   要么就是碰巧。脚本没传 `--no-dspark`, 换一个带参数的文件就会量成投机路而不自知。另外 `st->mainh` 只看元数据不看开关
   (`core_v41_forward.c:57,197`), 带三塔的文件每步白发 3 发 hc_mean, 小但是死活。
4. GEMV 核注释(`cuda_v41_4.inc.cu:82-83`)断言"4 B memcpy 让编译器发一条非对齐 32 位读"——CUDA 硬件没有非对齐 32 位读, 要么编译器发了
   4 条字节读, 要么它先按对齐读两个字再拼。没看 SASS 之前这条注释是假设不是事实, 而 §2.1/§2.2 的波前账取决于它。

## 3. 判决: 方案错在哪, 代码错在哪

**方案层面的错(四条)**:
- "核内到头"建立在一个 ncu 指标(24.92 扇区/请求)上, 没有 stall 原因分解, 没有波前/指令计数; 一个数解释不了两个核。
- A 路实验没用上平面格式的读法(§2.2), 拿它下的"格式不是主因"结论无效。
- 六刀估值全按"形态改了就到 95% 带宽"算, 没有一刀先算 LSU/在飞字节的账 —— 所以四刀落空(single.md §6.2 自己承认)。
- 在 3 token 上优化, 目标却在长上下文; 长上下文那 157 ms 连分账都没有。

**代码层面的错(不改数值就能修的, 按收益排)**:

| # | 病 | 位置 | 预期【估】 | 门 |
|---:|---|---|---:|---|
| 1 | VQ 内层形态: 位流字节读 + 激活 32 B 步长 + 每索引一次 gather | `cuda_v41_4.inc.cu:211-230` | −5~6 ms | 逐位同(乘加配对与序不动) |
| 2 | 注意力 split 阈值 128 | `cuda_v41_attn_split.inc.cu:21` | −4.5 ms | NLL 尺 |
| 3 | GEMV 每 lane 4 B; 平面副本没用 LDG.128; x 用 f32 | `cuda_v41_4.inc.cu:69-104` | −4 ms | NLL 尺(规约序变) |
| 4 | 每步 7 次隐式排空 | §2.4 五处 | −1.5 ms | 逐位同 |
| 5 | rsqrt/scale_rows/窗口 memcpy/copy-add-round 没融; 零 PDL | §2.5 | −2~3 ms | 逐位同 |
| 6 | indexer 打分核 1 block | `cuda_v41_2.inc.cu:140` | 长上下文 −20 ms/步 | 逐位同(每组独立) |
| 7 | 压缩 KV 缓存存 f32 | `core_v41_attn.c:75-81` | 长上下文 −1~3 ms | 逐位同 |

短上下文账: 54.2 − (5.5 + 4.5 + 4 + 1.5 + 2.5) ≈ **36 ms(28 t/s)【估】**; 再往下就是 6.3 GB 的字节地板 26 ms, 那一段要动字节。
每一格落地后按 §1 尺重出实账, 不按这里累加 —— single.md 六刀的教训就是估值不能当账。

## 3.5 实测补账(2026-09-16 本轮, 同一 `-fp4-mtp-native.gguf`, 同机器状态, `--no-dspark`)

§4 第 1 步的两张尺已经跑完, 两条最重要的猜测都被证实, 而且比猜的更狠。

**(a) 12k 上下文的 157 ms 是谁的** —— `d0a_decode_profile.sh 16 yes <模型> speed-bench/promessi_sposi.txt 40000`
(12478 token 提示, 解码 4.51 t/s = **222 ms/步**; nsys 逐核, 第 27~40 步平均):

| 核 | ms/步 | 发/步 | grid |
|---|---:|---:|---|
| **v41_indexer_score_kernel** | **80.33** | 8 | **(1, 1)** ← 解码 n_tok=1 |
| **v41_candidate_kernel** | **46.39** | 1 | **(1, 1)** |
| v41_fp4x32_gemv_kernel | 22.11 | 329 | |
| v41_sparse_attn_split_kernel | 10.75 | 40 | |
| v41_vq_gateup / down | 9.41 / 5.12 | 40 / 40 | |
| **v41_topk_kernel** | **5.89** | 8 | **(1, 1)** |
| 其余(hc/norm/rope/memcpy…) | ≈6 | ≈1100 | |
| 核忙合计 | 188 | 1631~1670 | 间隙另有 4~24 |

★indexer 三件套 = 132.6 ms = 整步的 66%, 三个核的 grid 全是一个 block(48 个 SM 用 1 个)★。
引擎自己的逐层毫秒也指着同一处: 40 层里 36 层都是 1.3 ms, 而 **L20 一层 60.6 ms**(候选源层),
L24/28/32/36 各 14.7 ms(其余 indexer 源层), L02/08/14 各 8~9 ms(kv 源层)。

定罪到行:
- `v41_candidate_kernel`(`cuda_v41_2.inc.cu`): 线程 0 串行 `O(kk×nb)` 选块。12k 时 nb=1558、
  kk=min(2048,nb)=1558 ⇒ **一个线程 240 万轮**。这与 09-14 给 topk 核做过的手术是同一个病, 那次漏了这个核。
- `v41_indexer_score_kernel`: `<<<n_tok, 256>>>`, 解码时 8 个 warp 轮流啃 ng 个组。
- 三个核在短提示下分别是 0.21 / ~0 / 0.11 ms —— 排在表的第 12 位以后, **所以 single.md 全篇没看见它们**。

**(b) 短上下文基线(32 步, 3 token 提示)**: 中位 **54.6 ms(18.30 t/s)**, 与 single.md §6.1 的 54.2 对得上。
逐核: GEMV 20.83(329 发) / gateup 9.13 / down 4.91 / **注意力 4.45(且从第 2 步的 1.6 线性涨到第 33 步的 7.3)** /
hc_fused 1.73 / engram wkv 1.39 / f32_gemv 1.16 / row_rsqrt 1.06 / bf16_gemv 0.97 / 其余 ≈1.8; 间隙 4.3。
**发射数 1591~1630/步** —— 坐实 §2.5(single.md 写的"≈400"不对)。

**(c) SASS 直接验了 §2.1/§2.2 的两条假设**(`cuobjdump -sass ./ds4`, 现役二进制):
- `v41_vq_gateup_kernel`: **212 条 `LDG.E.U8`** —— 位流那句 4 字节 `memcpy` 落在任意字节偏移上,
  硬件没有非对齐 32 位读, 编译器只能拆成 4 次字节读。§2.1 的波前账成立。
- `v41_fp4x32_gemv_kernel`: 激活那一行编出 **8 条标量 `LDG.E.CONSTANT`**, 而 214 GB/s 的
  `v41_fp8blk_gemv_kernel` 只有 16 条 32 位读却每条覆盖 16 个元素。按地址算, GEMV 一条激活读指令的
  32 个 lane 铺开 **1024 B**(每 lane 只取 4 B)⇒ 一次请求碰 32 个扇区。★ncu 那个"24.92 扇区/请求"的
  真正来源是激活, 不是权重的 17 B 跨距★ —— 所以平面副本(只动权重)只值 4%, 而 single.md 据此判"核内到头"。

## 3.6 第一批落地: 三刀一正两负(2026-09-16)

按 §3 的清单先动了四刀, 一次编译一起测, 靠**逐核表**分别归因(不是靠 A/B 猜):

| 刀 | 核 | 改前 ms/步 | 改后 ms/步 | 判 |
|---|---|---:|---:|---|
| #2 注意力 split 阈值 128 → 16 + 段长自适应 | sparse_attn(原核 → split+merge) | 4.45 | **1.40** | **正, 省 3.05** |
| #3 GEMV 8 lane 一块 + 激活 float4 | fp4x32_gemv | 20.83 | 22.75 | 负 9%, 回退 |
| #1 VQ 位流对齐读 + funnelshift | vq_gateup + down | 14.04 | 18.81 | 负 34%, 回退 |
| #6 indexer 打分 grid / 候选块 radix | (短上下文量不到) | — | — | 看 §3.7 |

整步: 18.30 → 17.03 t/s(四刀在一起) → 回退两负刀后 **18.98 t/s**。

★两刀为什么全负, 一句话★: 我拿 SASS 的**指令条数**和 ncu 的**扇区/请求**当瓶颈证据, 但这两个都是 L1 侧的量;
这两个核真正在等的东西没量过。省下的访存指令要么打在 L1 命中上(GEMV 的激活本来就被 block 内 8 行重用),
要么被新加的分支吃掉(VQ 的边界判断)。★教训写进代码注释了★: 动这两个核之前, 先用
`smsp__warp_issue_stalled_{long_scoreboard,mio_throttle,lg_throttle}` 看 warp 停在哪一类 ——
这正是 §4 第 1 步(b) 写的那趟 ncu, 我跳过它直接改码, 于是两刀全负。

## 3.7 第二批落地: indexer 两刀, 长上下文解码 2.8×(2026-09-16)

两刀都**逐位同**(只改"谁去算"与"怎么选", 每个 (i,g) 的算法、bf16 舍点、并列规则一个没动):

| | 改前 ms/步 | 改后 ms/步 |
|---|---:|---:|
| `v41_indexer_score_kernel` grid (n_tok) → (n_tok, 组块) | 80.33 | **0.90** |
| `v41_candidate_kernel` 线程 0 的 O(kk×nb) → radix select + 全选早出 | 46.39 | **掉出前 18 名(<0.12)** |
| 核忙合计 | 188 | **61.2** |
| 墙钟/步 | ~200 | **67** |

**12464 token 上下文解码: 4.51 → 12.60 t/s(222 → 79 ms/步, 2.8×)**; 短上下文 18.30 → 18.98 t/s。

现在 12k 一步 67 ms 的构成: GEMV 21.9 / 注意力 split 10.5 / 专家 14.3 / **topk 5.9** / mHC 等 ≈4 / 间隙 4.7。
`v41_topk_kernel` 顶上来了 —— 它也是一个 block(radix 部分并行, 但 256 线程一个 block 扫 12480 个组), 是下一刀。

## 4. 建议的顺序(每步分钟级门)

1. **先补两张尺, 不改代码**(半小时内出):
   (a) `d0a_decode_profile.sh` 的 nsys 分账在 12k 提示上跑一遍, 拿到长上下文逐核表(§2.3 的 157 ms 是谁的);
   (b) ncu 对 `v41_vq_gateup_kernel` 与 `v41_fp4x32_gemv_kernel<1,1>` 各问 §2.1 那三个指标 + `cuobjdump -sass` 看 memcpy 发了几条 LD。
   两张表定 §2.1/§2.2 的账是不是真的; 是, 就按 fused2 形态改; 不是, 本文 §2.1/§2.2 作废, 但 §2.3~§2.5 不受影响。
2. 逐位同的三刀先走(#2 阈值、#4 主机同步链、#6 indexer 多 block): 门是温 0 64 位 logits ×8 遍逐位同(#2 走 NLL)。
3. VQ 内层重写(#1), 门 = 逐位同 + nsys gateup ≤ 130 µs/层。
4. GEMV lane 16 B + 平面 + bf16 x(#3), 门 = NLL 尺 + 每形状带宽表(小矩阵 ≥ 180, 大矩阵 ≥ 200)。
5. 融合 + PDL(#5), 门 = 逐位同 + 发射数 ≤ 900。

## 5. 不做的 / 不变量(与 single.md 一致)

位宽不动; 不做 CPU 路; 不做超时/降级兜底; 判决只认同机器状态中位; 未验证不提交; 发车先批准; 不报时长。
本文的所有【估】只用来排优先级, 不进任何账。结论与后续实测按铁律进 fable5.md。
