# clear.md — 清掉 V4.1 引擎里的非 FP4 遗留(方案, 2026-09-15; 只方案不写码)

配套 `speed.md`(目标 预填 300 / 解码 100 / 1M)。speed.md §2 定了"每个张量该是什么精度",
本文回答另一件事: **今天的代码里, 哪些地方还没按 §2 走、各自吃掉多少速度、按什么顺序清。**
数字全是 09-15 盘上实读(`dump_gguf_meta --stats`、`d0a_decode_profile.sh`、grep 行数), 不是估的。

## 0. 三句话

1. **盘上早就是 FP4 了, 引擎还没是。** 110.27 GiB 的 GGUF 里 98.8% 是 VQ/fp4x32; 但引擎运行时
   每 token 读的 6.3 GB 里**约 1.2 GB(18%)是 f32/f16 张量**, 激活从头到尾用 **f32 缓冲 + 每个消费核
   各自重舍一遍 bf16**(35 处), 解码主路上还挂着 **每层 1 条 cuBLAS f32 GEMM + engram 一条 f16 GEMM**,
   预填 wo_a 还走 **f16 暂存 + cuBLAS**, KV 缓存**全 f32**(官方是 FP4/MXFP4/FP8, 1M 处每 token 多读 ≈1.3 GB)。
2. **V4 老代码是引擎的 90%。** core 24.2k 行里 V4.1 只有 1.2k; CUDA 22.6k 里 16.6k 是 V4 核; Metal 30k
   纯 V4(V4.1 只有一个 stub); GPU 契约 186 个函数 V4.1 只用 40 个。老代码本身不跑、不吃毫秒 ——
   它吃的是: V4.1 被迫借 V4 的原语(`ds4_gpu_tensor` 没有 dtype, 只有"一袋字节"), 所以激活只能是
   f32、精度语义只能靠每个核自己舍; 每改一次契约要陪 Metal 改一次 stub; 测试 11 套 6 套是 V4 的。
3. **分四段清, 每段自己过门。** C0 纯删(零数值风险, 门 = 主尺逐位同) → C1 张量落格(转换器出新 GGUF,
   门 = 五指标) → C2 运行时格式按官方(激活 bf16、KV 按 §5.1、cuBLAS 全退, 门 = 逐位同或五指标) →
   C3 契约收口(`ds4_gpu_tensor` 带 dtype)。**速度来自 C1+C2(估 解码 −20~25% 步时), C0/C3 是让 C2 能做。**

## 1. 现状盘点

### 1.1 盘上(`dump_gguf_meta --stats`, 带三塔的 GGUF, 110.27 GiB)

| 类型 | 张量数 | GiB | 占比 | 是什么 |
|---|---:|---:|---:|---|
| 42 VQ blob | 40 | 98.01 | 88.9% | routed 专家 1.519 bpw(码本 `__half`) |
| 43 fp4x32 | 1507 | 10.88 | 9.9% | 骨架/shared/embed/head/三塔 |
| **1 f16** | 2 | **0.59** | 0.5% | engram wkv ×2(官方 FP8) |
| **0 f32** | 596 | **0.78** | 0.7% | 路由 gate、hc 系数、norm、compressor、indexer、engram q/k |
| 26/27 | 4 | 0 | | 反修/杂项 |

⇒ **非 FP4 只占盘上 1.2%**。真正的账不在盘上, 在"每 token 读了什么"。

### 1.2 每 token 读的非 FP4 字节(解码, 6.3 GB 里的)

| 张量 | 现格式 | 每 token | speed.md §2 定的 | 到 §2 后 | 全 FP4 后 |
|---|---|---:|---|---:|---:|
| engram wkv 2 × 6144 × 25600 | **f16** | 0.63 GB | FP8(官方) | 0.31 | 0.16 |
| 路由 gate 384 × 5120 × 40 层 | **f32** | 0.31 GB | f16 存 f32 算 | 0.16 | 0.04 |
| hc_attn_fn + hc_ffn_fn 20480 × 24 × 2 × 40 | **f32** | 0.16 GB | fp32(§2 留高精度) | 0.16 | 0.02 |
| compressor kv/gate 4 源层 × 2 × 5120 × 512 | **f32** | ≈0.06 GB(ratio 摊) | — (§2 漏了这行) | 0.03 | 0.01 |
| indexer proj/wk、norm、engram q/k | f32 | <0.02 | f16 | <0.01 | <0.01 |
| **合计** | | **≈1.18 GB = 18%** | | **0.66(−0.5 GB, −8%)** | **0.24(−0.94 GB, −15%)** |

§2 表把 gate/hc 留高精度的理由是"位宽一动路由就变"。**这一条不是物理墙, 是没过门的猜测** —— 用户令全 FP4,
所以本方案把它们各自当一个 A/B 项过五指标门, 门过了就落 FP4, 门退了才留(§5 写清楚怎么判)。

### 1.3 运行时精度遗留(代码层; 这是吃指令的, 不是吃字节的)

| # | 遗留 | 在哪 | 吃什么 |
|---|---|---|---|
| R1 | **激活全 f32 缓冲, 值落 bf16 格点, 每个消费核自己 `v41_bf16r` 一遍** | `core_v41.h` 第 5 行自述"全 f32 行主序"; `v41_bf16r` 35 处(v41_1 ×8, v41_2 ×9, v41_4 ×11, v41_3 ×2, fused ×4, mma ×1) | 解码 gemv 内层每 8 个元素 8 次舍(4 条 ALU 一次); 激活字节 2×; 09-15 gemv 一针证明这个核是**指令受限**的 —— 舍入就是指令 |
| R2 | **cuBLAS f32 GEMM(`cublasSgemm`)当 GEMV 用** | `cuda_v41_1.inc.cu:146` `ds4_gpu_v41_matmul_f32_tensor`; 调用: gate(40 层/每 token)、compressor kv+gate(4 源层)、indexer wk(4)、indexer proj(8) | n=1 时 Sgemm 是最差的 GEMV 形态; 每层一发, 40 发/步 |
| R3 | **engram wkv 走 V4 老原语 `ds4_gpu_matmul_f16_tensor` + 独立 round_bf16 核** | `core_v41_engram.c:239-240` | 0.63 GB/token 走 cuBLAS hgemm(n=1), 再多一趟 round 核 |
| R4 | **预填 wo_a n>8: fp4x32 → f16 暂存 → cuBLAS hgemm** | `cuda_v41_1.inc.cu:126-143`(`v41_dequant_fp4x32` + `v41_x_to_f16_kernel`) | 唯一还落 f16 暂存的预填矩阵: 8 组 × 4096→1024 = 33.5M 元素/层, 写 67 MB + 读 67 MB, 40 层 = 5.4 GB/块 白流量; 张量核用的还是 f16(81~91 TFLOPS)不是 NVFP4(284~356) |
| R5 | **NVFP4 GEMM 出口再跑一趟 `round_bf16` 核** | `cuda_v41_1.inc.cu:112, 143` | 多一遍读写 [n_tok × out_dim] f32 |
| R6 | **KV 缓存全 f32** | `core_v41.h` `comp_kv/index_k/win`: [组][512] f32 = 2048 B/组; 官方 288 B(E2M1+E4M3/16), 索引 K 512 B vs 68 B(MXFP4), SWA 2048 B/行 vs 544 B(FP8) | ≤32k 时看不出; **1M 处 L20 扫 100 万组索引 K 是 512 MB 不是 68 MB, 四源层合计每 token ≈1.3 GB 而不是 0.17** ⇒ 1M 解码字节 +20%; 内存 5.1 GB 而不是 0.9 |
| R7 | **码本 `__half`** | VQ blob 内, 64 KB/矩阵 × 3 × 384 × 40 | 每 token 读 6 专家 × 3 × 64 KB × 40 = 46 MB(0.7%); 大头是 shared 占用: 每个融合核块先搬 64 KB 码本进 shared, 锁死占用率(speed.md B 段的真因) |
| R8 | **`cuda_vq_prefill.inc.cu`(f16 暂存的老预填专家路, 325 行)** | 09-15 起被 NVFP4 路硬接管, 已无调用者 | 死代码(不吃时间, 但它的 `g_vqp` 缓冲池还被 nvfp4 路借用, 见 §3 C0) |
| R9 | **`ds4_gpu_tensor` 无 dtype** | `ds4_gpu*.h` 契约 | 一切精度约定靠注释与调用者记性; R1 的根 |

**nsys 实账(09-15 傍晚二进制, 32 token 解码, 中位 66.5 ms/token; `d0a_decode_profile.sh 32 yes`)**:

| 核 | 占比 | 每 token | 是谁 |
|---|---:|---:|---|
| `v41_fp4x32_gemv_kernel<1>` | 31.8% | 22.6 ms | 骨架 FP4 GEMV(R1 的舍入税在里面, 没法单独量) |
| `v41_vq_gateup` + `v41_vq_down` | 38.0% | 26.9 ms | 专家 VQ 即乘(R7 码本 shared 占用在里面) |
| `v41_sparse_attn_kernel` | 6.5% | 4.6 ms | 稀疏注意力(解码走标量版) |
| **`matmul_f16_ordered_chunks_kernel`** | **4.5%** | **3.2 ms** | **R3: engram wkv 走 V4 老 f16 原语**(每 token 2 发, 每发 1.4 ms) |
| **cuBLAS `gemvx` + `dot_kernel` + `reduce_1Block`** | **4.6%** | **3.2 ms** | **R2: 六种 f32 权重的 `cublasSgemm`**(每 token 45+85 发小核) |
| `v41_hc_split` / `rms_norm` / `row_rsqrt` / `hc_post` | 8.3% | 5.9 ms | mHC 与归一化(R1 的 f32 缓冲各读各舍) |
| 其余(router/rope/topk/act_quant/indexer) | 1.4% | 1.0 ms | |

⇒ **R2+R3 两条非 FP4 老路直接吃掉 6.4 ms = 每 token 的 10%**, 这是最没争议的一刀: 权重 0.94 GB 走的
是最差形态的核(f32 GEMV 45 发 + f16 hgemm 2 发), 换成骨架同款 fp4x32 gemv 后按 gemv 今天的 86 GB/s 算只要 ≈2.8 ms
(C1 落 FP4 后 ≈0.6 ms)。R1 的舍入税藏在 gemv 22.6 ms 里, 只有 C2 做完才能量出来。

### 1.4 V4 老代码体量(不在 V4.1 路上, 但每次都一起编进 ds4)

| 区域 | 行数 | V4.1 用到的 |
|---|---:|---|
| `src/core/` 非 v41 文件 | 20 976 | gguf 读取/tokenizer/采样/session/payload/kv 那些是共用的(≈8k); `core_gpu_*`/`core_cpu_*`/`core_kern_*`/`core_attn`/`core_ffn*`/`core_moe`/`core_hc`/`core_sidecar`/`core_gpu_dspark`(V4 的草稿器)/`core_session_spec` ≈11.6k 是 V4 前向 |
| `src/cuda/` 非 V4.1 分片 | 16 595 | 只借 `lifecycle`(tensor 生命周期)、`graphcap`(命令流)、`modelmap`、`cuda_api_matmul_1` 里的一个 f16 matmul(R3) |
| `src/metal/` + `metal/` | 30 011 | 零(`metal_v41_stub.m`) |
| `src/common/ds4_quantfmt` 的 V4 类型(iq2_xxs/q2_k/q4_k/q8/…) | 部分 | V4.1 只用 fp4x32 + VQ + f16/f32 |
| 根目录小模块 ds4_z/loss/corr/zchain/zfinetune/multimodal/spatial/css | 2 379 | 零(反修放大器走 `core_v41_amp.c` 自己读 `gr_Lnn.bin`); ds4_posttrain 1 处 |
| GPU 契约 `ds4_gpu*.h` | 186 个函数 | 40 个 |
| 测试 11 套 | | V4.1 能跑的: `--server`/`--engine-units`/`--rax`/`--tp-allreduce`(离线); `--metal-*` ×4、`--mpp`、`--logprob-vectors`、`--long-context`、`--tool-call-quality`、`--local-golden-vectors` 全是 V4 模型的 |
| linecount 豁免 | 8 项 | 4 项是 V4 单函数(`core_gpu_prefill_attn`/`core_gpu_decode_layer`/`metal_*`/`cuda_moe_launch`) |

★说人话: **删 V4 代码本身不会让解码快一毫秒** —— 它是死代码, 不跑。它的代价是 R9: V4.1 骑在 V4 的
张量模型上, 想把激活改成 bf16、KV 改成 FP4, 就得先让 `ds4_gpu_tensor` 认 dtype, 而这个改动要么陪着
186 个 V4 函数 + Metal 一起改, 要么先把它们请出去。所以 C0 排在 C2 前面, 不是因为它快, 是因为它让 C2 便宜。

## 2. 目标态

精度表按 speed.md §2, 本文不复述; 这里只补 §2 没写的**运行时格式**三行(全部 = 官方, 不比官方低):

| 运行时数据 | 今天 | 目标 | 为什么 |
|---|---|---|---|
| 层间激活(hc 四路、x、xn、q/kv/o、专家中间量) | f32 缓冲, 值在 bf16 格点, 消费核各自舍 | **bf16 存**(官方 `torch.bfloat16`); 消费核读进来直接是格点值, 一次舍都不做; 需要 f32 累加的在寄存器里升 | 值一个不变(bf16 格点 ↔ f32 存同一个值), 只去掉 35 处重复舍入与一半激活字节 ⇒ **逐位同**是门 |
| GEMM 输入 | f32 → 核内舍 bf16 | 预填: bf16 → 核内量化 NVFP4(已落地); 解码: bf16 直乘 | 同上 |
| KV 缓存(主 KV / 索引 K / SWA) | f32 | §5.1 官方: E2M1+E4M3/16 / MXFP4 / FP8 | 官方 QAT 格式, 逐位可对拍 |
| 路由 gate / hc 系数 / compressor / indexer 权重 | f32 | 先按 §2(f16/fp32), 再各自 A/B 到 FP4 | 用户令全 FP4; 每一行单独过五指标门 |
| engram wkv | f16 | FP8(官方) → A/B FP4 | 0.63 → 0.31 → 0.16 GB/token |
| 码本 | `__half` | 先去重(−2.0 GB 常驻), 再 A/B FP8 | 码本值域窄, FP8 大概率过门; 过了 shared 占用减半 |

## 3. 分段与门(10 分钟律; 不报时长; 每段独立可回滚)

| 序 | 段 | 做什么 | 门(分钟级) | 预期 |
|---:|---|---|---|---|
| **C0** | **死路清退** | ①`cuda_vq_prefill.inc.cu` f16 专家路: 把 nvfp4 路借的 `g_vqp` 缓冲池挪到 nvfp4 文件, 删文件; ②R4/R5 之外的 V4 CUDA 分片(`f16_shadow`/`q8_repack`/`q4k*`/`qk_warp*`/`moe_kernels*`/`moe_launch`/`attn_kernels*`/`indexer_kernels*`/`compressor*`/`embed_norm*`/`api_*`/`dspark`/`router`)从 `ds4_cuda.cu` 摘掉; ③`src/core` 的 V4 前向(`core_gpu_*`/`core_cpu_*`/`core_kern_*`/`core_attn`/`core_ffn*`/`core_moe`/`core_hc`/`core_sidecar`/`core_gpu_dspark`/`core_session_spec`)、`core_shape_select` 里 FLASH/PRO 两个变体; ④Metal 整目录 + Makefile 的 Metal 目标(Mac 只剩编辑源, 本来也只在 spark 跑); ⑤11 套测试里 V4 的 7 套删, 留离线 4 套 + 新加 V4.1 主尺套件; ⑥`.linecount-exempt` 掉 4 项; ⑦根目录 8 个小模块与 `ds4_posttrain` 按引用零的删 | `make test` 绿; 主尺 `prefill_2048_ruler.sh` **PPL 逐位同**; `d0a_decode_profile.sh` 中位 ms 不退 | 引擎 ≈75k 行 → ≈20k; 契约 186 → ≈45; 速度 0(它不该动) |
| **C1a ✅** | **原生精度直存(2026-09-15 落地)** | ★发现: 这几项的"非 FP4"是**我们自己胀出来的**★ —— HF 原件里 gate/compressor/indexer 本来就是 **BF16**, engram wkv 本来就是 **F8_E4M3 + 32×32 块缩放**, 是转换器把它们展开成 f32/f16。存回原格式 = 拿回官方精度, **字节减半且更准**(f16 的 10 位尾数装不下 e4m3×2^k 的全部取值, 老代码专门数 over/nan 就是防这个) | 主尺 PPL(见 fable5) | 每 token **−0.51 GB**: wkv 0.63→0.31, gate 0.31→0.157, compressor/indexer 0.088→0.044 |
| **C1b** | **A/B 压 FP4** | gate / hc_attn_fn / hc_ffn_fn / 码本各一个开关, 一次只开一个; 码本去重(三矩阵共一份) | 五指标(金融主尺 j 8192)每项 ≤ 噪声带; 退了就留 C1a 的格式 | 再 −0.42 GB/token; 常驻 −2.0 GB |
| **C2** | **运行时格式按官方** | ①`ds4_gpu_tensor` 加 dtype(C0 后只剩 45 个函数要认); ②激活缓冲改 bf16, 35 处 `v41_bf16r` 里"舍输入"的全删、"舍输出"的变成"存 bf16"; ③R2 六个 cuBLAS Sgemm 换成 fp4x32/f16 gemv(与骨架同核, 权重按 C1 的格式); ④R3 engram wkv 改走 V4.1 自己的 gemv, R5 的独立 round 核删; ⑤R4 预填 wo_a 走 NVFP4(分组 = 8 个独立小 GEMM, 与 `cuda_v41_nvfp4` 同一条路); ⑥KV 缓存三种按 §5.1 落格, 读侧核内解码 | ①②: **逐位同**(格点值没变); ③④⑤: 累加序变 ⇒ 主尺 NLL ≤ 噪声带; ⑥: 512 token 缓存逐位与官方 model.py 同 | 解码: gemv 内层每元素少一次舍(估 −10%) + 40 发 Sgemm 退(估 −3~5 ms) + engram 路省一趟(−1 ms); 预填: wo_a 少 5.4 GB/块 白流量; 1M: 每 token −1.1 GB, 内存 −4.2 GB |
| **C3** | **契约收口** | `ds4_gpu.h` 六子头合并成 V4.1 一份; `src/common` 的 V4 量化类型移到 `gguf-tools/legacy`(工具链的旧代金标还要); `ds4_fp8.h` 改名 `ds4_numfmt.h`(它装的是 E2M1/E8M0/E4M3 三种 FP4 组件, 名字骗人) | `make test` + `make -C gguf-tools tools-test` 绿 | 代码卫生; 以后加一个核不用再看 V4 |

**顺序为什么是这样**: C1 与 C0 互不依赖, 可以并行(C1 是转换器 + 五指标, C0 是删代码 + 逐位); C2 依赖 C0(dtype 要改契约)
和 C1(gemv 要认新格式); C3 依赖全部。速度只在 C1/C2 出, **C0 必须做到"主尺一个字节不变"**, 变了就是删错了。

## 4. 账(【估】, 每段起跑前先 nsys 出实账)

| | 今天(09-15 傍晚) | C1 后(§2 态) | C1 后(全 FP4 态) | C2 后 |
|---|---:|---:|---:|---:|
| 解码每 token 字节 | 6.3 GB | 5.8 | 5.4 | 5.4(1M 处再 −1.1) |
| 解码 ms/token(中位) | 69.1 | ≈63 | ≈59 | ≈50 |
| 解码 t/s | 14.5 | ≈16 | ≈17 | **≈20** |
| 墙(字节 ÷ 240 GB/s) | 26 ms / 38 t/s | 24 / 41 | 22.5 / 44 | 22.5 / 44 |
| 离墙 | 38% | 38% | 38% | 45% |

诚实一句: **这一整套清完, 解码从 14.5 到 ≈20, 离 40 的墙还有一半, 离 100 还要 DSpark。** 它不是主菜,
它是把盘子洗干净 —— 让 speed.md 段 2(K 切 block 级)和段 6(DSpark)在一个只有 FP4 的引擎上做, 不用每写一个核
先问"这个缓冲是 f32 还是 bf16 格点"。但每一项都有实数, 没有一项是"应该会快"。

## 5. 会怎么出错

- **C0 删多了**: 症状是编译过、主尺 PPL 变了。主尺逐位同是唯一的门, 变了一律回退那一刀重删。别信"这个函数肯定没人用"——
  `grep` 一遍调用者, 引擎 + server + agent + dist + tests + gguf-tools 六处都查(V4.1 借 V4 原语的就是这么漏的)。
- **C0 删了 `src/common` 的 V4 dequant**: `ds4_unit` 的七类型金标和 `gguf-tools/legacy` 的旧代解算器都会断。
  所以 common 的清理在 C3 而不是 C0, 且是"搬到 legacy"不是"删"。
- **C1 把 gate 压成 FP4 路由翻了**: 五指标门会抓(routing 翻 ⇒ NLL 跳)。抓到了就留 f16, 这是门的裁决, 不许调阈值。
  ★但先做 A/B 再下结论, §2 那句"位宽一动路由就变"没有一个数支撑。
- **C2 改激活 bf16 却"顺手"把 f32 累加也降了**: 那是改模型不是清遗留。舍入点一个不动(speed.md §2 末行), 门是逐位同, 不是"差不多"。
- **C2 的 KV 落格用错 swizzle/缩放粒度**: 与 speed.md §7 同一坑 —— 不报错, 只出"形状对数值错"。夹具 = 512 token 缓存逐位对官方。
- **把 engram 表(203 GB FP8)也当遗留清**: 那是官方盘上格式, 不是我们的选择; 改它 = 重做 203 GB, 且 §2 已定"原样"。
- **删 `ds4_fp8.h`**: 它装的是 FP4 的 scale(E8M0/E4M3)与 nibble 解码, 删了 FP4 就没了。它该改名, 不该删。
- **C0 与 speed.md 段 2/段 6 同时开工**: 一次一条长跑线; C0 是大删, 期间不许另一条线改同一批文件(合并地狱)。

## 6. 不做的 / 不变量

- 不动官方盘上格式(engram 表 FP8、engram 行 264 B)。
- 不为"全 FP4"降 f32 累加、不动 mHC/softmax/RMSNorm 的舍入点 —— 那是 speed.md §2 末行的不变量。
- 不做 CPU 路(V4 的 `core_cpu_*` 随 C0 走, 不补 V4.1 版); 不做 Metal V4.1。
- 底座 GGUF 不动; C1 产新文件, 旧文件留到过门; `-m` 指回即回滚。
- 判决只认参考前向尺五指标; 逐位同用 `prefill_2048_ruler.sh`; 解码速度用 `d0a_decode_profile.sh` **中位数**(平均被 kswapd 离群步糊掉, 09-15 撞过)。
- 零 Python 数值、零 env、单文件 ≤500 行、未验证不提交、发车先批准。
