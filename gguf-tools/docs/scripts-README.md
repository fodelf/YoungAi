# go-onebit 编排脚本

go1b（严格 1-bit 路由专家）+ z 隐变量（四损失闭式）端到端流水线的编排脚本。
Python 前向/质量脚本在 `../pyfwd/`；C 工具（calib_run / emit_z / deepseek4-quantize）在 `..` 与 `../..`。

## 双机集群（M4 mini coordinator + M1 MacBook，各 16 GiB，≤12 GiB 预算）

- **Shard locality（硬约束）**：每机只能算它本地有专家 shard 的层。原始 HF 46 shard 大致 1 shard ≈ 1 层（shard N ≈ 层 N-1）。M4 默认有 shards 1-13（层 0-12），M1 有 14-46（层 13-42）。**M4↔M1 的 NFS 不可靠（会变 stale EPERM）**，所以跨机层必须把对应 shard 物理拷过去。
- **空闲机器加入剩余工作**：要让 M4 多做层，先腾空间（临时删 M4 的 go1b GGUF，M1 有副本可 recopy）→ 拷 shards 14-X + 对应 cap `ffn_in_L*.npy` 到 M4 → 均衡切分（如 M4 做 0-21、M1 做 22-42，各 ~22 层一起完成）。
- **harness 会杀长后台命令**：用 detached 启动（子 shell `( nohup ./calib_run … & )`，reparent 到 init，扛得住 wrapper 被杀）；进度用快命令轮询（`poll_z.sh`），每次顺手查 RSS 当看门狗。
- **内存安全（铁律）**：calib_run nx=256 峰值 ~2-6 GiB（安全）；nx=512 ~13 GiB（贴线）。watchdog @13 GiB。

## 流水线

1. **生成 go1b 模型** `gen_go1b.sh <HF> <TEMPLATE_GGUF> <OUT.gguf>`
   - 模板只需 GGUF 头部（元数据+tensor 结构）；权重从 HF 重量化。可从 published GGUF range-下载头部当模板（见脚本注释）。产物 ~45.6 GiB。
2. **算每层 z**（四损失闭式解，逐层独立）`z_compute_dual.sh`（双机均衡）或 `calib_split.sh`（单机/单段）
   - calib_run 加了 z-dump：算完 hv_solve 存 `z_L{L}.bin`（`[i32 d_model,n_exp,d_l][f32 U,V,C,b,beta,delta]`）。
3. **emit corr sidecar** `../emit_z --out gguf/ds4-go1b-corr.gguf --zdir zdump --layers 43`
   - 把 43 层 z 写成独立 ~90 MiB sidecar GGUF（`blk.{L}.corr_{U,V,C,b,beta,delta}` + `ds4.corr.present` KV）。不碰 45.6 GiB 主模型（磁盘安全）。
4. **运行** `./ds4 -m gguf/ds4-go1b.gguf --corr gguf/ds4-go1b-corr.gguf -p "..." --temp 0 --metal`
   - ds4 加载主模型 + sidecar，每层 MoE 加 `o += Σ_e U·diag(C_e)·(V·x) + n_sel·b + Σβ_e`，δ 修 router logit。x = post-RMSNorm FFN 激活（cap 的 ffn_in 同口径）。无 sidecar 时 = 纯 1-bit 原路径。

## 退化循环 ≠ bug

go1b temp=0 贪心会掉进重复吸引子（弱模型 + greedy）；非 runtime bug（experts 数值 0.0006 已验）。用 `DS4_REPEAT_FREQ=3`（repeat penalty）或 temp>0 治退化。1-bit 质量地板：连 "2+2=4"/"capital of France" 都生成不出 → 真能用需残差到 2-bit（见 `../pyfwd/residual_test.py`，端到端 0.778）。
