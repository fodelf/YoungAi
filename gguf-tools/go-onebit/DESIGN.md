# go-onebit 模块化设计（R3-b，2026-07-03 草案 v1）

> 目标：把 R1/R2 长成的平铺代码收敛为清晰模块——量化（Θ_fix）、隐变量（z）、四损失求解、MTP 加速、双机流水、语料工厂——每模块单一职责、接口文档化、脚本与工具同域放置。运行时（ds4.c/ds4_metal.m）保持薄接口不搬家。

## 模块图

```
gguf-tools/go-onebit/
├── DESIGN.md                    ← 本文件（模块契约）
├── pipeline.conf                ← 工厂唯一配置入口
├── quant/                       【Θ_fix：1-bit 量化】
│   ├── onebit_quant.c/.h        编码器（go1b/go1b_blk + L_fix 加权 scale）
│   └── gen_go1b.sh              发射（--imatrix gostats，RSS 看门狗）
├── latent/                      【z 隐变量：求解与发射】
│   ├── solve_rrr.c/.h           四损失闭式求解器（RRR/L_align+L_smooth+L_fix钉+Q度量）
│   ├── hiddenvar_solve.c/.h     旧 hv 求解器（对照保留）+ z_layer 容器
│   ├── linalg_small.c/.h        Cholesky/topk-eig/（svd_small）
│   ├── emit_z.c                 z→corr GGUF（--phi、逐层 k_L、merge/check）
│   └── solve_delta.py           L_cls δ 闭式（教师/学生 route_logits 对齐）
├── calib/                       【标定数据面】
│   ├── calib_run.c              装配+求解驱动（--solver rrr/--dump-sel/--load-sel/--go-stats）
│   ├── calib_diag.c / validate_fwd.c / traj_analyze.c   诊断三件
│   ├── layer_probe.c/.h / hf_read.c/.h / npy.c/.h       原语
│   └── pyfwd/                   教师世界（numpy 直跑原始 HF）
│       ├── dsv4_fwd.py          教师前向+八信号 capture
│       ├── teacher_nll.py / teacher_inject.py / gen_go_stats.py
│       ├── student_traj.py      Python 学生轨迹（浅层判官；深层已知世界鸿沟）
│       ├── cap_raw2npy.py       引擎 capture raw→npy（R3-f 新增）
│       └── court_acc.py / check_capture.py / merge_caps.py
├── mtp/                         【投机加速】
│   ├── build_go_trie.py         语料→GTRI trie
│   └── gotrie_ab.sh             无损+配对判决
├── cluster/                     【双机流水】
│   ├── e5_solve_dual.sh / e5_pump.sh / e5_consume.sh（LOCAL 模式）
│   ├── e5_balance.sh / v2_finalize.sh / cap_v2_dual.sh / poll 系
│   └── 约束文档：M4=L0-11+无shard线代；M1=全shard；BatchMode/绝对路径/单命令铁律
├── corpus/                      【语料工厂】
│   ├── harvest_repos.py / corpus_build.py（难度路由）
│   ├── books/ method/           用户自备合法文本
│   └── build/                   产出（不进 git）
├── posttrain/                   【后训练（阶段7，桩）】
│   └── README：冻结Θ_fix、只训 z 侧车；门=还原底座达标
└── scripts/e2e_build.sh         【总编排】阶段门幂等
```

## 运行时接口（ds4 侧，保持薄）
- 侧车协议：`blk.{L}.corr_{U,V,C,b,beta,delta}` + KV `ds4.corr.{present,phi_yhat,dl.{L}}`；`--corr`/自动探测；φ=ŷ 绑定 routed_out；δ≡0 跳 dispatch
- 引擎轨迹 capture：`DS4_CAP_DIR`/`DS4_CAP_LAYERS`（批路径，raw f16/i16 shards）
- 投机：`--go-trie`/`DS4_GO_TRIE*`（默认关，复用 copy-spec 验证/回滚）

## 数据契约（跨模块）
- cap 目录 schema：`{ffn_in,route,route_logits,route_w,routed}_L{L}.npy` + `final_topk_{idx,val}.npy`
- sel 两段件：`sel_L{L}.bin`（RSEL 头 + Xsel/Yhat/Yref/route）
- z 件：`z_L{L}.bin`（i32×3 头 + U/V/C/b/β/δ f32）
- gostats：llama.cpp imatrix .dat（entry 名=GGUF 张量名，per-expert 段）

## 迁移施工令（一次 PR 一步）
1. mkdir 六模块目录 + git mv（不改内容）
2. Makefile 路径更新 + include 路径（相对 include 改 `-I` 根）
3. 全脚本内路径改经 `$GO_ROOT` 变量
4. 回归门：make 全目标 + go-onebit-test + e2_lane 单层冒烟 + emit --check
迁移窗口选在 R3-f 引擎轨迹采集的等待期执行。

## 已知债务（按优先级）
1. e5_consume ZDIR 默认值陷阱 → 改必填参数
2. 批打分路径 43 层 corr 开销（R3-e profile）
3. z 写入原子化（tmp+rename）
4. hv/perexpert/denoise/genrf 四个旧 solver 的去留（保留 hv 作对照，其余归档）

---
# v2 增补（2026-07-03）：以两日事故实证驱动的结构对策

| 病灶（实证事故数） | 对策 | 状态 |
|---|---|---|
| 脚本私有 env 默认值漂移（5） | `cluster/lib.sh` 单源：conf()/require()（无值即死，禁静默默认） | lib.sh 已落，接入=重解窗执行 |
| 裸 ssh/scp（挂死/agent 丢失，2） | lib.sh rsh/rcp（BatchMode+超时+fail-fast） | 同上 |
| 产物非原子写（z 丢/append 污染，2） | atomic_out/atomic_land（tmp+rename）；sel 已原子，z dump 与 δ patch 待接 | 待接 |
| 观测黑盒（3） | lib.sh progress()（stderr+耗时）；C 侧已带节流进度 | 部分落地 |
| calib_run 巨石（5 职责） | Phase-2 拆 driver/assembly/gostats-io；四个废 solver 归档 legacy/ | 排期 |
| 长任务×Monitor 连坐（1） | 规程：任务独立 nohup、Monitor 只读 | 已固化记忆 |

接入顺序：重解窗改 cluster 脚本头 → 回归（e2_lane 单层冒烟）→ Phase-2 C 拆分与迁移（同一 PR）→ make 全绿 + go-onebit-test。

---
# v3 增补（2026-07-03 下午）：runtime corr 双模式（速度结构修复）

**债务 #2 已结案**：批打分/解码路径的 corr 开销根因=微 dispatch 写 routed_out 的 hazard
管线气泡（非 kernel 计算，非 CB 数量——scratch 探针裁决），φ=ŷ 时 out==x 自别名加倍。

| 模式 | kernel | 写目标 | 何时启用 |
|---|---|---|---|
| in-place（旧） | kernel_dsv4_corr_apply | routed_out（+=） | prefill 批路径、TP/quality/keep_ffn_out、CUDA、CPU 镜像 |
| **delta（新默认）** | kernel_dsv4_corr_delta | g->corr_delta（store） | 单机 decode 且 fused shared-down 消费者运行时 |

- 消费点=kernel_dsv4_shared_down_hc_expand4_q8_0 内 `routed[d]+delta[d]`（同操作数
  同序 fadd → 与 in-place **位精确一致**；ds4_test metal-kernels 有 bitwise 断言）。
- 选择逻辑在 ds4.c decode（消费者谓词 hoist 到 corr 调度点之前）；`corr_delta_live`
  按层置位防 stale delta 泄漏到无 corr 层。
- 回退/对照钮：`DS4_CORR_INPLACE=1`（强制旧路径）、`DS4_CORR_SKIP=1`（载不调度，
  perf 定位用）。CUDA/CPU 端 `ds4_gpu_corr_delta_supported()==0` 永走旧路径。
- 免费旋钮判决（对 go1b 生效，token 逐字节一致）：`DS4_METAL_EXPERT_PREAD=1 +
  DS4_METAL_EXPERT_PREFETCH_AHEAD=1` = bare 0.45→1.00 t/s；已作 test_go1b/emit_and_run
  的默认 env（可覆盖）。
