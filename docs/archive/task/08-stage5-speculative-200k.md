# Task 08 — Stage 5：投机解码叠加 + 拉到 200K + 逼近 20 t/s

**状态**：未开始（收尾集成）
**前置依赖**：#06（全层 TP + 专家并行）、#07（Q5 重量化过质量门）
**类型**：双机 · 集成 + 最终测量

## 目标

在 TP+Q5 基线上叠加投机解码（MTP），把上下文逐档拉到 200K，逼近 20 t/s；
产出诚实的「配置 → sustained t/s @200K / 峰值 footprint / 质量分」对照表与上限结论。

## 实施步骤

1. **MTP 是否启用由实测决定**：
   - 若 #06+#07 实测已 **≥20 t/s** → MTP **可选**（启用要权衡 3.54GB 占用 vs ~1.8× 加速；
     on-host MTP 在 `mac` 分支现成）。不启用则把 3.54GB 让给专家容量。
   - 若 **未达标** → 启用 MTP，调 `--mtp-draft N`，并确认**热专家缓存覆盖投机窗口的专家并集**
     （否则验证 N 个草稿需加载并集，冷专家在 SSD 时被 I/O 卡死，失去稀疏收益——MoE-Spec/PowerInfer-2 点此）。
2. **上下文逐档拉**：32K → 100K → 200K，每档看门狗确认两机不破线（M4≤12 / M1≤8）；
   必要时 `--kv-disk-dir` 落盘非活跃 KV。
3. **最终对照表**：每个配置记 sustained gen t/s @200K（用 `ds4-bench` 前沿或 `DS4_PROFILE`）+ 峰值 footprint + `ds4-eval` 质量分。

## 备选路线（E0 #04 判 TP no-go 时）

- 单 M4 + Q5 backbone + on-host MTP + 小热专家集（M1 退化为纯 KV / 冷专家盘）；
  靠 MTP ~1.8× 把单机 ~20 t/s（注：单机 Q5 地板）拉到 ~30 effective。
- 该路线不依赖 all-reduce，规避 #04 的延迟风险。

## 闸门 / 验收

- **速度**：给出 sustained t/s @200K 实测；诚实标注是否过 20 t/s 铁律(不预先打包票，以实测为准)。
- **内存**：200K 全程峰值 footprint ≤12/8GB，无 OOM/panic（保内存 > 保速度）。
- **质量**：`ds4-eval q1..q4 --temp 0 --seed 1` + `ds4_test --logprob-vectors` 仍过。
- **正确性**：`make`(Metal) + `ds4_test --metal-kernels` + `ds4_test --server` 绿。

## 内存安全

- MTP draft 状态不跨 disk-checkpoint 持久化（已知）；启用 MTP 时把 3.54GB 计入 L1 预算闸重算。
- 200K KV 落盘走 `ds4_kvstore`（plain read/write 不 mmap，不新增 VM 映射）。
- 两机二进制同步;不删另一台机器文件;改完等确认再跑。

## 验证命令

```sh
# MTP 叠加（按 #03 标定的最佳 N）
# 200K 逐档拉，看门狗守线
./ds4-bench -m gguf/<final>.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 32768 --ctx-max 200000 --step-incr 32768 --gen-tokens 128 --csv /tmp/ds4-200k.csv
./ds4-eval -m gguf/<final>.gguf --plain --questions 4 --tokens 2048 --temp 0 --seed 1
./ds4_test --logprob-vectors && ./ds4_test --metal-kernels && ./ds4_test --server
```

## 产出

- 「配置 → sustained t/s @200K / 峰值 footprint / 质量分」最终对照表 + 诚实上限结论（项目收尾）。
