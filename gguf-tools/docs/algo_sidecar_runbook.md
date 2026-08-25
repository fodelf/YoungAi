# 算法域侧车 runbook（2026-07-22 立项；遗留项③的执行入口）

目标: 治算法域缺口(12针仅1/3真体; Go 三针全败=Go域侧车同链顺带)。
机制: P2 插件工厂 — X 捕获 → 教师/学生参考 → z 四损失 RRR 逐层解 → corr 侧车热插拔(不动 base)。

## 已就绪
- [x] 语料: corpus/algo_calib.txt (TheAlgorithms 四语言 10 实现, ~1450 tok)
- [x] 捕获: M1:/tmp/capalgo raw_ffn_in_L0..L42 (973MB, capture_alllayers.sh MODEL=v2/v3 通用)
- [x] 判决器: corpus/algo_probes.txt 12针 + 基线报告 reports/algo_probes_v3_stack_2026-07-22.report
- [x] 求解器: z_v3_solve.sh (双机认领制) / obase_v3_m4.sh (学生 obase 双lane, RSS 看门狗@3.5G)

## 缺环(下次开工首查)
1. **ship/prep 步**: raw_ffn_in_L* (二进制) → /tmp/cap_v3prep 的 npy 三件套(ffn_in_L{L}.npy +
   route_w_L{L}.npy + ...)。工具在历史 v3prep 产线里(候选: cap_ef2.sh / onebit_calib.py 族 /
   M1 侧 ship 脚本) — 找到后固化为 scripts/cap_ship.sh。
2. **教师 o_ref/routed_L**: HF FP 前向逐层参考(双机各跑本机 shard 铁律, e5_balance 双指针工作窃取)。
   注意 obase_v3_m4.sh 头注: L24/L0-5 教师就绪前自动跳过。
3. 求解后: emit_z 汇集 → 侧车 gguf → svc.sh CORR= 挂载(两端都要) → algo_probes 12针 A/B(分钟级)。

## 铁律 checklist(启动前过一遍)
- 内存: obase lane RSS 看门狗@3.5G 已内建; z lane THREADS 调小可多 lane; 双机 ≤12G 红线。
- 判决前置: 侧车 A/B 用 12 针面板(对照腿=上面基线报告), 赢线=Go三针+全败针带至少 +3 针真体。
- 行为复核: 侧车挂载后必须重跑 behavior_gate(MAXTOK=192 口径) — 侧车不许伤工具帧(v2 反修教训同族)。
- 交接铁律: 侧车是外挂, 不动 base, 无删除风险; 但 CORR 两端同步(M1DIR 同名相对路径)。
