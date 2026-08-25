# Python 盘点（迁移工作清单基线, 2026-08-25 探索 agent 产出）

勘误: 本盘点扫描于阶段0回传之前 —— "9个C端口不存在"一节已失效(commit bc7a181 已回传
calib/{amp_solve,anchor_metrics,cap_raw2npy,dilution,ge_solve,hbase,hsolve,st_bridge,
zchain_merge}.c + teacher_routed.c + vq_em.cu)。其余结论有效。
进度: Wave A 已完成(过金标闸+删py): anchor_metrics / kl_forensic / trace_ladder / rec_fidelity。

## 汇总

| 类别 | 文件数 | 行数 |
|---|---|---|
| A 数值算法 (numpy/torch/纯py求解/量化/指标) | 144 | 16014 |
| B 编排/搬运 | 56 | 9792 |
| C 一次性诊断/报表 | 23 | 2323 |
| 合计 | 223 | 28129 |

分类口径: pack_stream.py / vq_merge_v4.py 这类"搬字节顺路 dequant/烘焙 float"算 A;
ops_to_zchain.py / dql_strip.py 纯格式搬运算 B。distill4/gen_train/train_codec/
proper_distill 是 `import os, numpy as np, torch` 一行多模块写法(torch+MPS 训练), 别漏判。

## 真实存在的 Python↔C 声明对

| Python | C | 备注 |
|---|---|---|
| calib/pyfwd/dsv4_fwd.py (300) | quant/ds4quant_fwd.c | "数值原语逐一对应"; 被 28 个 .py import(地基) |
| calib/pyfwd/ds4reader.py (93) | quant/st_read.c | e4m3 LUT + 块 scale dequant; 被 26 个 .py import |
| quant/go2b_encode.py (130) | quant/go2b_qc.h + go2b_parity.c | 逐行移植+对拍探针 |
| scripts/g1c_vq_sweep.py (99) | quant/vq_shim.c | 编码链同口径 |

★未声明重复实现(分歧高危): quant/corr_fit.py ↔ tools/zsolve.c(同做 corr 侧车闭式拟合,
双方注释互不知晓) —— Wave B 处置: 判谁是真源, 另一个删。

互补消费(非取代, 迁移时把"产格式"并进消费者或转bash): gguf_offsets.py→ds4quant_run.c;
make_expert_mask.py/shrink_gguf.py→deepseek4-quantize.c(DSXM); quant_assemble.py;
cap_raw2npy.py; gen_pinned.py→ds4_metal.m。

## 高优先级(算法铁律序)

1. ~~anchor_metrics.py~~ ✅(23 处消费方; 另 3 个 .sh heredoc `from anchor_metrics import` 待 Wave C 清)
2. zlayer.py (672, cupy) — 反修主力, 10 个 .sh 调用 — Wave B 主项
3. amp_solve.py (129) — C 版已回传, 需金标对拍+接线
4. vq_merge_v4.py (453) — 最热产物合并器(11 个 .sh), 懒加载 numpy+烘焙路由偏置
5. probe_layer_behavior.py(125)/probe_behavior_spectrum.py(108) — 19/14 个 .py import 的底座库
6. dsv4_fwd.py/ds4reader.py — C 已存在, 收编 import 者后删

## 关键风险面

- **72 个 .sh 内嵌 python3 heredoc**; 其中 p2_truncprobe.sh / q4t_teacher_spark.sh /
  r60_kl_probe.sh 直接 `from anchor_metrics import ...` 现算指标(算法藏在 .sh 里)。
- **悬空引用 4 处**(现在就是坏的): preflight_skeleton.sh→r36_rebake_bias.py(已改名
  route_bias_rebake.py); r29_campaign.sh→skel_from_hf.py(只有 .sh); cluster/cap_v2_dual.sh→
  capture_go_big.py; tools/make_small_mtp.sh→_filter_mtp.py。
- **零引用 60 个文件(27%)**(无 .sh/无 import/无 .md): 含 nlz_ladder.py(454)、
  dspark_replay.py(494) 等大件 — Wave D 按用户裁决仍 C 化, 但排最后且金标多不可得(如实标注)。
- 只在 fable5.md 留痕的手跑针 79 个: 非死码, Wave D 中段。

## 目录明细(行数|类|C对应|调用方)

### zlever/ 24 个 2544 行 (A20)
zlayer 672 A 无 10个.sh | nlz_ladder 454 A 无 零引用 | l1pipe 201 A 无 fable5 |
dspark_amp_fit 188 A 无 dspark_amp*.sh | full10 164 A 无 零引用 | ladder 158 A 无 fable5 |
amp_solve 129 A C已回传 4个.sh | solve 125 A 无 8个.sh | route_solve 107 A 无 rsolve43.sh |
rgate 91 A 无 rgate12_spark.sh | amp_diag 87 A 无 fable5 | dynprobe 86 A 无 零引用 |
route_probe 84 A 无 零引用 | mkzchain 73 A 无 fable5 | amp_probe/3/2 71/70/53 A 无 fable5 |
wproj 66 A 无 零引用 | layerz 63 A 无 fable5 | datascale 32 A 无 fable5 |
zrec_to_zchain 42 B 无 8个.sh | zrollback 36 B 无 fable5 | dql_to_zchain 30 B 无 4个.sh |
pop_layer 22 B 无 r30_campaign.sh
(kl_forensic/rec_fidelity/trace_ladder 已迁移删除)

### calib/pyfwd/ 21 个 1451 行 (全A)
dsv4_fwd 300(C有)/ds4reader 93(C有)=地基; 有.sh调用: cap_raw2npy 38(6个)/teacher_inject
75(6个)/merge_caps 21(2个); 零引用9个: full_stack_test 90/gen_eval 44/gen_train 63/
go_input_rank 30/go_quality_test 56/lowrank_resid_test 38/magnitude_predict_test 45/
precompute_obase 37/train_codec 76; 其余 fable5/DESIGN 留痕: student_traj 114/teacher_nll 57/
fork_teacher 65/residual_test 33/oref_on_student 31/check_capture 45/court_acc 13

### quant/ 26 个 2913 行 (A23 B3)
vq_merge_v4 453 A 无 ★11个.sh★ | gptq1_rewrite 219 A 无 v3_gptq1.sh+9py |
gr_refit_layer 155 A 无 l42_analysis | gptq1_pack_compute 139 A 无 m4_lane.sh |
l42_analysis 136 A 无 零引用 | go2b_encode 130 A C有 | build_monolithic/_ef 122/105 A 无 |
gen_go_stats 111 A 无 e2e_build.sh | corr_fit 99 A ★疑似zsolve.c重复★ fable5 |
ef_layer 98 A 无 3个.sh | d2_probe 96/build_go2b_hot 95/layer_truth 92/deep_survey 85/
pack_layer 84/build_go2b_sidecar 83/build_go2b_combined 76/obase_v3 69/scale_fold 53/
fix_l24_requant 47/splice_signs 37/pack_stream 32 A 多数零引用 |
vq_merge_gguf 222 B 无(被v4取代) | merge_sidecars 76/splice_go2b_inplace 55 B ef_grind.sh

### scripts/ 114 个 12127 行 (A76 B21 C17)
锚/指标A: bpw_audit 196(零)/anchor_hotcurve 108/bitexact_check 92/cap_to_anchor 45/
engine_smin 35(零)/anchor_top_experts 29
体积求解A(纯py): rplan_solve_r30 305/r29 291/v4 285/无印 152/v3 136 ★同事5版本并存★
路由A: route_alpha_set 108(7个.sh)/probe_router_agree 70/route_concentration 66/
route_bias_rebake 61(4个.sh)/fork_route_diff 34
行为谱底座A(全靠import): probe_layer_behavior 125(19py)/probe_behavior_spectrum 108(14py)/
probe_layer_z_validate 78
g系列A: g1a_lever_probe 248(8py)/g1b_stack_probe 177(7py)/g4_bpw01_ladder 154/
g3_layer3_validate 133/g4e_w23_spot 124/g4c_crossstyle_probe 120/g4d_layer_minbit 120/
g4b_manifold_ladder 104/g1c_vq_sweep 99(C有)
dspark对拍A: dspark_replay 494(零)/dspark_refcheck 233(零)
1-bit算法针A(~40个): full_optimize 273/scale_compute 203/seq_optimize 200/layer_design 199/
scale_patch 180/coadapt_layer 161/residual_rd 161/onebit_ef 133/onebit_rotate 131/
scale_apply 130/probe_error_spectrum 127/dsml_scale_test 125/dcoef_eval 122/onebit_iter 112/
restore_rate 98/metric_leniency 93/free3part_poc 91(torch)/probe_z_structure 90/
onebit_calib 87/emit_coding_imatrix 86/probe_condz 81/dsml_oref 79/verify_ampd_kernel 74/
probe_subspace_share 73/probe_klsolve 72/dsml_sidecar_* 59/63/58/p2_* 57/46/34/34/
vq_gateup_refcheck 57/logit_attr 56/gen_pinned 46/gen_active_from_anchor 37/
corr_reconstruct_check 33/drift_output_check{,_all} 29/34/gen_active_topk 23/
go2b_overlay_from_sidecars 75
编排B: pubbench 438(7个.sh)/dsml_verify 141/plan_json2rplan 129/preflight_r30 128/
dsml_corpus_gen 92/r30_plan_champ 75/go2b_backbone_extract 75/ops_sidecar_to_main 67/
ops_to_zchain 61/gguf_offsets 61/vq_overlay_from_sidecars 60/g4c_cells 58/
vq_blob_truesize 54/dql_strip 52/dql_down_offset 50/r30_blob_patch 48/punch_holes 44/
corpus_gate 44/build_cal8/9/10 44/40/36/dsml_cap_trim 37/sparsify_zeros 30/gen_zchain_test 17
报表C: v4bf_layer_metrics_doc 137/skel_bias_probe 129/r30_layer_table 116/skel_breakdown 111/
bf_metrics 95/r29_vs_r28 80/r29_evidence 79/layer_detail 77/gen_tables 71/r30_report 63/
r30_evidence_mix 56/pick_active_experts 49/p4_budget 47/rb_inspect 45/g2_compare 38/
p4_window 29/p4_kernel_ms 23

### 其余
imatrix/dataset/build_ds4_imatrix_dataset.py 2230 B(单文件最大, 仅README) |
shrink_gguf 655 B | mixed/splice_mixed 404 B | router_norms_from_imatrix 337 B |
quality-testing/collect_official 266 B(score_official.c互补)/compare_scores 88 C |
make_expert_mask 171 B | split_gguf_layers 123/balanced_split 85 B | dump_gguf_meta 71 C(零引用) |
layer_sizes 62 C | make_firstk_mask 42 B | go-onebit根: syntax_routing 79/distill4 75/
proper_distill 74/probe_func 45 A 全零引用(torch训练+sklearn) | split_corpus_halves 48 B |
corpus/: make_calib_prog_v5 166/harvest_repos 109/harvest_issue_fixes 76/corpus_build 66/
harvest_skills 49 B | cluster/: merge_go2b 129 A(零)/gguf_range_stream 90/quant_assemble 54 B |
latent/: train_z_sentinel 68/solve_delta 36 A(同目录 solve_rrr.c/hiddenvar_solve.c 为原生C非移植) |
mtp/build_go_trie 179 B | benchmarks/go-bench/run_bench 808/react-bench/run_bench 636 B |
dir-steering/build_direction 229/run_sweep 64 B | speed-bench/plot_speed 227 C |
tests/fetch_official_vectors 277 B/generate_long_context_story_prompt 196 B(零) |
tools/build_mtp_template 68 B(零, MTP已删) | competition/build_doc 437/make_figs 193 C
