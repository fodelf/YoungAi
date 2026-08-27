/* ds4quant_run.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 5800 行实现体的纯字节
 * 切割, 按序 cat p1..p14 与拆分前逐字节一致。符号/数值零变化。
 *
 * ★include 契约★ 自身的 .c include 群(../calib/st_read.c / ds4quant_fwd.c /
 * ../quantize/onebit_quant.c / ds4_z.c / ds4_loss.c 等, 原 24-41 行)全在 p1
 * 原字节位不动; 编译仍要求 -I..(仓库根解析 ds4_z.c/ds4_loss.c)。
 * ★片数说明★ 计划 12 片左右, 实际 14: coadapt_moe(544 行)/layer_fwd(505 行)/
 * fwd_all_tune(534 行)三个单函数各超 500, 各需一处体内切点(深度扫描过的
 * 顶层语句/阶段注释边界: 2109/2998/4070/4561), 其余切点全在函数边界。
 *
 * p1 : 头注释/include 群 + 全局参数与记账 + 模块日志/表格 + 层配置/加载(LW/LWH)
 *      + zpar_for + z_solve_dual/z_pick_rank/z_solve_fourloss
 * p2 : FP 锚定(anchor) + 专家量化/应用(quant_apply/expert_worker) + coadapt 前置
 * p3 : co_apply_mult/co_eval + 合并动态侧车(zfile_write/co_base_pass/payload)
 *      + 层文件重前向(lfile_load) + DQZ2 全链侧车(zchain_write)
 * p4 : bytes_moe(字节回放 MoE) + 序贯路由 RB + mv 体积记账 + append_rec
 * p5 : co_rounds + coadapt_moe 前半(候选轮询)
 * p6 : coadapt_moe 后半(胜者 payload/BWD/元素叠加) + 锚 rowmap + bf_fp_routed
 * p7 : layer_fwd 主体(锚捕获/路由/专家量化前向/z 段)
 * p8 : layer_fwd 收尾 + head_fwd(+stream) + bwd KL/solve_sym + fwd_all
 * p9 : GGUF 直写导出(export_worker/export_layer_file)
 * p10: export_gguf + 逐层渐进调优基建(CANDS/ckpt/plan/rr 锚) + fwd_all_tune 序幕
 * p11: fwd_all_tune 主循环(逐层贪心 L 循环整体)
 * p12: fwd_all_tune 收尾 + verdict + 全局联合回扫(backfit_layer_z/joint_round 等)
 * p13: backfit_prev(跨层反修主体)
 * p14: 冷专家修复 + global_sweep + main
 */
#include "ds4quant_run_p1.inc.c"
#include "ds4quant_xcap.inc.c"
#include "ds4quant_anchor.inc.c"
#include "ds4quant_run_p2.inc.c"
#include "ds4quant_run_p3.inc.c"
#include "ds4quant_run_p4.inc.c"
#include "ds4quant_run_p5.inc.c"
#include "ds4quant_run_p6.inc.c"
#include "ds4quant_run_p7.inc.c"
#include "ds4quant_run_p8.inc.c"
#include "ds4quant_run_p9.inc.c"
#include "ds4quant_run_p10.inc.c"
#include "ds4quant_run_p11.inc.c"
#include "ds4quant_run_p12.inc.c"
#include "ds4quant_run_p13.inc.c"
#include "ds4quant_run_p14.inc.c"
