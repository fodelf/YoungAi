/* solve_rrr.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 802 行实现体的纯字节
 * 切割, 按序 cat p1..p3 与拆分前逐字节一致。符号/数值零变化。
 * 注: 原计划 2 片, 但 rrr_solve 单函数横跨 293..718(426 行), 任何 2 片切法都会
 * 超 500 行/片或切进函数体 —— 按"切点必须在函数边界"改 3 片。
 *
 * p1: parallel_for + gram/trisolve/fit/fcov worker 群 + Q^{±1/2} Woodbury 施加器
 *     + 保真指标(agg_metrics/token_corr)
 * p2: rrr_solve 聚合级四损失闭式解算(整函数)
 * p3: -DRRR_TEST 合成恢复自测
 */
#include "solve_rrr_p1.inc.c"
#include "solve_rrr_p2.inc.c"
#include "solve_rrr_p3.inc.c"
