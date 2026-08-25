/* calib_run.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 865 行实现体的纯字节
 * 切割(切点在函数边界), 按序 cat p1..p2 与拆分前逐字节一致。符号/数值零变化。
 *
 * p1: worker 上下文 + imatrix L_fix 统计 + 基线指标 + aggregate 分支前半
 *     (两阶段 split/shard-free sel dump 解算)
 * p2: run_rrr_layer 聚合流水(逐层) + main
 */
#include "calib_run_p1.inc.c"
#include "calib_run_p2.inc.c"
