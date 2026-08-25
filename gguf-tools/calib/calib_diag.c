/* calib_diag.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 802 行实现体的纯字节
 * 切割(切点在节注释边界, 不切断 #ifndef CALIBDIAG_TEST 配对), 按序 cat p1..p2
 * 与拆分前逐字节一致。符号/数值零变化。
 *
 * p1: 诊断数学核心(rows_mean/协方差/eff-rank/Δo metrics, 生产与合成自测共用)
 * p2: FULL HF driver(#ifndef CALIBDIAG_TEST 区) + -DCALIBDIAG_TEST 合成自测
 */
#include "calib_diag_p1.inc.c"
#include "calib_diag_p2.inc.c"
