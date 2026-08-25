/* traj_analyze.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 614 行实现体的纯字节
 * 切割(切点在 PART 3 节注释边界), 按序 cat p1..p2 与拆分前逐字节一致。
 * 符号/数值零变化。
 *
 * p1: 小工具(6 元 id 集排序/交集) + route 加载 + PART 1-2(层内结构分析)
 * p2: PART 3 跨层路径结构(Jaccard/主专家一致率) + main
 */
#include "traj_analyze_p1.inc.c"
#include "traj_analyze_p2.inc.c"
