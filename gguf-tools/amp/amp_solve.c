/* amp_solve.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 559 行实现体的纯字节
 * 切割(切点在函数边界), 按序 cat p1..p2 与拆分前逐字节一致。符号/数值零变化。
 *
 * p1: 并行 for/cholesky/splitmix64+Box-Muller/f16 转换 + φ 提升 + 特征 matmul/
 *     ZtZ 累加/held 增量评估/rec worker 群
 * p2: main(乘性动态 z 放大器逐层闭式解算 CLI 驱动)
 */
#include "amp_solve_p1.inc.c"
#include "amp_solve_p2.inc.c"
