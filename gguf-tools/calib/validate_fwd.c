/* validate_fwd.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 922 行实现体的纯字节
 * 切割(切点在节注释边界), 按序 cat p1..p2 与拆分前逐字节一致。符号/数值零变化。
 *
 * p1: 路由几何/数值原语(sqrtsoftplus/cos/rel-L2) + token 并行收集 worker +
 *     前向验证与 1-bit 抵消主流程
 * p2: GO-AWARE 1-bit scale test(闭式 1-D 最小二乘节) + 聚合校正闸 + main
 */
#include "validate_fwd_p1.inc.c"
#include "validate_fwd_p2.inc.c"
