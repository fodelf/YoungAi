/* hiddenvar_solve.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 790 行实现体的纯字节
 * 切割(hv_solve 单函数 300 行不可再分, 切点在其后的节注释边界), 按序 cat p1..p2
 * 与拆分前逐字节一致。符号/数值零变化。
 *
 * p1: 确定性 dither PRNG + 四损失(classify/fixed/smooth/align) + hv_solve 分段闭式解
 * p2: hv_fidelity 重建保真(表 B) + z^ℓ 扁平文件序列化 + -DHVSOLVE_TEST 自测
 */
#include "hiddenvar_solve_p1.inc.c"
#include "hiddenvar_solve_p2.inc.c"
