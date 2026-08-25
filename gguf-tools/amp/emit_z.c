/* emit_z.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 807 行实现体的纯字节
 * 切割(切点在 writer 节注释边界), 按序 cat p1..p2 与拆分前逐字节一致。
 * 符号/数值零变化。
 *
 * p1: 文件头(双输出模式说明) + GGUF 读侧/corr 张量装配
 * p2: writer(sidecar/merge 两路写出) + main
 */
#include "emit_z_p1.inc.c"
#include "emit_z_p2.inc.c"
