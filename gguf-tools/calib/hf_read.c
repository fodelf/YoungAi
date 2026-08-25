/* hf_read.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 720 行实现体的纯字节
 * 切割(切点在函数边界), 按序 cat p1..p2 与拆分前逐字节一致。符号/数值零变化。
 *
 * p1: 数据结构 + 数值转换 + tiny JSON 扫描器 + 容器 + shard 发现
 * p2: shard 头懒解析(shard_load) + 张量查找/切片读取 + scale 配对 +
 *     -DHFREAD_TEST 自测
 */
#include "hf_read_p1.inc.c"
#include "hf_read_p2.inc.c"
