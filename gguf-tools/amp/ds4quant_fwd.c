/* ds4quant_fwd.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 682 行实现体的纯字节
 * 切割(切点在函数边界), 按序 cat p1..p2 与拆分前逐字节一致。符号/数值零变化。
 *
 * ★路径契约★ 本文件被 4 个 TU 以 "ds4quant_fwd.c" 引号 include(ds4quant_run.c/
 * ds4quant_layer.c/probe/z_explore.c/z_explore2.c), 聚合根路径名不得改; 分片
 * 引号 include 按本文件所在目录(amp/)解析, 对所有引用方一致。
 *
 * p1: 数值原语(rms/softmax/matmul 族含 strided/NT 版) + hc_sinkhorn/hc_pre/hc_post
 * p2: gate_route 路由(hash 与 L3+ 两路) + 专家前向/逐层驱动 + DS4QUANT_SELFTEST
 */
#include "ds4quant_fwd_p1.inc.c"
#include "ds4quant_fwd_p2.inc.c"
