/* zlayer.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 2554 行实现体的纯字节
 * 切割, 按序 cat p1..p7 与拆分前逐字节一致。符号/数值零变化。
 *
 * ★include 契约★ 三个 .c include 全在分片内原字节位不动:
 *   ../calib/st_read.c(原 88 行→p1), ../../src/common/ds4_quantfmt.c 与
 *   ds4_gguf.c(原 716-717 行→p2, GGUF 标量模式节内, 位置敏感勿移)。
 * ★片数说明★ 计划 6 片, 但 main 单函数横跨 1116..2554(≈1439 行), 函数边界切法
 * 不可能 ≤500 行/片; main 体内仅 1286/1699/1885/1921/2272 五处顶层语句边界
 * (深度扫描 brace=1/paren=0/#if=0/非注释), 其中 1699..2272 段 573 行必须再切
 * 一刀 —— 故 7 片, main 内切点全取阶段注释行。
 *
 * p1: 兼容 shim + 小助手/parallel_for + f16 + MT19937 + mm64/mm32 + Cholesky/LU
 * p2: Jacobi 特征分解 + npz 读写 + VQ 侧车反量化 + GGUF 标量模式(含上述两 include)
 *     + 锚(DQA2)读取 + swiglu/φ 提升
 * p3: 低秩 SVD + Householder QR + 单专家前向 + XCAP 读取 + rec 头/区间/env
 *     + main 前段(参数/装载)
 * p4: main: GGUF 标量模式装载 + 非 XCAP 学生重算/PREV 中段
 * p5: main: 活 k_L held 段 k 曲线 + 分域终验
 * p6: main: ERF 死层部件
 * p7: main: 注入载荷 + 写出收尾
 */
#include "row_layout.inc.c"   /* 行布局→分层行选取(与 ds4quant_run 共用一份); 必须在 p1 之前=文件作用域 */
#include "zlayer_p1.inc.c"
#include "zlayer_p2.inc.c"
#include "zlayer_build.inc.c" /* 配对构建唯一实现(主解算+跨语料闸两处同源); p3 起进 main, 须在此前 */
#include "zlayer_p3.inc.c"
#include "zlayer_p4.inc.c"
#include "zlayer_p5.inc.c"
#include "zlayer_p6.inc.c"
#include "zlayer_p7.inc.c"
