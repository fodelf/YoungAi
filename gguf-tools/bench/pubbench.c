/* pubbench.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 1871 行实现体的纯字节
 * 切割(切点在节注释/函数边界), 按序 cat p1..p4 与拆分前逐字节一致。符号/数值
 * 零变化。gates/pubbench_extract_test.c 仍 #include "../pubbench.c" 整体复用。
 *
 * p1: 文件头/基础设施(sb/str/py 字符串语义) + 最小 JSON
 * p2: HTTP(libcurl)/gunzip + 数据集拉取缓存 + 正则等价件(金标最密区起点)
 * p3: extract_completion 抽取器 + 子进程执行 + 判定器 + call_server + run_suite 前半
 * p4: run_suite 后半 + rejudge + verdict/compare + selftest + main
 */
#include "pubbench_p1.inc.c"
#include "pubbench_p2.inc.c"
#include "pubbench_p3.inc.c"
#include "pubbench_p4.inc.c"
