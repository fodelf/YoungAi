/* st_read.c — 转发 stub(2026-08-25 重构阶段2): 实现已升格为 src/common/ds4_st.c
 * (全仓唯一 safetensors 读器)。本文件保住 7 个"源文件 include"式消费方的路径,
 * 链接式转换见重构阶段7。新代码请直接 include src/common/ds4_st.c。 */

/* macOS 没有 posix_fadvise: ds4_st.c 的页缓存旁路是 Linux 专属 IO 优化(读完丢缓存),
 * Mac 上退化成 no-op —— 只影响读吞吐, 不碰任何数值。放在 include 之前, 免改 src/。
 * zlayer.c 在 include 本文件前已自带同款 shim(那里先定义了 POSIX_FADV_DONTNEED),
 * 所以用 !defined(POSIX_FADV_DONTNEED) 防重定义。 */
#if defined(__APPLE__) && !defined(POSIX_FADV_DONTNEED)
#define POSIX_FADV_DONTNEED 4
#include <sys/types.h>
static int posix_fadvise(int fd, off_t off, off_t len, int adv) {
    (void)fd; (void)off; (void)len; (void)adv; return 0;
}
#endif

#include "../../../src/common/ds4_st.c"
