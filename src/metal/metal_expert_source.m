/* metal_expert_source.m — 原"expert source cache"(显式 RAM 专家缓存: mlock/
 * hard_copy/async-madvise LRU)整套子系统已删除: 在案负结果家族 —— 显式 RAM
 * 专家缓存饿死 page cache 净变慢(实测 3.84→1.84 t/s), 页面缓存本身就是更好的
 * LRU。文件保留为空桩, 避免打散 src/metal 的机械拆分文件清单。 */
#import "metal_internal.h"
