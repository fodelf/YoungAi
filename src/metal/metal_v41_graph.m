/* metal_v41_graph.m — 解码整步 graph 原语在 Metal 上的形态(2026-10-08; 契约 ds4_gpu_core.h, CUDA 实现 src/cuda/cuda_decode_graph.inc.cu)。
 * Metal 没有"流捕获 → 图实例重放"这一套: 捕获入口返回 0 ⇒ core(core_decode_graph.c)按直发路走, 每步照常逐核发射 —— 结果与 CUDA 直发路同口径。
 * 两个异步拷贝退成同步版(直发路在 synchronize 之后读结果, 语义仍成立); zerocopy 同理; host_device_ptr 返回原指针(统一内存, 主机缓冲本来就是设备可见的)。
 * 这不是"未实现": V4.1 的全部前向原语已在 metal_v41_*.m 落地, 这里只是没有图这一层加速。 */
#import "metal_v41.h"

int ds4_gpu_decode_graph_capture_begin(void) { return 0; }
void *ds4_gpu_decode_graph_capture_end(void) { return NULL; }
int ds4_gpu_decode_graph_launch(void *exec) { (void)exec; return 0; }
void ds4_gpu_decode_graph_free(void *exec) { (void)exec; }
int ds4_gpu_host_flag_wait(const void *flag_pinned, const void *want_pinned, void *err_pinned) { (void)flag_pinned; (void)want_pinned; (void)err_pinned; return 0; }
int ds4_gpu_tensor_write_async(ds4_gpu_tensor *t, uint64_t offset, const void *pinned, uint64_t bytes) { return ds4_gpu_tensor_write(t, offset, pinned, bytes); }
int ds4_gpu_tensor_read_async(void *pinned, const ds4_gpu_tensor *t, uint64_t offset, uint64_t bytes) { return ds4_gpu_tensor_read(t, offset, pinned, bytes); }
int ds4_gpu_tensor_write_zerocopy(ds4_gpu_tensor *t, uint64_t offset, const void *pinned, uint64_t bytes) { return ds4_gpu_tensor_write(t, offset, pinned, bytes); }
void *ds4_gpu_host_device_ptr(void *pinned) { return pinned; }
int ds4_gpu_tensor_read_zerocopy(void *pinned, const ds4_gpu_tensor *t, uint64_t offset, uint64_t bytes) { return ds4_gpu_tensor_read(t, offset, pinned, bytes); }
