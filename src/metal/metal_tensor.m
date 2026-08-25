/* metal_tensor.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (!g_initialized && !ds4_gpu_init()) return NULL;
    if (bytes == 0 || bytes > (uint64_t)NSUIntegerMax) return NULL;

    @autoreleasepool {
        DS4MetalTensor *tensor = [DS4MetalTensor new];
        tensor.buffer = [g_device newBufferWithLength:(NSUInteger)bytes
                                              options:MTLResourceStorageModeShared];
        if (!tensor.buffer) {
            return NULL;
        }
        tensor.offset = 0;
        tensor.bytes = bytes;
        tensor.owner = 1;
        g_tensor_alloc_live_bytes += bytes;
        if (g_tensor_alloc_live_bytes > g_tensor_alloc_peak_bytes) {
            g_tensor_alloc_peak_bytes = g_tensor_alloc_live_bytes;
        }
        if (ds4_gpu_trace_allocs()) {
            fprintf(stderr,
                    "ds4: Metal tensor alloc %.3f MiB live %.3f MiB peak %.3f MiB\n",
                    (double)bytes / (1024.0 * 1024.0),
                    (double)g_tensor_alloc_live_bytes / (1024.0 * 1024.0),
                    (double)g_tensor_alloc_peak_bytes / (1024.0 * 1024.0));
        }
        return (__bridge_retained ds4_gpu_tensor *)tensor;
    }
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    return ds4_gpu_tensor_alloc(bytes);
}

int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    (void)kv_cache_bytes;
    (void)context_bytes;
    return 0;
}

ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base) return NULL;
    const DS4MetalTensor *base_obj = ds4_gpu_tensor_const_obj(base);
    if (offset > base_obj.bytes || bytes > base_obj.bytes - offset) return NULL;
    if (base_obj.offset > UINT64_MAX - offset) return NULL;
    const uint64_t absolute_offset = base_obj.offset + offset;
    if (absolute_offset > (uint64_t)NSUIntegerMax) return NULL;

    @autoreleasepool {
        DS4MetalTensor *view = [DS4MetalTensor new];
        view.buffer = base_obj.buffer;
        view.offset = absolute_offset;
        view.bytes = bytes;
        view.owner = 0;
        return (__bridge_retained ds4_gpu_tensor *)view;
    }
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    @autoreleasepool {
        DS4MetalTensor *obj = (__bridge_transfer DS4MetalTensor *)tensor;
        if (obj.owner) {
            if (obj.bytes <= g_tensor_alloc_live_bytes) {
                g_tensor_alloc_live_bytes -= obj.bytes;
            } else {
                g_tensor_alloc_live_bytes = 0;
            }
            if (ds4_gpu_trace_allocs()) {
                fprintf(stderr,
                        "ds4: Metal tensor free %.3f MiB live %.3f MiB peak %.3f MiB\n",
                        (double)obj.bytes / (1024.0 * 1024.0),
                        (double)g_tensor_alloc_live_bytes / (1024.0 * 1024.0),
                        (double)g_tensor_alloc_peak_bytes / (1024.0 * 1024.0));
            }
        }
        obj.buffer = nil;
        obj.offset = 0;
        obj.bytes = 0;
        obj.owner = 0;
    }
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    if (!tensor) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return obj.bytes;
}

void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    DS4MetalTensor *obj = ds4_gpu_tensor_obj(tensor);
    return (uint8_t *)[obj.buffer contents] + obj.offset;
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > ds4_gpu_tensor_bytes(tensor) / sizeof(float)) return 0;
    float *p = ds4_gpu_tensor_contents(tensor);
    if (!p && count != 0) return 0;
    for (uint64_t i = 0; i < count; i++) p[i] = value;
    return 1;
}

int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    DS4MetalTensor *obj = ds4_gpu_tensor_obj(tensor);
    if (offset > obj.bytes || bytes > obj.bytes - offset) return 0;
    if (bytes != 0) {
        memcpy((uint8_t *)[obj.buffer contents] + obj.offset + offset, data, (size_t)bytes);
    }
    return 1;
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    if (offset > obj.bytes || bytes > obj.bytes - offset) return 0;
    /* Host-read boundary: async-submitted CBs (corr sidecar) may still be
     * writing this buffer. Reading GPU memory with in-flight writes is a
     * race — drain pending before the memcpy (no-op when the list is empty,
     * which is the steady decode state: end_commands already swept it). */
    if ([g_pending_cbs count] != 0)
        (void)ds4_gpu_wait_pending_command_buffers("tensor read");
    if (bytes != 0) {
        memcpy(data, (const uint8_t *)[obj.buffer contents] + obj.offset + offset, (size_t)bytes);
    }
    return 1;
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes) {
    if (!dst || !src) return 0;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    DS4MetalTensor *d = ds4_gpu_tensor_obj(dst);
    const DS4MetalTensor *s = ds4_gpu_tensor_const_obj(src);
    if (dst_offset > d.bytes || bytes > d.bytes - dst_offset) return 0;
    if (src_offset > s.bytes || bytes > s.bytes - src_offset) return 0;
    if (bytes == 0) return 1;
    if (!g_batch_cb) return 0;

    ds4_gpu_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [g_batch_cb blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:s.buffer
            sourceOffset:(NSUInteger)(s.offset + src_offset)
                toBuffer:d.buffer
       destinationOffset:(NSUInteger)(d.offset + dst_offset)
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    return 1;
}

int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count) {
    if (!dst || !src) return 0;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    DS4MetalTensor *d = ds4_gpu_tensor_obj(dst);
    const DS4MetalTensor *s = ds4_gpu_tensor_const_obj(src);
    if (count == 0) return 1;
    if (count > UINT64_MAX / sizeof(float) ||
        count > UINT64_MAX / sizeof(uint16_t)) {
        return 0;
    }
    const uint64_t src_bytes = count * sizeof(float);
    const uint64_t dst_bytes = count * sizeof(uint16_t);
    if (src_offset > s.bytes || src_bytes > s.bytes - src_offset ||
        dst_offset > d.bytes || dst_bytes > d.bytes - dst_offset) {
        return 0;
    }

    @autoreleasepool {
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        uint64_t done = 0;
        int ok = 1;
        while (done < count && ok) {
            uint64_t chunk64 = count - done;
            if (chunk64 > UINT32_MAX) chunk64 = UINT32_MAX;
            const uint32_t chunk = (uint32_t)chunk64;
            ok = ds4_gpu_encode_cpy_f32_f16_1d(
                    cb,
                    s.buffer,
                    (NSUInteger)(s.offset + src_offset + done * sizeof(float)),
                    d.buffer,
                    (NSUInteger)(d.offset + dst_offset + done * sizeof(uint16_t)),
                    chunk);
            done += chunk;
        }
        if (ok) ok = ds4_gpu_finish_command_buffer(cb, owned, "tensor f32 to f16 copy");
        return ok;
    }
}
