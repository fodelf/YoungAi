/* Ordered CPU snapshots. A page is reusable only after its last reader finishes.
 * Like the rest of this backend, queue submission is externally serialized. */
#import "metal_internal.h"

@interface DS4MetalUploadPage : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, strong) id<MTLCommandBuffer> last;
@property(nonatomic) NSUInteger used;
@end
@implementation DS4MetalUploadPage
@end

static NSMutableArray<DS4MetalUploadPage *> *g_upload_pages;

void ds4_gpu_upload_pool_cleanup(void) { g_upload_pages = nil; }

static id<MTLBuffer> upload_slice(NSUInteger bytes, id<MTLCommandBuffer> cb, NSUInteger *offset) {
    const NSUInteger page_bytes = 1024u * 1024u, max_pages = 16u;
    if (!g_upload_pages) g_upload_pages = [NSMutableArray array];
    if (bytes <= page_bytes) {
        for (DS4MetalUploadPage *p in g_upload_pages) {
            if (p.last.status == MTLCommandBufferStatusCompleted || p.last.status == MTLCommandBufferStatusError) {
                p.used = 0; p.last = nil;
            }
            const NSUInteger start = (p.used + 255u) & ~(NSUInteger)255u;
            if (start <= page_bytes && bytes <= page_bytes - start) {
                *offset = start; p.used = start + bytes; p.last = cb;
                return p.buffer;
            }
        }
        if (g_upload_pages.count < max_pages) {
            DS4MetalUploadPage *p = [DS4MetalUploadPage new];
            p.buffer = [g_device newBufferWithLength:page_bytes options:MTLResourceStorageModeShared];
            if (!p.buffer) return nil;
            p.used = bytes; p.last = cb; *offset = 0;
            [g_upload_pages addObject:p];
            return p.buffer;
        }
    }
    /* Large/overflow uploads are retained by the command buffer, not the pool. */
    *offset = 0;
    return [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

int ds4_gpu_tensor_write_queued(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    @autoreleasepool {
        DS4MetalTensor *dst = ds4_gpu_tensor_obj(tensor);
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        NSUInteger source = 0;
        id<MTLBuffer> upload = upload_slice((NSUInteger)bytes, cb, &source);
        if (!upload) return 0;
        memcpy((uint8_t *)upload.contents + source, data, (size_t)bytes);
        const NSUInteger target = (NSUInteger)(dst.offset + offset);
        if (!(target % 4u) && !(bytes % 4u) && bytes / 4u <= UINT32_MAX) {
            id<MTLComputePipelineState> pso = ds4_gpu_get_pipeline("kernel_v41_copy_u32");
            if (!pso) return 0;
            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            if (!enc) return 0;
            const uint32_t n = (uint32_t)(bytes / 4u);
            [enc setComputePipelineState:pso];
            [enc setBuffer:upload offset:source atIndex:0];
            [enc setBuffer:dst.buffer offset:target atIndex:1];
            [enc setBytes:&n length:sizeof n atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((n + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
        } else {
            if (!owned) ds4_gpu_close_batch_encoder();
            id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
            if (!enc) return 0;
            [enc copyFromBuffer:upload sourceOffset:source toBuffer:dst.buffer destinationOffset:target size:(NSUInteger)bytes];
            [enc endEncoding];
        }
        return ds4_gpu_submit_command_buffer_async(cb, owned);
    }
}
