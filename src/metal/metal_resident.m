/* metal_resident.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

void ds4_gpu_model_views_clear(void) {
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        g_model_views[i].buffer = nil;
        g_model_views[i].model_map = NULL;
        g_model_views[i].model_size = 0;
        g_model_views[i].model_offset = 0;
        g_model_views[i].bytes = 0;
    }
    g_model_view_count = 0;
}

void ds4_gpu_model_residency_clear(void) {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (g_model_residency_set) {
            [g_model_residency_set endResidency];
            [g_model_residency_set removeAllAllocations];
            g_model_residency_set = nil;
        }
    }
#endif
    g_model_residency_count = 0;
}

static int ds4_gpu_model_residency_request_views(void) {
    if (g_model_view_count == 0 || getenv("DS4_METAL_NO_RESIDENCY") != NULL) return 1;

#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        /*
         * Register all model views as one residency set before inference. This
         * is a GPU residency/budgeting hint, not a request to fault the whole
         * 80+ GB file into memory. Its purpose is to make the driver see the
         * complete set of large shared allocations during setup instead of
         * discovering them lazily from the first measured graph command, where
         * VM validation and residency accounting would look like model compute.
         */
        MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ds4_model";
        desc.initialCapacity = g_model_view_count;

        NSError *error = nil;
        g_model_residency_set = [g_device newResidencySetWithDescriptor:desc error:&error];
        if (!g_model_residency_set) {
            fprintf(stderr, "ds4: Metal model residency set creation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 0;
        }

        uint32_t resident_views = 0;
        uint64_t resident_bytes = 0;
        for (uint32_t i = 0; i < g_model_view_count; i++) {
            /* Non-resident views (routed-expert offload) are wrapped but kept out
             * of the residency set so their clean pages stay reclaimable. */
            if (!g_model_views[i].resident_hint) continue;
            [g_model_residency_set addAllocation:g_model_views[i].buffer];
            resident_views++;
            resident_bytes += g_model_views[i].bytes;
        }
        [g_model_residency_set commit];
        [g_model_residency_set requestResidency];
        g_model_residency_count = resident_views;
        /* [diag] surface the actual wired working set vs the GPU ceiling. This
         * fires once per model map (base slice, then again after MTP appends its
         * views), so the second line shows the cumulative base+MTP residency. */
        fprintf(stderr,
                "ds4: [diag] residency set: %u/%u views wired, %.2f GiB; "
                "device recommendedMax %.2f GiB, currentAllocated %.2f GiB\n",
                resident_views, g_model_view_count,
                (double)resident_bytes / (1024.0 * 1024.0 * 1024.0),
                (double)[g_device recommendedMaxWorkingSetSize] / (1024.0 * 1024.0 * 1024.0),
                (double)[g_device currentAllocatedSize] / (1024.0 * 1024.0 * 1024.0));
    }
#endif

    return 1;
}

int ds4_gpu_add_model_view_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    map_offset,
        uint64_t    map_size,
        uint64_t    max_tensor_bytes,
        bool        resident,
        uint64_t   *mapped_model_size_out) {
    const uint64_t page = (uint64_t)getpagesize();
    const uintptr_t model_addr = (uintptr_t)model_map;

    if ((model_addr & (uintptr_t)(page - 1)) != 0) {
        fprintf(stderr, "ds4: Metal model mmap base is not page aligned\n");
        return 0;
    }
    if (map_offset > model_size || map_size > model_size - map_offset) {
        fprintf(stderr, "ds4: Metal model mapped range is outside the GGUF mapping\n");
        return 0;
    }
    const uint64_t page_model_offset = map_offset & ~(page - 1);
    const uint64_t leading = map_offset - page_model_offset;
    if (map_size > UINT64_MAX - leading ||
        leading + map_size > UINT64_MAX - (page - 1))
    {
        fprintf(stderr, "ds4: Metal model mapped range overflows page alignment\n");
        return 0;
    }
    const uint64_t mapped_model_size = round_up_u64(leading + map_size, page);
    uint64_t max_buffer = (uint64_t)[g_device maxBufferLength];
    max_buffer &= ~(page - 1);

    /*
     * Wrap only the tensor-data part of the GGUF file. Metadata is parsed by the
     * CPU and is never dereferenced by kernels, so exposing it to Metal only
     * grows the residency set and the VM range the driver must validate.
     *
     * Metal buffers have a device-specific maximum length, and this model is
     * larger than that maximum on the target machines. Creating one no-copy
     * buffer per tensor would avoid the length limit, but it would also move a
     * lot of VM-object creation and residency bookkeeping into graph setup. The
     * stable shape here is a tiny number of page-aligned views created once.
     *
     * Adjacent views intentionally overlap by more than the largest tensor, plus
     * one page for alignment. That invariant guarantees every tensor lies wholly
     * inside at least one view, so hot paths pass one buffer and one inner byte
     * offset. We never split a weight tensor across command encoders.
     */
    if (max_tensor_bytes > map_size) {
        fprintf(stderr, "ds4: Metal model max tensor span is larger than a mapped tensor span\n");
        return 0;
    }
    if (max_tensor_bytes > UINT64_MAX - (page - 1)) {
        fprintf(stderr, "ds4: Metal model max tensor span overflows page alignment\n");
        return 0;
    }
    const uint64_t max_tensor_rounded = round_up_u64(max_tensor_bytes, page);
    if (max_tensor_rounded > UINT64_MAX - page) {
        fprintf(stderr, "ds4: Metal model view overlap overflows page slack\n");
        return 0;
    }
    const uint64_t overlap = max_tensor_rounded + page;
    if (max_buffer == 0 || max_buffer <= overlap) {
        fprintf(stderr,
                "ds4: Metal maxBufferLength is too small for DS4 model views "
                "(max tensor %.2f GiB, max buffer %.2f GiB)\n",
                ds4_gpu_gib(max_tensor_bytes),
                ds4_gpu_gib(max_buffer));
        return 0;
    }

    const uint64_t step = max_buffer - overlap;
    uint64_t off = 0;
    while (off < mapped_model_size) {
        if (g_model_view_count == DS4_METAL_MAX_MODEL_VIEWS) {
            fprintf(stderr, "ds4: Metal model needs more mapped views than expected\n");
            return 0;
        }

        uint64_t view_bytes = mapped_model_size - off;
        if (view_bytes > max_buffer) view_bytes = max_buffer;

        id<MTLBuffer> buffer = [g_device newBufferWithBytesNoCopy:(void *)(model_addr + page_model_offset + off)
                                                           length:(NSUInteger)view_bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!buffer) {
            fprintf(stderr,
                    "ds4: Metal could not wrap mmaped model view at %.2f GiB, size %.2f GiB\n",
                    (double)(page_model_offset + off) / (1024.0 * 1024.0 * 1024.0),
                    (double)view_bytes / (1024.0 * 1024.0 * 1024.0));
            return 0;
        }
        buffer.label = [NSString stringWithFormat:@"ds4_model_view_%u", g_model_view_count];

        g_model_views[g_model_view_count].buffer = buffer;
        g_model_views[g_model_view_count].model_map = model_map;
        g_model_views[g_model_view_count].model_size = model_size;
        g_model_views[g_model_view_count].model_offset = page_model_offset + off;
        g_model_views[g_model_view_count].bytes = view_bytes;
        g_model_views[g_model_view_count].resident_hint = resident;
        g_model_view_count++;

        g_model_wrap_count++;
        g_model_wrap_bytes += view_bytes;
        if (view_bytes > g_model_wrap_max_bytes) g_model_wrap_max_bytes = view_bytes;

        if (off + view_bytes >= mapped_model_size) break;
        off += step;
    }

    if (mapped_model_size_out) *mapped_model_size_out += mapped_model_size;
    return 1;
}

int ds4_gpu_finish_model_views(
        double t0,
        uint64_t mapped_model_size,
        uint64_t display_offset) {
    const double t_mapped = ds4_gpu_now_ms();
    const int request_residency = getenv("DS4_METAL_NO_RESIDENCY") == NULL;
    if (request_residency) ds4_gpu_progress_begin("requesting Metal residency (may take tens of seconds)");
    if (!ds4_gpu_model_residency_request_views()) {
        if (request_residency) ds4_gpu_progress_failed();
        return 0;
    }
    if (request_residency) ds4_gpu_progress_done();
    const double t_resident = ds4_gpu_now_ms();
    int warmed = 1;
    const double t_warm0 = ds4_gpu_now_ms();
    const int warm_model_views = getenv("DS4_METAL_NO_RESIDENCY") == NULL &&
                                 getenv("DS4_METAL_NO_MODEL_WARMUP") == NULL;
    if (warm_model_views) {
        /*
         * The first GPU command touching no-copy mmap storage can pay command
         * queue setup, page-table validation, and shared-allocation residency
         * costs. Sample each model view here so timed graph execution starts
         * after that one-time work. The stride is intentionally coarse: this is
         * a validation touch over the VM ranges, not a full model prefetch. A
         * dense prefetch would create exactly the kind of memory pressure and
         * startup stalls this path is designed to avoid.
         */
        ds4_gpu_progress_begin("warming Metal model views");
        warmed = ds4_gpu_warm_model_views();
        if (warmed) ds4_gpu_progress_done();
        else ds4_gpu_progress_failed();
    }
    const double t_warm = ds4_gpu_now_ms();
    fprintf(stderr,
            "ds4: Metal model views created in %.3f ms, residency requested in %.3f ms, warmup %.3f ms (mapped %.2f MiB from offset %.2f MiB)\n",
            t_mapped - t0,
            t_resident - t_mapped,
            t_warm - t_warm0,
            mapped_model_size / 1024.0 / 1024.0,
            display_offset / 1024.0 / 1024.0);
    if (!warmed) return 0;
    return 1;
}

void ds4_gpu_set_model_map_nonresident_hint(int on) {
    g_model_view_force_nonresident = on ? 1 : 0;
}

int ds4_gpu_map_model_views(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    map_offset,
        uint64_t    map_size,
        uint64_t    max_tensor_bytes) {
    const double t0 = ds4_gpu_now_ms();
    uint64_t mapped_model_size = 0;
    const bool resident = g_model_view_force_nonresident ? false : true;
    if (!ds4_gpu_add_model_view_range(model_map,
                                      model_size,
                                      map_offset,
                                      map_size,
                                      max_tensor_bytes,
                                      resident,
                                      &mapped_model_size)) {
        return 0;
    }
    return ds4_gpu_finish_model_views(t0, mapped_model_size, map_offset);
}

id<MTLBuffer> ds4_gpu_new_transient_buffer(NSUInteger bytes, const char *label) {
    if (bytes == 0) bytes = 1;

    id<MTLBuffer> buffer = [g_device newBufferWithLength:bytes
                                                 options:MTLResourceStorageModeShared];
    if (!buffer) {
        fprintf(stderr, "ds4: failed to allocate Metal transient buffer %s (%llu bytes)\n",
                label ? label : "(unnamed)", (unsigned long long)bytes);
        return nil;
    }
    if (label) buffer.label = [NSString stringWithUTF8String:label];

    /*
     * CPU-filled buffers must survive until their command buffer completes.
     * A local ObjC strong variable is not enough when the encoder function
     * returns before the caller commits the command buffer.
     */
    [g_transient_buffers addObject:buffer];
    return buffer;
}
