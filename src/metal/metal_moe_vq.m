/* metal_moe_vq.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

/* Pass 2: optionally sort the active ids ascending (= ascending file offsets,
 * so cold reads walk the GGUF forward instead of jumping around) and rewrite
 * the selected-id buffer from original expert ids to compact scratch slots.
 * Bit-exact either way: per-pick results and their summation order are
 * unchanged, only the scratch slot numbering moves. */
#include "vq_fmt.h"

id<MTLBuffer> g_moe_vq_gate_scratch, g_moe_vq_up_scratch, g_moe_vq_down_scratch;

float *g_vq_diag_ref;

uint32_t g_vq_diag_layer, g_vq_diag_ntok, g_vq_diag_out;

static void *ds4_vq_gather_worker(void *arg) {
    ds4_vq_gather_task *t = (ds4_vq_gather_task *)arg;
    for (uint32_t i = t->lo; i < t->hi && !*t->err; i++) {
        const uint32_t e = t->active_ids[i];
        uint16_t *dg = t->gbase + (uint64_t)i * t->mid * t->in;
        uint16_t *du = t->ubase + (uint64_t)i * t->mid * t->in;
        uint16_t *dd = t->dbase + (uint64_t)i * t->out_dim * t->mid;
        uint64_t o1 = ds4vq_slot(t->blob, (int)e, 0), o3 = ds4vq_slot(t->blob, (int)e, 1), o2 = ds4vq_slot(t->blob, (int)e, 2);
        int rc1 = (!o1) ? -9 : ds4vq_dequant_f16(t->blob + o1, dg, (int)t->mid, (int)t->in);
        int rc3 = (rc1 == 0 && o3) ? ds4vq_dequant_f16(t->blob + o3, du, (int)t->mid, (int)t->in) : (!o3 ? -9 : 0);
        if (rc1 != 0 || rc3 != 0) {
            fprintf(stderr, "ds4: [vq-gather-err] e=%u o1=%llu o3=%llu rc1=%d rc3=%d mid=%u in=%u\n",
                    e, (unsigned long long)o1, (unsigned long long)o3, rc1, rc3, t->mid, t->in);
            *t->err = 1; return NULL;
        }
        if (o2) {
            int rc2 = ds4vq_dequant_f16(t->blob + o2, dd, (int)t->out_dim, (int)t->mid);
            if (rc2 != 0) {
                fprintf(stderr, "ds4: [vq-gather-err] e=%u o2=%llu rc2=%d out=%u mid=%u\n",
                        e, (unsigned long long)o2, rc2, t->out_dim, t->mid);
                *t->err = 1; return NULL;
            }
        } else {   /* 冷 w2: base go1b 34B/256el → ±d f16 */
            /* R28: down 影子张量(base w2 不入文件)⇒ 这条回退路没有字节可读。
             * 走到这里说明 blob 的 which=2 槽缺失但文件又没带 base down —— 产物与
             * 引擎不同代, 硬失败而不是读 offset 0 的垃圾当权重。 */
            if (t->down_expert_bytes == 0 || t->down_offset == 0) {
                fprintf(stderr, "ds4: [vq-gather-err] e=%u 冷 w2 槽缺失且 base down 不在文件里"
                                "(影子张量) -- aborting (no silent quality downgrade)\n", e);
                *t->err = 1; return NULL;
            }
            const uint8_t *sd = (const uint8_t *)t->model_map + t->down_offset + (uint64_t)e * t->down_expert_bytes;
            const uint64_t nblk_row = t->mid / 256u;
            for (uint32_t r = 0; r < t->out_dim; r++) {
                const uint8_t *rb = sd + (uint64_t)r * nblk_row * 34u;
                uint16_t *orow = dd + (uint64_t)r * t->mid;
                for (uint64_t b = 0; b < nblk_row; b++) {
                    uint16_t dsc; memcpy(&dsc, rb + b * 34u, 2);
                    const uint8_t *sg = rb + b * 34u + 2;
                    uint16_t *o = orow + b * 256u;
                    for (int k = 0; k < 256; k++)
                        o[k] = (sg[k >> 3] >> (k & 7)) & 1 ? dsc : (uint16_t)(dsc ^ 0x8000u);
                }
            }
        }
    }
    return NULL;
}

/* v2.2 VQ unified gather: 全活跃专家 dequant→f16 scratch(热三矩阵/冷 w1w3 走层 blob;
 * 冷 w2 从 base go1b 字节展开 ±d)。scratch 上限护栏 DS4_VQ_SCRATCH_GB(默认3)。 */
int ds4_gpu_vq_unified_gather(
        const void *model_map, const uint8_t *blob,
        uint32_t n_active, const uint32_t *active_ids,
        uint64_t down_offset, uint64_t down_expert_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim) {
    const uint64_t ge = (uint64_t)expert_mid_dim * expert_in_dim * 2u;   /* f16 gate/up 每专家 */
    const uint64_t de = (uint64_t)out_dim * expert_mid_dim * 2u;
    const uint64_t need_g = (uint64_t)n_active * ge, need_d = (uint64_t)n_active * de;
    static uint64_t cap = 0;
    if (!cap) { const char *e = getenv("DS4_VQ_SCRATCH_GB"); double g = e ? atof(e) : 3.0; cap = (uint64_t)(g * 1073741824.0); }
    if (2 * need_g + need_d > cap) {
        fprintf(stderr, "ds4: VQ gather scratch %.2fGB > cap(降 DS4_METAL_PREFILL_CHUNK 或调 DS4_VQ_SCRATCH_GB)\n",
                (2.0 * need_g + need_d) / 1073741824.0);
        return 0;
    }
    if (!g_moe_vq_gate_scratch || (uint64_t)g_moe_vq_gate_scratch.length < need_g) {
        g_moe_vq_gate_scratch = [g_device newBufferWithLength:(NSUInteger)need_g options:MTLResourceStorageModeShared];
        g_moe_vq_up_scratch   = [g_device newBufferWithLength:(NSUInteger)need_g options:MTLResourceStorageModeShared];
    }
    if (!g_moe_vq_down_scratch || (uint64_t)g_moe_vq_down_scratch.length < need_d)
        g_moe_vq_down_scratch = [g_device newBufferWithLength:(NSUInteger)need_d options:MTLResourceStorageModeShared];
    if (!g_moe_vq_gate_scratch || !g_moe_vq_up_scratch || !g_moe_vq_down_scratch) return 0;
    /* memset 移除: 下面 dequant 填满全部 n_active 专家 gate/up/down, 清零冗余(省~300ms/token, ~730MB memset) */
    double _vqgt = getenv("DS4_VQ_GT") ? ds4_gpu_now_ms() : 0.0;
    /* 8 线程并行 dequant(各专家写不相交 scratch, 无锁); 串行是 decode 慢主因 */
    int vqnth = 8; { const char *te = getenv("DS4_METAL_EXPERT_GATHER_THREADS"); if (te && atoi(te) > 0) vqnth = atoi(te); }
    if ((uint32_t)vqnth > n_active) vqnth = (int)(n_active ? n_active : 1);
    if (vqnth > 32) vqnth = 32;
    volatile int vqerr = 0;
    pthread_t vqth[32]; ds4_vq_gather_task vqtk[32];
    uint16_t *gbase = (uint16_t *)g_moe_vq_gate_scratch.contents;
    uint16_t *ubase = (uint16_t *)g_moe_vq_up_scratch.contents;
    uint16_t *dbase = (uint16_t *)g_moe_vq_down_scratch.contents;
    for (int ti = 0; ti < vqnth; ti++) {
        vqtk[ti] = (ds4_vq_gather_task){ model_map, blob, active_ids, gbase, ubase, dbase,
            down_offset, down_expert_bytes, expert_in_dim, expert_mid_dim, out_dim,
            (uint32_t)((uint64_t)ti * n_active / vqnth), (uint32_t)((uint64_t)(ti + 1) * n_active / vqnth), &vqerr };
        pthread_create(&vqth[ti], NULL, ds4_vq_gather_worker, &vqtk[ti]);
    }
    for (int ti = 0; ti < vqnth; ti++) pthread_join(vqth[ti], NULL);
    if (vqerr) return 0;
    if (_vqgt) fprintf(stderr, "[VQ_GT] n_active=%u nth=%d dequant=%.1fms\n", n_active, vqnth, ds4_gpu_now_ms() - _vqgt);
    if (getenv("DS4_VQ_DEBUG") && n_active > 0) {
        const uint16_t *g0 = (const uint16_t *)g_moe_vq_gate_scratch.contents;
        const uint16_t *d0 = (const uint16_t *)g_moe_vq_down_scratch.contents;
        int nz_g = 0, nz_d = 0;
        for (uint64_t k = 0; k < ge / 2u; k++) if (g0[k]) nz_g++;
        for (uint64_t k = 0; k < de / 2u; k++) if (d0[k]) nz_d++;
        uint64_t o1 = ds4vq_slot(blob, (int)active_ids[0], 0);
        uint64_t o2 = ds4vq_slot(blob, (int)active_ids[0], 2);
        fprintf(stderr, "[VQ_DBG] n_active=%u e0=%u gate_nz=%d/%llu(w0=%.4f) down_nz=%d/%llu o1=%llu o2=%llu(w2hot=%d)\n",
                n_active, active_ids[0], nz_g, (unsigned long long)(ge/2u), ds4vq_f16(g0[0]),
                nz_d, (unsigned long long)(de/2u), (unsigned long long)o1, (unsigned long long)o2, o2!=0);
    }
    return 1;
}

/* R5-C unified go2b gather: fill compact scratch with ALL active experts as go2b
 * blocks — hot experts memcpy'd from the merged sidecar, cold experts fabricated
 * on the fly from the base go1b bytes (d1=base scale, d2=0 => second plane inert;
 * s2 bits left as-is). One weight type per layer => the MoE keeps bare's exact
 * single map + three-tile dispatch shape. */
int ds4_gpu_hot_unified_gather(
        const void *model_map,
        uint32_t n_active, const uint32_t *active_ids, const float *lut,
        const void *hot_gate, const void *hot_up, const void *hot_down,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim) {
    const uint64_t g2_gate_row = (uint64_t)expert_in_dim / 256u * 68u;
    const uint64_t g2_gate_exp = (uint64_t)expert_mid_dim * g2_gate_row;
    const uint64_t g2_down_row = (uint64_t)expert_mid_dim / 256u * 68u;
    const uint64_t g2_down_exp = (uint64_t)out_dim * g2_down_row;
    const uint64_t need_g = (uint64_t)n_active * g2_gate_exp;
    const uint64_t need_d = (uint64_t)n_active * g2_down_exp;
    if (!g_moe_hot_gate_scratch || (uint64_t)g_moe_hot_gate_scratch.length < need_g) {
        g_moe_hot_gate_scratch = [g_device newBufferWithLength:(NSUInteger)need_g options:MTLResourceStorageModeShared];
        g_moe_hot_up_scratch   = [g_device newBufferWithLength:(NSUInteger)need_g options:MTLResourceStorageModeShared];
    }
    if (!g_moe_hot_down_scratch || (uint64_t)g_moe_hot_down_scratch.length < need_d)
        g_moe_hot_down_scratch = [g_device newBufferWithLength:(NSUInteger)need_d options:MTLResourceStorageModeShared];
    if (!g_moe_hot_gate_scratch || !g_moe_hot_up_scratch || !g_moe_hot_down_scratch) return 0;
    for (uint32_t i = 0; i < n_active; i++) {
        const uint32_t e = active_ids[i];
        const int slot = lut ? (int)lut[e] : -1;
        uint8_t *dg = (uint8_t *)g_moe_hot_gate_scratch.contents + (uint64_t)i * g2_gate_exp;
        uint8_t *du = (uint8_t *)g_moe_hot_up_scratch.contents   + (uint64_t)i * g2_gate_exp;
        uint8_t *dd = (uint8_t *)g_moe_hot_down_scratch.contents + (uint64_t)i * g2_down_exp;
        if (slot >= 0) {
            memcpy(dg, (const uint8_t *)hot_gate + (uint64_t)slot * g2_gate_exp, g2_gate_exp);
            memcpy(du, (const uint8_t *)hot_up   + (uint64_t)slot * g2_gate_exp, g2_gate_exp);
            memcpy(dd, (const uint8_t *)hot_down + (uint64_t)slot * g2_down_exp, g2_down_exp);
        } else {
            const uint8_t *sg = (const uint8_t *)model_map + gate_offset + (uint64_t)e * gate_expert_bytes;
            const uint8_t *su = (const uint8_t *)model_map + up_offset   + (uint64_t)e * gate_expert_bytes;
            const uint8_t *sd = (const uint8_t *)model_map + down_offset + (uint64_t)e * down_expert_bytes;
            const uint64_t nbg = g2_gate_exp / 68u, nbd = g2_down_exp / 68u;
            for (uint64_t b = 0; b < nbg; b++) {
                uint8_t *o = dg + b * 68u; const uint8_t *p = sg + b * 34u;
                o[0]=p[0]; o[1]=p[1]; o[2]=0; o[3]=0; memcpy(o+4, p+2, 32);
                o = du + b * 68u; p = su + b * 34u;
                o[0]=p[0]; o[1]=p[1]; o[2]=0; o[3]=0; memcpy(o+4, p+2, 32);
            }
            for (uint64_t b = 0; b < nbd; b++) {
                uint8_t *o = dd + b * 68u; const uint8_t *p = sd + b * 34u;
                o[0]=p[0]; o[1]=p[1]; o[2]=0; o[3]=0; memcpy(o+4, p+2, 32);
            }
        }
    }
    return 1;
}

int ds4_gpu_remap_selected_to_slots(
        id<MTLBuffer> selectedbuf,
        NSUInteger    selected_off,
        uint32_t      n_picks,
        uint32_t      n_expert_total,
        uint32_t     *active_ids,
        uint32_t      n_active) {
    if (!selectedbuf || !active_ids || n_active == 0 || n_active > 1024u) return 0;
    if (ds4_gpu_expert_sort_ids_enabled() && n_active > 1u) {
        qsort(active_ids, n_active, sizeof(uint32_t), ds4_gpu_cmp_u32);
    }
    int16_t compact_lut[1024];
    for (uint32_t i = 0; i < 1024u; i++) compact_lut[i] = -1;
    for (uint32_t s = 0; s < n_active; s++) {
        if (active_ids[s] >= 1024u) return 0;
        compact_lut[active_ids[s]] = (int16_t)s;
    }
    int32_t *sel_cpu =
        (int32_t *)((uint8_t *)selectedbuf.contents + (size_t)selected_off);
    for (uint32_t i = 0; i < n_picks; i++) {
        int32_t raw = sel_cpu[i];
        uint32_t id = (raw >= 0 && (uint32_t)raw < n_expert_total) ? (uint32_t)raw : 0u;
        if (id >= 1024u || compact_lut[id] < 0) return 0;
        sel_cpu[i] = (int32_t)compact_lut[id];
    }
    return 1;
}

int ds4_gpu_compact_selected_experts(
        id<MTLBuffer> selectedbuf,
        NSUInteger    selected_off,
        uint32_t      n_picks,
        uint32_t      n_expert_total,
        uint32_t     *active_ids,
        uint32_t      active_cap,
        uint32_t     *n_active_out) {
    if (!ds4_gpu_collect_active_experts(selectedbuf, selected_off, n_picks,
                                        n_expert_total, active_ids, active_cap,
                                        n_active_out)) {
        return 0;
    }
    return ds4_gpu_remap_selected_to_slots(selectedbuf, selected_off, n_picks,
                                           n_expert_total, active_ids,
                                           *n_active_out);
}
