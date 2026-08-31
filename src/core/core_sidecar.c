/* core_sidecar.c — residual/vq/zchain 侧车加载与 GPU 上载 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
#include "src/common/ds4_amp_fmt.h"   /* z 秩上限: 与工具/CUDA 同一契约 */
/* =========================================================================
 * go1b "hidden variable z^L" four-loss correction sidecar.
 * =========================================================================
 *
 * Loaded from a small separate GGUF (gguf/ds4-go1b-corr.gguf, ~94 MiB) that
 * carries only the per-layer corr_* tensors + the ds4.corr.present KV. The base
 * model GGUF is loaded unchanged. The correction is applied in the routed-MoE
 * forward at every layer that carries it; absent => exactly today's pure 1-bit. */


void residual_free(struct ds4_residual *r) {
    if (!r) return;
#ifndef DS4_NO_GPU
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(r->layer[il].g_gate);
        ds4_gpu_tensor_free(r->layer[il].g_up);
        ds4_gpu_tensor_free(r->layer[il].g_down);
    }
#endif
    model_close(&r->sidecar);
    free(r);
}

/* v2.2 直读 VQ 侧车目录(DS4_VQ_DIR): 逐层 mmap dql_vq_L%02d.bin(DQVL 校验), 零复制。
 * 盘账动机: overlay GGUF 需复制 34.5G 侧车字节, 战役机放不下; 直读=同字节同语义。 */
/* ★VQ blob 在场(2026-08-01): 专家字节全在 blk.L.ffn_exps_vq.blob 里, 前向走 CPU gather →
 * f16 scratch → GPU, blob 本身不进 Metal span; 且 --no-down 合并的文件里 gate/up 缺席、
 * down 是 bytes=0 影子张量 ⇒ experts span 必然为空, 不是切分失败。 */
bool g_vq_experts_blob;
struct ds4_residual *vq_dir_load(const char *dir) {
    struct ds4_residual *r = xcalloc(1, sizeof(*r));
    uint32_t loaded = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        char pth[512];
        snprintf(pth, sizeof pth, "%s/dql_vq_L%02u.bin", dir, il);
        int fd = open(pth, O_RDONLY);
        if (fd < 0) continue;
        struct stat st; fstat(fd, &st);
        void *mp = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (mp == MAP_FAILED) continue;
        uint32_t mg; memcpy(&mg, mp, 4);
        if (mg != 0x4C565144u || (size_t)st.st_size < 16 + 256 * 3 * 8) { munmap(mp, (size_t)st.st_size); continue; }
        r->layer[il].vq_raw = mp; r->layer[il].vq_sz = (size_t)st.st_size;
        r->layer[il].present = true;
        loaded++;
    }
    if (!loaded) { free(r); return NULL; }
    r->present = true;
    g_vq_experts_blob = true;
    fprintf(stderr, "ds4: v2.2 VQ 侧车直读 %s (%u 层)\n", dir, loaded);
    return r;
}

/* 合一 VQ GGUF(2026-07-27): blk.L.ffn_exps_vq.blob 张量(type 42 = DQVL 字节原样)在场即
 * 自动装载, 指针直指模型 mmap — 与目录直读同字节同语义, 无需 DS4_VQ_DIR/DS4_RESIDUAL。 */
struct ds4_residual *vq_model_load(const ds4_model *m) {
    struct ds4_residual *r = xcalloc(1, sizeof(*r));
    uint32_t loaded = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        char nm[64];
        snprintf(nm, sizeof nm, "blk.%u.ffn_exps_vq.blob", il);
        ds4_tensor *t = model_find_tensor(m, nm);
        if (!t || t->type != 42u) continue;
        const void *p = tensor_data(m, t);
        if (!p || t->bytes < 16 + 256 * 3 * 8) continue;
        uint32_t mg; memcpy(&mg, p, 4);
        if (mg != 0x4C565144u) {
            fprintf(stderr, "ds4: 内嵌 VQ blob 层 %u 魔数错 -- aborting (no silent quality downgrade)\n", il);
            exit(1);
        }
        r->layer[il].vq_raw = p;
        r->layer[il].vq_sz = (size_t)t->bytes;
        r->layer[il].present = true;
        loaded++;
    }
    if (!loaded) { free(r); return NULL; }
    r->present = true;
    g_vq_experts_blob = true;
    fprintf(stderr, "ds4: 合一 GGUF 内嵌 VQ blob 装载 (%u 层)\n", loaded);
    return r;
}

/* Open the 1-bit residual sidecar GGUF and bind per-layer go1b residual expert
 * tensors (blk.{L}.ffn_{gate,up,down}_exps_res.weight). Returns NULL (single
 * 1-bit path) when the file lacks ds4.residual.present or carries no tensors. */
struct ds4_residual *residual_load(const char *path, bool metal_mapping) {
    struct ds4_residual *r = xcalloc(1, sizeof(*r));
    model_open(&r->sidecar, path, metal_mapping, false);
    bool present = false;
    if (!model_get_bool(&r->sidecar, "ds4.residual.present", &present) || !present) {
        fprintf(stderr, "ds4: residual sidecar %s lacks ds4.residual.present=true; ignoring\n", path);
        model_close(&r->sidecar); free(r); return NULL;
    }
    uint32_t loaded = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        char nm[128];
        snprintf(nm, sizeof nm, "blk.%u.ffn_exps_vq.blob", il);
        ds4_tensor *vqb = model_find_tensor(&r->sidecar, nm);
        if (vqb) {                              /* v2.2 VQ overlay: 单 blob 携带全专家 */
            ds4_residual_layer *rl = &r->layer[il];
            rl->gate = rl->up = rl->down = vqb; rl->lut = NULL; rl->present = true;
            loaded++; continue;
        }
        snprintf(nm, sizeof nm, "blk.%u.ffn_gate_exps_res.weight", il);
        ds4_tensor *g = model_find_tensor(&r->sidecar, nm);
        if (!g) continue;                       /* layer not residual-corrected */
        ds4_residual_layer *rl = &r->layer[il];
        rl->gate = g;
        snprintf(nm, sizeof nm, "blk.%u.ffn_up_exps_res.weight", il);
        rl->up = model_find_tensor(&r->sidecar, nm);
        snprintf(nm, sizeof nm, "blk.%u.ffn_down_exps_res.weight", il);
        rl->down = model_find_tensor(&r->sidecar, nm);
        if (!rl->up || !rl->down) { fprintf(stderr, "ds4: residual layer %u missing up/down\n", il); exit(1); }
        /* Sparse residual (Go-active experts only): the LUT maps a routed expert id to
         * its residual slot (or -1). Absent => dense (indexed directly by expert id).
         * The metal path CPU-gathers residual experts from the sidecar mmap, so no
         * resident GPU upload is needed (unlike the corr sidecar). */
        snprintf(nm, sizeof nm, "blk.%u.ffn_res_lut.weight", il);
        rl->lut = model_find_tensor(&r->sidecar, nm);
        rl->present = true;
        loaded++;
    }
    if (loaded == 0) {
        fprintf(stderr, "ds4: residual sidecar %s carried no per-layer tensors; ignoring\n", path);
        model_close(&r->sidecar); free(r); return NULL;
    }
    r->present = true;
    fprintf(stderr, "ds4: go1b 1-bit residual loaded from %s (%u layers)\n", path, loaded);
    return r;
}

/* Build the zchain from in-model blk.L.opt_* tensors — the single merged GGUF
 * product (deepseek4-quantize --zchain lands the optimization chains next to
 * the 1bit expert bytes; KV ds4.zchain.present gates this). Op records mirror
 * the ds4_gpu packed layout (16 floats, [15]=layer-local V8 block). v8 points
 * into the model mmap (zero-copy); map stays NULL so ds4_zchain_free never
 * munmaps model memory. External --zchain/DS4_ZCHAIN overrides this loader. */
struct ds4_zchain *zchain_from_model(const ds4_model *m) {
    bool present = false;
    if (!model_get_bool(m, "ds4.zchain.present", &present) || !present) return NULL;
    ds4_zchain *z = xcalloc(1, sizeof(*z));
    z->n_layer = DS4_N_LAYER; z->n_expert = DS4_N_EXPERT; z->d_model = DS4_N_EMBD;
    z->layer = xcalloc(DS4_N_LAYER, sizeof(*z->layer));
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        char nm[64];
        snprintf(nm, sizeof nm, "blk.%u.opt_chain.weight", il);
        ds4_tensor *tc = model_find_tensor(m, nm);
        snprintf(nm, sizeof nm, "blk.%u.opt_ge.weight", il);
        ds4_tensor *tg = model_find_tensor(m, nm);
        snprintf(nm, sizeof nm, "blk.%u.opt_v8.weight", il);
        ds4_tensor *tv = model_find_tensor(m, nm);
        const uint16_t *v8 = tv ? (const uint16_t *)tensor_data(m, tv) : NULL;
        snprintf(nm, sizeof nm, "blk.%u.opt_zlm.weight", il);
        ds4_tensor *tzl = model_find_tensor(m, nm);
        const uint16_t *zlm = tzl ? (const uint16_t *)tensor_data(m, tzl) : NULL;
        if (tg && tg->dim[0] == DS4_N_EXPERT) {
            const float *ge = (const float *)tensor_data(m, tg);
            if (ge) {
                z->layer[il].ge = xmalloc((size_t)DS4_N_EXPERT * sizeof(float));
                memcpy(z->layer[il].ge, ge, (size_t)DS4_N_EXPERT * sizeof(float));
                z->n_ge_layers++;
            }
        }
        if (tc && tc->dim[0] >= DS4_AMP_CHAIN_FLOATS) {
            const float *ch = (const float *)tensor_data(m, tc);
            const uint32_t nops = (uint32_t)(tc->dim[0] / DS4_AMP_CHAIN_FLOATS);
            if (ch && nops) {
                z->layer[il].ops = xcalloc(nops, sizeof(ds4_zchain_op));
                uint32_t kept = 0;
                for (uint32_t i = 0; i < nops; i++) {
                    const float *f = ch + (size_t)i * DS4_AMP_CHAIN_FLOATS;
                    ds4_zchain_op *o = &z->layer[il].ops[kept];
                    memset(o, 0, sizeof(*o));
                    o->type = (uint32_t)f[0];
                    o->g = f[1];
                    memcpy(o->w2p, f + 2, 4 * sizeof(float));
                    memcpy(o->w8, f + 6, 9 * sizeof(float));
                    if (o->type == 3u) {
                        const int blk = (int)f[15];
                        if (blk < 0 || !v8 ||
                            (uint64_t)(blk + 1) * 8u * DS4_N_EMBD > (tv ? tv->dim[0] : 0)) {
                            fprintf(stderr, "ds4: zchain L%u dyn8 dropped: blk=%d v8=%d dim0=%llu need=%llu\n",
                                    il, blk, v8 != NULL, (unsigned long long)(tv ? tv->dim[0] : 0),
                                    (unsigned long long)((uint64_t)(blk + 1) * 8u * DS4_N_EMBD));
                            continue;   /* V8-less dyn8 = quantizer-replay no-op: drop */
                        }
                        o->v8 = v8 + (size_t)blk * 8u * DS4_N_EMBD;
                    }
                    if (o->type == 6u) {   /* 冻结 z^L: 槽{f[1]=tr,f[2]=k,f[3]=din} + opt_zlm 张量; 不进 λ 链 */
                        const uint32_t zk = (uint32_t)f[2];
                        /* f[3]=V 输入维: 旧 GGUF 写者不填(0)=线性 din=D; 3D=ftA 特征提升。
                         * 旧公式写死 din=D, ftA 载荷会被按错布局静默别解 —— 尺寸按真 din 算。 */
                        const uint32_t din = f[3] > 0.0f ? (uint32_t)f[3] : DS4_N_EMBD;
                        const uint64_t nh = DS4_AMP_ZL_ELEMS(zk, din, DS4_N_EMBD);
                        if (zk > 0 && zk <= DS4_AMP_ZK_MAX
                            && (din == DS4_N_EMBD || din == 3u * DS4_N_EMBD)
                            && zlm && (tzl ? tzl->dim[0] : 0) >= nh) {
                            z->layer[il].zl.zlk = zk;
                            z->layer[il].zl.zltr = f[1];
                            z->layer[il].zl.zdin = din;
                            z->layer[il].zl.zlm = zlm;   /* aliases model mmap */
                        } else {
                            fprintf(stderr, "ds4: zchain L%u z^L dropped: k=%u din=%u zlm_ne=%llu need=%llu\n",
                                    il, zk, din, (unsigned long long)(tzl ? tzl->dim[0] : 0),
                                    (unsigned long long)nh);
                        }
                        continue;
                    }
                    if (o->type >= 1u && o->type <= 4u) { kept++; z->n_ops_total++; }
                }
                z->layer[il].n_ops = kept;
            }
        }
    }
    uint32_t n_zl = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) if (z->layer[il].zl.zlk) n_zl++;
    if (!z->n_ops_total && !z->n_ge_layers && !n_zl) { ds4_zchain_free(z); return NULL; }
    fprintf(stderr, "ds4: zchain loaded from model tensors (merged GGUF): %u chain ops + %u GE layers + %u z^L layers\n",
            z->n_ops_total, z->n_ge_layers, n_zl);
    return z;
}

/* Build the GPU residual set for layer il (or NULL if absent). Static return: the
 * routed-MoE forward is serialized through one graph worker, so a single static is safe.
 * GPU builds only — the type itself lives behind the ds4_gpu.h GPU guard. */
#ifndef DS4_NO_GPU
const ds4_gpu_residual_set *residual_set_for(const ds4_model *m, uint32_t il) {
    static ds4_gpu_residual_set rs;
    if (!m || !m->residual || !m->residual->present ||
        il >= DS4_MAX_LAYER || !m->residual->layer[il].present) return NULL;
    const ds4_residual_layer *rl = &m->residual->layer[il];
    /* rs 是跨层复用的静态, 每条分支都要写全判别位, 否则上一层的取值会漏进本层。 */
    if (rl->vq_raw) {   /* 直读侧车: blob 即三矩阵之源 */
        rs.gate_ptr = rs.up_ptr = rs.down_ptr = rl->vq_raw;
        rs.lut = NULL; rs.merged2b = 0; rs.vq = 1;
        rs.vq_bytes = (uint64_t)rl->vq_sz;
        return &rs;
    }
    if (!rl->gate || !rl->up || !rl->down) return NULL;
    rs.gate_ptr = tensor_data(&m->residual->sidecar, rl->gate);
    rs.up_ptr   = tensor_data(&m->residual->sidecar, rl->up);
    rs.down_ptr = tensor_data(&m->residual->sidecar, rl->down);
    rs.vq_bytes = 0;
    if (!rs.gate_ptr || !rs.up_ptr || !rs.down_ptr) return NULL;
    rs.lut = rl->lut ? (const float *)tensor_data(&m->residual->sidecar, rl->lut) : NULL;
    /* Type 41 = go2b (offline-merged base+residual, R5-C): the metal path runs the
     * hot/cold two-source split instead of the legacy three-extra-matmul add. */
    rs.merged2b = (rl->gate->type == 41u);
    rs.vq = (rl->gate->type == 42u);
    return &rs;
}

/* dense matmul 的类型分发(backbone-q4k 配方): 同一批调用点此前硬编码 q8_0。
 * Q4_K 走新的 ds4_gpu_matmul_q4_K_tensor(CUDA; Metal 返回 0 即向上抛失败)。 */
int dense_matmul_typed(ds4_gpu_tensor *out, const ds4_model *m, const ds4_tensor *w,
                              uint64_t in_dim, uint64_t out_dim,
                              const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (w->type == DS4_TENSOR_Q4_K)
        return ds4_gpu_matmul_q4_K_tensor(out, m->map, m->size, w->abs_offset,
                                          in_dim, out_dim, x, n_tok);
    if (w->type == DS4_TENSOR_Q2_K)   /* 全q2 基座(2026-08-19) */
        return ds4_gpu_matmul_q2_K_tensor(out, m->map, m->size, w->abs_offset,
                                          in_dim, out_dim, x, n_tok);
    return ds4_gpu_matmul_q8_0_tensor(out, m->map, m->size, w->abs_offset,
                                      in_dim, out_dim, x, n_tok);
}

/* decode 同输入矩阵对(q_a+kv / shared gate+up): 两权重都 q4_K 时一次量化+一次发射;
 * 其他形态(q8/metal/批量)回退两次单矩阵。 */
int dense_matmul_pair_typed(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
                                   const ds4_model *m,
                                   const ds4_tensor *w0, const ds4_tensor *w1,
                                   uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
                                   const ds4_gpu_tensor *x) {
    if (w0->type == DS4_TENSOR_Q4_K && w1->type == DS4_TENSOR_Q4_K &&
        ds4_gpu_matmul_q4_K_pair_tensor(out0, out1, m->map, m->size,
                                        w0->abs_offset, w1->abs_offset,
                                        in_dim, out0_dim, out1_dim, x))
        return 1;
    if (w0->type == DS4_TENSOR_Q2_K && w1->type == DS4_TENSOR_Q2_K &&
        ds4_gpu_matmul_q2_K_pair_tensor(out0, out1, m->map, m->size,
                                        w0->abs_offset, w1->abs_offset,
                                        in_dim, out0_dim, out1_dim, x))
        return 1;
    return dense_matmul_typed(out0, m, w0, in_dim, out0_dim, x, 1) &&
           dense_matmul_typed(out1, m, w1, in_dim, out1_dim, x, 1);
}

/* attn_output 批量入口按 a 张量类型分发(q4_K/q2_K 同参; 2026-08-19 全q2)。 */
int attn_output_kq_batch(const ds4_tensor *a, ds4_gpu_tensor *out, ds4_gpu_tensor *low,
                                const void *map, uint64_t msize, uint64_t offb,
                                uint64_t group_dim, uint64_t rank, uint32_t n_groups,
                                uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    return a->type == DS4_TENSOR_Q2_K
        ? ds4_gpu_attention_output_q2k_batch_tensor(out, low, map, msize, a->abs_offset, offb,
                                                    group_dim, rank, n_groups, out_dim, heads, n_tokens)
        : ds4_gpu_attention_output_q4k_batch_tensor(out, low, map, msize, a->abs_offset, offb,
                                                    group_dim, rank, n_groups, out_dim, heads, n_tokens);
}

/* Pack the host-side zchain into the flat GPU tables (layout: ds4_gpu.h).
 * Ops are layer-major; dyn8 V8 blocks concatenate in encounter order and each
 * op's slot [15] carries its block index (-1 = none). GE rows default to 1.0
 * so a single [n_layer][n_expert] table serves every GE layer. */
int zchain_gpu_upload(const struct ds4_zchain *z) {
    const uint32_t nl = z->n_layer, ne = z->n_expert, dm = z->d_model;
    uint32_t *loff = xmalloc((size_t)(nl + 1) * sizeof(uint32_t));
    uint8_t *gep = xcalloc(nl, 1);
    uint32_t nops = 0, nblk = 0, has_ge = 0;
    for (uint32_t l = 0; l < nl; l++) {
        loff[l] = nops;
        nops += z->layer[l].n_ops;
        if (z->layer[l].ge) { gep[l] = 1; has_ge = 1; }
        for (uint32_t i = 0; i < z->layer[l].n_ops; i++)
            if (z->layer[l].ops[i].type == 3u && z->layer[l].ops[i].v8) nblk++;
    }
    loff[nl] = nops;
    float *ops = nops ? xmalloc((size_t)nops * 16u * sizeof(float)) : NULL;
    uint16_t *v8 = nblk ? xmalloc((size_t)nblk * 8u * dm * sizeof(uint16_t)) : NULL;
    uint32_t oi = 0, bi = 0;
    for (uint32_t l = 0; l < nl; l++) {
        for (uint32_t i = 0; i < z->layer[l].n_ops; i++, oi++) {
            const ds4_zchain_op *o = &z->layer[l].ops[i];
            float *f = ops + (size_t)oi * 16u;
            f[0] = (float)o->type;
            f[1] = o->g;
            memcpy(f + 2, o->w2p, 4 * sizeof(float));
            memcpy(f + 6, o->w8, 9 * sizeof(float));
            f[15] = -1.0f;
            if (o->type == 3u && o->v8) {
                memcpy(v8 + (size_t)bi * 8u * dm, o->v8, (size_t)8u * dm * sizeof(uint16_t));
                f[15] = (float)bi++;
            }
        }
    }
    float *ge = NULL;
    if (has_ge) {
        ge = xmalloc((size_t)nl * ne * sizeof(float));
        for (size_t i = 0; i < (size_t)nl * ne; i++) ge[i] = 1.0f;
        for (uint32_t l = 0; l < nl; l++)
            if (z->layer[l].ge)
                memcpy(ge + (size_t)l * ne, z->layer[l].ge, (size_t)ne * sizeof(float));
    }
    int r = ds4_gpu_zchain_set(ops, loff, v8, ge, has_ge ? gep : NULL,
                               nl, ne, dm, nops, nblk);
    /* frozen z^L (type 6): concat fp16 factor blocks + per-layer meta */
    if (r) {
        uint64_t total = 0; uint32_t nz = 0;
        for (uint32_t l = 0; l < nl; l++) if (z->layer[l].zl.zlk) {
            const uint32_t di = z->layer[l].zl.zdin ? z->layer[l].zl.zdin : dm;
            total += (uint64_t)z->layer[l].zl.zlk * (1u + dm + di); nz++; }
        if (nz) {
            uint16_t *zlm = xmalloc(total * sizeof(uint16_t));
            uint32_t *zo = xcalloc(nl, sizeof(uint32_t));
            uint32_t *zk = xcalloc(nl, sizeof(uint32_t));
            uint32_t *zd = xcalloc(nl, sizeof(uint32_t));
            float    *zt = xcalloc(nl, sizeof(float));
            uint32_t *zm = xcalloc(nl, sizeof(uint32_t));
            uint64_t cur = 0;
            for (uint32_t l = 0; l < nl; l++) {
                const ds4_zchain_zl *lz = &z->layer[l].zl;
                if (!lz->zlk) continue;
                const uint32_t di = lz->zdin ? lz->zdin : dm;
                const uint64_t nh = (uint64_t)lz->zlk * (1u + dm + di);
                memcpy(zlm + cur, lz->zlm, nh * sizeof(uint16_t));
                zo[l] = (uint32_t)cur; zk[l] = lz->zlk; zd[l] = di; zt[l] = lz->zltr;
                zm[l] = lz->zmul;
                cur += nh;
            }
            r = ds4_gpu_zchain_zl_set(zlm, zo, zk, zd, zt, zm, nl, total);
            free(zlm); free(zo); free(zk); free(zd); free(zt); free(zm);
        }
    }
    /* 路由闭式侧车(type8): z^L 同款 concat 上传, blob = z[k]|U[ne*k]|V[din*k] */
    if (r) {
        uint64_t total = 0; uint32_t nr = 0;
        for (uint32_t l = 0; l < nl; l++) if (z->layer[l].rte.zlk) {
            total += (uint64_t)z->layer[l].rte.zlk * (1u + ne + z->layer[l].rte.zdin); nr++; }
        if (nr) {
            uint16_t *rm = xmalloc(total * sizeof(uint16_t));
            uint32_t *ro = xcalloc(nl, sizeof(uint32_t));
            uint32_t *rk = xcalloc(nl, sizeof(uint32_t));
            float    *rt = xcalloc(nl, sizeof(float));
            uint64_t cur = 0;
            for (uint32_t l = 0; l < nl; l++) {
                const ds4_zchain_zl *lr = &z->layer[l].rte;
                if (!lr->zlk) continue;
                const uint64_t nh = (uint64_t)lr->zlk * (1u + ne + lr->zdin);
                memcpy(rm + cur, lr->zlm, nh * sizeof(uint16_t));
                ro[l] = (uint32_t)cur; rk[l] = lr->zlk; rt[l] = lr->zltr;
                cur += nh;
            }
            r = ds4_gpu_zchain_rte_set(rm, ro, rk, rt, nl, ne, total);
            free(rm); free(ro); free(rk); free(rt);
        }
    }
    free(ops); free(v8); free(ge); free(loff); free(gep);
    return r;
}
#endif

/* Per-layer router-logit bias (delta), or NULL when this layer has no correction.
 * Only score-routed layers consume it (hash layers select experts by token id). */

/* Optional startup pass that touches tensor pages before timing generation. */
void model_warm_weights(const ds4_model *m) {
    const uint64_t start = m->tensor_data_pos;
    const uint64_t end = m->size;
    if (start >= end) return;

    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint8_t *p = m->map;
    volatile uint64_t checksum = 0;
    const double t0 = now_sec();

    fprintf(stderr, "ds4: warming mapped tensor pages: %.2f GiB\n",
            (double)(end - start) / (1024.0 * 1024.0 * 1024.0));

#if defined(POSIX_MADV_WILLNEED)
    (void)posix_madvise((void *)(p + start), (size_t)(end - start), POSIX_MADV_WILLNEED);
#endif

    for (uint64_t off = start; off < end; off += page) {
        checksum += p[off];
    }
    checksum += p[end - 1];

    const double t1 = now_sec();
    fprintf(stderr, "ds4: warmed tensor pages in %.3fs (checksum=%llu)\n",
            t1 - t0, (unsigned long long)checksum);
}

