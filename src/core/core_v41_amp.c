/* core_v41_amp.c — DeepSeek V4.1 反修放大器(zchain 的 V4.1 形态)在引擎里的加载与应用(2026-09-12)。
 *
 * 产物: <dir>/amp_Lnn.bin, 头 <i32 D><i32 K><i32 存储类型>, 接 A[K][D], 再 B[K][D](B 内存 [K][D] 行主序 = 数学上的
 * D×K 列主序)。语义与 Python 判决态(v41_amp_hooks.install_apply → libv41amp v41_amp_apply_gpu)逐式同:
 * 每层 MoE 输出 y(bf16 格点) 做 y += x·(B·A) (f32), 再舍回 bf16; x = 该层 MoE 输入(ffn_norm 出口)。
 * 引擎侧算法: T = x·B (n×K), y += T·A —— 两发 cuBLAS Sgemm, 与 C 库先乘 B·A 再乘 x 只差累加序。
 * 没有的层就不挂(逐层 K 可不同)。
 *
 * 【存储类型(2026-09-13)】头第三字段: 43 = fp4x32(4.25 bpw, 骨架/出口头/HF 出厂专家同一种块格式, spark 原生),
 * 1 = f32(09-12 的第一版产物, 8 倍体积)。★必须按这个字段分派★ —— 两种格式的字节数差 7.5 倍, 拿 f32 的读法去读
 * fp4 文件不会报错(长度校验之外), 只会把码字当浮点位型解出一堆垃圾, 挂上去照跑, 出的是"一眼合理"的假账。
 * 解出来一律是 f32 上设备: 省的是磁盘与解码带宽, GEMM 仍走 f32(K≤512 的两发小 GEMM 不是瓶颈)。 */
#include "core_internal.h"
#include "../common/ds4_quantfmt.h"
#include "../common/ds4_float.h"
#include "../common/ds4_gr_fnv.h"   /* 指纹算法: 引擎核对与解算器落盘共用一份 */
#ifndef DS4_NO_GPU

/* 读一块 [K][D] 权重到 buf(f32)。ty 43 = fp4x32 按块解码, 1 = 直接 f32。返回 false = 截断/类型不认。 */
static bool amp_read_mat(FILE *f, int32_t ty, size_t nel, float *buf, uint8_t *pk) {
    if (ty == (int32_t)DS4_GGT_FP4X32) {
        const size_t nb = nel / 32, nby = nb * 17;
        if (fread(pk, 1, nby, f) != nby) return false;
        ds4_deq_fp4x32(pk, nb, buf);
        return true;
    }
    return fread(buf, 4, nel, f) == nel;
}

/* 读一层的增益缩放因子文件 gr_Lnn.bin 并【乘进】acc(acc 进来必须已初始化为 1)。
 * 头 <i32 n_expert><i32 D><i32 类型>, 类型: 1 = f32(s 本身) / 2 = f16(s 本身) / 43 = fp4x32(★存 s−1★)。
 *
 * ★为什么 fp4 必须存 s−1 而不是 s★(2026-09-13, 用户要求插件走 fp4 精度): fp4 是 E2M1 加每 32 个数
 * 一个块缩放, 格点 ±{0,.5,1,1.5,2,3,4,6}×块缩放。增益因子全挤在 1 附近(实测 |s−1| 中位 0.09~0.31),
 * 直接存 s 的话块缩放被最大值(≈1.3)定死在 0.22, 格点间隔 0.11 ⇒ 量化误差 ±0.055, 是修正量本身的
 * 20~60% —— 等于把插件的作用抹掉大半。存 s−1 则以 0 为中心、动态范围大, 正是块缩放擅长的形态,
 * 相对误差降到 8% 量级(失真的是修正的 8%, 不是 60%)。加载时 +1 还原。
 * 返回 1 = 有这个文件且读好了, 0 = 没这个文件, <0 = 有文件但坏了(不静默跳过: 显式给的目录里有
 * 坏文件, 跳过就是拿半份插件出假账)。 */
static int v41_gr_accum_layer(const char *dir, uint32_t il, float *acc, size_t nel, double *mb) {
    char p[4200]; snprintf(p, sizeof p, "%s/gr_L%02u.bin", dir, il);
    FILE *f = fopen(p, "rb");
    if (!f) return 0;
    int32_t hd[3];
    if (fread(hd, 4, 3, f) != 3 || hd[0] != (int32_t)DS4_N_EXPERT || hd[1] != (int32_t)DS4_N_EMBD ||
        (hd[2] != 1 && hd[2] != 2 && hd[2] != (int32_t)DS4_GGT_FP4X32)) {
        fprintf(stderr, "ds4: 增益覆盖 %s 头不对(专家 %d 通道 %d 类型 %d; 要 %u/%u/1|2|43)\n",
                p, hd[0], hd[1], hd[2], (unsigned)DS4_N_EXPERT, (unsigned)DS4_N_EMBD);
        fclose(f); return -1;
    }
    bool ok;
    double bytes;
    if (hd[2] == (int32_t)DS4_GGT_FP4X32) {
        if (nel % 32u) { fprintf(stderr, "ds4: %s 元素数 %zu 不是 32 的整数倍, fp4x32 装不下\n", p, nel); fclose(f); return -1; }
        const size_t nb = nel / 32u, nby = nb * 17u;
        uint8_t *pk = xmalloc(nby);
        float *raw = xmalloc(nel * 4);
        ok = fread(pk, 1, nby, f) == nby;
        if (ok) {
            ds4_deq_fp4x32(pk, nb, raw);
            for (size_t t = 0; t < nel; t++) acc[t] *= 1.0f + raw[t];   /* 盘上是 s−1 */
        }
        free(pk); free(raw);
        bytes = (double)nby;
    } else if (hd[2] == 2) {   /* f16(s 本身): 1 附近分辨率 2^-10 ≈ 0.001 */
        uint16_t *raw = xmalloc(nel * 2);
        ok = fread(raw, 2, nel, f) == nel;
        if (ok) for (size_t t = 0; t < nel; t++) acc[t] *= ds4_f16_to_f32(raw[t]);
        free(raw);
        bytes = (double)nel * 2;
    } else {
        float *raw = xmalloc(nel * 4);
        ok = fread(raw, 4, nel, f) == nel;
        if (ok) for (size_t t = 0; t < nel; t++) acc[t] *= raw[t];
        free(raw);
        bytes = (double)nel * 4;
    }
    fclose(f);
    if (!ok) { fprintf(stderr, "ds4: 增益覆盖 %s 截断\n", p); return -1; }
    *mb += bytes / 1e6;
    return 1;
}

/* 路由偏置侧车(2026-09-20, 路由反修的 V4.1 形态): rb_Lnn.bin = <i32 n_expert><i32 1=f32> + Δb[n_expert] f32(拟合侧已乘 α)。
 * 引擎把它加进该层路由的【选择分】(exp_probs_b 那一项), 权重分不动 —— 与 V4 路由偏置侧车同语义。
 * 设备表是进程级的: 每次加载先全卸, 没文件的层就是裸路由。返回 1 挂了 / 0 没文件 / <0 坏文件(不静默跳过)。 */
static int v41_rb_load_layer(const char *dir, uint32_t il, double *mb) {
    char p[4200]; snprintf(p, sizeof p, "%s/rb_L%02u.bin", dir, il);
    FILE *f = fopen(p, "rb");
    if (!f) return 0;
    int32_t hd[2];
    if (fread(hd, 4, 2, f) != 2 || hd[0] != (int32_t)DS4_N_EXPERT || hd[1] != 1) {
        fprintf(stderr, "ds4: 路由偏置 %s 头不对(专家 %d 类型 %d; 要 %u/1=f32)\n", p, hd[0], hd[1], (unsigned)DS4_N_EXPERT);
        fclose(f); return -1;
    }
    float *buf = xmalloc((size_t)DS4_N_EXPERT * 4);
    const bool ok = fread(buf, 4, DS4_N_EXPERT, f) == DS4_N_EXPERT;
    fclose(f);
    if (!ok) { fprintf(stderr, "ds4: 路由偏置 %s 截断\n", p); free(buf); return -1; }
    const ds4_model *m = g_ds4_v41_model;
    char nm[64]; snprintf(nm, sizeof nm, "blk.%u.exp_probs_b.bias", il);
    const ds4_tensor *t = m ? model_find_tensor(m, nm) : NULL;
    if (!t) { fprintf(stderr, "ds4: 路由偏置 L%02u: 模型里找不到 %s\n", il, nm); free(buf); return -1; }
    const int r = ds4_gpu_v41_set_rb_override(m->map, m->size, t->abs_offset, buf, (uint32_t)DS4_N_EXPERT);
    free(buf);
    if (!r) { fprintf(stderr, "ds4: 路由偏置 L%02u 挂不上\n", il); return -1; }
    *mb += (double)DS4_N_EXPERT * 4 / 1e6;
    return 1;
}
/* 目录里所有层的路由偏置; dir 为空 = 只全卸。返回挂上的层数, <0 = 有文件但读坏了。 */
static int v41_rb_load(const char *dir) {
    (void)ds4_gpu_v41_set_rb_override(NULL, 0, 0, NULL, 0);
    int n = 0; double mb = 0;
    for (uint32_t il = 0; dir && dir[0] && il < DS4_N_LAYER; il++) {
        const int r = v41_rb_load_layer(dir, il, &mb);
        if (r < 0) return -1;
        n += r;
    }
    if (n) fprintf(stderr, "ds4: [反修·路由] %d 层挂上路由偏置侧车 (盘上 %.3f MB)\n", n, mb);
    return n;
}
/* ★进程级插件缓存★(2026-10-17, 用户"切换体验速度"): 同一组 (②目录, ③目录, β) 只读一次盘、只上传一次。
 * 以前每个状态(每条请求态 + 批态 + 切换的试探态)都从盘重读 ②+③(146 MB: ② gr fp4 在 CPU 解码 + ③ amp f32 105 MB)、
 * 重算 ② 的 40 MB 指纹、重传设备 —— 17 token 的请求从 prompt start 到 done 0.42 s, 这段是大头; 切换插件要装三遍。
 * 现在: 放大器 A/B 设备张量住这里, 状态只拿指针(amp_shared=1, 释放时只减引用); gr/rb 进程级表装过就不动。
 * 盘上内容变没变只以切换为准: ds4_engine_v41_switch_plugins 先 v41_amp_cache_drop 再装 —— 页面开关关再开 = 重读盘。
 * 出错会怎样: 漏标 amp_shared 就会把共享张量当自己的 free, 下一个状态读到释放掉的显存(不报错, 出胡话)。 */
static struct {
    int valid; char amp[2056], pt[2056]; float scale;
    ds4_gpu_tensor *A[DS4_MAX_LAYER], *B[DS4_MAX_LAYER], *S[DS4_MAX_LAYER]; uint32_t K[DS4_MAX_LAYER];
    uint32_t n_arm, kmax, n_route;
    int refs;   /* 正拿着这些指针的状态数; >0 时不能丢 */
} g_pc;

static bool pc_same(const char *dir, const char *pt) {
    return g_pc.valid && !strcmp(g_pc.amp, dir ? dir : "") && !strcmp(g_pc.pt, pt ? pt : "") && g_pc.scale == g_ds4_v41_amp_scale;
}
void v41_amp_cache_drop(void) {
    if (!g_pc.valid) return;
    if (g_pc.refs) { fprintf(stderr, "ds4: ★插件缓存还有 %d 个状态在用, 不能丢(切换插件时不许有活着的状态)★\n", g_pc.refs); return; }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        if (g_pc.A[il]) ds4_gpu_tensor_free(g_pc.A[il]);
        if (g_pc.B[il]) ds4_gpu_tensor_free(g_pc.B[il]);
        if (g_pc.S[il]) ds4_gpu_tensor_free(g_pc.S[il]);
    }
    memset(&g_pc, 0, sizeof g_pc);
}
/* 状态拿缓存的指针; ampT 是每个状态自己的(按 cap 开) */
static bool pc_attach(ds4_v41_state *st) {
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) { st->ampA[il] = g_pc.A[il]; st->ampB[il] = g_pc.B[il]; st->ampS[il] = g_pc.S[il]; st->ampK[il] = g_pc.K[il]; }
    st->amp_shared = 1; g_pc.refs++;
    if (!g_pc.n_arm) return true;
    st->ampT = ds4_gpu_tensor_alloc((uint64_t)st->cap_tok * g_pc.kmax * 4);
    return st->ampT != NULL;
}

/* 训练态(core_ptrain_bwd.c pt_run_alloc)的 A/B/ampT 是它自己开的, 不该占插件缓存的引用。占着会怎样(10-22 实撞, 10-17 缓存上线后第一次训 ③):
 * 轮末探针要挂 (②, ckpt_eNN), 探针批态装插件时缓存换不了组(refs=1 拒丢) → "插件挂不上, 停车", 训练在轮 1 末退出; 退出时 v41_amp_free 又把
 * 训练态自己的张量当共享的漏放。只在状态没拿缓存张量时允许(② 没有低秩件 —— 训练器本来就不接按秩拼接), 拿着就返回 false 让调用方停车:
 * 认领了缓存的张量, 释放时会把别的状态还在用的显存 free 掉。 */
bool v41_amp_disown(ds4_v41_state *st) {
    if (!st->amp_shared) return true;
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) if (st->ampA[il] || st->ampB[il] || st->ampS[il]) return false;
    st->amp_shared = 0;
    if (g_pc.refs > 0) g_pc.refs--;
    return true;
}

/* 训练态叠在【带低秩件的 ②】上训 ③(2026-11-01, 用户: 行为类的轮不该碰 ② 的编程指标): 把缓存里 ② 的 A/B 拷成状态自己的张量(秩 k2 原样),
 * 放掉缓存引用; 调用方(core_ptrain_bwd.c pt_run_alloc)再把它扩成 [k2 + k3] 行 —— 前 k2 行冻结(梯度每步清零), 后 k3 行是 ③, 落盘只写后 k3 行。
 * 返回 ② 的秩 k2(训练层范围内各层要一样, 带路由 S 的 ② 不接 → −1); ② 没低秩件时等价 v41_amp_disown, 返回 0。
 * 为什么不改应用核: 引擎挂 ②+③ 本来就是按秩拼接成一份 A/B(v41_amp_load), 训练态里同一形状, 前向/反传/落盘后的挂载逐位同一条路。 */
int v41_amp_adopt(ds4_v41_state *st, uint32_t lo, uint32_t hi) {
    if (!st->amp_shared) return 0;
    uint32_t k2 = 0;
    for (uint32_t il = lo; il <= hi && il < DS4_MAX_LAYER; il++) {
        if (st->ampS[il]) { fprintf(stderr, "ds4: [ptrain] ② 在 L%u 带路由系数, 叠 ③ 不接\n", il); return -1; }
        if (!st->ampA[il]) continue;
        if (k2 && st->ampK[il] != k2) { fprintf(stderr, "ds4: [ptrain] ② 各层秩不一(L%u %u vs %u), 叠 ③ 不接\n", il, st->ampK[il], k2); return -1; }
        k2 = st->ampK[il];
    }
    if (!k2) { for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) st->ampA[il] = st->ampB[il] = st->ampS[il] = NULL, st->ampK[il] = 0; }
    else {
        float *buf = NULL; size_t bn = 0;
        for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {   /* 训练范围外有 ② 的层也拷成自己的: 放掉引用后不能再指着缓存 */
            if (!st->ampA[il]) continue;
            const uint64_t nel = (uint64_t)st->ampK[il] * DS4_N_EMBD;
            if (bn < nel * 4) { free(buf); buf = xmalloc((size_t)nel * 4); bn = nel * 4; }
            ds4_gpu_tensor *A = ds4_gpu_tensor_alloc(nel * 4), *B = ds4_gpu_tensor_alloc(nel * 4);
            const bool ok = A && B && ds4_gpu_tensor_read(st->ampA[il], 0, buf, nel * 4) && ds4_gpu_tensor_write(A, 0, buf, nel * 4) &&
                            ds4_gpu_tensor_read(st->ampB[il], 0, buf, nel * 4) && ds4_gpu_tensor_write(B, 0, buf, nel * 4);
            if (!ok) { fprintf(stderr, "ds4: [ptrain] 拷 ② L%u 放大器失败\n", il); if (A) ds4_gpu_tensor_free(A); if (B) ds4_gpu_tensor_free(B); free(buf); return -1; }
            st->ampA[il] = A; st->ampB[il] = B; st->ampS[il] = NULL;
        }
        free(buf);
    }
    st->amp_shared = 0;
    if (g_pc.refs > 0) g_pc.refs--;
    return (int)k2;
}

/* 裸底座: 两张进程级表全卸 + 缓存丢掉。★gr 也要卸★: 以前只卸 rb, 从"挂 ②③"切到"不挂"时上一套的增益覆盖还留在设备上, 裸底座不裸 */
void v41_plugins_clear_all(void) {
    (void)v41_rb_load(NULL);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) (void)ds4_gpu_v41_set_gr_override(il, NULL, 0, 0);
    v41_amp_cache_drop();
}

/* 三文件部署(2026-09-13): ②反修目录与③后训练目录的增益表【逐元素相乘】后一次上设备 ——
 *     g_eff = 载荷 g_r × s₂ × s₃
 * 盘上权重一个字节不动; 任一目录不挂, 该项就是 1(数学上就是不挂)。为什么必须先乘再上传:
 * ds4_gpu_v41_set_gr_override 是"这一层的表就是它", 两个目录各 set 一次 = 后者顶掉前者,
 * 而且不报错 —— 那就是安静地只挂了一件插件, 读数还长得挺合理。
 * 返回挂上的层数, <0 = 有文件但读坏了。 */
static int v41_gr_load(const char *amp_dir, const char *pt_dir) {
    int n = 0; double mb = 0;
    const size_t nel = (size_t)DS4_N_EXPERT * DS4_N_EMBD;
    float *acc = xmalloc(nel * 4);
    int rc = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER && rc >= 0; il++) {
        for (size_t t = 0; t < nel; t++) acc[t] = 1.0f;
        int got = 0;
        for (int src = 0; src < 2; src++) {
            const char *d = src == 0 ? amp_dir : pt_dir;
            if (!d || !d[0]) continue;
            const int r = v41_gr_accum_layer(d, il, acc, nel, &mb);
            if (r < 0) { rc = -1; break; }
            got += r;
        }
        if (rc < 0) continue;
        if (!got) { (void)ds4_gpu_v41_set_gr_override(il, NULL, 0, 0); continue; }   /* 这层没文件 = 裸; 上一套插件在这层的表不能留 */
        if (!ds4_gpu_v41_set_gr_override(il, acc, (uint32_t)DS4_N_EXPERT, (uint32_t)DS4_N_EMBD)) {
            fprintf(stderr, "ds4: L%02u 增益覆盖上传失败\n", il); rc = -1; break;
        }
        n++;
    }
    free(acc);
    if (rc < 0) return -1;
    if (n) fprintf(stderr, "ds4: [反修·权重侧] %d 层挂上逐专家增益覆盖 (盘上 %.1f MB%s)\n",
                   n, mb, (pt_dir && pt_dir[0]) ? ", 含后训练第三件" : "");
    return n;
}

/* 后训练目录的底座指纹核对(三文件部署的硬闸)。③ 是对"①+②这个部署态"解出来的修正:
 * ② 换了版, ③ 就打在错的基线上, 而且照样能跑、照样出一个像模像样的读数。所以 ③ 落盘时
 * 把 ② 目录的指纹写进 base.fnv, 加载时重算比对。指纹 = 按层号顺序对 ② 的 gr_Lnn.bin 全字节
 * 做 FNV-1a 64(不引哈希库; 这里防的是"拿错了文件", 不是防篡改)。
 * ③ 目录没有 base.fnv => 打警告放行(手搓的实验产物), 有但对不上 => 停车。 */
static bool v41_pt_base_ok(const char *pt_dir, const char *amp_dir) {
    char p[4200]; snprintf(p, sizeof p, "%s/base.fnv", pt_dir);
    FILE *f = fopen(p, "rb");
    if (!f) {
        fprintf(stderr, "ds4: ★后训练目录 %s 没有 base.fnv, 无法核对它是解在哪个反修态上的 —— 放行但读数自负★\n", pt_dir);
        return true;
    }
    unsigned long long want = 0; unsigned want_n = 0;
    const int got = fscanf(f, "%llx %u", &want, &want_n);
    fclose(f);
    if (got != 2) { fprintf(stderr, "ds4: %s 读不出指纹 -- aborting\n", p); return false; }
    uint32_t have_n = 0;
    unsigned hn = 0;
    const uint64_t have = (amp_dir && amp_dir[0]) ? ds4_gr_dir_fnv(amp_dir, DS4_N_LAYER, &hn) : DS4_GR_FNV_SEED;
    have_n = hn;
    if (have != (uint64_t)want || have_n != want_n) {
        fprintf(stderr, "ds4: ★后训练件 %s 是解在另一个反修态上的(指纹 %016llx/%u 层, 现挂 %016llx/%u 层)★\n"
                        "     ② 换版后 ③ 必须重解 —— 停车, 不出假读数\n",
                pt_dir, want, want_n, (unsigned long long)have, have_n);
        return false;
    }
    fprintf(stderr, "ds4: [后训练] 底座指纹核对通过(%016llx, %u 层)\n", (unsigned long long)have, have_n);
    return true;
}

/* 读一个目录里第 il 层的放大器 amp_Lnn.bin 到主机 f32(A 与 B 各 [K][D], 本函数分配, 调用方 free)。
 * 返回 K(>0) / 0 = 没这个文件 / <0 = 有文件但坏了(显式给的目录里有坏文件, 跳过就是拿半份插件出假账)。
 * scale 乘进 A: y += x·(B·(βA)) = β·(x·B·A) ⇒ 整层修正缩到 β 倍; 乘 A 不乘 B 是因为两发 GEMM 里 A 是第二发的权重, 缩它不改第一发 T = x·B 的数值范围。
 * mb 累加【盘上真实字节】(旧版固定按 f16 估, 盘上落的却是 f32, 报了一半的假账)。 */
static int amp_read_layer(const char *dir, uint32_t il, float **A, float **B, float scale, double *mb, const char **tyname) {
    *A = *B = NULL;
    if (!dir || !dir[0]) return 0;
    char p[4200]; snprintf(p, sizeof p, "%s/amp_L%02u.bin", dir, il);
    FILE *f = fopen(p, "rb");
    if (!f) return 0;
    int32_t hd[3];
    if (fread(hd, 4, 3, f) != 3 || hd[0] != (int32_t)DS4_N_EMBD || hd[1] <= 0 || hd[1] > 8192) {
        fprintf(stderr, "ds4: 放大器 %s 头不对(D %d K %d)\n", p, hd[0], hd[1]); fclose(f); return -1;
    }
    const uint32_t D = (uint32_t)hd[0], K = (uint32_t)hd[1];
    const int32_t ty = hd[2];
    if (ty != 1 && ty != (int32_t)DS4_GGT_FP4X32) {
        fprintf(stderr, "ds4: 放大器 %s 存储类型 %d 不认(1=f32 / 43=fp4x32)\n", p, ty); fclose(f); return -1;
    }
    const size_t nel = (size_t)K * D;
    if (ty == (int32_t)DS4_GGT_FP4X32 && nel % 32u) {
        fprintf(stderr, "ds4: 放大器 %s K×D=%zu 不是 32 的整数倍, fp4x32 装不下\n", p, nel); fclose(f); return -1;
    }
    float *a = xmalloc(nel * 4), *b = xmalloc(nel * 4);
    uint8_t *pk = ty == (int32_t)DS4_GGT_FP4X32 ? xmalloc(nel / 32u * 17u) : NULL;
    const bool ok = amp_read_mat(f, ty, nel, a, pk) && amp_read_mat(f, ty, nel, b, pk);
    free(pk); fclose(f);
    if (!ok) { fprintf(stderr, "ds4: 放大器 %s 读失败(截断)\n", p); free(a); free(b); return -1; }
    if (scale != 1.0f) for (size_t t = 0; t < nel; t++) a[t] *= scale;
    *mb += ty == (int32_t)DS4_GGT_FP4X32 ? 2.0 * nel / 32.0 * 17.0 / 1e6 : 2.0 * nel * 4.0 / 1e6;
    *tyname = ty == (int32_t)DS4_GGT_FP4X32 ? "fp4x32" : "f32";
    *A = a; *B = b;
    return (int)K;
}

/* 带路由的放大器系数 ampr_Lnn.bin({NE, K, 1} + f32 S[NE][K], 10-29 段 B 试验件, 见 src/cuda/cuda_amp_route.inc.cu): 只认 ② 目录、且这层没有 ③ 拼进来的秩
 * (S 的 K 是 ② 自己那份的秩; 拼了 ③ 再乘就错位)。返回 1 挂上 / 0 没这个文件 / <0 有文件但对不上(停车, 不悄悄当成不带路由)。 */
static int amp_route_read(const char *dir, uint32_t il, uint32_t k2, uint32_t k3, ds4_gpu_tensor **S, double *mb) {
    *S = NULL;
    if (!dir || !dir[0]) return 0;
    char p[4200]; snprintf(p, sizeof p, "%s/ampr_L%02u.bin", dir, il);
    FILE *f = fopen(p, "rb");
    if (!f) return 0;
    int32_t hd[3] = { 0, 0, 0 };
    const size_t nel = (size_t)DS4_N_EXPERT * k2;
    float *buf = NULL;
    bool ok = fread(hd, 4, 3, f) == 3 && hd[0] == (int32_t)DS4_N_EXPERT && hd[1] == (int32_t)k2 && hd[2] == 1 && k2 > 0 && k3 == 0;
    if (ok) { buf = xmalloc(nel * 4); ok = fread(buf, 4, nel, f) == nel; }
    fclose(f);
    if (ok) ok = (*S = ds4_gpu_tensor_alloc(nel * 4)) != NULL && ds4_gpu_tensor_write(*S, 0, buf, nel * 4);
    free(buf);
    if (!ok) { fprintf(stderr, "ds4: ★带路由系数 %s 对不上(要 NE %u、K = 本层 ② 的秩 %u、f32、这层没有 ③ 的秩; 头 %d/%d/%d)★\n", p, DS4_N_EXPERT, k2, hd[0], hd[1], hd[2]); return -1; }
    *mb += (double)nel * 4 / 1e6;
    return 1;
}

bool v41_amp_load(ds4_v41_state *st, const char *dir, const char *pt_dir) {
    if (pc_same(dir, pt_dir)) return pc_attach(st);   /* 装过同一组: 不读盘不上传 */
    v41_amp_cache_drop();
    if (g_pc.valid) return false;   /* 有状态还拿着另一组插件(上面已喊): 不许两套表混在一个进程里 */
    uint32_t n_arm = 0, kmin = 0, kmax = 0; double mb = 0.0;
    const char *tyname = "";
    if (pt_dir && pt_dir[0] && !v41_pt_base_ok(pt_dir, dir)) return false;
    const int n_rb = v41_rb_load(dir);   /* 路由偏置(先全卸再挂) */
    if (n_rb < 0) return false;
    const int n_gr = v41_gr_load(dir, pt_dir);
    if (n_gr < 0) return false;
    /* ★②③ 按秩拼接★(2026-10-01, 09-08 三文件设计的落地): 两个目录同一层各有 amp_Lnn.bin 时
     *   y += x·B₂·A₂ + x·B₃·A₃ = x·[B₂|B₃]·[A₂;A₃] —— A/B 内存都是 [K][D] 行主序, 拼接 = 两段顺序摆放, 核与应用零改动。
     * β(--zchain-scale)只缩 ②: 它是反修的步长旋钮, ③ 是另一件事。以前 ③ 目录只认增益表, 低秩候选只能靠"配对目录"
     * (② 软链 + 候选 amp 当 ② 挂); 门 = 同一候选两种挂法输出逐字节同(fable5 10-01)。 */
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        float *A2 = NULL, *B2 = NULL, *A3 = NULL, *B3 = NULL;
        const int k2 = amp_read_layer(dir, il, &A2, &B2, g_ds4_v41_amp_scale, &mb, &tyname);
        const int k3 = k2 < 0 ? 0 : amp_read_layer(pt_dir, il, &A3, &B3, 1.0f, &mb, &tyname);
        if (k2 < 0 || k3 < 0) { free(A2); free(B2); g_pc.valid = 1; v41_amp_cache_drop(); return false; }   /* 前几层已上传的随缓存放掉 */
        const uint32_t K = (uint32_t)k2 + (uint32_t)k3, D = DS4_N_EMBD;
        if (!K) continue;
        if (K > 8192u) { fprintf(stderr, "ds4: L%02u 放大器 ②(K %d)+③(K %d) 拼接超 8192\n", il, k2, k3); free(A2); free(B2); free(A3); free(B3); g_pc.valid = 1; v41_amp_cache_drop(); return false; }
        const uint64_t nel = (uint64_t)K * D, o3 = (uint64_t)k2 * D * 4u;
        bool ok = true;
        g_pc.A[il] = ds4_gpu_tensor_alloc(nel * 4); g_pc.B[il] = ds4_gpu_tensor_alloc(nel * 4);
        if (!g_pc.A[il] || !g_pc.B[il]) ok = false;
        if (ok && k2 > 0 && (!ds4_gpu_tensor_write(g_pc.A[il], 0, A2, o3) || !ds4_gpu_tensor_write(g_pc.B[il], 0, B2, o3))) ok = false;
        if (ok && k3 > 0 && (!ds4_gpu_tensor_write(g_pc.A[il], o3, A3, (uint64_t)k3 * D * 4u) ||
                             !ds4_gpu_tensor_write(g_pc.B[il], o3, B3, (uint64_t)k3 * D * 4u))) ok = false;
        free(A2); free(B2); free(A3); free(B3);
        if (!ok) { fprintf(stderr, "ds4: L%02u 放大器上传失败\n", il); g_pc.valid = 1; v41_amp_cache_drop(); return false; }   /* 半装的张量随缓存一起放 */
        g_pc.K[il] = K;
        const int kr = amp_route_read(dir, il, (uint32_t)k2, (uint32_t)k3, &g_pc.S[il], &mb);
        if (kr < 0) { g_pc.valid = 1; v41_amp_cache_drop(); return false; }
        g_pc.n_route += (uint32_t)kr;
        if (!n_arm || K < kmin) kmin = K;
        if (K > kmax) kmax = K;
        n_arm++;
    }
    /* 只有增益覆盖/路由偏置、没有低秩放大器也是合法的一种插件(权重侧反修与后训练的产物都长这样) */
    if (!n_arm && n_gr <= 0 && n_rb <= 0) {
        fprintf(stderr, "ds4: 目录里既没有 amp_Lnn.bin 也没有 gr_Lnn.bin / rb_Lnn.bin(反修 %s / 后训练 %s)\n",
                dir && dir[0] ? dir : "(无)", pt_dir && pt_dir[0] ? pt_dir : "(无)");
        return false;
    }
    g_pc.valid = 1; g_pc.scale = g_ds4_v41_amp_scale; g_pc.n_arm = n_arm; g_pc.kmax = kmax;
    snprintf(g_pc.amp, sizeof g_pc.amp, "%s", dir ? dir : ""); snprintf(g_pc.pt, sizeof g_pc.pt, "%s", pt_dir ? pt_dir : "");
    if (n_arm) fprintf(stderr, "ds4: [反修] 挂上 %u 层放大器 (K %u~%u, %s 盘上合计 %.1f MB, 步长 β=%.4g%s) ← ② %s%s%s\n",
                       n_arm, kmin, kmax, tyname, mb, (double)g_ds4_v41_amp_scale, g_pc.n_route ? ", 带路由" : "", dir && dir[0] ? dir : "(无)",
                       pt_dir && pt_dir[0] ? " + ③ " : "", pt_dir && pt_dir[0] ? pt_dir : "");
    return pc_attach(st);
}

/* 反修取料钩子: 放大器应用前把这一层的 MoE 入/出与路由交给解算方(主机内存)。先 flush+synchronize —— y 刚由 GPU 写完,
 * 不同步就 D2H 读到的是半成品。回调说 1 = 取完了 ⇒ 置 stop_early, 前向驱动跳过余下层与出口。 */
int v41_amp_hook(ds4_v41_state *st, uint32_t il) {
    if (!g_ds4_v41_hook) return 0;
    /* 层过滤要在所有拷贝之前: 下面那串 D2H 一层就是几十 MB(prefill 路还有 63 MB 的逐专家输出),
     * 取料方只要一层时, 不过滤等于每块白拷 39 层。 */
    if (g_ds4_v41_hook_layer >= 0 && (int)il != g_ds4_v41_hook_layer) return 0;
    const uint32_t n = st->n, D = DS4_N_EMBD, NU = DS4_N_EXPERT_USED;
    if (!ds4_gpu_flush_commands() || !ds4_gpu_synchronize()) return -1;
    const uint32_t HC = DS4_N_HC;
    float *x = xmalloc((size_t)n * D * 4), *y = xmalloc((size_t)n * D * 4), *rw = xmalloc((size_t)n * NU * 4);
    int32_t *sel = xmalloc((size_t)n * NU * 4);
    /* α[i] = Σ_k pre[i][k]·post[i][k]: y 经 hc_post 进各路、再由 hc_pre 用本层 ffn_pre 合成回来的系数。
     * 此刻 st->pre 正是本层 ffn_pre(层末才与 pre_mix 交换), st->post 是 ffn 的 post —— 末层出口用的就是这两件。 */
    float *pre = xmalloc((size_t)n * HC * 4), *post = xmalloc((size_t)n * HC * 4), *alpha = xmalloc((size_t)n * 4);
    /* 逐专家 down 输出(未乘路由权重): 只有 prefill GEMM 路物化, 取不到就递 NULL —— 不猜、不另算一遍。
     * 63 MB(n=512) 一层一块, 解码路(n≤8)与取不到时都不分配。 */
    float *ye = NULL, *ysh = NULL;
    if (n > 8u) {
        ye = xmalloc((size_t)n * NU * D * 4);
        ysh = xmalloc((size_t)n * D * 4);
        if (!ds4_gpu_v41_vq_capture_expert_out(ye, n, NU, D) ||
            !ds4_gpu_tensor_read(st->so, 0, ysh, (uint64_t)n * D * 4)) { free(ye); free(ysh); ye = ysh = NULL; }
    }
    int rc = -1;
    if (ds4_gpu_tensor_read(st->xn, 0, x, (uint64_t)n * D * 4) && ds4_gpu_tensor_read(st->y, 0, y, (uint64_t)n * D * 4) &&
        ds4_gpu_tensor_read(st->sel, 0, sel, (uint64_t)n * NU * 4) && ds4_gpu_tensor_read(st->rw, 0, rw, (uint64_t)n * NU * 4) &&
        ds4_gpu_tensor_read(st->pre, 0, pre, (uint64_t)n * HC * 4) && ds4_gpu_tensor_read(st->post, 0, post, (uint64_t)n * HC * 4)) {
        for (uint32_t i = 0; i < n; i++) {
            float a = 0.f;
            for (uint32_t k = 0; k < HC; k++) a += pre[i * HC + k] * post[i * HC + k];
            alpha[i] = a;
        }
        rc = g_ds4_v41_hook(g_ds4_v41_hook_ud, (int)il, (int)st->pos0, (int)n, (int)D, (int)NU, DS4_SWIGLU_CLAMP_EXP, x, y, sel, rw, alpha, ye, ysh);
        if (rc < 0) fprintf(stderr, "ds4: [反修钩子] L%02u 回调要求停车(rc %d)\n", il, rc);
        else if (rc > 0) { st->stop_early = 1; rc = 1; }
    } else fprintf(stderr, "ds4: [反修钩子] L%02u 取 x/y/路由/hc 系数 下主机失败\n", il);
    free(x); free(y); free(sel); free(rw); free(pre); free(post); free(alpha); free(ye); free(ysh);
    return rc;
}

void v41_amp_free(ds4_v41_state *st) {
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        if (!st->amp_shared) {   /* 自己的(草稿态塔件, core_v41_draft_amp.c); 共享的归缓存, 下面只减引用 */
            if (st->ampA[il]) ds4_gpu_tensor_free(st->ampA[il]);
            if (st->ampB[il]) ds4_gpu_tensor_free(st->ampB[il]);
            if (st->ampS[il]) ds4_gpu_tensor_free(st->ampS[il]);
        }
        st->ampA[il] = st->ampB[il] = st->ampS[il] = NULL; st->ampK[il] = 0;
    }
    if (st->amp_shared) { st->amp_shared = 0; if (g_pc.refs > 0) g_pc.refs--; }
    if (st->ampT) { ds4_gpu_tensor_free(st->ampT); st->ampT = NULL; }
}

/* y[n][D](bf16 格点) += x[n][D]·(B·A) → 舍 bf16 */
bool v41_amp_apply(ds4_v41_state *st, uint32_t il) {
    if (!st->ampA[il]) return true;
    if (st->ampS[il] && !st->draft) {   /* 带路由: 本层路由 st->sel/rw 此刻还是 v41_moe 刚算的 */
        if (!ds4_gpu_v41_amp_route_apply_tensor(st->y, st->xn, st->ampA[il], st->ampB[il], st->ampS[il], st->sel, st->rw, st->ampT, st->n,
                                                DS4_N_EMBD, st->ampK[il], DS4_N_EXPERT_USED, DS4_N_EXPERT)) return false;
        return ds4_gpu_v41_round_bf16_tensor(st->y, (uint64_t)st->n * DS4_N_EMBD) != 0;
    }
    /* 草稿态(塔件)走小批两发小核(cuda_draft_attn.inc.cu): cuBLAS 小 n 一发 ~0.2 ms, 草稿一轮 8 发把接受率的收益吃光(10-07 dkspeed 分账); 主干照旧 */
    if (st->draft ? !ds4_gpu_draft_amp_apply_tensor(st->y, st->xn, st->ampA[il], st->ampB[il], st->ampT, st->n, DS4_N_EMBD, st->ampK[il])
                  : !ds4_gpu_v41_amp_apply_tensor(st->y, st->xn, st->ampA[il], st->ampB[il], st->ampT, st->n, DS4_N_EMBD, st->ampK[il])) return false;
    return ds4_gpu_v41_round_bf16_tensor(st->y, (uint64_t)st->n * DS4_N_EMBD) != 0;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_amp_nonempty_tu;
