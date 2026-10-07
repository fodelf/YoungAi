/* core_ptrain_teacher.c — 后训练 ③ 第八版: 教师 top-K(2026-10-01)。总述见 core_ptrain.h。
 *
 * 教师 = 部署模型(①+②, 不挂 ③)读了复盘块之后答题。同块各题的前缀(BOS + <｜User｜> + 块正文 + 空行 …)逐 id 相同,
 * 所以前缀只按部署口径预填一次(CED: 中间块只跑编码器段 + 分界层 KV, 与生成时长提示的前几块同一种状态), 存快照;
 * 每题还原快照后续算"前缀最后 PT_TAIL 个 token + 问题 + 答案"一块(跑满解码器 —— 生成时提示的末块也是这样, 末块至少要
 * 留一个窗口宽的 token, 否则解码器段看不见块尾那 128 个位置)。
 * 快照只要存会被续算改写的东西: 各层窗口环的历史段、压缩源层的余行、几个计数; 压缩 KV 按组号写, 前缀的组不会被覆盖。
 * 结果缓存到 out/teacher.bin(料不变 ⇒ 第二次训练直接读, 只做训练)。 */
#include "core_ptrain.h"
#ifndef DS4_NO_GPU

#define PT_TAIL 256u          /* 续算块里带上的前缀尾巴(≥ 窗口 128, 余量给注意力) */
#define PT_TROWS 1024u        /* 续算块的 logits 行上限(内存: 1024 × 129280 × 4 = 0.5 GB) */
/* 教师表算法的版本: 改了 top-K/余量的算法就 +1, 旧缓存自动作废(2 = 余量改为榜外直接累加, 10-01) */
#define PT_TEACHER_VER 2u

void pt_state_reset(ds4_v41_state *st) {
    st->n_past = 0; st->snap_on = 0; st->mainh_n = 0; st->mainh_end = -1; st->idx_owner = -1; st->cand_owner = -1;
    memset(st->cpend, 0, sizeof st->cpend); memset(st->ng_src, 0, sizeof st->ng_src); memset(st->win_from, 0, sizeof st->win_from);
}

typedef struct {
    ds4_gpu_tensor *win[DS4_MAX_LAYER], *ckv[DS4_MAX_LAYER], *csc[DS4_MAX_LAYER];
    uint32_t n_past, cpend[DS4_MAX_LAYER], ng_src[DS4_MAX_LAYER], win_from[DS4_MAX_LAYER];
} pt_snap;

static bool pt_snap_io(ds4_v41_state *st, pt_snap *sn, int save) {
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4, ringb = (uint64_t)DS4_N_SWA * rowb;
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        if (!sn->win[il]) sn->win[il] = v41_alloc(ringb, &ok);
        if (ok) ok = save ? ds4_gpu_tensor_copy(sn->win[il], 0, st->win[il], 0, ringb) : ds4_gpu_tensor_copy(st->win[il], 0, sn->win[il], 0, ringb);
        if (!ok || !st->cpre_kv[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint64_t nb = (uint64_t)ratio * rowb;
        if (!sn->ckv[il]) { sn->ckv[il] = v41_alloc(nb, &ok); sn->csc[il] = v41_alloc(nb, &ok); }
        if (ok) ok = save ? (ds4_gpu_tensor_copy(sn->ckv[il], 0, st->cpre_kv[il], 0, nb) && ds4_gpu_tensor_copy(sn->csc[il], 0, st->cpre_sc[il], 0, nb))
                          : (ds4_gpu_tensor_copy(st->cpre_kv[il], 0, sn->ckv[il], 0, nb) && ds4_gpu_tensor_copy(st->cpre_sc[il], 0, sn->csc[il], 0, nb));
    }
    if (save) { sn->n_past = st->n_past; memcpy(sn->cpend, st->cpend, sizeof sn->cpend); memcpy(sn->ng_src, st->ng_src, sizeof sn->ng_src); memcpy(sn->win_from, st->win_from, sizeof sn->win_from); }
    else { pt_state_reset(st); st->n_past = sn->n_past; memcpy(st->cpend, sn->cpend, sizeof sn->cpend); memcpy(st->ng_src, sn->ng_src, sizeof sn->ng_src); memcpy(st->win_from, sn->win_from, sizeof sn->win_from); }
    return ok;
}
static void pt_snap_free(pt_snap *sn) {
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        if (sn->win[il]) ds4_gpu_tensor_free(sn->win[il]);
        if (sn->ckv[il]) ds4_gpu_tensor_free(sn->ckv[il]);
        if (sn->csc[il]) ds4_gpu_tensor_free(sn->csc[il]);
    }
    memset(sn, 0, sizeof *sn);
}

/* ★教师表按题缓存★(10-03, 每日叠加训练): 键 = FNV(算法版本, K, 教师序列长, 答案起点, 全部教师 ids) —— 块正文、问题、答案任一字变就是
 * 另一道题; 与学生渲染无关(教师分布只由教师序列决定)。文件 = 头 {magic, K} + 记录串 {键 u64, m u32, top_id[m·K] i32, top_p[m·K] f32,
 * top_rest[m] f32}, 只追加, 多日共用一份(cfg teacher=; 缺省 out/teacher.bin)。
 * 为什么按题不按整份料: 叠加训练的清单 = 当日新题 + 回放旧题, 按整份料哈希存的表(10-01 版)一混进回放就整份重算(~1 s/题);
 * 按题存则回放永远命中, 一天只算当日新题。同一题在清单里出现几次(保持料列几遍)只算一次, 其余拷贝。 */
#define PT_CACHE_MAGIC 0x324b5450u   /* 'PTK2'; 10-01 的整份料格式是 'PTK1', 见 pt_cache_v1 */

static uint64_t pt_sample_key(const pt_sample *s, uint32_t K) {
    uint64_t h = 1469598103934665603ull;
    #define PT_H(x) do { h ^= (uint64_t)(uint32_t)(x); h *= 1099511628211ull; } while (0)
    PT_H(PT_TEACHER_VER); PT_H(K); PT_H(s->tn); PT_H(s->ta0);
    for (uint32_t t = 0; t < s->tn; t++) PT_H(s->tids[t]);
    #undef PT_H
    return h;
}

/* 键 → 代表题号(清单里第一次出现的那道)的开放寻址表; 槽空 = UINT32_MAX */
typedef struct { uint64_t *k; uint32_t *v, mask; } pt_kmap;
static uint32_t *pt_kmap_slot(pt_kmap *m, uint64_t key) {
    uint32_t i = (uint32_t)(key ^ (key >> 29)) & m->mask;
    while (m->v[i] != UINT32_MAX && m->k[i] != key) i = (i + 1u) & m->mask;
    m->k[i] = key;
    return &m->v[i];
}

/* 10-01 的整份料格式(头 'PTK1' + 份数 + K + 整份哈希, 逐题串排): 只为同清单的老目录(kd-review_v2-* 的 teacher.bin)不白算一遍 ——
 * 读进来之后按新格式重写整个文件。整份哈希里带学生长度 sn, 照旧算 */
static bool pt_cache_v1(const char *path, pt_data *d, uint32_t K) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    uint64_t h = 1469598103934665603ull;
    #define PT_H(x) do { h ^= (uint64_t)(uint32_t)(x); h *= 1099511628211ull; } while (0)
    PT_H(PT_TEACHER_VER); PT_H(K); PT_H(d->ns);
    for (uint32_t i = 0; i < d->ns; i++) {
        const pt_sample *s = &d->s[i];
        PT_H(s->tn); PT_H(s->ta0); PT_H(s->sn);
        for (uint32_t t = 0; t < s->tn; t++) PT_H(s->tids[t]);
    }
    #undef PT_H
    uint32_t hd[3] = { 0, 0, 0 }; uint64_t k2 = 0;
    bool ok = fread(hd, 4, 3, f) == 3 && fread(&k2, 8, 1, f) == 1 && hd[0] == 0x314b5450u && hd[1] == d->ns && hd[2] == K && k2 == h;
    for (uint32_t i = 0; ok && i < d->ns; i++) {
        pt_sample *s = &d->s[i];
        uint32_t m = 0;
        ok = fread(&m, 4, 1, f) == 1 && (m == 0 || m == s->m);
        if (!ok || !m) continue;
        const size_t mk = (size_t)m * K;
        s->top_id = xmalloc(mk * 4); s->top_p = xmalloc(mk * 4); s->top_rest = xmalloc((size_t)m * 4);
        ok = fread(s->top_id, 4, mk, f) == mk && fread(s->top_p, 4, mk, f) == mk && fread(s->top_rest, 4, m, f) == m;
    }
    fclose(f);
    if (!ok) for (uint32_t i = 0; i < d->ns; i++) { pt_sample *s = &d->s[i]; free(s->top_id); free(s->top_p); free(s->top_rest); s->top_id = NULL; s->top_p = s->top_rest = NULL; }
    return ok;
}

/* 读按题缓存: 代表题命中就填表。返回 0 = 文件在但不是本格式/K 不同(不读也不往里写, 免得两种记录混一个文件), 1 = 可用(含文件还不存在) */
static int pt_cache_load(const char *path, pt_data *d, uint32_t K, pt_kmap *m, uint32_t *hit) {
    FILE *f = fopen(path, "rb");
    *hit = 0;
    if (!f) return 1;
    uint32_t hd[2] = { 0, 0 };
    if (fread(hd, 4, 2, f) != 2 || hd[0] != PT_CACHE_MAGIC || hd[1] != K) { fclose(f); return 0; }
    uint64_t key; uint32_t mm;
    while (fread(&key, 8, 1, f) == 1 && fread(&mm, 4, 1, f) == 1) {
        const size_t mk = (size_t)mm * K;
        const uint32_t idx = *pt_kmap_slot(m, key);
        pt_sample *s = idx != UINT32_MAX ? &d->s[idx] : NULL;
        if (s && !s->top_id && s->m == mm) {
            s->top_id = xmalloc(mk * 4); s->top_p = xmalloc(mk * 4); s->top_rest = xmalloc((size_t)mm * 4);
            if (fread(s->top_id, 4, mk, f) != mk || fread(s->top_p, 4, mk, f) != mk || fread(s->top_rest, 4, mm, f) != mm) {
                free(s->top_id); free(s->top_p); free(s->top_rest); s->top_id = NULL; s->top_p = s->top_rest = NULL;
                break;   /* 尾部残缺(上次写到一半被杀): 后面没有了, 残缺那条本趟重算并追加在后面 */
            }
            (*hit)++;
        } else if (fseek(f, (long)(mk * 8 + (size_t)mm * 4), SEEK_CUR)) break;
    }
    fclose(f);
    return 1;
}

/* 写回: rewrite = 整个文件按新格式重写(老格式转换, 全部代表题), 否则只追加本趟新算的代表题(fresh) */
static bool pt_cache_save(const char *path, const pt_data *d, uint32_t K, const uint32_t *rep, const uint8_t *fresh, int rewrite) {
    FILE *t = fopen(path, "rb");
    const bool had = t && fgetc(t) != EOF;
    if (t) fclose(t);
    FILE *f = fopen(path, had && !rewrite ? "ab" : "wb");
    if (!f) return false;
    bool ok = true;
    if (!had || rewrite) { const uint32_t hd[2] = { PT_CACHE_MAGIC, K }; ok = fwrite(hd, 4, 2, f) == 2; }
    uint32_t n = 0;
    for (uint32_t i = 0; ok && i < d->ns; i++) {
        const pt_sample *s = &d->s[i];
        if (!s->top_id || s->hard || s->nocache || rep[i] != i || !(rewrite || fresh[i])) continue;
        const uint64_t key = pt_sample_key(s, K); const size_t mk = (size_t)s->m * K;
        ok = fwrite(&key, 8, 1, f) == 1 && fwrite(&s->m, 4, 1, f) == 1 && fwrite(s->top_id, 4, mk, f) == mk && fwrite(s->top_p, 4, mk, f) == mk && fwrite(s->top_rest, 4, s->m, f) == s->m;
        n++;
    }
    fclose(f);
    if (ok) fprintf(stderr, "ds4: [ptrain] 教师表缓存 %s: %s %u 题\n", path, rewrite ? "按题格式重写" : "追加", n);
    return ok;
}

/* 一趟教师前向: 算"代表题(rep[i]==i)、还没表、锚标志 == want"的题。pt_dir 非空 = 教师状态挂这份 ③(kl 锚行的教师 = 本轮起点自己, 10-04):
 * 插件是分配状态那一刻挂的(v41_state_alloc → v41_state_plugins), 所以只在 alloc 前后动全局 g_ds4_v41_pt_dir 并立刻还原 ——
 * 学生状态(pt_run_alloc, 在教师之后分配)看见它就会多挂一份 ③ 报"L%u 已有放大器"停车。 */
static bool pt_teacher_pass(ds4_engine *e, pt_data *d, uint32_t K, const uint32_t *rep, uint8_t *fresh, int want, const char *pt_dir, uint32_t need) {
    uint32_t maxtn = 16;
    for (uint32_t i = 0; i < d->ns; i++) if (d->s[i].tn > maxtn) maxtn = d->s[i].tn;
    const uint32_t cap = maxtn < DS4_V41_CHUNK ? maxtn : DS4_V41_CHUNK;
    ds4_v41_state st;
    const char *keep = g_ds4_v41_pt_dir;
    g_ds4_v41_pt_dir = pt_dir;
    const bool al = v41_state_alloc(&st, cap, maxtn + 8u, PT_TROWS);
    g_ds4_v41_pt_dir = keep;
    if (!al) return false;
    bool ok = true;
    ds4_gpu_tensor *tid = v41_alloc((uint64_t)PT_TROWS * K * 4, &ok), *tp = v41_alloc((uint64_t)PT_TROWS * K * 4, &ok), *tr = v41_alloc((uint64_t)PT_TROWS * 4, &ok);
    pt_snap sn; memset(&sn, 0, sizeof sn);
    const double t0 = now_sec();
    uint32_t done = 0, skip = 0;
    #define PT_WANT(S) (rep[(S) - d->s] == (uint32_t)((S) - d->s) && !(S)->top_id && ((S)->nocache != 0) == (want != 0))   /* 形参不能叫 s: 体里有 d->s */
    for (uint32_t ci = 0; ok && ci < d->nch; ci++) {
        const pt_sample *f = NULL; uint32_t todo = 0;   /* 这块有没有要算的代表题: 全命中的块(回放的旧块)连前缀都不预填 */
        for (uint32_t i = 0; i < d->ns; i++) if (d->s[i].chunk == ci) { if (!f) f = &d->s[i]; if (PT_WANT(&d->s[i])) todo++; }
        if (!todo) continue;
        uint32_t P0 = d->ch[ci].lcp > PT_TAIL ? d->ch[ci].lcp - PT_TAIL : 0u;
        pt_state_reset(&st);
        for (uint32_t c0 = 0; ok && c0 < P0; ) {   /* 前缀: 部署 CED 口径(中间块只跑编码器段 + 分界层 KV) */
            const uint32_t nc = P0 - c0 < cap ? P0 - c0 : cap;
            st.ced_skip = g_ds4_v41_decoder_full ? 0 : 1;
            ok = v41_forward(e, &st, f->tids + c0, nc);
            c0 += nc;
        }
        st.ced_skip = 0;
        if (ok) ok = pt_snap_io(&st, &sn, 1);
        for (uint32_t i = 0; ok && i < d->ns; i++) {
            pt_sample *s = &d->s[i];
            if (s->chunk != ci || !PT_WANT(s)) continue;
            const uint32_t n = s->tn - P0;
            if (n > PT_TROWS || n > cap) { skip++; continue; }
            ok = pt_snap_io(&st, &sn, 0) && v41_forward(e, &st, s->tids + P0, n);
            if (!ok) break;
            const uint32_t r0 = s->ta0 - 1u - P0;
            ok = ds4_gpu_bwd_topk_tensor(tid, tp, tr, st.logits, r0, s->m, DS4_N_VOCAB, K) != 0;
            const size_t mk = (size_t)s->m * K;
            s->top_id = xmalloc(mk * 4); s->top_p = xmalloc(mk * 4); s->top_rest = xmalloc((size_t)s->m * 4);
            if (ok) ok = ds4_gpu_tensor_read(tid, 0, s->top_id, mk * 4) && ds4_gpu_tensor_read(tp, 0, s->top_p, mk * 4) &&
                         ds4_gpu_tensor_read(tr, 0, s->top_rest, (uint64_t)s->m * 4);
            fresh[i] = 1; done++;
        }
        fprintf(stderr, "ds4: [ptrain] 教师%s 块 %u/%u(前缀 %u token 预填一次)累计 %u/%u 题 %.0f s\n", want ? "(锚)" : "", ci + 1, d->nch, P0, done, need, now_sec() - t0);
    }
    #undef PT_WANT
    pt_snap_free(&sn);
    if (tid) ds4_gpu_tensor_free(tid);
    if (tp) ds4_gpu_tensor_free(tp);
    if (tr) ds4_gpu_tensor_free(tr);
    v41_state_free(&st);
    if (!ok) fprintf(stderr, "ds4: [ptrain] 教师前向失败\n");
    else if (skip) fprintf(stderr, "ds4: [ptrain] 教师续算块超 %u 行跳过 %u 题(不进训练/评估)\n", PT_TROWS, skip);
    return ok;
}

bool pt_teacher(ds4_engine *e, const pt_cfg *c, pt_data *d) {
    char path[1200];
    if (c->teacher[0]) snprintf(path, sizeof path, "%s", c->teacher); else snprintf(path, sizeof path, "%s/teacher.bin", c->out);
    const uint32_t K = c->topk;
    pt_kmap km; { uint32_t cap = 16; while (cap < 2u * d->ns) cap <<= 1; km.k = xmalloc((size_t)cap * 8); km.v = xmalloc((size_t)cap * 4); memset(km.v, 0xff, (size_t)cap * 4); km.mask = cap - 1u; }
    uint32_t *rep = xmalloc((size_t)(d->ns + 1) * 4);   /* 每题的代表题号(自己 = 代表) */
    uint8_t *fresh = xmalloc_zeroed(d->ns + 1, 1);
    for (uint32_t i = 0; i < d->ns; i++) {
        /* 硬目标题的表是 one-hot(pt_add_sample 已填), 不是教师; 锚行的教师挂着另一份 ③: 都不进键表、不当别题的代表、不进缓存 */
        if (d->s[i].hard || d->s[i].nocache) { rep[i] = i; continue; }
        uint32_t *v = pt_kmap_slot(&km, pt_sample_key(&d->s[i], K)); if (*v == UINT32_MAX) *v = i; rep[i] = *v;
    }
    uint32_t hit = 0, need = 0;
    int usable = pt_cache_load(path, d, K, &km, &hit), rewrite = 0;
    if (!usable) {
        if (pt_cache_v1(path, d, K)) { rewrite = 1; usable = 1; for (uint32_t i = 0; i < d->ns; i++) if (d->s[i].top_id) hit++;
            fprintf(stderr, "ds4: [ptrain] 教师表 %s 是 10-01 的整份料格式且料没变, 读完按题格式重写\n", path); }
        else { rewrite = 1; usable = 1;   /* 对不上的缓存文件没有别的用处(kd_teacher_cache 从别的目录拷来的整份料表就是这种): 本趟算完按题格式盖掉 */
            fprintf(stderr, "ds4: [ptrain] 教师表 %s 不是按题格式(或 K 不同)也对不上整份料: 本趟算完按题格式重写它\n", path); }
    }
    uint32_t uniq = 0, need_a = 0;
    for (uint32_t i = 0; i < d->ns; i++) if (rep[i] == i && !d->s[i].hard) { uniq++; if (!d->s[i].top_id) { if (d->s[i].nocache) need_a++; else need++; } }
    fprintf(stderr, "ds4: [ptrain] 教师表缓存 %s: 清单 %u 题(去重 %u), 命中 %u, 要算 %u%s\n", path, d->ns, uniq, hit, need, need_a ? "(另有锚行, 见下)" : "");
    free(km.k); free(km.v);
    if (need && !pt_teacher_pass(e, d, K, rep, fresh, 0, NULL, need)) { free(rep); free(fresh); return false; }
    if (need_a) {   /* 锚行(kl)另起一个教师状态: 挂 anchor= 的 ③(本轮起点)答同一串 token; 没给 anchor = 部署态(GRPO 式固定参考) */
        fprintf(stderr, "ds4: [ptrain] 锚行 %u 题的教师 = %s\n", need_a, c->anchor[0] ? c->anchor : "部署态(①+②, 没给 anchor=)");
        if (!pt_teacher_pass(e, d, K, rep, fresh, 1, c->anchor[0] ? c->anchor : NULL, need_a)) { free(rep); free(fresh); return false; }
    }
    for (uint32_t i = 0; i < d->ns; i++) {   /* 重复题(同键)从代表题拷贝 */
        pt_sample *s = &d->s[i]; const pt_sample *o = &d->s[rep[i]];
        if (rep[i] == i || s->top_id || !o->top_id || o->hard) continue;
        const size_t mk = (size_t)s->m * K;
        s->top_id = xmalloc(mk * 4); s->top_p = xmalloc(mk * 4); s->top_rest = xmalloc((size_t)s->m * 4);
        memcpy(s->top_id, o->top_id, mk * 4); memcpy(s->top_p, o->top_p, mk * 4); memcpy(s->top_rest, o->top_rest, (size_t)s->m * 4);
    }
    if (usable && (need || rewrite) && !pt_cache_save(path, d, K, rep, fresh, rewrite))
        fprintf(stderr, "ds4: [ptrain] ★教师表缓存写不进 %s(不影响本趟)★\n", path);
    free(rep); free(fresh);
    return true;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_ptrain_teacher_nonempty_tu;
