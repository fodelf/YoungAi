/* core_v41_state.c — V4.1 状态的分配 / 释放 / 收缩(2026-09-30 从 core_v41_forward.c 拆出, 那片顶到 500 行)。
 *
 * 一个状态 = KV 侧(随位置长: comp_kv/index_k/cpre/窗口环/hist/mainh)+ 行缓冲(随行数 cap: x/hc/q/…/logits)。
 * ★并发(batch.md, 用户 09-30 "支持 CUDA 架构的并发")★: 行缓冲按用途分两份 ——
 *   稠密段(embed/hc/norm/MoE/出口, v41_alloc_dense): 多请求合批时由**批态**统一持有一份(core_v41_multi.c), 权重每步只读一遍;
 *   注意力侧(q 路/压缩源/indexer/输出投影的中间量, v41_alloc_attn): 每个请求态自己一份, KV 本来就分开, 合批不摊。
 * 预填仍用整状态(块 2048 行, 与单请求路一字不差); 预填完 v41_state_shrink 把稠密段的行释放、注意力侧缩到 1 行, 请求态
 * 只剩 KV + 几十 MB —— 三个 1M 请求态同时活着才装得下(单请求 3.1 GB 固定开销里, 2.2 GB 是这两份 2048 行的缓冲)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU

ds4_gpu_tensor *v41_alloc(uint64_t bytes, bool *ok) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes ? bytes : 16);
    if (!t) *ok = false;
    return t;
}

/* 稠密段的行缓冲: 单状态与批态共用这一份分配式(形状只随 cap) */
static void v41_alloc_dense(ds4_v41_state *st, uint32_t cap, uint32_t logits_rows, bool *ok) {
    const uint64_t E = DS4_N_EMBD, HC = DS4_N_HC, FF = DS4_N_FF_EXP, NE = DS4_N_EXPERT, K = DS4_N_EXPERT_USED;
    const uint64_t mix = 2 * HC + HC * HC;
    st->tok = v41_alloc((uint64_t)cap * 4, ok);          st->pos = v41_alloc((uint64_t)cap * 4, ok);
    st->hc = v41_alloc((uint64_t)cap * HC * E * 4, ok);  st->hc2 = v41_alloc((uint64_t)cap * HC * E * 4, ok);
    st->mix = v41_alloc((uint64_t)cap * mix * 4, ok);    st->pre = v41_alloc((uint64_t)cap * HC * 4, ok);
    st->post = v41_alloc((uint64_t)cap * HC * 4, ok);    st->comb = v41_alloc((uint64_t)cap * HC * HC * 4, ok);
    st->pre_mix = v41_alloc((uint64_t)cap * HC * 4, ok);
    st->x = v41_alloc((uint64_t)cap * E * 4, ok);        st->xn = v41_alloc((uint64_t)cap * E * 4, ok);
    st->attn_out = v41_alloc((uint64_t)cap * E * 4, ok);
    st->glog = v41_alloc((uint64_t)cap * NE * 4, ok);    st->sel = v41_alloc((uint64_t)cap * K * 4, ok);
    st->rw = v41_alloc((uint64_t)cap * K * 4, ok);       st->routed = v41_alloc((uint64_t)cap * E * 4, ok);
    st->sg = v41_alloc((uint64_t)cap * FF * 4, ok);      st->su = v41_alloc((uint64_t)cap * FF * 4, ok);
    st->sh = v41_alloc((uint64_t)cap * FF * 4, ok);      st->so = v41_alloc((uint64_t)cap * E * 4, ok);
    st->y = v41_alloc((uint64_t)cap * E * 4, ok);
    /* logits 只按真要读的行数开(2026-09-29, 见 core_v41.h head_last_only): 生成路给 DS4_MTP_MAX_BLOCK+2 行, 打分路 0 = cap 行 */
    st->logits_rows = (logits_rows && logits_rows < cap) ? logits_rows : cap;
    st->logits = v41_alloc((uint64_t)st->logits_rows * DS4_N_VOCAB * 4, ok);
    st->xlast = v41_alloc((uint64_t)E * 4, ok);
}

/* 注意力投影段的行缓冲(q 路 / kv / o / low): 按行, 合批时由批态持有(v41_attn_in / v41_attn_out 在批态上一次发) */
static void v41_alloc_attn_rows(ds4_v41_state *st, uint32_t cap, bool *ok) {
    const uint64_t HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, low = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    st->qr = v41_alloc((uint64_t)cap * Q * 4, ok);       st->qrn = v41_alloc((uint64_t)cap * Q * 4, ok);
    st->q = v41_alloc((uint64_t)cap * NH * HD * 4, ok);
    st->kv = v41_alloc((uint64_t)cap * HD * 4, ok);      st->kvn = v41_alloc((uint64_t)cap * HD * 4, ok);
    st->o = v41_alloc((uint64_t)cap * NH * HD * 4, ok);  st->low = v41_alloc((uint64_t)cap * low * 4, ok);
}
/* 注意力缓存段的中间量(压缩源 / indexer): 每个请求态一份(iscore/cand 不在这里: 按用到的组数长, v41_index_scratch_prepare) */
static void v41_alloc_attn_cache(ds4_v41_state *st, uint32_t cap, bool *ok) {
    const uint64_t HD = DS4_N_HEAD_DIM, IH = DS4_N_INDEXER_HEAD, IK = DS4_N_INDEXER_HEAD_DIM;
    st->posg = v41_alloc((uint64_t)cap * 4, ok);
    st->ckv = v41_alloc((uint64_t)cap * HD * 4, ok);     st->csc = v41_alloc((uint64_t)cap * HD * 4, ok);
    st->pooled = v41_alloc((uint64_t)cap * HD * 4, ok);  st->latent = v41_alloc((uint64_t)cap * HD * 4, ok);
    st->ktmp = v41_alloc((uint64_t)cap * IK * 4, ok);
    st->iq = v41_alloc((uint64_t)cap * IH * IK * 4, ok); st->iw = v41_alloc((uint64_t)cap * IH * 4, ok);
    st->idx = v41_alloc((uint64_t)cap * DS4_N_INDEXER_TOP_K * 4, ok);
}
static void v41_alloc_attn(ds4_v41_state *st, uint32_t cap, bool *ok) { v41_alloc_attn_rows(st, cap, ok); v41_alloc_attn_cache(st, cap, ok); }

static void v41_free_list(ds4_gpu_tensor ***list, size_t n) {
    for (size_t i = 0; i < n; i++) { if (*list[i]) ds4_gpu_tensor_free(*list[i]); *list[i] = NULL; }
}
#define V41_FREE(...) do { ds4_gpu_tensor **l_[] = { __VA_ARGS__ }; v41_free_list(l_, sizeof(l_) / sizeof(l_[0])); } while (0)
static void v41_free_dense(ds4_v41_state *st) {
    V41_FREE(&st->tok, &st->pos, &st->hc, &st->hc2, &st->mix, &st->pre, &st->post, &st->comb, &st->pre_mix, &st->x, &st->xn, &st->attn_out,
             &st->glog, &st->sel, &st->rw, &st->routed, &st->sg, &st->su, &st->sh, &st->so, &st->y, &st->logits, &st->xlast);
}
static void v41_free_attn(ds4_v41_state *st) {
    V41_FREE(&st->posg, &st->qr, &st->qrn, &st->q, &st->kv, &st->kvn, &st->ckv, &st->csc, &st->pooled, &st->latent, &st->ktmp,
             &st->iq, &st->iw, &st->idx, &st->o, &st->low);
}
/* 合批时请求态里指向批态行的视图(core_v41_multi.c 每步挂/摘): 这几个字段在收缩后的请求态里恒 NULL, 释放时不许当自己的 */
static void v41_null_views(ds4_v41_state *st) { st->tok = st->pos = st->xn = st->attn_out = st->erows = st->qrn = st->q = st->kvn = st->o = NULL; }

/* 三文件部署的插件(②反修 ③后训练)按状态挂: 显式要了的挂不上就停车, 不许静默裸跑 */
static bool v41_state_plugins(ds4_v41_state *st) {
    /* 没挂任何目录 = 裸底座。路由偏置的设备表是进程级的, 同一进程里上一个状态可能挂过 —— 先全卸(没挂过是 no-op)。 */
    if (!g_ds4_v41_amp_dir && !g_ds4_v41_pt_dir) { v41_rb_clear_all(); return true; }
    /* ③ 是解在 ①+② 那个态上的, 单挂 = 把修正打在另一个基线上: 放行但打一行明白话(每个进程一次, 服务端有多个状态) */
    static int pt_alone_noted;
    if (g_ds4_v41_pt_dir && !g_ds4_v41_amp_dir && !pt_alone_noted++)
        fprintf(stderr, "ds4: ★只挂了后训练件、没挂反修件 —— 它是解在反修态上的, 这个组合不是判决态★\n");
    if (v41_amp_load(st, g_ds4_v41_amp_dir, g_ds4_v41_pt_dir)) return true;
    fprintf(stderr, "ds4: 插件挂不上, 停车(不做静默裸模型对照): 反修 %s / 后训练 %s\n",
            g_ds4_v41_amp_dir ? g_ds4_v41_amp_dir : "(无)", g_ds4_v41_pt_dir ? g_ds4_v41_pt_dir : "(无)");
    return false;
}

/* 服务里热切换 ②③(2026-10-10, 工作台聊天页"切侧车" + 训完挂新 ③): 插件本来就是每个状态创建时从盘上读进来的(上面那个函数),
 * 所以切换 = 换掉两个目录, 下一个状态就用新的, 113 GB 的底座不用重装。
 * ★调用方保证此刻没有活着的状态★(服务端在两条请求之间调; 合批的批态先关掉再调): 路由偏置 / 增益覆盖表是进程级的, 而且每次装都是
 * 先 cudaFree 再 cudaMalloc —— 有请求在跑时换, 它后半段就用错了表, 已捕获的图还会指到释放掉的显存。
 * 换之前先开一个 16 行的试探状态把新插件真装一遍: 坏目录、③ 的 base.fnv 对不上 ② 都在这里就拒, 不等下一条请求才炸; 拒了换回老目录。
 * 返回 0 = 已换; -1 = 没换(err 写原因, 细节在服务日志里 —— 装载失败的具体行是 v41_amp_load 打的)。 */
int ds4_engine_v41_switch_plugins(const char *amp_dir, const char *pt_dir, char *err, size_t errn) {
    if (g_ds4_v41_draft_amp) { snprintf(err, errn, "挂了草稿器对齐件(--draft-amp), 它绑着当前侧车的指纹, 不能热切"); return -1; }
    static char *own_amp, *own_pt;   /* 这里 strdup 的那份; 命令行给的指向 argv, 不归这里释放 */
    const char *old_amp = g_ds4_v41_amp_dir, *old_pt = g_ds4_v41_pt_dir;
    char *na = (amp_dir && amp_dir[0]) ? strdup(amp_dir) : NULL, *np = (pt_dir && pt_dir[0]) ? strdup(pt_dir) : NULL;
    g_ds4_v41_amp_dir = na; g_ds4_v41_pt_dir = np;
    ds4_v41_state st;
    if (!v41_state_alloc(&st, 16, 16, 2)) {   /* 失败时它自己已经 free 过 */
        g_ds4_v41_amp_dir = old_amp; g_ds4_v41_pt_dir = old_pt;
        free(na); free(np);
        snprintf(err, errn, "新侧车/后训练件挂不上(原因在服务日志), 已保持原来的");
        return -1;   /* 进程级表此刻可能是半装的: 下一个状态创建时会按老目录整套重装, 不用在这里补 */
    }
    v41_state_free(&st);
    free(own_amp); free(own_pt);
    own_amp = na; own_pt = np;
    return 0;
}

static uint64_t g_v41_state_uid = 0;   /* 状态序号发号器(见 core_v41.h uid) */
bool v41_state_alloc(ds4_v41_state *st, uint32_t cap, uint32_t ctx, uint32_t logits_rows) {
    memset(st, 0, sizeof *st);
    st->uid = ++g_v41_state_uid;
    st->cap_tok = cap; st->ctx = ctx; st->idx_owner = -1; st->cand_owner = -1;
    for (uint32_t i = 0; i < DS4_V41_MAX_ENGRAM; i++) st->eshard[i].fd = -1;
    if (cap == 0 || ctx < cap) { fprintf(stderr, "ds4: V4.1 状态参数错(cap %u ctx %u)\n", cap, ctx); return false; }
    if (ctx > g_ds4_v41.ctx) { fprintf(stderr, "ds4: V4.1 上下文 %u(模型元数据 deepseek4.context_length), 状态要了 %u\n", g_ds4_v41.ctx, ctx); return false; }
    const uint64_t E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, SWA = DS4_N_SWA;
    bool ok = true;
    st->hist = xmalloc((size_t)ctx * 4);
    v41_alloc_dense(st, cap, logits_rows, &ok);
    v41_alloc_attn(st, cap, &ok);
    st->wintmp = v41_alloc(SWA * HD * 4, &ok);
    /* DSpark 的 main_hidden: 留最后 SWA 行 —— 预填完要用它把三塔的 128 行窗口一次填满
     * (官方 DSparkBlock.forward(start_pos=0) 就是拿整段 main_kv 灌窗口), 之后每轮只进几行。 */
    if (g_ds4_v41.mtp_block && g_ds4_v41.n_mtp_target) {
        st->mainh_cap = DS4_N_SWA;
        st->mainh = v41_alloc((uint64_t)st->mainh_cap * g_ds4_v41.n_mtp_target * E * 4, &ok);
        st->mainh_end = -1; st->mainh_n = 0;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        st->win[il] = v41_alloc((SWA + cap) * HD * 4, &ok);
        /* ★环清零★(2026-09-21, bug.md §6.2): cudaMalloc 回收的多半是上一条请求同层的环 —— CED 下解码器段的环槽在最后
         * 一块之前从没写过, 不清就是把别的请求的 KV 当历史键读(读不报错, 只出假注意力)。下界钳位(win_from)是正解,
         * 清零是它的兜网: 万一哪条核路漏了钳位, 读到的也是零键而不是别人的键。 */
        if (ok && !ds4_gpu_tensor_fill_f32(st->win[il], 0.f, (uint64_t)(SWA + cap) * HD)) ok = false;
        if (!g_ds4_v41.is_kv_source[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ratio) { fprintf(stderr, "ds4: V4.1 kv 源层 L%u 压缩比为 0\n", il); ok = false; break; }
        /* ★多两格★(2026-09-18): 第 ctx/ratio 格是解码整步 graph 的**垃圾槽** —— 没凑满组的那些步, 打包核照样发,
         * 写到这一格(永远没人读; 组号上限是 ctx/ratio − 1)。core_v41_attn.c 算垃圾槽下标用的就是 ctx/ratio, 两处同源。 */
        const uint64_t ngcap = (uint64_t)ctx / ratio + 2;
        /* ★按官方打包尺寸分配★(decode.md D1): 主 KV 288 B/组、索引 K 72 B/组, 不是 512/128 个 f32。
         * 常量在 ds4_gpu_v41.h 只有一份 —— 这里与核里的解包各写各的就会静默越界。 */
        st->comp_kv[il] = v41_alloc(ngcap * DS4_V41_CKV_BYTES, &ok);
        st->index_k[il] = v41_alloc(ngcap * DS4_V41_IDXK_BYTES, &ok);
        st->posg_pin[il] = ds4_gpu_host_alloc((uint64_t)cap * 4);   /* 一块最多 cap 个新组; 收缩后仍按预填 cap 留着(几 KB) */
        if (!st->posg_pin[il]) ok = false;
        if (ratio > 1) {
            st->cpre_kv[il] = v41_alloc((ratio + (uint64_t)cap) * HD * 4, &ok);
            st->cpre_sc[il] = v41_alloc((ratio + (uint64_t)cap) * HD * 4, &ok);
        }
    }
    if (ok && !v41_state_plugins(st)) ok = false;
    if (!ok) { fprintf(stderr, "ds4: V4.1 状态缓冲分配失败(cap=%u ctx=%u)\n", cap, ctx); v41_state_free(st); }
    return ok;
}

/* 批态(core_v41_multi.c): 只有稠密段的行缓冲 + 反修表, 没有 KV、没有注意力侧行缓冲 —— 那些在各请求态里 */
bool v41_batch_rows_alloc(ds4_v41_state *st, uint32_t cap) {
    memset(st, 0, sizeof *st);
    st->cap_tok = cap; st->ctx = g_ds4_v41.ctx; st->idx_owner = -1; st->cand_owner = -1;
    for (uint32_t i = 0; i < DS4_V41_MAX_ENGRAM; i++) st->eshard[i].fd = -1;
    bool ok = true;
    v41_alloc_dense(st, cap, cap, &ok);
    v41_alloc_attn_rows(st, cap, &ok);   /* q/kv/o/low 也按批: 注意力的投影段在批态上发(core_v41_attn.c 三段拆分) */
    if (ok && !v41_state_plugins(st)) ok = false;
    if (!ok) { fprintf(stderr, "ds4: V4.1 批态行缓冲分配失败(cap=%u)\n", cap); v41_state_free(st); }
    return ok;
}

/* ★预填完收缩★(batch.md §3.1): 稠密段的行全放掉(合批时用批态的), 注意力侧缩到 rcap 行, 窗口环 / 压缩器余行按新行数重开并把
 * 历史那几行搬过去; 图、反修表、索引草稿、engram 行缓冲一并放掉(合批步用视图指进批态; 反修由批态应用)。
 * 出错会怎样: 漏搬环的前 SWA 行 = 窗口历史全丢, 下一步注意力只看新 token, 不报错只出胡话(逐字节门抓)。 */
bool v41_state_shrink(ds4_v41_state *st, uint32_t rcap) {
    const uint64_t HD = DS4_N_HEAD_DIM, SWA = DS4_N_SWA, rowb = HD * 4;
    if (!rcap || rcap > st->cap_tok) { fprintf(stderr, "ds4: V4.1 状态收缩参数错(rcap %u cap %u)\n", rcap, st->cap_tok); return false; }
    v41_graph_free(st);
    v41_amp_free(st);
    V41_FREE(&st->erows, &st->ekv, &st->iscore, &st->cand);
    for (uint32_t k = 0; k < DS4_V41_MAX_ENGRAM; k++) V41_FREE(&st->eraw[k]);
    st->iscap = 0; st->isrows = 0; st->icrows = 0; st->iscap_gen++;
    v41_free_dense(st);
    v41_free_attn(st);
    bool ok = true;
    v41_alloc_attn_cache(st, rcap, &ok);   /* 投影段的行(q/kv/o/low)不再自己持有: 合批时是批态行的视图 */
    if (g_ds4_v41.n_engram) {   /* 本请求自己的 engram 原始行(取行的上传目标); 解码行/wkv 出口用批态的 */
        const ds4_v41_cfg *v = &g_ds4_v41;
        const uint64_t cols = (uint64_t)(v->engram_max_ngram - 1) * v->engram_heads, stride = v->engram_head_dim + v->engram_head_dim / 32u;
        for (uint32_t k = 0; k < v->n_engram; k++) st->eraw[k] = v41_alloc((uint64_t)rcap * cols * stride, &ok);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        ds4_gpu_tensor *w = v41_alloc((SWA + rcap) * rowb, &ok);   /* 环在前 SWA 行, 后面是本批的行(core_v41.h) */
        if (ok && !ds4_gpu_tensor_fill_f32(w, 0.f, (SWA + rcap) * HD)) ok = false;   /* 与 alloc 同一姿势: 先清零再搬环 */
        if (ok && !ds4_gpu_tensor_copy(w, 0, st->win[il], 0, SWA * rowb)) ok = false;
        ds4_gpu_tensor_free(st->win[il]); st->win[il] = w;
        V41_FREE(&st->snap_win[il], &st->snap_cpre_kv[il], &st->snap_cpre_sc[il]);   /* 按 cap 开的快照, 用到时按新 cap 重建 */
        if (!g_ds4_v41.is_kv_source[il] || !st->cpre_kv[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        ds4_gpu_tensor *ck = v41_alloc((ratio + (uint64_t)rcap) * rowb, &ok), *cs = v41_alloc((ratio + (uint64_t)rcap) * rowb, &ok);
        if (ok && (!ds4_gpu_tensor_copy(ck, 0, st->cpre_kv[il], 0, (uint64_t)ratio * rowb) ||
                   !ds4_gpu_tensor_copy(cs, 0, st->cpre_sc[il], 0, (uint64_t)ratio * rowb))) ok = false;   /* 余行(< ratio 行)在头部 */
        ds4_gpu_tensor_free(st->cpre_kv[il]); ds4_gpu_tensor_free(st->cpre_sc[il]); st->cpre_kv[il] = ck; st->cpre_sc[il] = cs;
    }
    st->cap_tok = rcap; st->logits_rows = 0; st->n_direct1 = 0; memset(st->n_direct_n, 0, sizeof st->n_direct_n);
    if (!ok) fprintf(stderr, "ds4: V4.1 状态收缩失败(rcap %u)\n", rcap);
    return ok;
}

/* ★收缩态克隆★(2026-10-09, 奖励回路同题多份采样): 同一提示的 G 份采样以前各自预填一遍(20 题 × 8 份 × ~1.0 s = 160 s, 其中 7/8 是重复算),
 * 现在只预填一份, 收缩后把它的 KV 深拷给其余几份。预填是确定的(同提示两趟逐字节同, 10-06 门), 所以克隆出的 KV 与各自预填的逐字节同。
 * 只认收缩态(稠密段 / logits 已放、图与反修表已放): 那时状态里剩的全是本请求自己的缓冲, 一个个深拷; 其余指针字段在收缩态里恒 NULL,
 * 开头就核, 不是 NULL 说明调用顺序错了(漏一个就是两个状态共用一块缓冲, 一个写另一个读到 —— 不报错只出胡话)。
 * engram 的后台任务与表分片 fd 不拷: 首次前向时懒建(core_v41_engram.c), 共用 fd 会在 free 时关两遍。 */
static ds4_gpu_tensor *v41_dup(const ds4_gpu_tensor *s, bool *ok) {
    if (!s || !*ok) return NULL;
    const uint64_t n = ds4_gpu_tensor_bytes(s);
    ds4_gpu_tensor *d = v41_alloc(n, ok);
    if (d && !ds4_gpu_tensor_copy(d, 0, s, 0, n)) *ok = false;
    return d;
}
bool v41_state_clone(ds4_v41_state *dst, const ds4_v41_state *src) {
    memset(dst, 0, sizeof *dst);
    if (src->hc || src->logits || src->tok || src->qr || src->iscore || src->cand || src->dgraph || src->tsave || src->ampT || src->snap_on || src->ced_skip) {
        fprintf(stderr, "ds4: V4.1 状态克隆只认刚收缩的请求态\n"); return false;
    }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++)
        if (src->ampA[il] || src->snap_win[il] || src->snap_cpre_kv[il]) { fprintf(stderr, "ds4: V4.1 状态克隆只认刚收缩的请求态(L%u)\n", il); return false; }
    *dst = *src;   /* 标量(位置/组数/余行/窗口下界/mainh 账…)整份带过去; 下面把每个自有缓冲换成新拷贝 —— 失败也照样赋 NULL, 不留指向 src 的指针 */
    dst->uid = ++g_v41_state_uid;   /* 新状态新序号: 合批图按 (指针, uid) 认状态 */
    dst->ejob = NULL; dst->psamp = NULL; dst->spec_q = NULL;
    dst->eg_job_s = dst->eg_wait_s = dst->eg_enter_s = 0.0; dst->eg_n = 0;   /* engram 取行账按状态记, 关时打印; 不带 src 的 */
    for (uint32_t i = 0; i < DS4_V41_MAX_ENGRAM; i++) { dst->eshard[i].fd = -1; dst->eshard[i].dio = 0; dst->eshard[i].size = 0; }
    bool ok = true;
    dst->hist = xmalloc((size_t)src->ctx * 4);
    memcpy(dst->hist, src->hist, (size_t)src->ctx * 4);
    dst->posg = v41_dup(src->posg, &ok);     dst->ckv = v41_dup(src->ckv, &ok);   dst->csc = v41_dup(src->csc, &ok);
    dst->pooled = v41_dup(src->pooled, &ok); dst->latent = v41_dup(src->latent, &ok); dst->ktmp = v41_dup(src->ktmp, &ok);
    dst->iq = v41_dup(src->iq, &ok);         dst->iw = v41_dup(src->iw, &ok);     dst->idx = v41_dup(src->idx, &ok);
    dst->wintmp = v41_dup(src->wintmp, &ok); dst->mainh = v41_dup(src->mainh, &ok);
    for (uint32_t k = 0; k < DS4_V41_MAX_ENGRAM; k++) dst->eraw[k] = v41_dup(src->eraw[k], &ok);
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        dst->win[il] = v41_dup(src->win[il], &ok);
        dst->comp_kv[il] = v41_dup(src->comp_kv[il], &ok); dst->index_k[il] = v41_dup(src->index_k[il], &ok);
        dst->cpre_kv[il] = v41_dup(src->cpre_kv[il], &ok); dst->cpre_sc[il] = v41_dup(src->cpre_sc[il], &ok);
        dst->posg_pin[il] = NULL;
        if (src->posg_pin[il] && ok) {   /* 零拷贝位置槽: 内容每块重写, 不用拷; 收缩态一块最多 cap_tok 个新组 */
            dst->posg_pin[il] = ds4_gpu_host_alloc((uint64_t)src->cap_tok * 4);
            if (!dst->posg_pin[il]) ok = false;
        }
    }
    if (!ok) { fprintf(stderr, "ds4: V4.1 状态克隆失败(ctx %u)\n", src->ctx); v41_state_free(dst); }
    return ok;
}

/* ★索引打分的草稿按"这一趟走到哪"长, 不按配置的 ctx★(2026-09-22, 用户: "最新的架构 1m 上下文只好 0.89g")
 *
 * iscore [cap_tok][ng] f32 与 cand [cap_tok][ng] u8 是**每次前向都重写**的草稿(核里的行距就是当次的 ng,
 * 见 cuda_v41_indexer.inc.cu 的 score[i*ng+g] / cand[i*ng+g]), 不跨前向保存内容 —— cand_owner/idx_owner
 * 每次 v41_forward 开头都清, 所以两次前向之间换指针是安全的。
 * 以前按 ctx 一次开满: 1M 档 2.0 + 0.5 GiB, 而 1M 真正的 KV 只有 0.879 GiB —— "配大上下文很贵"是这块草稿造成的错觉。
 * 翻倍长: 1M 一趟最多长 11 次, 每次几十毫秒的 cudaMalloc, 摊到几万步里看不见。
 * ★捕获态下不许分配★(CUDA graph 捕获期间任何 cudaMalloc 都让捕获作废): 走图那条在 capture_begin 之前
 * 按桶上限先长够(core_decode_graph.c dg_capture), 与 attn/候选块暂存同一套做法; 这里撞见 st->graph 就是 bug, 直接喊。 */
bool v41_index_scratch_prepare(ds4_v41_state *st, uint32_t ng_need, uint32_t rows_score, uint32_t rows_cand) {
    /* ★C2(2026-09-30)★: cand 不再是按组/按块的掩码, 是候选块核写的紧凑列表 i32 [行][1 + topk_blocks](第 0 项 = 块数), 与 ng 无关 ——
     * 1M 预填块 2048 行只要 16.8 MB(掩码 268 MB); 打分/topk 核拿它只走候选(契约见 ds4_gpu_v41.h)。没有候选源层的接线表下 kcap = 0, 不开。 */
    const uint32_t kcap = g_ds4_v41.candidate_source_layer >= 0 && g_ds4_v41.candidate_topk_blocks > 0 ? (uint32_t)g_ds4_v41.candidate_topk_blocks : 0u;
    if (ng_need <= st->iscap && rows_score <= st->isrows && rows_cand <= st->icrows && st->iscore && (st->cand || !kcap)) return true;
    if (st->graph) {   /* 图那条应当在 capture 前长够(core_decode_graph.c), 这里撞见就是 bug */
        fprintf(stderr, "ds4: ★捕获态下要长索引草稿(要 %u/%u 行 × %u 组, 现有 %u/%u × %u)★\n", rows_score, rows_cand, ng_need, st->isrows, st->icrows, st->iscap); return false; }
    uint32_t want = st->iscap ? st->iscap * 2u : ng_need;
    if (want < ng_need) want = ng_need;
    if (want > st->ctx) want = st->ctx;
    /* ★行数按这一趟真要写的行开, 不按 cap_tok★(2026-09-29 实撞, fable5 09-29 深夜): 以前恒按 2048 行分而解码每步只写 1~7 行, 这台机器
     * cudaMalloc 分了就占(cuda_commit_probe.cu), 一条长请求每过一个 2 的幂多占一大块(65536→131072 组要一次 1.34 GB), 64.7k/79k 上下文两次被看门狗杀。
     * 解码档一次开到验证批上限(1+block+1 行, 与 logits_rows 同源): 1→7 行之间不反复重分(重分会作废解码图)。
     * ★C1(2026-09-30)★: 打分草稿只按行块 Rb 行开(预填块内按 Rb 行分几趟打分/选块/topk, core_v41_attn.c), 候选掩码按整块行数开但按候选块存(一块一字节):
     * 1M 上下文的预填块以前要 2048 × 1M × 5 B = 10.7 GB(过 52 万 token 就分不出 ⇒ 1M 跑不完), 现在 64 × 1M × 4 + 2048 × 131072 = 0.5 GB。 */
    const uint32_t dec_rows = DS4_MTP_MAX_BLOCK + 2u;
    uint32_t rs = rows_score <= dec_rows ? dec_rows : rows_score, rc = rows_cand <= dec_rows ? dec_rows : rows_cand;
    if (rs > st->cap_tok) rs = st->cap_tok < rows_score ? rows_score : st->cap_tok;
    if (rc > st->cap_tok) rc = st->cap_tok < rows_cand ? rows_cand : st->cap_tok;
    bool ok = true;
    ds4_gpu_tensor *is = v41_alloc((uint64_t)rs * want * 4, &ok);
    ds4_gpu_tensor *cd = kcap ? v41_alloc((uint64_t)rc * (1u + kcap) * 4, &ok) : NULL;
    if (!ok) {
        if (is) ds4_gpu_tensor_free(is);
        if (cd) ds4_gpu_tensor_free(cd);
        fprintf(stderr, "ds4: V4.1 索引草稿长不动(%u 组 × 打分 %u 行 / 候选列表 %u 行)\n", want, rs, rc);
        return false;
    }
    if (st->iscore) ds4_gpu_tensor_free(st->iscore);
    if (st->cand) ds4_gpu_tensor_free(st->cand);
    st->iscore = is; st->cand = cd; st->iscap = want; st->isrows = rs; st->icrows = rc; st->iscap_gen++;
    return true;
}

void v41_state_free(ds4_v41_state *st) {
    v41_graph_free(st);
    v41_amp_free(st);
    v41_engram_close(st);
    if (st->cap_tok && !st->hc && st->hist) v41_null_views(st);   /* 收缩后的请求态: 视图字段本该已摘, 保险再清一遍 */
    v41_free_dense(st);
    v41_free_attn(st);
    V41_FREE(&st->wintmp, &st->iscore, &st->cand, &st->mainh, &st->erows, &st->ekv);
    for (uint32_t k = 0; k < DS4_V41_MAX_ENGRAM; k++) V41_FREE(&st->eraw[k]);
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        V41_FREE(&st->win[il], &st->comp_kv[il], &st->index_k[il], &st->cpre_kv[il], &st->cpre_sc[il],
                 &st->snap_win[il], &st->snap_cpre_kv[il], &st->snap_cpre_sc[il]);
        if (st->posg_pin[il]) { ds4_gpu_host_free(st->posg_pin[il]); st->posg_pin[il] = NULL; }
    }
    free(st->hist); st->hist = NULL;
}
#else
/* 只为让服务端在 -DDS4_NO_GPU 构建里链得上: V4.1 插件只在 GPU 上挂, 这里不是 CPU 实现 */
int ds4_engine_v41_switch_plugins(const char *amp_dir, const char *pt_dir, char *err, size_t errn) {
    (void)amp_dir; (void)pt_dir; snprintf(err, errn, "CPU 构建没有 V4.1 插件"); return -1;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_state_nonempty_tu;
