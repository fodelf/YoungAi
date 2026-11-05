/* core_v41_kvreuse.c — V4.1 单请求路的 KV 前缀复用(2026-10-11)。
 *
 * 为什么要: 以前每条请求都从第 0 个 token 整段预填。聊天页和助理每一步发的都是"上一条提示 + 模型刚吐的 + 新的一段",
 * 前面那一大截与上一条一模一样, 却每次重读一遍。10-11 实测助理一趟 31 次工具调用: 每步只新增 5~9 千 token,
 * 却要预填 4~19 万 token, 一步 40 秒到 3 分钟, 半小时没跑完(预填 1000 t/s、解码 40 t/s 都正常, 钱全花在重读上)。
 *
 * 怎么做: 两份检查点(只存会被解码改掉或之后要读的那几样):
 *   - 压缩 KV / 索引键: 按组号只追加, 存已用的 ng_src 组(20 万 token ≈ 180 MB, 1M ≈ 0.9 GB);
 *   - 窗口环: win 缓冲的前 SWA 行(位置 p 恒在 p % SWA 格, 见 cuda_kv_ring.inc.cu), 解码会把它们一格格盖掉;
 *   - 压缩器余行: cpre_kv/cpre_sc 头部 cpend 行(还没凑满一组的输入);
 *   - 草稿器的 main_hidden 环 + 计数; 各层 ng_src / cpend / win_from; 到这个位置为止的 token 序列; 末位那一行 logits。
 *   槽 0 = 提示末尾(预填刚完、解码前存): 接"同提示再来一遍"(页面重新生成、后训练一题 8 份采样: 10-06 账 160 s 里 140 s 是重复预填)
 *         与"提示以上一条提示开头"的请求。
 *   槽 1 = 解码结束(2026-10-20): 位置 = 提示 + 真喂进 KV 的生成 token。接"上一条提示 + 模型刚吐的 + 新的一段"——
 *         助理/聊天每一轮都是这个形状, 以前这一轮的第一条只能接到槽 0, 刚生成的几万 token 明明在 KV 里却重算一遍
 *         (10-20 实撞: 思考 36k 位后下一条 47,498 的提示只接上 11,358, 重新预填 36,140 位 36.5 s)。
 *         服务端把上一条的采样 id 原样拼进下一条提示(server_v41_live.c), 才对得上: 重新分词在第 12,229 位就与采样序列分叉。
 *   两槽各自整份存(不做增量): 1M 一份 0.9 GB, 存之前按内存余量(ds4_runtime_mem_headroom_bytes)判, 放不下就只少这一份复用,
 *   不碰看门狗红线。
 * 下一条请求的提示若以某槽的 token 序列开头(且不短于它), 就新开一份状态, 把那槽拷进去, 只预填后面那段; 两槽都接得上取长的。
 * 解码图 / 采样 / 投机 / 稠密段的缓冲全是新状态自己的, 与冷启动同一套代码 —— 复用只改"预填从哪开始"。
 * ★提示与槽等长也复用★(2026-10-12): 槽里带末位那一行 logits(0.5 MB); 等长 ⇒ 状态拷回、logits 行拷回第 0 行、一块都不预填,
 * 直接从那一行采第一个 token。
 *
 * 与冷启动的差别(如实): 后面那段按自己的块切(从检查点位置起切), 冷启动从 0 起切; 块边界不同 ⇒ GEMM 形状不同, 浮点上不是逐位同,
 * 与把 --v41-chunk 换个值同一量级。CED 的"最后一块跑满解码器"规则对后面那段照样生效(v41_prefill_chunk 按 c0 起算)。
 * 精确门(检查点存全了没有): A = 提示 P 温 0 关投机吐 t1 t2…; B = P+t1 ⇒ B 从槽 1 整份接上(P+t1 已在 KV 里), 第一个 token 从存下的
 * logits 行采, B 的输出必须逐 token 等于 A 去掉 t1(A 解码几十步把环/余行/组数全改过, 漏存一样就过不去)。脚本 gguf-tools/scripts/kv_reuse_gate.sh
 * (含解码结束槽的长解码门: A 吐 N 位后 D = P + A 的 N 位文本, D 必须等于一条 N+M 位的 A 的后 M 位)。
 *
 * 作废: 侧车 / 后训练件热切(KV 是带着放大器算的)—— ds4_engine_v41_switch_plugins 调 v41_kvreuse_drop。
 * 出错会怎样: 拷贝失败 ⇒ 这一条照冷启动整段预填(结果与没有复用时相同), 日志打一行; 不会拿半份检查点去算。 */
#include "core_internal.h"

int g_ds4_v41_kv_reuse = 1;
void ds4_engine_v41_set_kv_reuse(int on) { g_ds4_v41_kv_reuse = on; }
static int g_last_reused;
int ds4_engine_v41_last_reused(void) { return g_last_reused; }

typedef struct {
    int valid;
    uint32_t n;                 /* 检查点位置 = 到这里为止的 token 数 */
    int32_t *tok; uint32_t tok_cap;
    uint64_t *sig;              /* [n] 每位所属图片的指纹(0 = 文本): 图片 span 的 token 全是同一个占位 id, 不比图片字节就会把别的图的 KV 接上(2026-10-13) */
    ds4_gpu_tensor *ckv[DS4_MAX_LAYER], *idk[DS4_MAX_LAYER], *win[DS4_MAX_LAYER], *cpk[DS4_MAX_LAYER], *cps[DS4_MAX_LAYER], *mainh;
    ds4_gpu_tensor *logits;     /* 末位的 logits 一行 [V] f32: 等长复用时不预填, 第一个 token 从它采 */
    uint32_t ng_src[DS4_MAX_LAYER], cpend[DS4_MAX_LAYER], win_from[DS4_MAX_LAYER];
    int64_t mainh_end; uint32_t mainh_n;
    uint64_t bytes;             /* 检查点占的显存(日志用) */
} ckpt;
enum { CK_PROMPT = 0, CK_END = 1, CK_SLOTS = 2 };
static ckpt K[CK_SLOTS];

/* 作废但留提示: 切换插件后服务端空闲时拿它用新插件重新预填一遍(server_plugins.c), 用户切完再发消息就接得上, 不用等整段预填
 * (7,240 token 的工具会话实测每次开关后首条 7.1 s)。只记一次: 取走就清, 预热被新请求打断也不再补(那条新请求自己会存新检查点)。 */
static int g_stale;
void v41_kvreuse_drop(void) { g_stale = K[CK_PROMPT].valid && K[CK_PROMPT].n > 0; K[CK_PROMPT].valid = 0; K[CK_END].valid = 0; }
int32_t *ds4_engine_v41_kv_stale_prompt(uint32_t *n) {
    *n = 0;
    if (!g_stale) return NULL;
    g_stale = 0;
    const ckpt *k = &K[CK_PROMPT];
    for (uint32_t i = 0; i < k->n; i++) if (k->sig[i]) return NULL;   /* 带图片的提示: 视觉嵌入没留, 预热不了 */
    int32_t *tok = xmalloc((size_t)k->n * 4);
    memcpy(tok, k->tok, (size_t)k->n * 4);
    *n = k->n;
    return tok;
}

/* 缓冲不够大就换一块大的(按 1.5 倍长, 免得长对话每步都重分配); 内容不保留 —— 调用方紧接着整段重写 */
static bool ensure(ds4_gpu_tensor **t, uint64_t need) {
    if (!need) return true;
    if (*t && ds4_gpu_tensor_bytes(*t) >= need) return true;
    if (*t) ds4_gpu_tensor_free(*t);
    bool ok = true;
    *t = v41_alloc(need + need / 2, &ok);
    return ok && *t;
}
static uint64_t have(const ds4_gpu_tensor *t) { return t ? ds4_gpu_tensor_bytes(t) : 0; }

static bool cp(ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t bytes) {
    return !bytes || (dst && src && ds4_gpu_tensor_copy(dst, 0, src, 0, bytes));
}

/* 这一份要新分多少显存(已有缓冲够大的不算): 存之前先算, 超过内存余量就不存 —— 槽 1 在 1M 上下文是 0.9 GB, 不能等 cudaMalloc 说不 */
static uint64_t ckpt_need(const ckpt *k, const ds4_v41_state *st) {
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    uint64_t need = 0;
#define NEED(t, b) do { const uint64_t b_ = (b); if (b_ > have(t)) need += b_ + b_ / 2; } while (0)
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        NEED(k->win[il], DS4_N_SWA * rowb);
        if (!st->comp_kv[il]) continue;
        NEED(k->ckv[il], (uint64_t)st->ng_src[il] * DS4_V41_CKV_BYTES);
        NEED(k->idk[il], (uint64_t)st->ng_src[il] * DS4_V41_IDXK_BYTES);
        if (st->cpre_kv[il] && st->cpend[il]) { NEED(k->cpk[il], (uint64_t)st->cpend[il] * rowb); NEED(k->cps[il], (uint64_t)st->cpend[il] * rowb); }
    }
    if (st->mainh) NEED(k->mainh, ds4_gpu_tensor_bytes(st->mainh));
    NEED(k->logits, (uint64_t)DS4_N_VOCAB * 4);
#undef NEED
    return need;
}

/* st 此刻停在 head + tail 的末尾(tail 可空); logit_row = 末位 logits 在 st->logits 的哪一行 */
static void ckpt_save(ckpt *k, const char *what, const ds4_v41_state *st, const int32_t *head, uint32_t nh, const uint64_t *hsig,
                      const int32_t *tail, uint32_t nt, uint32_t logit_row) {
    k->valid = 0;
    const uint64_t need = ckpt_need(k, st), room = ds4_runtime_mem_headroom_bytes();
    if (need > room) {
        fprintf(stderr, "ds4: [v41] %s检查点要新分 %.0f MB, 内存余量只剩 %.0f MB, 这一份不存(下一条请求少这一段复用)\n",
                what, (double)need / 1048576.0, (double)room / 1048576.0);
        return;
    }
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    bool ok = true; uint64_t total = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        ok = ensure(&k->win[il], DS4_N_SWA * rowb) && cp(k->win[il], st->win[il], DS4_N_SWA * rowb);
        total += DS4_N_SWA * rowb;
        k->ng_src[il] = st->ng_src[il]; k->cpend[il] = st->cpend[il]; k->win_from[il] = st->win_from[il];
        if (!ok || !st->comp_kv[il]) continue;
        const uint64_t kb = (uint64_t)st->ng_src[il] * DS4_V41_CKV_BYTES, ib = (uint64_t)st->ng_src[il] * DS4_V41_IDXK_BYTES;
        ok = ensure(&k->ckv[il], kb) && cp(k->ckv[il], st->comp_kv[il], kb) && ensure(&k->idk[il], ib) && cp(k->idk[il], st->index_k[il], ib);
        total += kb + ib;
        if (ok && st->cpre_kv[il] && st->cpend[il]) {
            const uint64_t pb = (uint64_t)st->cpend[il] * rowb;
            ok = ensure(&k->cpk[il], pb) && cp(k->cpk[il], st->cpre_kv[il], pb) && ensure(&k->cps[il], pb) && cp(k->cps[il], st->cpre_sc[il], pb);
            total += 2 * pb;
        }
    }
    if (ok && st->mainh) {
        const uint64_t mb = ds4_gpu_tensor_bytes(st->mainh);
        ok = ensure(&k->mainh, mb) && cp(k->mainh, st->mainh, mb);
        total += mb;
    }
    k->mainh_end = st->mainh_end; k->mainh_n = st->mainh_n;
    const uint64_t vb = (uint64_t)DS4_N_VOCAB * 4;
    if (ok) ok = ensure(&k->logits, vb) && st->logits && ds4_gpu_tensor_copy(k->logits, 0, st->logits, (uint64_t)logit_row * vb, vb);
    total += vb;
    if (ok && !ds4_gpu_synchronize()) ok = false;
    if (!ok) { fprintf(stderr, "ds4: [v41] %s检查点存不下(显存), 下一条请求少这一段复用\n", what); return; }
    const uint32_t n = nh + nt;
    if (k->tok_cap < n) { free(k->tok); free(k->sig); k->tok_cap = n + n / 2; k->tok = xmalloc((size_t)k->tok_cap * 4); k->sig = xmalloc((size_t)k->tok_cap * 8); }
    memcpy(k->tok, head, (size_t)nh * 4);
    if (nt) memcpy(k->tok + nh, tail, (size_t)nt * 4);
    if (hsig) memcpy(k->sig, hsig, (size_t)nh * 8); else memset(k->sig, 0, (size_t)nh * 8);
    if (nt) memset(k->sig + nh, 0, (size_t)nt * 8);   /* 生成的 token 没有图片 */
    k->n = n; k->bytes = total; k->valid = 1;
}

/* 预填刚完(解码前)调: st 此刻正好停在提示末尾 */
void v41_kvreuse_save(const ds4_v41_state *st, const int32_t *prompt, uint32_t np, const uint64_t *sig) {
    if (!g_ds4_v41_kv_reuse) return;
    ckpt_save(&K[CK_PROMPT], "提示末尾的 KV ", st, prompt, np, sig, NULL, 0, st->last_logit_row);
}

/* 解码结束调: fed = 这一趟真喂进 KV 的生成 token(按喂入顺序, 不含没喂的末位/EOS); st 停在 prompt + fed 的末尾, 末位 logits 在 logit_row 行 */
void v41_kvreuse_save_end(const ds4_v41_state *st, const int32_t *prompt, uint32_t np, const uint64_t *sig,
                          const int32_t *fed, uint32_t nf, uint32_t logit_row) {
    if (!g_ds4_v41_kv_reuse || !nf) return;
    if (st->n_past != np + nf) {   /* 位置账对不上 = 喂入清单漏记了, 这份存了也接不对; 报出来别装没事 */
        fprintf(stderr, "ds4: ★[v41] 解码结束位置 %u ≠ 提示 %u + 喂入 %u, 解码结束检查点不存★\n", st->n_past, np, nf);
        return;
    }
    ckpt_save(&K[CK_END], "解码结束的 KV ", st, prompt, np, sig, fed, nf, logit_row);
}

/* 前 n 位的图片指纹要逐位相同(没图片 = 全 0) */
static bool sig_match(const ckpt *k, const uint64_t *a, uint32_t n) {
    if (!a) { for (uint32_t i = 0; i < n; i++) if (k->sig[i]) return false; return true; }
    return memcmp(k->sig, a, (size_t)n * 8) == 0;
}

/* 新状态刚分配完(预填前)调: 能接上就把检查点拷进 st, 返回预填起点(= 检查点位置; = np 时一块都不用填, 末位 logits 已在第 0 行);
 * 接不上返回 0(整段预填)。两槽都接得上取长的(槽 1 ⊇ 槽 0 时它更长)。 */
static const ckpt *ckpt_match(const int32_t *prompt, uint32_t np, uint32_t ctx, const uint64_t *sig) {
    if (!g_ds4_v41_kv_reuse) return NULL;
    const ckpt *k = NULL;
    for (int s = 0; s < CK_SLOTS; s++) {
        const ckpt *c = &K[s];
        if (!c->valid || c->n > np || c->n > ctx || (k && c->n <= k->n)) continue;
        if (memcmp(c->tok, prompt, (size_t)c->n * 4) || !sig_match(c, sig, c->n)) continue;
        k = c;
    }
    return k;
}

/* Match before allocating rows: a 70K-token prefix plus five new tokens must
 * not allocate a 2048-row prefill workspace. Keep enough rows for DSpark even
 * on an exact hit. The context/KV capacity still covers the entire request. */
bool v41_kvreuse_alloc(ds4_v41_state *st, const int32_t *prompt, uint32_t np,
                       uint32_t ctx, uint32_t chunk, const uint64_t *sig, uint32_t *reused) {
    const ckpt *k = ckpt_match(prompt, np, ctx, sig);
    const uint32_t planned = k ? k->n : 0, remaining = np - planned;
    uint32_t rows = remaining < chunk ? remaining : chunk;
    const uint32_t floor = DS4_MTP_MAX_BLOCK + 2u;
    if (rows < floor) rows = floor;
    if (rows > ctx) rows = ctx;
    *reused = 0;
    if (!v41_state_alloc(st, rows, ctx, floor)) return false;
    *reused = v41_kvreuse_restore(st, prompt, np, sig);
    if (planned && *reused != planned) {
        /* A failed copy may have queued writes into the partial state. Drain
         * it before freeing, then cold-prefill a fresh, full-sized workspace. */
        (void)ds4_gpu_synchronize();
        v41_state_free(st);
        rows = np < chunk ? np : chunk;
        if (rows < floor) rows = floor;
        if (rows > ctx) rows = ctx;
        if (!v41_state_alloc(st, rows, ctx, floor)) return false;
        *reused = 0;
    }
    fprintf(stderr, "[v41] request workspace: prompt=%u reused=%u pending=%u rows=%u ctx=%u\n",
            np, *reused, np - *reused, rows, ctx);
    return true;
}

uint32_t v41_kvreuse_restore(ds4_v41_state *st, const int32_t *prompt, uint32_t np, const uint64_t *sig) {
    g_last_reused = 0;
    const ckpt *k = ckpt_match(prompt, np, st->ctx, sig);
    if (!k) return 0;
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        ok = cp(st->win[il], k->win[il], DS4_N_SWA * rowb);
        if (!ok || !st->comp_kv[il]) continue;
        ok = cp(st->comp_kv[il], k->ckv[il], (uint64_t)k->ng_src[il] * DS4_V41_CKV_BYTES) && cp(st->index_k[il], k->idk[il], (uint64_t)k->ng_src[il] * DS4_V41_IDXK_BYTES);
        if (ok && st->cpre_kv[il] && k->cpend[il])
            ok = cp(st->cpre_kv[il], k->cpk[il], (uint64_t)k->cpend[il] * rowb) && cp(st->cpre_sc[il], k->cps[il], (uint64_t)k->cpend[il] * rowb);
    }
    if (ok && st->mainh && k->mainh) ok = cp(st->mainh, k->mainh, ds4_gpu_tensor_bytes(st->mainh));
    if (ok && k->n == np) { ok = cp(st->logits, k->logits, (uint64_t)DS4_N_VOCAB * 4); st->last_logit_row = 0; }   /* 等长: 末位 logits 落第 0 行 */
    if (ok && !ds4_gpu_synchronize()) ok = false;
    if (!ok) {   /* 状态里可能已拷了一半: 冷启动会从第 0 位整段重写(环/余行/组数都按位置重算), 不残留 */
        fprintf(stderr, "ds4: [v41] KV 检查点拷不进新状态, 这一条整段预填\n");
        return 0;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) { st->ng_src[il] = k->ng_src[il]; st->cpend[il] = k->cpend[il]; st->win_from[il] = k->win_from[il]; }
    if (st->mainh) { st->mainh_end = k->mainh_end; st->mainh_n = k->mainh_n; }
    memcpy(st->hist, k->tok, (size_t)k->n * 4);   /* engram 哈希要回看前 3 个 token, 跨检查点边界 */
    for (uint32_t i = 0; i < k->n; i++) st->himg[i] = k->sig[i] != 0;   /* 图片位同样要回看得到 */
    st->n_past = k->n;
    g_last_reused = (int)k->n;
    fprintf(stderr, "[v41] KV 复用 %u/%u token(%s检查点 %.1f MB), 只预填后 %u\n", k->n, np, k == &K[CK_END] ? "解码结束的" : "提示末尾的",
            (double)k->bytes / 1048576.0, np - k->n);
    return k->n;
}
