/* core_ptrain_data.c — 后训练 ③ 的料: 配置、jsonl 读取、分词(2026-10-10 收口版; 总述见 core_ptrain.h)。
 *
 * 料 = jsonl, 一行一题: {"messages":[{"role":"user","content":"问"},{"role":"assistant","content":"答"}], "context":"材料(可省)"}
 *   或一行一段原文: {"text":"..."}(unsloth 的 continued pretraining 口径: 不套聊天模板, BOS + 原文 + EOS, 每个 token 都是目标)。
 *   messages 前面可以多一条 {"role":"system",...}(两边渲染都带); 别的字段一律报错 —— 10-09 那版收 reward/mode/split, 用户 10-10 判为跑偏, 不留口子。
 *   带 context = 上下文蒸馏(同一份 context 的题共一块, 教师前缀只预填一次); 不带 = 答案 one-hot(交叉熵), 不要教师。
 *   留出: 问题(text 行: 整段原文)去空白后的 FNV-1a 五取一(措辞没进训练); 保持料(cfg hold=)同规则, 只是教师 = 不看材料的部署态自己。
 * 渲染一律走部署同一个函数(ds4_encode_chat_prompt: BOS <｜User｜>正文 <｜Assistant｜></think>, 不思考):
 *   学生 = 问题;  教师 = 材料 + 空行 + 问题;  两边接同一串答案 token + EOS(答案单独分词, 两边逐 id 相同, 位置一一对齐)。
 * 出错会怎样: 一行坏了整趟停(打行号 + 原因), 不跳过 —— 静默跳过会让"料少了一半"看起来像"没学会"。 */
#include "core_ptrain.h"
#include "../common/ds4_json.h"
#ifndef DS4_NO_GPU

static char *pt_slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = xmalloc((size_t)n + 1);
    if (n > 0 && fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0; fclose(f);
    return b;
}
static void pt_rstrip(char *s) { size_t n = strlen(s); while (n && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ')) s[--n] = 0; }

bool pt_cfg_load(pt_cfg *c, const char *path) {
    memset(c, 0, sizeof *c);
    c->layer_lo = 0; c->layer_hi = DS4_N_LAYER - 1u;   /* 缺省全层(10-01 子集 1 轮: 全层 −32% 对 L21-39 −26%) */
    c->rank = 64; c->topk = 64; c->epochs = 3; c->batch = 4; c->maxlen = 1024; c->probe_n = 6; c->probe_tok = 96; c->seed = 1; c->dbg_layer = -1;
    /* 多形式扩展缺省开(10-11 用户: 验证过的机制进默认, 不藏在 extra= 里): youngai 24 行 换问法 12 题 挂 ③ 6/12 → 11/12, 泛化留出 2.13 → 1.80; 代价 = 每题一次性改写 ~7 s + 每轮训练量 ×5~7。
     * expand=0 回到只训原料(老路逐位同) */
    c->expand = 4;
    c->lr = 2e-4f; c->clip = 1.0f; c->init_std = 0.f;
    c->vq_hot = 4; c->vq_knn = 16; c->vq_group = 5; c->vq_bs = 16; c->vq_check = 1; c->vq_frac = 0.005f; c->vq_docs = 4;   /* V 步缺省(见 pt_cfg.vq_*; vq_layers 不给 = 不走 V 步) */
    char *b = pt_slurp(path);
    if (!b) { fprintf(stderr, "ds4: --ptrain 读不了配置 %s\n", path); return false; }
    for (char *ln = strtok(b, "\n"); ln; ln = strtok(NULL, "\n")) {
        while (*ln == ' ' || *ln == '\t') ln++;
        if (!*ln || *ln == '#') continue;
        char *eq = strchr(ln, '=');
        if (!eq) { fprintf(stderr, "ds4: --ptrain 配置行没有 '=': %s\n", ln); free(b); return false; }
        *eq = 0; char *k = ln, *v = eq + 1; pt_rstrip(k); pt_rstrip(v);
        while (*v == ' ') v++;
        if (!strcmp(k, "data")) snprintf(c->data, sizeof c->data, "%s", v);
        else if (!strcmp(k, "out")) snprintf(c->out, sizeof c->out, "%s", v);
        else if (!strcmp(k, "hold")) snprintf(c->hold, sizeof c->hold, "%s", v);
        else if (!strcmp(k, "init")) snprintf(c->init, sizeof c->init, "%s", v);
        else if (!strcmp(k, "eval")) snprintf(c->evalf, sizeof c->evalf, "%s", v);
        else if (!strcmp(k, "eval_kl")) snprintf(c->evalkl, sizeof c->evalkl, "%s", v);
        else if (!strcmp(k, "teacher")) {   /* 只认 fp:<表>(core_ptrain_fp.c); 别的教师来源不存在, 写错直接停 */
            if (strncmp(v, "fp:", 3) || !v[3]) { fprintf(stderr, "ds4: --ptrain teacher= 只认 fp:<top-K 表>, 给的是 %s\n", v); free(b); return false; }
            snprintf(c->teacher_fp, sizeof c->teacher_fp, "%s", v + 3);
        }
        else if (!strcmp(k, "dumpids")) snprintf(c->dumpids, sizeof c->dumpids, "%s", v);
        else if (!strcmp(k, "sample_n")) c->sample_n = (uint32_t)atoi(v);
        else if (!strcmp(k, "expand")) c->expand = (uint32_t)atoi(v);
        else if (!strcmp(k, "layers")) { unsigned a = 0, z = 0; if (sscanf(v, "%u-%u", &a, &z) == 2) { c->layer_lo = a; c->layer_hi = z; } else c->layer_lo = c->layer_hi = (uint32_t)atoi(v); }
        else if (!strcmp(k, "rank")) c->rank = (uint32_t)atoi(v);
        else if (!strcmp(k, "topk")) c->topk = (uint32_t)atoi(v);
        else if (!strcmp(k, "epochs")) c->epochs = (uint32_t)atoi(v);
        else if (!strcmp(k, "budget_s")) c->budget_s = (uint32_t)atoi(v);   /* 训练墙上时间预算(见 pt_cfg.budget_s) */
        else if (!strcmp(k, "vq_layers")) {   /* V 步的层(见 pt_cfg.vq_l): a-b 或单层 */
            unsigned a = 0, z = 0; c->vq_nl = 0;
            if (sscanf(v, "%u-%u", &a, &z) != 2) a = z = (unsigned)atoi(v);
            for (unsigned il = a; il <= z && il < DS4_MAX_LAYER && c->vq_nl < DS4_MAX_LAYER; il++) c->vq_l[c->vq_nl++] = il;
        }
        else if (!strcmp(k, "vq_hot")) c->vq_hot = (uint32_t)atoi(v);
        else if (!strcmp(k, "vq_knn")) c->vq_knn = (uint32_t)atoi(v);
        else if (!strcmp(k, "vq_group")) c->vq_group = (uint32_t)atoi(v);
        else if (!strcmp(k, "vq_bs")) c->vq_bs = (uint32_t)atoi(v);
        else if (!strcmp(k, "vq_check")) c->vq_check = (uint32_t)atoi(v);
        else if (!strcmp(k, "vq_tail")) c->vq_tail = (float)atof(v);   /* 尾权阈 τ(ds4_gpu_bwd_kl_tail): 0 关 */
        else if (!strcmp(k, "vq_docs")) c->vq_docs = (uint32_t)atoi(v);
        else if (!strcmp(k, "vq_init")) snprintf(c->vq_init, sizeof c->vq_init, "%s", v);   /* V 步起点补丁目录(见 pt_cfg.vq_init) */
        else if (!strcmp(k, "vq_frac")) c->vq_frac = (float)atof(v);
        else if (!strcmp(k, "data_eval")) c->data_eval = (uint32_t)atoi(v);   /* 1 = 给了 eval= 料也照常五取一留出(V 步两边信赖域要两份留出) */
        else if (!strcmp(k, "batch")) c->batch = (uint32_t)atoi(v);
        else if (!strcmp(k, "maxlen")) c->maxlen = (uint32_t)atoi(v);
        else if (!strcmp(k, "ctxlen")) c->ctxlen = (uint32_t)atoi(v);   /* 前缀形态的样本上限(见 core_ptrain.h pt_cfg.ctxlen) */
        else if (!strcmp(k, "eval_n")) c->eval_n = (uint32_t)atoi(v);
        else if (!strcmp(k, "doc_tail")) c->doc_tail = (uint32_t)atoi(v);   /* 每篇只训末 K 段(见 pt_cfg.doc_tail) */   /* 留出段只评几段(见 pt_cfg.eval_n) */
        else if (!strcmp(k, "probe_n")) c->probe_n = (uint32_t)atoi(v);
        else if (!strcmp(k, "probe_tok")) c->probe_tok = (uint32_t)atoi(v);
        else if (!strcmp(k, "seed")) c->seed = (uint32_t)atoi(v);
        else if (!strcmp(k, "gradcheck")) c->gradcheck = (uint32_t)atoi(v);
        else if (!strcmp(k, "prof")) c->prof = (uint32_t)atoi(v);
        else if (!strcmp(k, "rccheck")) c->rccheck = (uint32_t)atoi(v);
        else if (!strcmp(k, "contcheck")) c->contcheck = (uint32_t)atoi(v);
        else if (!strcmp(k, "diag")) c->diag = (uint32_t)atoi(v);
        else if (!strcmp(k, "max_steps")) c->max_steps = (uint32_t)atoi(v);
        else if (!strcmp(k, "epoch_tok")) c->epoch_tok = (uint32_t)atoi(v);
        else if (!strcmp(k, "packcheck")) c->packcheck = (uint32_t)atoi(v);
        else if (!strcmp(k, "teachercheck")) c->teachercheck = (uint32_t)atoi(v);
        else if (!strcmp(k, "nopack")) c->nopack = (uint32_t)atoi(v);
        else if (!strcmp(k, "dbglayer")) c->dbg_layer = atoi(v);
        else if (!strcmp(k, "poison")) c->poison = (uint32_t)atoi(v);
        else if (!strcmp(k, "gclayers")) {
            for (char *q = v, *nx; *q && c->ngcl < 16u; q = nx) {
                c->gcl[c->ngcl] = (uint32_t)strtoul(q, &nx, 10);
                if (nx == q) break;
                c->ngcl++;
                while (*nx == '/' || *nx == ' ') nx++;
            }
        }
        else if (!strcmp(k, "hccheck")) {   /* 层号用 / 隔开(脚本的额外配置按逗号拆行, 这里不能用逗号) */
            for (char *q = v, *nx; *q && c->nhcl < 16u; q = nx) {
                c->hcl[c->nhcl] = (uint32_t)strtoul(q, &nx, 10);
                if (nx == q) break;
                c->hcmid[c->nhcl++] = *nx == 'm' ? 1u : 0u;
                if (*nx == 'm') nx++;
                while (*nx == '/' || *nx == ' ') nx++;
            }
        }
        else if (!strcmp(k, "lr")) c->lr = (float)atof(v);
        else if (!strcmp(k, "clip")) c->clip = (float)atof(v);
        else if (!strcmp(k, "init_std")) c->init_std = (float)atof(v);
        else { fprintf(stderr, "ds4: --ptrain 不认识的配置项 %s\n", k); free(b); return false; }
    }
    free(b);
    if (!c->data[0] || !c->out[0]) { fprintf(stderr, "ds4: --ptrain 配置缺 data= 或 out=\n"); return false; }
    if (c->layer_hi >= DS4_N_LAYER || c->layer_lo > c->layer_hi) { fprintf(stderr, "ds4: --ptrain layers 越界(%u-%u, 模型 %u 层)\n", c->layer_lo, c->layer_hi, DS4_N_LAYER); return false; }
    /* 反传从出口往下逐层推, 出口梯度直接进第 hi 层: hi 不是末层时中间那几层被跳过, 梯度是错的(不报错, 只是训歪) */
    if (c->layer_hi != DS4_N_LAYER - 1u) { fprintf(stderr, "ds4: --ptrain layers 必须到末层 L%u(给的是 %u-%u)\n", DS4_N_LAYER - 1u, c->layer_lo, c->layer_hi); return false; }
    if (c->evalf[0] && !strcmp(c->evalf, c->data)) { fprintf(stderr, "ds4: --ptrain eval= 与 data= 是同一份料: 泛化留出必须是没进训练的另一份\n"); return false; }
    if (c->evalkl[0] && (!strcmp(c->evalkl, c->data) || c->data_eval)) { fprintf(stderr, "ds4: --ptrain eval_kl= 不能是 data= 本身, 也不能和 data_eval=1 同给(两种留出规则打架)\n"); return false; }
    if (!c->rank || c->topk < 1 || c->topk > 128 || !c->batch || c->maxlen < 16) { fprintf(stderr, "ds4: --ptrain rank/topk(1..128)/batch/maxlen 不合法\n"); return false; }
    if (c->ctxlen && c->ctxlen < c->maxlen) { fprintf(stderr, "ds4: --ptrain ctxlen %u 小于 maxlen %u(ctxlen 是样本上限, 尾窗 = maxlen)\n", c->ctxlen, c->maxlen); return false; }
    if (c->teacher_fp[0] && c->sample_n) { fprintf(stderr, "ds4: --ptrain teacher=fp: 与 sample_n 不能同开: 步 2 的教师是部署态自己当场打分, 两种教师不能混一趟\n"); return false; }
    return true;
}

/* 块: 同一份材料一块(名字 = 材料的 FNV, 教师前缀只预填一次); 保持料共一块(hold); 不带材料的普通题共一块(sft); 原文段共一块(text) */
uint32_t pt_chunk_of(pt_data *d, const char *ctx, int hold, int raw) {
    char name[32];
    if (hold) snprintf(name, sizeof name, "hold");
    else if (raw) snprintf(name, sizeof name, "text");
    else if (!ctx[0]) snprintf(name, sizeof name, "sft");
    else {
        uint64_t h = 1469598103934665603ull;
        for (const unsigned char *p = (const unsigned char *)ctx; *p; p++) { h ^= *p; h *= 1099511628211ull; }
        snprintf(name, sizeof name, "c_%016llx", (unsigned long long)h);
    }
    for (uint32_t i = 0; i < d->nch; i++) if (!strcmp(d->ch[i].name, name)) return i;
    d->ch = realloc(d->ch, (d->nch + 1) * sizeof *d->ch);
    pt_chunk *c = &d->ch[d->nch];
    memset(c, 0, sizeof *c);
    c->name = strdup(name); c->text = strdup(ctx); c->hold = hold; c->sft = !hold && !ctx[0];
    return d->nch++;
}

static int32_t *pt_cat(const ds4_tokens *a, const ds4_tokens *b, uint32_t *n) {
    *n = (uint32_t)(a->len + b->len);
    int32_t *v = xmalloc((size_t)*n * 4);
    for (int i = 0; i < a->len; i++) v[i] = a->v[i];
    for (int i = 0; i < b->len; i++) v[a->len + i] = b->v[i];
    return v;
}

/* 留出 = 问题去空白后 FNV-1a 五取一: 同一道题不管出现在哪份料、哪一趟, 永远在同一边 */
int pt_is_eval(const char *q) {
    uint64_t h = 1469598103934665603ull;
    for (const unsigned char *p = (const unsigned char *)q; *p; p++) { if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') continue; h ^= *p; h *= 1099511628211ull; }
    return h % 5u == 0u;
}

/* one-hot 的表(目标 = 序列里答案位的 token 自己, 余量 0): 同一个 KL 核算出来就是 −log p(目标), 梯度 = 交叉熵的; 不进教师缓存 */
static void pt_fill_onehot(const pt_cfg *c, pt_sample *s) {
    const uint32_t K = c->topk, m = s->m;
    if (c->teacher_fp[0]) return;   /* FP 教师在: 表留空等 pt_teacher_fp 填, 这题不是 one-hot(sft 保持 0) */
    s->sft = 1;
    s->top_id = xmalloc((size_t)m * K * 4); s->top_p = xmalloc((size_t)m * K * 4); s->top_rest = xmalloc((size_t)m * 4);
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t k = 0; k < K; k++) { s->top_id[i * K + k] = k ? -1 : s->sids[s->sa0 + i]; s->top_p[i * K + k] = k ? 0.f : 1.f; }
        s->top_rest[i] = 0.f;
    }
}

/* text 行: BOS + 原文 + EOS, 从第 1 个 token 起每个都是目标。超 maxlen 的按 token 切成连续几段(每段 BOS + ≤ maxlen−1 个 token, EOS 只跟末段),
 * 各段自成样本、同一边留出(按整段原文算 FNV) —— unsloth 是截到 max_seq_length 丢尾, 这里不丢: 料就是一篇文档一行, 丢尾 = 后半篇白传。
 * ★给了 ctxlen(前缀形态)★: 第 k 段 = 原文 [0, en_k)(超 ctxlen 才从前面截), 目标只是本段新的 token, 前面全当上下文。
 *   为什么不按 ctxlen 切不重叠的段: 训练时只有尾窗进损失(core_ptrain_bwd.c), 段长 4096 就有 3/4 的 token 永远不当靶。
 *   段界: en_k = (k+1)·S − 1(S = maxlen−2 以下 pt_pfx_align() 的最大倍数, 现役 1022) ⇒ 第 k 段的序列长 (k+1)·S 对齐到组, 正好是第 k+1 段的前缀长 —— 训练器按篇接力
 *   (pt_run.cont_s), 上一段跑完的状态直接当下一段的前缀, 不重新预填。段界那 1 位(原文第 kS−1 个)由上一段多带的"只当目标"token 补上(xt);
 *   末位(EOS)若正好落在段界, 并进上一段(尾窗 S+1 ≤ maxlen)。行里 "eos": false(原文没写完)时末尾不补 EOS。
 *   整条轨迹一行进来就行 —— 别在料里先切窗: 每行末尾都补 EOS, 切在思考中间的窗 = 教它想到一半就停。 */
uint32_t pt_pfx_align(void) {
    uint32_t a = 2u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) { const uint32_t r = ds4_layer_compress_ratio(il); if (r > a) a = r; }
    return a;
}

/* 每读一行(text 或问答)占一个 doc 号, docids/docn 与 ndoc 同步长, 问答行是 NULL/0。
 * 10-26 实撞: 只在 text 行上长数组, 保持料 63 行也占 doc 号, 收尾按 ndoc 释放就越界 —— 训完、③ 落盘后 SIGSEGV(rc 139), 链误判训练失败。 */
static void pt_doc_slot(pt_data *d) {
    d->docids = realloc(d->docids, (size_t)(d->ndoc + 1u) * sizeof *d->docids); d->docn = realloc(d->docn, (size_t)(d->ndoc + 1u) * sizeof *d->docn);
    d->docids[d->ndoc] = NULL; d->docn[d->ndoc] = 0;
}

/* ★"prefix" 行(10-26 晚, 分支续写料)★: 序列 = BOS + prefix + text(+EOS), prefix 的 token 只当上下文、不进损失(pt_sample.l0), 落在 prefix 里的段不出样本。
 * 为什么: 学生自己生成到两万位后走进官方从没去过的区域(过度核算 / 绕圈), 官方轨迹上的靶教不了它从那里怎么回来; 让官方 API 从学生的前缀往下续写,
 *   前缀是学生的(只给上下文, 零梯度), 靶 100% 是原件的分布 —— 09-30 铁律禁的是"自吐当靶", 这里自吐只是提问的方式。
 * prefix 与 text 各自分词再拼: 分界正好是学生停笔 / 官方接笔的那个 token 边界(API 续写就是从这个前缀 token 串起的), 拼起来再切会在边界合并出别的 token。 */
static bool pt_add_text(ds4_engine *e, const pt_cfg *c, pt_data *d, int src, const char *pfxtxt, const char *txt, int eos) {
    const int eval = src == 2 ? 2 : (src == 0 && c->evalf[0] && !c->data_eval) ? 0 : pt_is_eval(txt);   /* 给了 eval= 时料不留出(同 pt_add_sample); data_eval=1 照常五取一 */
    ds4_tokens at = {0};
    /* 认特殊 token 的切分(10-22): 原文段里的 <think> / </think> / <｜end▁of▁sentence｜> 要成为它们的 id, 思考轨迹尾段才能把 </think> 这个决策位放进 KL;
     * 纯 BPE 会把 "</think>" 切成几个普通片。不含这些字串的原文两条路逐 token 同(只有一个 span, 同一个 BPE) */
    uint32_t np = 0;
    if (pfxtxt[0]) { ds4_tokenize_rendered_chat(e, pfxtxt, &at); np = (uint32_t)at.len; }
    ds4_tokens tt = {0};
    ds4_tokenize_rendered_chat(e, txt, &tt);
    for (int i = 0; i < tt.len; i++) ds4_tokens_push(&at, tt.v[i]);
    ds4_tokens_free(&tt);
    if (eos) ds4_tokens_push(&at, ds4_token_eos(e));
    const bool ok = pt_add_tokens(e, c, d, src, eval, e->vocab.bos_id, at.v, (uint32_t)at.len, np);
    ds4_tokens_free(&at);
    return ok;
}

/* 原文段的切段与建样本(text 行与 ids 行共用, 11-02): 序列 = first + tok[0..T), 前 np 个 tok 只当上下文。text 行 first = BOS;
 * ids 行(core_ptrain_ids.c, 判决尺那种裸 token 流)first = 流的第 0 个 id, 不另加 BOS —— FP 锚就是对这串原样算的, 加了 BOS 教师/学生看到的就不是同一条流 */
bool pt_add_tokens(ds4_engine *e, const pt_cfg *c, pt_data *d, int src, int eval, int32_t first, const int32_t *tok, uint32_t T, uint32_t np) {
    const uint32_t ci = pt_chunk_of(d, "", 0, 1);
    const bool pf = c->ctxlen > c->maxlen;
    const uint32_t S = pf ? (c->maxlen - 2u) / pt_pfx_align() * pt_pfx_align() : c->maxlen - 1u, Wc = c->ctxlen - 1u;
    uint32_t nseg = 0;   /* 前缀形态的段数(与下面的段界同式): doc_tail 按它数末 K 段 */
    if (pf) for (uint32_t k = 0, off = 0; off < T; k++) { uint32_t en = (k + 1u) * S - 1u < T ? (k + 1u) * S - 1u : T; if (en + 1u == T) en = T; nseg++; off = (k + 1u) * S; if (en >= T) break; }
    /* 整篇一条教师序列(10-26): 前缀形态下各段的 tids 都是它的前缀(没被 ctxlen 截时), FP 表按它导一次填所有段, 不按段的窗各导一条(一篇 doc_tail 遍白算) */
    pt_doc_slot(d); uint64_t fpkey = 0;
    if (pf) {
        int32_t *ds = xmalloc((size_t)(T + 1u) * 4); ds[0] = first; memcpy(ds + 1, tok, (size_t)T * 4);
        d->docids[d->ndoc] = ds; d->docn[d->ndoc] = T + 1u; fpkey = pt_ids_fnv(ds, T + 1u);
    }
    for (uint32_t k = 0, off = 0; off < T; k++) {
        uint32_t en = !pf ? (T - off < S ? T : off + S) : ((k + 1u) * S - 1u < T ? (k + 1u) * S - 1u : T);
        if (pf && en + 1u == T) en = T;   /* EOS 落在段界: 并进本段 */
        if (pf && c->doc_tail && k + c->doc_tail < nseg) { off = (k + 1u) * S; continue; }   /* 末 K 段以前: 只当上下文, 不出样本 */
        /* xt: 非末段多带 1 个"只当目标"的 token(原文第 en 个, 存在 sids[sn], 不进前向): 它的 logits 是本段尾窗末行, 下一段的前缀正好到它之前 ——
         * 不带的话段界那一位谁都不当靶(每 S 位丢 1 个, 10-25 晚补上)。sn 仍是输入长度, 接力判据(下一段前缀 == 本段 sn)不变 */
        const uint32_t xt = pf && en < T ? 1u : 0u, st0 = !pf ? off : en > Wc ? en - Wc : 0u, m = en - off + xt;
        if (en + xt <= np) { off = pf ? (k + 1u) * S : en; continue; }   /* 整段都在 prefix 里: 没有一位当靶, 不出样本(与 doc_tail 跳段同理, 下一段从头预填) */
        d->s = realloc(d->s, (d->ns + 1) * sizeof *d->s);
        pt_sample *s = &d->s[d->ns++];
        memset(s, 0, sizeof *s);
        s->chunk = ci; s->eval = eval; s->raw = 1; s->doc = d->ndoc; s->q = strdup(""); s->a = strdup(""); s->src = src;
        s->sn = en - st0 + 1u; s->sa0 = off - st0 + 1u; s->m = m;
        s->l0 = np > off ? np - off : 0u;   /* 本段靶里落在 prefix 的头几位: 只当上下文(上面保证了 l0 ≤ m−1, 至少 1 位进损失) */
        s->sids = xmalloc((size_t)(s->sn + xt) * 4);
        s->sids[0] = first;
        memcpy(s->sids + 1, tok + st0, (size_t)(en - st0 + xt) * 4);
        s->tids = xmalloc((size_t)(s->sn + xt) * 4); memcpy(s->tids, s->sids, (size_t)(s->sn + xt) * 4); s->tn = s->sn; s->ta0 = s->sa0;
        s->fpkey = (pf && st0 == 0u) ? fpkey : 0;   /* 被 ctxlen 截过头的窗不是整篇的前缀, 仍按自己的窗做键 */
        pt_fill_onehot(c, s);
        off = pf ? (k + 1u) * S : en;
    }
    d->ndoc++;
    return true;
}

static bool pt_add_sample(ds4_engine *e, const pt_cfg *c, pt_data *d, int src, const char *sys, const char *ctx, const char *q, const char *a, uint32_t *skipped) {
    const int hold = src == 1;
    const uint32_t ci = pt_chunk_of(d, ctx, hold, 0);
    ds4_tokens sp = {0}, tp = {0}, at = {0};
    ds4_encode_chat_prompt(e, sys, q, DS4_THINK_NONE, &sp);
    if (ctx[0]) {
        const size_t tl = strlen(ctx) + strlen(q) + 3;
        char *tt = xmalloc(tl);
        snprintf(tt, tl, "%s\n\n%s", ctx, q);
        ds4_encode_chat_prompt(e, sys, tt, DS4_THINK_NONE, &tp);
        free(tt);
    } else ds4_encode_chat_prompt(e, sys, q, DS4_THINK_NONE, &tp);   /* 保持料 / 普通题: 教师提示 = 学生提示 */
    ds4_tokenize_text(e, a, &at);
    ds4_tokens_push(&at, ds4_token_eos(e));
    /* one-hot 的题(不带材料)可到 ctxlen(前缀形态); 带材料/保持料的教师表是部署态按整条算的, 前缀路不接, 仍以 maxlen 为限 */
    const uint32_t lim = (d->ch[ci].sft && c->ctxlen > c->maxlen) ? c->ctxlen : c->maxlen;
    if ((uint32_t)(sp.len + at.len) > lim) { (*skipped)++; ds4_tokens_free(&sp); ds4_tokens_free(&tp); ds4_tokens_free(&at); return true; }
    d->s = realloc(d->s, (d->ns + 1) * sizeof *d->s);
    pt_sample *s = &d->s[d->ns++];
    memset(s, 0, sizeof *s);
    /* 留出: eval 文件的题全是泛化留出(2); 料的题按 FNV 五取一 —— ★给了 eval= 时料不再留出★(10-11 下午): 泛化留出已经是没进训练的一份,
     * 再从料里抽 20% 走只会吃掉小料里一条事实唯一的信息行(youngai 24 行被抽走 5 行, fact 2/8/9/10 各只剩答案漏在问句里的那行, 挂上去这几条全错);
     * unsloth/TRL 也没有自动留出, eval_dataset 才是留出。保持料(src 1)照旧五取一: 它的留出 KL 量的是"通用回答没被带偏", 与 eval= 无关 */
    pt_doc_slot(d); s->doc = d->ndoc++;
    s->chunk = ci; s->eval = src == 2 ? 2 : (src == 0 && c->evalf[0] && !c->data_eval) ? 0 : pt_is_eval(q); s->q = strdup(q); s->a = strdup(a); s->src = src;
    s->sids = pt_cat(&sp, &at, &s->sn); s->sa0 = (uint32_t)sp.len;
    s->tids = pt_cat(&tp, &at, &s->tn); s->ta0 = (uint32_t)tp.len;
    s->m = (uint32_t)at.len;
    if (d->ch[ci].sft) pt_fill_onehot(c, s);   /* 不带材料: 目标 = 答案 token 本身 */
    if (d->ch[ci].sft && src == 0 && c->sample_n) {   /* 步 2 的教师提示: 原回答当材料, 与带材料的题同一个形状(材料 + 空行 + 问题) —— 全是料里的字, 不加训练链自己的措辞(10-11 用户令) */
        const size_t dl = strlen(q) + strlen(a) + 8;
        char *dt = xmalloc(dl);
        snprintf(dt, dl, "%s\n\n%s", a, q);
        ds4_tokens dp = {0};
        ds4_encode_chat_prompt(e, sys, dt, DS4_THINK_NONE, &dp);
        s->dn = (uint32_t)dp.len; s->dids = xmalloc((size_t)dp.len * 4);
        for (int i = 0; i < dp.len; i++) s->dids[i] = dp.v[i];
        ds4_tokens_free(&dp); free(dt);
    }
    ds4_tokens_free(&sp); ds4_tokens_free(&tp); ds4_tokens_free(&at);
    return true;
}

/* 一行 jsonl → system/context/user/assistant 四串(malloc; 没有的给 ""), 或 text 一串(*txt 非空时另四串都是 ""); 失败时 *err 指向原因(静态串)。
 * 全仓唯一的料行解析器: 扩展(core_ptrain_expand.c)收模型写的行也走它 —— 不是合法料行的就不是料 */
bool pt_parse_line(const char *ln, char **sys, char **ctx, char **q, char **a, char **txt, char **pfx, int *eos, const char **err) {
    *eos = 1;
    const char *p = ln; char *key = NULL, *role = NULL, *content = NULL;
    *sys = *ctx = *q = *a = *txt = *pfx = NULL; *err = "不是 JSON 对象";
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    for (;;) {
        json_ws(&p);
        if (*p == '}') { p++; break; }
        if (!json_string(&p, &key)) { *err = "键不是字符串"; return false; }
        json_ws(&p);
        if (*p != ':') { free(key); *err = "键后面没有冒号"; return false; }
        p++;
        if (!strcmp(key, "context")) {
            if (!json_string(&p, ctx)) { free(key); *err = "context 要是字符串"; return false; }
        } else if (!strcmp(key, "text")) {
            if (!json_string(&p, txt)) { free(key); *err = "text 要是字符串"; return false; }
        } else if (!strcmp(key, "eos")) {   /* text 行可选: false = 原文没写完(撞长度上限的轨迹), 末尾不补 EOS */
            bool b = true;
            if (!json_bool(&p, &b)) { free(key); *err = "eos 要是 true/false"; return false; }
            *eos = b ? 1 : 0;
        } else if (!strcmp(key, "prefix")) {   /* text 行可选: 排在 text 前、只当上下文的原文(分支续写料, 见 pt_add_text) */
            if (!json_string(&p, pfx)) { free(key); *err = "prefix 要是字符串"; return false; }
        } else if (!strcmp(key, "messages")) {
            json_ws(&p);
            if (*p != '[') { free(key); *err = "messages 要是数组"; return false; }
            p++;
            for (;;) {
                json_ws(&p);
                if (*p == ']') { p++; break; }
                if (*p != '{') { free(key); *err = "messages 的元素要是对象"; return false; }
                p++; role = content = NULL;
                for (;;) {
                    json_ws(&p);
                    if (*p == '}') { p++; break; }
                    char *mk = NULL;
                    if (!json_string(&p, &mk)) { free(key); *err = "message 的键不是字符串"; return false; }
                    json_ws(&p);
                    if (*p != ':') { free(key); free(mk); *err = "message 键后面没有冒号"; return false; }
                    p++;
                    bool ok = !strcmp(mk, "role") ? json_string(&p, &role) : !strcmp(mk, "content") ? json_string(&p, &content) : false;
                    if (!ok) { *err = !strcmp(mk, "role") || !strcmp(mk, "content") ? "role/content 要是字符串" : "message 只认 role / content"; free(key); free(mk); return false; }
                    free(mk);
                    json_ws(&p);
                    if (*p == ',') p++;
                }
                if (!role || !content) { free(key); *err = "message 缺 role 或 content"; return false; }
                char **slot = !strcmp(role, "system") ? sys : !strcmp(role, "user") ? q : !strcmp(role, "assistant") ? a : NULL;
                if (!slot) { free(key); *err = "role 只认 system / user / assistant"; return false; }
                if (*slot) { free(key); *err = "只收单轮(system 可选 + user + assistant 各一条)"; return false; }
                *slot = content; free(role); role = content = NULL;
                json_ws(&p);
                if (*p == ',') p++;
            }
        } else { free(key); *err = "只认 messages / context / text(+eos, +prefix) 这几个字段(reward/mode/split 10-10 起不收)"; return false; }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
    }
    json_ws(&p);
    if (*p) { *err = "行尾有多余内容"; return false; }
    if (!*sys) *sys = strdup("");
    if (!*ctx) *ctx = strdup("");
    if (!*txt && !*eos) { *err = "eos 只给 text 行(问答的答案恒以 EOS 收尾)"; return false; }
    if (!*txt && *pfx) { *err = "prefix 只给 text 行(它是 text 前面只当上下文的原文)"; return false; }
    if (!*pfx) *pfx = strdup("");
    if (*txt) {   /* 原文行: 不许再带 messages/context(一行只能是一种料, 混着写说不清目标是什么) */
        if (*q || *a || (*ctx)[0]) { *err = "text 行不能再带 messages / context"; return false; }
        pt_rstrip(*txt);
        if (!(*txt)[0]) { *err = "text 是空的"; return false; }
        if (!*q) *q = strdup(""); if (!*a) *a = strdup("");
        return true;
    }
    *txt = strdup("");
    if (!*q || !*a) { *err = "messages 要有 user 和 assistant"; return false; }
    pt_rstrip(*q); pt_rstrip(*a); pt_rstrip(*ctx);
    if (!(*q)[0] || !(*a)[0]) { *err = "问或答是空的"; return false; }
    return true;
}

/* 读一份 jsonl(src 0 = data, 1 = hold, 2 = eval) */
static bool pt_load_file(ds4_engine *e, const pt_cfg *c, pt_data *d, const char *path, int src, uint32_t *skipped, uint32_t *npfx) {
    char *b = pt_slurp(path);
    if (!b) { fprintf(stderr, "ds4: --ptrain 读不了料 %s\n", path); return false; }
    uint32_t lineno = 0, n0 = d->ns;
    bool ok = true;
    for (char *p = b; ok && p && *p; ) {
        char *nl = strchr(p, '\n');
        if (nl) *nl = 0;
        lineno++;
        const char *q0 = p; while (*q0 == ' ' || *q0 == '\t' || *q0 == '\r') q0++;
        int32_t *ids = NULL; uint32_t nids = 0;
        if (*q0 && pt_parse_ids_line(p, &ids, &nids)) {   /* ids 行(裸 token 流, core_ptrain_ids.c): 保持料不收(教师 = 部署态按问答算, 没有流的形态) */
            if (src == 1) { fprintf(stderr, "ds4: --ptrain %s 第 %u 行: 保持料不收 ids 行\n", path, lineno); ok = false; }
            else ok = pt_add_ids(e, c, d, src, ids, nids);
            free(ids);
        } else if (*q0) {
            char *sys, *ctx, *q, *a, *txt, *pfx; const char *err; int eos = 1;
            if (!pt_parse_line(p, &sys, &ctx, &q, &a, &txt, &pfx, &eos, &err)) { fprintf(stderr, "ds4: --ptrain %s 第 %u 行: %s\n", path, lineno, err); ok = false; }
            else if (src == 1 && (ctx[0] || txt[0])) { fprintf(stderr, "ds4: --ptrain %s 第 %u 行: 保持料只收不带 context 的问答(教师 = 部署态自己)\n", path, lineno); ok = false; }
            else if (txt[0]) { ok = pt_add_text(e, c, d, src, pfx, txt, eos); if (pfx[0]) (*npfx)++; }
            else ok = pt_add_sample(e, c, d, src, sys[0] ? sys : NULL, ctx, q, a, skipped);
            free(sys); free(ctx); free(q); free(a); free(txt); free(pfx);
        }
        p = nl ? nl + 1 : NULL;
    }
    free(b);
    if (ok) fprintf(stderr, "ds4: [ptrain] %s: %u 行, 收 %u 题\n", path, lineno, d->ns - n0);
    return ok;
}

bool pt_data_load(ds4_engine *e, const pt_cfg *c, pt_data *d) {
    memset(d, 0, sizeof *d);
    uint32_t skipped = 0, npfx = 0;
    if (!pt_load_file(e, c, d, c->data, 0, &skipped, &npfx)) return false;
    uint32_t nexp = 0;
    if (c->expand) {   /* 多形式扩展: 训练题改写成 N 种问法(core_ptrain_expand.c), 当 src 0 装进来、全标训练 —— 留出题不扩展(扩展了就是把留出的措辞教给它) */
        char cache[1100]; snprintf(cache, sizeof cache, "%s.expand%u.jsonl", c->data, c->expand);
        if (!pt_expand(e, c, d, cache)) return false;
        const uint32_t n0 = d->ns;
        if (!pt_load_file(e, c, d, cache, 0, &skipped, &npfx)) return false;
        for (uint32_t i = n0; i < d->ns; i++) d->s[i].eval = 0;
        nexp = d->ns - n0;
    }
    if (c->evalkl[0]) {   /* 同形式留出整份从 eval_kl= 来: 当 src 0 装(教师缓存/FP 配表与 data 同一路), 装完全标 eval 1; data= 的行此时全是训练(eval= 给了且 data_eval=0 ⇒ 不五取一) */
        const uint32_t n0 = d->ns;
        if (!pt_load_file(e, c, d, c->evalkl, 0, &skipped, &npfx)) return false;
        for (uint32_t i = n0; i < d->ns; i++) d->s[i].eval = 1;
        fprintf(stderr, "ds4: [ptrain] 同形式留出(eval_kl= %s): %u 题, data= 整份进训练\n", c->evalkl, d->ns - n0);
    }
    if (c->hold[0]) { if (!pt_load_file(e, c, d, c->hold, 1, &skipped, &npfx)) return false; }
    else fprintf(stderr, "ds4: ★[ptrain] 没给 hold=(保持料): ③ 在通用问题上没有东西钉着, 10-01 实撞局部性崩 —— 只在故意对照时这么跑★\n");
    if (c->evalf[0] && !pt_load_file(e, c, d, c->evalf, 2, &skipped, &npfx)) return false;
    if (npfx) {   /* 分支续写料的账: 多少行带 prefix、prefix 位占多少(只当上下文)、靶位多少 —— 看一眼就知道这批料的梯度有多少真落在"从学生前缀往下走" */
        uint64_t ctxt = 0, tgt = 0; uint32_t nsp = 0;
        for (uint32_t i = 0; i < d->ns; i++) if (d->s[i].l0) { nsp++; ctxt += d->s[i].l0; tgt += d->s[i].m - d->s[i].l0; }
        fprintf(stderr, "ds4: [ptrain] 带 prefix 的 text 行 %u: 跨分界的段 %u(前缀位 %llu 只当上下文 / 靶位 %llu)\n", npfx, nsp, (unsigned long long)ctxt, (unsigned long long)tgt);
    }
    pt_data_lcp(d);
    if (c->eval_n) {   /* 料的留出段(eval 1, src 0)按篇等间隔留, 约 eval_n 段: 第 j 篇留下当且仅当 ⌊(j+1)·N/nv⌋ > ⌊j·N/nv⌋(按段数比例抽篇, 散在全料)。
                        * 整篇留不按段抽: 前缀形态按篇接力(pt_run.cont_s), 抽散的段每段都要从 0 重新预填前缀(18:4x 那趟 649 段 43 分钟没评完) */
        uint32_t nv = 0, nd = 0, nk = 0;
        for (uint32_t i = 0; i < d->ns; i++) nv += d->s[i].eval == 1 && d->s[i].src == 0 ? 1u : 0u;
        if (nv > c->eval_n) {
            uint32_t j = 0, last = UINT32_MAX; bool keep = false;
            for (uint32_t i = 0; i < d->ns; i++) if (d->s[i].eval == 1 && d->s[i].src == 0) {
                if (d->s[i].doc != last) { last = d->s[i].doc; keep = (uint64_t)(j + 1u) * c->eval_n / nv > (uint64_t)j * c->eval_n / nv; j++; }
                if (keep) nk++; else { d->s[i].eval = 3; nd++; }
            }
            fprintf(stderr, "ds4: [ptrain] eval_n=%u: 留出 %u 篇 %u 段, 整篇抽留 %u 段评估, 其余 %u 段不训也不评\n", c->eval_n, j, nv, nk, nd);
        }
    }
    uint32_t ne = 0, ng = 0, nsft = 0, nraw = 0, nh = 0, mx = 0, h[4] = { 0, 0, 0, 0 };   /* 学生序列长度分布: 训练缓冲按 maxlen 行预分配, 定 maxlen 要看尾巴有多长 */
    for (uint32_t i = 0; i < d->ns; i++) {
        ne += d->s[i].eval ? 1u : 0u; ng += d->s[i].eval == 2 ? 1u : 0u; nsft += d->s[i].sft && !d->s[i].raw ? 1u : 0u; nraw += d->s[i].raw ? 1u : 0u; nh += d->s[i].src == 1 ? 1u : 0u;
        const uint32_t L = d->s[i].sn;
        if (L > mx) mx = L;
        h[L <= 256u ? 0 : L <= 512u ? 1 : L <= 768u ? 2 : 3]++;
    }
    fprintf(stderr, "ds4: [ptrain] 料: %u 块, %u 题(训练 %u / 留出 %u, 其中泛化留出 %u; 带材料 %u, 不带材料 %u, 原文段 %u, 保持料 %u), 超 maxlen %u 跳过 %u 题; 长度 ≤256/≤512/≤768/更长 = %u/%u/%u/%u, 最长 %u\n",
            d->nch, d->ns, d->ns - ne, ne, ng, d->ns - nsft - nraw - nh, nsft, nraw, nh, c->maxlen, skipped, h[0], h[1], h[2], h[3], mx);
    if (c->expand) fprintf(stderr, "ds4: [ptrain] 多形式扩展 expand=%u: 新问答 %u 组进训练(问答都改写, 教师看原问答; 缓存 %s.expand%u.jsonl)\n", c->expand, nexp, c->data, c->expand);
    if (c->sample_n) fprintf(stderr, "ds4: [ptrain] 步 2 开: 每轮每道训练题学生自己采 %u 份, 教师在学生采出来的 token 上打分\n", c->sample_n);
    return d->ns > 0;
}

/* 每块的公共前缀: 同块各题教师序列逐 id 相同的那一段(封顶到最短提示 −1: 前缀里不许含任何一题自己的问题 token) */
void pt_data_lcp(pt_data *d) {
    for (uint32_t ci = 0; ci < d->nch; ci++) {
        const pt_sample *f = NULL; uint32_t lcp = 0;
        for (uint32_t i = 0; i < d->ns; i++) {
            const pt_sample *s = &d->s[i];
            if (s->chunk != ci) continue;
            if (!f) { f = s; lcp = s->ta0 - 1u; continue; }
            uint32_t k = 0; while (k < lcp && k < s->ta0 - 1u && f->tids[k] == s->tids[k]) k++;
            lcp = k;
        }
        d->ch[ci].lcp = lcp;
    }
}

void pt_data_free(pt_data *d) {
    for (uint32_t i = 0; i < d->nch; i++) { free(d->ch[i].name); free(d->ch[i].text); }
    for (uint32_t i = 0; i < d->ns; i++) {
        pt_sample *s = &d->s[i];
        free(s->q); free(s->a); free(s->sids); free(s->tids); free(s->top_id); free(s->top_p); free(s->top_rest); free(s->dids);
    }
    if (d->docids) for (uint32_t i = 0; i < d->ndoc; i++) free(d->docids[i]);
    free(d->docids); free(d->docn);
    free(d->ch); free(d->s); memset(d, 0, sizeof *d);
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_ptrain_data_nonempty_tu;
