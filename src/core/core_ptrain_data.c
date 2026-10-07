/* core_ptrain_data.c — 后训练 ③ 第八版的料: 配置、分词、教师 top-K(2026-10-01)。总述见 core_ptrain.h。
 *
 * 料的形状(z_nightly_spark.sh kdgen 产出): 清单每行 `<块.txt> <问答.qa> <train|eval> [hard|rft|kl] [权重]`; .qa 是 "#Q\n问题\n#A\n答案\n[#W 数]\n" 的串。
 * hard = 答案是代码算的(账本料): 目标换成答案 token 的 one-hot + 数字位加权(pt_add_sample), 不走教师。
 * rft = 模型自己抽的答案 + 代码结算的奖励(#W 行; 10-04 奖励回路): one-hot × (奖励 − 同题组均值)(pt_rft_group) = REINFORCE 的梯度, 负的往下压;
 * kl = 自采样的锚: 同一串 token 的教师表(教师挂 anchor= 的 ③ = 本轮起点), 全 token 权重 = 清单第 5 列 β。
 * 渲染一律走部署同一个函数(ds4_encode_chat_prompt: BOS <｜User｜>正文 <｜Assistant｜></think>, 无 system、不思考):
 *   学生 = 问题;  教师 = 块正文 + 空行 + 问题;  两边接同一串答案 token + EOS(答案单独分词, 两边逐 id 相同, 位置一一对齐)。
 * 教师的块前缀(同块各题共享的那几千个 token)只预填一次: 按部署的 CED 口径跑完前缀 → 存状态快照 → 每题从快照续算
 * "前缀最后 256 个 token + 问题 + 答案"(这一段跑满解码器, 与生成时提示末块同一种状态)。一题从 3~4 s 降到半秒。 */
#include "core_ptrain.h"
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
    c->layer_lo = c->layer_hi = DS4_N_LAYER - 1u;   /* 第一阶段: 只训末层 */
    c->rank = 64; c->topk = 64; c->epochs = 2; c->batch = 4; c->maxlen = 1024; c->probe_n = 4; c->probe_tok = 96; c->seed = 1; c->dbg_layer = -1;
    c->lr = 1e-3f; c->clip = 1.0f; c->init_std = 0.f;
    c->hard_num = 4.f; c->hard_txt = 0.25f;   /* 一道决策题 ~70 个答案 token 里数字只有 ~10 个: 4 : 0.25 让数字段与叙事段的总权重相当(10-04) */
    c->rft_scale = 1.f;   /* 奖励单位是百分点收益(一天 ±几个点), 优势 ±1~3 与硬目标的 4 同量级, 不另放大 */
    c->eval0 = c->probe0 = 1;
    c->probe_batch = 1; c->sample_batch = 8;
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
        else if (!strcmp(k, "init")) snprintf(c->init, sizeof c->init, "%s", v);
        else if (!strcmp(k, "probe_q")) snprintf(c->probe_q, sizeof c->probe_q, "%s", v);
        else if (!strcmp(k, "teacher")) snprintf(c->teacher, sizeof c->teacher, "%s", v);
        else if (!strcmp(k, "anchor")) snprintf(c->anchor, sizeof c->anchor, "%s", v);
        else if (!strcmp(k, "sample_n")) c->sample_n = (uint32_t)atoi(v);
        else if (!strcmp(k, "probe_base")) snprintf(c->probe_base, sizeof c->probe_base, "%s", v);
        else if (!strcmp(k, "eval0")) c->eval0 = (uint32_t)atoi(v);
        else if (!strcmp(k, "probe0")) c->probe0 = (uint32_t)atoi(v);
        else if (!strcmp(k, "probe_batch")) c->probe_batch = (uint32_t)atoi(v);
        else if (!strcmp(k, "sample_batch")) c->sample_batch = (uint32_t)atoi(v);
        else if (!strcmp(k, "rft_scale")) c->rft_scale = (float)atof(v);
        else if (!strcmp(k, "layers")) { unsigned a = 0, z = 0; if (sscanf(v, "%u-%u", &a, &z) == 2) { c->layer_lo = a; c->layer_hi = z; } else c->layer_lo = c->layer_hi = (uint32_t)atoi(v); }
        else if (!strcmp(k, "rank")) c->rank = (uint32_t)atoi(v);
        else if (!strcmp(k, "topk")) c->topk = (uint32_t)atoi(v);
        else if (!strcmp(k, "epochs")) c->epochs = (uint32_t)atoi(v);
        else if (!strcmp(k, "batch")) c->batch = (uint32_t)atoi(v);
        else if (!strcmp(k, "maxlen")) c->maxlen = (uint32_t)atoi(v);
        else if (!strcmp(k, "probe_n")) c->probe_n = (uint32_t)atoi(v);
        else if (!strcmp(k, "probe_tok")) c->probe_tok = (uint32_t)atoi(v);
        else if (!strcmp(k, "seed")) c->seed = (uint32_t)atoi(v);
        else if (!strcmp(k, "gradcheck")) c->gradcheck = (uint32_t)atoi(v);
        else if (!strcmp(k, "prof")) c->prof = (uint32_t)atoi(v);
        else if (!strcmp(k, "rccheck")) c->rccheck = (uint32_t)atoi(v);
        else if (!strcmp(k, "diag")) c->diag = (uint32_t)atoi(v);
        else if (!strcmp(k, "max_steps")) c->max_steps = (uint32_t)atoi(v);
        else if (!strcmp(k, "epoch_tok")) c->epoch_tok = (uint32_t)atoi(v);
        else if (!strcmp(k, "packcheck")) c->packcheck = (uint32_t)atoi(v);
        else if (!strcmp(k, "nopack")) c->nopack = (uint32_t)atoi(v);
        else if (!strcmp(k, "dbglayer")) c->dbg_layer = atoi(v);
        else if (!strcmp(k, "poison")) c->poison = (uint32_t)atoi(v);
        else if (!strcmp(k, "hard_num")) c->hard_num = (float)atof(v);
        else if (!strcmp(k, "hard_txt")) c->hard_txt = (float)atof(v);
        else if (!strcmp(k, "gclayers")) {
            for (char *q = v, *nx; *q && c->ngcl < 16u; q = nx) {
                c->gcl[c->ngcl] = (uint32_t)strtoul(q, &nx, 10);
                if (nx == q) break;
                c->ngcl++;
                while (*nx == '/' || *nx == ' ') nx++;
            }
        }
        else if (!strcmp(k, "hccheck")) {   /* 层号用 / 隔开(kdtrain 段的额外配置按逗号拆行, 这里不能用逗号) */
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
    if (!c->rank || c->topk < 1 || c->topk > 128 || !c->batch || c->maxlen < 16) { fprintf(stderr, "ds4: --ptrain rank/topk(1..128)/batch/maxlen 不合法\n"); return false; }
    return true;
}

uint32_t pt_userq_load(const char *path, char ***out) {
    *out = NULL;
    if (!path[0]) return 0;
    char *b = pt_slurp(path);
    if (!b) { fprintf(stderr, "ds4: --ptrain 读不了探针题 %s\n", path); return UINT32_MAX; }
    uint32_t n = 0;
    for (char *ln = strtok(b, "\n"); ln; ln = strtok(NULL, "\n")) {
        pt_rstrip(ln);
        if (!*ln) continue;
        char *q = ln[0] == '@' ? pt_slurp(ln + 1) : strdup(ln);   /* "@路径" = 整个文件当一道题(多行提示, 如决策日材料节选 + 问题; 10-03 kdfwd 决策探针) */
        if (!q) { fprintf(stderr, "ds4: --ptrain 探针题文件 %s 读不了\n", ln + 1); for (uint32_t k = 0; k < n; k++) free((*out)[k]); free(*out); *out = NULL; free(b); return UINT32_MAX; }
        pt_rstrip(q);
        *out = realloc(*out, (n + 1) * sizeof **out);
        (*out)[n++] = q;
    }
    free(b);
    return n;
}

static uint32_t pt_chunk_id(pt_data *d, const char *path) {
    for (uint32_t i = 0; i < d->nch; i++) if (!strcmp(d->ch[i].name, path)) return i;
    d->ch = realloc(d->ch, (d->nch + 1) * sizeof *d->ch);
    pt_chunk *c = &d->ch[d->nch];
    memset(c, 0, sizeof *c);
    c->name = strdup(path); c->text = pt_slurp(path);
    if (!c->text) { fprintf(stderr, "ds4: --ptrain 读不了块 %s\n", path); return UINT32_MAX; }
    pt_rstrip(c->text);
    c->hold = c->text[0] == 0;
    return d->nch++;
}

static int32_t *pt_cat(const ds4_tokens *a, const ds4_tokens *b, uint32_t *n) {
    *n = (uint32_t)(a->len + b->len);
    int32_t *v = xmalloc((size_t)*n * 4);
    for (int i = 0; i < a->len; i++) v[i] = a->v[i];
    for (int i = 0; i < b->len; i++) v[a->len + i] = b->v[i];
    return v;
}

/* 清单第 4 列: 教师表(0) / hard / rft / kl(见文件头); wt = 这道题的权重(rft: 原始奖励; kl: β; 其余不用) */
enum { PT_TAG_SOFT = 0, PT_TAG_HARD = 1, PT_TAG_RFT = 2, PT_TAG_KL = 3 };
static int pt_tag(const char *s) { return !strcmp(s, "hard") ? PT_TAG_HARD : !strcmp(s, "rft") ? PT_TAG_RFT : !strcmp(s, "kl") ? PT_TAG_KL : PT_TAG_SOFT; }

static bool pt_add_sample(ds4_engine *e, const pt_cfg *c, pt_data *d, uint32_t ci, int eval, int tag, float wt, const char *q, const char *a, uint32_t *skipped) {
    ds4_tokens sp = {0}, tp = {0}, at = {0};
    ds4_encode_chat_prompt(e, NULL, q, DS4_THINK_NONE, &sp);
    const size_t tl = strlen(d->ch[ci].text) + strlen(q) + 3;
    char *tt = xmalloc(tl);
    if (d->ch[ci].hold) snprintf(tt, tl, "%s", q);   /* 保持料: 教师不看任何材料 = 部署态自己 */
    else snprintf(tt, tl, "%s\n\n%s", d->ch[ci].text, q);
    ds4_encode_chat_prompt(e, NULL, tt, DS4_THINK_NONE, &tp);
    free(tt);
    ds4_tokenize_text(e, a, &at);
    ds4_tokens_push(&at, ds4_token_eos(e));
    if ((uint32_t)(sp.len + at.len) > c->maxlen) { (*skipped)++; ds4_tokens_free(&sp); ds4_tokens_free(&tp); ds4_tokens_free(&at); return true; }
    d->s = realloc(d->s, (d->ns + 1) * sizeof *d->s);
    pt_sample *s = &d->s[d->ns++];
    memset(s, 0, sizeof *s);
    s->chunk = ci; s->eval = eval; s->q = strdup(q); s->a = strdup(a);
    s->sids = pt_cat(&sp, &at, &s->sn); s->sa0 = (uint32_t)sp.len;
    s->tids = pt_cat(&tp, &at, &s->tn); s->ta0 = (uint32_t)tp.len;
    s->m = (uint32_t)at.len;
    if (tag == PT_TAG_HARD || tag == PT_TAG_RFT) {
        /* ★硬目标★(10-04): 答案是代码算的(账本料), 不该再过"教师信不信"这一道 —— 10-03 实撞: 教师读着规则仍给自己习惯的高目标价 0.5、给规则数字 0.3,
         * 学生学到的是这个混合, 第 2 轮从没把先验带到新股上。教师表直接写成答案 token 的 one-hot(余量 0), 同一个 KL 核算出来就是 −log p(答案),
         * 梯度 = 交叉熵的; 不进教师缓存(core_ptrain_teacher.c 跳过 hard 题)。权重: 含数字的 token(决策数字)= hard_num, 其余(格式/理由)= hard_txt。
         * 自采样(rft)同一张 one-hot, 权重先记原始奖励 r, 全题装完后 pt_rft_group 换成组内优势(同一个核: 权重为负就是把这份答案往下压) */
        const uint32_t K = c->topk, m = s->m;
        s->hard = tag == PT_TAG_RFT ? 2 : 1; s->r = wt;
        s->top_id = xmalloc((size_t)m * K * 4); s->top_p = xmalloc((size_t)m * K * 4); s->top_rest = xmalloc((size_t)m * 4); s->w = xmalloc((size_t)m * 4);
        for (uint32_t i = 0; i < m; i++) {
            for (uint32_t k = 0; k < K; k++) { s->top_id[i * K + k] = k ? -1 : s->sids[s->sa0 + i]; s->top_p[i * K + k] = k ? 0.f : 1.f; }
            s->top_rest[i] = 0.f;
            size_t len = 0; char *t = ds4_token_text(e, s->sids[s->sa0 + i], &len); int dig = 0;
            for (size_t j = 0; t && j < len; j++) if (t[j] >= '0' && t[j] <= '9') { dig = 1; break; }
            free(t);
            s->w[i] = tag == PT_TAG_RFT ? 0.f : dig ? c->hard_num : c->hard_txt;
        }
    } else if (tag == PT_TAG_KL) {   /* 锚: 教师表照算(教师挂 anchor= 的 ③), 全 token 权重 β; 一次性的题, 不进教师缓存 */
        s->nocache = 1; s->w = xmalloc((size_t)s->m * 4);
        for (uint32_t i = 0; i < s->m; i++) s->w[i] = wt;
    }
    ds4_tokens_free(&sp); ds4_tokens_free(&tp); ds4_tokens_free(&at);
    return true;
}

/* .qa: "#Q" 行起一题, "#A" 行起答案, 各自到下一个标记行为止(可多行) */
typedef struct { char *p; size_t n, cap; } pt_buf;
static void pt_buf_line(pt_buf *b, const char *s) {
    const size_t L = strlen(s), need = b->n + L + 2;
    if (need > b->cap) { b->cap = need * 2; b->p = realloc(b->p, b->cap); }
    if (b->n) b->p[b->n++] = '\n';
    memcpy(b->p + b->n, s, L + 1); b->n += L;
}
static bool pt_flush_qa(ds4_engine *e, const pt_cfg *c, pt_data *d, uint32_t ci, int eval, int tag, float wt, pt_buf *q, pt_buf *a, uint32_t *skipped) {
    bool ok = true;
    if (q->n && a->n) {
        pt_rstrip(q->p); pt_rstrip(a->p);
        if (q->p[0] && a->p[0]) ok = pt_add_sample(e, c, d, ci, eval, tag, wt, q->p, a->p, skipped);
    }
    q->n = a->n = 0;
    if (q->p) q->p[0] = 0;
    if (a->p) a->p[0] = 0;
    return ok;
}
/* filew = 清单第 5 列(这份问答每题的缺省权重); 答案段后的 "#W 数" 行只管它前面那一题(rft 的奖励按份不同) */
static bool pt_load_qa(ds4_engine *e, const pt_cfg *c, pt_data *d, uint32_t ci, const char *path, int eval, int tag, float filew, uint32_t *skipped) {
    char *b = pt_slurp(path);
    if (!b) { fprintf(stderr, "ds4: --ptrain 读不了问答 %s\n", path); return false; }
    pt_buf q = {0}, a = {0};
    int mode = 0;   /* 0 还没进题 / 1 问题段 / 2 答案段 */
    float wt = filew;
    bool ok = true;
    for (char *p = b; ok && p; ) {
        char *nl = strchr(p, '\n');
        if (nl) *nl = 0;
        if (!strcmp(p, "#Q")) { ok = pt_flush_qa(e, c, d, ci, eval, tag, wt, &q, &a, skipped); wt = filew; mode = 1; }
        else if (!strcmp(p, "#A")) mode = 2;
        else if (mode == 2 && !strncmp(p, "#W", 2)) wt = (float)atof(p + 2);
        else if (mode == 1) pt_buf_line(&q, p);
        else if (mode == 2) pt_buf_line(&a, p);
        p = nl ? nl + 1 : NULL;
    }
    if (ok) ok = pt_flush_qa(e, c, d, ci, eval, tag, wt, &q, &a, skipped);
    free(q.p); free(a.p); free(b);
    return ok;
}

/* ★组内优势★(10-04 奖励回路): 同一道提示抽的 G 份答案互为基线 —— 优势 = 奖励 − 组均值, 全组同分 = 这组没信息(优势 0, 不进训练)。
 * 不除标准差: 组内收益只差 0.1% 时除出来是 ±1 的噪声放大(Dr. GRPO 的理由); 奖励本身就是百分点收益, 量级够。 */
static void pt_rft_group(pt_data *d, const pt_cfg *c) {
    uint32_t nr = 0, ng = 0, nz = 0; double sa = 0;
    uint8_t *done = xmalloc_zeroed(d->ns + 1, 1);
    for (uint32_t i = 0; i < d->ns; i++) {
        if (d->s[i].hard != 2 || done[i]) continue;
        double sum = 0; uint32_t n = 0;
        for (uint32_t j = i; j < d->ns; j++) if (d->s[j].hard == 2 && !done[j] && !strcmp(d->s[j].q, d->s[i].q)) { sum += d->s[j].r; n++; }
        const double mean = sum / n; ng++;
        for (uint32_t j = i; j < d->ns; j++) {
            pt_sample *s = &d->s[j];
            if (s->hard != 2 || done[j] || strcmp(s->q, d->s[i].q)) continue;
            done[j] = 1; nr++;
            const float A = (float)((s->r - mean) * c->rft_scale);
            if (A == 0.f) { nz++; free(s->top_id); s->top_id = NULL; continue; }   /* 没表 = 不可训(pt_runnable), 不占步 */
            for (uint32_t t = 0; t < s->m; t++) s->w[t] = A;
            /* ★负优势不压 EOS★(10-04 第 0007 次实撞): 答案末位是 pt_add_sample 补的 EOS, 负权重打在它上面 = 教模型"别在这停", 三轮重抽收口 56 → 36 → 22 → 9 份/88,
             * 答案越写越长、可训样本枯竭。差样本差在数字, 不差在结束; 只把它的数字/叙事往下压, 结束位不动(正优势照常加强 EOS) */
            if (A < 0.f) s->w[s->m - 1u] = 0.f;
            sa += fabs((double)A);
        }
    }
    free(done);
    if (nr) fprintf(stderr, "ds4: [ptrain] 自采样 %u 份 / %u 题(同题互为基线): 权重 = (奖励 − 组均值) × %.2g, |权重| 均值 %.3f, 全组同分跳过 %u 份\n",
                    nr, ng, (double)c->rft_scale, nr > nz ? sa / (double)(nr - nz) : 0.0, nz);
}

bool pt_data_load(ds4_engine *e, const pt_cfg *c, pt_data *d) {
    memset(d, 0, sizeof *d);
    char *b = pt_slurp(c->data);
    if (!b) { fprintf(stderr, "ds4: --ptrain 读不了清单 %s\n", c->data); return false; }
    uint32_t skipped = 0;
    for (char *ln = strtok(b, "\n"); ln; ln = strtok(NULL, "\n")) {
        char cp[1024], qp[1024], sp[16], hp[16] = "", wp[32] = "";
        const int nf = ln[0] == '#' ? 0 : sscanf(ln, "%1023s %1023s %15s %15s %31s", cp, qp, sp, hp, wp);   /* 第 4 列 hard/rft/kl, 第 5 列权重(见文件头) */
        if (nf < 3) continue;
        const uint32_t ci = pt_chunk_id(d, cp);
        if (ci == UINT32_MAX) { free(b); return false; }
        const int tag = nf >= 4 ? pt_tag(hp) : PT_TAG_SOFT;
        if (nf >= 4 && tag == PT_TAG_SOFT) { fprintf(stderr, "ds4: --ptrain 清单第 4 列不认识: %s(只认 hard / rft / kl)\n", hp); free(b); return false; }
        if (!pt_load_qa(e, c, d, ci, qp, !strcmp(sp, "eval"), tag, nf >= 5 ? (float)atof(wp) : 1.f, &skipped)) { free(b); return false; }
    }
    free(b);
    pt_rft_group(d, c);
    /* 每块的公共前缀: 同块各题教师序列逐 id 相同的那一段(封顶到最短提示 −1: 前缀里不许含任何一题自己的问题 token) */
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
    uint32_t ne = 0, nh = 0, nr = 0, nk = 0, mx = 0, h[4] = { 0, 0, 0, 0 };   /* 学生序列长度分布: 训练缓冲按 maxlen 行预分配, 定 maxlen 要看尾巴有多长 */
    for (uint32_t i = 0; i < d->ns; i++) {
        ne += d->s[i].eval ? 1u : 0u; nh += d->s[i].hard == 1 ? 1u : 0u; nr += d->s[i].hard == 2 ? 1u : 0u; nk += d->s[i].nocache ? 1u : 0u;
        const uint32_t L = d->s[i].sn;
        if (L > mx) mx = L;
        h[L <= 256u ? 0 : L <= 512u ? 1 : L <= 768u ? 2 : 3]++;
    }
    fprintf(stderr, "ds4: [ptrain] 料: %u 块, %u 题(训练 %u / 留出 %u; 硬目标 %u, 数字 ×%.2g 其余 ×%.2g; 自采样 %u, 锚 %u), 超 maxlen %u 跳过 %u 题; 长度 ≤256/≤512/≤768/更长 = %u/%u/%u/%u, 最长 %u\n",
            d->nch, d->ns, d->ns - ne, ne, nh, (double)c->hard_num, (double)c->hard_txt, nr, nk, c->maxlen, skipped, h[0], h[1], h[2], h[3], mx);
    return d->ns > 0;
}

void pt_data_free(pt_data *d) {
    for (uint32_t i = 0; i < d->nch; i++) { free(d->ch[i].name); free(d->ch[i].text); }
    for (uint32_t i = 0; i < d->ns; i++) {
        pt_sample *s = &d->s[i];
        free(s->q); free(s->a); free(s->sids); free(s->tids); free(s->top_id); free(s->top_p); free(s->top_rest); free(s->w);
    }
    free(d->ch); free(d->s); memset(d, 0, sizeof *d);
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_ptrain_data_nonempty_tu;
