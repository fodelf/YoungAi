/* server_v41_live.c — V4.1 单 worker 路: 上一条生成的 token 序列接进下一条提示(2026-10-20)。
 *
 * 为什么要: 助理/聊天每一轮发的提示 = 上一条提示 + 模型刚吐的 + 新的一段。引擎在解码结束存了检查点(core_v41_kvreuse.c 槽 1),
 * 位置 = 提示 + 真喂进 KV 的生成 token, 接得上的条件是下一条提示的 **id 序列** 以它开头。可下一条提示是客户端 JSON 重新渲染、
 * 重新分词的: 文本与模型吐的逐字节同, id 却不同 —— 模型采样出的 token 边界不是分词器的规范切法
 * (10-20 实撞: 36,089 位生成里第 12,229 位分叉, 一个 token 被切成两个), 于是 47,498 的提示只接上 11,358, 重填 36,140 位 36.5 s。
 *
 * 怎么做: 生成完记下这条的渲染文本 + 真吃/真吐的 id(store)。下一条生成前(splice): 它的渲染文本若以记下的文本开头, 且重新分词的
 * 某个 token 边界恰好落在这段文本末尾(后面紧跟的是 <｜end▁of▁sentence｜> 这类特殊 token, 分词器不会跨它合并), 提示 id 就换成
 * "记下的 id + 边界之后重新分词的 id"。模型看到的是它自己吐出的那串 token, 与在线那次完全一样; 引擎的两份检查点按 id 前缀自己判。
 * 边界落在 token 中间(文本续在半个 token 上)就不接, 照旧整段冷分词 —— 不猜。
 * 不接的(如实): 带图片的请求(图片 span 在分词之后才展开, 位置随 id 数变动, 展开表要重算; 助理里图片与多轮工具同现很少, 先不做);
 * 并发调度器那条路(没有检查点, 不需要)。
 * 出错会怎样: 文本不是前缀 ⇒ 不接, 一字节不改; 边界落不到 ⇒ 不接并打一行日志; 两种都只是少一段 KV 复用, 输出不变。 */
#include "server_internal.h"

/* 生成完调(v41_gen_end 释放 g 之前): 文本 = 渲染提示 + 吐出的全文; id = 提示 id + 吐出的 id(去掉没有文本的末尾 EOS、超出 max_tokens 没进正文的那一位) */
void v41_live_store(server *s, const job *j, const v41_gen *g) {
    v41_live_seq *L = &s->v41_live;
    free(L->text); free(L->ids); memset(L, 0, sizeof *L);
    const request *r = &j->req;
    if (!r->prompt_text || r->prompt.len < 1 || !g->ids) return;
    int ng = g->n_ids;
    size_t total = 0;
    for (int i = 0; i < ng; i++) {
        if (g->ids[i] == g->eos) { ng = i; break; }   /* EOS 之后不会再吐 */
        size_t pn = 0; char *p = ds4_token_text(s->engine, g->ids[i], &pn); free(p);
        total += pn;
    }
    while (ng > 0 && total > g->text.len) {   /* 超出 max_tokens 的那一位: 记了 id, 没进正文 */
        size_t pn = 0; char *p = ds4_token_text(s->engine, g->ids[ng - 1], &pn); free(p);
        total -= pn; ng--;
    }
    if (total != g->text.len) {
        server_log(DS4_LOG_WARNING, "ds4-server: ★V4.1 吐出的 id 拼不回正文(id 文本 %zu 字节, 正文 %zu), 这条序列不记★", total, g->text.len);
        return;
    }
    snprintf(L->conv, sizeof L->conv, "%s", r->conv);   /* 最新一份按会话归属(2026-11-23): 换 agent 的请求不接它 */
    const size_t pl = strlen(r->prompt_text);
    L->text = xmalloc(pl + g->text.len + 1);
    memcpy(L->text, r->prompt_text, pl);
    if (g->text.len) memcpy(L->text + pl, g->text.ptr, g->text.len);
    L->text_len = pl + g->text.len; L->text[L->text_len] = '\0';
    L->ids = xmalloc(((size_t)r->prompt.len + (size_t)ng) * sizeof(int32_t));
    memcpy(L->ids, r->prompt.v, (size_t)r->prompt.len * sizeof(int32_t));
    if (ng) memcpy(L->ids + r->prompt.len, g->ids, (size_t)ng * sizeof(int32_t));
    L->n = r->prompt.len + ng;
    L->has_img = r->images != NULL;
}

/* 生成前调(v41_gen_begin 之前, 提示 id 还没进 trace/监控): 条件齐了就把 j->req.prompt 换成 上一条的 id + 边界后重新分词的 id。
 * 返回 true = 本条是上一条会话的续写; false = 不同对话/agent 或边界落进 token 中间 —— 服务端据此把旧检查点离载(2026-11-15)。 */
bool v41_live_splice(server *s, job *j) {
    const v41_live_seq *L = &s->v41_live;
    request *r = &j->req;
    if (!L->ids || L->n < 1 || !r->prompt_text || r->images || L->has_img) return false;
    if (strcmp(L->conv, r->conv) != 0) return false;   /* 不同会话/agent 的同文本也不接(2026-11-23): 旧检查点对它不是 KV */
    const size_t pl = strlen(r->prompt_text);
    if (pl < L->text_len || memcmp(r->prompt_text, L->text, L->text_len) != 0) return false;
    size_t acc = 0; int b = -1;   /* b = 重新分词里第一个落在记下文本之外的 token 序号 */
    for (int i = 0; i <= r->prompt.len; i++) {
        if (acc == L->text_len) { b = i; break; }
        if (acc > L->text_len || i == r->prompt.len) break;
        size_t pn = 0; char *p = ds4_token_text(s->engine, r->prompt.v[i], &pn); free(p);
        acc += pn;
    }
    if (b < 0) {
        server_log(DS4_LOG_GENERATION, "ds4-server: V4.1 上一条序列(%d token, %zu 字节)是本条提示的前缀, 但重新分词的边界落在 token 中间, 不接, 整段冷分词", L->n, L->text_len);
        return false;
    }
    const int rest = r->prompt.len - b;
    int32_t *nv = xmalloc(((size_t)L->n + (size_t)rest) * sizeof(int32_t));
    memcpy(nv, L->ids, (size_t)L->n * sizeof(int32_t));
    if (rest) memcpy(nv + L->n, r->prompt.v + b, (size_t)rest * sizeof(int32_t));
    ds4_tokens_free(&r->prompt);
    r->prompt.v = nv; r->prompt.len = r->prompt.cap = L->n + rest;
    server_log(DS4_LOG_GENERATION, "ds4-server: V4.1 提示沿用上一条的 %d 个 token(重新分词是 %d 个)+ 新段 %d 个 = %d", L->n, b, rest, r->prompt.len);
    return true;
}

/* 本条与记下的最新一份是否同会话: false = 换 agent/会话, 服务端要把进程级两槽离载到盘(仍归旧 conv), 放显存;
 * true 但 splice 失败(边界落 token 中/图片) = 同会话的不同分支, 两槽留着不离载, 冷预填后覆盖槽(2026-11-23)。 */
bool v41_live_same_conversation(server *s, const job *j) {
    const v41_live_seq *L = &s->v41_live;
    return L->ids && L->n > 0 && strcmp(L->conv, j->req.conv) == 0;
}
