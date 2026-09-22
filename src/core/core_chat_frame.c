/* core_chat_frame.c — 聊天角色帧的拼法(token 级): BOS / <｜System｜> / <｜User｜> / <｜Assistant｜> / <think> 怎么排。
 * 2026-09-21 从 core_bpe.c 拆出(那片顶到 500 行); 文本级的同一套拼法在 src/server/server_dsml_render.c, 两边都以
 * 官方 encoding/encoding.py 为准, 金标单测 tests/server_tests_render_v41.c。
 *
 * ★V4.1 的 system 块只开一次★(官方 render_message: <｜System｜> 在 BOS 之后写一次, thinking 的 effort 前缀与 system 正文
 * 同住这一块; 会话中途再来一条 system 消息才再写一个)。判"块开着没有": 从尾巴往回找最近的一个角色 token ——
 * 是 <｜System｜> = 开着(直接接正文); 是 BOS/User/Assistant/EOS/think = 没开(先写一个)。
 * tokenizer 没这个 token(V4)= 什么都不写, 渲染退回原样。
 * 2026-09-21 实撞: 原来照 V4 写法 BOS 后直接 system 正文, V4.1 每条带 system 的请求都跑在模型没训练过的格式上(bug.md §1)。 */
#include "core_internal.h"

static bool chat_system_block_open(const ds4_vocab *vocab, const token_vec *tokens) {
    for (int i = tokens->len - 1; i >= 0; i--) {
        const int t = tokens->v[i];
        if (t == vocab->system_id) return true;
        if (t == vocab->bos_id || t == vocab->user_id || t == vocab->assistant_id || t == vocab->eos_id ||
            t == vocab->think_start_id || t == vocab->think_end_id) return false;
    }
    return false;
}

static void chat_open_system_block(const ds4_vocab *vocab, token_vec *tokens) {
    if (vocab->system_id >= 0 && !chat_system_block_open(vocab, tokens)) token_vec_push(tokens, vocab->system_id);
}

/* 一次性聊天提示(CLI 单发, V4.1 官方 encoding.py 同序): BOS → [<｜System｜> effort 前缀 system 正文] → <｜User｜>prompt
 * → <｜Assistant｜> → <think> 或 </think>。system 块只在有东西可写(thinking 开 或 有 system 正文)时才开。 */
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out) {
    const ds4_vocab *vocab = &e->vocab;
    if (!prompt) prompt = "";
    token_vec_push(out, vocab->bos_id);
    const char *effort = ds4_think_effort_prefix(think_mode);
    const bool has_system = system && system[0];
    if (effort[0] || has_system) chat_open_system_block(vocab, out);
    if (effort[0]) bpe_tokenize_text(vocab, effort, out);
    if (has_system) bpe_tokenize_text(vocab, system, out);
    token_vec_push(out, vocab->user_id);
    bpe_tokenize_text(vocab, prompt, out);
    token_vec_push(out, vocab->assistant_id);
    token_vec_push(out, ds4_think_mode_enabled(think_mode) ? vocab->think_start_id : vocab->think_end_id);
}

/* 多轮对话(REPL/agent)按块追加: begin(BOS) → [open_system / max 前缀 / system 消息] → user … → assistant 前缀 */
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens) {
    token_vec_push(tokens, e->vocab.bos_id);
}

void ds4_chat_open_system(ds4_engine *e, ds4_tokens *tokens) {
    chat_open_system_block(&e->vocab, tokens);
}

void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens) {
    chat_open_system_block(&e->vocab, tokens);   /* 前缀住 system 块里(官方: <｜System｜>Reasoning Effort: …\n\n{system}) */
    bpe_tokenize_text(&e->vocab, DS4_REASONING_EFFORT_MAX_PREFIX, tokens);
}

void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content) {
    ds4_vocab *vocab = &e->vocab;
    if (!role) role = "user";
    if (!content) content = "";

    if (!strcmp(role, "system") || !strcmp(role, "developer")) {
        chat_open_system_block(vocab, tokens);   /* 紧接 effort 前缀 = 同一块不再写; 会话中途 = 新开一块(官方中途 system 语义) */
        bpe_tokenize_text(vocab, content, tokens);
    } else if (!strcmp(role, "assistant")) {
        token_vec_push(tokens, vocab->assistant_id);
        if (strncmp(content, "<think>", 7) != 0 && strncmp(content, "</think>", 8) != 0) {
            token_vec_push(tokens, vocab->think_end_id);
        }
        bpe_tokenize_text(vocab, content, tokens);
    } else {
        token_vec_push(tokens, vocab->user_id);
        if (!strcmp(role, "tool") || !strcmp(role, "function")) {
            bpe_tokenize_text(vocab, "Tool: ", tokens);
        }
        bpe_tokenize_text(vocab, content, tokens);
    }
}

void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode) {
    token_vec_push(tokens, e->vocab.assistant_id);
    token_vec_push(tokens, ds4_think_mode_enabled(think_mode) ?
                   e->vocab.think_start_id : e->vocab.think_end_id);
}
