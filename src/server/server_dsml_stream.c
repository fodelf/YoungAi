/* server_dsml_stream.c — 机械拆分自 ds4_server.c (5815-6232 行): DSML 流式解码状态机。 */

#include "server_internal.h"

const dsml_syntax dsml_syntaxes[DSML_SYN_COUNT] = {
    [DSML_SYN_LONG] = {
        DS4_TOOL_CALLS_START, DS4_TOOL_CALLS_END,
        DS4_INVOKE_START, DS4_INVOKE_END,
        DS4_PARAM_START, DS4_PARAM_END, true,
    },
    [DSML_SYN_SHORT] = {
        DS4_TOOL_CALLS_START_SHORT, DS4_TOOL_CALLS_END_SHORT,
        DS4_INVOKE_START_SHORT, DS4_INVOKE_END_SHORT,
        DS4_PARAM_START_SHORT, DS4_PARAM_END_SHORT, true,
    },
    [DSML_SYN_PLAIN] = {
        "<tool_calls>", "</tool_calls>",
        "<invoke", "</invoke>",
        "<parameter", "</parameter>", true,
    },
    [DSML_SYN_SPACED] = {
        DS4_TOOL_CALLS_START_SPACED, DS4_TOOL_CALLS_END_SPACED,
        DS4_INVOKE_START_SPACED, DS4_INVOKE_END_SPACED,
        DS4_PARAM_START_SPACED, DS4_PARAM_END_SPACED, false,
    },
};

/* 渲染(工具说明/历史工具调用)用哪种写法。GGUF 里没有 chat template, tokenizer 能看出编码代际的
 * 唯一标记是 <｜System｜>: 官方 encoding.py 加它的同一版把 DSML 标签改成了带空格的 " calls"/" invoke"/" parameter"。
 * 写错代际的代价见 server_types.h DS4_TOOL_CALLS_START_SPACED 处。单测里 ds4_chat_set_system_token 切换两态。 */
const dsml_syntax *dsml_render_syntax(void) {
    return &dsml_syntaxes[ds4_chat_system_token()[0] ? DSML_SYN_SPACED : DSML_SYN_LONG];
}

/* string 值从 DSML 文本还原成真值。实体写法(旧格式)全反转义; 原样写法只还原渲染侧为闭合标签加的
 * 那一个 "&lt;"(append_dsml_parameter_text), 其余字节不动 —— 值里本来就有的 "&lt;" 必须原样交给客户端。 */
char *dsml_value_unescape(const dsml_syntax *syn, const char *s) {
    if (syn->entity_values) return dsml_unescape_text(s);
    buf b = {0};
    const char *end = syn->param_end;
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, "&lt;", 4) && !strncmp(s + 4, end + 1, strlen(end + 1))) {
            ds4_buf_putc(&b, '<');
            s += 4;
        } else {
            ds4_buf_putc(&b, *s++);
        }
    }
    return ds4_buf_take(&b);
}

static size_t dsml_max_tool_start_len(void) {
    size_t max = 0;
    for (size_t i = 0; i < sizeof(dsml_syntaxes) / sizeof(dsml_syntaxes[0]); i++) {
        size_t n = strlen(dsml_syntaxes[i].tool_calls_start);
        if (n > max) max = n;
    }
    return max;
}

static bool dsml_find_tool_start(const char *raw, size_t raw_len,
                                 size_t *pos_out,
                                 const dsml_syntax **syn_out) {
    const char *best = NULL;
    const dsml_syntax *best_syn = NULL;
    for (size_t i = 0; i < sizeof(dsml_syntaxes) / sizeof(dsml_syntaxes[0]); i++) {
        const char *p = find_lit_bounded(raw, raw_len, dsml_syntaxes[i].tool_calls_start);
        if (p && (!best || p < best)) {
            best = p;
            best_syn = &dsml_syntaxes[i];
        }
    }
    if (!best) return false;
    *pos_out = (size_t)(best - raw) + strlen(best_syn->tool_calls_start);
    *syn_out = best_syn;
    return true;
}

static bool dsml_find_tool_start_from(const char *raw, size_t raw_len,
                                      size_t start,
                                      size_t *pos_out,
                                      const dsml_syntax **syn_out) {
    if (start > raw_len) return false;
    size_t rel = 0;
    if (!dsml_find_tool_start(raw + start, raw_len - start, &rel, syn_out)) {
        return false;
    }
    *pos_out = start + rel;
    return true;
}

static bool dsml_attr_is_string_true(const char *raw, size_t raw_len,
                                     size_t tag_start, size_t tag_end) {
    if (tag_end <= tag_start || tag_end > raw_len) return false;
    char *tag = xstrndup(raw + tag_start, tag_end - tag_start);
    char *is_string = dsml_attr(tag, "string");
    bool result = !is_string || !strcmp(is_string, "true");   /* 缺 string 属性 = 字符串: 与最终解析器(dsml_parse_*)的缺省一致, 裸 <parameter 就不带它 */
    free(is_string);
    free(tag);
    return result;
}

#ifdef DS4_SERVER_TEST
/* 缓冲尾巴是某个闭标签(两种写法)的真前缀(≥ min_len 字节) */
static bool raw_suffix_partial_close(const dsml_syntax *syn, const char *raw, size_t raw_len, size_t min_len) {
    for (size_t n = min_len; n <= raw_len && n < 32; n++)
        if (dsml_param_close_partial(syn, raw, raw_len, raw_len - n, min_len)) return true;
    return false;
}

static dsml_decode_state dsml_decode_scan_json_param(const char *raw,
                                                     size_t raw_len,
                                                     size_t pos,
                                                     const dsml_syntax *syn) {
    bool in_string = false;
    bool escaped = false;
    while (pos < raw_len) {
        if (!in_string && dsml_param_close_len(syn, raw, raw_len, pos)) {
            return DSML_DECODE_STRUCTURAL;
        }
        unsigned char c = (unsigned char)raw[pos++];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
        } else if (c == '"') {
            in_string = true;
        }
    }
    if (!in_string && raw_suffix_partial_close(syn, raw, raw_len, 2)) {
        return DSML_DECODE_STRUCTURAL;
    }
    return in_string ? DSML_DECODE_JSON_STRING : DSML_DECODE_JSON_STRUCTURAL;
}

/* Slow reference recognizer used by tests. */
dsml_decode_state dsml_decode_state_for_text(const char *raw, size_t raw_len) {
    if (!raw || raw_len == 0) return DSML_DECODE_OUTSIDE;

    size_t pos = 0;
    const dsml_syntax *syn = NULL;
    if (!dsml_find_tool_start(raw, raw_len, &pos, &syn)) {
        return DSML_DECODE_OUTSIDE;
    }
    bool in_invoke = false;
    char invoke_name[64] = "", param_name[64] = "";

    for (;;) {
        while (pos < raw_len && isspace((unsigned char)raw[pos])) pos++;
        if (pos >= raw_len) return DSML_DECODE_STRUCTURAL;

        if (raw_full_lit(raw, raw_len, pos, syn->tool_calls_end)) {
            return DSML_DECODE_OUTSIDE;
        }
        if (dsml_invoke_close_len(syn, raw, raw_len, pos)) {
            pos += dsml_invoke_close_len(syn, raw, raw_len, pos);
            in_invoke = false;
            continue;
        }
        if (raw_full_lit(raw, raw_len, pos, syn->invoke_start)) {
            const char *tag_end = memchr(raw + pos, '>', raw_len - pos);
            if (!tag_end) return DSML_DECODE_STRUCTURAL;
            pos = (size_t)(tag_end - raw) + 1;
            in_invoke = true;
            continue;
        }
        /* 伪 XML invoke 闭标签 </lightpanda_search>: 已开 invoke 时按工具名判 */
        if (in_invoke) {
            char got[64]; size_t got_len = 0;
            if (dsml_tag_close_at(raw, raw_len, pos, got, sizeof got, &got_len) &&
                !strcmp(got, invoke_name)) {
                pos = got_len;   /* dsml_tag_close_at 的 end_out 是绝对下标 */
                in_invoke = false;
                continue;
            }
        }
        if (dsml_param_open_full(syn, raw, raw_len, pos)) {
            size_t tag_start = pos;
            const char *tag_end_ptr = memchr(raw + pos, '>', raw_len - pos);
            if (!tag_end_ptr) return DSML_DECODE_STRUCTURAL;
            size_t tag_end = (size_t)(tag_end_ptr - raw) + 1;
            bool string_value = dsml_attr_is_string_true(raw, raw_len, tag_start, tag_end);
            pos = tag_end;

            size_t end_len = 0;
            if (string_value) {
                const char *end = dsml_param_close_find(syn, raw + pos, raw_len - pos, &end_len);
                if (!end) {
                    if (raw_suffix_partial_close(syn, raw, raw_len, 2)) {
                        return DSML_DECODE_STRUCTURAL;
                    }
                    return DSML_DECODE_STRING_BODY;
                }
                pos = (size_t)(end - raw) + end_len;
                continue;
            }

            dsml_decode_state json_state =
                dsml_decode_scan_json_param(raw, raw_len, pos, syn);
            if (json_state == DSML_DECODE_STRUCTURAL) {
                const char *end = dsml_param_close_find(syn, raw + pos, raw_len - pos, &end_len);
                if (!end) return DSML_DECODE_STRUCTURAL;
                pos = (size_t)(end - raw) + end_len;
                continue;
            }
            return json_state;
        }
        /* 伪 XML 标签: 第一枚(还没开 invoke)是 invoke, 后面的是参数; 参数一律 string=true */
        char pseudo_name[64]; size_t pseudo_end = 0;
        if (dsml_tag_open_at(raw, raw_len, pos, pseudo_name, sizeof pseudo_name, &pseudo_end)) {
            if (!in_invoke) {
                in_invoke = true;
                snprintf(invoke_name, sizeof invoke_name, "%s", pseudo_name);
                pos = pseudo_end;   /* dsml_tag_open_at 的 end_out 是绝对下标 */
                continue;
            }
            snprintf(param_name, sizeof param_name, "%s", pseudo_name);
            pos = pseudo_end;

            size_t end_len = 0;
            const char *end = dsml_tag_close_find(raw + pos, raw_len - pos, param_name, &end_len);
            if (!end) {
                size_t scan = raw_len > 64 ? raw_len - 64 : 0;
                for (size_t i = raw_len; i > scan; i--) {
                    if (raw[i - 1] == '<' && dsml_tag_close_partial(raw, raw_len, i - 1))
                        return DSML_DECODE_STRUCTURAL;
                }
                return DSML_DECODE_STRING_BODY;
            }
            pos = (size_t)(end - raw) + end_len;
            continue;
        }

        for (size_t i = 0; i < sizeof(dsml_syntaxes) / sizeof(dsml_syntaxes[0]); i++) {
            if (raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].tool_calls_end) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].invoke_start) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].invoke_end) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].param_start) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].param_end))
            {
                return DSML_DECODE_STRUCTURAL;
            }
        }
        return DSML_DECODE_STRUCTURAL;
    }
}
#endif

bool dsml_decode_state_is_tool(dsml_decode_state state) {
    return state != DSML_DECODE_OUTSIDE;
}

/* 下一位是工具调用的协议语法(标签 / 参数头 / JSON 标点 / 闭合标记), 不是参数值。采样请求在这些位取 argmax
 * (server_generate_v41.c v41_syntax_greedy); string 值体与 JSON 字符串照请求采样 —— 值也贪心的话长文件体会复读(上游 22ca6ab 的教训)。 */
bool dsml_decode_state_is_syntax(dsml_decode_state state) {
    return state != DSML_DECODE_OUTSIDE && state != DSML_DECODE_STRING_BODY && state != DSML_DECODE_JSON_STRING;
}

void dsml_decode_tracker_init(dsml_decode_tracker *dt) {
    memset(dt, 0, sizeof(*dt));
    dt->mode = DSML_TRACK_SEARCH;
    dt->decode = DSML_DECODE_OUTSIDE;
}

/* Track where generation is inside a DSML tool call.  This is intentionally a
 * forgiving recognizer, not a validator: malformed DSML still gets parsed later
 * by the normal tool-call parser.  Here we only need enough state to decide
 * whether the next token belongs to protocol syntax or arbitrary payload. */
void dsml_decode_tracker_update(dsml_decode_tracker *dt,
                                       const char *raw, size_t raw_len) {
    if (!dt || !raw) return;

    for (;;) {
        if (dt->mode == DSML_TRACK_DONE) {
            dt->decode = DSML_DECODE_OUTSIDE;
            return;
        }

        if (dt->mode == DSML_TRACK_SEARCH) {
            size_t pos = 0;
            const dsml_syntax *syn = NULL;
            if (!dsml_find_tool_start_from(raw, raw_len, dt->pos, &pos, &syn)) {
                size_t hold = dsml_max_tool_start_len();
                dt->pos = raw_len > hold ? raw_len - hold : 0;
                dt->decode = DSML_DECODE_OUTSIDE;
                return;
            }
            dt->syn = syn;
            dt->pos = pos;
            dt->mode = DSML_TRACK_STRUCTURAL;
            dt->decode = DSML_DECODE_STRUCTURAL;
        }

        if (dt->mode == DSML_TRACK_STRING_BODY) {
            while (dt->pos < raw_len) {
                size_t close = dsml_param_close_len(dt->syn, raw, raw_len, dt->pos);
                size_t close_len = 0;
                if (dt->pseudo_param) {
                    char got[64]; size_t got_len = 0;
                    if (dsml_tag_close_at(raw, raw_len, dt->pos, got, sizeof got, &got_len) &&
                        !strcmp(got, dt->param_name)) { close_len = got_len; }
                    else if (dsml_tag_close_partial(raw, raw_len, dt->pos)) {
                        dt->decode = DSML_DECODE_STRUCTURAL;
                        return;
                    }
                } else if (dsml_param_close_partial(dt->syn, raw, raw_len, dt->pos, 2)) {
                    dt->decode = DSML_DECODE_STRUCTURAL;
                    return;
                }
                if (close || close_len || dsml_calls_end_at(dt->syn, raw, raw_len, dt->pos)) {   /* 块闭合直接接值 = 隐式闭合, 不前进(server_dsml_alias.c) */
                    if (close) dt->pos += close;
                    else if (close_len) dt->pos = close_len;   /* pseudo close_len 是 dsml_tag_close_at 的绝对下标 */
                    dt->pseudo_param = false;
                    dt->mode = DSML_TRACK_STRUCTURAL;
                    dt->decode = DSML_DECODE_STRUCTURAL;
                    goto structural;
                }
                dt->pos++;
            }
            dt->decode = DSML_DECODE_STRING_BODY;
            return;
        }

        if (dt->mode == DSML_TRACK_JSON_PARAM) {
            while (dt->pos < raw_len) {
                if (!dt->json_in_string) {
                    const size_t close = dsml_param_close_len(dt->syn, raw, raw_len, dt->pos);
                    if (close || dsml_calls_end_at(dt->syn, raw, raw_len, dt->pos)) {
                        dt->pos += close;
                        dt->mode = DSML_TRACK_STRUCTURAL;
                        dt->decode = DSML_DECODE_STRUCTURAL;
                        goto structural;
                    }
                    if (dsml_param_close_partial(dt->syn, raw, raw_len, dt->pos, 2)) {
                        dt->decode = DSML_DECODE_STRUCTURAL;
                        return;
                    }
                }

                unsigned char c = (unsigned char)raw[dt->pos++];
                if (dt->json_in_string) {
                    if (dt->json_escaped) {
                        dt->json_escaped = false;
                    } else if (c == '\\') {
                        dt->json_escaped = true;
                    } else if (c == '"') {
                        dt->json_in_string = false;
                    }
                } else if (c == '"') {
                    dt->json_in_string = true;
                }
            }
            dt->decode = dt->json_in_string ?
                DSML_DECODE_JSON_STRING : DSML_DECODE_JSON_STRUCTURAL;
            return;
        }

structural:
        while (dt->mode == DSML_TRACK_STRUCTURAL) {
            while (dt->pos < raw_len && isspace((unsigned char)raw[dt->pos])) dt->pos++;
            if (dt->pos >= raw_len) {
                dt->decode = DSML_DECODE_STRUCTURAL;
                return;
            }

            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->tool_calls_end)) {
                dt->mode = DSML_TRACK_DONE;
                dt->pos += strlen(dt->syn->tool_calls_end);
                dt->decode = DSML_DECODE_OUTSIDE;
                return;
            }
            if (dsml_invoke_close_len(dt->syn, raw, raw_len, dt->pos)) {
                dt->pos += dsml_invoke_close_len(dt->syn, raw, raw_len, dt->pos);
                dt->in_invoke = false;
                continue;
            }
            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->invoke_start)) {
                const char *tag_end = memchr(raw + dt->pos, '>', raw_len - dt->pos);
                if (!tag_end) {
                    dt->decode = DSML_DECODE_STRUCTURAL;
                    return;
                }
                dt->pos = (size_t)(tag_end - raw) + 1;
                dt->in_invoke = true;
                continue;
            }
            /* 伪 XML invoke 闭标签 </lightpanda_search>: 已开 invoke 时按工具名判, 与最终解析同一套标签规则 */
            if (dt->in_invoke) {
                char got[64]; size_t got_len = 0;
                if (dsml_tag_close_at(raw, raw_len, dt->pos, got, sizeof got, &got_len) &&
                    !strcmp(got, dt->invoke_name)) {
                    dt->pos = got_len;   /* dsml_tag_close_at 的 end_out 是绝对下标 */
                    dt->in_invoke = false;
                    continue;
                }
            }
            if (dsml_param_open_full(dt->syn, raw, raw_len, dt->pos)) {
                size_t tag_start = dt->pos;
                const char *tag_end = memchr(raw + dt->pos, '>', raw_len - dt->pos);
                if (!tag_end) {
                    dt->decode = DSML_DECODE_STRUCTURAL;
                    return;
                }
                size_t tag_after = (size_t)(tag_end - raw) + 1;
                bool string_value = dsml_attr_is_string_true(raw, raw_len, tag_start, tag_after);
                dt->pos = tag_after;
                if (string_value) {
                    dt->mode = DSML_TRACK_STRING_BODY;
                    dt->decode = DSML_DECODE_STRING_BODY;
                } else {
                    dt->mode = DSML_TRACK_JSON_PARAM;
                    dt->json_in_string = false;
                    dt->json_escaped = false;
                    dt->decode = DSML_DECODE_JSON_STRUCTURAL;
                }
                break;
            }
            /* 伪 XML 标签: 块里的第一枚(还没开 invoke)是 invoke, 后面的是参数名标签; 参数一律 string=true */
            char pseudo_name[64]; size_t pseudo_end = 0;
            if (dsml_tag_open_at(raw, raw_len, dt->pos, pseudo_name, sizeof pseudo_name, &pseudo_end)) {
                if (!dt->in_invoke) {
                    dt->in_invoke = true;
                    snprintf(dt->invoke_name, sizeof dt->invoke_name, "%s", pseudo_name);
                    dt->pos = pseudo_end;   /* dsml_tag_open_at 的 end_out 是绝对下标 */
                    continue;
                }
                dt->pseudo_param = true;
                snprintf(dt->param_name, sizeof dt->param_name, "%s", pseudo_name);
                dt->pos = pseudo_end;
                dt->mode = DSML_TRACK_STRING_BODY;
                dt->decode = DSML_DECODE_STRING_BODY;
                break;
            }
            if (dsml_tag_open_partial(raw, raw_len, dt->pos) || dsml_tag_close_partial(raw, raw_len, dt->pos)) {
                dt->decode = DSML_DECODE_STRUCTURAL;
                return;
            }

            if (raw_partial_lit(raw, raw_len, dt->pos, dt->syn->tool_calls_end) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->invoke_start) ||
                dsml_invoke_close_partial(dt->syn, raw, raw_len, dt->pos) ||
                dsml_param_open_partial(dt->syn, raw, raw_len, dt->pos) ||
                dsml_param_close_partial(dt->syn, raw, raw_len, dt->pos, 1))
            {
                dt->decode = DSML_DECODE_STRUCTURAL;
                return;
            }

            dt->decode = DSML_DECODE_STRUCTURAL;
            return;
        }
    }
}

/* 流式 string 值按块反转义: 块尾若停在一个转义序列的前半截, 先扣住不发, 否则客户端收到半截 "&l"。
 * 原样写法唯一的转义序列是 "&lt;" + 闭合标签剩余部分(dsml_value_unescape)。 */
size_t dsml_entity_stream_safe_len(const dsml_syntax *syn, const char *raw,
                                          size_t start, size_t limit) {
    static const char *ents[] = {"&amp;", "&lt;", "&gt;", "&quot;", "&apos;"};
    char sentinel[64];
    snprintf(sentinel, sizeof(sentinel), "&lt;%s", syn->param_end + 1);
    const char *raw_ents[] = {sentinel};
    const char **list = syn->entity_values ? ents : raw_ents;
    const size_t n = syn->entity_values ? sizeof(ents) / sizeof(ents[0]) : 1;
    const size_t max_ent = syn->entity_values ? 6 : strlen(sentinel);
    size_t scan = limit > start + max_ent ? limit - max_ent : start;
    for (size_t i = limit; i > scan; i--) {
        if (raw[i - 1] != '&') continue;
        size_t amp = i - 1;
        size_t tail = limit - amp;
        for (size_t ei = 0; ei < n; ei++) {
            size_t elen = strlen(list[ei]);
            if (tail < elen && !memcmp(raw + amp, list[ei], tail)) return amp;
        }
        break;
    }
    return limit;
}

size_t dsml_pseudo_value_stream_safe_len(const char *raw, size_t start,
                                               size_t raw_len, const dsml_syntax *syn,
                                               const char *name, bool is_string) {
    (void)name;   /* 完整闭标签由 update 先经 dsml_tag_close_find 认出; 这里只看还没写完的任意 <... 标签 */
    /* 伪 XML 参数的闭标签是 </名字>, 不是 syn->param_end: 值体尾巴停在 "</名字" 的前半截时扣住,
     * 也不发一个还没写完的任意 <... 标签(下一块才能判是不是闭标签)。完整闭标签由 update 先经 dsml_tag_close_find 认出。 */
    size_t limit = raw_len;
    size_t scan = raw_len > start + 64 ? raw_len - 64 : start;
    for (size_t i = raw_len; i > scan; i--) {
        if (raw[i - 1] != '<') continue;
        size_t marker = i - 1;
        if (memchr(raw + marker, '>', raw_len - marker) == NULL) limit = marker;
        break;
    }
    if (is_string) limit = dsml_entity_stream_safe_len(syn, raw, start, limit);
    return utf8_stream_safe_len(raw, start, limit, false);
}

size_t tool_param_value_stream_safe_len(const char *raw, size_t start,
                                               size_t raw_len, const dsml_syntax *syn,
                                               bool is_string) {
    size_t limit = raw_len;
    const size_t end_len = strlen(syn->param_end);   /* 两种闭标签里 DSML 的更长, 往回扫这么多就够 */
    size_t scan = raw_len > start + end_len ? raw_len - end_len : start;
    for (size_t i = raw_len; i > scan; i--) {
        if (raw[i - 1] != '<') continue;
        size_t marker = i - 1;
        if (dsml_param_close_partial(syn, raw, raw_len, marker, 1)) limit = marker;   /* 尾巴像闭标签的开头: 扣住不发 */
        break;
    }
    if (is_string) limit = dsml_entity_stream_safe_len(syn, raw, start, limit);
    return utf8_stream_safe_len(raw, start, limit, false);
}
