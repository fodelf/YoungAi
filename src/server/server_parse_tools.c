/* server_parse_tools.c — 机械拆分自 ds4_server.c (1511-1920 行): tools/messages/图像块解析。 */

#include "server_internal.h"

void tool_schema_orders_add_json(tool_schema_orders *orders, const char *json) {
    tool_schema_orders_add_json_wire(orders, json, NULL, NULL, false);
}

static bool append_responses_namespace_tool_schemas(buf *schemas,
                                                    tool_schema_orders *orders,
                                                    const char *raw) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return false;
    p++;

    char *type = NULL;
    char *name = NULL;
    char *tools = NULL;
    bool appended = false;

    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto done;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto done;
        }
        p++;
        if (!strcmp(key, "type")) {
            free(type);
            if (!json_string(&p, &type)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "name")) {
            free(name);
            if (!json_string(&p, &name)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "tools")) {
            free(tools);
            if (!json_raw_value(&p, &tools)) {
                free(key);
                goto done;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto done;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }

    if (!type || strcmp(type, "namespace") || !name || !tools) goto done;

    const char *tp = tools;
    json_ws(&tp);
    if (*tp != '[') goto done;
    tp++;
    json_ws(&tp);
    while (*tp && *tp != ']') {
        char *tool_raw = NULL;
        if (!json_raw_value(&tp, &tool_raw)) goto done;
        char *wire_name = NULL;
        char *schema =
            responses_namespace_function_schema_from_tool(tool_raw, name, &wire_name);
        if (schema) {
            append_raw_json_line(schemas, schema);
            tool_schema_orders_add_json_wire(orders, schema, name, wire_name, false);
            appended = true;
        }
        free(schema);
        free(wire_name);
        free(tool_raw);
        json_ws(&tp);
        if (*tp == ',') tp++;
        json_ws(&tp);
    }

done:
    free(type);
    free(name);
    free(tools);
    return appended;
}

/* OpenAI wraps tools as {"type":"function","function":{...}}. Anthropic sends
 * the function schema directly as {"name":...,"input_schema":...}. The DS4
 * prompt wants one raw function schema per line, so unwrap OpenAI tools and keep
 * already-direct schemas unchanged. Responses can additionally group tools in a
 * namespace item; those are flattened for DSML prompt rendering while preserving
 * their client-facing name and namespace for response output. */
bool parse_tools_value(const char **p, char **out, tool_schema_orders *orders) {
    json_ws(p);
    if (json_lit(p, "null")) {
        *out = xstrdup("");
        return true;
    }
    if (**p != '[') return false;
    (*p)++;
    buf schemas = {0};

    json_ws(p);
    while (**p && **p != ']') {
        char *raw = NULL;
        if (!json_raw_value(p, &raw)) goto bad;
        char *function = openai_function_schema_from_tool(raw);
        if (function) {
            append_raw_json_line(&schemas, function);
            tool_schema_orders_add_json(orders, function);
        } else if (!append_responses_namespace_tool_schemas(&schemas, orders, raw)) {
            char *special = responses_special_schema_from_tool(raw);
            if (special) {
                append_raw_json_line(&schemas, special);
                tool_schema_orders_add_json_wire(orders, special,
                                                 NULL, NULL, true);
            } else {
                append_raw_json_line(&schemas, raw);
                tool_schema_orders_add_json(orders, raw);
            }
            free(special);
        }
        free(function);
        free(raw);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto bad;
    (*p)++;
    *out = ds4_buf_take(&schemas);
    return true;
bad:
    ds4_buf_free(&schemas);
    return false;
}

bool parse_messages(const char **p, chat_msgs *msgs) {
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;

    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') return false;
        (*p)++;
        chat_msg msg = {0};
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) goto fail;
            json_ws(p);
            if (**p != ':') {
                free(key);
                goto fail;
            }
            (*p)++;
            if (!strcmp(key, "role")) {
                free(msg.role);
                if (!json_string(p, &msg.role)) {
                    free(key);
                    goto fail;
                }
            } else if (!strcmp(key, "content")) {
                free(msg.content);
                if (!json_content(p, &msg.content)) {
                    free(key);
                    goto fail;
                }
            } else if (!strcmp(key, "reasoning_content")) {
                free(msg.reasoning);
                if (!json_content(p, &msg.reasoning)) {
                    free(key);
                    goto fail;
                }
            } else if (!strcmp(key, "tool_call_id")) {
                char *id = NULL;
                if (!json_string(p, &id)) {
                    free(key);
                    goto fail;
                }
                chat_msg_add_tool_call_id(&msg, id);
                free(id);
            } else if (!strcmp(key, "tool_calls")) {
                tool_calls_free(&msg.calls);
                if (!parse_tool_calls_value(p, &msg.calls)) {
                    free(key);
                    goto fail;
                }
            } else if (!json_skip_value(p)) {
                free(key);
                goto fail;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != '}') goto fail;
        (*p)++;
        if (!msg.role) msg.role = xstrdup("user");
        if (!msg.content) msg.content = xstrdup("");
        chat_msgs_push(msgs, msg);
        memset(&msg, 0, sizeof(msg));
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
        continue;
fail:
        chat_msg_free(&msg);
        return false;
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

bool append_anthropic_block_content(buf *dst, const char *text) {
    if (!text || !text[0]) return true;
    ds4_buf_puts(dst, text);
    return true;
}

/* Parse an Anthropic image "source" value: {"type","media_type","data"}.
 * Unknown members are skipped; a non-object value is skipped whole so the
 * caller can reject on block classification instead of JSON shape. */
bool json_parse_image_source(const char **p, char **src_type,
                                    char **src_media, char **src_data) {
    json_ws(p);
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *skey = NULL;
        if (!json_string(p, &skey)) return false;
        json_ws(p);
        if (**p != ':') {
            free(skey);
            return false;
        }
        (*p)++;
        bool ok;
        if (!strcmp(skey, "type")) {
            free(*src_type);
            *src_type = NULL;
            ok = json_string(p, src_type);
        } else if (!strcmp(skey, "media_type")) {
            free(*src_media);
            *src_media = NULL;
            ok = json_string(p, src_media);
        } else if (!strcmp(skey, "data")) {
            free(*src_data);
            *src_data = NULL;
            ok = json_string(p, src_data);
        } else {
            ok = json_skip_value(p);
        }
        free(skey);
        if (!ok) return false;
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

/* Decode one base64 image source through the modality registry and append the
 * canonical <image> section.  Shared by user-message image blocks and images
 * nested inside tool_result content (the MCP screenshot loop).  Fail closed on
 * anything unsupported -- no registry (encoder tool not built), url sources
 * (the server does not fetch), unknown media type, bad base64, encoder
 * rejection: a 400 beats answering on top of silently dropped pixels. */
bool mm_image_source_to_text(ds4_mm *mm, const char *src_type,
                                    const char *src_media, const char *src_data,
                                    buf *dst) {
    if (!mm || !src_type || strcmp(src_type, "base64") != 0 ||
        !src_media || !src_data || !ds4_mm_supported(mm, src_media))
        return false;
    uint8_t *raw = NULL;
    size_t raw_len = 0;
    if (ds4_mm_b64_decode(src_data, strlen(src_data), &raw, &raw_len) != 0)
        return false;
    char *mm_text = NULL;
    int mm_rc = ds4_mm_encode_as_text(mm, src_media, raw, raw_len, &mm_text);
    free(raw);
    if (mm_rc != 0 || !mm_text) return false;
    ds4_buf_puts(dst, "<image>\n");
    ds4_buf_puts(dst, mm_text);
    if (mm_text[0] && mm_text[strlen(mm_text) - 1] != '\n') ds4_buf_puts(dst, "\n");
    ds4_buf_puts(dst, "</image>");
    free(mm_text);
    return true;
}

/* tool_result "content": a bare string, null, or an array of content blocks.
 * Real MCP tool results arrive as block arrays; text blocks concatenate exactly
 * like the legacy string path (rendered bytes stay stable, so existing disk-KV
 * prefixes keep matching) and image blocks -- browser/screenshot MCP tools --
 * resolve through the modality registry into the same <image> sections user
 * messages get.  Fail closed only on pixels: an undeliverable image rejects the
 * request, while text-bearing blocks of other types keep their text as before. */
bool json_tool_result_content(const char **p, ds4_mm *mm, char **out) {
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (json_lit(p, "null")) {
        *out = xstrdup("");
        return true;
    }
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = xstrdup("");
        return true;
    }

    (*p)++;
    buf b = {0};
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) goto fail;
            ds4_buf_puts(&b, s);
            free(s);
        } else if (**p == '{') {
            (*p)++;
            char *type = NULL;
            char *text = NULL;
            char *src_type = NULL;
            char *src_media = NULL;
            char *src_data = NULL;
            bool ok = true;
            json_ws(p);
            while (ok && **p && **p != '}') {
                char *key = NULL;
                ok = json_string(p, &key);
                if (ok) {
                    json_ws(p);
                    ok = **p == ':';
                    if (ok) (*p)++;
                }
                if (ok) {
                    if (!strcmp(key, "type")) {
                        free(type);
                        type = NULL;
                        ok = json_string(p, &type);
                    } else if (!strcmp(key, "text")) {
                        free(text);
                        text = NULL;
                        ok = json_string(p, &text);
                    } else if (!strcmp(key, "source")) {
                        ok = json_parse_image_source(p, &src_type, &src_media,
                                                     &src_data);
                    } else {
                        ok = json_skip_value(p);
                    }
                }
                free(key);
                if (ok) {
                    json_ws(p);
                    if (**p == ',') (*p)++;
                    json_ws(p);
                }
            }
            if (ok) ok = **p == '}';
            if (ok) {
                (*p)++;
                if (type && !strcmp(type, "image"))
                    ok = mm_image_source_to_text(mm, src_type, src_media,
                                                 src_data, &b);
                else if (text)
                    ds4_buf_puts(&b, text);
            }
            free(type);
            free(text);
            free(src_type);
            free(src_media);
            free(src_data);
            if (!ok) goto fail;
        } else if (!json_skip_value(p)) {
            goto fail;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto fail;
    (*p)++;
    *out = ds4_buf_take(&b);
    return true;
fail:
    ds4_buf_free(&b);
    return false;
}
