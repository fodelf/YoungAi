/* server_tool_schema.c — 机械拆分自 ds4_server.c (1093-1510 行): 工具 schema 归一化与属性序。 */

#include "server_internal.h"

static bool parse_function_call(const char **p, tool_call *tc) {
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) goto bad;
        json_ws(p);
        if (**p != ':') {
            free(key);
            goto bad;
        }
        (*p)++;
        if (!strcmp(key, "name")) {
            free(tc->name);
            if (!json_string(p, &tc->name)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "arguments")) {
            free(tc->arguments);
            json_ws(p);
            if (**p == '"') {
                if (!json_string(p, &tc->arguments)) {
                    free(key);
                    goto bad;
                }
            } else if (!json_raw_value(p, &tc->arguments)) {
                free(key);
                goto bad;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') goto bad;
    (*p)++;
    return true;
bad:
    return false;
}

bool parse_tool_calls_value(const char **p, tool_calls *calls) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') return false;
        (*p)++;
        tool_call tc = {0};
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) goto bad;
            json_ws(p);
            if (**p != ':') {
                free(key);
                goto bad;
            }
            (*p)++;
            if (!strcmp(key, "id")) {
                free(tc.id);
                if (!json_string(p, &tc.id)) {
                    free(key);
                    goto bad;
                }
            } else if (!strcmp(key, "function")) {
                if (!parse_function_call(p, &tc)) {
                    free(key);
                    goto bad;
                }
            } else if (!json_skip_value(p)) {
                free(key);
                goto bad;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != '}') goto bad;
        (*p)++;
        if (tc.name && tc.arguments) {
            tool_calls_push(calls, tc);
            memset(&tc, 0, sizeof(tc));
        }
        tool_call_free(&tc);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
        continue;
bad:
        tool_call_free(&tc);
        return false;
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

void append_raw_json_line(buf *b, const char *json) {
    if (!json || !json[0]) return;
    if (b->len) buf_putc(b, '\n');
    buf_puts(b, json);
}

char *openai_function_schema_from_tool(const char *raw) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        char *value = NULL;
        if (!json_string(&p, &key)) return NULL;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            return NULL;
        }
        p++;
        if (!strcmp(key, "function")) {
            free(key);
            if (!json_raw_value(&p, &value)) return NULL;
            return value;
        }
        free(key);
        if (!json_skip_value(&p)) return NULL;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    return NULL;
}

char *responses_special_schema_from_tool(const char *raw) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;

    char *type = NULL;
    char *description = NULL;
    char *parameters = NULL;
    char *out = NULL;

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
        } else if (!strcmp(key, "description")) {
            free(description);
            if (!json_string(&p, &description)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "parameters")) {
            free(parameters);
            if (!json_raw_value(&p, &parameters)) {
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

    if (type && !strcmp(type, "tool_search")) {
        buf b = {0};
        buf_puts(&b, "{\"name\":\"tool_search\",\"description\":");
        json_escape(&b, description ? description : "Search available tools.");
        buf_puts(&b, ",\"parameters\":");
        buf_puts(&b, parameters ? parameters :
                 "{\"type\":\"object\",\"properties\":{}}");
        buf_putc(&b, '}');
        out = buf_take(&b);
    }

done:
    free(type);
    free(description);
    free(parameters);
    return out;
}

char *responses_namespace_function_schema_from_tool(const char *raw,
                                                           const char *namespace,
                                                           char **wire_name) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;

    char *type = NULL;
    char *name = NULL;
    char *description = NULL;
    char *parameters = NULL;
    char *out = NULL;

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
        } else if (!strcmp(key, "description")) {
            free(description);
            if (!json_string(&p, &description)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "parameters") || !strcmp(key, "input_schema")) {
            free(parameters);
            if (!json_raw_value(&p, &parameters)) {
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

    if ((!type || !strcmp(type, "function")) && namespace && name && name[0]) {
        buf prompt_name = {0};
        buf_puts(&prompt_name, namespace);
        buf_puts(&prompt_name, name);

        buf b = {0};
        buf_puts(&b, "{\"name\":");
        json_escape(&b, prompt_name.ptr ? prompt_name.ptr : name);
        buf_puts(&b, ",\"description\":");
        json_escape(&b, description ? description : "");
        buf_puts(&b, ",\"parameters\":");
        buf_puts(&b, parameters ? parameters :
                 "{\"type\":\"object\",\"properties\":{}}");
        buf_putc(&b, '}');
        out = buf_take(&b);
        if (wire_name) *wire_name = xstrdup(name);
        buf_free(&prompt_name);
    }

done:
    free(type);
    free(name);
    free(description);
    free(parameters);
    return out;
}

static bool parse_schema_properties(const char *json, tool_schema_order *order) {
    const char *p = json;
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) return false;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            return false;
        }
        p++;
        if (!strcmp(key, "properties")) {
            free(key);
            json_ws(&p);
            if (*p != '{') return false;
            p++;
            json_ws(&p);
            while (*p && *p != '}') {
                char *prop = NULL;
                if (!json_string(&p, &prop)) return false;
                json_ws(&p);
                if (*p != ':') {
                    free(prop);
                    return false;
                }
                p++;
                tool_schema_order_prop_push(order, prop);
                if (!json_skip_value(&p)) return false;
                json_ws(&p);
                if (*p == ',') p++;
                json_ws(&p);
            }
            if (*p != '}') return false;
            p++;
        } else if (!strcmp(key, "required")) {
            free(key);
            json_ws(&p);
            if (*p != '[') return false;
            p++;
            json_ws(&p);
            while (*p && *p != ']') {
                char *rq = NULL;
                if (!json_string(&p, &rq)) return false;
                tool_schema_order_req_push(order, rq);
                json_ws(&p);
                if (*p == ',') p++;
                json_ws(&p);
            }
            if (*p != ']') return false;
            p++;
        } else {
            free(key);
            if (!json_skip_value(&p)) return false;
        }
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    return *p == '}';
}

void tool_schema_orders_add_json_wire(tool_schema_orders *orders,
                                             const char *json,
                                             const char *namespace,
                                             const char *wire_name,
                                             bool responses_tool_search) {
    if (!orders || !json) return;
    const char *p = json;
    json_ws(&p);
    if (*p != '{') return;
    p++;
    tool_schema_order order = {0};
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
        if (!strcmp(key, "name")) {
            free(order.name);
            if (!json_string(&p, &order.name)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "input_schema") || !strcmp(key, "parameters")) {
            char *schema = NULL;
            if (!json_raw_value(&p, &schema)) {
                free(key);
                goto done;
            }
            parse_schema_properties(schema, &order);
            free(schema);
        } else if (!json_skip_value(&p)) {
            free(key);
            goto done;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (order.name) {
        if (namespace && namespace[0]) order.namespace = xstrdup(namespace);
        if (wire_name && wire_name[0]) order.wire_name = xstrdup(wire_name);
        order.responses_tool_search = responses_tool_search;
        tool_schema_orders_push(orders, order);
        memset(&order, 0, sizeof(order));
    }
done:
    tool_schema_order_free(&order);
}
