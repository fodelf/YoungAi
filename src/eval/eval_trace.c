/* ds4-eval trace files: writing the run trace and re-grading a saved trace
 * offline (moved verbatim from ds4_eval.c). */

#include "eval_internal.h"

static void trace_write_block(FILE *trace, const char *label, const char *text) {
    if (!trace) return;
    size_t len = text ? strlen(text) : 0;
    /* Counted blocks make regrading robust against model output that happens
     * to contain trace-looking delimiters or embedded CASE headers. */
    fprintf(trace, "%s_BEGIN bytes=%zu\n", label, len);
    if (len) {
        fwrite(text, 1, len, trace);
        if (text[len - 1] != '\n') fputc('\n', trace);
    }
    fprintf(trace, "%s_END\n", label);
}

static const char *think_close_kind_name(eval_think_close_kind kind) {
    switch (kind) {
    case EVAL_THINK_CLOSE_NATURAL: return "natural";
    case EVAL_THINK_CLOSE_SOFT: return "soft_forced";
    case EVAL_THINK_CLOSE_HARD: return "hard_forced";
    case EVAL_THINK_CLOSE_NONE:
    default: return "none";
    }
}

int token_rank_in_top(ds4_session *session, int token, int max_rank) {
    if (token < 0 || max_rank <= 0) return 0;
    ds4_token_score *top = malloc((size_t)max_rank * sizeof(*top));
    if (!top) return 0;
    int n = ds4_session_top_logprobs(session, top, max_rank);
    int rank = 0;
    for (int i = 0; i < n; i++) {
        if (top[i].id == token) {
            rank = i + 1;
            break;
        }
    }
    free(top);
    return rank;
}

void trace_write_header(FILE *trace, const eval_config *cfg,
                        const char *model_name,
                        int ncases,
                        int max_prompt_tokens) {
    if (!trace) return;
    fprintf(trace,
            "# ds4-eval trace\n"
            "started_unix: %lld\n"
            "model: %s\n"
            "model_shape: %s\n"
            "backend: %s\n"
            "ctx: %d\n"
            "max_tokens: %d\n"
            "max_prompt_tokens: %d\n"
            "questions: %d\n"
            "temperature: %.6g\n"
            "top_p: %.6g\n"
            "min_p: %.6g\n"
            "seed: %llu\n"
            "think_mode_requested: %s\n"
            "soft_limit_reply_budget: %d\n"
            "hard_limit_reply_budget: %d\n"
            "soft_limit_think_close_rank: %d\n"
            "\n",
            (long long)time(NULL),
            cfg->model_path,
            model_name ? model_name : "unknown",
            ds4_backend_name(cfg->backend),
            cfg->ctx_size,
            cfg->max_tokens,
            max_prompt_tokens,
            ncases,
            cfg->temperature,
            cfg->top_p,
            cfg->min_p,
            (unsigned long long)cfg->seed,
            ds4_think_mode_name(cfg->think_mode),
            cfg->soft_limit_reply_budget,
            cfg->hard_limit_reply_budget,
            cfg->soft_limit_think_close_rank);
    fflush(trace);
}

void trace_write_case(FILE *trace,
                      const eval_config *cfg,
                      const eval_case *tc,
                      int idx,
                      int ncases,
                      const char *status,
                      const char *error,
                      const char *system_prompt,
                      const char *question_prompt,
                      const char *model_output,
                      ds4_think_mode effective_think_mode,
                      int prompt_tokens,
                      int generated_tokens,
                      double elapsed_sec,
                      const char *picked,
                      const eval_think_close_info *think_close) {
    if (!trace) return;
    int nchoices = eval_case_nchoices(tc);
    fprintf(trace,
            "===== CASE %d/%d %s/%s =====\n"
            "timestamp_unix: %lld\n"
            "source: %s\n"
            "id: %s\n"
            "domain: %s\n"
            "title: %s\n"
            "status: %s\n"
            "picked: %s\n"
            "expected: %s\n"
            "prompt_tokens: %d\n"
            "generated_tokens: %d\n"
            "elapsed_sec: %.3f\n"
            "temperature: %.6g\n"
            "top_p: %.6g\n"
            "min_p: %.6g\n"
            "think_mode_effective: %s\n",
            idx + 1, ncases, tc->source, tc->id,
            (long long)time(NULL),
            tc->source,
            tc->id,
            tc->domain,
            tc->title,
            status,
            picked && *picked ? picked : "?",
            tc->answer,
            prompt_tokens,
            generated_tokens,
            elapsed_sec,
            cfg->temperature,
            cfg->top_p,
            cfg->min_p,
            ds4_think_mode_name(effective_think_mode));
    if (think_close && think_close->kind != EVAL_THINK_CLOSE_NONE) {
        fprintf(trace,
                "think_close: %s\n"
                "think_close_token_index: %d\n"
                "think_close_remaining_budget: %d\n"
                "think_close_rank: %d\n",
                think_close_kind_name(think_close->kind),
                think_close->token_index,
                think_close->remaining_budget,
                think_close->rank);
    } else {
        fprintf(trace, "think_close: none\n");
    }
    if (error && *error) fprintf(trace, "error: %s\n", error);

    if (nchoices > 0) {
        fprintf(trace, "choices:\n");
        for (int i = 0; i < nchoices; i++) {
            fprintf(trace, "  %c. %s\n", 'A' + i, tc->choice[i]);
        }
    } else {
        fprintf(trace, "answer_kind: exact_integer\n");
    }
    trace_write_block(trace, "SYSTEM_PROMPT", system_prompt);
    trace_write_block(trace, "QUESTION_PROMPT", question_prompt);
    trace_write_block(trace, "MODEL_OUTPUT", model_output);
    fputc('\n', trace);
    fflush(trace);
}

static char *read_text_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-eval: cannot open trace '%s': %s\n",
                path, strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4-eval: cannot seek trace '%s': %s\n",
                path, strerror(errno));
        fclose(fp);
        return NULL;
    }
    long len_long = ftell(fp);
    if (len_long < 0) {
        fprintf(stderr, "ds4-eval: cannot tell trace size '%s': %s\n",
                path, strerror(errno));
        fclose(fp);
        return NULL;
    }
    rewind(fp);

    size_t len = (size_t)len_long;
    char *buf = malloc(len + 1);
    if (!buf) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        fclose(fp);
        return NULL;
    }
    if (len && fread(buf, 1, len, fp) != len) {
        fprintf(stderr, "ds4-eval: cannot read trace '%s': %s\n",
                path, ferror(fp) ? strerror(errno) : "short read");
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    buf[len] = '\0';
    if (len_out) *len_out = len;
    return buf;
}

static const char *bounded_strstr(const char *start, const char *end,
                                  const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0) return start;
    if ((size_t)(end - start) < nlen) return NULL;
    for (const char *p = start; p + nlen <= end; p++) {
        if (!memcmp(p, needle, nlen)) return p;
    }
    return NULL;
}

static void copy_span(char *dst, size_t dstlen, const char *start, const char *end) {
    if (dstlen == 0) return;
    while (end > start && (end[-1] == '\r' || end[-1] == '\n')) end--;
    size_t n = (size_t)(end - start);
    if (n >= dstlen) n = dstlen - 1;
    memcpy(dst, start, n);
    dst[n] = '\0';
}

static const char *trace_skip_counted_block(const char *line,
                                            const char *line_end,
                                            const char *end) {
    /*
     * Trace regrading must scan metadata lines without being confused by model
     * output.  Prefer the byte count written by trace_write_block(); the older
     * delimiter-only parser is still accepted in trace_copy_model_output for
     * compatibility with existing trace files.
     */
    const char *begin = bounded_strstr(line, line_end, "_BEGIN bytes=");
    if (!begin || begin == line) return NULL;

    size_t label_len = (size_t)(begin - line);
    if (label_len > 96) return NULL;

    char end_marker[128];
    int marker_len = snprintf(end_marker, sizeof(end_marker), "%.*s_END",
                              (int)label_len, line);
    if (marker_len <= 0 || (size_t)marker_len >= sizeof(end_marker)) return NULL;

    const char *bytes = begin + strlen("_BEGIN bytes=");
    char *endptr = NULL;
    unsigned long long declared = strtoull(bytes, &endptr, 10);
    if (endptr == bytes || endptr > line_end) return NULL;

    const char *content = line_end < end ? line_end + 1 : end;
    if (declared > (unsigned long long)(end - content)) return NULL;

    const char *marker = content + (size_t)declared;
    if (marker < end && *marker == '\n') marker++;
    if ((size_t)(end - marker) < (size_t)marker_len ||
        memcmp(marker, end_marker, (size_t)marker_len) != 0)
        return NULL;

    const char *after_marker = marker + marker_len;
    const char *after_line = memchr(after_marker, '\n', (size_t)(end - after_marker));
    return after_line ? after_line + 1 : end;
}

const char *trace_find_next_case(const char *start, const char *end) {
    const char *p = start;
    while (p < end) {
        const char *line_end = memchr(p, '\n', (size_t)(end - p));
        if (!line_end) line_end = end;

        const char *skip = trace_skip_counted_block(p, line_end, end);
        if (skip && skip > p) {
            p = skip;
            continue;
        }

        const char *case_marker = "===== CASE ";
        size_t marker_len = strlen(case_marker);
        if ((size_t)(line_end - p) >= marker_len &&
            !memcmp(p, case_marker, marker_len))
            return p;

        p = line_end < end ? line_end + 1 : end;
    }
    return NULL;
}

static const char *trace_find_block_begin(const char *start, const char *end,
                                          const char *label) {
    size_t label_len = strlen(label);
    const char *p = start;
    while (p < end) {
        const char *line_end = memchr(p, '\n', (size_t)(end - p));
        if (!line_end) line_end = end;

        if ((size_t)(line_end - p) >= label_len &&
            !memcmp(p, label, label_len))
            return p;

        const char *skip = trace_skip_counted_block(p, line_end, end);
        if (skip && skip > p) {
            p = skip;
            continue;
        }

        p = line_end < end ? line_end + 1 : end;
    }
    return NULL;
}

static bool trace_get_line_field(const char *start, const char *end,
                                 const char *key, char *dst, size_t dstlen) {
    size_t keylen = strlen(key);
    if (dstlen > 0) dst[0] = '\0';

    const char *p = start;
    while (p < end) {
        const char *line_end = memchr(p, '\n', (size_t)(end - p));
        if (!line_end) line_end = end;
        if ((size_t)(line_end - p) >= keylen && !memcmp(p, key, keylen)) {
            copy_span(dst, dstlen, p + keylen, line_end);
            return true;
        }
        p = line_end < end ? line_end + 1 : end;
    }
    return false;
}

static const eval_case *find_eval_case_by_source_id(const char *source,
                                                    const char *id) {
    size_t ncases = eval_case_count();
    for (size_t i = 0; i < ncases; i++) {
        if (eval_case_at(i)->source && eval_case_at(i)->id &&
            !strcmp(eval_case_at(i)->source, source) &&
            !strcmp(eval_case_at(i)->id, id))
            return eval_case_at(i);
    }
    return NULL;
}

char *trace_copy_model_output(const char *case_start, const char *case_end) {
    const char *begin = trace_find_block_begin(case_start, case_end, "MODEL_OUTPUT_BEGIN");
    if (!begin) return NULL;
    const char *line_end = memchr(begin, '\n', (size_t)(case_end - begin));
    if (!line_end) return NULL;
    const char *content = line_end + 1;

    size_t len = 0;
    const char *bytes = bounded_strstr(begin, line_end, "bytes=");
    if (bytes) {
        char *endptr = NULL;
        unsigned long long declared = strtoull(bytes + 6, &endptr, 10);
        if (endptr == bytes + 6 || endptr > line_end ||
            declared > (unsigned long long)(case_end - content))
            return NULL;
        len = (size_t)declared;
        const char *marker = content + len;
        if (marker < case_end && *marker == '\n') marker++;
        if ((size_t)(case_end - marker) < strlen("MODEL_OUTPUT_END") ||
            memcmp(marker, "MODEL_OUTPUT_END", strlen("MODEL_OUTPUT_END")) != 0)
            return NULL;
    } else {
        const char *finish = bounded_strstr(content, case_end, "\nMODEL_OUTPUT_END");
        if (!finish) return NULL;
        len = (size_t)(finish - content);
    }

    char *out = malloc(len + 1);
    if (!out) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        return NULL;
    }
    memcpy(out, content, len);
    out[len] = '\0';
    return out;
}

int regrade_trace_file(const char *path) {
    size_t len = 0;
    char *text = read_text_file(path, &len);
    if (!text) return 2;

    const char *start = text;
    const char *end = text + len;
    int total = 0;
    int passed = 0;
    int failed = 0;
    int changed = 0;
    int unknown = 0;
    int parse_errors = 0;

    while (true) {
        const char *case_start = trace_find_next_case(start, end);
        if (!case_start) break;
        const char *after_header = memchr(case_start, '\n', (size_t)(end - case_start));
        after_header = after_header ? after_header + 1 : end;
        const char *case_end = trace_find_next_case(after_header, end);
        if (!case_end) case_end = end;
        start = case_end;
        total++;

        char source[64];
        char id[128];
        char traced_status[32];
        char traced_pick[EVAL_ANSWER_MAX];
        if (!trace_get_line_field(case_start, case_end, "source: ", source, sizeof(source)) ||
            !trace_get_line_field(case_start, case_end, "id: ", id, sizeof(id))) {
            fprintf(stderr, "ds4-eval: trace case %d is missing source/id\n", total);
            parse_errors++;
            continue;
        }
        trace_get_line_field(case_start, case_end, "status: ", traced_status, sizeof(traced_status));
        trace_get_line_field(case_start, case_end, "picked: ", traced_pick, sizeof(traced_pick));

        const eval_case *tc = find_eval_case_by_source_id(source, id);
        if (!tc) {
            fprintf(stderr, "ds4-eval: trace case %d not found in embedded cases: %s/%s\n",
                    total, source, id);
            unknown++;
            continue;
        }

        char *model_output = trace_copy_model_output(case_start, case_end);
        if (!model_output) {
            fprintf(stderr, "ds4-eval: trace case %d is missing MODEL_OUTPUT block: %s/%s\n",
                    total, source, id);
            parse_errors++;
            continue;
        }

        char got[EVAL_ANSWER_MAX];
        find_case_answer(tc, model_output, got, sizeof(got));
        bool ok = answer_matches(tc, got);
        if (ok) passed++;
        else failed++;

        bool traced_ok = !strcmp(traced_status, "PASSED");
        if ((traced_status[0] && ok != traced_ok) ||
            (traced_pick[0] && strcmp(got, traced_pick) != 0)) {
            changed++;
            printf("case %d %s/%s: trace %s picked=%s -> regrade %s picked=%s expected=%s\n",
                   total, source, id,
                   traced_status[0] ? traced_status : "?",
                   traced_pick[0] ? traced_pick : "?",
                   ok ? "PASSED" : "FAILED", got, tc->answer ? tc->answer : "?");
        }
        free(model_output);
    }

    printf("ds4-eval: regraded %d cases from %s: passed=%d failed=%d changed=%d unknown=%d parse_errors=%d\n",
           total, path, passed, failed, changed, unknown, parse_errors);
    free(text);
    return (unknown || parse_errors || total == 0) ? 1 : 0;
}
