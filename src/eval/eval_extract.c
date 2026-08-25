/* ds4-eval answer extraction and normalization, plus the extractor
 * self-tests behind --self-test-extractors (moved verbatim from ds4_eval.c). */

#include "eval_internal.h"

/* Model outputs can contain provisional "answer" text after a forced
 * </think> and then a later final line.  A strict final Answer: marker is the
 * best grading target; outputs without one keep the original loose fallback. */
static char *strcasestr_local(const char *hay, const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0) return (char *)hay;
    for (; *hay; hay++) {
        if (tolower((unsigned char)*hay) == tolower((unsigned char)needle[0]) &&
            strncasecmp(hay, needle, nlen) == 0)
            return (char *)hay;
    }
    return NULL;
}

static bool is_letter_boundary(char before, char after) {
    return !isalpha((unsigned char)before) && !isalpha((unsigned char)after);
}

static char *find_last_answer_marker(const char *visible) {
    char *last = NULL;
    const size_t nlen = strlen("answer");

    for (const char *p = visible; *p; p++) {
        if (tolower((unsigned char)*p) != 'a' || strncasecmp(p, "answer", nlen) != 0)
            continue;
        char before = p == visible ? ' ' : p[-1];
        char after = p[nlen];
        if (!is_letter_boundary(before, after)) continue;

        const char *q = p + nlen;
        while (*q && isspace((unsigned char)*q)) q++;
        if (*q == ':') last = (char *)p;
    }
    return last ? last : strcasestr_local(visible, "answer");
}

static char find_answer_letter(const char *generated, int nchoices) {
    if (nchoices <= 0) return '?';
    const char *visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;
    char max_answer = (char)('A' + nchoices - 1);

    char *answer = find_last_answer_marker(visible);
    if (answer) {
        const char *end = answer + strlen(answer);
        if (strlen(answer) > 96) end = answer + 96;
        for (const char *p = answer; p < end && *p; p++) {
            char c = (char)toupper((unsigned char)*p);
            if (c >= 'A' && c <= max_answer) {
                char before = p == visible ? ' ' : p[-1];
                char after = p[1];
                if (is_letter_boundary(before, after)) return c;
            }
        }
    }

    for (const char *p = visible + strlen(visible); p > visible; p--) {
        char c = (char)toupper((unsigned char)p[-1]);
        if (c >= 'A' && c <= max_answer) {
            char before = p - 1 == visible ? ' ' : p[-2];
            char after = p[0];
            if (is_letter_boundary(before, after)) return c;
        }
    }
    return '?';
}

static void normalize_integer_answer(const char *p, size_t len,
                                     char *dst, size_t dstlen) {
    while (len > 1 && *p == '0') {
        p++;
        len--;
    }
    if (dstlen == 0) return;
    size_t n = len < dstlen - 1 ? len : dstlen - 1;
    memcpy(dst, p, n);
    dst[n] = '\0';
}

static bool scan_first_integer(const char *start, const char *end,
                               char *dst, size_t dstlen) {
    const char *p = start;
    while (p < end && *p) {
        if (isdigit((unsigned char)*p)) {
            const char *q = p + 1;
            while (q < end && isdigit((unsigned char)*q)) q++;
            normalize_integer_answer(p, (size_t)(q - p), dst, dstlen);
            return true;
        }
        p++;
    }
    return false;
}

static void find_integer_answer(const char *generated, char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    snprintf(dst, dstlen, "?");
    const char *visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;

    char *answer = find_last_answer_marker(visible);
    if (answer) {
        const char *end = answer + strlen(answer);
        if (strlen(answer) > 160) end = answer + 160;
        if (scan_first_integer(answer, end, dst, dstlen)) return;
    }

    const char *last_start = NULL;
    const char *last_end = NULL;
    for (const char *p = visible; *p; p++) {
        if (isdigit((unsigned char)*p)) {
            const char *q = p + 1;
            while (isdigit((unsigned char)*q)) q++;
            last_start = p;
            last_end = q;
            p = q - 1;
        }
    }
    if (last_start && last_end) {
        normalize_integer_answer(last_start, (size_t)(last_end - last_start),
                                 dst, dstlen);
    }
}

static void normalize_compsec_line_spec(const char *p, const char *end,
                                        char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    size_t n = 0;

    /* The model is instructed to finish with:
     *
     *     Answer: <line number or comma-separated line numbers>
     *
     * We still accept harmless surface variants in that final line: "line 9",
     * "9 and 15", "9, 15", or "20-22".  The hidden key decides what set of
     * exact line numbers is acceptable; this routine only normalizes the model
     * text into the same compact line-spec syntax.
     */
    for (; p < end && *p; p++) {
        if (!isdigit((unsigned char)*p)) continue;

        if (n > 0 && n + 1 < dstlen) dst[n++] = ',';
        while (p < end && isdigit((unsigned char)*p)) {
            if (n + 1 < dstlen) dst[n++] = *p;
            p++;
        }
        while (p < end && isspace((unsigned char)*p)) p++;

        if (p < end && *p == '-') {
            if (n + 1 < dstlen) dst[n++] = '-';
            p++;
            while (p < end && isspace((unsigned char)*p)) p++;
            while (p < end && isdigit((unsigned char)*p)) {
                if (n + 1 < dstlen) dst[n++] = *p;
                p++;
            }
        }
        if (p >= end || !*p) break;
    }
    while (n > 0 && (dst[n - 1] == ',' || dst[n - 1] == '-')) n--;
    dst[n] = '\0';
    if (n == 0) snprintf(dst, dstlen, "?");
}

static void find_compsec_answer(const char *generated, char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    snprintf(dst, dstlen, "?");
    const char *visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;

    char *answer = find_last_answer_marker(visible);
    if (answer) {
        const char *end = answer + strlen(answer);
        if (strlen(answer) > 160) end = answer + 160;
        const char *newline = memchr(answer, '\n', (size_t)(end - answer));
        if (newline) end = newline;
        normalize_compsec_line_spec(answer, end, dst, dstlen);
        if (strcmp(dst, "?") != 0) return;
    }
    find_integer_answer(generated, dst, dstlen);
}

static bool parse_line_spec(const char *spec, bool *set, size_t setlen) {
    bool any = false;
    const char *p = spec;
    while (p && *p) {
        while (*p && !isdigit((unsigned char)*p)) p++;
        if (!*p) break;
        char *end = NULL;
        long a = strtol(p, &end, 10);
        long b = a;
        p = end;
        if (*p == '-') {
            p++;
            b = strtol(p, &end, 10);
            p = end;
        }
        if (a > b) {
            long tmp = a;
            a = b;
            b = tmp;
        }
        if (a < 0) a = 0;
        if (b >= (long)setlen) b = (long)setlen - 1;
        for (long i = a; i <= b; i++) {
            set[i] = true;
            any = true;
        }
    }
    return any;
}

static bool compsec_answer_matches(const char *expected_spec, const char *got_spec) {
    bool expected[256] = {0};
    bool got[256] = {0};
    if (!parse_line_spec(expected_spec, expected, sizeof(expected))) return false;
    if (!parse_line_spec(got_spec, got, sizeof(got))) return false;

    bool hit = false;
    for (size_t i = 0; i < sizeof(got); i++) {
        if (!got[i]) continue;
        if (!expected[i]) return false;
        hit = true;
    }
    /* The prompt asks the model for the single best line, or the smallest exact
     * set when the bug cannot be localized to one line.  The hidden expected
     * answer may be a small audited range when adjacent lines are equivalent
     * locations for the same bug.  Any model-supplied line must be inside that
     * accepted set, and at least one accepted line must be present. */
    return hit;
}

void find_case_answer(const eval_case *tc, const char *generated,
                      char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    if (eval_case_is_multiple_choice(tc)) {
        dst[0] = find_answer_letter(generated, eval_case_nchoices(tc));
        if (dstlen > 1) dst[1] = '\0';
    } else if (eval_case_is_compsec(tc)) {
        find_compsec_answer(generated, dst, dstlen);
    } else {
        find_integer_answer(generated, dst, dstlen);
    }
}

bool answer_matches(const eval_case *tc, const char *got) {
    if (eval_case_is_multiple_choice(tc)) {
        return got && got[0] && tc->answer && got[0] == tc->answer[0];
    }
    if (eval_case_is_compsec(tc)) {
        return got && tc->answer && compsec_answer_matches(tc->answer, got);
    }
    char expected[EVAL_ANSWER_MAX];
    normalize_integer_answer(tc->answer, strlen(tc->answer), expected, sizeof(expected));
    return got && strcmp(got, expected) == 0;
}

static int extractor_self_test_case(const char *name, const eval_case *tc,
                                    const char *generated,
                                    const char *expected_extract) {
    char got[EVAL_ANSWER_MAX];
    find_case_answer(tc, generated, got, sizeof(got));
    if (strcmp(got, expected_extract) == 0 && answer_matches(tc, got)) return 0;

    fprintf(stderr,
            "ds4-eval: extractor self-test failed: %s (got %s, expected %s, key %s)\n",
            name, got, expected_extract, tc->answer ? tc->answer : "?");
    return 1;
}

static int trace_copy_self_test_case(void) {
    const char *prompt_output =
        "Prompt text may mention\n"
        "MODEL_OUTPUT_BEGIN without being the model block.\n";
    const char *model_output =
        "</think>Trace payload may mention\n"
        "MODEL_OUTPUT_END before the real marker.\n"
        "===== CASE not a real case =====\n"
        "Answer: F";
    char trace_case[1024];
    int n = snprintf(trace_case, sizeof(trace_case),
                     "===== CASE 1/2 SuperGPQA/trace-copy-self-test =====\n"
                     "source: SuperGPQA\n"
                     "id: trace-copy-self-test\n"
                     "QUESTION_PROMPT_BEGIN bytes=%zu\n"
                     "%s\n"
                     "QUESTION_PROMPT_END\n"
                     "MODEL_OUTPUT_BEGIN bytes=%zu\n"
                     "%s\n"
                     "MODEL_OUTPUT_END\n"
                     "\n"
                     "===== CASE 2/2 SuperGPQA/trace-copy-self-test-2 =====\n",
                     strlen(prompt_output), prompt_output,
                     strlen(model_output), model_output);
    if (n < 0 || (size_t)n >= sizeof(trace_case)) {
        fprintf(stderr, "ds4-eval: trace self-test setup failed\n");
        return 1;
    }

    const char *trace_start = trace_case;
    const char *trace_end = trace_case + strlen(trace_case);
    const char *first = trace_find_next_case(trace_start, trace_end);
    const char *after_header = first ? memchr(first, '\n', (size_t)(trace_end - first)) : NULL;
    after_header = after_header ? after_header + 1 : trace_end;
    const char *second = first ? trace_find_next_case(after_header, trace_end) : NULL;
    if (!first || !second || !strstr(second, "===== CASE 2/2")) {
        fprintf(stderr, "ds4-eval: trace self-test failed: embedded case marker skip\n");
        return 1;
    }

    char *copied = trace_copy_model_output(first, second);
    if (!copied || strcmp(copied, model_output) != 0) {
        fprintf(stderr, "ds4-eval: trace self-test failed: MODEL_OUTPUT byte copy\n");
        free(copied);
        return 1;
    }
    free(copied);
    return 0;
}

int run_extractor_self_tests(void) {
    int failed = 0;

    failed += trace_copy_self_test_case();

    const eval_case mc = {
        .source = "SuperGPQA",
        .choice[0] = "A",
        .choice[1] = "B",
        .choice[2] = "C",
        .choice[3] = "D",
        .choice[4] = "E",
        .choice[5] = "F",
        .choice[6] = "G",
        .choice[7] = "H",
        .choice[8] = "I",
        .choice[9] = "J",
        .answer = "F",
    };
    failed += extractor_self_test_case(
        "multiple-choice prefers final answer marker",
        &mc,
        "</think>So answer is 0.716 H+/O2. That corresponds to option F.\n"
        "Thus final answer: F.</think>The visible explanation repeats the calculation.\n"
        "Answer: F",
        "F");
    failed += extractor_self_test_case(
        "multiple-choice prefers Answer-colon over later prose",
        &mc,
        "</think>Answer: F\nThis answer is final; option H is a tempting distractor.",
        "F");
    failed += extractor_self_test_case(
        "multiple-choice preserves loose-answer fallback",
        &mc,
        "</think>The answer is F. This answer is final; option H is tempting.",
        "F");

    const eval_case integer = {
        .source = "AIME2025",
        .answer = "82",
    };
    failed += extractor_self_test_case(
        "integer prefers final answer marker",
        &integer,
        "</think>I first thought the answer was 80.\nFinal answer: 082",
        "82");
    failed += extractor_self_test_case(
        "integer preserves loose-answer fallback",
        &integer,
        "</think>The answer is 082. This answer comes from AIME 2025.",
        "82");

    const eval_case compsec = {
        .source = "COMPSEC",
        .answer = "9-10",
    };
    failed += extractor_self_test_case(
        "COMPSEC prefers final answer marker",
        &compsec,
        "</think>I think the answer should be line 10, because CWE-122 may apply.\n"
        "**Answer:** 10</think>The primary write is at line 10.\n"
        "Answer: 10",
        "10");

    if (failed) return 1;
    printf("ds4-eval: answer extractor self-tests passed\n");
    return 0;
}
