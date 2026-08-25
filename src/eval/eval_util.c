/* ds4-eval shared small helpers: byte/style buffers, wall clocks, and
 * eval_case classification (moved verbatim from ds4_eval.c). */

#include "eval_internal.h"

void buf_append(byte_buf *b, const char *p, size_t n) {
    if (n == 0) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 1024;
        while (cap < b->len + n + 1) cap *= 2;
        char *v = realloc(b->v, cap);
        if (!v) {
            fprintf(stderr, "ds4-eval: out of memory\n");
            exit(1);
        }
        b->v = v;
        b->cap = cap;
    }
    memcpy(b->v + b->len, p, n);
    b->len += n;
    b->v[b->len] = '\0';
}

void buf_appendf(byte_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list copy;
    va_copy(copy, ap);
    int n = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);
    if (n < 0) {
        va_end(ap);
        return;
    }
    char *tmp = malloc((size_t)n + 1);
    if (!tmp) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        exit(1);
    }
    vsnprintf(tmp, (size_t)n + 1, fmt, ap);
    va_end(ap);
    buf_append(b, tmp, (size_t)n);
    free(tmp);
}

int eval_case_nchoices(const eval_case *tc) {
    int n = 0;
    while (n < EVAL_MAX_CHOICES && tc->choice[n]) n++;
    return n;
}

bool eval_case_is_multiple_choice(const eval_case *tc) {
    return eval_case_nchoices(tc) > 0;
}

bool eval_case_is_compsec(const eval_case *tc) {
    return tc->source && !strcmp(tc->source, "COMPSEC");
}

void style_append(style_buf *b, unsigned char style, size_t n) {
    if (n == 0) return;
    if (b->len + n > b->cap) {
        size_t cap = b->cap ? b->cap : 1024;
        while (cap < b->len + n) cap *= 2;
        unsigned char *v = realloc(b->v, cap);
        if (!v) {
            fprintf(stderr, "ds4-eval: out of memory\n");
            exit(1);
        }
        b->v = v;
        b->cap = cap;
    }
    memset(b->v + b->len, style, n);
    b->len += n;
}

void buf_free(byte_buf *b) {
    free(b->v);
    memset(b, 0, sizeof(*b));
}

void style_free(style_buf *b) {
    free(b->v);
    memset(b, 0, sizeof(*b));
}

double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

double run_clock_sec(void) {
    struct timespec ts;
#ifdef CLOCK_UPTIME_RAW
    /* On Darwin this clock excludes system sleep, which is exactly what the TUI
     * elapsed counter wants: benchmark runtime, not lid-closed wall time. */
    clock_gettime(CLOCK_UPTIME_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}
