/* core_sample.c — token 文本/采样核 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
static uint32_t utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    const uint8_t c = (uint8_t)s[*pos];
    if (c < 0x80 || *pos + 1 >= len) {
        (*pos)++;
        return c;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        uint32_t cp = ((uint32_t)(c & 0x1f) << 6) | ((uint8_t)s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        uint32_t cp = ((uint32_t)(c & 0x0f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    (*pos)++;
    return c;
}

static int gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        return (int)cp;
    }

    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
            continue;
        }
        if (cp == 256 + n) return (int)b;
        n++;
    }
    return -1;
}

static bool vocab_token_is_literal_special(ds4_str s) {
    const unsigned char bar[] = {0xef, 0xbd, 0x9c}; /* U+FF5C fullwidth vertical bar. */
    if (s.len < sizeof(bar)) return false;
    for (uint64_t i = 0; i + sizeof(bar) <= s.len; i++) {
        if (!memcmp(s.ptr + i, bar, sizeof(bar))) return true;
    }
    return false;
}

char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    ds4_vocab *vocab = &e->vocab;
    if (token < 0 || token >= vocab->n_vocab) {
        if (len) *len = 0;
        char *out = xmalloc(1);
        out[0] = '\0';
        return out;
    }

    ds4_str s = vocab->token[token];
    char *out = xmalloc((size_t)s.len + 1);
    if (vocab_token_is_literal_special(s)) {
        memcpy(out, s.ptr, (size_t)s.len);
        out[s.len] = '\0';
        if (len) *len = (size_t)s.len;
        return out;
    }

    size_t n = 0;
    uint64_t pos = 0;
    while (pos < s.len) {
        uint32_t cp = utf8_decode_one(s.ptr, s.len, &pos);
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

int ds4_token_eos(ds4_engine *e) {
    return e->vocab.eos_id;
}

int ds4_token_user(ds4_engine *e) {
    return e->vocab.user_id;
}

int ds4_token_assistant(ds4_engine *e) {
    return e->vocab.assistant_id;
}

int sample_argmax(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best = (int)i;
        }
    }
    return best;
}

static DS4_MAYBE_UNUSED void logits_top2(const float *logits, uint32_t n_vocab,
                        int *top0, float *logit0,
                        int *top1, float *logit1) {
    int b0 = -1, b1 = -1;
    float v0 = DS4_NEG_INF, v1 = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > v0) {
            b1 = b0; v1 = v0;
            b0 = (int)i; v0 = v;
        } else if (v > v1) {
            b1 = (int)i; v1 = v;
        }
    }
    if (top0) *top0 = b0;
    if (logit0) *logit0 = v0;
    if (top1) *top1 = b1;
    if (logit1) *logit1 = v1;
}

static uint64_t sample_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dULL;
}

static float sample_rng_f32(uint64_t *state) {
    const uint64_t x = sample_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

typedef struct {
    int id;
    float logit;
    float prob;
} sample_candidate;

static int sample_candidate_cmp_desc(const void *a, const void *b) {
    const sample_candidate *ca = a;
    const sample_candidate *cb = b;
    return (cb->logit > ca->logit) - (cb->logit < ca->logit);
}

static int sample_full_vocab(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        float        top_p,
        float        min_p,
        uint64_t    *rng) {
    float max_logit = DS4_NEG_INF;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return sample_argmax(logits, n_vocab);

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = sample_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    sample_candidate *cand = xmalloc((size_t)finite * sizeof(cand[0]));
    uint32_t n = 0;
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        const float p = expf((v - max_logit) / temperature);
        cand[n++] = (sample_candidate){.id = (int)i, .logit = v, .prob = p};
        sum += p;
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        free(cand);
        return best;
    }

    qsort(cand, n, sizeof(cand[0]), sample_candidate_cmp_desc);
    const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = cand[i].prob / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) {
        free(cand);
        return best;
    }

    float r = sample_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= cand[i].prob;
        if (r <= 0.0f) {
            const int id = cand[i].id;
            free(cand);
            return id;
        }
    }
    const int id = cand[filtered - 1].id;
    free(cand);
    return id;
}

int sample_top_p_min_p(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        uint64_t    *rng) {
    if (temperature <= 0.0f) return sample_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0) return sample_full_vocab(logits, n_vocab, temperature, top_p, min_p, rng);
    /* 上限=栈数组容量。客户端 JSON 的 top_k 直达这里且上游零校验, 静默截断=引擎
     * 擅自改采样分布(铁律禁) —— 至少让截断可见(打一次)。 */
    #define DS4_SAMPLE_TOPK_MAX 1024
    if (top_k > DS4_SAMPLE_TOPK_MAX) {
        static bool topk_warned = false;
        if (!topk_warned) {
            topk_warned = true;
            fprintf(stderr, "ds4: top_k %d exceeds cap %d, clamped (warned once)\n",
                    top_k, DS4_SAMPLE_TOPK_MAX);
        }
        top_k = DS4_SAMPLE_TOPK_MAX;
    }
    if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;

    int ids[DS4_SAMPLE_TOPK_MAX];
    float vals[DS4_SAMPLE_TOPK_MAX];
    int n = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (!isfinite(v)) continue;
        if (n == top_k && v <= vals[n - 1]) continue;
        int j = n < top_k ? n++ : n - 1;
        while (j > 0 && vals[j - 1] < v) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j--;
        }
        vals[j] = v;
        ids[j] = (int)i;
    }
    if (n == 0) return sample_argmax(logits, n_vocab);

    float probs[1024];
    const float max_logit = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((vals[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) return ids[0];

    const float min_prob = (probs[0] / sum) * min_p;
    float filtered_sum = 0.0f;
    int filtered = 0;
    for (int i = 0; i < n; i++) {
        float p = probs[i] / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += probs[i];
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered <= 0) return ids[0];

    float r = sample_rng_f32(rng) * filtered_sum;
    for (int i = 0; i < filtered; i++) {
        r -= probs[i];
        if (r <= 0.0f) return ids[i];
    }
    return ids[filtered - 1];
}

void print_top_logits(
        FILE          * fp,
        const char    * label,
        const ds4_vocab * vocab,
        const float   * logits,
        uint32_t        n_vocab,
        int             k) {
    int best[16];
    if (k > 16) k = 16;
    for (int i = 0; i < k; i++) best[i] = -1;

    for (uint32_t i = 0; i < n_vocab; i++) {
        for (int j = 0; j < k; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int l = k - 1; l > j; l--) best[l] = best[l - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    fprintf(fp, "ds4: top logits %s:\n", label);
    for (int i = 0; i < k && best[i] >= 0; i++) {
        const int id = best[i];
        fprintf(fp, "  %2d %7d % .9g  ", i, id, logits[id]);
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%.*s", (int)vocab->token[id].len, vocab->token[id].ptr);
        }
        fputc('\n', fp);
    }
}

/* CPU generation entry point.  It runs layer-major prefill once, then decodes
 * one token at a time using the persistent KV cache and scratch arena. */
