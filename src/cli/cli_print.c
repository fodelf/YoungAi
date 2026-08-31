#include "ds4.h"
#include "ds4_distributed.h"
#include "linenoise.h"

/* ds4 CLI.
 *
 * One-shot mode builds a single DeepSeek chat prompt and exits.  Interactive
 * mode keeps a rendered token transcript plus one ds4_session, so follow-up
 * turns reuse the live Metal KV checkpoint just like the server does.  The CLI
 * deliberately keeps policy here and leaves graph/cache mechanics inside the
 * engine API. */

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include "cli_internal.h"

void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4 [(-p PROMPT | --prompt-file FILE)] [options]\n"
        "\n"
        "Invocation modes:\n"
        "  ds4\n"
        "      Start the interactive chat prompt with a session backend: ds4>\n"
        "  ds4 -p TEXT\n"
        "      Run one prompt and exit.\n"
        "  ds4 --prompt-file FILE\n"
        "      Run one prompt read from FILE and exit. Useful for long prompts.\n"
        "\n"
        "Model and runtime:\n"
        "  -m, --model FILE\n"
        "      GGUF model path. Default: ds4flash.gguf\n"
        "  -c, --ctx N\n"
        "      Context size allocated for the session. Default: " DS4_STRINGIFY(DS4_DEFAULT_CTX_SIZE) "\n"
        "  --metal\n"
        "      Use the Metal graph backend. This is the normal fast path on macOS.\n"
        "  --cuda\n"
        "      Use the CUDA graph backend. This is the normal fast path on CUDA builds.\n"
        "  --cpu\n"
        "      Use the CPU reference/debug backend. Not recommended for normal inference.\n"
        "  --backend NAME\n"
        "      Select backend explicitly: metal, cuda, or cpu.\n"
        "  -t, --threads N\n"
        "      CPU helper threads for host-side or reference work.\n"
        "  --quality\n"
        "      Prefer exact kernels where faster approximate paths exist; MTP uses strict verification.\n"
        "  --dir-steering-file FILE\n"
        "      Load one f32 direction vector per layer for directional steering.\n"
        "  --dir-steering-ffn F\n"
        "      Apply steering after FFN outputs: y -= F*v*dot(v,y). Default with file: 1\n"
        "  --dir-steering-attn F\n"
        "      Apply steering after attention outputs. Default: 0\n"
        "  --warm-weights\n"
        "      Touch mapped tensor pages before generation. Slower startup, fewer first-use stalls.\n"
        "  --power N\n"
        "      Target GPU duty cycle percentage, 1..100. Default: 100\n"
        "  --no-residency\n"
        "      Skip MTLResidencySet wiring and view warmup; survival flag for\n"
        "      oversized single-host models. Metal only.\n"
        "  --strict-fp\n"
        "      Strict IEEE-754 shader math (safe math + f32 raw KV + exp2/log2 RoPE)\n"
        "      for cross-GPU parity lanes. Metal only.\n"
        "  --expert-pool-mb N\n"
        "      Resident routed-expert LRU pool budget in MiB (0 = off). Metal only.\n"
        "  --expert-pool-pinned SPEC\n"
        "      Pool pin whitelist, e.g. \"L20:1,2;L21:7\". Metal only.\n"
        "  --expert-pool-auto-pin-top N\n"
        "      Auto-pin the measured top-N experts per served layer. Metal only.\n"
        "  --expert-pool-prefetch-top N\n"
        "      Predictor prefetch margin for the pool (0 = off). Metal only.\n"
        "  --expert-pin-file FILE\n"
        "      Frequency hot-expert pin list; mlocks their mmap ranges. Metal only.\n"
        "  --expert-pin-mlock-mb N\n"
        "      Wired budget for --expert-pin-file ranges (0 = pins off). Metal only.\n"
        "  --resid-pin-mlock-mb N\n"
        "      Wired budget for the residual sidecar's pin ranges (0 = off). Metal only.\n"
        "  --mem-budget-mb N\n"
        "      Arm the memory guardrails: watchdog aborts at 90%% of N, the L1 load gate\n"
        "      refuses startup over 85%%, and the expert resident/stream AUTO verdict\n"
        "      compares against it. Unset = guardrails disarmed.\n"
        "  --prefill-chunk N\n"
        "      Prefill batch chunk cap in tokens (0 = whole prompt as one batch).\n"
        "      Default: backend-specific (Metal min(prompt,4096), CUDA 256).\n"
        "  --spec\n"
        "      DSpark speculative decoding + online speculate-vs-flat scheduler.\n"
        "      Greedy verify is position-exact: output tokens match flat decode.\n"
        "  --draft-gguf FILE\n"
        "      Standalone DSpark drafter GGUF (mtp.* tensors only).\n"
        "  --draft-zchain FILE\n"
        "      Drafter amplifier sidecar (3-layer chain merged at slots 43..45).\n"
        "  --vq-dir DIR\n"
        "      VQ codebook sidecar directory (takes precedence over --residual).\n"
        "  --residual FILE\n"
        "      1-bit residual expert sidecar GGUF.\n"
        "  --zchain FILE\n"
        "      External amplifier chain sidecar (else embedded blk.L.opt_* auto-load).\n"
        "  --cap-dir DIR | --cap-layers LO-HI | --eval-ids FILE | --eval-logits FILE |\n"
        "  --eval-hdump DIR | --eval-no-bos | --amp-anchor FILE [--amp-anchor-route] |\n"
        "  --multi-bench N\n"
        "      反修数据面 instruments: per-layer capture, teacher-forced scoring,\n"
        "      anchored replay, and the multi-session batching bench.\n"
    );
    ds4_dist_usage(fp);
    fprintf(fp,
        "\n"
        "Prompt and generation:\n"
        "  -p, --prompt TEXT\n"
        "      Prompt to generate from.\n"
        "  --prompt-file FILE\n"
        "      Read the prompt text from FILE.\n"
        "  -sys, --system TEXT\n"
        "      System prompt. Empty string disables the default. Default: You are a helpful assistant\n"
        "  -n, --tokens N\n"
        "      Maximum tokens to generate. Default: 50000\n"
        "  --temp F\n"
        "      Sampling temperature. 0 is greedy/deterministic. Default: 1\n"
        "  --top-p F\n"
        "      Nucleus sampling probability. Default: 1\n"
        "  --min-p F\n"
        "      Keep tokens scoring at least F times the top token. Default: 0.05\n"
        "  --seed N\n"
        "      Sampling seed for reproducible non-greedy runs. Default: time-based\n"
        "  --think\n"
        "      Use normal thinking mode. This is the default.\n"
        "  --think-max\n"
        "      Use Think Max when --ctx is at least 393216 tokens; otherwise normal thinking.\n"
        "  --nothink\n"
        "      Start assistant turns with </think> for direct non-thinking replies.\n"
        "\n"
        "Interactive commands:\n"
        "  /help\n"
        "      Show interactive commands.\n"
        "  /think, /think-max, /nothink\n"
        "      Select normal thinking, context-gated Think Max, or non-thinking mode.\n"
        "  /ctx N\n"
        "      Recreate the interactive session with a new context size.\n"
        "  /power N\n"
        "      Set GPU duty cycle percentage, 1..100.\n"
        "  /read FILE\n"
        "      Read a prompt from FILE and run it as the next user message.\n"
        "  /quit, /exit\n"
        "      Leave the interactive prompt.\n"
        "  Ctrl+C\n"
        "      Stop the current generation and return to ds4> without exiting.\n"
        "\n"
        "Diagnostics:\n"
        "  --inspect\n"
        "      Load the model and print a summary only.\n"
        "  --dump-tokens\n"
        "      Tokenize -p/--prompt-file exactly as written, then exit without inference.\n"
        "  --dump-logits FILE\n"
        "      Write full next-token logits as JSON after prompt prefill, then exit.\n"
        "  --dump-logprobs FILE\n"
        "      Write greedy continuation top-logprobs as JSON without printing text.\n"
        "  --logprobs-top-k N\n"
        "      Number of local alternatives stored by --dump-logprobs. Default: 20\n"
        "  --perplexity-file FILE\n"
        "      Score raw text with teacher-forced next-token negative log likelihood.\n"
        "  --imatrix-dataset FILE\n"
        "      Rendered DS4 prompt dataset produced by misc/imatrix_dataset.\n"
        "  --imatrix-out FILE\n"
        "      Collect a routed-MoE activation imatrix and write llama-compatible .dat.\n"
        "  --imatrix-max-prompts N\n"
        "      Stop imatrix collection after N prompts. Default: no prompt limit\n"
        "  --imatrix-max-tokens N\n"
        "      Stop imatrix collection after N prompt tokens. Default: no token limit\n"
        "  --head-test\n"
        "      Run the output HC/logits head after the native slice.\n"
        "  --first-token-test\n"
        "      Run an exact CPU whole-model pass for the first prompt token.\n"
        "  --metal-graph-test\n"
        "      Compare first GPU-resident graph stages with CPU.\n"
        "  --metal-graph-full-test\n"
        "      Run the GPU-resident self-token graph across all layers.\n"
        "  --metal-graph-prompt-test\n"
        "      Compare CPU and GPU graph logits for the full prompt.\n"
        "\n"
        "Normal CLI commands:\n"
        "  ./ds4\n"
        "  ./ds4 -p \"Scrivi una storia su una papera scansafatiche\"\n"
        "  ./ds4 --think-max --prompt-file prompt.txt --ctx 393216\n"
        "\n"
        "Notes:\n"
        "  The CLI keeps KV cache state across interactive turns on session backends.\n"
        "  CPU mode supports interactive chat too, but it is a slow reference/debug path.\n"
        "  Long added input is processed with batched prefill; short continuations use decode.\n"
        "  Startup prints the extra context-buffer memory for the selected context size.\n"
        "\n"
        "  -h, --help\n"
        "      Show this help.\n");
}

void log_context_memory(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = ds4_context_memory_estimate(backend, ctx_size);
    fprintf(stderr,
            "ds4: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

ds4_think_mode cli_effective_think_mode(const cli_generation_options *gen) {
    return ds4_think_mode_for_context(gen->think_mode, gen->ctx_size);
}

bool cli_think_max_downgraded(const cli_generation_options *gen) {
    return gen->think_mode == DS4_THINK_MAX &&
           cli_effective_think_mode(gen) != DS4_THINK_MAX;
}

void cli_warn_think_max_downgraded(const cli_generation_options *gen, const char *name) {
    if (!cli_think_max_downgraded(gen)) return;
    ds4_log(stderr,
        DS4_LOG_WARNING,
        "ds4: warning: %s needs --ctx >= %u; ctx=%d uses normal thinking instead\n",
        name,
        ds4_think_max_min_context(),
        gen->ctx_size);
}

double cli_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

void cli_prefill_progress_cb(void *ud, const char *event, int current, int total) {
    (void)total;
    cli_prefill_progress *p = ud;
    if (!p || !event || p->input_tokens <= 0) return;
    const bool is_display = strcmp(event, "prefill_display") == 0;
    if (strcmp(event, "prefill_chunk") && !is_display) return;
    if (is_display && !p->use_color) return;

    int processed = current - p->base_tokens;
    if (processed < 0) processed = 0;
    if (processed > p->input_tokens) processed = p->input_tokens;
    double pct = 100.0 * (double)processed / (double)p->input_tokens;
    if (pct > 100.0) pct = 100.0;

    const bool complete = processed >= p->input_tokens;
    if (complete && p->finished) return;

    if (p->use_color) {
        fputc('\r', stderr);
        ds4_log(stderr,
                DS4_LOG_PREFILL,
                "processing %d input tokens: %d/%d (%.1f%%)",
                p->input_tokens,
                processed,
                p->input_tokens,
                pct);
        fputs("\x1b[K", stderr);
        if (complete) fputc('\n', stderr);
    } else {
        fprintf(stderr,
                "processing %d input tokens: %d/%d (%.1f%%)\n",
                p->input_tokens,
                processed,
                p->input_tokens,
                pct);
    }
    if (complete) p->finished = true;
    fflush(stderr);
}

static bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static void token_printer_set_grey(token_printer *p) {
    if (p->use_color && !p->color_open) {
        fputs("\x1b[90m", p->fp);
        p->color_open = true;
    }
}

static void token_printer_reset_color(token_printer *p) {
    if (p->use_color && p->color_open) {
        fputs("\x1b[0m", p->fp);
        p->color_open = false;
    }
}

static void token_printer_write_char(token_printer *p, char c) {
    if (p->in_think) token_printer_set_grey(p);
    fputc((unsigned char)c, p->fp);
    p->last_output_newline = c == '\n';
}

void token_printer_process(token_printer *p, const char *text, size_t len, bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = p->pending_len + len;
    char *buf = malloc(total ? total : 1);
    if (!buf) return;
    if (p->pending_len) memcpy(buf, p->pending, p->pending_len);
    if (len) memcpy(buf + p->pending_len, text, len);
    p->pending_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = buf + i;
        const size_t rem = total - i;
        if (bytes_has_prefix(cur, rem, think_open)) {
            p->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (bytes_has_prefix(cur, rem, think_close)) {
            p->in_think = false;
            token_printer_reset_color(p);
            if (!p->last_output_newline) {
                fputc('\n', p->fp);
                p->last_output_newline = true;
            }
            i += strlen(think_close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(p->pending)) {
                memcpy(p->pending, cur, rem);
                p->pending_len = rem;
            }
            break;
        }
        token_printer_write_char(p, cur[0]);
        i++;
    }

    free(buf);
}

void token_printer_finish(token_printer *p) {
    if (p->format_thinking) {
        token_printer_process(p, NULL, 0, true);
        token_printer_reset_color(p);
    }
    fflush(p->fp);
}

void generation_done(void *ud) {
    token_printer *p = ud;
    token_printer_finish(p);
    if (!p->last_output_newline) {
        fputc('\n', p->fp);
        p->last_output_newline = true;
    }
    fflush(p->fp);
}

void token_printer_write_text(token_printer *p, const char *text, size_t len) {
    if (p->format_thinking) {
        token_printer_process(p, text, len, false);
    } else if (len) {
        fwrite(text, 1, len, p->fp);
        p->last_output_newline = text[len - 1] == '\n';
    }
}

static bool json_utf8_valid(const char *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned char c = (unsigned char)s[i++];
        if (c < 0x80) continue;
        int need = 0;
        if (c >= 0xc2 && c <= 0xdf) need = 1;
        else if (c >= 0xe0 && c <= 0xef) need = 2;
        else if (c >= 0xf0 && c <= 0xf4) need = 3;
        else return false;
        if (i + (size_t)need > n) return false;
        unsigned char c1 = (unsigned char)s[i];
        if (c == 0xe0 && c1 < 0xa0) return false;
        if (c == 0xed && c1 >= 0xa0) return false;
        if (c == 0xf0 && c1 < 0x90) return false;
        if (c == 0xf4 && c1 >= 0x90) return false;
        for (int j = 0; j < need; j++) {
            unsigned char cc = (unsigned char)s[i + (size_t)j];
            if ((cc & 0xc0) != 0x80) return false;
        }
        i += (size_t)need;
    }
    return true;
}

void json_write_string(FILE *fp, const char *s, size_t n) {
    bool valid_utf8 = json_utf8_valid(s, n);
    fputc('"', fp);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc((char)c, fp);
        } else if (c == '\n') {
            fputs("\\n", fp);
        } else if (c == '\r') {
            fputs("\\r", fp);
        } else if (c == '\t') {
            fputs("\\t", fp);
        } else if (c < 0x20) {
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else if (!valid_utf8 && c >= 0x80) {
            /* Tokenizer pieces can be arbitrary byte fragments.  The bytes
             * array is authoritative; this escape keeps the JSON valid. */
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else {
            fputc((char)c, fp);
        }
    }
    fputc('"', fp);
}

void json_write_token(FILE *fp, ds4_engine *engine, int token) {
    size_t n = 0;
    char *text = ds4_token_text(engine, token, &n);
    fprintf(fp, "{\"id\":%d,\"text\":", token);
    json_write_string(fp, text, n);
    fputs(",\"bytes\":[", fp);
    for (size_t i = 0; i < n; i++) {
        if (i) fputc(',', fp);
        fprintf(fp, "%u", (unsigned)(unsigned char)text[i]);
    }
    fputc(']', fp);
    fputc('}', fp);
    free(text);
}
