/* Native attention-only diagnostic. Feed saved normalized inputs to each layer;
 * never treat these outputs as full-model validation or generated logits. */
#include "src/core/core_internal.h"
#include <sys/stat.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "attention replay failed at line %d\n", __LINE__); return 1; } } while (0)

static int write_tensor(const ds4_gpu_tensor *t, uint64_t off, uint64_t bytes,
                        const char *prefix, const char *kind, uint32_t layer, int append) {
    if (!bytes) return 1;
    char path[4096];
    if (snprintf(path, sizeof path, "%s.%s_L%02u.bin", prefix, kind, layer) >= (int)sizeof path) return 0;
    void *data = malloc((size_t)bytes);
    if (!data) return 0;
    int ok = ds4_gpu_flush_commands() && ds4_gpu_tensor_read(t, off, data, bytes);
    FILE *f = ok ? fopen(path, append ? "ab" : "wbx") : NULL;
    ok = f && fwrite(data, 1, (size_t)bytes, f) == bytes;
    if (f && fclose(f)) ok = 0;
    free(data);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        fprintf(stderr, "usage: %s MODEL AX_PREFIX OUT_PREFIX TOKENS CHUNK [--strict-fp]\n"
                        "Diagnostic only; TOKENS 2..128, CHUNK 1..8. Use a fresh output prefix.\n", argv[0]);
        return 2;
    }
    char *end;
    const long tokens = strtol(argv[4], &end, 10); REQUIRE(!*end && tokens >= 2 && tokens <= 128);
    const long chunk = strtol(argv[5], &end, 10); REQUIRE(!*end && chunk >= 1 && chunk <= 8);
    if (argc == 7) { REQUIRE(!strcmp(argv[6], "--strict-fp")); ds4_gpu_set_strict_fp(1); }
    ds4_engine_options opt = {0};
    opt.model_path = argv[1]; opt.backend = DS4_BACKEND_METAL; opt.vision_gguf_path = "off";
    ds4_engine *e = NULL;
    REQUIRE(ds4_engine_open(&e, &opt) == 0 && ds4_engine_is_v41(e));
    const uint32_t cap = (uint32_t)(chunk < tokens ? chunk : tokens), E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM;
    /* Validate every input before running any attention. Sidecars/Engram/MoE are
     * already represented by these frozen inputs; none executes in this tool. */
    FILE *inputs[DS4_MAX_LAYER];
    REQUIRE(DS4_N_LAYER <= DS4_MAX_LAYER);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        char path[4096]; struct stat sb;
        REQUIRE(snprintf(path, sizeof path, "%s.ax_L%02u.bin", argv[2], il) < (int)sizeof path);
        REQUIRE(!stat(path, &sb) && sb.st_size == (off_t)((uint64_t)tokens * E * 4));
        inputs[il] = fopen(path, "rb"); REQUIRE(inputs[il]);
    }
    ds4_v41_state st; REQUIRE(v41_state_alloc(&st, cap, (uint32_t)tokens, 1));
    st.no_engram = 1; st.half_prefill = 1;
    REQUIRE(ds4_gpu_v41_set_half_prefill(1));
    float *x = malloc((size_t)cap * E * 4); REQUIRE(x);
    for (uint32_t pos = 0; pos < (uint32_t)tokens; pos += cap) {
        st.n = (uint32_t)tokens - pos < cap ? (uint32_t)tokens - pos : cap;
        st.pos0 = pos; st.idx_owner = st.cand_owner = -1; st.idx_topk = st.idx_ratio = 0;
        int32_t positions[8]; for (uint32_t j = 0; j < st.n; j++) positions[j] = (int32_t)(pos + j);
        REQUIRE(ds4_gpu_tensor_write(st.pos, 0, positions, st.n * 4));
        REQUIRE(ds4_gpu_begin_commands());
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            REQUIRE(fread(x, 4, (size_t)st.n * E, inputs[il]) == (size_t)st.n * E);
            REQUIRE(ds4_gpu_tensor_write(st.xn, 0, x, (uint64_t)st.n * E * 4));
            REQUIRE(v41_attn_in(e, &st, il));
            REQUIRE(write_tensor(st.qr, 0, (uint64_t)st.n * DS4_N_LORA_Q * 4, argv[3], "qr", il, pos));
            REQUIRE(write_tensor(st.qrn, 0, (uint64_t)st.n * DS4_N_LORA_Q * 4, argv[3], "qrn", il, pos));
            REQUIRE(write_tensor(st.q, 0, (uint64_t)st.n * DS4_N_HEAD * HD * 4, argv[3], "q", il, pos));
            REQUIRE(write_tensor(st.kvn, 0, (uint64_t)st.n * HD * 4, argv[3], "wkv", il, pos));
            REQUIRE(v41_attn_cache(e, &st, il));
            REQUIRE(write_tensor(st.o, 0, (uint64_t)st.n * DS4_N_HEAD * HD * 4, argv[3], "o", il, pos));
            if (g_ds4_v41.is_kv_source[il]) {
                const uint32_t g0 = pos / ds4_layer_compress_ratio(il), ng = st.ng_src[il] - g0;
                REQUIRE(write_tensor(st.comp_kv[il], (uint64_t)g0 * DS4_V41_CKV_BYTES, (uint64_t)ng * DS4_V41_CKV_BYTES,
                                     argv[3], "packed", il, g0));
            }
            REQUIRE(v41_attn_out(e, &st, il));
            REQUIRE(write_tensor(st.attn_out, 0, (uint64_t)st.n * E * 4, argv[3], "ao", il, pos));
        }
        REQUIRE(ds4_gpu_end_commands());
        fprintf(stderr, "attention-only diagnostic: rows %u..%u complete\n", pos, pos + st.n);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) fclose(inputs[il]);
    v41_state_free(&st); ds4_engine_close(e); free(x);
    return 0;
}
