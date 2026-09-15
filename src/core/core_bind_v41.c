/* core_bind_v41.c — V4.1 张量绑定与形状/类型校验(2026-09-12 战役 P1)。
 *
 * 与 V4 的 weights_bind/weights_validate_layout 分开写, 因为三处不同:
 *  ① 骨架是 fp4x32(type 43), 不是 Q4_K/Q8_0; routed 专家只以 blk.L.ffn_exps_vq.blob 在场,
 *     gate/up/down 张量一律缺席(down 用影子张量补维度, 与 V4 合一 VQ 文件同法);
 *  ② 压缩器只在 kv 源层(无 ape, ratio 1 无 gate); indexer 的 q_b/proj 在索引源层, wk/k_norm 在 kv 源层;
 *  ③ 没有 output_hc_*(头直接用末层 ffn_pre 做 hc_pre); 多 engram 三件与哈希常量张量。
 * 名字来自转换器 v41_to_gguf.c, 这里是它的消费端 —— 改名两边同改。 */
#include "core_internal.h"

static ds4_tensor *need(const ds4_model *m, const char *fmt, uint32_t il) { return required_tensorf(m, fmt, il); }
/* 按整名取(三塔的专家名字里有两个数字, 那几个头则没有层号可带) */
static ds4_tensor *required_tensor_name(const ds4_model *m, const char *name) {
    ds4_tensor *t = model_find_tensor(m, name);
    if (!t) { fprintf(stderr, "ds4: 缺张量 %s\n", name); exit(1); }
    return t;
}

static void expect(const ds4_tensor *t, uint32_t type, uint32_t ndim, uint64_t d0, uint64_t d1) {
    if (!t) return;
    if (t->type != type) {
        fprintf(stderr, "ds4: V4.1 tensor %.*s has type %s, expected %s\n", (int)t->name.len, t->name.ptr,
                tensor_type_name(t->type), tensor_type_name(type));
        exit(1);
    }
    const uint64_t want[2] = {d0, d1};
    if (t->ndim != ndim) { fprintf(stderr, "ds4: V4.1 tensor %.*s has %u dims, expected %u\n", (int)t->name.len, t->name.ptr, t->ndim, ndim); exit(1); }
    for (uint32_t i = 0; i < ndim; i++) if (t->dim[i] != want[i]) {
        fprintf(stderr, "ds4: V4.1 tensor %.*s dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n", (int)t->name.len, t->name.ptr, i, t->dim[i], want[i]);
        exit(1);
    }
}

void weights_bind_v41(ds4_weights *w, const ds4_model *m) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    if (!v->active) ds4_die("weights_bind_v41 called without V4.1 metadata");
    w->token_embd  = model_find_tensor(m, "token_embd.weight");
    w->output_norm = model_find_tensor(m, "output_norm.weight");
    w->output      = model_find_tensor(m, "output.weight");
    w->engram_token_map   = model_find_tensor(m, "engram.token_map");
    w->engram_multipliers = model_find_tensor(m, "engram.multipliers");
    w->engram_primes      = model_find_tensor(m, "engram.primes");
    w->engram_offsets     = model_find_tensor(m, "engram.offsets");
    if (v->n_engram && !(w->engram_token_map && w->engram_multipliers && w->engram_primes && w->engram_offsets))
        ds4_die("V4.1 engram hash constant tensors missing (engram.token_map/multipliers/primes/offsets)");

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_layer_weights *l = &w->layer[il];
        if (!tensor_by_namef(m, "blk.%u.attn_norm.weight", il)) continue;   /* 分片切层: 缺层跳过 */
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        l->hc_attn_fn      = need(m, "blk.%u.hc_attn_fn.weight", il);
        l->hc_attn_scale   = need(m, "blk.%u.hc_attn_scale.weight", il);
        l->hc_attn_base    = need(m, "blk.%u.hc_attn_base.weight", il);
        l->attn_norm       = need(m, "blk.%u.attn_norm.weight", il);
        l->attn_q_a        = need(m, "blk.%u.attn_q_a.weight", il);
        l->attn_q_a_norm   = need(m, "blk.%u.attn_q_a_norm.weight", il);
        l->attn_q_b        = need(m, "blk.%u.attn_q_b.weight", il);
        l->attn_kv         = need(m, "blk.%u.attn_kv.weight", il);
        l->attn_kv_a_norm  = need(m, "blk.%u.attn_kv_a_norm.weight", il);
        l->attn_sinks      = need(m, "blk.%u.attn_sinks.weight", il);
        l->attn_output_a   = need(m, "blk.%u.attn_output_a.weight", il);
        l->attn_output_b   = need(m, "blk.%u.attn_output_b.weight", il);
        if (v->is_kv_source[il]) {
            l->attn_compressor_kv   = need(m, "blk.%u.attn_compressor_kv.weight", il);
            l->attn_compressor_norm = need(m, "blk.%u.attn_compressor_norm.weight", il);
            if (ratio > 1) l->attn_compressor_gate = need(m, "blk.%u.attn_compressor_gate.weight", il);
            l->indexer_wk     = need(m, "blk.%u.indexer.wk.weight", il);
            l->indexer_k_norm = need(m, "blk.%u.indexer.k_norm.weight", il);
        }
        if (v->is_index_source[il]) {
            l->indexer_attn_q_b = need(m, "blk.%u.indexer.attn_q_b.weight", il);
            l->indexer_proj     = need(m, "blk.%u.indexer.proj.weight", il);
        }
        l->hc_ffn_fn       = need(m, "blk.%u.hc_ffn_fn.weight", il);
        l->hc_ffn_scale    = need(m, "blk.%u.hc_ffn_scale.weight", il);
        l->hc_ffn_base     = need(m, "blk.%u.hc_ffn_base.weight", il);
        l->ffn_norm        = need(m, "blk.%u.ffn_norm.weight", il);
        l->ffn_gate_inp    = need(m, "blk.%u.ffn_gate_inp.weight", il);
        l->ffn_exp_probs_b = need(m, "blk.%u.exp_probs_b.bias", il);
        if (!tensor_by_namef(m, "blk.%u.ffn_exps_vq.blob", il)) { fprintf(stderr, "ds4: V4.1 layer %u lacks ffn_exps_vq.blob\n", il); exit(1); }
        l->ffn_gate_exps = NULL; l->ffn_up_exps = NULL;
        l->ffn_down_exps = routed_down_shadow(il);          /* 维度/类型口径, bytes=0 不读 */
        l->ffn_gate_shexp  = need(m, "blk.%u.ffn_gate_shexp.weight", il);
        l->ffn_up_shexp    = need(m, "blk.%u.ffn_up_shexp.weight", il);
        l->ffn_down_shexp  = need(m, "blk.%u.ffn_down_shexp.weight", il);
        if (v->engram_index_of[il] >= 0) {
            l->engram_wkv = need(m, "blk.%u.engram_wkv.weight", il);
            l->engram_q   = need(m, "blk.%u.engram_q.weight", il);
            l->engram_k   = need(m, "blk.%u.engram_k.weight", il);
        }
    }

    /* ---- DSpark 三塔(speed.md 段 6, 2026-09-15): 结构与普通层同构, 专家不走 VQ 而是逐专家 FP4 ---- */
    for (uint32_t T = 0; T < v->mtp_towers; T++) {
        ds4_layer_weights *t = &w->mtp.tower[T];
        t->hc_attn_fn     = need(m, "mtp.%u.hc_attn_fn.weight", T);
        t->hc_attn_scale  = need(m, "mtp.%u.hc_attn_scale.weight", T);
        t->hc_attn_base   = need(m, "mtp.%u.hc_attn_base.weight", T);
        t->attn_norm      = need(m, "mtp.%u.attn_norm.weight", T);
        t->attn_q_a       = need(m, "mtp.%u.attn_q_a.weight", T);
        t->attn_q_a_norm  = need(m, "mtp.%u.attn_q_a_norm.weight", T);
        t->attn_q_b       = need(m, "mtp.%u.attn_q_b.weight", T);
        t->attn_kv        = need(m, "mtp.%u.attn_kv.weight", T);
        t->attn_kv_a_norm = need(m, "mtp.%u.attn_kv_a_norm.weight", T);
        t->attn_sinks     = need(m, "mtp.%u.attn_sinks.weight", T);
        t->attn_output_a  = need(m, "mtp.%u.attn_output_a.weight", T);
        t->attn_output_b  = need(m, "mtp.%u.attn_output_b.weight", T);
        t->hc_ffn_fn      = need(m, "mtp.%u.hc_ffn_fn.weight", T);
        t->hc_ffn_scale   = need(m, "mtp.%u.hc_ffn_scale.weight", T);
        t->hc_ffn_base    = need(m, "mtp.%u.hc_ffn_base.weight", T);
        t->ffn_norm       = need(m, "mtp.%u.ffn_norm.weight", T);
        t->ffn_gate_inp   = need(m, "mtp.%u.ffn_gate_inp.weight", T);
        t->ffn_exp_probs_b= need(m, "mtp.%u.exp_probs_b.bias", T);
        t->ffn_gate_shexp = need(m, "mtp.%u.ffn_gate_shexp.weight", T);
        t->ffn_up_shexp   = need(m, "mtp.%u.ffn_up_shexp.weight", T);
        t->ffn_down_shexp = need(m, "mtp.%u.ffn_down_shexp.weight", T);
        for (uint32_t e = 0; e < v->mtp_experts; e++) {
            char nm[96];
            snprintf(nm, sizeof nm, "mtp.%u.ffn_exp.%u.gate.weight", T, e); w->mtp.exp_gate[T][e] = required_tensor_name(m, nm);
            snprintf(nm, sizeof nm, "mtp.%u.ffn_exp.%u.up.weight", T, e);   w->mtp.exp_up[T][e]   = required_tensor_name(m, nm);
            snprintf(nm, sizeof nm, "mtp.%u.ffn_exp.%u.down.weight", T, e); w->mtp.exp_down[T][e] = required_tensor_name(m, nm);
        }
    }
    if (v->mtp_towers) {
        w->mtp.main_proj   = required_tensor_name(m, "mtp.main_proj.weight");
        w->mtp.main_norm   = required_tensor_name(m, "mtp.main_norm.weight");
        w->mtp.markov_embd = required_tensor_name(m, "mtp.markov_embd.weight");
        w->mtp.markov_head = required_tensor_name(m, "mtp.markov_head.weight");
        w->mtp.confidence  = required_tensor_name(m, "mtp.confidence.weight");
        w->mtp.out_norm    = required_tensor_name(m, "mtp.out_norm.weight");
        fprintf(stderr, "ds4: [v41] DSpark 三塔已接线: %u 塔 × %u 专家\n", v->mtp_towers, v->mtp_experts);
    }

    /* ---- 形状/类型校验: 一处错就停(V4.1 的格式解错不报错只出假数) ---- */
    const uint64_t E = DS4_N_EMBD, hc_dim = (uint64_t)E * DS4_N_HC, mix = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM, out_low = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t grp_in = DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    expect(w->token_embd, DS4_TENSOR_FP4X32, 2, E, DS4_N_VOCAB);
    expect(w->output_norm, DS4_TENSOR_F32, 1, E, 0);
    expect(w->output, DS4_TENSOR_FP4X32, 2, E, DS4_N_VOCAB);
    if (v->n_engram) {
        expect(w->engram_token_map, DS4_TENSOR_I32, 1, DS4_N_VOCAB, 0);
        expect(w->engram_multipliers, 27u, 2, v->engram_max_ngram, v->n_engram);
        expect(w->engram_offsets, 27u, 2, (uint64_t)(v->engram_max_ngram - 1) * v->engram_heads, v->n_engram);
        if (!w->engram_primes || w->engram_primes->type != 27u || w->engram_primes->ndim != 3) ds4_die("engram.primes must be I64 [heads][ngram-1][n_engram]");
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        if (!l->attn_norm) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        expect(l->hc_attn_fn, DS4_TENSOR_F32, 2, hc_dim, mix);
        expect(l->hc_attn_scale, DS4_TENSOR_F32, 1, 3, 0);
        expect(l->hc_attn_base, DS4_TENSOR_F32, 1, mix, 0);
        expect(l->attn_norm, DS4_TENSOR_F32, 1, E, 0);
        expect(l->attn_q_a, DS4_TENSOR_FP4X32, 2, E, DS4_N_LORA_Q);
        expect(l->attn_q_a_norm, DS4_TENSOR_F32, 1, DS4_N_LORA_Q, 0);
        expect(l->attn_q_b, DS4_TENSOR_FP4X32, 2, DS4_N_LORA_Q, q_dim);
        expect(l->attn_kv, DS4_TENSOR_FP4X32, 2, E, DS4_N_HEAD_DIM);
        expect(l->attn_kv_a_norm, DS4_TENSOR_F32, 1, DS4_N_HEAD_DIM, 0);
        expect(l->attn_sinks, DS4_TENSOR_F32, 1, DS4_N_HEAD, 0);
        expect(l->attn_output_a, DS4_TENSOR_FP4X32, 2, grp_in, out_low);
        expect(l->attn_output_b, DS4_TENSOR_FP4X32, 2, out_low, E);
        if (v->is_kv_source[il]) {
            expect(l->attn_compressor_kv, DS4_TENSOR_BF16, 2, E, DS4_N_HEAD_DIM);
            if (ratio > 1) expect(l->attn_compressor_gate, DS4_TENSOR_BF16, 2, E, DS4_N_HEAD_DIM);
            expect(l->attn_compressor_norm, DS4_TENSOR_F32, 1, DS4_N_HEAD_DIM, 0);
            expect(l->indexer_wk, DS4_TENSOR_BF16, 2, DS4_N_HEAD_DIM, DS4_N_INDEXER_HEAD_DIM);
            expect(l->indexer_k_norm, DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0);
        }
        if (v->is_index_source[il]) {
            expect(l->indexer_attn_q_b, DS4_TENSOR_FP4X32, 2, DS4_N_LORA_Q, (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM);
            expect(l->indexer_proj, DS4_TENSOR_BF16, 2, E, DS4_N_INDEXER_HEAD);
        }
        expect(l->hc_ffn_fn, DS4_TENSOR_F32, 2, hc_dim, mix);
        expect(l->hc_ffn_scale, DS4_TENSOR_F32, 1, 3, 0);
        expect(l->hc_ffn_base, DS4_TENSOR_F32, 1, mix, 0);
        expect(l->ffn_norm, DS4_TENSOR_F32, 1, E, 0);
        expect(l->ffn_gate_inp, DS4_TENSOR_BF16, 2, E, DS4_N_EXPERT);
        expect(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0);
        expect(l->ffn_gate_shexp, DS4_TENSOR_FP4X32, 2, E, DS4_N_FF_EXP);
        expect(l->ffn_up_shexp, DS4_TENSOR_FP4X32, 2, E, DS4_N_FF_EXP);
        expect(l->ffn_down_shexp, DS4_TENSOR_FP4X32, 2, DS4_N_FF_EXP, E);
        if (v->engram_index_of[il] >= 0) {
            expect(l->engram_wkv, DS4_TENSOR_FP8_32X32, 2, (uint64_t)(v->engram_max_ngram - 1) * v->engram_heads * v->engram_head_dim, E * (DS4_N_HC + 1));
            expect(l->engram_q, DS4_TENSOR_F32, 2, E, DS4_N_HC);
            expect(l->engram_k, DS4_TENSOR_F32, 2, E, DS4_N_HC);
        }
    }
}
