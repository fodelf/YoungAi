/* core_validate_v41.c — V4.1 接线元数据装填(2026-09-12 战役 P1)。
 *
 * V4.1 与 V4 Flash 的结构差异不在维度而在"接线": 只有 kv_source_layers 压缩并持有压缩 KV,
 * 其余压缩层读最近一个源层的缓存; 只有 index_source_layers 跑 indexer, 其余层复用最近源层的
 * topk; candidate_source_layer 之后各层的 topk 限制在它筛出的候选块内; engram 只在两层。
 * 这些全部来自转换器写进 GGUF 的 deepseek4.* 键(与官方 config.json 一一对应), 这里装进
 * g_ds4_v41 供热路径按层查表。缺键 = 硬停, 不给默认值。 */
#include "core_internal.h"

/* --engram-dir: table_path 是转换器 realpath 出来的绝对路径(v41_to_gguf.c), 只在转换那台机器上成立。
 * 发布出去的 GGUF 不能为此重写(改一个字符串长度 = 数据区整体挪位 = 40 个分块全部重传), 所以在装载时换目录:
 * 只留 GGUF 里的文件名, 拼到这个目录下。不传 = 原样用 GGUF 里的路径(转换机本机)。 */
static const char *g_v41_engram_dir = NULL;
void ds4_engine_v41_set_engram_dir(const char *dir) { g_v41_engram_dir = (dir && dir[0]) ? dir : NULL; }

static uint32_t v41_arr_i32(const ds4_model *m, const char *key, int32_t *out, uint32_t cap) {
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) || (arr.type != GGUF_VALUE_INT32 && arr.type != GGUF_VALUE_UINT32)) {
        fprintf(stderr, "ds4: V4.1 required int32 array key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len > cap) ds4_die("V4.1 metadata array longer than capacity");
    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint64_t i = 0; i < arr.len; i++) if (!cursor_read(&c, &out[i], 4)) ds4_die(c.error);
    return (uint32_t)arr.len;
}

static uint32_t v41_arr_u64(const ds4_model *m, const char *key, uint64_t *out, uint32_t cap) {
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) || arr.type != GGUF_VALUE_UINT64) {
        fprintf(stderr, "ds4: V4.1 required uint64 array key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len > cap) ds4_die("V4.1 metadata array longer than capacity");
    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint64_t i = 0; i < arr.len; i++) if (!cursor_u64(&c, &out[i])) ds4_die(c.error);
    return (uint32_t)arr.len;
}

static uint64_t v41_req_u64(const ds4_model *m, const char *key) {
    uint64_t v = 0;
    if (!model_get_u64_compat(m, key, &v)) { fprintf(stderr, "ds4: V4.1 required key is missing: %s\n", key); exit(1); }
    return v;
}

void v41_load_metadata(const ds4_model *m) {
    ds4_v41_cfg *v = &g_ds4_v41;
    memset(v, 0, sizeof *v);
    v->active = 1;
    /* ★上下文只从模型元数据来★(用户 2026-09-22 "不要任何写死的上下文, 上下文大小只有 1M 这一个选择"): 转换器把 HF config 的
     * max_position_embeddings 写成 deepseek4.context_length, 引擎只认这个键 —— 没有默认值、没有上限常量、没有 --ctx。 */
    const uint64_t ctx = v41_req_u64(m, "deepseek4.context_length");
    if (ctx == 0 || ctx > UINT32_MAX) ds4_die("V4.1 deepseek4.context_length 不合法");
    v->ctx = (uint32_t)ctx;
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        v->kv_source_of[il] = -1; v->index_source_of[il] = -1; v->engram_index_of[il] = -1;
    }
    int32_t ids[DS4_MAX_LAYER];
    uint32_t n = v41_arr_i32(m, "deepseek4.attention.kv_source_layers", ids, DS4_MAX_LAYER);
    for (uint32_t i = 0; i < n; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= DS4_N_LAYER) ds4_die("kv_source_layers out of range");
        v->is_kv_source[ids[i]] = 1;
    }
    n = v41_arr_i32(m, "deepseek4.attention.index_source_layers", ids, DS4_MAX_LAYER);
    for (uint32_t i = 0; i < n; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= DS4_N_LAYER) ds4_die("index_source_layers out of range");
        v->is_index_source[ids[i]] = 1;
    }
    /* 接线: 压缩层读"最近一个 ≤ 本层的源层"。官方 SharedAttentionRuntime 的语义就是
     * "源层写一槽, 后面的层读同一槽", 层按序执行, 所以等价于最近源层。 */
    int16_t last_kv = -1, last_idx = -1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (v->is_kv_source[il]) last_kv = (int16_t)il;
        if (v->is_index_source[il]) last_idx = (int16_t)il;
        if (ds4_layer_compress_ratio(il) == 0) continue;
        if (last_kv < 0 || last_idx < 0) {
            fprintf(stderr, "ds4: V4.1 layer %u compresses but has no kv/index source before it\n", il);
            exit(1);
        }
        v->kv_source_of[il] = last_kv;
        v->index_source_of[il] = last_idx;
        if (v->is_kv_source[il] && !v->is_index_source[il])
            ds4_die("V4.1: a kv source layer must also be an index source (indexer keys come from its latent)");
    }
    /* DSpark 三塔: 可选 —— 09-15 之前转出来的 GGUF 没带, 那时 mtp_towers=0, 引擎就走纯单 token 解码。
     * 不用 required_u32: 那会让旧模型直接起不来(铁律: 不破坏已有产物)。 */
    if (!model_get_u32(m, "deepseek4.mtp.tower_count", &v->mtp_towers)) v->mtp_towers = 0;
    if (!model_get_u32(m, "deepseek4.mtp.expert_count", &v->mtp_experts)) v->mtp_experts = 0;
    if (v->mtp_towers > DS4_MTP_MAX_TOWERS || v->mtp_experts > DS4_MTP_MAX_EXPERTS)
        ds4_die("mtp tower/expert count over engine limit");
    /* 草稿器跑起来要的五件(speed.md 段 6 D1)。同样可选: 少一件就把 mtp_block 归零 = 投机路不武装,
     * 引擎照常单 token 解码 —— 不猜默认值(猜错了不报错, 只是草稿全不对, 接受率掉到 0)。 */
    for (uint32_t i = 0; i < DS4_MAX_LAYER; i++) v->mtp_target_slot[i] = -1;
    if (v->mtp_towers) {
        uint32_t blk = 0, used = 0, noise = 0, rank = 0;
        const int have = model_get_u32(m, "deepseek4.mtp.block_size", &blk) &&
                         model_get_u32(m, "deepseek4.mtp.expert_used_count", &used) &&
                         model_get_u32(m, "deepseek4.mtp.noise_token_id", &noise) &&
                         model_get_u32(m, "deepseek4.mtp.markov_rank", &rank);
        int32_t tids[DS4_MTP_MAX_TOWERS * 2];
        const uint32_t nt = have ? v41_arr_i32(m, "deepseek4.mtp.target_layers", tids, DS4_MTP_MAX_TOWERS * 2) : 0;
        if (have && nt) {
            v->mtp_block = blk; v->mtp_used = used; v->mtp_noise_id = noise; v->mtp_markov_rank = rank;
            v->n_mtp_target = nt;
            for (uint32_t i = 0; i < nt; i++) {
                if (tids[i] < 0 || (uint32_t)tids[i] >= DS4_N_LAYER) ds4_die("mtp target layer out of range");
                v->mtp_target[i] = (int16_t)tids[i];
                v->mtp_target_slot[tids[i]] = (int16_t)i;
            }
        } else {
            fprintf(stderr, "ds4: [v41] 这份 GGUF 带了三塔但没带 DSpark 运行参数(block_size/target_layers 等) ⇒ 投机解码不武装\n");
        }
    }
    v->candidate_source_layer = (int32_t)required_u32(m, "deepseek4.attention.candidate.source_layer");
    v->candidate_topk_blocks = (int32_t)required_u32(m, "deepseek4.attention.candidate.topk_blocks");
    v->candidate_block_size = (int32_t)required_u32(m, "deepseek4.attention.candidate.block_size");

    n = v41_arr_i32(m, "deepseek4.engram.layer_ids", ids, DS4_V41_MAX_ENGRAM);
    v->n_engram = n;
    uint64_t rows[DS4_V41_MAX_ENGRAM];
    if (v41_arr_u64(m, "deepseek4.engram.num_embeddings", rows, DS4_V41_MAX_ENGRAM) != n)
        ds4_die("engram.num_embeddings count != engram.layer_ids count");
    for (uint32_t i = 0; i < n; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= DS4_N_LAYER) ds4_die("engram layer id out of range");
        v->engram_layer[i] = ids[i];
        v->engram_index_of[ids[i]] = (int16_t)i;
        v->engram_rows[i] = rows[i];
        char key[96]; ds4_str s;
        snprintf(key, sizeof key, "deepseek4.engram.%u.table_path", i);
        if (!model_get_string(m, key, &s) || s.len == 0 || s.len >= sizeof v->engram_table_path[i]) {
            fprintf(stderr, "ds4: V4.1 required key is missing: %s\n", key); exit(1);
        }
        memcpy(v->engram_table_path[i], s.ptr, s.len); v->engram_table_path[i][s.len] = 0;
        if (g_v41_engram_dir) {
            char name[sizeof v->engram_table_path[i]];
            const char *slash = strrchr(v->engram_table_path[i], '/');
            snprintf(name, sizeof name, "%s", slash ? slash + 1 : v->engram_table_path[i]);
            const int w = snprintf(v->engram_table_path[i], sizeof v->engram_table_path[i], "%s/%s", g_v41_engram_dir, name);
            if (w < 0 || (size_t)w >= sizeof v->engram_table_path[i]) ds4_die("--engram-dir path too long");
            fprintf(stderr, "ds4: engram 表 %u 用 %s(--engram-dir)\n", i, v->engram_table_path[i]);
        }
        /* 表到第一次前向才打开; 加载要 2 分钟, 不在这里先说一声, 人要等到第一个请求失败才知道。
         * 只警告不停: --score-ids 的 no-engram 对拍口径本来就不读表。 */
        if (access(v->engram_table_path[i], R_OK) != 0)
            fprintf(stderr, "ds4: ★engram 表打不开 %s, 第一个请求会失败 —— 用 --engram-dir 指向放官方分片的目录★\n", v->engram_table_path[i]);
        snprintf(key, sizeof key, "deepseek4.engram.%u.weight_offset", i); v->engram_weight_off[i] = v41_req_u64(m, key);
        snprintf(key, sizeof key, "deepseek4.engram.%u.scale_offset", i);  v->engram_scale_off[i]  = v41_req_u64(m, key);
    }
    v->engram_max_ngram = required_u32(m, "deepseek4.engram.max_ngram_size");
    v->engram_heads     = required_u32(m, "deepseek4.engram.head_count");
    v->engram_head_dim  = required_u32(m, "deepseek4.engram.head_dim");
    v->engram_vocab     = required_u32(m, "deepseek4.engram.vocab_size");
    v->engram_cvocab    = required_u32(m, "deepseek4.engram.compressed_vocab_size");
    v->engram_pad       = required_u32(m, "deepseek4.engram.pad_id_compressed");
    v->swiglu_limit     = DS4_SWIGLU_CLAMP_EXP;

    int nk = 0, ni = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) { nk += v->is_kv_source[il]; ni += v->is_index_source[il]; }
    fprintf(stderr, "ds4: V4.1 接线: kv 源 %d 层 / indexer 源 %d 层 / 候选源 L%d(%d 块×%d) / engram %u 层 (表 %llu+%llu 行, 在盘)\n",
            nk, ni, v->candidate_source_layer, v->candidate_topk_blocks, v->candidate_block_size, v->n_engram,
            (unsigned long long)(n > 0 ? v->engram_rows[0] : 0), (unsigned long long)(n > 1 ? v->engram_rows[1] : 0));
}
