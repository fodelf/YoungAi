/* v41_fp_bind.inc.c — 反修 FP 靶要的 HF 出厂权重绑定(2026-09-13 从 v41_amp_run.c 拆出, 只为守 500 行)。
 * 做的事就一件: 按 safetensors 里的张量名找到某层某专家的 w1/w3/w2, 把指针与形状填进 v41_fp_mat
 * (数据仍在 mmap 上, 不拷)。routed 专家是 I8+分组 scale, shared 专家是 E4M3 —— dtype 对不上直接停车,
 * 因为"当成另一种格式解出来"不会报错, 只会出一堆看着合理的假数。 */

/* HF 张量 → v41_fp_mat(指针指向 mmap, 分片按需映射) */
static int fill_mat(v41_st *S, const char *base, v41_fp_mat *M, int fp8) {
    char wn[256], sn[256]; snprintf(wn, sizeof wn, "%s.weight", base); snprintf(sn, sizeof sn, "%s.scale", base);
    const v41_st_ent *W = v41_st_find(S, wn), *Sc = v41_st_find(S, sn);
    if (!W || !Sc) { fprintf(stderr, "★HF 缺 %s / .scale★\n", wn); return -1; }
    if (strcmp(W->dtype, fp8 ? "F8_E4M3" : "I8")) { fprintf(stderr, "★%s dtype %s, 期 %s★\n", wn, W->dtype, fp8 ? "F8_E4M3" : "I8"); return -1; }
    if (W->nd != 2 || Sc->nd != 2) { fprintf(stderr, "★%s 不是二维★\n", wn); return -1; }
    M->w = v41_st_data(S, W); M->s = v41_st_data(S, Sc);
    if (!M->w || !M->s) return -1;
    M->rows = (int)W->shape[0]; M->cols = fp8 ? (int)W->shape[1] : (int)W->shape[1] * 2; M->fp8 = fp8;
    if (fp8) { M->sbr = M->rows / (int)Sc->shape[0]; M->sbc = M->cols / (int)Sc->shape[1]; }
    else {
        M->sbr = 1; M->sbc = 32;
        if (Sc->shape[0] != M->rows || Sc->shape[1] * 32 != M->cols) { fprintf(stderr, "★%s scale 形状 %lld×%lld 对不上 %d×%d★\n", wn, (long long)Sc->shape[0], (long long)Sc->shape[1], M->rows, M->cols); return -1; }
    }
    return 0;
}
static int resolve_layer(ctx_t *c, int il) {
    char nm[256];
    for (int e = 0; e < c->n_expert; e++) {
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w1", il, e); if (fill_mat(&c->S, nm, &c->ex[e].w1, 0)) return -1;
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w3", il, e); if (fill_mat(&c->S, nm, &c->ex[e].w3, 0)) return -1;
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w2", il, e); if (fill_mat(&c->S, nm, &c->ex[e].w2, 0)) return -1;
    }
    snprintf(nm, sizeof nm, "layers.%d.ffn.shared_experts.w1", il); if (fill_mat(&c->S, nm, &c->sh.w1, 1)) return -1;
    snprintf(nm, sizeof nm, "layers.%d.ffn.shared_experts.w3", il); if (fill_mat(&c->S, nm, &c->sh.w3, 1)) return -1;
    snprintf(nm, sizeof nm, "layers.%d.ffn.shared_experts.w2", il); if (fill_mat(&c->S, nm, &c->sh.w2, 1)) return -1;
    return 0;
}

