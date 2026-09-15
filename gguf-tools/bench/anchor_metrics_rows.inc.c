/* anchor_metrics_rows.inc.c — --rows 的取行号工具(2026-09-13 夜)。anchor_metrics.c 单 TU include;
 * 拆出来只为守单文件 ≤500 行, 逻辑一个字没改。 */
/* --rows "a:b" 或 "a:b,c:d,..." → 绝对行号数组(升序去重不做, 由调用方保证不重叠)。
 * 为什么要多段: 每条样本是[当日材料][本应写的报告], 只有报告那一段是后训练的目标 token,
 * 材料段不算数; N 条样本 = N 段不连续区间, 但要合成一个池子出一份平均 NLL。 */
static int *rows_parse(const char *spec, int S, int *out_n) {
    int cap = 64, n = 0;
    int *r = malloc((size_t)cap * sizeof(int));
    /* ★"@文件" = 每行一个行号★(2026-09-13 夜): 后训练的决策点/约束行是采样出来的, 行号不连续,
     * 写成区间就是几千段撑爆命令行 —— 解算器侧(v41_sft_run.inc.c 的 rows_spec_parse)早就认这个
     * 前缀了, 判决器不认。两处口径漂的代价是实撞过的: 尺 A 报 0/0, 错误被 `|| true` 吞掉,
     * 看起来像"一个决策点都没有", 其实是参数没被接受。 */
    if (spec[0] == '@') {
        FILE *f = fopen(spec + 1, "r");
        if (!f) { fprintf(stderr, "--rows 行号文件打不开 %s\n", spec + 1); free(r); return NULL; }
        int v;
        while (fscanf(f, "%d", &v) == 1) {
            if (v < 0 || v >= S) continue;   /* 越界行直接丢: 判决器的 S 是这一趟表的行数 */
            if (n == cap) { cap *= 2; r = realloc(r, (size_t)cap * sizeof(int)); }
            r[n++] = v;
        }
        fclose(f);
        if (!n) { free(r); return NULL; }
        *out_n = n;
        return r;
    }
    const char *p = spec;
    while (*p) {
        int a, b, adv = 0;
        if (sscanf(p, "%d:%d%n", &a, &b, &adv) != 2) { free(r); return NULL; }
        if (b > S) b = S;
        for (int i = a; i < b; i++) {
            if (i < 0) { free(r); return NULL; }
            if (n == cap) { cap *= 2; r = realloc(r, (size_t)cap * sizeof(int)); }
            r[n++] = i;
        }
        p += adv;
        if (*p == ',') p++;
        else if (*p) { free(r); return NULL; }
    }
    if (!n) { free(r); return NULL; }
    *out_n = n;
    return r;
}
static int *rows_range(int lo, int hi) {
    int *r = malloc((size_t)(hi - lo) * sizeof(int));
    for (int i = lo; i < hi; i++) r[i - lo] = i;
    return r;
}