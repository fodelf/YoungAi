/* v41_sft_run.inc.c — 后训练(三文件部署的第三件)的驱动(2026-09-13 夜第二版)。
 * v41_amp_run.c 单 TU include; 拆出来只为守单文件 ≤500 行, 数值全在 v41_gr_solve.cu / v41_kl_target.cu。
 *
 * 【解什么】每层每个专家每个输出通道一个增益缩放因子(与 ② 反修插件同构, 引擎里两张表逐元素相乘)。
 *
 * 【靶是什么 —— 第二版换了】不再拟合"损失对这一层输出的梯度"。09-13 段 1 实测那条路 train 只挽回
 * 0.20%、val 0.04%, 缩放因子却已经跑到 [0.29, 2.00] —— 梯度是由目标 token 词向量定的 5120 维稠密方向,
 * 与"这一行选了哪 6 个专家"没有结构关系, 增益形态表达不了它; 而那 5119 个无关方向占了靶能量的 99.8%。
 * 换成"谁压过谁"的 logit 差: 末层之后只有 norm→head, 所以它对本层 MoE 输出是精确线性的
 * (v41_gr_solve.cu 第二段有推导), 撬动一个错误判断 = 一条线性方程。
 * ★解算与主动集在 v41_margin_solve.inc.c★(拆文件只为守 500 行; 本文件只管取料与行表)。
 *
 * 【为什么一条样本一趟】引擎 V4.1 的上下文硬上限 32768(core_v41.h), 一条样本约 5~6k token, 八条串起来
 * 必然撞墙 —— 撞了不会明着报错, 只会给一串安静的错数。所以逐条取料, 把【选中的行】累积进同一批。
 *
 * 【产物是 fp4】落盘是 fp4x32(4 bit/元素)。所以每个候选都是 ★解 → 量化 → 解回来 → 用落地那份重算
 * 预测★; 量化误差把某个决策点又踩回 τ 以下时, 那一行进主动集再解一遍。报出来的翻转率永远是落地态的。 */


/* "a:b" 或 "@文件"(每行一个行号) → 绝对行号数组。越界直接判错而不是悄悄裁剪: 行号是脚本按 token
 * 序列算出来的, 对不上说明配错了文件。 */
static int *rows_spec_parse(const char *spec, int ntok, int *out_n) {
    int cap = 1024, n = 0;
    int *r = malloc((size_t)cap * sizeof(int));
    if (spec[0] == '@') {
        FILE *f = fopen(spec + 1, "r");
        if (!f) { fprintf(stderr, "★行号文件打不开 %s★\n", spec + 1); free(r); return NULL; }
        int v;
        while (fscanf(f, "%d", &v) == 1) {
            if (v < 0 || v >= ntok) { fprintf(stderr, "★%s 里的行号 %d 越界(ntok=%d)★\n", spec + 1, v, ntok); fclose(f); free(r); return NULL; }
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
        if (a < 0 || b > ntok || b <= a) { fprintf(stderr, "★行段 %d:%d 越界(ntok=%d)★\n", a, b, ntok); free(r); return NULL; }
        for (int i = a; i < b; i++) {
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

/* 清单一行 = 一条样本: <对版ids> <topk表> <rms表> <错版ids 或 -> <约束行段> <决策点行段>
 * ★决策点只能是"每处改动的第一个分歧位置"★: 在那里两版的前缀逐字相同、行号也对齐, 所以
 * "该写对版还是错版"是可比的。块内往后的位置, 对版序列的前缀已经是对版自己写的了 —— 拿错版
 * 序列的同行号 token 去比, 比的是两条已经分叉的序列, 量的是 teacher-forcing 的自洽偏置。 */
typedef struct { char ids[1024], top[1024], rms[1024], alt[1024], ctr[4096], dec[4096], nll[1100]; } sft_item;

static sft_item *sft_list_read(const char *path, int *n_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "★清单打不开 %s★\n", path); return NULL; }
    int cap = 32, n = 0;
    sft_item *it = malloc((size_t)cap * sizeof(sft_item));
    char line[12288];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (n == cap) { cap *= 2; it = realloc(it, (size_t)cap * sizeof(sft_item)); }
        if (sscanf(line, "%1023s %1023s %1023s %1023s %4095s %4095s", it[n].ids, it[n].top, it[n].rms,
                   it[n].alt, it[n].ctr, it[n].dec) != 6) {
            fprintf(stderr, "★清单第 %d 行不是 6 列(<对版ids> <topk> <rms> <错版ids|-> <约束行段> <决策点行段>)★\n", n + 1);
            fclose(f); free(it); return NULL;
        }
        snprintf(it[n].nll, sizeof it[n].nll, "%.1023s.nll", it[n].top);   /* NLL 与榜单同一趟落, 名字跟着榜单走 */
        n++;
    }
    fclose(f);
    if (!n) { fprintf(stderr, "★清单是空的★\n"); free(it); return NULL; }
    *n_out = n;
    return it;
}

static int *sft_ids_read(const char *path, int *n_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "★ids 打不开 %s★\n", path); return NULL; }
    int cap = 8192, n = 0, v;
    int *a = malloc((size_t)cap * 4);
    while (fscanf(f, "%d", &v) == 1) {
        if (n == cap) { cap *= 2; a = realloc(a, (size_t)cap * 4); }
        a[n++] = v;
    }
    fclose(f);
    *n_out = n;
    return a;
}

/* "<目录>/<名字>" → "<目录>/<前缀><名字>"。验证趟的表要和取料趟的表并排放、名字只差一个前缀,
 * 判决器那边就能按同一套规则找到两张表做对比(守门 1 就是比这两张)。 */
static void sft_pref_path(const char *path, const char *pref, char *out, size_t cap) {
    const char *b = strrchr(path, '/');
    if (!b) { snprintf(out, cap, "%s%s", pref, path); return; }
    snprintf(out, cap, "%.*s%s%s", (int)(b - path + 1), path, pref, b + 1);
}

/* 引擎 --score-rms 的产物: f32[S], 每位置一个 inv = rsqrt(mean(x²)+eps)。行数必须等于 ids 行数 ——
 * 少一行就整体错位, 而错位的方向表照样解得出一个数。 */
static float *sft_rms_read(const char *path, int nids) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "★rms 表打不开 %s(引擎要传 --score-rms)★\n", path); return NULL; }
    float *a = malloc((size_t)nids * 4);
    const size_t got = fread(a, 4, (size_t)nids, f);
    long extra = 0;
    if (!fseek(f, 0, SEEK_END)) extra = ftell(f) - (long)nids * 4;
    fclose(f);
    if (got != (size_t)nids || extra != 0) {
        fprintf(stderr, "★%s 是 %zu+%ld 行, ids 是 %d 行 —— 不是同一条序列的产物★\n", path, got, extra / 4, nids);
        free(a); return NULL;
    }
    return a;
}

/* 累积缓冲: 行按 [决策拟合行, 决策 val 行, 约束行] 排好 —— 解算器按权重区分, 预测按段统计。
 * ★整张 top-K 榜也要留着★(09-13 夜第三针): 主动集要看"修正之后榜上谁冒头了", 只留一对不够。 */
typedef struct {
    float *ye, *rw, *alpha, *inv;  int *sel;   /* 取料侧 */
    float *xh;                                 /* 本层输入隐状态 [n][D]: ★x 键控门控★的原料(back.md 段 4) */
    int   *ta, *tb;                            /* 首对: 决策行 = (对版, 最强对手), 约束行 = (top1, top2) */
    float *m0;                                 /* 首对当前的 logit 差(从 top-K 表算) */
    int   *want; float *wlp;                   /* 这一行"该赢的那个 token"及其 log p(决策行=对版, 约束行=top1) */
    int   *tid;  float *tlp;                   /* top-K 榜的 id 与 log p, [n][K] */
    int   *smp, *rowid;                        /* 属于第几条样本 / 在那条样本序列里的绝对行号 */
    float *repw;                               /* 词对重复度权重(back.md §4.7b): 同一对 token 跨几条样本出现过 */
    int    n, cap, nu, D, K;
} mg_acc;

static int mg_acc_alloc(mg_acc *a, int cap, int nu, int D, int K) {
    memset(a, 0, sizeof *a);
    a->cap = cap; a->nu = nu; a->D = D; a->K = K;
    a->ye = malloc((size_t)cap * nu * D * 4); a->rw = malloc((size_t)cap * nu * 4);
    a->xh = malloc((size_t)cap * D * 4);
    a->sel = malloc((size_t)cap * nu * 4); a->alpha = malloc((size_t)cap * 4);
    a->inv = malloc((size_t)cap * 4); a->ta = malloc((size_t)cap * 4); a->tb = malloc((size_t)cap * 4);
    a->m0 = malloc((size_t)cap * 4); a->smp = malloc((size_t)cap * 4); a->rowid = malloc((size_t)cap * 4);
    a->want = malloc((size_t)cap * 4); a->wlp = malloc((size_t)cap * 4);
    a->tid = malloc((size_t)cap * K * 4); a->tlp = malloc((size_t)cap * K * 4);
    a->repw = malloc((size_t)cap * 4);
    if (!a->ye || !a->xh || !a->rw || !a->sel || !a->alpha || !a->inv || !a->ta || !a->tb || !a->m0 ||
        !a->smp || !a->rowid || !a->want || !a->wlp || !a->tid || !a->tlp || !a->repw) {
        fprintf(stderr, "★累积缓冲(%d 行 × %.1f KB)分配失败★\n", cap, (double)(nu * D * 4) / 1024); return -1; }
    return 0;
}
/* ★把第 src 槽整体搬到第 dst 槽 —— 字段清单只许有这一份★。
 * 实撞(09-13 22:29): 压实那段原来把字段一个个抄了两遍, 后来给累积区加了"整张榜"四个数组,
 * 只在填充处加了、两处搬运都漏了 —— 于是 val 段与约束段读到错位的榜, 扫榜核当场越界
 * (运气好; 它也完全可能不越界, 只是安静地出一组错数)。 */
static void mg_acc_move(mg_acc *a, int dst, int src) {
    const int nu = a->nu, D = a->D, K = a->K;
    memmove(a->ye + (size_t)dst * nu * D, a->ye + (size_t)src * nu * D, (size_t)nu * D * 4);
    memmove(a->xh + (size_t)dst * D, a->xh + (size_t)src * D, (size_t)D * 4);
    memmove(a->rw + (size_t)dst * nu, a->rw + (size_t)src * nu, (size_t)nu * 4);
    memmove(a->sel + (size_t)dst * nu, a->sel + (size_t)src * nu, (size_t)nu * 4);
    memmove(a->tid + (size_t)dst * K, a->tid + (size_t)src * K, (size_t)K * 4);
    memmove(a->tlp + (size_t)dst * K, a->tlp + (size_t)src * K, (size_t)K * 4);
    a->alpha[dst] = a->alpha[src]; a->inv[dst] = a->inv[src];
    a->ta[dst] = a->ta[src]; a->tb[dst] = a->tb[src]; a->m0[dst] = a->m0[src];
    a->want[dst] = a->want[src]; a->wlp[dst] = a->wlp[src];
    a->smp[dst] = a->smp[src]; a->rowid[dst] = a->rowid[src];
    a->repw[dst] = a->repw[src];
}

static void mg_acc_free(mg_acc *a) {
    free(a->ye); free(a->xh); free(a->rw); free(a->sel); free(a->alpha); free(a->inv);
    free(a->ta); free(a->tb); free(a->m0); free(a->smp); free(a->rowid);
    free(a->want); free(a->wlp); free(a->tid); free(a->tlp); free(a->repw);
}

/* 一行的 (a, b, m⁰) —— 这一行要把谁压过谁:
 *   决策行: a = 对版 token, b = ★除它之外当前概率最高的 token★(见函数里的实测注释)
 *   约束行: a/b = 这一行自己的 top1 与 top2(意思是"这里原本最想说什么, 别给我改")
 * 概率只从 top-K 表来(K=64 实测覆盖 97.3%)。返回 0 = 这一行不可用(没有可比的两个 token)。 */
static int mg_row_pair(const ds4_etgd *tk, int r, int alt_tok, int *pa, int *pb, float *pm0) {
    const int *ids = tk->ids + (size_t)r * tk->K;
    const float *ps = tk->ps + (size_t)r * tk->K;
    int i1 = 0, i2 = -1;
    for (int q = 1; q < tk->K; q++) if (ps[q] > ps[i1]) i1 = q;
    for (int q = 0; q < tk->K; q++) if (q != i1 && (i2 < 0 || ps[q] > ps[i2])) i2 = q;
    if (i2 < 0) return 0;
    if (alt_tok < 0) {   /* 约束行 */
        if (!(ps[i1] > 0.f) || !(ps[i2] > 0.f)) return 0;
        *pa = ids[i1]; *pb = ids[i2]; *pm0 = logf(ps[i1]) - logf(ps[i2]);
        return 1;
    }
    const int t = tk->tgt[r];
    if (t < 0 || alt_tok == t) return 0;
    const float pt = tk->tgt_p[r];
    /* ★对手 = 除对版之外概率最高的那个 token, 不是错版 token★(2026-09-13 夜, 实测定的)
     * 第一版拿错版当对手, 于是 m = ℓ_对 − ℓ_错。冒烟读数当场拆穿了它: 80 个决策点上
     * m⁰>0 的有 81.25%, 平均 m⁰ = +5.86, val 段里一堆 +12/+13/+14 —— 也就是说这些位置上
     * 模型本来就极度偏向对版, 根本不是"错误判断", 解算器在那里无事可做(val 翻转率解前解后
     * 都是 90.0%, 一个没动)。而同一批样本的 argmax 口径只有 50.75%。
     * 差在哪: 部署是贪心 argmax, 比的是"对版 vs 全词表"; 赢过对版的往往是第三方 token,
     * 不是错版。把对版压过错版, argmax 照样不选它 —— 优化目标与部署行为不是同一个量。
     * 取"最强对手"后 m>0 ★严格等价于★ argmax = 对版, 两者变成同一个量; 而且它天然覆盖
     * 错版(错版若就是最强对手, 取到的就是它)。 */
    int ib = -1;
    for (int q = 0; q < tk->K; q++) {
        if (ids[q] == t || !(ps[q] > 0.f)) continue;
        if (ib < 0 || ps[q] > ps[ib]) ib = q;
    }
    if (ib < 0 || !(pt > 0.f)) return 0;
    *pa = t; *pb = ids[ib]; *pm0 = logf(pt) - logf(ps[ib]);
    return 1;
}

/* 解算与主动集在 v41_margin_solve.inc.c(它在本文件之后 include, 所以这里前置声明一次)。
 * 拆两个文件是为了守单文件 ≤500 行: 本文件只管"把料取全、把行摆对", 数值判断一行都不在这里。 */
static int mg_solve_grid(ctx_t *c, mg_acc *ab, int il, const char *out, int ndec_fit, int ndec);

/* 后训练主流程: 逐条样本取料 → 决策行/约束行累积 → 建方向表 → τ×ρ×λ 网格各出一个候选。
 * 真前向判决(尺 A/B 与三把守门尺)不在这里 —— 这里只出候选与【预测】, 由夜间脚本的 gate 段挂上去真跑。
 * 为什么网格在一次取料里做完: 取料一条样本一趟前向(实测 161s), 8 条就是 21 分钟; 解一个候选是秒级。 */
static int sft_run(ctx_t *c, ds4_engine *e, const char *list_path, const char *out,
                   int il, int ntok_cap, int no_engram) {
    int nit = 0;
    sft_item *it = sft_list_read(list_path, &nit);
    if (!it) return -1;
    /* 行预算: 先把清单里所有决策行/约束行数加起来, 缓冲一次配够 —— 中途扩容会把已上传的设备
     * 缓冲搞乱, 而"行数超预算"这种错在解算结果里是看不出来的。 */
    int tot_dec = 0, tot_ctr = 0;
    for (int m = 0; m < nit; m++) {
        int nc = 0, nd = 0;
        int *cr = rows_spec_parse(it[m].ctr, ntok_cap, &nc), *dr = rows_spec_parse(it[m].dec, ntok_cap, &nd);
        if (!cr || !dr) { fprintf(stderr, "★清单第 %d 条的行段不合法★\n", m + 1); free(cr); free(dr); free(it); return -1; }
        free(cr); free(dr); tot_ctr += nc; tot_dec += nd;
    }
    printf("[后训练] %d 条样本, 决策点 %d + 约束行 %d, 层 L%02d\n", nit, tot_dec, tot_ctr, il);
    c->nsmp = nit;   /* 尺 L 按样本条留一, 折数 = 条数(按点随机分折会泄题, 见 v41_margin_gen.inc.c) */
    mg_acc ab; memset(&ab, 0, sizeof ab);
    /* ★一块缓冲三个写指针★: 决策拟合行从 0、决策 val 行从 ndec_fit、约束行从 tot_dec 往后写。
     * 分三块最后再拼要多占一整份(7.5k 行 × 123 KB ≈ 0.9 GB), GB10 是统一内存, 多占就是 GPU 少。
     * 决策 val = 每 4 个决策点留 1 个不解 —— 解算内部的"举一反三"读数(只作诊断, 不当上线依据)。 */
    const int ndec_val = tot_dec / 4, ndec_fit = tot_dec - ndec_val;
    int inited = 0, rc = 0, wf = 0, wv = ndec_fit, wc = tot_dec, dseq = 0;
    for (int m = 0; m < nit && rc == 0; m++) {
        int nids = 0;
        int *ids = sft_ids_read(it[m].ids, &nids);
        if (!ids) { rc = -1; break; }
        if (nids > ntok_cap) { fprintf(stderr, "★第 %d 条 %d 行 > 取料缓冲 %d 行★\n", m + 1, nids, ntok_cap); free(ids); rc = -1; break; }
        c->layer = il; c->got = 0; c->n = nids;
        /* ★部署同路取料★(2026-09-23): 最小的约束/决策行 = 提示最后一位(约束行从报告段前一行起取), 切分点 P = 它 + 1。
         * 提示照生成路 CED 预填, 报告跑满解码器 —— 否则表量的是"提示也跑满解码器"那种部署里不存在的状态,
         * 09-23 实撞: 按那种表解出的 ③ 在服务端一个字没翻(fable5 09-23 夜)。 */
        int split = 0;
        {
            int nc0 = 0, nd0 = 0, mn = nids;
            int *cr0 = rows_spec_parse(it[m].ctr, nids, &nc0), *dr0 = rows_spec_parse(it[m].dec, nids, &nd0);
            for (int q = 0; cr0 && q < nc0; q++) if (cr0[q] < mn) mn = cr0[q];
            for (int q = 0; dr0 && q < nd0; q++) if (dr0[q] < mn) mn = dr0[q];
            free(cr0); free(dr0);
            split = mn < nids ? mn + 1 : 0;
        }
        ds4_engine_v41_set_score_split(split);
        ds4_engine_v41_set_amp_dir(c->base_amp);
        ds4_engine_v41_set_posttrain_dir(c->base_pt);
        ds4_engine_v41_set_moe_hook(hook, c);
        ds4_engine_v41_set_moe_hook_layer(il);   /* 只要末层: 不设的话每块白拷 39 层(实测 33s → 个位数) */
        /* ★一趟拿两样★: 逐专家输出(钩子)与榜单/NLL/rms(出口)同一次前向。不写全词表 logits
         * (那是 517 KB/位置, 统一内存上写它就是掏 GPU 内存)。--reuse-tables 时表已在盘上,
         * 钩子取完就停车, 省掉出口那一段。
         * --verify-tabs: 挂着候选只出表不取料(gatea 的真前向验证), 表名 = 清单里的名字加前缀。 */
        char vt[1200], vr[1200], vn[1300];
        if (c->verify_pref) {
            sft_pref_path(it[m].top, c->verify_pref, vt, sizeof vt);
            sft_pref_path(it[m].rms, c->verify_pref, vr, sizeof vr);
            snprintf(vn, sizeof vn, "%.1199s.nll", vt);
            ds4_engine_v41_set_score_aux(vn, vt, 64, vr, 1);
            ds4_engine_v41_set_moe_hook(NULL, NULL);   /* 只要出口的表, 不取料 */
        } else if (!c->reuse_tab) {
            ds4_engine_v41_set_score_aux(it[m].nll, it[m].top, 64, it[m].rms, 1);
        }
        const double tf = now_s();
        const int frc = ds4_engine_v41_score_ids(e, ids, nids, "/dev/null", no_engram, 0);
        ds4_engine_v41_set_score_split(0);
        ds4_engine_v41_set_moe_hook(NULL, NULL);
        ds4_engine_v41_set_score_aux(NULL, NULL, 0, NULL, 0);
        if (c->verify_pref) {   /* 验证模式: 出完表就下一条, 不累积不解算 */
            free(ids);
            if (frc != 0) { fprintf(stderr, "★第 %d 条验证前向失败 rc=%d★\n", m + 1, frc); rc = -1; break; }
            printf("  [验证 %d/%d] %s (%.0fs)\n", m + 1, nit, vt, now_s() - tf);
            fflush(stdout);
            continue;
        }
        c->t_fwd += now_s() - tf;
        free(ids);
        /* CED 块不过 L39(没有钩子行), 所以只要求"提示最后一位起"全部取到; 末个提示块跑满解码器, 会多给几行, 不少就行 */
        const int need_rows = split ? nids - (split - 1) : nids;
        if (frc != 0 || c->got < need_rows) { fprintf(stderr, "★第 %d 条取料失败(rc=%d, %d 行 < 需要 %d)★\n", m + 1, frc, c->got, need_rows); rc = -1; break; }
        ds4_etgd tk;
        if (ds4_etgd_read(it[m].top, &tk)) { rc = -1; break; }
        if (tk.n < nids) { fprintf(stderr, "★第 %d 条 top-K 表 %d 行 < ids %d 行 —— 不是同一条序列的产物★\n", m + 1, tk.n, nids); ds4_etgd_free(&tk); rc = -1; break; }
        float *inv = sft_rms_read(it[m].rms, nids);
        int na = 0; int *alt = NULL;
        if (strcmp(it[m].alt, "-")) alt = sft_ids_read(it[m].alt, &na);
        if (!inv || (!alt && strcmp(it[m].alt, "-"))) { ds4_etgd_free(&tk); free(inv); free(alt); rc = -1; break; }
        if (!inited) {   /* nu 要等第一条取完才知道(V4.1 = 6), 所以缓冲在这里才配 */
            if (mg_acc_alloc(&ab, tot_dec + tot_ctr, c->n_used, c->D, tk.K)) { ds4_etgd_free(&tk); free(inv); free(alt); rc = -1; break; }
            inited = 1;
        }
        int nc = 0, nd = 0;
        int *cr = rows_spec_parse(it[m].ctr, nids, &nc), *dr = rows_spec_parse(it[m].dec, nids, &nd);
        if (!cr || !dr) { ds4_etgd_free(&tk); free(inv); free(alt); free(cr); free(dr); rc = -1; break; }
        int used_d = 0, used_c = 0;
        for (int q = 0; q < nd + nc && rc == 0; q++) {
            const int is_dec = q < nd, r = is_dec ? dr[q] : cr[q - nd];
            /* 决策行的"另一版 token" = 错版序列在 r+1 的 token —— 与 tgt 同一个错开一格的口径 */
            const int w = (is_dec && alt && r + 1 < na) ? alt[r + 1] : -1;
            int ta = -1, tb = -1; float m0 = 0.f;
            if (!mg_row_pair(&tk, r, is_dec ? w : -1, &ta, &tb, &m0)) continue;
            /* 决策行按到达顺序 3:1 分进拟合段与 val 段(不用随机: 同一份清单解两次必须出同一个东西) */
            int slot;
            if (is_dec) { slot = ((dseq++ % 4) == 3 && wv < tot_dec) ? wv++ : wf++; used_d++; }
            else { slot = wc++; used_c++; }
            if (slot < 0 || slot >= ab.cap) { fprintf(stderr, "★累积槽位 %d 超预算 %d★\n", slot, ab.cap); rc = -1; break; }
            const int nu = ab.nu, D = ab.D;
            memcpy(ab.ye + (size_t)slot * nu * D, c->rye + (size_t)r * nu * D, (size_t)nu * D * 4);
            memcpy(ab.xh + (size_t)slot * D, c->rx + (size_t)r * D, (size_t)D * 4);
            memcpy(ab.rw + (size_t)slot * nu, c->rrw + (size_t)r * nu, (size_t)nu * 4);
            memcpy(ab.sel + (size_t)slot * nu, c->rsel + (size_t)r * nu, (size_t)nu * 4);
            ab.alpha[slot] = c->ralpha[r]; ab.inv[slot] = inv[r];
            ab.ta[slot] = ta; ab.tb[slot] = tb; ab.m0[slot] = m0; ab.smp[slot] = m; ab.rowid[slot] = r;
            /* 整张榜(id + log p)与"该赢的那个" —— 主动集扫榜要它。log 在这里取一次:
             * 后面比的全是 log 域的差, 而 p 的动态范围跨十几个数量级。 */
            /* "该赢的那个"的 log p: 决策行的对版 token 可能掉出 top-K, 那时只有 tgt_p 这一个来源;
             * 在榜内的话下面的循环会用榜上那份覆盖它(同一个数, 口径统一)。 */
            ab.repw[slot] = 1.0f;   /* 真值在 mg_pair_rep 里按 (a,b) 跨条统计后覆盖 */
            ab.want[slot] = ta;
            ab.wlp[slot] = (is_dec && tk.tgt_p[r] > 0.f) ? logf(tk.tgt_p[r]) : -1e30f;
            for (int z = 0; z < ab.K; z++) {
                const int vid = tk.ids[(size_t)r * tk.K + z];
                const float pv = tk.ps[(size_t)r * tk.K + z];
                ab.tid[(size_t)slot * ab.K + z] = (pv > 0.f) ? vid : -1;
                ab.tlp[(size_t)slot * ab.K + z] = (pv > 0.f) ? logf(pv) : -1e30f;
                if (vid == ta && pv > 0.f) ab.wlp[slot] = logf(pv);
            }
        }
        printf("  [样本 %d/%d] 取 %d 行 → 决策点 %d/%d 可用, 约束行 %d/%d (前向 %.0fs)\n",
               m + 1, nit, nids, used_d, nd, used_c, nc, now_s() - tf);
        fflush(stdout);
        ds4_etgd_free(&tk); free(inv); free(alt); free(cr); free(dr);
    }
    if (c->verify_pref) {   /* 验证趟不解算: 表落盘了就算完 */
        free(it);
        if (inited) mg_acc_free(&ab);
        return rc;
    }
    free(it);
    const int nctr = wc - tot_dec;
    /* 槽位没写满是正常的(有些行取不到可比的两个 token), 但要把"没写过的槽"剔掉: 决策拟合段
     * [wf, ndec_fit) 与 val 段 [wv, tot_dec) 里面是垃圾, 解算照样跑得出一个数。做法是把三段
     * 压实成连续的 [0, wf) + [ndec_fit, wv) + [tot_dec, wc) → 重排到 [0, nfit+nval+nctr)。 */
    /* 门槛是 1 不是 4(2026-09-23 改): 当天复盘只有一条事实(大盘方向)时决策点就 1 个, 解它正是"拿当天复盘训练"
     * 的本意。代价如实: val 段为空、尺 L 单折无决策方程(零解, 读数 = 基线), 这时候选只能看拟合与约束行保住率,
     * 结论以挂 ③ 重跑原请求(reviewrun 段)为准。 */
    if (rc == 0 && (!inited || wf < 1)) { fprintf(stderr, "★可用决策点 %d 个, 不解★\n", wf); rc = -1; }
    if (rc) { if (inited) mg_acc_free(&ab); return -1; }
    {
        const int nu = ab.nu, D = ab.D;
        int dst = wf;
        for (int src = ndec_fit; src < wv; src++, dst++) mg_acc_move(&ab, dst, src);
        const int nvfin = wv - ndec_fit;
        for (int src = tot_dec; src < wc; src++, dst++) mg_acc_move(&ab, dst, src);
        ab.n = dst;
        printf("[后训练] 累积 %d 行: 决策点 拟合 %d / val %d, 约束 %d; ye 缓冲 %.2f GB\n",
               ab.n, wf, nvfin, nctr, (double)ab.n * nu * D * 4 / 1e9);
        const int grc = mg_solve_grid(c, &ab, il, out, wf, wf + nvfin);
        mg_acc_free(&ab);
        return grc;
    }
}
