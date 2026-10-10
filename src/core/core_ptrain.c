/* core_ptrain.c — 后训练 ③ 的主循环: --ptrain <配置>(2026-10-01; 10-10 收口)。总述见 core_ptrain.h。
 *
 * 一趟 = 读料(jsonl) → 教师 top-K(有缓存就读; 不带材料的题不要教师) → 第 0 步评估(部署态, ③ 恒等) → 若干轮: 打乱训练题、每 batch 题攒梯度一次 Adam
 * → 每轮末评估 + 贪心探针(留出题) + 存 ckpt_eNN → 落 ③(amp_Lnn.bin + base.fnv, 引擎 --posttrain 直接挂)。
 * 判据(全打印, 判读归人): ①留出题答案位损失(带材料 = KL 教师‖学生, 不带 = NLL)较第 0 步降多少 —— 措辞没进训练, 降 = 学到了内容而不是背了题;
 * ②同一道留出题的贪心回答: 部署态 / 挂 ③ / 参考答案三份原样并排; ③守门(wt2 五指标)在脚本里另跑(docgate 段)。
 * 轮数照配置训满(10-02 用户: "支持设置训练轮数, 然后选出最优就行了"), 挑哪一轮归脚本 kdpick 段(过 wt2 门的轮里留出损失最低者);
 * 每轮末另打一行"留出损失较上轮逐题配对降多少 ± 标准误"(见 pt_paired_gain), 只打不判 —— 末轮还在显著降 = 下次轮数可以给多些。 */
#include "core_ptrain.h"
#include <sys/stat.h>
#ifndef DS4_NO_GPU

/* 可用内存(MB, 读 /proc/meminfo; 没有就 -1): 全层训练只剩 ~3 GB 余量, 每段都打出来, 看门狗(2.5 GB)杀之前能看见是哪一段在吃 */
static long pt_mem_mb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char ln[256]; long kb = -1;
    while (fgets(ln, sizeof ln, f)) if (sscanf(ln, "MemAvailable: %ld kB", &kb) == 1) break;
    fclose(f);
    return kb >= 0 ? kb / 1024 : -1;
}
static void pt_release(const char *after) {   /* 阶段之间放掉按需长的暂存槽(见 ds4_gpu_v41_scratch_release) */
    const uint64_t b = ds4_gpu_v41_scratch_release();
    fprintf(stderr, "ds4: [ptrain] %s后放掉暂存 %.0f MB, 可用内存 %ld MB\n", after, (double)b / 1048576.0, pt_mem_mb());
}

#define PT_MEM_FLOOR_MB 3000   /* 开训前预热后的可用内存下限(见 ds4_engine_ptrain 的预热段) */

/* prof 累计表(秒/题; Adam 与每步墙钟按秒/步): 各段之和对每题墙钟 —— 差额 = 段外开销。为什么要整步一张表: 10-02 实测每题 7.75 s,
 * 而以前只量逐层反传四段(+ 前向估计)合计 3.4 s, 剩下 ~4 s 看不见在哪(铁律: 速度问题先出逐段表再照最大项动刀) */
static void pt_prof_print(const pt_run *r, const char *tag) {
    if (!r->tm_n) return;
    const double n = (double)r->tm_n, ns = r->tm_steps ? (double)r->tm_steps : 1.0, *t = r->tm;
    const double seg = t[PT_TM_FWD] + t[PT_TM_LOSS] + t[PT_TM_EXIT] + t[PT_TM_LAYERS];
    fprintf(stderr, "ds4: [ptrain prof %s] %u 题 / %u 步(秒/题): 前向 %.3f | 损失 %.3f | 出口反传 %.3f | 逐层反传 %.3f = 重算 %.3f + routed 专家反向 %.3f"
            " + 注意力半层 %.3f + 层内其余 %.3f + 层间 %.3f | 段和 %.3f vs 每题墙钟 %.3f | Adam+范数 %.3f 秒/步 | 每步墙钟 %.3f 秒\n",
            tag, r->tm_n, r->tm_steps, t[PT_TM_FWD] / n, t[PT_TM_LOSS] / n, t[PT_TM_EXIT] / n, t[PT_TM_LAYERS] / n,
            t[PT_TM_RECOMP] / n, t[PT_TM_ROUTED] / n, t[PT_TM_ATTN] / n, t[PT_TM_LREST] / n,
            (t[PT_TM_LAYERS] - t[PT_TM_RECOMP] - t[PT_TM_ROUTED] - t[PT_TM_ATTN] - t[PT_TM_LREST]) / n,
            seg / n, t[PT_TM_Q] / n, t[PT_TM_ADAM] / ns, t[PT_TM_STEP] / ns);
}

static bool pt_in_eval(const pt_data *d, const pt_sample *s, int which, int hold) {
    return s->eval == which && s->top_id && d->ch[s->chunk].hold == hold;
}

/* 评估: 留出题(eval=1)或训练题前若干道的平均 KL(nat/答案 token)。hold: 0 = 复盘题, 1 = 保持料(通用题, 教师 = 部署态), 分开报。
 * li/ni 非空 = 逐题记 KL 和 / 答案 token 数(第 k 道入选题进第 k 格, 顺序与 pt_heldout_dump 同): 轮间逐题配对比用 */
static bool pt_eval(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d, int which, uint32_t cap_n, double *kl, int hold,
                    double *li, uint32_t *ni) {
    const pt_sample **ss = xmalloc((size_t)(d->ns + 1) * sizeof *ss);
    uint32_t k = 0, nt = 0;
    for (uint32_t i = 0; i < d->ns && k < cap_n; i++) if (pt_in_eval(d, &d->s[i], which, hold)) ss[k++] = &d->s[i];
    double ls = 0.0;
    const bool ok = pt_run_list(e, c, r, ss, k, 0, &ls, &nt, li, ni);   /* 合批装包; li/ni 仍按入选顺序逐题记 */
    free(ss);
    *kl = nt ? ls / nt : 0.0;
    return ok;
}

/* ★轮间逐题配对比★: 降幅 D = Σ(上轮_i − 本轮_i) / Σm_i, 与留出 KL 同口径(按答案 token 加权)。标准误把"题"当独立单位 ——
 * 同一题的各 token 高度相关, 按 token 算会把误差低估好几倍; 比值估计线性化: SE = sqrt(Q/(Q−1)·Σ(δ_i − D·m_i)²) / Σm_i。
 * 两轮用的是同一批题、同一串教师答案, 配对把"题有难有易"这部分波动消掉, 只剩这一轮训练带来的变化。 */
static void pt_paired_gain(const double *prev, const double *cur, const uint32_t *m, uint32_t q, double *gain, double *se) {
    double sd = 0.0, sm = 0.0, ss = 0.0;
    for (uint32_t i = 0; i < q; i++) { sd += prev[i] - cur[i]; sm += m[i]; }
    const double g = sm > 0 ? sd / sm : 0.0;
    for (uint32_t i = 0; i < q; i++) { const double u = prev[i] - cur[i] - g * m[i]; ss += u * u; }
    *gain = g;
    *se = (q > 1 && sm > 0) ? sqrt(ss * q / (q - 1)) / sm : 0.0;
}

/* 逐题留出 KL 落盘(out/heldout_eNN.tsv: 块名 / 样本号 / 答案 token 数 / KL 和): 事后任意两轮配对比、按块看哪块学会了又被挤掉
 * (10-02 全量: 601069、603881 的探针第 1 轮答对、第 3 轮答错, 只有 12 道探针时说不清是挤掉还是探针抽样) */
static void pt_heldout_dump(const pt_cfg *c, const pt_data *d, const double *li, const uint32_t *ni, const char *tag) {
    char p[1200]; snprintf(p, sizeof p, "%s/heldout_%s.tsv", c->out, tag);
    FILE *f = fopen(p, "w");
    if (!f) { fprintf(stderr, "ds4: [ptrain] 写不了 %s\n", p); return; }
    for (uint32_t i = 0, k = 0; i < d->ns; i++)
        if (pt_in_eval(d, &d->s[i], 1, 0)) { fprintf(f, "%s\t%u\t%u\t%.6f\n", d->ch[d->s[i].chunk].name, i, ni[k], li[k]); k++; }
    fclose(f);
}

/* 梯度检查 / hc 检查用的题: 第一道【非保持料】的训练题。保持料的教师 = 部署态自己, A = 0 时学生与教师逐位相同 ⇒ 损失 0、梯度 ~0,
 * 比值全是噪声(10-02 实撞: 全量清单保持料排第一行, L0..L39 比值 -2996 / 1400 / -317 / 4.07, 初始损失 0.00000)。 */
static const pt_sample *pt_check_sample(const pt_data *d) {
    for (uint32_t i = 0; i < d->ns; i++)
        if (!d->s[i].eval && d->s[i].top_id && !d->ch[d->s[i].chunk].hold) return &d->s[i];
    fprintf(stderr, "ds4: [ptrain] 检查要一道非保持料的训练题, 清单里没有\n");
    return NULL;
}

/* ★梯度检查★: 反传写对没有, 只认有限差分。取第一道非保持料训练题, 每个训练层的 A 沿它自己的梯度方向 D = g/|g| 走 ±ε,
 * (L(A+εD) − L(A−εD))/2ε 应等于 |g|(L = 本题平均 KL / batch, 与梯度同一个尺度)。ε 太小会被前向的 bf16 舍入淹没(y 每次放大器之后舍 bf16),
 * 太大进入非线性区 —— 所以扫四档, 看中间几档比值是否 ≈ 1。比值远离 1 或符号反 = 反传有错, 不许开训。 */
static bool pt_gradcheck(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d) {
    const pt_sample *s = pt_check_sample(d);
    if (!s) return false;
    const uint64_t nel = (uint64_t)r->K * DS4_N_EMBD;
    const double norm = 1.0 / ((double)s->m * (double)c->batch);
    bool ok = true;
    ds4_gpu_tensor *D = v41_alloc(nel * 4, &ok);
    for (uint32_t il = c->layer_lo; ok && il <= c->layer_hi; il++) {
        bool want = c->ngcl == 0;
        for (uint32_t q = 0; q < c->ngcl; q++) if (c->gcl[q] == il) want = true;
        if (!want) continue;
        double l0 = 0; uint32_t nt = 0, nt2 = 0;
        ok = ds4_gpu_tensor_fill_f32(r->gA[il], 0.f, nel) && ds4_gpu_tensor_fill_f32(r->gB[il], 0.f, nel) && pt_step_sample(e, c, r, s, 1, &l0, &nt);
        double ss = 0;
        if (ok) ok = ds4_gpu_bwd_sumsq_tensor(r->gA[il], nel, &ss);
        const double gn = sqrt(ss);
        if (!ok || gn <= 0) { fprintf(stderr, "ds4: [ptrain 梯度检查] L%u |gA| = %g, 查不了\n", il, gn); break; }
        ok = ds4_gpu_tensor_fill_f32(D, 0.f, nel) && ds4_gpu_bwd_axpy_tensor(D, r->gA[il], (float)(1.0 / gn), nel);
        /* 1e-4/1e-3 两档在全层检查里纯是舍入噪声(y 舍 bf16 后小扰动被整个吃掉), 换成 0.3 / 1 两档看比值随 ε 的走向 */
        const float eps[4] = { 1e-2f, 1e-1f, 3e-1f, 1.f };
        /* 扰动的两趟走冻结选择前向(路由 top-6 与 indexer 选组照上面这趟基准前向存的): 反传把选择当常量, 量的也该是同一个函数;
         * 不冻的话下游几十层总有并列边被翻, 一次 0.005~0.01 的损失跳变就盖过 ε·|g|(10-01 全层检查实撞)。仅多层模式有存档。 */
        r->save.replay = r->lb.ready ? 1 : 0;
        for (int q = 0; ok && q < 4; q++) {
            double lp = 0, lm = 0;
            ok = ds4_gpu_bwd_axpy_tensor(r->st.ampA[il], D, eps[q], nel) && pt_step_sample(e, c, r, s, 0, &lp, &nt2) &&
                 ds4_gpu_bwd_axpy_tensor(r->st.ampA[il], D, -2.f * eps[q], nel) && pt_step_sample(e, c, r, s, 0, &lm, &nt2) &&
                 ds4_gpu_bwd_axpy_tensor(r->st.ampA[il], D, eps[q], nel);
            const double fd = (lp - lm) * norm / (2.0 * eps[q]);
            fprintf(stderr, "ds4: [ptrain 梯度检查] L%u ε=%.0e: 有限差分 %.6g / 反传 %.6g = %.4f (L0 %.5f L+ %.5f L− %.5f)\n",
                    il, (double)eps[q], fd, gn, fd / gn, l0 * norm, lp * norm, lm * norm);
        }
        r->save.replay = 0;
        if (ok) ok = ds4_gpu_tensor_fill_f32(r->gA[il], 0.f, nel) && ds4_gpu_tensor_fill_f32(r->gB[il], 0.f, nel);
    }
    r->save.replay = 0;
    if (D) ds4_gpu_tensor_free(D);
    return ok;
}

/* ★hc 边界检查(逐层定位)★: 在第 k 层入口沿该处的解析梯度方向 D = g/|g| 扰动 hc(f32, 加在层入口、存档之前), 冻结选择前向量损失,
 * (L+ − L−)/2ε 对 |g|。它与放大器梯度检查的区别: 放大器那个只告诉你"第 k 层的放大器梯度偏了", 偏在哪一层的反传说不清;
 * 这个量的是"第 k 层入口之上整段反传", 相邻两层一比, 比值从 ≈1 掉下去的那一层就是反传写错的那层。
 * ε 扫四档: 残差流每层出口舍 bf16, 深层 hc 量级上百(bf16 一格 0.5 起), 太小的扰动大半被舍掉, 只有中间几档有信号。 */
static bool pt_hccheck(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d) {
    const pt_sample *s = pt_check_sample(d);
    if (!s || !r->lb.ready) return false;
    const uint64_t nel = (uint64_t)s->sn * DS4_N_HC * DS4_N_EMBD;
    const double norm = 1.0 / ((double)s->m * (double)c->batch);
    bool ok = true;
    ds4_gpu_tensor *D = v41_alloc(nel * 4, &ok);
    r->hcap = v41_alloc(nel * 4, &ok);
    for (uint32_t q = 0; ok && q < c->nhcl; q++) {
        const uint32_t k = c->hcl[q];
        if (k < c->layer_lo || k > c->layer_hi) { fprintf(stderr, "ds4: [ptrain hc 检查] L%u 不在训练段 L%u..L%u 里, 跳过\n", k, c->layer_lo, c->layer_hi); continue; }
        const int mid = c->hcmid[q] ? 1 : 0;
        double l0 = 0, ss = 0; uint32_t nt = 0, nt2 = 0;
        r->hcap_layer = (int32_t)k; r->hcap_mid = mid;
        ok = pt_step_sample(e, c, r, s, 1, &l0, &nt) && ds4_gpu_bwd_sumsq_tensor(r->hcap, nel, &ss);
        r->hcap_layer = -1;
        const double gn = sqrt(ss);
        if (!ok || gn <= 0) { fprintf(stderr, "ds4: [ptrain hc 检查] L%u |g| = %g, 查不了\n", k, gn); break; }
        ok = ds4_gpu_tensor_fill_f32(D, 0.f, nel) && ds4_gpu_bwd_axpy_tensor(D, r->hcap, (float)(1.0 / gn), nel);
        r->save.hpert = D; r->save.hpert_layer = (int32_t)k; r->save.hpert_mid = mid; r->save.replay = 1;
        const float eps[4] = { 1e-1f, 1.f, 1e1f, 1e2f };
        for (int qe = 0; ok && qe < 4; qe++) {
            double lp = 0, lm = 0;
            r->save.hpert_eps = eps[qe];  ok = pt_step_sample(e, c, r, s, 0, &lp, &nt2);
            r->save.hpert_eps = -eps[qe]; if (ok) ok = pt_step_sample(e, c, r, s, 0, &lm, &nt2);
            const double fd = (lp - lm) * norm / (2.0 * eps[qe]);
            fprintf(stderr, "ds4: [ptrain hc 检查] L%02u %s ε=%.0e: 有限差分 %.6g / 反传 %.6g = %.4f (L0 %.5f L+ %.5f L− %.5f)\n",
                    k, mid ? "中段" : "入口", (double)eps[qe], fd, gn, fd / gn, l0 * norm, lp * norm, lm * norm);
        }
        r->save.hpert = NULL; r->save.hpert_layer = -1; r->save.hpert_mid = 0; r->save.replay = 0; r->hcap_mid = 0;
        for (uint32_t il = c->layer_lo; ok && il <= c->layer_hi; il++)   /* 基准那趟攒进来的放大器梯度清掉 */
            ok = ds4_gpu_tensor_fill_f32(r->gA[il], 0.f, (uint64_t)r->K * DS4_N_EMBD) && ds4_gpu_tensor_fill_f32(r->gB[il], 0.f, (uint64_t)r->K * DS4_N_EMBD);
    }
    if (D) ds4_gpu_tensor_free(D);
    if (r->hcap) { ds4_gpu_tensor_free(r->hcap); r->hcap = NULL; }
    return ok;
}

int ds4_engine_ptrain(ds4_engine *e, const char *spec) {
    if (!e || !spec || !ds4_engine_is_v41(e) || !e->metal_ready) { fprintf(stderr, "ds4: --ptrain 要 V4.1 模型 + GPU 后端\n"); return 1; }
    pt_cfg c;
    if (!pt_cfg_load(&c, spec)) return 1;
    mkdir(c.out, 0755);
    if (g_ds4_v41_pt_dir) { fprintf(stderr, "ds4: --ptrain 不接 --posttrain(训练的就是 ③; 起点恒为 ①+②)\n"); return 1; }
    pt_data d;
    if (!pt_data_load(e, &c, &d)) { pt_data_free(&d); return 1; }
    if (!pt_teacher(e, &c, &d)) { pt_data_free(&d); return 1; }
    pt_release("教师表");
    pt_run r;
    if (!pt_run_alloc(e, &c, &r)) { pt_data_free(&d); return 1; }
    fprintf(stderr, "ds4: [ptrain] 训练缓冲分配后可用内存 %ld MB\n", pt_mem_mb());
    uint32_t *ord = xmalloc((size_t)d.ns * 4), ntr = 0, nev = 0;
    const pt_sample **bs = xmalloc((size_t)(c.batch + 1) * sizeof *bs);   /* 一批的题(装包用) */
    for (uint32_t i = 0; i < d.ns; i++) if (!d.s[i].eval && d.s[i].top_id) ord[ntr++] = i;
    for (uint32_t i = 0; i < d.ns; i++) if (pt_in_eval(&d, &d.s[i], 1, 0)) nev++;
    /* 逐题留出 KL: 上一轮 / 本轮 / 答案 token 数(各轮同一批题, token 数不变) */
    double *ev_prev = xmalloc((size_t)(nev + 1) * 8), *ev_cur = xmalloc((size_t)(nev + 1) * 8);
    uint32_t *ev_m = xmalloc((size_t)(nev + 1) * 4);
    char **base = xmalloc_zeroed(c.probe_n + PT_PROBE_HOLD, sizeof(char *));
    char ck[1100];   /* 每轮末的 ③ 各存一份 ckpt_eNN(一份 19 层 50 MB): 守门挑轮次用, 不让下一轮覆盖 */
    char lp[1100]; snprintf(lp, sizeof lp, "%s/train.log", c.out);
    FILE *lf = fopen(lp, "a");
    int rc = 1;
    double kl_ev0 = 0, kl_tr0 = 0, kl = 0, kl_h0 = 0, kh = 0;
    uint64_t rs = 0x2545F4914F6CDD1Dull ^ c.seed;
    uint32_t step = 0;
    double t0 = now_sec();
    /* 逐位诊断: 只读不训, 不要预热(预热会把一道训练题的梯度算进缓冲, 诊断用不上); 它会把 A 置零, 所以跑完只能退出 */
    if (c.diag) { rc = pt_diag(e, &c, &r, &d) ? 0 : 1; goto out; }
    {   /* ★预热探底★: 最满的包(不合批时 = 最长的训练题)完整走一趟前向 + 反传, 按需长的暂存一次长到顶, 内存峰值在开训前就量出来 ——
         * 10-02 第一次全量全层训练在第 0 步之后被看门狗(2.5 GB 连续两次)杀掉, 那时已经花了 38 分钟。
         * 余量低于 PT_MEM_FLOOR_MB 直接停车: 看门狗 2500 MB 动手, 多留 500 MB 给评估/探针的零碎。 */
        uint32_t lmax = 0;
        if (!pt_pack_warm(e, &c, &r, &d, &lmax)) goto out;
        for (uint32_t il = c.layer_lo; il <= c.layer_hi; il++)   /* 预热这趟的梯度不算数 */
            if (!ds4_gpu_tensor_fill_f32(r.gA[il], 0.f, (uint64_t)r.K * DS4_N_EMBD) || !ds4_gpu_tensor_fill_f32(r.gB[il], 0.f, (uint64_t)r.K * DS4_N_EMBD)) goto out;
        const long mem = pt_mem_mb();
        fprintf(stderr, "ds4: [ptrain] 预热(最满一包 %u 行前向 + 反传)后可用内存 %ld MB, 暂存 %.0f MB\n", lmax, mem, (double)ds4_gpu_v41_scratch_bytes() / 1048576.0);
        if (mem >= 0 && mem < PT_MEM_FLOOR_MB) {
            fprintf(stderr, "ds4: ★[ptrain] 可用内存 %ld MB 低于 %d MB, 训练中途会撞看门狗 —— 停车(降 maxlen 或减训练层)★\n", mem, PT_MEM_FLOOR_MB);
            pt_release("预热(停车前, 看暂存大头)");
            goto out;
        }
        pt_release("预热");
    }
    memset(r.tm, 0, sizeof r.tm); r.tm_n = r.tm_steps = 0;   /* 预热那题是冷启动(暂存一次长到顶), 不进计时 */
    /* 检查(一次装载可串着跑): 合批对账 → 重算对拍 → hc 边界检查 → 放大器梯度检查。除 gradcheck=1(查完接着训)外都是"查完就退出" */
    bool only_check = false;
    if (c.packcheck) { if (!pt_pack_check(e, &c, &r, &d)) goto out; only_check = true; }
    if (c.rccheck) {   /* 重算对拍: 第一道训练题前向 + 反传一次(层里打对拍行) */
        double l0 = 0; uint32_t nt = 0;
        if (!ntr || !r.lb.ready || !pt_step_sample(e, &c, &r, &d.s[ord[0]], 1, &l0, &nt)) goto out;
        for (uint32_t il = c.layer_lo; il <= c.layer_hi; il++)   /* 这一趟攒进来的放大器梯度不算数 */
            if (!ds4_gpu_tensor_fill_f32(r.gA[il], 0.f, (uint64_t)r.K * DS4_N_EMBD) || !ds4_gpu_tensor_fill_f32(r.gB[il], 0.f, (uint64_t)r.K * DS4_N_EMBD)) goto out;
        c.rccheck = 0;   /* 只对拍这一趟, 后面的检查/训练不再打 */
        only_check = true;
    }
    if (c.nhcl) { if (!pt_hccheck(e, &c, &r, &d)) goto out; only_check = true; }
    if (c.gradcheck) {
        if (!pt_gradcheck(e, &c, &r, &d)) goto out;
        if (c.gradcheck >= 2) only_check = true;
    }
    if (only_check) { rc = 0; goto out; }
    fprintf(stderr, "ds4: [ptrain] 训练 L%u-L%u 放大器 K=%u, lr %.2g, batch %u, %u 轮, 训练题 %u; 第 0 步评估…\n",
            c.layer_lo, c.layer_hi, c.rank, (double)c.lr, c.batch, c.epochs, ntr);
    if (c.max_steps) fprintf(stderr, "ds4: [ptrain] max_steps=%u: 计时模式, 不做第 0 步评估/探针, 到步数就退出, 不评估不存盘\n", c.max_steps);
    else {
        if (!pt_eval(e, &c, &r, &d, 1, UINT32_MAX, &kl_ev0, 0, ev_prev, ev_m) || !pt_eval(e, &c, &r, &d, 0, 64, &kl_tr0, 0, NULL, NULL) ||
            !pt_eval(e, &c, &r, &d, 1, UINT32_MAX, &kl_h0, 1, NULL, NULL)) goto out;
        fprintf(stderr, "ds4: [ptrain] 第 0 步(%s): 留出损失 %.4f(%u 题) / 训练前 64 题 %.4f / 保持料留出 KL %.4f\n",
                c.init[0] ? "起点 = init 的 ③" : "部署态", kl_ev0, nev, kl_tr0, kl_h0);
        if (lf) { fprintf(lf, "step 0 eval_kl %.5f train_kl %.5f hold_kl %.5f\n", kl_ev0, kl_tr0, kl_h0); fflush(lf); }
        pt_heldout_dump(&c, &d, ev_prev, ev_m, "e00");
        pt_release("第 0 步评估");   /* 探针走解码路, 它的暂存别叠在评估/训练暂存上面 */
        /* 给了 init 时第 0 步的"挂 ③"= 起点那份 ③(续训: 起点对留出题答什么, 与部署态并排); 没给就只有部署态 */
        if (!pt_probes(e, &c, &d, base, c.init[0] ? c.init : NULL, "e00")) goto out;
        pt_release("第 0 步探针");
    }
    /* ★每轮按 token 预算抽题(10-03 用户定)★: epoch_tok > 0 时一轮 = 打乱序里连续的一段, 各题行数累计到预算就收, 下一轮从游标接着取,
     * 取完整个打乱序才重新打乱 —— 一个周期内每题恰好一次、不放回; 轮与轮之间是相邻切片, 不是同一批题。预算 0 = 老口径(一轮 = 全量过一遍)。 */
    uint64_t rows_all = 0;
    for (uint32_t i = 0; i < ntr; i++) rows_all += (uint64_t)d.s[ord[i]].sn;
    fprintf(stderr, "ds4: [ptrain] 训练料 %u 题共 %llu 行(含提示)%s", ntr, (unsigned long long)rows_all, c.epoch_tok ? "" : ", 一轮 = 全量过一遍\n");
    if (c.epoch_tok) fprintf(stderr, "; 每轮 token 预算 %u ⇒ 约 %.1f 轮过一遍, 轮末\"留出 KL 较上轮降\"比的是相邻切片\n", c.epoch_tok, (double)rows_all / (double)c.epoch_tok);
    uint32_t cur = ntr;   /* 打乱序的游标: 到头就重新打乱 */
    uint64_t qall = 0;    /* 开训以来过了多少题(算覆盖了几遍) */
    t0 = now_sec();
    for (uint32_t ep = 1; ep <= c.epochs; ep++) {
        uint64_t erows = 0; uint32_t eq = 0;   /* 本轮已过的行 / 题 */
        for (;;) {
            if (cur >= ntr) {
                if (!c.epoch_tok && eq) break;   /* 老口径: 全量过一遍就是一轮 */
                for (uint32_t i = ntr; i > 1; i--) {   /* Fisher–Yates, 种子固定 */
                    rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27;
                    const uint32_t j = (uint32_t)((rs * 2685821657736338717ull) % i);
                    const uint32_t t = ord[i - 1]; ord[i - 1] = ord[j]; ord[j] = t;
                }
                cur = 0;
            }
            double ls = 0; uint32_t nt = 0;
            const uint32_t b0 = cur, b1 = cur + c.batch < ntr ? cur + c.batch : ntr;   /* 一批 = c.batch 道题 */
            const double s0 = pt_tick(&c);
            for (uint32_t b = b0; b < b1; b++) { bs[b - b0] = &d.s[ord[b]]; erows += (uint64_t)d.s[ord[b]].sn; }
            cur = b1; eq += b1 - b0;
            const bool last = c.epoch_tok ? erows >= c.epoch_tok : cur >= ntr;   /* 本轮最后一批 */
            if (!pt_run_list(e, &c, &r, bs, b1 - b0, 1, &ls, &nt, NULL, NULL)) goto out;   /* 一批的题装包后逐包过, 梯度照旧攒到批末 */
            const double s1 = pt_tick(&c);
            double ss = 0;
            for (uint32_t il = c.layer_lo; il <= c.layer_hi; il++) {
                double a = 0, bb = 0;
                if (!ds4_gpu_bwd_sumsq_tensor(r.gA[il], (uint64_t)r.K * DS4_N_EMBD, &a) || !ds4_gpu_bwd_sumsq_tensor(r.gB[il], (uint64_t)r.K * DS4_N_EMBD, &bb)) goto out;
                ss += a + bb;
            }
            const double gn = sqrt(ss);
            const float gs = (c.clip > 0.f && gn > c.clip) ? (float)(c.clip / gn) : 1.0f;   /* 全局范数裁剪(标准配方, 防单批炸步) */
            step++;
            for (uint32_t il = c.layer_lo; il <= c.layer_hi; il++)
                if (!ds4_gpu_bwd_adam_tensor(r.st.ampA[il], r.gA[il], r.mA[il], r.vA[il], (uint64_t)r.K * DS4_N_EMBD, c.lr, 0.9f, 0.999f, 1e-8f, gs, step) ||
                    !ds4_gpu_bwd_adam_tensor(r.st.ampB[il], r.gB[il], r.mB[il], r.vB[il], (uint64_t)r.K * DS4_N_EMBD, c.lr, 0.9f, 0.999f, 1e-8f, gs, step)) goto out;
            if (c.prof) {
                const double s2 = pt_tick(&c);
                r.tm[PT_TM_ADAM] += s2 - s1; r.tm[PT_TM_STEP] += s2 - s0; r.tm_steps++;
                if (r.tm_steps % 5u == 0u) pt_prof_print(&r, "累计");
            }
            const long mem = pt_mem_mb();
            if (lf) { fprintf(lf, "step %u ep %u loss %.5f gnorm %.4g t %.0f mem %ld\n", step, ep, nt ? ls / nt : 0.0, gn, now_sec() - t0, mem); fflush(lf); }
            if (step % 10 == 0 || last)
                fprintf(stderr, "ds4: [ptrain] 轮 %u 步 %u: 批 KL %.4f, 梯度范数 %.3g%s, %.0f s, 可用内存 %ld MB(暂存 %.0f MB)\n", ep, step, nt ? ls / nt : 0.0, gn,
                        gs < 1.f ? "(裁剪)" : "", now_sec() - t0, mem, (double)ds4_gpu_v41_scratch_bytes() / 1048576.0);
            if (c.max_steps && (step >= c.max_steps || last)) {   /* 轮先跑完也收: 计时模式不评估不存盘(10-09 实撞: 合并后一轮 63 步 < 67, 白跑了评估/探针/采样 13 分钟) */
                pt_prof_print(&r, "终");
                fprintf(stderr, "ds4: [ptrain] max_steps=%u 到了: 计时模式退出(不评估不存盘, 产物不能挂), 共 %.0f s\n", c.max_steps, now_sec() - t0);
                rc = 0; goto out;
            }
            if (last) break;
        }
        double kt = 0, gain = 0, se = 0;
        if (!pt_eval(e, &c, &r, &d, 1, UINT32_MAX, &kl, 0, ev_cur, ev_m) || !pt_eval(e, &c, &r, &d, 0, 64, &kt, 0, NULL, NULL) ||
            !pt_eval(e, &c, &r, &d, 1, UINT32_MAX, &kh, 1, NULL, NULL)) goto out;
        pt_paired_gain(ev_prev, ev_cur, ev_m, nev, &gain, &se);
        qall += eq;
        fprintf(stderr, "ds4: [ptrain] ★轮 %u 末(本轮 %u 题 %llu 行, 累计覆盖 %.2f 遍): 留出损失 %.4f(第 0 步 %.4f, %+.1f%%) / 训练前 64 题 %.4f(第 0 步 %.4f) / 保持料留出 KL %.4f(第 0 步 %.4f)★\n",
                ep, eq, (unsigned long long)erows, ntr ? (double)qall / (double)ntr : 0.0,
                kl, kl_ev0, kl_ev0 > 0 ? (kl / kl_ev0 - 1.0) * 100.0 : 0.0, kt, kl_tr0, kh, kl_h0);
        fprintf(stderr, "ds4: [ptrain] 轮 %u 留出损失较上轮降 %.4f ± %.4f(%u 题逐题配对, %.1f 个标准误; 不到 2 个 = 这轮的进步分不清是学到了还是题目抽样)\n",
                ep, gain, se, nev, se > 0 ? gain / se : 0.0);
        if (lf) { fprintf(lf, "epoch %u eval_kl %.5f train_kl %.5f hold_kl %.5f gain %.5f se %.5f nq %u rows %llu\n", ep, kl, kt, kh, gain, se, eq, (unsigned long long)erows); fflush(lf); }
        char tag[16]; snprintf(tag, sizeof tag, "e%02u", ep);
        pt_heldout_dump(&c, &d, ev_cur, ev_m, tag);
        snprintf(ck, sizeof ck, "%s/ckpt_e%02u", c.out, ep);
        if (!pt_save_amp(&c, &r, ck)) goto out;
        pt_release("轮末评估");
        if (!pt_probes(e, &c, &d, base, ck, tag)) goto out;
        pt_release("轮末探针");
        double *t = ev_prev; ev_prev = ev_cur; ev_cur = t;
    }
    if (c.epochs) {   /* epochs=0 = 只做第 0 步评估 + 探针, 没训过的 ③ 不落盘(落了只是 init 的一份拷贝) */
        if (!pt_save_amp(&c, &r, c.out)) goto out;
        fprintf(stderr, "ds4: [ptrain] ③ 已落 %s(amp_L%02u..L%02u.bin + base.fnv); 挂法: --zchain ② --posttrain %s\n", c.out, c.layer_lo, c.layer_hi, c.out);
    }
    rc = 0;
out:
    if (lf) fclose(lf);
    for (uint32_t k = 0; k < c.probe_n + PT_PROBE_HOLD; k++) free(base[k]);
    free(base); free(ord); free(bs); free(ev_prev); free(ev_cur); free(ev_m);
    pt_run_free(&r);
    pt_data_free(&d);
    return rc;
}
#else
int ds4_engine_ptrain(ds4_engine *e, const char *spec) { (void)e; (void)spec; return 1; }
#endif /* !DS4_NO_GPU */
