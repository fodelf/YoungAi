/* core_decode_penalty.c — 解码采样的复读惩罚(2026-09-21, 113-1.md §4.2)。
 *
 * 只在请求显式给了参数时才被调用, 默认路(裸 argmax / 纯采样)一次都不进来 —— 铁律"引擎不得改模型输出"。
 * 作用域 = 生成段(不含提示): 引用/摘录提示里的原文永远不被罚(V4 路 repeat_penalize_buf 同一口径)。
 *
 * ① 频率/出现惩罚(OpenAI 语义, 与 V4 路逐字同式): logits[t] −= freq·count(t) + (count(t)>0 ? presence : 0)。
 *    它按 token 计数, 挡得住"同一个词反复出现", 挡不住"整段逐字抄"(抄的每个 token 单独看都只出现过一两次)。
 * ② DRY(序列复读惩罚, llama.cpp 的 Don't Repeat Yourself): 对每个候选 t, 找生成段里"当前后缀 + t"能接上的最长已现序列长 L
 *    (= 以某个更早出现的 t 为终点、往回与当前后缀逐 token 相同的长度), L ≥ allowed 时 logits[t] −= mult·base^(L−allowed)。
 *    抄得越长罚得越狠(指数), 短的合法重复(数字 / 名字 / 术语, L 1~2)不受影响; 序列断点(token 文本含 换行 冒号 引号 星号)把匹配截断,
 *    列表 / 段落这种结构性的重复不被当成抄。★温 0 下也生效★(改的是 logits, argmax 照样看得见) ⇒ "确定性直出 + 不死循环"只有它。
 *
 * 为什么是这个算法而不是 z-algorithm: n ≤ 几千、正常文本 L 个位数, 一步几万次比较可忽略; 复读区 L 会涨到几百, 那正是要罚死它的时候,
 * 匹配上限 DRY_MAX_L 一封, 最坏也只是 n × 512 次比较(≈ 0.5 ms), 而且惩罚早就把那个 token 压到 −1e15 了。
 * 出错会怎样: 断点表给错(全 0)= 段落边界也接着算匹配, 列表格式会被误罚, 输出格式塌; 给成全 1 = 没有任何匹配, DRY 静默失效。 */
#include "core_internal.h"

#define DRY_MAX_L 512u          /* 往回匹配上限; base^(512) 早已是天文数字, 再长没有意义 */
#define DRY_MAX_EXP 64          /* 指数封顶: 1.75^64 ≈ 3e15, 有限但等于禁用该 token; 不封会算成 inf */

void ds4_decode_penalize(float *logits, uint32_t n_vocab, const int32_t *gen, uint32_t n, const uint8_t *breaker,
                         float freq, float presence, float dry_mult, float dry_base, int dry_allowed) {
    if (!logits || !gen || n == 0) return;
    if (freq != 0.f || presence != 0.f) {
        /* 与 V4 路同一个两遍走法: 第一遍每次出现都扣 freq、首次出现再扣 presence, 第二遍把标记擦掉(O(n), 不碰整个词表) */
        static uint8_t *seen = NULL;
        if (!seen) seen = xcalloc((size_t)n_vocab, 1);
        for (uint32_t i = 0; i < n; i++) {
            const int32_t t = gen[i];
            if (t < 0 || (uint32_t)t >= n_vocab) continue;
            logits[t] -= freq;
            if (!seen[t]) { seen[t] = 1; logits[t] -= presence; }
        }
        for (uint32_t i = 0; i < n; i++) if (gen[i] >= 0 && (uint32_t)gen[i] < n_vocab) seen[gen[i]] = 0;
    }
    if (dry_mult <= 0.f || dry_base <= 1.f || dry_allowed < 1 || n < 2 || !breaker) return;
    /* best[t] = 候选 t 的最长匹配 L; 只有被碰过的项才非零, 用完按碰过的清单擦掉(不 memset 整个词表) */
    static uint32_t *best = NULL; static int32_t *touched = NULL; static uint32_t ntouched = 0;
    if (!best) { best = xcalloc((size_t)n_vocab, sizeof(uint32_t)); touched = xmalloc((size_t)n_vocab * sizeof(int32_t)); }
    ntouched = 0;
    const int32_t last = gen[n - 1];
    if (last < 0 || (uint32_t)last >= n_vocab || breaker[last]) return;   /* 当前后缀以断点结尾: 匹配长度恒 0, 没有可罚的 */
    /* 候选 t = gen[j](j 可到 n-1: 那就是"同一个 token 连着重复"), 它前面那个 gen[j-1] 必须等于当前末 token 才有 L ≥ 1;
     * 再往回逐个比 gen[j-1-L] 与 gen[n-1-L](两段窗口允许重叠, 周期文本就是这样), 碰到断点或到头就停。 */
    for (uint32_t j = 1; j < n; j++) {
        if (gen[j - 1] != last) continue;
        const int32_t t = gen[j];
        if (t < 0 || (uint32_t)t >= n_vocab || breaker[t]) continue;   /* 候选本身是断点: 不罚(llama.cpp 同口径) */
        uint32_t L = 1;
        while (L < DRY_MAX_L && j - 1 >= L && gen[j - 1 - L] == gen[n - 1 - L] && !breaker[gen[n - 1 - L]]) L++;
        if (best[t] == 0) touched[ntouched++] = t;
        if (L > best[t]) best[t] = L;
    }
    for (uint32_t k = 0; k < ntouched; k++) {
        const int32_t t = touched[k];
        const uint32_t L = best[t]; best[t] = 0;
        if (L < (uint32_t)dry_allowed) continue;
        int e = (int)(L - (uint32_t)dry_allowed); if (e > DRY_MAX_EXP) e = DRY_MAX_EXP;
        logits[t] -= dry_mult * powf(dry_base, (float)e);
    }
}

/* 断点表: 逐 token 看文本里有没有 换行 / 引号 / 星号。llama.cpp 的默认集还有冒号, 这里故意不要:
 * ★实撞(2026-09-21 第三批探针)★: 带冒号(含全角"：")当断点时, 模型找到了空子 ——
 * "贵州茅台2024年上半年：营业总收入？净利润？" 一行一行抄, 冒号把每行切成两段短匹配(≤ 6 token, 罚不到 8 nat),
 * 顶不过复读区 9~10 nat 的置信差, 整段 100% 复读照旧。去掉冒号后整行是一段(~12 token), 罚到 1.75^10 ≈ 200, 抄到第 7 个 token 就被翻掉。
 * 换行仍是断点(不然列表/表格的固定格式会被误罚), 所以 ≤ 6 token 的短行死循环还是挡不住 —— 那要靠频率惩罚, 另立。
 * 一次性建(12.9 万个 token 各解一次文本, 几十毫秒), 进程内复用。 */
uint8_t *ds4_decode_breakers(ds4_engine *e, uint32_t n_vocab) {
    uint8_t *b = xcalloc((size_t)n_vocab, 1);
    for (uint32_t t = 0; t < n_vocab; t++) {
        size_t len = 0;
        char *s = ds4_token_text(e, (int)t, &len);
        if (!s) continue;
        for (size_t i = 0; i < len; i++) {
            const unsigned char c = (unsigned char)s[i];
            if (c == '\n' || c == '"' || c == '*') { b[t] = 1; break; }
        }
        free(s);
    }
    return b;
}
