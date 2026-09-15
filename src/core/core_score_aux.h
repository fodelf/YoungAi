/* core_score_aux.h — teacher-forced 打分的两个【小】出口: 逐位 NLL 与逐位 top-K。
 * V4 的 --eval-ids 批路与 V4.1 的 --score-ids 分块路共用这一份实现(2026-09-13)。
 *
 * ★为什么必须有这两个出口★: 全词表 logits 每个位置 129280×4 = 517 KB, 一趟 7.3k token
 * 就是 8.1 GB。GB10 是 121 GB 的 CPU/GPU 统一内存 —— 那 8 GB 落盘会灌满 page cache,
 * 下一趟引擎启动时 MemAvailable 被掏空, GPU 分配直接失败。2026-09-08 夜就是这么把 spark
 * 写崩的(内核日志 NVRM: Out of memory, 六个时间点全对上那几轮判决)。★写大文件 = 占 GPU 内存★。
 * 判决只要 4 字节(目标 token 的 −log p), 后训练的靶只要 top-K 的 (id, p)。
 *
 * ★为什么两条路共用一份★: 两边写的是同一种文件, 下游(finetune_solve / anchor_metrics)按
 * 同一套字节读。抄第二份 = 迟早一边改了另一边没改, 而读的人不知道信谁。
 *
 * 文件格式(与 2026-09-10 落地的 V4 产物逐字节相同, 下游零改动):
 *   NLL : f32[S] —— 位置 i 的值 = −log p(ids[i+1] | ids[0..i])。最后一位没有下一个 token,
 *         写 NaN 占位(行数恒等于 S, 下游按行号取值不用边界特判; NaN 参与任何统计都会立刻
 *         暴露, 而 0.0 会被当成"这个位置预测得完美"悄悄拉低平均)。
 *   topK: 头 <u32 'ETGD'><u32 K><u32 S><u32 VOCAB>, 之后每行
 *         <row u32><tgt i32><tgt_p f32><mass f32><ids i32[K]><ps f32[K]>, ps 是归一化后的概率。
 *         mass = top-K 覆盖的概率质量 —— ★近似有多糙必须是能读出来的★, 不能靠"应该够了"。
 *   rms : f32[S] —— 位置 i 的 ★inv★ = rsqrt(mean(x²)+eps), 就是出口 RMSNorm 那一步乘进去的
 *         那个标量本身(不是 rms, 是它的倒数; 存能直接用的那个数, 省得下游再倒一次还要对 eps)。
 *         后训练第二版要它: logit 差对末层 MoE 输出的系数里带 inv(back.md §4.1), 少了它
 *         "改动能把 logit 差推多少"就只有形状没有单位。 */
#ifndef DS4_CORE_SCORE_AUX_H
#define DS4_CORE_SCORE_AUX_H

#include <stdint.h>

typedef struct ds4_score_aux ds4_score_aux;

/* 两条路径都为空(或 topk<=0 且 nll 为空) => 返回 NULL, 调用方照常跑、什么都不写。
 * 显式给了路径却打不开 => 直接 exit(1): 请求了出口却静默不产, 下游会拿旧文件当新读数。
 * tag 只进日志(如 "EVAL_IDS" / "v41"), 方便在混合日志里认出是哪条路写的。 */
ds4_score_aux *ds4_score_aux_open(const char *nll_path, const char *topk_path, int topk,
                                  const char *rms_path, uint32_t n, uint32_t vocab, const char *tag);

/* 写一块行的 inv(= rsqrt(mean(x²)+eps))。x = 出口 RMSNorm 【之前】的隐状态, 行主序 [nrow][dim],
 * i0 = 这一块第一行的绝对行号。★必须与后端 kernel 同式同 eps★(cuda_v41_1.inc.cu 的
 * v41_rms_norm_kernel: f32 累加、除 dim、加 eps、rsqrt) —— 差一点点不会报错, 只会让下游
 * 预测的 logit 差有个不知从哪来的系数。没开 rms 出口时是空操作。 */
void ds4_score_aux_rms_rows(ds4_score_aux *a, uint32_t i0, const float *x,
                            uint32_t nrow, uint32_t dim, float eps);

/* 写一行。i = 绝对行号(与 ids 同一口径), lg = 该行的全词表 logits(未 softmax),
 * tgt = 位置 i 要预测的 token(即 ids[i+1]); i 是最后一位时传 -1。 */
void ds4_score_aux_row(ds4_score_aux *a, uint32_t i, const float *lg, int tgt);

/* 关文件并打印统计(平均 NLL/PPL、topK 平均覆盖质量)。a 可以是 NULL。 */
void ds4_score_aux_close(ds4_score_aux *a);

#endif /* DS4_CORE_SCORE_AUX_H */
