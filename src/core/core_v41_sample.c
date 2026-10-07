/* core_v41_sample.c — V4.1 生成路"取下一个 token"的分岔(2026-09-28): 设备采样核 / 设备 argmax / 主机惩罚路, 以及验证批结果的拼装。
 *
 * 为什么单拆一个文件: 09-28 采样默认改成模型卡配方(温 1.0)之后, 请求不带 temperature 就走采样路; 09-28 尺(真实 CFO 请求, 同一二进制):
 * 贪心纯解码 29.24 t/s / 采样纯解码 28.92 / 贪心+投机 42.49 —— 主机采样本身每步只贵 0.32 ms(1%), 真正丢的是投机(−32%)。
 * 投机要在采样下成立, 验证就得是拒绝采样(接受 ⇔ 均匀数 < 目标分布给草稿的概率, 拒绝从残差抽), 那一步的分子分母都在设备上,
 * 所以采样搬进设备核(cuda_v41_sample.inc.cu), 主机只读 16 B/行 —— 与贪心路 argmax 同一个槽位、同一条读回路。
 *
 * 三条路的分岔(core_v41_api.c 决定, 这里执行):
 *   温度 ≤ 0 且无惩罚 → 设备 argmax(老路一字不动, 门 = 温 0 输出逐字节回归);
 *   温度 > 0 且无惩罚 → 设备采样核(st->dev_sample), 投机照走;
 *   任一惩罚非零     → 读回整行, 主机罚完交给 V4 路同一份采样器(温 0 时它给 argmax); 惩罚要看 token 史, 投机不接。
 * 出错会怎样: want 拼错(接受位填了残差或反之)⇒ 投机与纯解码分布分叉, 采样下看不出逐字节, 门在 tests/cuda_sample_selftest.c
 * (核的边缘分布 = 目标分布)与 d1 的 dflt 模式(贪心路逐字节)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU

void v41_sample_pick(const int32_t *slot, const int32_t *batch, uint32_t n, int dev_sample, int32_t *want) {
    for (uint32_t i = 0; i < n; i++) {
        const int32_t *s = slot + 4u * i;
        want[i] = (dev_sample && batch && i + 1u < n) ? (s[1] ? batch[i + 1u] : s[2]) : s[0];
    }
}

bool v41_device_next(ds4_v41_state *st, ds4_gpu_tensor *am, uint32_t row0, uint32_t n, const int32_t *batch, int32_t *want) {
    int32_t slot[(DS4_MTP_MAX_BLOCK + 2u) * 4u];
    if (!am || n == 0u || n > DS4_MTP_MAX_BLOCK + 2u) return false;   /* 槽够不够 16·n 字节由核入口按 out->bytes 判 */
    if (st->dev_sample) {
        if (!ds4_gpu_v41_sample_tensor(am, st->logits, row0, n, DS4_N_VOCAB, st->pos, st->tok, &st->samp, n > 1u ? st->spec_q : NULL) ||
            !ds4_gpu_synchronize() || !ds4_gpu_tensor_read(am, 0, slot, (uint64_t)n * 16u)) return false;
    } else {
        /* argmax 核只写槽的第 0 个 int, 逐行发、逐行读(直发路只在暖身步/捕获失败重来时走, n 次同步无所谓) */
        for (uint32_t i = 0; i < n; i++) {
            slot[4u * i + 1u] = slot[4u * i + 2u] = slot[4u * i + 3u] = 0;
            if (!ds4_gpu_v41_argmax_tensor(am, st->logits, row0 + i, DS4_N_VOCAB) || !ds4_gpu_synchronize() ||
                !ds4_gpu_tensor_read(am, 0, &slot[4u * i], 4)) return false;
        }
    }
    v41_sample_pick(slot, batch, n, st->dev_sample, want);
    return true;
}

bool v41_next_token(ds4_v41_state *st, ds4_gpu_tensor *am, uint32_t row, float *rowbuf, uint64_t *rng, v41_hist *h, int32_t *out) {
    if (!rowbuf) return v41_device_next(st, am, row, 1u, NULL, out);
    const ds4_decode_sampling *sp = st->psamp ? st->psamp : &g_decode_sampling;   /* 并发: 每请求自己的采样面 */
    if (!ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(st->logits, (uint64_t)row * DS4_N_VOCAB * 4u, rowbuf, (uint64_t)DS4_N_VOCAB * 4u)) return false;
    if (h && h->n && (sp->dry_multiplier > 0.f || sp->freq_penalty != 0.f || sp->presence_penalty != 0.f))
        ds4_decode_penalize(rowbuf, DS4_N_VOCAB, h->tok, h->n, h->brk, sp->freq_penalty, sp->presence_penalty,
                            sp->dry_multiplier, sp->dry_base, sp->dry_allowed_length);
    *out = (int32_t)ds4_sample_logits(rowbuf, (int)DS4_N_VOCAB, sp->temperature, sp->top_k, sp->top_p, sp->min_p, rng);
    if (h && h->n < h->cap) h->tok[h->n++] = *out;
    return true;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_sample_nonempty_tu;
