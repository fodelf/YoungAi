/* cuda_vq_decode_launch.inc.cu — cuda_vq_decode.inc.cu 的尾部发射器(2026-09-19 拆出, 守单文件 ≤500 行)。
 * 聚合根按序 #include: 必须紧跟 cuda_vq_decode.inc.cu(用它的 v41_vq_fused_moe_n 模板与 v41_vq_mat)。
 * 这里只做一件事: 按【盘上版本 + 码本词数】挑实例 + 形状前置校验。数值全在被它调的模板里。 */
/* 按码本词数分发: 位宽 = ⌈log2(词数)⌉ 决定一轮的字数, 核按它实例化(现役配方 vq8x4096 ⇒ 12 位)。
 * 一轮 = 32 个索引 ⇒ 行的索引数(cols/8)必须是 32 的倍数(IN/MID 是 256 的倍数); 不满足就硬错, 不留慢路。
 *
 * ★版本也要进实例(2026-09-21, 113.md 方案 v3)★: v2 的载荷自带 f16 码本、位流按 NBIT 直排;
 * v3 的码本一层一本且只存 E4M3、位流恒 12 位主流 + (13 位层)一个位平面。两版的布局差别全在
 * cuda_vq_row.inc.cu 的那一族里, 这里只负责"选对实例" —— 选错了不会报错, 只会出一整套假权重,
 * 所以版本是从 blob 头的 ver 字段读的(调用方传进来), 不是猜的、也不按文件名。
 * 实例表(× SORTED 两份): v2 12 位 / v2 11 位 / v3 12 位 / v3 13 位(带位平面)。 */
static int v41_vq_fused_moe(float *out, const uint8_t *blob, uint32_t IN, uint32_t MID, uint32_t OUT,
                            const int32_t *sel, const float *w, uint32_t K, float clamp, const float *x, uint32_t n_tok, uint32_t nc,
                            const float *gr, uint32_t ver) {
    uint32_t nbit = 0; while ((1u << nbit) < nc) nbit++;
    /* 一轮 32 个索引 ⇒ IN/MID 是 256 的倍数(V4.1 Flash: 5120 → 20 轮, 2304 → 9 轮); 尾块几轮都行 */
    if ((IN % 256u) || (MID % 256u)) {
        fprintf(stderr, "ds4: [v41] VQ 解码核要求 IN/MID 是 256 的倍数(一轮 32 个索引), 现 %u/%u\n", IN, MID);
        return 0;
    }
    if (ver == 3u) {
        /* v3: 12 位层无位平面, 13 位层带一个。更宽要再加一个平面(转换器那边同样是硬停, 不静默) */
        if (nbit == 12u) return v41_vq_fused_moe_n<12, 1, 0>(out, blob, IN, MID, OUT, sel, w, K, clamp, x, n_tok, nc, gr);
        if (nbit == 13u) return v41_vq_fused_moe_n<13, 1, 1>(out, blob, IN, MID, OUT, sel, w, K, clamp, x, n_tok, nc, gr);
        fprintf(stderr, "ds4: [v41] DQVL v3 码本 %u 词(%u 位)没有对应的解码核实例(只有 12/13 位)\n", nc, nbit);
        return 0;
    }
    /* v2 两档实例(几何与代价见 cuda_vq_row.inc.cu 文件头): nc4096 = 12 位, nc2048 = 11 位 */
    if (nbit == 12u) return v41_vq_fused_moe_n<12, 0, 0>(out, blob, IN, MID, OUT, sel, w, K, clamp, x, n_tok, nc, gr);
    if (nbit == 11u) return v41_vq_fused_moe_n<11, 0, 0>(out, blob, IN, MID, OUT, sel, w, K, clamp, x, n_tok, nc, gr);
    fprintf(stderr, "ds4: [v41] VQ 码本 %u 词(%u 位)没有对应的解码核实例\n", nc, nbit);
    return 0;
}
