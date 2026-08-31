/* calib_shared.inc.c — calib 工具族共享的判决原语(2026-08-31 魔数扫除收拢)。
 * baseline_metrics 曾在 calib_run_p1 / calib_diag_p2 逐字两份, quant_dequant 曾在
 * calib_diag_p2 / validate_fwd_p1 逐字两份 —— 判决指标两份实现一旦漂移就是两把尺。
 * 单 TU 纹理包含: 各聚合根在原定义位 include 本片, 符号仍是各 TU 的 static。 */
#include <math.h>
#include "onebit_quant.h"

/* 基线指标: rel_l2 = ‖delta‖/‖ref‖(整体), cos_mean = 逐(专家,token)行余弦均值 */
static void baseline_metrics(const double *o_ref, const double *o_hat, const double *delta,
                             int n_exp, int n_x, int d_model, double *rel_l2, double *cos_mean) {
    double num = 0, den = 0, csum = 0; long cnt = 0;
    for (long ei = 0; ei < (long)n_exp * n_x; ei++) {
        const double *r = o_ref + (size_t)ei * d_model;
        const double *h = o_hat + (size_t)ei * d_model;
        const double *d = delta + (size_t)ei * d_model;
        double dot = 0, nr = 0, nh = 0, dd = 0, rr = 0;
        for (int j = 0; j < d_model; j++) { dot += r[j]*h[j]; nr += r[j]*r[j]; nh += h[j]*h[j]; dd += d[j]*d[j]; rr += r[j]*r[j]; }
        num += dd; den += rr;
        if (nr > 0 && nh > 0) { csum += dot / (sqrt(nr) * sqrt(nh)); cnt++; }
    }
    *rel_l2 = den > 0 ? sqrt(num / den) : 0;
    *cos_mean = cnt ? csum / cnt : 0;
}

/* go1b round-trip a weight matrix [rows×cols] → w_hat (the 1-bit reconstruction). */
static void quant_dequant(const float *w, float *wh, int rows, int cols, unsigned char *scratch) {
    size_t rb = go1b_row_bytes(cols);
    go1b_quantize(w, scratch, rows, cols);
    for (int r = 0; r < rows; r++)
        go1b_dequantize_row(scratch + (size_t)r * rb, wh + (size_t)r * cols, cols);
}
