/* dilution —— 量化"放大器杠杆"沿前向链的稀释系数。
 *
 * 侧车修的是 routed_out(专家聚合)。它到最终 logits 之间隔着:
 *   routed_out → 加进 hyper-connection 残差流 → 层出口 h(4 条流) → 后续 42 层 → logits
 * 每一跳都按范数比稀释。本工具用引擎真值捕获量出第一跳的硬比例:
 *   分子 = ‖routed_teacher − routed_student‖  (侧车能修的全部量, 修满即上界)
 *   分母 = ‖层出口 h‖                          (下游实际看到的量)
 * 比值给出"单层 routed 完全修好, 层出口最多变多少"的上界。
 *
 * 输入: capnpy/{routed,obase_v3}_L{L}.npy [n,4096] + hdump/h_L%02d.bin [n,NHC,4096] f32
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../calib/npy.h"

#define DM  4096
#define NHC 4

static double l2(const float *a, long n) {
    double s = 0; for (long i = 0; i < n; i++) s += (double)a[i] * a[i]; return sqrt(s);
}
static double cosv(const float *a, const float *b, long n) {
    double d = 0, na = 0, nb = 0;
    for (long i = 0; i < n; i++) { d += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
    return (na > 0 && nb > 0) ? d / (sqrt(na) * sqrt(nb)) : 0;
}

int main(int argc, char **argv) {
    const char *cap = NULL, *hd = NULL, *layers = "0,8,16,24,32,42";
    long nmax = 1024; double RK = 0.44;   /* ‖Δ‖/‖routed‖ 实测常数(无教师降级用) */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cap") && i+1 < argc) cap = argv[++i];
        else if (!strcmp(argv[i], "--hdump") && i+1 < argc) hd = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i+1 < argc) layers = argv[++i];
        else if (!strcmp(argv[i], "--ntok") && i+1 < argc) nmax = atol(argv[++i]);
    }
    if (!cap || !hd) { fprintf(stderr, "用法: dilution --cap DIR --hdump DIR [--layers ..] [--ntok N]\n"); return 2; }

    printf("%-4s %10s %10s %9s %10s %12s %10s\n",
           "L", "|routed|", "|Δ学生|", "base_cos", "|h出口|", "Δ/|h| (上界)", "Δ/|routed|");
    char buf[512]; strncpy(buf, layers, sizeof buf - 1); buf[sizeof buf - 1] = 0;
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        int L = atoi(tok);
        char p[1024]; npy_meta m1, m2;
        snprintf(p, sizeof p, "%s/routed_L%d.npy", cap, L);
        float *ref = npy_read_f32(p, &m1);
        snprintf(p, sizeof p, "%s/obase_v3_L%d.npy", cap, L);
        float *stu = npy_read_f32(p, &m2);
        if (!stu) { fprintf(stderr, "L%d: obase 缺\n", L); free(ref); continue; }
        /* 无教师(routed_L 缺)时降级: Δ 按 ‖Δ‖/‖routed‖ 的实测常数估。L8/32/42 三层
         * 分别 44.3%/46.5%/40.8%, 全深度稳在 40~47% —— 用作跨层杠杆排序足够。 */
        int est = (ref == NULL);
        long n = (est ? m2.shape[0] : m1.shape[0]); if (n > nmax) n = nmax;

        snprintf(p, sizeof p, "%s/h_L%02d.bin", hd, L);
        FILE *f = fopen(p, "rb");
        if (!f) { fprintf(stderr, "L%d: %s 打不开\n", L, p); free(ref); free(stu); continue; }

        double sr = 0, sd = 0, sc = 0, sh = 0; long cnt = 0;
        float *hrow = malloc((size_t)NHC * DM * sizeof(float));
        float *d = malloc((size_t)DM * sizeof(float));
        for (long t = 0; t < n; t++) {
            if (fread(hrow, sizeof(float), (size_t)NHC*DM, f) != (size_t)NHC*DM) break;
            const float *s = stu + t*DM;
            if (est) { double ns = l2(s, DM); sr += ns; sd += RK * ns; sc += 0; }
            else {
                const float *r = ref + t*DM;
                for (int j = 0; j < DM; j++) d[j] = r[j] - s[j];
                sr += l2(r, DM); sd += l2(d, DM); sc += cosv(r, s, DM);
            }
            sh += l2(hrow, (long)NHC*DM);
            cnt++;
        }
        fclose(f); free(hrow); free(d);
        if (!cnt) { fprintf(stderr, "L%d: 无样本\n", L); free(ref); free(stu); continue; }
        double R = sr/cnt, D = sd/cnt, C = sc/cnt, H = sh/cnt;
        printf("%-4d %10.2f %10.2f %9s %10.2f %11.3f%% %9.3f%%\n",
               L, R, D, est ? "(估)" : (snprintf(p,sizeof p,"%.5f",C), p), H, 100.0*D/H, 100.0*D/R);
        free(ref); free(stu);
    }
    return 0;
}
