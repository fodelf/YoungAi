/* z_static_shim.c — 只为验证: 把 ds4quant_run.c 里 static 的 z 求解链暴露出来。
 * 手法: 直接 #include 该 .c 的前段(到 z_solve_fourloss 结束), 通过宏屏蔽 main 与量化主体。 */
#define main ds4quant_unused_main
#include "../quant/ds4quant_run.c"
#undef main
ds4_z *ztest_solve_fourloss(const float *X,const float *R,int n,int d,int rank,float lambda,
                            const float *colw,int naug,float augw,unsigned long long seed){
    return z_solve_fourloss(X,R,n,d,rank,lambda,colw,naug,augw,(uint64_t)seed);
}
