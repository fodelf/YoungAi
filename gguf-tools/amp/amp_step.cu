/* 沿两份已存rank64补丁之间的实际更新做小扰动，只供有限差分审计。
 * 用法：amp_step <父本目录> <更新后目录> <新目录> <有符号比例>，|比例|<=1/8。
 * GPU以FP64求差和插值，落盘仍f32；不含优化器，不改变rank，不覆盖已有目录。 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);} } while(0)
#define CU(x) CHECK((x)==cudaSuccess)

__global__ static void interpolate(float *out,const float *a,const float *b,size_t n,double alpha,int *bad) {
    const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    const double value=(double)a[i]+alpha*((double)b[i]-(double)a[i]);
    out[i]=(float)value;
    if(!isfinite(a[i]) || !isfinite(b[i]) || !isfinite(out[i]))atomicOr(bad,1);
}
static void step(float *out,const float *a,const float *b,size_t n,double alpha) {
    float *d[3]={0};int *bad=NULL,invalid=0;
    for(unsigned k=0;k<3;k++)CU(cudaMalloc((void**)&d[k],n*4));
    CU(cudaMalloc((void**)&bad,sizeof(int)));CU(cudaMemset(bad,0,sizeof(int)));
    CU(cudaMemcpy(d[0],a,n*4,cudaMemcpyHostToDevice));CU(cudaMemcpy(d[1],b,n*4,cudaMemcpyHostToDevice));
    interpolate<<<(n+255)/256,256>>>(d[2],d[0],d[1],n,alpha,bad);CU(cudaGetLastError());
    CU(cudaMemcpy(&invalid,bad,sizeof(int),cudaMemcpyDeviceToHost));CHECK(!invalid);
    CU(cudaMemcpy(out,d[2],n*4,cudaMemcpyDeviceToHost));
    for(unsigned k=0;k<3;k++)CU(cudaFree(d[k]));CU(cudaFree(bad));
}
static float *read_amp(const char *dir,unsigned layer) {
    char path[1600];int z=snprintf(path,sizeof path,"%s/amp_L%02u.bin",dir,layer);CHECK(z>0&&(size_t)z<sizeof path);
    FILE *f=fopen(path,"rb");CHECK(f);int32_t h[3];
    CHECK(fread(h,4,3,f)==3&&h[0]==5120&&h[1]==64&&h[2]==1);
    const size_t n=2u*5120*64;float *v=(float*)malloc(n*4);CHECK(v);
    CHECK(fread(v,4,n,f)==n&&fgetc(f)==EOF&&!ferror(f));CHECK(!fclose(f));return v;
}
int main(int argc,char **argv) {
    CHECK(argc==5);char *end=NULL;const double alpha=strtod(argv[4],&end);
    CHECK(end&&!*end&&isfinite(alpha)&&alpha!=0&&fabs(alpha)<=.125);
    const float a[]={1,2,-3},b[]={3,-2,5},plus[]={1.125,1.75,-2.5},minus[]={.875,2.25,-3.5};float out[3];
    step(out,a,b,3,.0625);for(unsigned i=0;i<3;i++)CHECK(out[i]==plus[i]);
    step(out,a,b,3,-.0625);for(unsigned i=0;i<3;i++)CHECK(out[i]==minus[i]);
    step(out,a,a,3,.0625);for(unsigned i=0;i<3;i++)CHECK(out[i]==a[i]);
    fprintf(stderr,"known_answer_positive_negative_identical PASS\n");
    CHECK(!mkdir(argv[3],0755));const size_t n=2u*5120*64;float *value=(float*)malloc(n*4);CHECK(value);
    for(unsigned l=0;l<40;l++) {
        float *base=read_amp(argv[1],l),*next=read_amp(argv[2],l);step(value,base,next,n,alpha);
        char path[1600];int z=snprintf(path,sizeof path,"%s/amp_L%02u.bin",argv[3],l);CHECK(z>0&&(size_t)z<sizeof path);
        FILE *f=fopen(path,"wb");CHECK(f);const int32_t h[]={5120,64,1};
        CHECK(fwrite(h,4,3,f)==3&&fwrite(value,4,n,f)==n);CHECK(!fclose(f));free(base);free(next);
    }
    free(value);printf("PASS layers=40 rank=64 alpha=%.17g optimizer_steps=0\n",alpha);return 0;
}
