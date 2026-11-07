/* 比较diag13保存的梯度，不加载底座、不更新权重。矩阵减法与平方和复用生产GPU原语。
 * 相对距离/余弦描述梯度差异，不证明实际更新或主判收益；平方和之差不是梯度距离。 */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while(0)

static void measure(const float *a, const float *b, const float *c, uint64_t n, double *aa, double *bb, double *dd) {
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(n*4), *y=ds4_gpu_tensor_alloc(n*4);
    CHECK(x && y);
    CHECK(ds4_gpu_tensor_write(x,0,a,n*4) && ds4_gpu_tensor_write(y,0,b,n*4));
    if(c) {
        ds4_gpu_tensor *z=ds4_gpu_tensor_alloc(n*4); CHECK(z);
        CHECK(ds4_gpu_tensor_write(z,0,c,n*4) && ds4_gpu_bwd_axpy_tensor(y,z,1.f,n));
        ds4_gpu_tensor_free(z);
    }
    CHECK(ds4_gpu_bwd_sumsq_tensor(x,n,aa) && ds4_gpu_bwd_sumsq_tensor(y,n,bb));
    CHECK(ds4_gpu_bwd_axpy_tensor(x,y,-1.f,n) && ds4_gpu_bwd_sumsq_tensor(x,n,dd));
    CHECK(isfinite(*aa) && isfinite(*bb) && isfinite(*dd) && *aa>=0 && *bb>=0 && *dd>=0);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(y);
}

int main(int argc, char **argv) {
    CHECK((argc==3 || argc==4) && ds4_gpu_init());
    const float a[]={1,2,3}, b[]={1,2,4}; double aa,bb,dd;
    measure(a,b,NULL,3,&aa,&bb,&dd); CHECK(aa==14 && bb==21 && dd==1);
    measure(a,a,NULL,3,&aa,&bb,&dd); CHECK(aa==14 && bb==14 && dd==0);
    const float tail[]={0,0,1}; measure(b,a,tail,3,&aa,&bb,&dd); CHECK(aa==21 && bb==21 && dd==0);
    fprintf(stderr,"known_answer_and_identical PASS\n");
    FILE *fa=fopen(argv[1],"rb"), *fb=fopen(argv[2],"rb"); CHECK(fa && fb);
    uint32_t ha[5],hb[5];
    CHECK(fread(ha,4,5,fa)==5 && fread(hb,4,5,fb)==5 && !memcmp(ha,hb,sizeof ha));
    CHECK(ha[0]==0x31445247u && ha[1]==0 && ha[2]==39 && ha[3]==64 && ha[4]==5120);
    FILE *fc=argc==4?fopen(argv[3],"rb"):NULL;
    if(argc==4) { uint32_t hc[5]; CHECK(fc && fread(hc,4,5,fc)==5 && !memcmp(ha,hc,sizeof ha)); }
    const uint64_t n=2ull*ha[3]*ha[4];
    float *x=malloc(n*4), *y=malloc(n*4); CHECK(x && y);
    float *z=fc?malloc(n*4):NULL; CHECK(!fc || z);
    double sa=0,sb=0,sd=0;
    puts("layer\tnorm_a\tnorm_b\tdifference_norm\trelative_to_a\tcosine");
    for(unsigned l=0;l<40;l++) {
        CHECK(fread(x,4,n,fa)==n && fread(y,4,n,fb)==n);
        if(fc) CHECK(fread(z,4,n,fc)==n);
        measure(x,y,z,n,&aa,&bb,&dd); CHECK(aa>0 && bb>0);
        printf("%u\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n",l,sqrt(aa),sqrt(bb),sqrt(dd),sqrt(dd/aa),.5*(aa+bb-dd)/sqrt(aa*bb));
        sa+=aa; sb+=bb; sd+=dd;
    }
    CHECK(fgetc(fa)==EOF && fgetc(fb)==EOF && !ferror(fa) && !ferror(fb));
    printf("TOTAL\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n",sqrt(sa),sqrt(sb),sqrt(sd),sqrt(sd/sa),.5*(sa+sb-sd)/sqrt(sa*sb));
    if(fc) { CHECK(fgetc(fc)==EOF && !ferror(fc)); fclose(fc); }
    free(x); free(y); free(z); fclose(fa); fclose(fb); return 0;
}
