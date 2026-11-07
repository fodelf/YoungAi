/* 量两份rank64补丁的完整有效权重差，不加载底座、不更新参数。
 * 编译：nvcc -O2 -arch=native amp_delta_norm.cu -lcublas -o amp_delta_norm
 * 使用：./amp_delta_norm <旧checkpoint目录> <新checkpoint目录>；无参数只跑已知答案检查。
 * ΔW=B1ᵀA1-B0ᵀA0=UᵀV，U=[B0;B1-B0]，V=[A1-A0;A1]。
 * ||ΔW||²=〈UUᵀ,VVᵀ〉，只做128×128 Gram；GPU用FP64避免近邻权重相减丢精度。
 * 这是整个线性补丁的矩阵范数，不等于端到端输出变化或准确率。 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CU(call) do { cudaError_t audit_cuda_rc=(call); if(audit_cuda_rc!=cudaSuccess){fprintf(stderr,"CUDA line %d: %s\n",__LINE__,cudaGetErrorString(audit_cuda_rc));exit(2);} } while(0)
#define BL(call) do { cublasStatus_t audit_blas_rc=(call); if(audit_blas_rc!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS line %d: status=%d\n",__LINE__,(int)audit_blas_rc);exit(2);} } while(0)

template <typename T> __global__ static void make_uv(const float *a0, const float *b0, const float *a1, const float *b1,
                               T *u, T *v, uint32_t n) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){u[i]=(T)b0[i];u[n+i]=(T)b1[i]-(T)b0[i];
             v[i]=(T)a1[i]-(T)a0[i];v[n+i]=(T)a1[i];}
}

template <typename T> static double delta2(cublasHandle_t h, const float *a0, const float *b0,
                     const float *a1, const float *b1, int e, int r) {
    const uint32_t n=(uint32_t)e*r; const int k=2*r;
    float *d[4]={0}; T *u=NULL,*v=NULL,*gu=NULL,*gv=NULL;
    const float *src[]={a0,b0,a1,b1};
    for(int j=0;j<4;j++){CU(cudaMalloc((void**)&d[j],n*sizeof(float)));CU(cudaMemcpy(d[j],src[j],n*sizeof(float),cudaMemcpyHostToDevice));}
    CU(cudaMalloc((void**)&u,2*n*sizeof(T)));CU(cudaMalloc((void**)&v,2*n*sizeof(T)));
    CU(cudaMalloc((void**)&gu,k*k*sizeof(T)));CU(cudaMalloc((void**)&gv,k*k*sizeof(T)));
    make_uv<T><<<(n+255)/256,256>>>(d[0],d[1],d[2],d[3],u,v,n);CU(cudaGetLastError());
    const T one=1,zero=0; T result=0;
    /* 行主序的k×e视作列主序e×k，转置乘自身即所需Gram。 */
    if constexpr(sizeof(T)==sizeof(double)) {
        BL(cublasDgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,k,k,e,&one,u,e,u,e,&zero,gu,k));
        BL(cublasDgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,k,k,e,&one,v,e,v,e,&zero,gv,k));
        BL(cublasDdot(h,k*k,gu,1,gv,1,&result));
    } else {
        BL(cublasSgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,k,k,e,&one,u,e,u,e,&zero,gu,k));
        BL(cublasSgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,k,k,e,&one,v,e,v,e,&zero,gv,k));
        BL(cublasSdot(h,k*k,gu,1,gv,1,&result));
    }
    CU(cudaDeviceSynchronize());
    for(int j=0;j<4;j++)CU(cudaFree(d[j]));
    CU(cudaFree(u));CU(cudaFree(v));CU(cudaFree(gu));CU(cudaFree(gv));
    if(!isfinite(result)||result<0){fprintf(stderr,"invalid matrix delta norm: %.17g\n",result);exit(2);}
    return result;
}

static void selftest(cublasHandle_t h) {
    const float eye[]={1,0,0,1},a[]={1.1f,.2f,0,.9f},b[]={.9f,.1f,.3f,1.2f};
    const float twice[]={2,0,0,2},half[]={.5f,0,0,.5f};
    const double actual=delta2<double>(h,eye,eye,a,b,2,2),same=delta2<double>(h,eye,eye,eye,eye,2,2);
    const double basis=delta2<double>(h,eye,eye,twice,half,2,2),single=delta2<float>(h,eye,eye,a,b,2,2);
    const int ok=fabs(actual-.2247)<1e-6 && same==0 && basis==0 && fabs(single-actual)<1e-6;
    fprintf(stderr,"matrix_delta_selftest known=%.12g expected=.2247 identical=%.12g equivalent_basis=%.12g fp32_error=%.12g %s\n",actual,same,basis,fabs(single-actual),ok?"PASS":"FAIL");
    if(!ok)exit(2);
}

static float *read_amp(const char *dir,int layer) {
    char path[1600]; const int sz=snprintf(path,sizeof path,"%s/amp_L%02d.bin",dir,layer);
    if(sz<0||(size_t)sz>=sizeof path){fprintf(stderr,"checkpoint path too long\n");exit(2);}
    FILE *f=fopen(path,"rb");if(!f){perror(path);exit(2);}
    int32_t hd[3];if(fread(hd,sizeof(int32_t),3,f)!=3||hd[0]!=5120||hd[1]!=64||hd[2]!=1){fprintf(stderr,"%s: expected 5120/rank64/f32\n",path);exit(2);}
    const size_t n=2u*5120*64;float *p=(float*)malloc(n*sizeof(float));
    if(!p||fread(p,sizeof(float),n,f)!=n||fgetc(f)!=EOF||ferror(f)){fprintf(stderr,"%s: invalid payload\n",path);exit(2);}
    fclose(f);return p;
}

int main(int argc,char **argv) {
    if(argc!=1&&argc!=3){fprintf(stderr,"usage: %s [old_checkpoint new_checkpoint]\n",argv[0]);return 2;}
    cublasHandle_t h;BL(cublasCreate(&h));BL(cublasSetMathMode(h,CUBLAS_PEDANTIC_MATH));selftest(h);
    if(argc==3){
        double sum=0,single_sum=0,max_error=0;const size_t n=5120u*64;
        puts("layer\teffective_weight_delta_squared\tfrobenius_norm\tfp32_squared\trelative_error");
        for(int l=0;l<40;l++){
            float *old=read_amp(argv[1],l),*next=read_amp(argv[2],l);
            const double d=delta2<double>(h,old,old+n,next,next+n,5120,64),s=delta2<float>(h,old,old+n,next,next+n,5120,64);
            const double error=fabs(s-d)/fmax(d,1e-30);sum+=d;single_sum+=s;max_error=fmax(max_error,error);
            printf("%d\t%.17g\t%.17g\t%.17g\t%.9g\n",l,d,sqrt(d),s,error);free(old);free(next);
        }
        printf("TOTAL\t%.17g\t%.17g\t%.17g\t%.9g\n",sum,sqrt(sum),single_sum,max_error);
        if(max_error>1e-4){fprintf(stderr,"FP32/FP64 discrepancy exceeds1e-4\n");return 2;}
    }
    BL(cublasDestroy(h));return 0;
}
