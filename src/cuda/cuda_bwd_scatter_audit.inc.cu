/* 只供反传准入诊断：DS4_BWD_SCATTER_MODE=check 重放相同输入，fixed 固定槽位累加。
 * 默认仍走原核；不改变权重、前向或部署文件。逆表只重排整数索引，浮点求和在GPU。
 * 一个线程独占一个(token,维度)，按槽位0..K-1相加，避免多个专家块争写同一值。 */
__global__ static void bwd_scatter_ordered_kernel(float *dst, const float *src, const int32_t *inverse,
                                                uint32_t K, uint32_t D) {
    const uint32_t token=blockIdx.x;
    for(uint32_t d=threadIdx.x;d<D;d+=blockDim.x) {
        float value=dst[(uint64_t)token*D+d];
        for(uint32_t k=0;k<K;k++) {
            const int32_t row=inverse[(uint64_t)token*K+k];
            if(row>=0) value+=src[(uint64_t)row*D+d];
        }
        dst[(uint64_t)token*D+d]=value;
    }
}
static bool bwd_scatter_fixture(cudaStream_t cs) {
    const float src[]={3,30,1,10,4,40}, initial[]={100,1000,200,2000}, expected[]={101,1010,207,2070};
    const int32_t inverse[]={1,-1,0,2}; float actual[4];
    float *s=NULL,*d=NULL; int32_t *v=NULL;
    bool ok=cudaMalloc(&s,sizeof src)==cudaSuccess && cudaMalloc(&d,sizeof initial)==cudaSuccess && cudaMalloc(&v,sizeof inverse)==cudaSuccess;
    if(ok) ok=cudaMemcpyAsync(s,src,sizeof src,cudaMemcpyHostToDevice,cs)==cudaSuccess &&
        cudaMemcpyAsync(d,initial,sizeof initial,cudaMemcpyHostToDevice,cs)==cudaSuccess &&
        cudaMemcpyAsync(v,inverse,sizeof inverse,cudaMemcpyHostToDevice,cs)==cudaSuccess;
    if(ok) {
        bwd_scatter_ordered_kernel<<<2,256,0,cs>>>(d,s,v,2,2);
        ok=cudaMemcpyAsync(actual,d,sizeof actual,cudaMemcpyDeviceToHost,cs)==cudaSuccess && cudaStreamSynchronize(cs)==cudaSuccess &&
           !memcmp(actual,expected,sizeof actual);
    }
    cudaFree(s); cudaFree(d); cudaFree(v);
    fprintf(stderr,"[scatter-check] known_answer=%s missing_slot=covered\n",ok?"PASS":"FAIL");
    return ok;
}
static v41_scratch g_bwd_inverse;
static bool bwd_scatter_dispatch(float *dst, const float *src, const int32_t *pair, const int32_t *list,
                                 uint32_t K, uint32_t D, uint32_t n, uint32_t nv, uint32_t layer, cudaStream_t cs) {
    static int mode=-1; static bool fixture=false;
    if(mode<0) {
        const char *s=getenv("DS4_BWD_SCATTER_MODE");
        mode=s && !strcmp(s,"check")?1:s && !strcmp(s,"fixed")?2:0;
        if(s && mode==0) { fprintf(stderr,"[scatter-check] unknown mode\n"); return false; }
    }
    if(!mode || (mode==1 && layer!=39)) {
        bwd_scatter_add_kernel<<<nv,256,0,cs>>>(dst,src,pair,K,D);
        return cuda_ok(cudaGetLastError(),"bwd original scatter");
    }
    if(!fixture) { if(!bwd_scatter_fixture(cs)) return false; fixture=true; }
    const size_t slots=(size_t)n*K, bytes=(size_t)n*D*4;
    int32_t *inv=(int32_t *)malloc(slots*4);
    if(!inv) return false;
    for(size_t i=0;i<slots;i++) inv[i]=-1;
    bool ok=true;
    for(uint32_t r=0;r<nv;r++) {
        if(list[r]<0 || (size_t)list[r]>=slots || inv[list[r]]!=-1) { ok=false; break; }
        inv[list[r]]=(int32_t)r;
    }
    int32_t *device=ok?(int32_t *)v41_grow(&g_bwd_inverse,slots*4,"bwd inverse pairs"):NULL;
    ok=ok && device && cudaMemcpyAsync(device,inv,slots*4,cudaMemcpyHostToDevice,cs)==cudaSuccess && cudaStreamSynchronize(cs)==cudaSuccess;
    free(inv); if(!ok) return false;
    if(mode==2) {
        bwd_scatter_ordered_kernel<<<n,256,0,cs>>>(dst,src,device,K,D);
        return cuda_ok(cudaGetLastError(),"bwd ordered scatter");
    }
    float *initial=NULL,*original=NULL; void *a=malloc(bytes),*b=malloc(bytes);
    ok=a && b && cudaMalloc(&initial,bytes)==cudaSuccess && cudaMalloc(&original,bytes)==cudaSuccess &&
       cudaMemcpyAsync(initial,dst,bytes,cudaMemcpyDeviceToDevice,cs)==cudaSuccess;
    size_t different[2]={0,0};
    for(unsigned kind=0;ok && kind<2;kind++) for(unsigned rep=0;ok && rep<2;rep++) {
        ok=cudaMemcpyAsync(dst,initial,bytes,cudaMemcpyDeviceToDevice,cs)==cudaSuccess;
        if(kind) bwd_scatter_ordered_kernel<<<n,256,0,cs>>>(dst,src,device,K,D);
        else bwd_scatter_add_kernel<<<nv,256,0,cs>>>(dst,src,pair,K,D);
        ok=ok && cuda_ok(cudaGetLastError(),"bwd scatter replay");
        if(!kind && !rep) ok=ok && cudaMemcpyAsync(original,dst,bytes,cudaMemcpyDeviceToDevice,cs)==cudaSuccess;
        ok=ok && cudaMemcpyAsync(rep?b:a,dst,bytes,cudaMemcpyDeviceToHost,cs)==cudaSuccess && cudaStreamSynchronize(cs)==cudaSuccess;
        if(ok && rep) for(size_t j=0;j<bytes;j++) different[kind]+=((unsigned char*)a)[j]!=((unsigned char*)b)[j];
    }
    /* 重放是观测，不让测试改写本次原核输出。 */
    if(ok) ok=cudaMemcpyAsync(dst,original,bytes,cudaMemcpyDeviceToDevice,cs)==cudaSuccess && cudaStreamSynchronize(cs)==cudaSuccess;
    fprintf(stderr,"[scatter-check] L%u tokens=%u pairs=%u bytes=%zu original_repeat_different_bytes=%zu ordered_repeat_different_bytes=%zu restored_original=%u\n",
            layer,n,nv,bytes,different[0],different[1],ok);
    free(a); free(b); cudaFree(initial); cudaFree(original);
    return ok && !different[1];
}
