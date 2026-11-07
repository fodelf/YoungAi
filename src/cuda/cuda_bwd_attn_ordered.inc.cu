/* DS4_BWD_ATTN_MODE=check/fixed仅供准入。逐query发射原键梯度核，仍在同一CUDA流。
 * 每个query的窗口键唯一；压缩键先检查唯一性，防止同一发射中仍有争写。
 * 不改变点积、精度或内存布局；只固定不同query对同一键的累加顺序。 */
static bool bwd_attn_unique(const int32_t *idx, uint32_t n, uint32_t topk, uint32_t ng, cudaStream_t cs) {
    if(!idx || !topk || !ng) return true;
    int32_t *h=(int32_t*)malloc((size_t)n*topk*4);
    uint32_t *seen=(uint32_t*)calloc(ng,4);
    bool ok=h && seen && cudaMemcpyAsync(h,idx,(size_t)n*topk*4,cudaMemcpyDeviceToHost,cs)==cudaSuccess && cudaStreamSynchronize(cs)==cudaSuccess;
    for(uint32_t i=0;ok && i<n;i++) for(uint32_t k=0;k<topk;k++) {
        const int32_t g=h[(size_t)i*topk+k];
        if(g<0 || (uint32_t)g>=ng) continue;
        if(seen[g]==i+1) { ok=false; break; }
        seen[g]=i+1;
    }
    free(h); free(seen);
    if(!ok) fprintf(stderr,"[attn-order] compressed indices not unique or unreadable\n");
    return ok;
}
static bool bwd_attn_kg_dispatch(float *gkv, float *gcomp, const uint16_t *wb, uint32_t nkmax, const float *q,
                                const float *go, const int32_t *idx, uint32_t n_head, uint32_t pos0, uint32_t window,
                                uint32_t ng, uint32_t topk, uint32_t n, cudaStream_t cs) {
    static int mode=-1; static bool checked=false;
    if(mode<0) {
        const char *s=getenv("DS4_BWD_ATTN_MODE");
        mode=s && !strcmp(s,"check")?1:s && !strcmp(s,"fixed")?2:0;
        if(s && !mode) { fprintf(stderr,"[attn-order] unknown mode\n"); return false; }
    }
    auto launch=[&](bool ordered) {
        const uint32_t count=ordered?n:1;
        for(uint32_t i=0;i<count;i++)
            bwd_attn_kg_kernel<<<dim3(ordered?1:n,(nkmax+BWD_KG_BN-1)/BWD_KG_BN,512/BWD_KG_BM),512,0,cs>>>
                (gkv,gcomp,wb,nkmax,q,go,idx,n_head,pos0,window,ng,topk,i);
        return cuda_ok(cudaGetLastError(),"bwd attn kg dispatch");
    };
    if(!mode || (mode==1 && checked)) return launch(false);
    if(gcomp && !bwd_attn_unique(idx,n,topk,ng,cs)) return false;
    if(mode==2) return launch(true);
    checked=true;
    const size_t kvbytes=(size_t)n*512*4, cpbytes=gcomp?(size_t)ng*512*4:0, bytes=kvbytes+cpbytes;
    float *initial=NULL,*original=NULL; void *a=malloc(bytes),*b=malloc(bytes);
    auto save=[&](float *p) {
        return cudaMemcpyAsync(p,gkv,kvbytes,cudaMemcpyDeviceToDevice,cs)==cudaSuccess &&
            (!cpbytes || cudaMemcpyAsync((char*)p+kvbytes,gcomp,cpbytes,cudaMemcpyDeviceToDevice,cs)==cudaSuccess);
    };
    auto restore=[&](const float *p) {
        return cudaMemcpyAsync(gkv,p,kvbytes,cudaMemcpyDeviceToDevice,cs)==cudaSuccess &&
            (!cpbytes || cudaMemcpyAsync(gcomp,(const char*)p+kvbytes,cpbytes,cudaMemcpyDeviceToDevice,cs)==cudaSuccess);
    };
    bool ok=a && b && cudaMalloc(&initial,bytes)==cudaSuccess && cudaMalloc(&original,bytes)==cudaSuccess && save(initial);
    size_t different[2]={0,0};
    for(unsigned kind=0;ok && kind<2;kind++) for(unsigned rep=0;ok && rep<2;rep++) {
        ok=restore(initial) && launch(kind!=0);
        if(ok && !kind && !rep) ok=save(original);
        char *host=(char*)(rep?b:a);
        ok=ok && cudaMemcpyAsync(host,gkv,kvbytes,cudaMemcpyDeviceToHost,cs)==cudaSuccess &&
            (!cpbytes || cudaMemcpyAsync(host+kvbytes,gcomp,cpbytes,cudaMemcpyDeviceToHost,cs)==cudaSuccess) &&
            cudaStreamSynchronize(cs)==cudaSuccess;
        if(ok && rep) for(size_t j=0;j<bytes;j++) different[kind]+=((unsigned char*)a)[j]!=((unsigned char*)b)[j];
    }
    if(ok) ok=restore(original) && cudaStreamSynchronize(cs)==cudaSuccess;
    fprintf(stderr,"[attn-order] first_call tokens=%u bytes=%zu original_repeat_different_bytes=%zu ordered_repeat_different_bytes=%zu restored_original=%u\n",
            n,bytes,different[0],different[1],ok);
    free(a); free(b); cudaFree(initial); cudaFree(original);
    return ok && !different[1];
}
