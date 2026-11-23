/* 原KL/排序核保持不变；只给冻结正确且原p1门排除的位置补原hinge项。
 * 暂存每行两个ID和一个损失，不分配词表大小的额外梯度矩阵。 */
#include "ds4_gpu_keepmargin.h"
__global__ static void keepmargin_prepare(int32_t *pair, float *extra, const float *z, uint32_t row0,
    uint32_t V, const int32_t *ids, const float *p, uint32_t K, const unsigned char *keep, float lam, float pmin, float gap) {
    const uint32_t r=blockIdx.x;
    __shared__ int32_t target, shj[32];
    __shared__ float delta, sh[32];
    if(threadIdx.x==0) {
        pair[r*2]=-1; pair[r*2+1]=-1; extra[r]=0; target=-1; delta=0;
        unsigned best=0; float p2=0;
        for(unsigned k=1;k<K;k++) if(p[(uint64_t)r*K+k]>p[(uint64_t)r*K+best]) best=k;
        for(unsigned k=0;k<K;k++) if(k!=best) p2=fmaxf(p2,p[(uint64_t)r*K+k]);
        const float p1=p[(uint64_t)r*K+best]; const int32_t id=ids[(uint64_t)r*K+best];
        if(keep[r] && p1<pmin && p1>0 && id>=0 && (uint32_t)id<V) {
            target=id; delta=fminf(gap,logf(p1/fmaxf(p2,1e-6f)));
        }
    }
    __syncthreads();
    if(target<0) return;
    const float *zr=z+(uint64_t)(row0+r)*V;
    float value=-INFINITY; int32_t which=0x7fffffff;
    for(uint32_t j=threadIdx.x;j<V;j+=blockDim.x) if((int32_t)j!=target && zr[j]>value) {value=zr[j];which=(int32_t)j;}
    bwd_block_argmax(&value,&which,sh,shj);
    if(threadIdx.x==0 && zr[target]-value<delta) {
        pair[r*2]=target; pair[r*2+1]=which; extra[r]=lam*(delta-(zr[target]-value));
    }
}
__global__ static void keepmargin_apply(float *g,float *loss,const int32_t *pair,const float *extra,uint32_t m,uint32_t V,float step) {
    const uint32_t r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=m || pair[r*2]<0) return;
    g[(uint64_t)r*V+pair[r*2]]-=step;
    g[(uint64_t)r*V+pair[r*2+1]]+=step;
    loss[r]+=extra[r];
}
int ds4_gpu_bwd_kl_keepmargin_tensor(ds4_gpu_tensor *g,ds4_gpu_tensor *loss,const ds4_gpu_tensor *z,
    uint32_t row,uint32_t m,uint32_t V,const ds4_gpu_tensor *tid,const ds4_gpu_tensor *tp,const ds4_gpu_tensor *rest,
    uint32_t K,float scale,const ds4_gpu_tensor *mask,float lam,float pmin,float gap) {
    if(!mask || !g || !loss || !z || !tid || !tp || !rest || !m || V<2 || V>131072 || !K || K>128 ||
       mask->bytes<m || g->bytes<(uint64_t)m*V*4 || loss->bytes<(uint64_t)m*4 ||
       z->bytes<((uint64_t)row+m)*V*4 || tid->bytes<(uint64_t)m*K*4 || tp->bytes<(uint64_t)m*K*4 ||
       rest->bytes<(uint64_t)m*4 || !(lam>0) || !(gap>=0)) return 0;
    ds4_gpu_tensor *pair=ds4_gpu_tensor_alloc((uint64_t)m*8),*extra=ds4_gpu_tensor_alloc((uint64_t)m*4);
    bool ok=pair && extra;
    if(ok) {
        keepmargin_prepare<<<m,1024,0,g_cur_stream>>>((int32_t*)pair->ptr,(float*)extra->ptr,(const float*)z->ptr,row,V,
            (const int32_t*)tid->ptr,(const float*)tp->ptr,K,(const unsigned char*)mask->ptr,lam,pmin,gap);
        ok=cuda_ok(cudaGetLastError(),"keepmargin prepare") && ds4_gpu_bwd_kl_topk_tensor(g,loss,z,row,m,V,tid,tp,rest,NULL,K,scale);
        if(ok) {
            keepmargin_apply<<<(m+255)/256,256,0,g_cur_stream>>>((float*)g->ptr,(float*)loss->ptr,(const int32_t*)pair->ptr,(const float*)extra->ptr,m,V,lam*scale);
            ok=cuda_ok(cudaGetLastError(),"keepmargin apply");
        }
    }
    if(pair) ds4_gpu_tensor_free(pair);
    if(extra) ds4_gpu_tensor_free(extra);
    return ok;
}
