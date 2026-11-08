/* TIE-KD 概率排序的 k=1 特例；不改变前向或原 KL 核。
 * 教师仅有 top-K：学生榜首在表外时用末位概率作上界；无法证明严格排序则不加项。
 * 学生并列而 argmax 选到非教师词时取半梯度，与 torch.maximum 的折点约定一致。 */
#include "ds4_gpu_rankprob.h"
#include "cuda_bwd_keepmargin.inc.cu"

__global__ static void rankprob_kernel(float *g, float *loss, const float *z, uint32_t row0,
    uint32_t V, const int32_t *tid, const float *tp, const float *w, uint32_t K, float scale, float lam) {
    __shared__ float sh[32], pj, pt, delta, mult;
    __shared__ int32_t jbest, target;
    const uint32_t r = blockIdx.x;
    const float *zr = z + (uint64_t)(row0 + r) * V;
    float mx = -INFINITY;
    for (uint32_t k = threadIdx.x; k < V; k += blockDim.x) mx = fmaxf(mx, zr[k]);
    mx = bwd_block_max(mx, sh);
    if (threadIdx.x == 0) jbest = 0x7fffffff;
    __syncthreads();
    int32_t local = 0x7fffffff;
    for (uint32_t k = threadIdx.x; k < V; k += blockDim.x)
        if (zr[k] == mx && (int32_t)k < local) local = (int32_t)k;
    if (local != 0x7fffffff) atomicMin(&jbest, local);
    __syncthreads();
    float sum = 0.f;
    for (uint32_t k = threadIdx.x; k < V; k += blockDim.x) sum += expf(zr[k] - mx);
    sum = bwd_block_sum(sum, sh);
    if (threadIdx.x == 0) {
        float best = -1.f, lower = INFINITY, qj = -1.f;
        target = -1;
        for (uint32_t k = 0; k < K; k++) {
            const int32_t id = tid[(uint64_t)r * K + k];
            const float q = tp[(uint64_t)r * K + k];
            if (id < 0 || (uint32_t)id >= V) continue;
            if (q > best) { best = q; target = id; }
            lower = fminf(lower, q);
            if (id == jbest) qj = q;
        }
        /* 表外词的教师概率未知，不能把整个余量桶当作该词的概率。 */
        if (qj < 0.f) qj = lower;
        pj = 1.f / sum;
        pt = target >= 0 ? expf(zr[target] - mx) / sum : 0.f;
        delta = fmaxf(0.f, pj - pt);
        const bool on = target >= 0 && target != jbest && best > qj;
        const float wi = w ? w[r] : 1.f;
        mult = on ? lam * wi * scale * (delta == 0.f ? 0.5f : 1.f) : 0.f;
        loss[r] = on ? lam * wi * delta : 0.f;
    }
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < V; k += blockDim.x) {
        const float pk = expf(zr[k] - mx) / sum;
        g[(uint64_t)r * V + k] = mult * (((int32_t)k == jbest ? pj : 0.f) -
            ((int32_t)k == target ? pt : 0.f) - delta * pk);
    }
}

int ds4_gpu_bwd_kl_rankprob_tensor(ds4_gpu_tensor *g, ds4_gpu_tensor *loss, const ds4_gpu_tensor *logits,
    uint32_t row0, uint32_t m, uint32_t V, const ds4_gpu_tensor *tid, const ds4_gpu_tensor *tp,
    const ds4_gpu_tensor *rest, const ds4_gpu_tensor *w, uint32_t K, float scale, float lam) {
    if (!isfinite(lam) || lam < 0.f) return 0;
    if (lam == 0.f) return ds4_gpu_bwd_kl_topk_tensor(g, loss, logits, row0, m, V, tid, tp, rest, w, K, scale);
    if (!g || !loss || !logits || !tid || !tp || !rest || !m || !V || !K || K > 128u || V > 131072u ||
        g->bytes < (uint64_t)m * V * 4 || loss->bytes < (uint64_t)m * 4 ||
        logits->bytes < ((uint64_t)row0 + m) * V * 4 || tid->bytes < (uint64_t)m * K * 4 ||
        tp->bytes < (uint64_t)m * K * 4 || rest->bytes < (uint64_t)m * 4 || (w && w->bytes < (uint64_t)m * 4)) return 0;
    ds4_gpu_tensor *rg = ds4_gpu_tensor_alloc((uint64_t)m * V * 4), *rl = ds4_gpu_tensor_alloc((uint64_t)m * 4);
    bool ok = rg && rl;
    if (ok) {
        rankprob_kernel<<<m, 512, 0, g_cur_stream>>>((float *)rg->ptr, (float *)rl->ptr, (const float *)logits->ptr,
            row0, V, (const int32_t *)tid->ptr, (const float *)tp->ptr, w ? (const float *)w->ptr : NULL, K, scale, lam);
        ok = cuda_ok(cudaGetLastError(), "probability rank") &&
            ds4_gpu_bwd_kl_topk_tensor(g, loss, logits, row0, m, V, tid, tp, rest, w, K, scale) &&
            ds4_gpu_bwd_axpy_tensor(g, rg, 1.f, (uint64_t)m * V) && ds4_gpu_bwd_axpy_tensor(loss, rl, 1.f, m);
    }
    if (rg) ds4_gpu_tensor_free(rg);
    if (rl) ds4_gpu_tensor_free(rl);
    return ok;
}
