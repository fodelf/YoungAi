/* diag=10：一次真实梯度之后冻结方向，只量权重重构和计算精度的影响。
 * 所有变体从同一raw快照构造，不重新反传、不累计缩放、不保存候选。
 * quality开关影响多个矩阵核；差异只能归到整体精度路径，不能直接归罪某个核。 */
#include "core_ptrain.h"
#ifndef DS4_NO_GPU

static void sc_drop(ds4_gpu_tensor **p) { if (*p) ds4_gpu_tensor_free(*p); *p = NULL; }

static bool sc_set(const pt_cfg *c, pt_run *r, ds4_gpu_tensor **a, ds4_gpu_tensor **b,
                   ds4_gpu_tensor *raw[DS4_MAX_LAYER][2], ds4_gpu_tensor *tmp, unsigned variant) {
    const uint64_t n = (uint64_t)c->rank * DS4_N_EMBD;
    bool ok = true;
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        ds4_gpu_tensor *all[] = {r->st.ampA[l],r->st.ampB[l]}, *parent[] = {a[l],b[l]};
        for (unsigned k = 0; ok && k < 2; k++) {
            ds4_gpu_tensor *w = pt_rows(all[k],r->k2,c->rank,DS4_N_EMBD);
            ds4_gpu_tensor *p = pt_rows(parent[k],r->k2,c->rank,DS4_N_EMBD);
            ok = w && p;
            if (ok && variant < 2) ok = ds4_gpu_tensor_copy(w,0,variant ? raw[l][k] : p,0,n*4);
            else if (ok) {
                const float scale = variant == 2 ? 1.f : .999667525291f;
                ok = ds4_gpu_tensor_copy(tmp,0,raw[l][k],0,n*4) && ds4_gpu_bwd_axpy_tensor(tmp,p,-1.f,n) &&
                     ds4_gpu_tensor_copy(w,0,p,0,n*4) && ds4_gpu_bwd_axpy_tensor(w,tmp,scale,n);
            }
            sc_drop(&w); sc_drop(&p);
        }
    }
    return ok;
}

/* 主机只比位模式；差的范数仍用GPU求。unit若不等于raw，先记证据，不跳过后续精度对照。 */
static bool sc_difference(const pt_cfg *c, pt_run *r, ds4_gpu_tensor *raw[DS4_MAX_LAYER][2],
                          ds4_gpu_tensor *tmp, uint64_t *changed, double *norm) {
    const uint64_t n = (uint64_t)c->rank * DS4_N_EMBD;
    uint32_t *x = xmalloc(n*4), *y = xmalloc(n*4); bool ok = true; double sum = 0;
    *changed = 0;
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        ds4_gpu_tensor *all[] = {r->st.ampA[l],r->st.ampB[l]};
        for (unsigned k = 0; ok && k < 2; k++) {
            ds4_gpu_tensor *w = pt_rows(all[k],r->k2,c->rank,DS4_N_EMBD); double ss = 0;
            ok = w && ds4_gpu_tensor_read(w,0,x,n*4) && ds4_gpu_tensor_read(raw[l][k],0,y,n*4) &&
                 ds4_gpu_tensor_copy(tmp,0,w,0,n*4) && ds4_gpu_bwd_axpy_tensor(tmp,raw[l][k],-1.f,n) &&
                 ds4_gpu_bwd_sumsq_tensor(tmp,n,&ss);
            if (ok) { for (uint64_t j = 0; j < n; j++) *changed += x[j] != y[j]; sum += ss; }
            sc_drop(&w);
        }
    }
    free(x); free(y); *norm = sqrt(sum);
    return ok && isfinite(*norm);
}

static bool sc_objective(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *const *ss,
                         uint32_t ns, uint32_t expected, double *mean, double *each, uint32_t *counts) {
    double loss = 0; uint32_t nt = 0; r->cont_s = NULL; r->grad_tokens = 0;
    if (!pt_loss_mode(c,true) || !pt_run_list(e,c,r,ss,ns,0,&loss,&nt,each,counts,NULL) || !nt || nt != expected) return false;
    *mean = loss / nt;
    return isfinite(*mean);
}

bool pt_stability(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *const *ss,
                  uint32_t ns, uint32_t nt, ds4_gpu_tensor **a, ds4_gpu_tensor **b) {
    if (c->diag != 10 || c->diag_save || ns != 8 || !nt || c->rank != 64) return false;
    ds4_gpu_tensor *raw[DS4_MAX_LAYER][2] = {{0}}, *tmp = NULL;
    bool ok = true, checks = true, stable[2] = {false,false};
    const uint64_t n = (uint64_t)c->rank * DS4_N_EMBD;
    char path[1200]; snprintf(path,sizeof path,"%s/stability.tsv",c->out); FILE *f = fopen(path,"w");
    snprintf(path,sizeof path,"%s/stability-docs.tsv",c->out); FILE *docs = fopen(path,"w");
    if (!f || !docs) ok = false;
    if (f) fprintf(f,"precision\tvariant\tloss\trepeat_error\tchanged_vs_raw\tfactor_delta_norm\n");
    if (docs) fprintf(docs,"precision\tvariant\tdoc\ttokens\tobjective\n");
    tmp = v41_alloc(n*4,&ok);
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        ds4_gpu_tensor *all[] = {r->st.ampA[l],r->st.ampB[l]};
        for (unsigned k = 0; ok && k < 2; k++) {
            raw[l][k] = v41_alloc(n*4,&ok);
            if (ok) ok = ds4_gpu_tensor_copy(raw[l][k],0,all[k],(uint64_t)r->k2*DS4_N_EMBD*4,n*4);
        }
    }
    const char *names[] = {"parent","raw","unit","micro"};
    for (unsigned quality = 0; ok && quality < 2; quality++) {
        ds4_gpu_set_quality(quality != 0);
        const char *precision = quality ? "FP32" : "TF32";
        double losses[4] = {0}; bool local_checks = true;
        for (unsigned v = 0; ok && v < 4; v++) {
            uint64_t changed = 0; double norm = 0, again = 0, each[8] = {0}; uint32_t counts[8] = {0};
            ok = sc_set(c,r,a,b,raw,tmp,v) && sc_difference(c,r,raw,tmp,&changed,&norm) &&
                 sc_objective(e,c,r,ss,ns,nt,&losses[v],each,counts) && sc_objective(e,c,r,ss,ns,nt,&again,NULL,NULL);
            if (!ok) break;
            const double repeat = fabs(again-losses[v]);
            if (repeat > 1e-7 || (v == 1 && changed) || (v == 2 && (changed || fabs(losses[2]-losses[1]) > 1e-7))) local_checks = false;
            fprintf(f,"%s\t%s\t%.12g\t%.12g\t%llu\t%.12g\n",precision,names[v],losses[v],repeat,(unsigned long long)changed,norm); fflush(f);
            for (uint32_t j = 0; j < ns; j++) fprintf(docs,"%s\t%s\t%u\t%u\t%.12g\n",precision,names[v],ss[j]->doc,counts[j],counts[j]?each[j]/counts[j]:NAN);
            fflush(docs);
            fprintf(stderr,"[stability] %s %s loss=%.12g repeat=%.12g changed=%llu factor_delta=%.12g\n",precision,names[v],losses[v],repeat,(unsigned long long)changed,norm);
            if (pt_mem_mb() >= 0 && pt_mem_mb() < 3000) ok = false;
        }
        double restored = 0;
        if (ok) ok = sc_set(c,r,a,b,raw,tmp,1) && sc_objective(e,c,r,ss,ns,nt,&restored,NULL,NULL);
        const double gain = losses[0]-losses[1], shift = losses[3]-losses[1];
        if (fabs(restored-losses[1]) > 1e-7) local_checks = false;
        checks = checks && local_checks;
        stable[quality] = ok && local_checks && gain > 0 && fabs(shift) <= .1*gain;
        fprintf(stderr,"[stability] %s raw_gain=%.12g micro_shift=%.12g shift_over_gain=%.12g restored_error=%.12g integrity=%s micro_stable=%s\n",
                precision,gain,shift,gain>0?fabs(shift)/gain:NAN,fabs(restored-losses[1]),local_checks?"PASS":"FAIL",stable[quality]?"PASS":"FAIL");
    }
    ds4_gpu_set_quality(false);
    snprintf(path,sizeof path,"%s/stability.verdict",c->out); FILE *verdict = fopen(path,"w");
    if (verdict) { fprintf(verdict,"execution=%s integrity=%s TF32=%s FP32=%s coverage_training=%s\n",ok?"PASS":"FAIL",checks?"PASS":"FAIL",stable[0]?"PASS":"FAIL",stable[1]?"PASS":"FAIL",ok&&checks&&stable[0]?"ELIGIBLE":"BLOCKED"); fclose(verdict); }
    else ok = false;
    for (uint32_t l = c->layer_lo; l <= c->layer_hi; l++) for (unsigned k = 0; k < 2; k++) sc_drop(&raw[l][k]);
    sc_drop(&tmp); if (f) fclose(f); if (docs) fclose(docs);
    return ok;
}

#endif
typedef int ds4_core_ptrain_stability_nonempty_tu;
