/* diag15：只量原始logit领先幅度，区分排序资格与实际激活。
 * top-K核会覆盖logits，必须在副本上找竞争词；原件保留给标量读回。
 * 全部固定训练位置都导出，不根据历史改坏位置挑样本。 */
#include "core_ptrain_split.h"
#ifndef DS4_NO_GPU
enum { MARGIN_DOCS=8, MARGIN_N=1022 };
typedef struct {
    int32_t teacher, first, second, competitor;
    float p1, p2, target, z_teacher, z_competitor, margin;
} margin_row;

static bool margin_finite(float x) {
    uint32_t bits; memcpy(&bits,&x,4);
    return (bits & 0x7f800000u)!=0x7f800000u;
}

static bool margin_predict(ds4_engine *e, const pt_cfg *c, pt_run *r,
                           const pt_sample *s, margin_row *out) {
    pt_state_reset(&r->st); r->cont_s=NULL; r->grad_tokens=0;
    r->st.ced_skip=0; r->st.head_last_only=0;
    if (!v41_forward(e,&r->st,s->sids,s->sn)) return false;
    bool ok=true;
    const uint64_t bytes=(uint64_t)MARGIN_N*DS4_N_VOCAB*4;
    ds4_gpu_tensor *copy=v41_alloc(bytes,&ok);
    int32_t ids[MARGIN_N*2];
    if (ok) ok=ds4_gpu_tensor_copy(copy,0,r->st.logits,0,bytes) &&
        ds4_gpu_bwd_topk_tensor(r->tid,r->tp,r->trest,copy,0,MARGIN_N,DS4_N_VOCAB,2) &&
        ds4_gpu_tensor_read(r->tid,0,ids,sizeof ids);
    if (copy) ds4_gpu_tensor_free(copy);
    memset(out,0,sizeof(*out)*MARGIN_N);
    for (unsigned j=0;ok && j<MARGIN_N;j++) {
        const size_t at=(size_t)j*c->topk; unsigned best=0;
        for (unsigned k=1;k<c->topk;k++) if (s->top_p[at+k]>s->top_p[at+best]) best=k;
        margin_row *v=&out[j];
        v->teacher=s->top_id[at+best]; v->p1=s->top_p[at+best];
        for (unsigned k=0;k<c->topk;k++) if(k!=best && s->top_p[at+k]>v->p2) v->p2=s->top_p[at+k];
        v->first=ids[j*2]; v->second=ids[j*2+1];
        v->competitor=v->first==v->teacher ? v->second : v->first;
        if (v->teacher<0 || v->first<0 || v->second<0 ||
            v->teacher>=(int32_t)DS4_N_VOCAB || v->first>=(int32_t)DS4_N_VOCAB || v->second>=(int32_t)DS4_N_VOCAB) return false;
        const uint64_t row=(uint64_t)j*DS4_N_VOCAB;
        ok=ds4_gpu_tensor_read(r->st.logits,(row+v->teacher)*4,&v->z_teacher,4) &&
           ds4_gpu_tensor_read(r->st.logits,(row+v->competitor)*4,&v->z_competitor,4);
        v->margin=v->z_teacher-v->z_competitor;
        v->target=fminf(c->margin_gap,logf(v->p1/fmaxf(v->p2,1e-6f)));
        ok=ok && margin_finite(v->z_teacher) && margin_finite(v->z_competitor) &&
            margin_finite(v->margin) && margin_finite(v->target);
    }
    return ok;
}

bool pt_margincheck(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d) {
    if (c->epochs || c->diag_save || c->sample_n || c->max_steps || !c->init[0] ||
        !c->token_mean || !c->nopack || c->maxlen!=1024 || c->rank!=64 ||
        c->topk!=20 || c->layer_lo || c->layer_hi!=39 || c->margin!=.5f ||
        c->margin_p1!=.5f || c->margin_gap!=.3f) return false;
    const pt_sample *samples[MARGIN_DOCS]={0};
    for (uint32_t i=0;i<d->ns;i++) {
        const pt_sample *s=&d->s[i];
        if (s->doc<MARGIN_DOCS && !samples[s->doc] && s->raw && !s->eval && s->src==0 &&
            s->sn==MARGIN_N && s->m==MARGIN_N && !s->l0 && s->sa0==1 && s->top_id)
            samples[s->doc]=s;
    }
    for (unsigned j=0;j<MARGIN_DOCS;j++) if(!samples[j]) return false;
    char path[1200]; snprintf(path,sizeof path,"%s/margin.tsv",c->out);
    FILE *f=fopen(path,"w"); if(!f) return false;
    fprintf(f,"repeat\tdoc\tpos\tteacher\tfirst\tsecond\tcompetitor\tp1\tp2\ttarget_gap\tz_teacher\tz_competitor\tmargin\teligible\tactive\tgap_violation\n");
    margin_row *first=xmalloc(sizeof(*first)*MARGIN_DOCS*MARGIN_N);
    margin_row current[MARGIN_N];
    bool ok=pt_diag_weights(c,r) && pt_loss_mode(c,false); uint32_t diff=0;
    for (unsigned rep=0;ok && rep<2;rep++) for(unsigned doc=0;ok && doc<MARGIN_DOCS;doc++) {
        ok=margin_predict(e,c,r,samples[doc],current);
        for(unsigned j=0;ok && j<MARGIN_N;j++) {
            margin_row *v=&current[j], *saved=&first[doc*MARGIN_N+j];
            if(!rep) *saved=*v; else if(memcmp(saved,v,sizeof *v)) diff++;
            const int eligible=v->p1>=c->margin_p1, violation=v->margin<v->target;
            fprintf(f,"%u\t%u\t%u\t%d\t%d\t%d\t%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\t%d\t%d\n",
                rep,doc,j,v->teacher,v->first,v->second,v->competitor,v->p1,v->p2,v->target,
                v->z_teacher,v->z_competitor,v->margin,eligible,eligible&&violation,violation);
        }
        fflush(f); fprintf(stderr,"[margin] repeat=%u doc=%u ok=%u differing_rows=%u\n",rep,doc,ok,diff);
    }
    const bool weights=pt_diag_weights(c,r); free(first);
    if(fclose(f)) ok=false;
    snprintf(path,sizeof path,"%s/margin.verdict",c->out); f=fopen(path,"w");
    if(!f) return false;
    const bool pass=ok && weights && !diff;
    fprintf(f,"%s repeat_different_rows=%u weights_equal=%u optimizer_steps=0\n",pass?"PASS":"FAIL",diff,weights);
    if(fclose(f)) return false;
    return pass;
}
#endif
typedef int ds4_core_ptrain_margin_nonempty_tu;
