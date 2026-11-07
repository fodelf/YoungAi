/* diag14：只分解原训练目标，不更新参数。预测只决定分组，监督始终取原FP教师。
 * 掩码在原损失/排序项算完后作用于出口梯度；所有组共用总token分母。
 * bf16反传未必保持梯度可加性，因此必须另量full与keep+repair的重构差。 */
#include "core_ptrain_split.h"
#include <sys/stat.h>
#ifndef DS4_NO_GPU
enum { SPLIT_DOCS=8, SPLIT_N=1022 };
static const unsigned char *split_keep;
static int split_group;

bool pt_split_mask(const pt_cfg *c, pt_run *r, ds4_gpu_tensor *g, uint32_t off, uint32_t m) {
    if (c->diag!=14 || !split_keep || split_group==0) return true;
    if (off || m!=SPLIT_N) return false;
    /* 连续非本组行一起置零，不改变前向/教师表/分母，也不在CPU上算矩阵。 */
    for (uint32_t j=0;j<m;) {
        const int active=split_group==1 ? split_keep[j] : !split_keep[j];
        if (active) { j++; continue; }
        const uint32_t begin=j;
        while (j<m && !(split_group==1 ? split_keep[j] : !split_keep[j])) r->loss_h[j++]=0;
        ds4_gpu_tensor *v=pt_rows(g,begin,j-begin,DS4_N_VOCAB);
        const bool ok=v && ds4_gpu_tensor_fill_f32(v,0.f,(uint64_t)(j-begin)*DS4_N_VOCAB);
        if (v) ds4_gpu_tensor_free(v);
        if (!ok) return false;
    }
    return true;
}

static bool split_predict(ds4_engine *e, pt_run *r, const pt_sample *s, int32_t *ids) {
    pt_state_reset(&r->st); r->cont_s=NULL;
    r->st.ced_skip=0; r->st.head_last_only=0;
    return v41_forward(e,&r->st,s->sids,s->sn) &&
        ds4_gpu_bwd_topk_tensor(r->tid,r->tp,r->trest,r->st.logits,0,s->m,DS4_N_VOCAB,1) &&
        ds4_gpu_tensor_read(r->tid,0,ids,(uint64_t)s->m*4);
}

static bool split_zero(const pt_cfg *c, pt_run *r) {
    for (uint32_t l=c->layer_lo;l<=c->layer_hi;l++)
        if (!ds4_gpu_tensor_fill_f32(r->gA[l],0.f,(uint64_t)r->K*DS4_N_EMBD) ||
            !ds4_gpu_tensor_fill_f32(r->gB[l],0.f,(uint64_t)r->K*DS4_N_EMBD)) return false;
    return true;
}

bool pt_splitcheck(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d) {
    if (c->epochs || c->diag_save || c->sample_n || c->max_steps || !c->init[0] ||
        !c->token_mean || !c->nopack || c->maxlen!=1024 || c->rank!=64 || c->layer_lo || c->layer_hi!=39) return false;
    const pt_sample *samples[SPLIT_DOCS]={0};
    for (uint32_t i=0;i<d->ns;i++) {
        const pt_sample *s=&d->s[i];
        if (s->doc<SPLIT_DOCS && !samples[s->doc] && s->raw && !s->eval && s->src==0 &&
            s->sn==SPLIT_N && s->m==SPLIT_N && !s->l0 && s->sa0==1 && s->top_id)
            samples[s->doc]=s;
    }
    for (unsigned j=0;j<SPLIT_DOCS;j++) if (!samples[j]) return false;
    char path[1200]; snprintf(path,sizeof path,"%s/split.tsv",c->out);
    FILE *f=fopen(path,"w"); if (!f) return false;
    unsigned char keep[SPLIT_DOCS][SPLIT_N]; int32_t base[SPLIT_DOCS][SPLIT_N], ids[SPLIT_N];
    uint32_t n_keep=0;
    bool ok=pt_diag_weights(c,r) && pt_loss_mode(c,false);
    for (unsigned j=0;ok && j<SPLIT_DOCS;j++) {
        const pt_sample *s=samples[j];
        ok=split_predict(e,r,s,base[j]) && split_predict(e,r,s,ids) && !memcmp(ids,base[j],sizeof ids);
        uint32_t good=0;
        for (uint32_t t=0;ok && t<SPLIT_N;t++) {
            const size_t at=(size_t)t*c->topk; uint32_t best=0;
            for (uint32_t k=1;k<c->topk;k++) if (s->top_p[at+k]>s->top_p[at+best]) best=k;
            keep[j][t]=base[j][t]==s->top_id[at+best]; good+=keep[j][t];
        }
        n_keep+=good; fprintf(f,"partition\tdoc=%u\tkeep=%u\trepair=%u\trepeat=%u\n",j,good,SPLIT_N-good,ok); fflush(f);
    }
    const uint32_t total=SPLIT_DOCS*SPLIT_N;
    ok=ok && n_keep>0 && n_keep<total;
    if (ok) {
        snprintf(path,sizeof path,"%s/partition.bin",c->out);
        FILE *p=fopen(path,"wb"); const uint32_t hd[]={SPLIT_DOCS,SPLIT_N};
        ok=p && fwrite(hd,sizeof hd,1,p)==1 && fwrite(base,sizeof base,1,p)==1 && fwrite(keep,sizeof keep,1,p)==1;
        if(p && fclose(p)) ok=false;
    }
    fprintf(f,"counts\tall=%u\tkeep=%u\trepair=%u\n",total,n_keep,total-n_keep); fflush(f);
    const char *names[]={"full","keep","repair"};
    double objectives[2][3]={{0}};
    for (unsigned rep=0;ok && rep<2;rep++) for (int group=0;ok && group<3;group++) {
        ok=split_zero(c,r) && pt_loss_mode(c,true);
        r->grad_tokens=total; split_group=group;
        double loss=0; uint32_t nt=0;
        for (unsigned j=0;ok && j<SPLIT_DOCS;j++) {
            split_keep=keep[j]; r->cont_s=NULL;
            ok=r->grad_tokens==total && pt_step_sample(e,c,r,samples[j],1,&loss,&nt) && r->grad_tokens==total;
            fprintf(stderr,"[split] %s_%u doc=%u ok=%u\n",names[group],rep,j,ok);
        }
        split_keep=NULL; split_group=0;
        objectives[rep][group]=loss/total;
        pt_cfg out=*c;
        const int pathlen=snprintf(out.out,sizeof out.out,"%s/%s_%u",c->out,names[group],rep);
        if(pathlen<0 || (size_t)pathlen>=sizeof out.out) ok=false;
        if (ok) ok=nt==total && mkdir(out.out,0755)==0 && pt_diag_gradient(&out,r,f) && pt_diag_weights(c,r);
        fprintf(f,"pass\t%s_%u\tdenominator=%u\tselected=%u\tobjective=%.12g\tok=%u\n",
                names[group],rep,total,group==0?total:group==1?n_keep:total-n_keep,loss/total,ok); fflush(f);
    }
    split_keep=NULL; split_group=0;
    for (unsigned rep=0;rep<2;rep++) {
        const double err=fabs(objectives[rep][0]-objectives[rep][1]-objectives[rep][2]);
        fprintf(f,"loss_additivity\trepeat=%u\terror=%.12g\n",rep,err);
        if (!(err<1e-7)) ok=false;
    }
    if (ok) ok=pt_loss_mode(c,false);
    for (unsigned j=0;ok && j<SPLIT_DOCS;j++)
        ok=split_predict(e,r,samples[j],ids) && !memcmp(ids,base[j],sizeof ids);
    const bool weights=pt_diag_weights(c,r);
    fprintf(f,"verdict\t%s\tweights_equal=%u\tpost_predictions_equal=%u\toptimizer_steps=0\n",ok&&weights?"PASS":"FAIL",weights,ok);
    fclose(f); return ok&&weights;
}
#endif
typedef int ds4_core_ptrain_split_nonempty_tu;
