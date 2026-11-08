/* diag16：固定形状、单次Adam的保护项对照；diag17复用原1024完整留出口径。
 * 父本只决定掩码，标签仍是原FP教师。两臂都恢复47，只有预定candidate参与选轮。 */
#include "core_ptrain_split.h"
#include "ds4_gpu_keepmargin.h"
#include <sys/stat.h>
#ifndef DS4_NO_GPU
enum { KEEP_DOCS=8, KEEP_N=1022, KEEP_TOTAL=8176 };
static ds4_gpu_tensor *protect_mask;
const ds4_gpu_tensor *pt_protect_mask(void) { return protect_mask; }
static void kp_free(ds4_gpu_tensor **p) { if(*p) ds4_gpu_tensor_free(*p); *p=NULL; }

#ifndef __APPLE__
static bool kp_fixture(void) {
    const float z[]={.1f,0,-1,-1,.1f,0,-1,-1,.1f,0,-1,-1,.5f,0,-1,-1};
    const int32_t id[]={0,1,0,1,0,1,0,1};
    const float tp[]={.4f,.35f,.4f,.35f,.6f,.3f,.4f,.35f}, rest[]={.25f,.25f,.1f,.25f};
    const unsigned char keep[]={1,0,1,1};
    const size_t sizes[]={sizeof z,sizeof id,sizeof tp,sizeof rest,sizeof keep,sizeof z,sizeof z,16,16};
    ds4_gpu_tensor *p[9]={0}; bool ok=true; float a[16],b[16],la[4],lb[4];
    for(unsigned i=0;i<9;i++) p[i]=v41_alloc(sizes[i],&ok);
    if(ok) ok=ds4_gpu_tensor_write(p[0],0,z,sizeof z) && ds4_gpu_tensor_write(p[1],0,id,sizeof id) &&
        ds4_gpu_tensor_write(p[2],0,tp,sizeof tp) && ds4_gpu_tensor_write(p[3],0,rest,sizeof rest) &&
        ds4_gpu_tensor_write(p[4],0,keep,sizeof keep) && ds4_gpu_bwd_kl_margin(.5f,.5f,.3f) &&
        ds4_gpu_bwd_kl_topk_tensor(p[5],p[7],p[0],0,4,4,p[1],p[2],p[3],NULL,2,.25f) &&
        ds4_gpu_bwd_kl_keepmargin_tensor(p[6],p[8],p[0],0,4,4,p[1],p[2],p[3],2,.25f,p[4],.5f,.5f,.3f) &&
        ds4_gpu_tensor_read(p[5],0,a,sizeof a) && ds4_gpu_tensor_read(p[6],0,b,sizeof b) &&
        ds4_gpu_tensor_read(p[7],0,la,sizeof la) && ds4_gpu_tensor_read(p[8],0,lb,sizeof lb);
    double worst=0;
    for(unsigned i=0;ok && i<16;i++) {
        const double expected=i==0?-.125:i==1?.125:0;
        worst=fmax(worst,fabs((double)b[i]-a[i]-expected));
    }
    for(unsigned i=0;ok && i<4;i++) worst=fmax(worst,fabs((double)lb[i]-la[i]-(i?0:.5*(log(.4/.35)-.1))));
    ok=ok && isfinite(worst) && worst<1e-6;
    fprintf(stderr,"[keep] known_answer max_error=%.12g %s\n",worst,ok?"PASS":"FAIL");
    for(unsigned i=0;i<9;i++) kp_free(&p[i]);
    return ok;
}
#endif

static bool kp_restore(const pt_cfg *c,pt_run *r,ds4_gpu_tensor *snap[DS4_MAX_LAYER][2]) {
    const uint64_t bytes=(uint64_t)r->K*DS4_N_EMBD*4;
    for(unsigned l=c->layer_lo;l<=c->layer_hi;l++) {
        if(!ds4_gpu_tensor_copy(r->st.ampA[l],0,snap[l][0],0,bytes) || !ds4_gpu_tensor_copy(r->st.ampB[l],0,snap[l][1],0,bytes)) return false;
        ds4_gpu_tensor *zero[]={r->gA[l],r->gB[l],r->mA[l],r->mB[l],r->vA[l],r->vB[l]};
        for(unsigned i=0;i<6;i++) if(!ds4_gpu_tensor_fill_f32(zero[i],0.f,bytes/4)) return false;
    }
    r->cont_s=NULL; r->grad_tokens=0; pt_state_reset(&r->st); return pt_diag_weights(c,r);
}
static bool kp_frozen(const pt_cfg *c,pt_run *r,ds4_gpu_tensor *snap[DS4_MAX_LAYER][2]) {
    const size_t bytes=(size_t)r->k2*DS4_N_EMBD*4;
    if(!bytes) return true;
    void *a=xmalloc(bytes),*b=xmalloc(bytes); bool ok=true;
    for(unsigned l=c->layer_lo;ok && l<=c->layer_hi;l++) {
        ds4_gpu_tensor *p[]={r->st.ampA[l],r->st.ampB[l]};
        for(unsigned k=0;ok && k<2;k++) ok=ds4_gpu_tensor_read(p[k],0,a,bytes) && ds4_gpu_tensor_read(snap[l][k],0,b,bytes) && !memcmp(a,b,bytes);
    }
    free(a);free(b);return ok;
}
static bool kp_score(ds4_engine *e,const pt_cfg *c,pt_run *r,const pt_sample *s,int32_t *ids,double *loss) {
    pt_state_reset(&r->st);r->cont_s=NULL;r->grad_tokens=0;r->st.ced_skip=0;r->st.head_last_only=0;
    if(!v41_forward(e,&r->st,s->sids,s->sn)) return false;
    bool ok=true; const uint64_t bytes=(uint64_t)KEEP_N*DS4_N_VOCAB*4;
    ds4_gpu_tensor *scores=v41_alloc(bytes,&ok),*g=NULL;
    if(ok) ok=ds4_gpu_tensor_copy(scores,0,r->st.logits,0,bytes) &&
        ds4_gpu_bwd_topk_tensor(r->tid,r->tp,r->trest,scores,0,KEEP_N,DS4_N_VOCAB,1) && ds4_gpu_tensor_read(r->tid,0,ids,KEEP_N*4);
    kp_free(&scores); uint32_t nt=0; *loss=0;
    if(ok) ok=pt_loss_rows(c,r,r->st.logits,0,s,0,loss,&nt,&g) && nt==KEEP_N;
    kp_free(&g);return ok;
}
static bool kp_objective(ds4_engine *e,const pt_cfg *c,pt_run *r,pt_sample *s,ds4_gpu_tensor **mask,bool enabled,int grad,double *mean) {
    double sum=0;uint32_t nt=0; bool ok=pt_loss_mode(c,true);
    r->grad_tokens=KEEP_TOTAL;
    for(unsigned i=0;ok && i<KEEP_DOCS;i++) {
        protect_mask=enabled?mask[i]:NULL;r->cont_s=NULL;
        ok=r->grad_tokens==KEEP_TOTAL && pt_step_sample(e,c,r,&s[i],grad,&sum,&nt);
        fprintf(stderr,"[keep] %s grad=%d doc=%u rows=%u ok=%u\n",enabled?"candidate":"control",grad,i,s[i].sn,ok);
    }
    protect_mask=NULL;*mean=sum/KEEP_TOTAL;
    return ok && nt==KEEP_TOTAL && isfinite(*mean);
}

bool pt_protectcheck(ds4_engine *e,const pt_cfg *c,pt_run *r,const pt_data *d) {
    if(c->diag==17) {
        if(c->maxlen!=1024 || c->epochs || c->max_steps || c->diag_save || c->sample_n) return false;
        double kl=0; const bool eval=pt_loss_mode(c,false) && pt_eval(e,c,r,d,2,d->ns,&kl,0,NULL,NULL,NULL) && isfinite(kl) && kl>0;
        const bool weights=pt_diag_weights(c,r);
        fprintf(stderr,"[keep-eval] full_dev_KL=%.12g weights_equal=%u ok=%u\n",kl,weights,eval&&weights);return eval&&weights;
    }
#ifdef __APPLE__
    return false;
#else
    if(c->diag!=16 || c->maxlen!=2048 || c->max_steps!=1 || c->rank!=64 || c->layer_lo || c->layer_hi!=39 ||
       !c->token_mean || !c->nopack || c->sample_n || c->rankprob || c->margin!=.5f || c->margin_p1!=.5f || c->margin_gap!=.3f ||
       c->lr!=5e-6f || (c->lr_b && c->lr_b!=c->lr)) return false;
    pt_sample samples[KEEP_DOCS];bool found[KEEP_DOCS]={0};
    for(unsigned i=0;i<d->ns;i++) {
        const pt_sample *s=&d->s[i]; const unsigned doc=s->doc;
        if(doc<KEEP_DOCS && !found[doc] && s->raw && !s->eval && !s->src && !s->l0 && s->sa0==1 && s->m>=KEEP_N && s->top_id && d->docn[doc]>KEEP_N) {
            samples[doc]=*s;samples[doc].sids=d->docids[doc];samples[doc].m=KEEP_N;
            samples[doc].sn=d->docn[doc]-1<2048?d->docn[doc]-1:2048; found[doc]=true;
        }
    }
    for(unsigned i=0;i<KEEP_DOCS;i++) if(!found[i]) return false;
    char path[1200];snprintf(path,sizeof path,"%s/protect.tsv",c->out);FILE *f=fopen(path,"w");if(!f)return false;
    fprintf(f,"stage\tdoc\tpos\tteacher\tstudent\tbase\tkeep\tp1\n");
    int32_t base[KEEP_DOCS][KEEP_N],ids[KEEP_N],rep[KEEP_N];unsigned char keep[KEEP_N];
    ds4_gpu_tensor *mask[KEEP_DOCS]={0},*snap[DS4_MAX_LAYER][2]={{0}};bool ok=kp_fixture() && pt_diag_weights(c,r) && pt_loss_mode(c,false);
    const uint64_t bytes=(uint64_t)r->K*DS4_N_EMBD*4;
    for(unsigned l=c->layer_lo;ok && l<=c->layer_hi;l++) for(unsigned k=0;ok && k<2;k++) {
        snap[l][k]=v41_alloc(bytes,&ok);if(ok)ok=ds4_gpu_tensor_copy(snap[l][k],0,k?r->st.ampB[l]:r->st.ampA[l],0,bytes);
    }
    unsigned total_keep=0,low_keep=0;
    for(unsigned doc=0;ok && doc<KEEP_DOCS;doc++) {
        double a=0,b=0;ok=kp_score(e,c,r,&samples[doc],base[doc],&a) && kp_score(e,c,r,&samples[doc],rep,&b) && !memcmp(base[doc],rep,sizeof rep) && a==b;
        for(unsigned j=0;ok && j<KEEP_N;j++) {
            unsigned best=0;const size_t at=(size_t)j*c->topk;
            for(unsigned k=1;k<c->topk;k++)if(samples[doc].top_p[at+k]>samples[doc].top_p[at+best])best=k;
            const int32_t t=samples[doc].top_id[at+best];const float p=samples[doc].top_p[at+best];
            keep[j]=base[doc][j]==t;total_keep+=keep[j];low_keep+=keep[j] && p<.5f;
            fprintf(f,"base\t%u\t%u\t%d\t%d\t%d\t%u\t%.9g\n",doc,j,t,base[doc][j],base[doc][j],keep[j],p);
        }
        mask[doc]=v41_alloc(KEEP_N,&ok);if(ok)ok=ds4_gpu_tensor_write(mask[doc],0,keep,KEEP_N);
    }
    fprintf(stderr,"[keep] frozen_base same=%u total=%u low_conf_correct=%u repeat=%u\n",total_keep,KEEP_TOTAL,low_keep,ok);
    const char *names[]={"control","candidate"};unsigned updates=0;
    for(unsigned arm=0;ok && arm<2;arm++) {
        double pre=0,post=0,gn=0;int clipped=0;
        ok=kp_restore(c,r,snap) && kp_objective(e,c,r,samples,mask,arm,1,&pre);
        if(ok) {ok=pt_opt_step(c,r,1,&gn,&clipped);if(ok)updates++;}
        if(ok)ok=kp_frozen(c,r,snap) && kp_objective(e,c,r,samples,mask,arm,0,&post);
        const bool decreased=ok && post<pre;
        fprintf(stderr,"[keep] %s objective_before=%.12g after=%.12g gnorm=%.12g clip=%d decreased=%u\n",names[arm],pre,post,gn,clipped,decreased);
        unsigned hit=0,fixed=0,broken=0;
        if(ok)ok=pt_loss_mode(c,false);
        for(unsigned doc=0;ok && doc<KEEP_DOCS;doc++) {
            double a=0,b=0;ok=kp_score(e,c,r,&samples[doc],ids,&a) && kp_score(e,c,r,&samples[doc],rep,&b) && !memcmp(ids,rep,sizeof ids) && a==b;
            for(unsigned j=0;ok && j<KEEP_N;j++) {
                unsigned best=0;const size_t at=(size_t)j*c->topk;
                for(unsigned k=1;k<c->topk;k++)if(samples[doc].top_p[at+k]>samples[doc].top_p[at+best])best=k;
                const int32_t t=samples[doc].top_id[at+best];const bool was=base[doc][j]==t,now=ids[j]==t;
                hit+=now;fixed+=!was&&now;broken+=was&&!now;
                fprintf(f,"%s\t%u\t%u\t%d\t%d\t%d\t%u\t%.9g\n",names[arm],doc,j,t,ids[j],base[doc][j],was,samples[doc].top_p[at+best]);
            }
        }
        fprintf(stderr,"[keep] %s same=%u total=%u fixed=%u broken=%u post_repeat=%u\n",names[arm],hit,KEEP_TOTAL,fixed,broken,ok);
        snprintf(path,sizeof path,"%s/%s",c->out,names[arm]);
        if(ok && (arm==0 || decreased)) ok=mkdir(path,0755)==0 && pt_save_amp(c,r,path);
        else if(arm==1) ok=false;
        fflush(f);
    }
    protect_mask=NULL;bool restored=false;
    if(snap[c->layer_hi][1])restored=kp_restore(c,r,snap);
    fprintf(stderr,"[keep] restored_parent=%u optimizer_steps=%u success=%u\n",restored,updates,ok);
    for(unsigned i=0;i<KEEP_DOCS;i++)kp_free(&mask[i]);
    for(unsigned l=c->layer_lo;l<=c->layer_hi;l++)for(unsigned k=0;k<2;k++)kp_free(&snap[l][k]);
    fclose(f);return ok && restored;
#endif
}
#endif
typedef int ds4_core_ptrain_protect_nonempty_tu;
