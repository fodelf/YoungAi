/* diag16：单步保护项对照；diag17：原1024完整留出；diag18：完整父本上的末层连续训练。
 * 父本只决定掩码，标签仍是原FP教师；末层模式也完整加载/保存40层，避免漏掉其他层已有补丁。 */
#include "core_ptrain_split.h"
#include "ds4_gpu_keepmargin.h"
#include <sys/stat.h>
#ifndef DS4_NO_GPU
enum { KEEP_DOCS=8, KEEP_N=1022, KEEP_TOTAL=8176 };
/* 独立实验构建声明父本计数，防止接力新父本时沿用旧轮次的准入值。 */
#ifndef PT_TAIL_EXPECT_SAME
#define PT_TAIL_EXPECT_SAME 7343
#endif
#ifndef PT_TAIL_EXPECT_LOW
#define PT_TAIL_EXPECT_LOW 618
#endif
#ifndef PT_TAIL_RETAIN
#define PT_TAIL_RETAIN 0
#endif
#ifndef PT_TAIL_ALLKEEP
#define PT_TAIL_ALLKEEP 0
#endif
#ifndef PT_TAIL_EXPECT_OBJECTIVE
#define PT_TAIL_EXPECT_OBJECTIVE 0.
#endif
#ifndef PT_TAIL_DIRECTION
#define PT_TAIL_DIRECTION 0
#endif
#ifndef PT_TAIL_RADIUS
#define PT_TAIL_RADIUS 0.
#endif
#ifndef PT_TAIL_BACKTRACK
#define PT_TAIL_BACKTRACK 0
#endif
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
        ds4_gpu_bwd_kl_keepmargin_tensor(p[6],p[8],p[0],0,4,4,p[1],p[2],p[3],2,.25f,p[4],.5f,PT_TAIL_ALLKEEP?1.01f:.5f,.3f) &&
        ds4_gpu_tensor_read(p[5],0,a,sizeof a) && ds4_gpu_tensor_read(p[6],0,b,sizeof b) &&
        ds4_gpu_tensor_read(p[7],0,la,sizeof la) && ds4_gpu_tensor_read(p[8],0,lb,sizeof lb);
    double worst=0;
    for(unsigned i=0;ok && i<16;i++) {
        const double expected=(i==0 || (PT_TAIL_ALLKEEP && i==8))?-.125:(i==1 || (PT_TAIL_ALLKEEP && i==9))?.125:0;
        worst=fmax(worst,fabs((double)b[i]-a[i]-expected));
    }
    for(unsigned i=0;ok && i<4;i++) {
        const double expected=i==0?.5*(log(.4/.35)-.1):(PT_TAIL_ALLKEEP && i==2)?.1:0;
        worst=fmax(worst,fabs((double)lb[i]-la[i]-expected));
    }
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
    const size_t capacity=(size_t)(c->diag==18?r->K:r->k2)*DS4_N_EMBD*4;
    if(!capacity) return true;
    void *a=xmalloc(capacity),*b=xmalloc(capacity); bool ok=true;
    for(unsigned l=c->layer_lo;ok && l<=c->layer_hi;l++) {
        const size_t bytes=(size_t)(c->diag==18 && l<c->layer_hi?r->K:r->k2)*DS4_N_EMBD*4;
        if(!bytes)continue;
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
    pt_cfg step_cfg=*c;
    /* 只改变反传入口；分配、父本加载和落盘仍使用完整40层配置。 */
    if(c->diag==18)step_cfg.layer_lo=c->layer_hi;
    r->grad_tokens=KEEP_TOTAL;
    for(unsigned i=0;ok && i<KEEP_DOCS;i++) {
        protect_mask=enabled?mask[i]:NULL;r->cont_s=NULL;
        ok=r->grad_tokens==KEEP_TOTAL && pt_step_sample(e,&step_cfg,r,&s[i],grad,&sum,&nt);
        fprintf(stderr,"[keep] %s grad=%d doc=%u rows=%u ok=%u\n",enabled?"candidate":"control",grad,i,s[i].sn,ok);
    }
    protect_mask=NULL;*mean=sum/KEEP_TOTAL;
    return ok && nt==KEEP_TOTAL && isfinite(*mean);
}

/* 正确总数不降也可能发生互换；这里必须逐位判断，不能只比较总数。 */
static unsigned kp_breaks(const int32_t *old,const int32_t *now,const int32_t *teacher,unsigned n) {
    unsigned broken=0;for(unsigned j=0;j<n;j++)broken+=old[j]==teacher[j] && now[j]!=teacher[j];
    return broken;
}
static bool kp_retained(ds4_engine *e,const pt_cfg *c,pt_run *r,pt_sample *samples,
                        int32_t previous[KEEP_DOCS][KEEP_N],unsigned *broken,unsigned *hit,FILE *audit,unsigned step) {
    int32_t next[KEEP_DOCS][KEEP_N],rep[KEEP_N],teacher[KEEP_N];bool ok=pt_loss_mode(c,false);
    *broken=0;*hit=0;
    for(unsigned doc=0;ok && doc<KEEP_DOCS;doc++) {
        double a=0,b=0;
        ok=kp_score(e,c,r,&samples[doc],next[doc],&a) && kp_score(e,c,r,&samples[doc],rep,&b) &&
            a==b && !memcmp(next[doc],rep,sizeof rep);
        for(unsigned j=0;ok && j<KEEP_N;j++) {
            unsigned best=0;const size_t at=(size_t)j*c->topk;
            for(unsigned k=1;k<c->topk;k++)if(samples[doc].top_p[at+k]>samples[doc].top_p[at+best])best=k;
            teacher[j]=samples[doc].top_id[at+best];*hit+=next[doc][j]==teacher[j];
            fprintf(audit,"%u\t%u\t%u\t%d\t%d\t%d\t%.9g\n",step,doc,j,teacher[j],previous[doc][j],next[doc][j],samples[doc].top_p[at+best]);
        }
        if(ok)*broken+=kp_breaks(previous[doc],next[doc],teacher,KEEP_N);
    }
    if(ok && !*broken)memcpy(previous,next,sizeof next);
    return ok && fflush(audit)==0;
}

/* 只在GPU上量实际参数差；同L2不等于同逐坐标步长，所以两者都记录。 */
static bool kp_update_stats(ds4_gpu_tensor **p,ds4_gpu_tensor **old,ds4_gpu_tensor **g,
                            uint32_t n,double gn,double *norm,double *largest,double *alignment) {
    bool ok=true;double ss=0,cross=0;float v=0;int32_t at=0;
    ds4_gpu_tensor *d[2]={0},*tmp=v41_alloc((uint64_t)n*4,&ok),*idx=v41_alloc(4,&ok);
    *largest=0;
    for(unsigned k=0;ok && k<2;k++) {
        d[k]=v41_alloc((uint64_t)n*4,&ok);double q=0;
        if(ok)ok=ds4_gpu_tensor_copy(d[k],0,p[k],0,(uint64_t)n*4) && ds4_gpu_bwd_axpy_tensor(d[k],old[k],-1.f,n) &&
            ds4_gpu_bwd_sumsq_tensor(d[k],n,&q);
        ss+=q;
        for(unsigned sign=0;ok && sign<2;sign++) {
            ok=ds4_gpu_tensor_fill_f32(tmp,0.f,n) && ds4_gpu_bwd_axpy_tensor(tmp,d[k],sign?-1.f:1.f,n) &&
                ds4_gpu_v41_argmax_tensor(idx,tmp,0,n) && ds4_gpu_tensor_read(idx,0,&at,4) && at>=0 && (uint32_t)at<n &&
                ds4_gpu_tensor_read(tmp,(uint64_t)at*4,&v,4);
            *largest=fmax(*largest,v);
        }
    }
    *norm=sqrt(ss);ok=ok && isfinite(*norm) && *norm>0 && isfinite(gn) && gn>0;
    for(unsigned k=0;ok && k<2;k++) {
        double q=0;ok=ds4_gpu_tensor_fill_f32(tmp,0.f,n) && ds4_gpu_bwd_axpy_tensor(tmp,d[k],(float)(1 / *norm),n) &&
            ds4_gpu_bwd_axpy_tensor(tmp,g[k],(float)(1/gn),n) && ds4_gpu_bwd_sumsq_tensor(tmp,n,&q);
        cross+=q;
    }
    *alignment=(2-cross)/2;
    kp_free(&d[0]);kp_free(&d[1]);kp_free(&tmp);kp_free(&idx);
    return ok && isfinite(*largest) && isfinite(*alignment);
}
static bool kp_direction_fixture(void) {
    const float before[4]={0},after[4]={3,-4,0,0},grad[4]={-3,4,0,0};
    bool ok=true;ds4_gpu_tensor *p[2]={0},*old[2]={0},*g[2]={0};
    for(unsigned k=0;ok && k<2;k++) {
        p[k]=v41_alloc(16,&ok);old[k]=v41_alloc(16,&ok);g[k]=v41_alloc(16,&ok);
        if(ok)ok=ds4_gpu_tensor_write(p[k],0,k?before:after,16) && ds4_gpu_tensor_write(old[k],0,before,16) &&
            ds4_gpu_tensor_write(g[k],0,k?before:grad,16);
    }
    double norm=0,max=0,align=0;
    if(ok)ok=kp_update_stats(p,old,g,4,5,&norm,&max,&align) && fabs(norm-5)<1e-6 && fabs(max-4)<1e-6 && fabs(align-1)<1e-6;
    fprintf(stderr,"[direction] known_answer norm=%.12g max=%.12g alignment=%.12g %s\n",norm,max,align,ok?"PASS":"FAIL");
    for(unsigned k=0;k<2;k++){kp_free(&p[k]);kp_free(&old[k]);kp_free(&g[k]);}
    return ok;
}
static bool kp_direction_step(ds4_engine *e,const pt_cfg *c,pt_run *r,pt_sample *samples,ds4_gpu_tensor **mask,
                              ds4_gpu_tensor *snap[DS4_MAX_LAYER][2],ds4_gpu_tensor **last,
                              int32_t base[KEEP_DOCS][KEEP_N],FILE *audit,unsigned step,
                              double *radius,double *gn,int *clipped,unsigned *updates) {
    const uint32_t n=r->K*DS4_N_EMBD;bool ok=true;double gg=0;
    ds4_gpu_tensor *g[2]={0},*p[]={r->st.ampA[c->layer_hi],r->st.ampB[c->layer_hi]};
    ds4_gpu_tensor *live[]={r->gA[c->layer_hi],r->gB[c->layer_hi]};
    for(unsigned k=0;ok && k<2;k++) {
        if(r->k2) {
            ds4_gpu_tensor *prefix=pt_rows(live[k],0,r->k2,DS4_N_EMBD);
            ok=prefix && ds4_gpu_tensor_fill_f32(prefix,0.f,(uint64_t)r->k2*DS4_N_EMBD);kp_free(&prefix);
        }
        g[k]=v41_alloc((uint64_t)n*4,&ok);double q=0;
        if(ok)ok=ds4_gpu_tensor_copy(g[k],0,live[k],0,(uint64_t)n*4) && ds4_gpu_bwd_sumsq_tensor(g[k],n,&q);
        gg+=q;
    }
    *gn=sqrt(gg);*clipped=0;ok=ok && isfinite(*gn) && *gn>0 && *gn<c->clip;
    if(ok && step==1) {
        ok=kp_direction_fixture();
        if(PT_TAIL_RADIUS>0) {
            /* 接力已获准的总步长；不把旧父本专用的Adam控制组读数套到新父本。 */
            *radius=PT_TAIL_RADIUS;
            fprintf(stderr,"[direction] inherited_radius=%.12g adam_control_steps=0\n",*radius);
        }
    }
    if(ok && step==1 && PT_TAIL_RADIUS<=0) {
        double adam_gn=0,post=0,max=0,align=0;int adam_clip=0;unsigned broken=0,hit=0;
        int32_t control[KEEP_DOCS][KEEP_N];memcpy(control,base,sizeof control);
        ok=pt_opt_step(c,r,1,&adam_gn,&adam_clip);if(ok)(*updates)++;
        if(ok)ok=kp_frozen(c,r,snap) && kp_update_stats(p,last,g,n,*gn,radius,&max,&align) &&
            kp_objective(e,c,r,samples,mask,true,0,&post) && kp_retained(e,c,r,samples,control,&broken,&hit,audit,0);
        fprintf(stderr,"[direction] adam_control norm=%.12g max=%.12g alignment=%.12g objective=%.12g same=%u broken=%u repeat=%u\n",
                *radius,max,align,post,hit,broken,ok);
        ok=ok && hit==7351 && broken==1 && fabs(post-.164124521423)<1e-10 && !adam_clip && fabs(adam_gn-*gn)<1e-10;
        if(ok)ok=kp_restore(c,r,snap);
    }
    const double scale=*radius / *gn;
    ok=ok && isfinite(scale) && scale>0;
    for(unsigned k=0;ok && k<2;k++)ok=ds4_gpu_bwd_axpy_tensor(p[k],g[k],(float)-scale,n) && ds4_gpu_tensor_fill_f32(live[k],0.f,n);
    if(ok)(*updates)++;
    double actual=0,max=0,align=0;
    if(ok)ok=kp_update_stats(p,last,g,n,*gn,&actual,&max,&align) && fabs(actual / *radius-1)<.001 && align>.999;
    fprintf(stderr,"[direction] raw step=%u radius=%.12g actual=%.12g max=%.12g alignment=%.12g scale=%.12g matched=%u\n",
            step,*radius,actual,max,align,scale,ok);
    for(unsigned k=0;k<2;k++)kp_free(&g[k]);
    return ok;
}
/* 只量训练切片B中已登记的脆弱位置；固定竞争词便于比较，最终仍检查全部位置。 */
static bool kp_boundary_probe(ds4_engine *e,pt_run *r,const pt_sample *s,double *margin) {
    pt_state_reset(&r->st);r->cont_s=NULL;r->st.ced_skip=0;r->st.head_last_only=0;
    float a=0,b=0;const uint64_t row=(uint64_t)575*DS4_N_VOCAB;
    const bool ok=v41_forward(e,&r->st,s->sids,s->sn) &&
        ds4_gpu_tensor_read(r->st.logits,(row+769)*4,&a,4) && ds4_gpu_tensor_read(r->st.logits,(row+6946)*4,&b,4);
    *margin=(double)a-b;return ok && isfinite(*margin);
}

/* 每次都从同一快照重新施加原梯度，避免权重插值误差；失败试算不能改保持集合。 */
static int kp_backtrack(ds4_engine *e,const pt_cfg *c,pt_run *r,pt_sample *samples,ds4_gpu_tensor **mask,
                        ds4_gpu_tensor *snap[DS4_MAX_LAYER][2],ds4_gpu_tensor **last,
                        int32_t retained[KEEP_DOCS][KEEP_N],FILE *audit,unsigned step,double pre,
                        double *post,double *radius,unsigned *trials,unsigned *updates) {
    bool ok=true;const uint32_t n=r->K*DS4_N_EMBD;double gg=0;
    ds4_gpu_tensor *g[2]={0},*p[]={r->st.ampA[c->layer_hi],r->st.ampB[c->layer_hi]};
    ds4_gpu_tensor *live[]={r->gA[c->layer_hi],r->gB[c->layer_hi]};
    for(unsigned k=0;ok && k<2;k++) {
        if(r->k2) {
            ds4_gpu_tensor *prefix=pt_rows(live[k],0,r->k2,DS4_N_EMBD);
            ok=prefix && ds4_gpu_tensor_fill_f32(prefix,0.f,(uint64_t)r->k2*DS4_N_EMBD);kp_free(&prefix);
        }
        g[k]=v41_alloc((uint64_t)n*4,&ok);double q=0;
        if(ok)ok=ds4_gpu_tensor_copy(g[k],0,live[k],0,(uint64_t)n*4) && ds4_gpu_bwd_sumsq_tensor(g[k],n,&q) &&
            ds4_gpu_tensor_fill_f32(live[k],0.f,n);
        gg+=q;
    }
    const double gn=sqrt(gg);double boundary=0;bool accepted=false;
    ok=ok && isfinite(gn) && gn>0 && gn<c->clip && pre>0;
    if(ok && step==1) {
        *radius=PT_TAIL_RADIUS;
        ok=kp_direction_fixture() && kp_boundary_probe(e,r,&samples[3],&boundary) &&
            fabs(boundary-.000299453735)<1e-9 && fabs(gn-.0236464977571)<1e-10 && fabs(pre-.130914387941)<1e-10;
        fprintf(stderr,"[backtrack] parent_boundary=%.12g gnorm=%.12g baseline_match=%u\n",boundary,gn,ok);
    }
    double trial_radius=*radius;
    for(unsigned attempt=0;ok && attempt<4 && *trials<8;attempt++,trial_radius*=.5) {
        for(unsigned k=0;ok && k<2;k++)ok=ds4_gpu_tensor_copy(p[k],0,last[k],0,(uint64_t)n*4) &&
            ds4_gpu_bwd_axpy_tensor(p[k],g[k],(float)(-trial_radius/gn),n);
        if(ok){(*updates)++;(*trials)++;}
        double actual=0,max=0,align=0,trial_loss=0;
        if(ok)ok=kp_frozen(c,r,snap) && kp_update_stats(p,last,g,n,gn,&actual,&max,&align) &&
            fabs(actual/trial_radius-1)<.001 && align>.999 &&
            kp_objective(e,c,r,samples,mask,true,0,&trial_loss) && kp_boundary_probe(e,r,&samples[3],&boundary);
        int32_t next[KEEP_DOCS][KEEP_N];memcpy(next,retained,sizeof next);unsigned broken=0,hit=0;
        if(ok)ok=kp_retained(e,c,r,samples,next,&broken,&hit,audit,*trials);
        const bool replay=step!=1 || attempt!=0 ||
            (fabs(trial_loss-.130811669869)<1e-10 && hit==7512 && broken==1);
        ok=ok && replay;
        const bool armijo=isfinite(trial_loss) && trial_loss<=pre-1e-4*trial_radius*gn;
        accepted=ok && armijo && !broken;
        fprintf(stderr,"[backtrack] step=%u attempt=%u trial=%u radius=%.12g actual=%.12g max=%.12g alignment=%.12g gnorm=%.12g before=%.12g after=%.12g boundary=%.12g same=%u broken=%u armijo=%u repeat=%u replay98_check=%d accepted=%u\n",
                step,attempt+1,*trials,trial_radius,actual,max,align,gn,pre,trial_loss,boundary,hit,broken,armijo,ok,step==1&&attempt==0?(int)replay:-1,accepted);
        if(accepted){memcpy(retained,next,sizeof next);*post=trial_loss;*radius=trial_radius;break;}
        for(unsigned k=0;k<2;k++)if(!ds4_gpu_tensor_copy(p[k],0,last[k],0,(uint64_t)n*4))ok=false;
        fprintf(stderr,"[backtrack] trial=%u restored_previous=%u\n",*trials,ok);
    }
    for(unsigned k=0;k<2;k++)kp_free(&g[k]);
    return !ok?-1:accepted?1:0;
}
static bool kp_tail_train(ds4_engine *e,const pt_cfg *c,pt_run *r,pt_sample *samples,
                         ds4_gpu_tensor **mask,ds4_gpu_tensor *snap[DS4_MAX_LAYER][2],
                         int32_t base[KEEP_DOCS][KEEP_N],FILE *f,unsigned *updates) {
    bool ok=kp_restore(c,r,snap);double previous=0,radius=0;unsigned accepted=0,trials=0;
    ds4_gpu_tensor *last[2]={0};const uint64_t bytes=(uint64_t)r->K*DS4_N_EMBD*4;
    FILE *audit=NULL;
    int32_t retained[KEEP_DOCS][KEEP_N];memcpy(retained,base,sizeof retained);
    if(PT_TAIL_RETAIN) {
        char path[1200];snprintf(path,sizeof path,"%s/retention.tsv",c->out);audit=fopen(path,"w");
        ok=ok && audit!=NULL;
        if(audit)fprintf(audit,"step\tdoc\tpos\tteacher\tprevious\ttrial\tp1\n");
        const int32_t t[]={1,2,3},old[]={1,0,3},swap[]={0,2,3},gain[]={1,2,3};
        ok=ok && kp_breaks(old,swap,t,3)==1 && kp_breaks(old,gain,t,3)==0;
        fprintf(stderr,"[retain] known_answer=%s\n",ok?"PASS":"FAIL");
        for(unsigned k=0;ok && k<2;k++)last[k]=v41_alloc(bytes,&ok);
    }
    for(unsigned step=1;ok && step<=c->max_steps;step++) {
        double pre=0,post=0,gn=0;int clipped=0;
        if(PT_TAIL_RETAIN)for(unsigned k=0;ok && k<2;k++)
            ok=ds4_gpu_tensor_copy(last[k],0,k?r->st.ampB[c->layer_hi]:r->st.ampA[c->layer_hi],0,bytes);
        if(!ok)break;
        ok=kp_objective(e,c,r,samples,mask,true,1,&pre);
        const bool repeated=step==1?(PT_TAIL_EXPECT_OBJECTIVE<=0 || fabs(pre-PT_TAIL_EXPECT_OBJECTIVE)<1e-6):pre==previous;
        ok=ok && repeated;
        if(ok && PT_TAIL_BACKTRACK) {
            const int result=kp_backtrack(e,c,r,samples,mask,snap,last,retained,audit,step,pre,&post,&radius,&trials,updates);
            ok=result>=0;
            if(result<=0){fprintf(stderr,"[backtrack] stopped_at=%u accepted_steps=%u trials=%u rollback=%u\n",step,accepted,trials,ok);break;}
            previous=post;accepted++;
            fprintf(stderr,"[tail] step=%u before=%.12g after=%.12g pre_repeat=%u frozen_L0_38=1 decreased=1 accepted=1\n",step,pre,post,repeated);
            continue;
        }
        if(ok && PT_TAIL_DIRECTION)ok=kp_direction_step(e,c,r,samples,mask,snap,last,base,audit,step,&radius,&gn,&clipped,updates);
        else if(ok){ok=pt_opt_step(c,r,step,&gn,&clipped);if(ok)(*updates)++;}
        const bool frozen=ok && kp_frozen(c,r,snap);
        if(ok)ok=frozen && kp_objective(e,c,r,samples,mask,true,0,&post);
        ok=ok && isfinite(gn) && post<pre;
        fprintf(stderr,"[tail] step=%u before=%.12g after=%.12g gnorm=%.12g clip=%d pre_repeat=%u frozen_L0_38=%u decreased=%u\n",
                step,pre,post,gn,clipped,repeated,frozen,ok);
        previous=post;
        if(ok && PT_TAIL_RETAIN) {
            unsigned broken=0,hit=0;ok=kp_retained(e,c,r,samples,retained,&broken,&hit,audit,step);
            fprintf(stderr,"[retain] step=%u same=%u broken_previous=%u repeat=%u accepted=%u\n",step,hit,broken,ok,ok&&!broken);
            if(ok && broken) {
                /* 首次违反保持条件就整步撤回并结束，不再使用该步的Adam状态续训。 */
                for(unsigned k=0;ok && k<2;k++)
                    ok=ds4_gpu_tensor_copy(k?r->st.ampB[c->layer_hi]:r->st.ampA[c->layer_hi],0,last[k],0,bytes);
                fprintf(stderr,"[retain] stopped_at=%u accepted_steps=%u rollback=%u\n",step,accepted,ok);
                break;
            }
        }
        if(ok)accepted++;
    }
    for(unsigned k=0;k<2;k++)kp_free(&last[k]);
    if(audit && fclose(audit)!=0)ok=false;
    unsigned hit=0,fixed=0,broken=0,baseline=0;int32_t ids[KEEP_N],rep[KEEP_N];
    if(ok)ok=pt_loss_mode(c,false);
    for(unsigned doc=0;ok && doc<KEEP_DOCS;doc++) {
        double a=0,b=0;
        ok=kp_score(e,c,r,&samples[doc],ids,&a) && kp_score(e,c,r,&samples[doc],rep,&b) && !memcmp(ids,rep,sizeof ids) && a==b;
        if(PT_TAIL_RETAIN)ok=ok && !memcmp(ids,retained[doc],sizeof ids);
        for(unsigned j=0;ok && j<KEEP_N;j++) {
            unsigned best=0;const size_t at=(size_t)j*c->topk;
            for(unsigned k=1;k<c->topk;k++)if(samples[doc].top_p[at+k]>samples[doc].top_p[at+best])best=k;
            const int32_t t=samples[doc].top_id[at+best];const bool was=base[doc][j]==t,now=ids[j]==t;
            baseline+=was;hit+=now;fixed+=!was&&now;broken+=was&&!now;
            fprintf(f,"candidate\t%u\t%u\t%d\t%d\t%d\t%u\t%.9g\n",doc,j,t,ids[j],base[doc][j],was,samples[doc].top_p[at+best]);
        }
    }
    ok=ok && hit>=baseline && kp_frozen(c,r,snap);
    if(PT_TAIL_RETAIN)ok=ok && accepted>0 && hit>baseline && broken==0;
    fprintf(stderr,"[tail] baseline=%u same=%u total=%u fixed=%u broken=%u post_repeat=%u\n",baseline,hit,KEEP_TOTAL,fixed,broken,ok);
    if(PT_TAIL_RETAIN)fprintf(stderr,"[retain] accepted_steps=%u final_gate=%s\n",accepted,ok?"PASS":"FAIL");
    char path[1200];snprintf(path,sizeof path,"%s/candidate",c->out);
    if(ok)ok=mkdir(path,0755)==0 && pt_save_amp(c,r,path);
    fflush(f);return ok;
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
    const bool tail=c->diag==18;
    if(PT_TAIL_BACKTRACK && (!tail || !PT_TAIL_RETAIN || !PT_TAIL_DIRECTION || PT_TAIL_RADIUS<=0))return false;
    if(tail && PT_TAIL_RETAIN && c->max_steps!=(PT_TAIL_DIRECTION?4u:8u))return false;
    if((c->diag!=16 && !tail) || c->maxlen!=2048 || (tail?(c->max_steps!=8 && c->max_steps!=16 && !(PT_TAIL_DIRECTION && c->max_steps==4)):c->max_steps!=1) || c->rank!=64 || c->layer_lo || c->layer_hi!=39 ||
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
    /* 全层分配器靠重算取输入，不留末层闭式反传所需的 xn；混合入口必须显式存这一份。 */
    if(ok && tail && !r->save.moe_in[c->layer_hi])
        r->save.moe_in[c->layer_hi]=v41_alloc((uint64_t)c->maxlen*DS4_N_EMBD*4,&ok);
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
    if(tail) {
        fprintf(stderr,"[tail] expected_parent same=%u low_conf_correct=%u\n",(unsigned)PT_TAIL_EXPECT_SAME,(unsigned)PT_TAIL_EXPECT_LOW);
        ok=ok && total_keep==PT_TAIL_EXPECT_SAME && low_keep==PT_TAIL_EXPECT_LOW && kp_tail_train(e,c,r,samples,mask,snap,base,f,&updates);
        goto finish;
    }
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
finish:
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
