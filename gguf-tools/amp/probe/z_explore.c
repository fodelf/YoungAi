/* z_explore.c — 输出空间低秩动态侧车 z (在 32.4GiB 纯1bit Θ_fix 上叠 MB 级校正).
 * 载入 dump 的 final hidden H_fp/H_q(全43层前向), 在 yv(hc-mix后 DIM 空间)拟合低秩 z,
 * 判决(2026-07-10 还原率铁律): held-out 的【分布还原率 Σmin + KL(fp‖q) + PPL ratio】主判据,
 * top1 仅参考; head 投影批量 sgemm(扫参分钟级)。每组合报告侧车体积(MB)。
 * 编译: cc -O3 -lm -framework Accelerate -I../../.. z_explore.c -o z_explore (或 -DDS4QUANT_NO_BLAS 免链)
 * 跑: DS4_NFIT=770 ./z_explore /tmp/rr_calib1075.ids  (读 /tmp/hfp.bin /tmp/hq.bin; ids 须与 dump 同文件) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "../../calib/st_read.c"
#include "../ds4quant_fwd.c"   /* dq_rms, dq_sigmoid */
#include "ds4_z.c"
#include "ds4_loss.c"

#define DIM 4096
#define HCM 4
#define VOCAB 129280
#define EPSF 1e-6f
static st_ctx C;

/* z_solve_dual: 与 ds4quant_run.c 逐字一致(dual-form ridge RRR 低秩). */
static ds4_z *z_solve_dual(const float *X, const float *R, uint32_t n,
                           uint32_t d_in, uint32_t d_out, uint32_t rank, float lambda){
    if(!X||!R||!n||!d_in||!d_out||!rank) return NULL;
    if(rank>d_in)rank=d_in; if(rank>d_out)rank=d_out; if(rank>n)rank=n;
    double *G=calloc((size_t)n*n,sizeof(double)); if(!G) return NULL;
    for(uint32_t a=0;a<n;a++){ const float*xa=X+(size_t)a*d_in;
        for(uint32_t b=a;b<n;b++){ const float*xb=X+(size_t)b*d_in; double s=0;
            for(uint32_t i=0;i<d_in;i++) s+=(double)xa[i]*xb[i]; G[(size_t)a*n+b]=s; } }
    double tr=0; for(uint32_t a=0;a<n;a++) tr+=G[(size_t)a*n+a];
    double ridge=(double)lambda*(tr/(double)d_in)+1e-10;
    for(uint32_t a=0;a<n;a++){ G[(size_t)a*n+a]+=ridge;
        for(uint32_t b=a+1;b<n;b++) G[(size_t)b*n+a]=G[(size_t)a*n+b]; }
    if(cholesky(G,n)!=0){ free(G); return NULL; }
    double *al=malloc((size_t)n*d_out*sizeof(double)),*cc=malloc((size_t)n*sizeof(double));
    for(uint32_t j=0;j<d_out;j++){ for(uint32_t a=0;a<n;a++) cc[a]=R[(size_t)a*d_out+j];
        cholesky_solve(G,n,cc); for(uint32_t a=0;a<n;a++) al[(size_t)a*d_out+j]=cc[a]; }
    free(cc); free(G);
    float *W=malloc((size_t)d_in*d_out*sizeof(float));
    for(uint32_t i=0;i<d_in;i++){ float*Wi=W+(size_t)i*d_out; for(uint32_t j=0;j<d_out;j++)Wi[j]=0.0f;
        for(uint32_t a=0;a<n;a++){ double xai=X[(size_t)a*d_in+i]; if(xai==0.0)continue;
            const double*ala=al+(size_t)a*d_out; for(uint32_t j=0;j<d_out;j++) Wi[j]+=(float)(xai*ala[j]); } }
    free(al);
    ds4_z *zl=calloc(1,sizeof(*zl));
    float *V=malloc((size_t)d_in*rank*4),*T=malloc((size_t)d_out*rank*4),*M=malloc((size_t)rank*d_out*4),*U=malloc((size_t)d_out*rank*4),*z=malloc((size_t)rank*4);
    uint64_t seed=0x5A5A1EEDULL;
    for(size_t i=0;i<(size_t)d_in*rank;i++) V[i]=lcg_unit(&seed);
    mgs(V,d_in,rank,&seed);
    for(int it=0;it<12;it++){
        memset(T,0,(size_t)d_out*rank*4);
        for(uint32_t i=0;i<d_in;i++){ const float*Wi=W+(size_t)i*d_out,*Vi=V+(size_t)i*rank;
            for(uint32_t j=0;j<d_out;j++){ float wij=Wi[j]; if(wij==0.0f)continue; float*Tj=T+(size_t)j*rank;
                for(uint32_t c=0;c<rank;c++) Tj[c]+=wij*Vi[c]; } }
        for(uint32_t i=0;i<d_in;i++){ const float*Wi=W+(size_t)i*d_out; float*Vi=V+(size_t)i*rank;
            for(uint32_t c=0;c<rank;c++)Vi[c]=0.0f;
            for(uint32_t j=0;j<d_out;j++){ float wij=Wi[j]; if(wij==0.0f)continue; const float*Tj=T+(size_t)j*rank;
                for(uint32_t c=0;c<rank;c++) Vi[c]+=wij*Tj[c]; } }
        mgs(V,d_in,rank,&seed);
    }
    memset(M,0,(size_t)rank*d_out*4);
    for(uint32_t i=0;i<d_in;i++){ const float*Wi=W+(size_t)i*d_out,*Vi=V+(size_t)i*rank;
        for(uint32_t c=0;c<rank;c++){ float v=Vi[c]; if(v==0.0f)continue; float*Mc=M+(size_t)c*d_out;
            for(uint32_t j=0;j<d_out;j++) Mc[j]+=v*Wi[j]; } }
    for(uint32_t c=0;c<rank;c++){ double nrm=0; const float*Mc=M+(size_t)c*d_out;
        for(uint32_t j=0;j<d_out;j++) nrm+=(double)Mc[j]*Mc[j]; nrm=sqrt(nrm); z[c]=(float)nrm;
        float inv=nrm>1e-20?(float)(1.0/nrm):0.0f; for(uint32_t j=0;j<d_out;j++) U[(size_t)j*rank+c]=Mc[j]*inv; }
    free(M);free(T);free(W);
    for(uint32_t a=0;a<rank;a++){ uint32_t best=a; for(uint32_t b=a+1;b<rank;b++) if(z[b]>z[best])best=b;
        if(best!=a){ float tz=z[a];z[a]=z[best];z[best]=tz;
            for(uint32_t i=0;i<d_in;i++){float tv=V[(size_t)i*rank+a];V[(size_t)i*rank+a]=V[(size_t)i*rank+best];V[(size_t)i*rank+best]=tv;}
            for(uint32_t j=0;j<d_out;j++){float tu=U[(size_t)j*rank+a];U[(size_t)j*rank+a]=U[(size_t)j*rank+best];U[(size_t)j*rank+best]=tu;} } }
    zl->U=U;zl->V=V;zl->z=z;zl->d_in=d_in;zl->d_out=d_out;zl->rank=rank;zl->k=rank; return zl;
}

/* H[S×HCM×DIM] → yv[S×DIM] (hc-mix, head 前半). */
static void hcmix(const float*H,int S,const float*hcfn,const float*hcb,const float*hcs,float*yv){
    int HD_=HCM*DIM;
    for(int s=0;s<S;s++){ const float*x=H+(size_t)s*HD_; double v=0;for(int i=0;i<HD_;i++)v+=(double)x[i]*x[i];
        float rsq=(float)(1.0/sqrt(v/HD_+EPSF));
        float mix[8],pre[8]; for(int j=0;j<HCM;j++){const float*fr=hcfn+(size_t)j*HD_;float aa=0;for(int k=0;k<HD_;k++)aa+=x[k]*fr[k];mix[j]=aa*rsq;pre[j]=dq_sigmoid(mix[j]*hcs[0]+hcb[j])+EPSF;}
        for(int d=0;d<DIM;d++){float aa=0;for(int j=0;j<HCM;j++)aa+=pre[j]*H[((size_t)s*HCM+j)*DIM+d];yv[(size_t)s*DIM+d]=aa;}
    }
}
/* 批量 yv[n,DIM] → logits[n,VOCAB] (rms + sgemm head). */
static void yv2logits_batch(const float*yv,int n,const float*norm,const float*hw,float*out){
    float *yn=malloc((size_t)n*DIM*4);
    for(int s=0;s<n;s++) dq_rms(yv+(size_t)s*DIM,norm,yn+(size_t)s*DIM,DIM,EPSF);
    dq_matmul(yn,hw,out,n,DIM,VOCAB);
    free(yn);
}
/* 一行判决指标累计 (与 ds4quant_run verdict 同式): a=FP logits, b=quant logits. */
typedef struct { double nllf,nllq,smin,kl; int tf,tq,ag,n; } zmet_t;
static void row_metrics(const float*a,const float*b,long tgt,zmet_t*m){
    float ma=a[0],mb=b[0]; int amax=0,bmax=0;
    for(int v=1;v<VOCAB;v++){ if(a[v]>ma){ma=a[v];amax=v;} if(b[v]>mb){mb=b[v];bmax=v;} }
    double sa=0,sb=0; for(int v=0;v<VOCAB;v++){ sa+=exp((double)a[v]-ma); sb+=exp((double)b[v]-mb); }
    double lsa=log(sa),lsb=log(sb),srow=0,krow=0;
    for(int v=0;v<VOCAB;v++){ double lpf=(double)a[v]-ma-lsa,lpq=(double)b[v]-mb-lsb;
        double pf=exp(lpf),pq=exp(lpq); srow+=pf<pq?pf:pq; if(pf>0)krow+=pf*(lpf-lpq); }
    m->smin+=srow; m->kl+=krow;
    m->nllf+=-((double)a[tgt]-ma-lsa); m->nllq+=-((double)b[tgt]-mb-lsb);
    m->tf+=(amax==(int)tgt); m->tq+=(bmax==(int)tgt); m->ag+=(amax==bmax); m->n++;
}

static float*loadH(const char*p,int*S){FILE*f=fopen(p,"rb");if(!f){printf("no %s\n",p);exit(1);}int hd[3];fread(hd,4,3,f);
    *S=hd[0]; float*H=malloc((size_t)hd[0]*hd[1]*hd[2]*4); fread(H,4,(size_t)hd[0]*hd[1]*hd[2],f); fclose(f); return H;}

int main(int argc,char**argv){
    const char*hf=getenv("DS4_HF");if(!hf)hf="/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base"; st_open(&C,hf);
    int Sfp,Sq; float*Hfp=loadH("/tmp/hfp.bin",&Sfp),*Hq=loadH("/tmp/hq.bin",&Sq);
    int S=Sfp<Sq?Sfp:Sq;
    int n_fit=(int)(S*3/4);
    if(getenv("DS4_NFIT")) n_fit=atoi(getenv("DS4_NFIT"));
    if(n_fit<1)n_fit=1; if(n_fit>=S)n_fit=S-1;
    long *ids=malloc(2048*sizeof(long));int ni=0;FILE*f=fopen(argc>1?argv[1]:"/tmp/rr_calib1075.ids","r");
    if(!f){printf("no ids\n");return 1;}
    { char ln[64]; while(ni<S&&ni<2048&&fgets(ln,sizeof(ln),f))ids[ni++]=atol(ln); fclose(f); }
    /* head 权重 */
    float*hcfn=st_read_weight(&C,"hc_head_fn",NULL,NULL),*hcb=st_read_weight(&C,"hc_head_base",NULL,NULL),*hcs=st_read_weight(&C,"hc_head_scale",NULL,NULL);
    float*norm=st_read_weight(&C,"norm.weight",NULL,NULL),*hw=st_read_weight(&C,"head.weight",NULL,NULL);
    /* yv */
    float*yfp=malloc((size_t)S*DIM*4),*yq=malloc((size_t)S*DIM*4);
    hcmix(Hfp,S,hcfn,hcb,hcs,yfp); hcmix(Hq,S,hcfn,hcb,hcs,yq);
    /* held 行(tgt=下一 token): s ∈ [n_fit, S-1) */
    int nh=S-1-n_fit;
    fprintf(stderr,"S=%d n_fit=%d held=%d\n",S,n_fit,nh);
    long *tg=malloc((size_t)nh*sizeof(long));
    float *yfh=malloc((size_t)nh*DIM*4),*yqh=malloc((size_t)nh*DIM*4);
    for(int i=0;i<nh;i++){ int s=n_fit+i; tg[i]=ids[s+1];
        memcpy(yfh+(size_t)i*DIM,yfp+(size_t)s*DIM,(size_t)DIM*4);
        memcpy(yqh+(size_t)i*DIM,yq +(size_t)s*DIM,(size_t)DIM*4); }
    float *LFb=malloc((size_t)nh*VOCAB*4),*LQb=malloc((size_t)nh*VOCAB*4);
    yv2logits_batch(yfh,nh,norm,hw,LFb);
    /* 基线: 未校正 1bit */
    yv2logits_batch(yqh,nh,norm,hw,LQb);
    zmet_t m0; memset(&m0,0,sizeof(m0));
    for(int i=0;i<nh;i++) row_metrics(LFb+(size_t)i*VOCAB,LQb+(size_t)i*VOCAB,tg[i],&m0);
    double pplf=exp(m0.nllf/m0.n);
    printf("=== 动态侧车判决 (held-out %d tok · 主判据 Σmin/KL/PPL, top1 参考) ===\n",nh);
    printf("FP:    PPL=%.4f  top1(vs真值)=%.1f%%\n",pplf,100.0*m0.tf/m0.n);
    printf("Θ_fix(纯1bit, 无侧车): Σmin=%.4f KL=%.4f ratio=%.4f 一致=%.1f%%\n",
           m0.smin/m0.n,m0.kl/m0.n,exp(m0.nllq/m0.n)/pplf,100.0*m0.ag/m0.n);
    printf("=== z 扫描 (输出空间低秩: yq→(yfp-yq) 残差, fit=%d 行) ===\n",n_fit);
    printf("rank λ     | Σmin(Δ)          KL      ratio   一致%%  | 侧车MB\n");
    int ranks[]={1,2,4,8,16,32,64,128}; double lams[]={0.1,1.0,10.0};
    float*R=malloc((size_t)n_fit*DIM*4);
    for(int s=0;s<n_fit;s++)for(int d=0;d<DIM;d++)R[(size_t)s*DIM+d]=yfp[(size_t)s*DIM+d]-yq[(size_t)s*DIM+d];
    float *yc=malloc((size_t)nh*DIM*4);
    for(int ri=0;ri<8;ri++){ int rk=ranks[ri]; if(rk>=n_fit)continue;
      for(int li=0;li<3;li++){ double lam=lams[li];
        ds4_z*zl=z_solve_dual(yq,R,n_fit,DIM,DIM,rk,(float)lam); if(!zl){printf("r%-3d λ%-4g NULL\n",rk,lam);continue;}
        for(int i=0;i<nh;i++){ memcpy(yc+(size_t)i*DIM,yqh+(size_t)i*DIM,(size_t)DIM*4);
            ds4_z_apply(zl,yqh+(size_t)i*DIM,yc+(size_t)i*DIM); }
        yv2logits_batch(yc,nh,norm,hw,LQb);
        zmet_t m; memset(&m,0,sizeof(m));
        for(int i=0;i<nh;i++) row_metrics(LFb+(size_t)i*VOCAB,LQb+(size_t)i*VOCAB,tg[i],&m);
        ds4_z_free(zl);
        double mb=(double)rk*(2.0*DIM+1.0)*4.0/1048576.0;   /* U+V fp32 + z */
        printf("r%-3d λ%-4g | Σmin=%.4f(%+.4f) KL=%.4f ratio=%.4f %5.1f%% | %.2f\n",rk,lam,
               m.smin/m.n, (m.smin-m0.smin)/m.n, m.kl/m.n, exp(m.nllq/m.n)/pplf, 100.0*m.ag/m.n, mb);
        fflush(stdout);
      } }
    return 0;
}
