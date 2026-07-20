/* z_explore.c — 找代码语法的"隐形的手"z(输出/行为空间低秩隐变量).
 * 载入 dump 的 final hidden H_fp/H_q(全43层前向), 在 yv(hc-mix后 DIM 空间)拟合低秩 z,
 * 过 head 测 held-out 代码 token 的【下一token top1一致率(vs FP) + top1准确(vs真值) + NLL】.
 * 秒级扫 rank×λ×机制. 编译: cc -O3 -lm -framework Accelerate -I../../.. z_explore.c -o z_explore (或 -DDS4QUANT_NO_BLAS 免链)
 * 跑: ./z_explore  (读 /tmp/hfp.bin /tmp/hq.bin /tmp/rr_code.ids) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "st_read.c"
#include "ds4quant_fwd.c"   /* dq_rms, dq_sigmoid */
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
/* yv[DIM] → logits[VOCAB] (rms + head). */
static void yv2logits(const float*yv,const float*norm,const float*hw,float*logits){
    float yn[DIM]; dq_rms(yv,norm,yn,DIM,EPSF);
    for(int vv=0;vv<VOCAB;vv++){const float*hr=hw+(size_t)vv*DIM;float aa=0;for(int d=0;d<DIM;d++)aa+=yn[d]*hr[d];logits[vv]=aa;}
}
static int argmax(const float*L){int m=0;for(int v=1;v<VOCAB;v++)if(L[v]>L[m])m=v;return m;}
static double nll_of(const float*L,long tgt){ float mx=L[0];for(int v=1;v<VOCAB;v++)if(L[v]>mx)mx=L[v];
    double se=0;for(int v=0;v<VOCAB;v++)se+=exp((double)L[v]-mx); return -((double)L[tgt]-mx-log(se)); }

static float*loadH(const char*p,int*S){FILE*f=fopen(p,"rb");if(!f){printf("no %s\n",p);exit(1);}int hd[3];fread(hd,4,3,f);
    *S=hd[0]; float*H=malloc((size_t)hd[0]*hd[1]*hd[2]*4); fread(H,4,(size_t)hd[0]*hd[1]*hd[2],f); fclose(f); return H;}

int main(int argc,char**argv){
    const char*hf=getenv("DS4_HF");if(!hf)hf="/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base"; st_open(&C,hf);
    int Sfp,Sq; float*Hfp=loadH("/tmp/hfp.bin",&Sfp),*Hq=loadH("/tmp/hq.bin",&Sq);
    int S=Sfp<Sq?Sfp:Sq; int n_fit=(int)(S*3/4);
    long ids[1024];int ni=0;FILE*f=fopen(argc>1?argv[1]:"/tmp/rr_code.ids","r");char ln[64];while(ni<S&&fgets(ln,sizeof(ln),f))ids[ni++]=atol(ln);fclose(f);
    /* head 权重 */
    float*hcfn=st_read_weight(&C,"hc_head_fn",NULL,NULL),*hcb=st_read_weight(&C,"hc_head_base",NULL,NULL),*hcs=st_read_weight(&C,"hc_head_scale",NULL,NULL);
    float*norm=st_read_weight(&C,"norm.weight",NULL,NULL),*hw=st_read_weight(&C,"head.weight",NULL,NULL);
    /* yv */
    float*yfp=malloc((size_t)S*DIM*4),*yq=malloc((size_t)S*DIM*4);
    hcmix(Hfp,S,hcfn,hcb,hcs,yfp); hcmix(Hq,S,hcfn,hcb,hcs,yq);
    fprintf(stderr,"S=%d n_fit=%d held=%d\n",S,n_fit,S-n_fit);
    /* 基线: held-out FP argmax + 未校正 quant */
    int nh=0, agree0=0, accF=0, accQ0=0; double nllF=0,nllQ0=0;
    float*LF=malloc((size_t)VOCAB*4),*LQ=malloc((size_t)VOCAB*4);
    int *fpArg=malloc((size_t)S*sizeof(int));
    for(int s=n_fit;s<S-1;s++){ long tgt=ids[s+1];
        yv2logits(yfp+(size_t)s*DIM,norm,hw,LF); yv2logits(yq+(size_t)s*DIM,norm,hw,LQ);
        int af=argmax(LF),aq=argmax(LQ); fpArg[s]=af;
        agree0+=(aq==af); accF+=(af==(int)tgt); accQ0+=(aq==(int)tgt);
        nllF+=nll_of(LF,tgt); nllQ0+=nll_of(LQ,tgt); nh++; }
    printf("=== 基线(held-out %d code token) ===\n",nh);
    printf("FP:    top1准确(vs真值)=%.1f%%  NLL=%.3f\n",100.0*accF/nh,nllF/nh);
    printf("1-bit: top1一致(vs FP)=%.1f%%  top1准确=%.1f%%  NLL=%.3f\n",100.0*agree0/nh,100.0*accQ0/nh,nllQ0/nh);
    printf("=== z 校正扫描 (★目标=真值token: yq→(head[真值]-head[当前argmax])) ===\n");
    printf("rank α    | held top1准确(vs真值) top1一致(vsFP) NLL  (Δ准确 vs 1-bit)\n");
    /* fit 目标 R[s]=head_row[tgt]-head_row[argmax_q(s)] (顶真值token的logit方向); 已对的置0 */
    float*R=malloc((size_t)n_fit*DIM*4);
    for(int s=0;s<n_fit;s++){ long tgt=ids[s+1];
        yv2logits(yq+(size_t)s*DIM,norm,hw,LQ); int aq=argmax(LQ);
        const float*ht=hw+(size_t)tgt*DIM,*ha=hw+(size_t)aq*DIM;
        for(int d=0;d<DIM;d++) R[(size_t)s*DIM+d]=(aq==(int)tgt)?0.0f:(ht[d]-ha[d]); }
    int ranks[]={1,2,4,8,16,32}; double alphas[]={0.05,0.1,0.2,0.5,1.0}; double lam=1.0;
    for(int ri=0;ri<6;ri++){ int rk=ranks[ri]; if(rk>=n_fit)continue;
        ds4_z*zl=z_solve_dual(yq,R,n_fit,DIM,DIM,rk,(float)lam); if(!zl)continue;
        for(int ai=0;ai<5;ai++){ double al=alphas[ai];
            int ag=0,acq=0; double nllq=0; float*yc=malloc((size_t)DIM*4),*zc=malloc((size_t)DIM*4);
            for(int s=n_fit;s<S-1;s++){ long tgt=ids[s+1];
                memset(zc,0,(size_t)DIM*4); ds4_z_apply(zl,yq+(size_t)s*DIM,zc);
                for(int d=0;d<DIM;d++) yc[d]=yq[(size_t)s*DIM+d]+(float)al*zc[d];
                yv2logits(yc,norm,hw,LQ); int aq=argmax(LQ);
                ag+=(aq==fpArg[s]); acq+=(aq==(int)tgt); nllq+=nll_of(LQ,tgt); }
            free(yc);free(zc);
            printf("r%-3d α%-5g | %.1f%%  %.1f%%  %.3f   (%+.1f)\n",rk,al,
                   100.0*acq/nh,100.0*ag/nh,nllq/nh, 100.0*(acq-accQ0)/nh);
            fflush(stdout); }
        ds4_z_free(zl); }
    return 0;
}
