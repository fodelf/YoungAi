/* z_tune.c — 秒级 z 调优: 载入 dump 的某层 (Fin/Ffp/Fout), 扫 λ×rank, 打印
 * held-out 上 z 校正的 cos/relL2 提升%. 用来找让 z 转正且最大的 (λ,rank).
 * 编译: cc -O3 -lm z_tune.c -o z_tune ; 跑: ./z_tune /tmp/ldump/L0.bin */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "ds4_z.c"
#include "ds4_loss.c"

/* --- z_solve_dual: 与 ds4quant_run.c 逐字一致 --- */
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
    if(!al||!cc){free(G);free(al);free(cc);return NULL;}
    for(uint32_t j=0;j<d_out;j++){ for(uint32_t a=0;a<n;a++) cc[a]=R[(size_t)a*d_out+j];
        cholesky_solve(G,n,cc); for(uint32_t a=0;a<n;a++) al[(size_t)a*d_out+j]=cc[a]; }
    free(cc); free(G);
    float *W=malloc((size_t)d_in*d_out*sizeof(float)); if(!W){free(al);return NULL;}
    for(uint32_t i=0;i<d_in;i++){ float*Wi=W+(size_t)i*d_out; for(uint32_t j=0;j<d_out;j++)Wi[j]=0.0f;
        for(uint32_t a=0;a<n;a++){ double xai=X[(size_t)a*d_in+i]; if(xai==0.0)continue;
            const double*ala=al+(size_t)a*d_out; for(uint32_t j=0;j<d_out;j++) Wi[j]+=(float)(xai*ala[j]); } }
    free(al);
    ds4_z *zl=calloc(1,sizeof(*zl));
    float *V=malloc((size_t)d_in*rank*4),*T=malloc((size_t)d_out*rank*4),*M=malloc((size_t)rank*d_out*4),*U=malloc((size_t)d_out*rank*4),*z=malloc((size_t)rank*4);
    if(!zl||!V||!T||!M||!U||!z){free(zl);free(V);free(T);free(M);free(U);free(z);free(W);return NULL;}
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

static double held_cos(ds4_z*zl,int k,const float*Fout,const float*Fin,const float*Ffp,int s0,int S,int DIMv){
    double dot=0,na=0,nb=0; float *tmp=malloc((size_t)DIMv*4);
    for(int s=s0;s<S;s++){ memcpy(tmp,Fout+(size_t)s*DIMv,(size_t)DIMv*4);
        if(zl&&k>0){ ds4_z_set_rank(zl,k); ds4_z_apply(zl,Fin+(size_t)s*DIMv,tmp); }
        for(int d=0;d<DIMv;d++){ double A=Ffp[(size_t)s*DIMv+d],B=tmp[d]; dot+=A*B;na+=A*A;nb+=B*B; } }
    free(tmp); return dot/(sqrt(na)*sqrt(nb)+1e-12);
}
static double held_relL2(ds4_z*zl,int k,const float*Fout,const float*Fin,const float*Ffp,int s0,int S,int DIMv){
    double a2=0,e2=0; float*tmp=malloc((size_t)DIMv*4);
    for(int s=s0;s<S;s++){ memcpy(tmp,Fout+(size_t)s*DIMv,(size_t)DIMv*4);
        if(zl&&k>0){ ds4_z_set_rank(zl,k); ds4_z_apply(zl,Fin+(size_t)s*DIMv,tmp); }
        for(int d=0;d<DIMv;d++){ double A=Ffp[(size_t)s*DIMv+d],B=tmp[d]; a2+=A*A; double e=B-A; e2+=e*e; } }
    free(tmp); return sqrt(e2)/(sqrt(a2)+1e-12);
}
/* fit-行自身拟合度(诊断是否 solve 有效) */
static double fit_relL2(ds4_z*zl,int k,const float*Fout,const float*Fin,const float*Ffp,int nfit,int DIMv){
    return held_relL2(zl,k,Fout,Fin,Ffp,0,nfit,DIMv);
}

int main(int argc,char**argv){
    const char*path=argc>1?argv[1]:"/tmp/ldump/L0.bin";
    FILE*f=fopen(path,"rb"); if(!f){printf("open fail %s\n",path);return 1;}
    int hd[3]; if(fread(hd,4,3,f)!=3){printf("hdr fail\n");return 1;}
    int S=hd[0],n_fit=hd[1],DIM=hd[2];
    float*Fin=malloc((size_t)S*DIM*4),*Ffp=malloc((size_t)S*DIM*4),*Fout=malloc((size_t)S*DIM*4);
    fread(Fin,4,(size_t)S*DIM,f); fread(Ffp,4,(size_t)S*DIM,f); fread(Fout,4,(size_t)S*DIM,f); fclose(f);
    int s0=n_fit;
    double bc=held_cos(NULL,0,Fout,Fin,Ffp,s0,S,DIM), br=held_relL2(NULL,0,Fout,Fin,Ffp,s0,S,DIM);
    printf("%s  S=%d n_fit=%d held=%d DIM=%d\n",path,S,n_fit,S-n_fit,DIM);
    printf("1bit基线(held-out): cos=%.4f relL2=%.4f\n",bc,br);
    printf("扫描 [Δcos%% / ΔrelL2%%提升 / fit-relL2] (Δ正=z有帮助; fit低=拟合好)\n");
    int ranks[]={1,2,4,8,16,32}; int nr=sizeof(ranks)/sizeof(int);
    double lams[]={0.03,0.1,0.3,1,3,10,30,100,300,1000};
    printf("%-8s","λ\\r");
    for(int r=0;r<nr;r++) if(ranks[r]<n_fit) printf(" r=%-2d            ",ranks[r]);
    printf("\n");
    for(int li=0;li<10;li++){ double lam=lams[li]; printf("%-8g",lam);
        for(int r=0;r<nr;r++){ int rk=ranks[r]; if(rk>=n_fit)continue;
            float*R=malloc((size_t)n_fit*DIM*4);
            for(int s=0;s<n_fit;s++)for(int d=0;d<DIM;d++)R[(size_t)s*DIM+d]=Ffp[(size_t)s*DIM+d]-Fout[(size_t)s*DIM+d];
            ds4_z*zl=z_solve_dual(Fin,R,n_fit,DIM,DIM,rk,(float)lam);
            if(zl){ double hc=held_cos(zl,rk,Fout,Fin,Ffp,s0,S,DIM);
                double hr=held_relL2(zl,rk,Fout,Fin,Ffp,s0,S,DIM);
                double fr=fit_relL2(zl,rk,Fout,Fin,Ffp,n_fit,DIM);
                printf(" %+4.1f/%+4.1f/%.2f",100*(hc-bc)/fabs(bc),100*(br-hr)/br,fr);
                ds4_z_free(zl);
            } else printf("   NULL      ");
            free(R); }
        printf("\n"); }
    return 0;
}
