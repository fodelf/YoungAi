static void co_apply_mult(coexp_t*ex,const float*afin,const float*Fin,const float*shared,
                          int S,float*Ftest,double*cmean,double*cmx){
    memcpy(Ftest,shared,(size_t)S*DIM*4);
    double csum=0,cm=0; long cc=0;
    for(int e=0;e<NEXP;e++){ coexp_t*ce=&ex[e];
        if(ce->nhit<1||!ce->yq_all) continue;
        for(int i=0;i<ce->nhit;i++){
            double s=1.0;
            if(ce->gstat!=0.0f) s=ce->gstat;
            else if(ce->alpha&&ce->ncal){
                const float*xh=Fin+(size_t)ce->hitS[i]*DIM;
                for(int t=0;t<ce->ncal;t++){ if(ce->alpha[t]==0.0f)continue;
                    const float*xc=afin+(size_t)ce->calS[t]*DIM;
                    double d=0; for(int d2=0;d2<DIM;d2++) d+=(double)xc[d2]*xh[d2];
                    s+=ce->alpha[t]*d; }
            }
            if(s<0.25)s=0.25; if(s>4.0)s=4.0;
            double ac=fabs(s-1.0); csum+=ac; if(ac>cm)cm=ac; cc++;
            float wc=ce->hitW[i]*(float)s; float*dst=Ftest+(size_t)ce->hitS[i]*DIM;
            const float*yi=ce->yq_all+(size_t)i*DIM;
            for(int d2=0;d2<DIM;d2++) dst[d2]+=wc*yi[d2];
        }
    }
    if(cmean)*cmean=cc?csum/(double)cc:0; if(cmx)*cmx=cm;
}
static void co_rel(const float*Hq,const float*Hf,int a,int b,size_t rowsz,double*rel);
static double co_score(const float*Hq,const float*Hf,int a,int b,size_t rowsz);
/* 出口评分: fit/val/held 三段 relL2 + val 四损失评分(align+0.5·classify=感知加权) */
static void co_eval(const float*Ftest,const float*H2,const float*post2,const float*comb2,
                    const float*Hf,float*Hq,int S,int vs,int n_fit,size_t rowsz,
                    double*fit,double*val,double*held,double*sc){
    dq_hc_post(Ftest,H2,post2,comb2,Hq,S,HCM,DIM);
    co_rel(Hq,Hf,0,vs,rowsz,fit);
    co_rel(Hq,Hf,vs,n_fit,rowsz,val);
    co_rel(Hq,Hf,n_fit,S,rowsz,held);
    *sc = (n_fit-vs)>0 ? co_score(Hq,Hf,vs,n_fit,rowsz) : co_score(Hq,Hf,0,vs,rowsz);
}
/* 累积回拉目标 DF: hc_post 对 Fout 线性 → Fout 空间闭式反解到本层为止的漂移 */
static void co_df(const float*Fcur,const float*H2,const float*post2,const float*comb2,
                  const float*Hf,float*Hq,float*DF,int S){
    dq_hc_post(Fcur,H2,post2,comb2,Hq,S,HCM,DIM);
    for(int s=0;s<S;s++){
        const float *ps=post2+(size_t)s*HCM; double pd=1e-12;
        for(int j=0;j<HCM;j++) pd+=(double)ps[j]*ps[j];
        for(int d2=0;d2<DIM;d2++){ double a2=0;
            for(int j=0;j<HCM;j++) a2+=(double)ps[j]*((double)Hf[((size_t)s*HCM+j)*DIM+d2]-(double)Hq[((size_t)s*HCM+j)*DIM+d2]);
            DF[(size_t)s*DIM+d2]=(float)(a2/pd); }
    }
}
/* ===== 合并动态侧车(最终产物): 逐层胜者 {algo,params,payload}, 结束一次性落盘 ===== */
typedef struct { int L; char algo[12]; float lam,g; int k,backward;
                 float valb,vala,heldb,helda; uint8_t*pay; uint64_t paysz; } zrec_t;
static zrec_t ZREC[NL]; static int NZREC=0;
static void zfile_write(void){
    if(!NZREC) return;
    const char*p=getenv("DS4_ZFILE"); if(!p)p="/tmp/ds4quant_zfile.bin";
    FILE*f=fopen(p,"wb"); if(!f){ fprintf(stderr,"[zfile] 写 %s 失败\n",p); return; }
    uint32_t magic=0x315A5144; uint32_t nr=(uint32_t)NZREC;   /* "DQZ1" */
    fwrite(&magic,4,1,f); fwrite(&nr,4,1,f);
    uint64_t tot=8;
    for(int i=0;i<NZREC;i++){ zrec_t*r=&ZREC[i];
        fwrite(&r->L,4,1,f); fwrite(r->algo,1,12,f); fwrite(&r->lam,4,1,f);
        fwrite(&r->g,4,1,f); fwrite(&r->k,4,1,f); fwrite(&r->backward,4,1,f);
        fwrite(&r->paysz,8,1,f);
        if(r->paysz&&r->pay) fwrite(r->pay,1,r->paysz,f);
        tot+=40+r->paysz;
    }
    fclose(f);
    printf("ZFILE %s layers=%d bytes=%llu (%.2f MB)\n",p,NZREC,(unsigned long long)tot,(double)tot/1048576.0);
    fflush(stdout);
}

/* base 量化遍(worker 并行): iter=0 全量+缓存, iter=1 向后重解(w2 目标=系数校正后残差) */
static long co_base_pass(int L,int S,int n_fit,int iter,const float*Fin,const int*idx,const float*rw,
                         coexp_t*ex,float*routed,lstat_t*st){
    memset(routed,0,(size_t)S*DIM*4);
    int e_next=0,nth=NTHREADS<1?1:(NTHREADS>NEXP?NEXP:NTHREADS);
    cowork_t *ws=calloc((size_t)nth,sizeof(cowork_t));
    pthread_t *th=malloc((size_t)nth*sizeof(pthread_t));
    long nflip=0;
    for(int t=0;t<nth;t++){
        ws[t]=(cowork_t){L,S,n_fit,iter,Fin,idx,rw,ex,&e_next,calloc((size_t)S*DIM,4),0,0,0,0};
        pthread_create(&th[t],NULL,coadapt_worker,&ws[t]);
    }
    for(int t=0;t<nth;t++){ pthread_join(th[t],NULL);
        for(size_t i=0;i<(size_t)S*DIM;i++) routed[i]+=ws[t].fout[i];
        free(ws[t].fout); nflip+=ws[t].nflip;
        if(iter==0&&st){ st->calib_rows+=ws[t].calib_rows; st->calib_empty+=ws[t].calib_empty; st->nhit+=ws[t].nhit_experts; }
    }
    free(ws);free(th);
    return nflip;
}
/* 胜者 payload 打包(合并侧车用, fp16) */
static uint8_t *co_pay_mult_stat(coexp_t*ex,uint64_t*sz){   /* GE: g_e ×NEXP */
    *sz=(uint64_t)NEXP*2; uint8_t*p=malloc((size_t)*sz); uint16_t*h=(uint16_t*)p;
    for(int e=0;e<NEXP;e++) h[e]=go1b_fp32_to_fp16(ex[e].gstat!=0.0f?ex[e].gstat:1.0f);
    return p;
}
static uint8_t *co_pay_mult_dyn(coexp_t*ex,const float*afin,uint64_t*sz){   /* CE: v_e=Σα·x fp16[DIM]×NEXP */
    *sz=(uint64_t)NEXP*DIM*2; uint8_t*p=malloc((size_t)*sz); uint16_t*h=(uint16_t*)p;
    for(int e=0;e<NEXP;e++){ coexp_t*ce=&ex[e];
        for(int d2=0;d2<DIM;d2++){ double v=0;
            if(ce->alpha&&ce->ncal)
                for(int t=0;t<ce->ncal;t++){ if(ce->alpha[t]==0.0f)continue;
                    v+=(double)ce->alpha[t]*afin[(size_t)ce->calS[t]*DIM+d2]; }
            h[(size_t)e*DIM+d2]=go1b_fp32_to_fp16((float)v); } }
    return p;
}
static uint8_t *co_pay_zl(const ds4_z*zl,int k,uint64_t*sz){   /* ZL: V[d,k],U[d,k],z[k] fp16 */
    int d=(int)zl->d_in,dout=(int)zl->d_out;
    *sz=(uint64_t)k*(d+dout+1)*2; uint8_t*p=malloc((size_t)*sz); uint16_t*h=(uint16_t*)p; size_t o=0;
    for(int c=0;c<k;c++) for(int i=0;i<d;i++)    h[o++]=go1b_fp32_to_fp16(zl->V[(size_t)i*zl->rank+c]);
    for(int c=0;c<k;c++) for(int j=0;j<dout;j++) h[o++]=go1b_fp32_to_fp16(zl->U[(size_t)j*zl->rank+c]);
    for(int c=0;c<k;c++) h[o++]=go1b_fp32_to_fp16(zl->z[c]);
    return p;
}
static void co_rel(const float*Hq,const float*Hf,int a,int b,size_t rowsz,double*rel){
    double e2=0,x2=0;
    for(size_t i=(size_t)a*rowsz;i<(size_t)b*rowsz;i++){
        double dd=(double)Hq[i]-Hf[i]; e2+=dd*dd; x2+=(double)Hf[i]*Hf[i]; }
    *rel=sqrt(e2/(x2+1e-30));
}
static double co_score(const float*Hq,const float*Hf,int a,int b,size_t rowsz){
    int n=b-a; if(n<1) return 0.0;
    const float *A=Hq+(size_t)a*rowsz,*B=Hf+(size_t)a*rowsz;
    float *wv=malloc(rowsz*4); ds4_loss_dim_variance(B,(uint32_t)n,(uint32_t)rowsz,wv);
    double la=ds4_loss_align(A,B,(uint32_t)n,(uint32_t)rowsz);
    double lc=ds4_loss_classify(A,B,wv,(uint32_t)n,(uint32_t)rowsz);
    free(wv);
    return la+0.5*lc;
}
/* ===== 从层文件重前向(跨层反向的执行引擎): 文件即真相 =====
 * 读 dql_L<NN>.bin: 1bit 权重字节 + 落地修正记录(按时间序回放 GL/dyn2/dyn8/TREF)。
 * 不重量化 — 直接 dequant 字节 → 专家前向 → 修正链回放。 */
typedef struct { int type; float g,t; float w2p[4]; float w8[9]; float *V8; float *ge;
                 float *zlU,*zlV,*zlz; int zlk; float zltr;   /* 6=zl.RRR 冻结秩-k 方向修正(产物③) */
                 int zdin;   /* ★md86: 输入维 DIM=线性 | 3*DIM=ftA 特征提升(φ=[x,x²/rms,relu]) */
                 int erf_ne,erf_r; uint32_t *erf_eid; float *erf_tau; float *erf_UV;
                 /* 8=zl.ERF 死层部件(2026-08-12): 每专家 w2 低秩补丁 U[D,r](折S)·V[r,MOEI](折α)+token能量门τ */
                 size_t foff;     float ghc[2];   /* type7 GLhc: g_hot, g_cold(2026-08-06 冷热双通道) */
} lop_t;   /* 1=GL 2=dyn2 3=dyn8 4=TREF 5=GE(per-expert增益,累加时乘) 6=zl.RRR 8=zl.ERF; foff=载荷文件偏移(反修原地改写用) */
typedef struct { uint8_t *map; size_t msz; const uint8_t *w1,*w3,*w2; size_t szG,szD;
                 uint8_t *g2map; size_t g2msz;                                    /* go2b 侧车 mmap(可异机盘/NFS; 内嵌时 NULL, g2w* 指进主 map) */
                 const uint8_t *g2w1,*g2w3,*g2w2; int g2k; int16_t g2slot[256];   /* 热专家 2bit 覆盖 */
                 uint8_t *vqmap; size_t vqmsz;                                    /* v2.2 VQ 侧车 mmap(dql_vq_L%02d.bin) */
                 uint8_t *opsmap; size_t opsmsz; int has_ops;                     /* op 侧车 mmap(dql_ops_L%02d.bin, 平行架构权威源) */
                 lop_t ops[32]; int nops;
                 int zl_stub_at; } lfile_t;   /* zl.RRR 空壳占位序号(真载荷回填正位用; -1=无) */
/* go2b 侧车路径: DS4_GO2B_DIR(默认=层文件同目录)/dql_go2b_L<NN>.bin — 分储设计:
 * 冷 go1b dql 在本机(热专家稀疏洞), 热 go2b 侧车可放对机 NFS(两机 16G 盘都装不下合体) */
static void g2_sidecar_path(const char*dql_path,int L,char*out,size_t outsz){
    const char*gd=getenv("DS4_GO2B_DIR");
    if(gd){ snprintf(out,outsz,"%s/dql_go2b_L%02d.bin",gd,L); return; }
    char dir[512]; snprintf(dir,sizeof(dir),"%s",dql_path);
    char*sl=strrchr(dir,'/'); if(sl)*sl=0; else snprintf(dir,sizeof(dir),".");
    snprintf(out,outsz,"%s/dql_go2b_L%02d.bin",dir,L);
}
/* 侧车头: 'DQG2' u32 | ver u32 | L u32 | khot u32 | mean_cos f32 | rsv u32 | ids u16[khot] | pad8 → 载荷 */
#define G2SC_MAGIC 0x32475144u
static size_t g2_sidecar_hdr(int khot){ return (24+2*(size_t)khot+7)&~(size_t)7; }
/* ★op 侧车(2026-08-04 用户架构令: 量化模型与 z/动态平行, 反修不许动量化文件)★
 * dql_LXX.bin = 纯权重字节(1bit/g2hot), 量化后只读; 全部 op 记录(z 家族/zl.RRR/loss/bf/bwd)
 * 住 dql_ops_LXX.bin('DQO2' u32|L u32|nrec u32 + DQL2 同款记录), 应用序=侧车文件序(单一权威)。
 * 反修/sweep/回扫只写侧车 — 重量化在架构上不再可能被需要。旧混装 dql 兼容读(侧车缺席时)。 */
#define OPSC_MAGIC 0x324F5144u   /* 'DQO2' */
static void ops_sidecar_path(const char*dql_path,int L,char*out,size_t outsz){
    char dir[512]; snprintf(dir,sizeof(dir),"%s",dql_path);
    char*sl=strrchr(dir,'/'); if(sl)*sl=0; else snprintf(dir,sizeof(dir),".");
    snprintf(out,outsz,"%s/dql_ops_L%02d.bin",dir,L);
}
/* op 宿主文件 = dql 主文件(超冠 573b7f5 混装架构; 2026-08-08 用户令还原:
 * "侧车不是我要求加入的" — 平行架构 op 侧车 2026-08-04 系擅自引入, 已删)。 */
static void op_host_path(int L,char*out,size_t outsz){
    snprintf(out,outsz,"%s/dql_L%02d.bin",
        getenv("DS4_LAYER_DIR")?getenv("DS4_LAYER_DIR"):".",L);
}
static lfile_t BF_LFF; static int BF_LFF_ON=0;   /* 反修字节起步的层 mmap 持有者 */
/* op 记录解析(主文件旧混装 与 op 侧车 共用): curfoff=载荷在其宿主文件内的偏移 */
static void parse_op_rec(lfile_t*lf,const char*nm,const uint8_t*pay,uint64_t psz,size_t curfoff){
    if(lf->nops>=32) return;
    lop_t*o=&lf->ops[lf->nops];
    if(strstr(nm,"GLhc")&&psz>=8){ o->type=7; memcpy(o->ghc,pay,8); o->foff=curfoff; lf->nops++; }
    else if(strstr(nm,"GLdyn2")&&psz>=16){ o->type=2; memcpy(o->w2p,pay,16); o->foff=curfoff; lf->nops++; }
    else if(strstr(nm,"GLdyn8")&&psz>=36){ o->type=3; memcpy(o->w8,pay,36); o->foff=curfoff;
        if(psz>=36+(uint64_t)8*DIM*2){ o->V8=malloc((size_t)8*DIM*4);
            const uint16_t*h=(const uint16_t*)(pay+36);
            for(size_t j=0;j<(size_t)8*DIM;j++) o->V8[j]=go1b_fp16_to_fp32(h[j]); }
        lf->nops++; }
    else if(strstr(nm,"bf.GE")&&psz>=(uint64_t)NEXP*2){   /* per-expert 增益(fp16×NEXP), 累加时乘 */
        o->type=5; o->ge=malloc((size_t)NEXP*4);
        const uint16_t*h=(const uint16_t*)pay;
        for(int e=0;e<NEXP;e++) o->ge[e]=go1b_fp16_to_fp32(h[e]);
        o->foff=curfoff; lf->nops++; }
    else if(strstr(nm,"zl.RRR")&&psz>=16){   /* 冻结 z^L: u32 k|f32 tr|u32 din|u32 dout|fp16 z[k],U[dout·k],V[din·k] */
        uint32_t zk,din,dout; float ztr;
        memcpy(&zk,pay,4); memcpy(&ztr,pay+4,4); memcpy(&din,pay+8,4); memcpy(&dout,pay+12,4);
        if(zk>0&&zk<=1024&&(din==(uint32_t)DIM||din==3u*(uint32_t)DIM)&&dout==(uint32_t)DIM
           &&psz>=16+(uint64_t)2*(zk+(uint64_t)zk*din+(uint64_t)zk*dout)){
            o->type=6; o->zlk=(int)zk; o->zltr=ztr; o->zdin=(int)din;
            const uint16_t*h=(const uint16_t*)(pay+16);
            o->zlz=malloc((size_t)zk*4);
            for(uint32_t i=0;i<zk;i++) o->zlz[i]=go1b_fp16_to_fp32(h[i]);
            o->zlU=malloc((size_t)dout*zk*4);
            for(size_t i=0;i<(size_t)dout*zk;i++) o->zlU[i]=go1b_fp16_to_fp32(h[zk+i]);
            o->zlV=malloc((size_t)din*zk*4);
            for(size_t i=0;i<(size_t)din*zk;i++) o->zlV[i]=go1b_fp16_to_fp32(h[zk+(size_t)dout*zk+i]);
            o->foff=curfoff;
            /* 错序回填(旧混装兼容; 侧车模式 zl 在自然表位带载荷, 空壳不出现, 此段不触发):
             * 真载荷曾是 export 后 append 的 ⇒ 文件序在全部 op 之后, 而评估注入点在缩放族之前
             * (空壳占位序); 修正链非交换 → 回填到空壳位。 */
            { int at=-1;
              if(lf->zl_stub_at>=0&&lf->zl_stub_at<=lf->nops) at=lf->zl_stub_at;
              if(at>=0&&at<lf->nops){
                  lop_t tmp=*o;
                  memmove(&lf->ops[at+1],&lf->ops[at],(size_t)(lf->nops-at)*sizeof(lop_t));
                  lf->ops[at]=tmp;
              } }
            lf->nops++; } }
    else if(strstr(nm,"zl.ERF")&&psz>=8){   /* ★死层部件(2026-08-12 用户令"修死层并入反修"): 每专家低秩补丁 */
        uint32_t ne=0; uint16_t r16=0;
        memcpy(&ne,pay,4); memcpy(&r16,pay+4,2);
        size_t blkf=(size_t)DIM*r16+(size_t)r16*MOEI;
        size_t per=8+2*blkf;
        if(ne>0&&ne<=256&&r16>0&&r16<=64&&psz>=8+(uint64_t)ne*per){
            o->type=8; o->erf_ne=(int)ne; o->erf_r=(int)r16;
            o->erf_eid=malloc((size_t)ne*4); o->erf_tau=malloc((size_t)ne*4);
            o->erf_UV=malloc((size_t)ne*blkf*4);
            const uint8_t*pp=pay+8;
            for(uint32_t i=0;i<ne;i++){
                memcpy(&o->erf_eid[i],pp,4); memcpy(&o->erf_tau[i],pp+4,4); pp+=8;
                const uint16_t*hh=(const uint16_t*)pp;
                float*df=o->erf_UV+(size_t)i*blkf;
                for(size_t j=0;j<blkf;j++) df[j]=go1b_fp16_to_fp32(hh[j]);
                pp+=2*blkf;
            }
            o->foff=curfoff; lf->nops++; } }
    else if(strstr(nm,"zl.RRR")){ lf->zl_stub_at=lf->nops; }   /* 空壳占位(psz<16): 记正位 */
    else if(strstr(nm,".GL")&&psz>=4){ o->type=1; memcpy(&o->g,pay,4); o->foff=curfoff; lf->nops++; }
    else if((strstr(nm,"TREF")||strstr(nm,"xlayer"))&&psz>=4){
        o->type=4; memcpy(&o->t,pay,4); o->foff=curfoff; lf->nops++; }
}
static int lfile_load(const char*path,lfile_t*lf){
    memset(lf,0,sizeof(*lf));
    lf->zl_stub_at=-1;
    int fd=open(path,O_RDONLY); if(fd<0) return -1;
    struct stat st2; if(fstat(fd,&st2)!=0){ close(fd); return -1; }
    lf->msz=(size_t)st2.st_size;
    lf->map=mmap(NULL,lf->msz,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    if(lf->map==MAP_FAILED){ lf->map=NULL; return -1; }
    const uint8_t*p=lf->map,*end=lf->map+lf->msz;
    if(lf->msz<12||memcmp(p,"DQL2",4)!=0){ munmap(lf->map,lf->msz); lf->map=NULL; return -1; }
    uint32_t nrec; memcpy(&nrec,p+8,4); p+=12;
    lf->szG=(size_t)MOEI*go1b_blk_row_bytes(DIM);
    lf->szD=(size_t)DIM*go1b_blk_row_bytes(MOEI);
    for(uint32_t i=0;i<nrec&&p+112<=end;i++){
        char nm[17]; memcpy(nm,p,16); nm[16]=0;
        uint64_t vol,psz; memcpy(&vol,p+80,8); memcpy(&psz,p+88,8);
        float m4; memcpy(&m4,p+108,4);
        int vd; memcpy(&vd,p+112,4);
        const uint8_t*pay=p+116;
        p=pay+psz; if(p>end) break;
        if(!strcmp(nm,"1bit")&&psz>=(uint64_t)NEXP*(2*lf->szG+lf->szD)){
            lf->w1=pay; lf->w3=pay+(size_t)NEXP*lf->szG; lf->w2=pay+2*(size_t)NEXP*lf->szG;
        } else if(!strcmp(nm,"g2hot")&&psz>=24){ /* 热 go2b 内嵌(DQG2 布局原样): 指针直指主 map, 无独立 mmap */
            uint32_t mg2,kh; memcpy(&mg2,pay,4); memcpy(&kh,pay+12,4);
            size_t szG2=(size_t)MOEI*go2b_row_bytes(DIM), szD2=(size_t)DIM*go2b_row_bytes(MOEI);
            size_t hdr2=g2_sidecar_hdr((int)kh);
            if(mg2==G2SC_MAGIC&&kh>0&&kh<=256&&psz>=hdr2+2*(uint64_t)kh*szG2+(uint64_t)kh*szD2){
                lf->g2k=(int)kh;
                for(int e2=0;e2<256;e2++) lf->g2slot[e2]=-1;
                const uint16_t*idsp=(const uint16_t*)(pay+24);
                for(uint32_t i2=0;i2<kh;i2++){ uint16_t ee=idsp[i2]; if(ee<256) lf->g2slot[ee]=(int16_t)i2; }
                lf->g2w1=pay+hdr2; lf->g2w3=lf->g2w1+(size_t)kh*szG2; lf->g2w2=lf->g2w3+(size_t)kh*szG2;
            }
        } else if(vd==1){   /* 超冠混装: 主文件 op 记录即权威 */
            parse_op_rec(lf,nm,pay,psz,(size_t)(pay-lf->map));
        }
    }
    if(!lf->w1){ munmap(lf->map,lf->msz); lf->map=NULL; return -1; }
    /* go2b 侧车(有则挂): 热专家 2bit 覆盖 — 回放/反修在合并态前向。冷 dql 热槽位是稀疏洞,
     * 侧车缺失时热专家会 dequant 全零 → 硬拒加载(禁静默错) */
    { char gp[512];
      uint32_t Lh; memcpy(&Lh,lf->map+4,4);   /* L 从主文件头取(侧车路径推导需层号) */
      g2_sidecar_path(path,(int)Lh,gp,sizeof(gp));
      int g2fd=lf->g2k>0?-1:open(gp,O_RDONLY);   /* 内嵌 g2hot 已认领 → 独立侧车不再挂 */
      if(g2fd>=0){
          struct stat gst; fstat(g2fd,&gst); lf->g2msz=(size_t)gst.st_size;
          lf->g2map=mmap(NULL,lf->g2msz,PROT_READ,MAP_PRIVATE,g2fd,0); close(g2fd);
          if(lf->g2map==MAP_FAILED){ lf->g2map=NULL; }
          else {
              uint32_t mg2,kh; memcpy(&mg2,lf->g2map,4); memcpy(&kh,lf->g2map+12,4);
              size_t szG2=(size_t)MOEI*go2b_row_bytes(DIM), szD2=(size_t)DIM*go2b_row_bytes(MOEI);
              size_t hdr2=g2_sidecar_hdr((int)kh);
              if(mg2==G2SC_MAGIC&&kh>0&&kh<=256&&lf->g2msz>=hdr2+2*(size_t)kh*szG2+(size_t)kh*szD2){
                  lf->g2k=(int)kh;
                  for(int e2=0;e2<256;e2++) lf->g2slot[e2]=-1;
                  const uint16_t*idsp=(const uint16_t*)(lf->g2map+24);
                  for(uint32_t i2=0;i2<kh;i2++){ uint16_t ee=idsp[i2]; if(ee<256) lf->g2slot[ee]=(int16_t)i2; }
                  lf->g2w1=lf->g2map+hdr2; lf->g2w3=lf->g2w1+(size_t)kh*szG2; lf->g2w2=lf->g2w3+(size_t)kh*szG2;
              } else { munmap(lf->g2map,lf->g2msz); lf->g2map=NULL; }
          }
      }
      /* v2.2 VQ 侧车(有则挂): 冷 w1/w3 + 热全三矩阵字节 */
      { char vqp[512]; vq_sidecar_path(path,(int)Lh,vqp,sizeof(vqp));
        int vfd=open(vqp,O_RDONLY);
        if(vfd>=0){
            struct stat vst; fstat(vfd,&vst); lf->vqmsz=(size_t)vst.st_size;
            lf->vqmap=mmap(NULL,lf->vqmsz,PROT_READ,MAP_PRIVATE,vfd,0); close(vfd);
            if(lf->vqmap==MAP_FAILED){ lf->vqmap=NULL; }
            else { uint32_t mgv; memcpy(&mgv,lf->vqmap,4);
                   if(mgv!=VQSC_MAGIC||lf->vqmsz<vq_hdr_bytes()){ munmap(lf->vqmap,lf->vqmsz); lf->vqmap=NULL; } }
        }
      }
      if(GO2B_HOT&&(int)Lh<64&&G2_K[Lh]>0&&lf->g2k<=0&&!lf->vqmap){
          fprintf(stderr,"[go2b] ★L%u 侧车 %s 缺失/损坏(无 VQ 侧车) — 热槽位是稀疏洞, 拒加载★\n",Lh,gp);
          munmap(lf->map,lf->msz); lf->map=NULL; return -1;
      }
    }
    return 0;
}
/* (旧 lfile_append_xlayer "追加标量伪反修" 已删 — 反修改为 zfile_commit 原地重解 z, 见 backfit_prev) */
static void lfile_free(lfile_t*lf){
    for(int i=0;i<lf->nops;i++){
        if(lf->ops[i].type==3&&lf->ops[i].V8) free(lf->ops[i].V8);
        if(lf->ops[i].type==5&&lf->ops[i].ge) free(lf->ops[i].ge);
        if(lf->ops[i].type==6){ free(lf->ops[i].zlU); free(lf->ops[i].zlV); free(lf->ops[i].zlz); }
        if(lf->ops[i].type==8){ free(lf->ops[i].erf_eid); free(lf->ops[i].erf_tau); free(lf->ops[i].erf_UV); }
    }
    if(lf->vqmap) munmap(lf->vqmap,lf->vqmsz);
    if(lf->g2map) munmap(lf->g2map,lf->g2msz);
    if(lf->opsmap) munmap(lf->opsmap,lf->opsmsz);
    if(lf->map) munmap(lf->map,lf->msz);
}
/* ===== DQZ2 运行时全链侧车: 最终落地 op 链 1:1 序列化(引擎回放的唯一权威载体) =====
 * DQZ1(zfile_all.bin) 只存 SEARCH 阶段每层单条胜者; 反修/堆叠后的最终链只活在层文件的
 * vd=1 记录里, 而层文件会被 merge consume 释放 → 必须独立落盘全链(教训: 2026-07-12 首版
 * GGUF 出炉后链数据随层文件消失, 只能重跑量化找回)。
 * 格式: "DQZ2" u32 | nlayers u32 | 每层{ L u32, nops u32, 每op{ type u32, paysz u32, payload } }
 * payload 与 lfile_load/bytes_moe 口径一致: 1=GL g f32 | 2=GLdyn2 w2p f32[4](w0,w1,特征均值,SD;
 * 特征=‖Fin_s‖₂, clamp[0.25,4]) | 3=GLdyn8 w8 f32[9]+V8 fp16[8*DIM](特征=V8·Fin_s, 同 clamp) |
 * 4=TREF t f32(锚 Fcur=专家累加后) | 5=GE fp16[NEXP](乘 gate 权重, 回放取最后一条)。
 * 1/2/3 锚 shared 基 → 引擎侧整链塌缩为 routed 贡献逐 token 标量: λ←g·λ | λ←c_s·λ | λ←1+t·(λ−1)。 */
/* 单层 op 链发射(DQZ2 层块: {L,nops,每op{type,paysz,payload}}); 返回写入字节数 */
static uint64_t zc_emit_layer(FILE*f, uint32_t Lw, lfile_t*lf){
    uint32_t nops = lf ? (uint32_t)lf->nops : 0;
    fwrite(&Lw,4,1,f); fwrite(&nops,4,1,f);
    uint64_t tot=8;
    for(int i=0;lf&&i<lf->nops;i++){ lop_t*o=&lf->ops[i];
        uint32_t ty=(uint32_t)o->type, psz=0;
        if(o->type==1){ psz=4; fwrite(&ty,4,1,f); fwrite(&psz,4,1,f); fwrite(&o->g,4,1,f); }
        else if(o->type==2){ psz=16; fwrite(&ty,4,1,f); fwrite(&psz,4,1,f); fwrite(o->w2p,4,4,f); }
        else if(o->type==3){ psz=36+(o->V8?(uint32_t)(8*DIM*2):0);
            fwrite(&ty,4,1,f); fwrite(&psz,4,1,f); fwrite(o->w8,4,9,f);
            if(o->V8){ uint16_t*h=malloc((size_t)8*DIM*2);
                for(size_t j=0;j<(size_t)8*DIM;j++) h[j]=go1b_fp32_to_fp16(o->V8[j]);
                fwrite(h,2,(size_t)8*DIM,f); free(h); } }
        else if(o->type==4){ psz=4; fwrite(&ty,4,1,f); fwrite(&psz,4,1,f); fwrite(&o->t,4,1,f); }
        else if(o->type==5&&o->ge){ psz=(uint32_t)NEXP*2;
            fwrite(&ty,4,1,f); fwrite(&psz,4,1,f);
            uint16_t*h=malloc((size_t)NEXP*2);
            for(int e=0;e<NEXP;e++) h[e]=go1b_fp32_to_fp16(o->ge[e]);
            fwrite(h,2,(size_t)NEXP,f); free(h); }
        else if(o->type==6&&o->zlk>0&&o->zlU&&o->zlV&&o->zlz){   /* 冻结 z^L: 头16B + fp16{z,U[dout·k],V[din·k]} */
            uint32_t zk=(uint32_t)o->zlk, din=(uint32_t)DIM, dout=(uint32_t)DIM;
            size_t nh=(size_t)zk+(size_t)zk*din+(size_t)zk*dout;
            psz=16+(uint32_t)(2*nh);
            fwrite(&ty,4,1,f); fwrite(&psz,4,1,f);
            fwrite(&zk,4,1,f); fwrite(&o->zltr,4,1,f); fwrite(&din,4,1,f); fwrite(&dout,4,1,f);
            uint16_t*h=malloc(nh*2); size_t off2=0;
            for(uint32_t i=0;i<zk;i++) h[off2++]=go1b_fp32_to_fp16(o->zlz[i]);
            for(size_t i=0;i<(size_t)dout*zk;i++) h[off2++]=go1b_fp32_to_fp16(o->zlU[i]);
            for(size_t i=0;i<(size_t)din*zk;i++)  h[off2++]=go1b_fp32_to_fp16(o->zlV[i]);
            fwrite(h,2,nh,f); free(h); }
        else { uint32_t z=0; fwrite(&ty,4,1,f); fwrite(&z,4,1,f); }   /* 未知型: 空载荷占位 */
        tot+=8+psz;
    }
    return tot;
}
/* 每层"优化文件"(用户产品形态: 一层两份 = dql_LXX.bin 量化 + opt_LXX.bin 优化):
 * 单层 DQZ2(nlayers=1), 任何 DQZ2 读者可直接消费。导出时写初版, zchain_write 刷终值。 */
static void zc_opt_emit(int L, lfile_t*lf){
    const char*ld=getenv("DS4_LAYER_DIR"); if(!ld) return;
    char op2[512]; snprintf(op2,sizeof(op2),"%s/opt_L%02d.bin",ld,L);
    FILE*f=fopen(op2,"wb"); if(!f) return;
    uint32_t magic=0x325A5144, one=1;
    fwrite(&magic,4,1,f); fwrite(&one,4,1,f);
    zc_emit_layer(f,(uint32_t)L,lf);
    fclose(f);
}
static void zchain_write(void){
    const char*p=getenv("DS4_ZCHAIN"); if(!p) return;
    if(getenv("DS4_MINVOL_MAXL")){   /* ★探针/部分层跑禁写(2026-08-03 事故: 7层探针把 43 层终值 zchain 覆盖成空链) */
        fprintf(stderr,"[zchain] 探针模式(MAXL)跳过落盘, 防覆盖全量终值\n"); return; }
    const char*ld=getenv("DS4_LAYER_DIR"); if(!ld) return;
    FILE*f=fopen(p,"wb"); if(!f){ fprintf(stderr,"[zchain] 写 %s 失败\n",p); return; }
    uint32_t magic=0x325A5144, nlay=(uint32_t)NL;
    fwrite(&magic,4,1,f); fwrite(&nlay,4,1,f);
    uint64_t tot=8; int lay_ok=0, ops_tot=0;
    for(int L=0;L<NL;L++){
        char lp[512]; snprintf(lp,sizeof(lp),"%s/dql_L%02d.bin",ld,L);
        lfile_t lf;
        if(lfile_load(lp,&lf)!=0){ tot+=zc_emit_layer(f,(uint32_t)L,NULL); continue; }
        tot+=zc_emit_layer(f,(uint32_t)L,&lf);
        zc_opt_emit(L,&lf);   /* 刷新每层优化文件为终值(反修/回扫后的链) */
        lay_ok++; ops_tot+=lf.nops; lfile_free(&lf);
    }
    fclose(f);
    printf("ZCHAIN %s layers=%d/%d ops=%d bytes=%llu (%.2f MB) (+opt_LXX.bin×%d 终值)\n",
           p,lay_ok,NL,ops_tot,(unsigned long long)tot,(double)tot/1048576.0,lay_ok);
    fflush(stdout);
}
/* bytes MoE: 量化字节前向 + 修正链回放(shared FP 已在 Fout 里) */
/* 并行专家 worker(bytes_moe): 原子计数器分发 e, 私有 partial 累加, 主线程归约 */
typedef struct { lfile_t*lf; int S; const float*Fin; const int*idx; const float*rw;
                 int *e_next; float *partial; const float*ge; float *partial_c; int ti;
                 int _pad; void *bar; int nchunk; } bmw_t;   /* 分块: 屏障 + 块数(上界走全局 ws_e_end) */
extern volatile int ws_e_end;   /* 本块专家上界: 主线程每块更新一次, 屏障保证可见性 */
/* ★冷热分桶缓存(2026-08-06 用户令"冷热双通道")★: bytes_moe 按专家冷热分离累计
 * routed = R_hot + R_cold(贡献项分桶, 非按 token)。热判定=合并态口径(vq w2 槽非零 /
 * g2slot>=0)。供 GLhc(type7) 求解与回放; 每次 bytes_moe 重写。 */
static float *BM_RH=NULL,*BM_RC=NULL; static int BM_S=0;
/* ★层级消融门(2026-08-19 五指标诊断)★ DS4_REPLAY_SKIP_LAYERS="24,25,..": 回放时
 * 整层跳过修正链(权重字节前向保留) — 与 DS4_REPLAY_SKIP_TYPES 正交组合。 */
static int g_replay_cur_L=-1;
static int replay_layer_skipped(void){
    static int init=0, skip[64]={0};
    if(!init){ init=1; const char*sv=getenv("DS4_REPLAY_SKIP_LAYERS");
        if(sv&&*sv){ char b2[256]; snprintf(b2,256,"%s",sv);
            for(char*tk=strtok(b2,",");tk;tk=strtok(NULL,",")){ int t2=atoi(tk);
                if(t2>=0&&t2<64) skip[t2]=1; }
            fprintf(stderr,"[replay] 层消融: 跳过修正链 层{%s}\n",sv); } }
    return g_replay_cur_L>=0&&g_replay_cur_L<64&&skip[g_replay_cur_L];
}
