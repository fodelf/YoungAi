/* vq_qc.h — VQ 码本量化(v2.2 shipping 配置) 量化器侧: 编码/打包/解码/逐层门日志。
 * 配置(2026-07-25 L23 层级判决 −8.8% GO): 热(64/层) w1/w3/w2 → vq4×512 (9bit/4w=2.25bpw);
 * 冷(192/层) w1/w3 → vq8×256 (1B/8w=1.0bpw), w2 → signref 保持(码本对 1bit×w2 无效实测)。
 * 每矩阵布局: [u32 'DQVQ'][u16 dim][u16 nc][u32 rows][u32 cols]
 *             [码本 nc*dim f16][g_r rows f16][索引流: dim4→9bit×4w 8索引9字节; dim8→1B/8w]
 * 依赖: dq_matmul(fwd.c), g2_inv(go2b_qc.h), go1b_fp16 helpers(onebit_quant)。
 * 编码链与 vq_shim.c/g1c_vq_sweep.vq_gptq 同口径(质量对齐已验: C 0.5120 vs py 0.5121)。 */
#ifndef DS4QUANT_VQ_H
#define DS4QUANT_VQ_H
#include <Accelerate/Accelerate.h>   /* vDSP argmin(vq_assign_q 热点, 2026-07-27) */

#define VQ_MAGIC 0x51565144u   /* 'DQVQ' LE */

static int dq_vq_on(void){ static int v=-1; if(v<0) v=getenv("DS4_VQ")?1:0; return v; }

static size_t vq_idx_bytes(int rows,int cols,int dim,int nc){
    size_t nidx=(size_t)rows*cols/dim;
    if(dim==8&&nc==256) return nidx;                 /* 1B/索引 */
    if(dim==4&&nc==512) return (nidx*9+7)/8+1;       /* 9bit/索引, +1 防解包尾读越界 */
    return (nidx*((size_t)ceil(log2((double)nc)))+7)/8;
}
static size_t vq_payload_bytes(int rows,int cols,int dim,int nc){
    return 16 + (size_t)nc*dim*2 + (size_t)rows*2 + vq_idx_bytes(rows,cols,dim,nc);
}

/* ---- 最近邻(分块 sgemm 语义, 此处用 dq_matmul 批量) ---- */
static void vq_assign_q(const float *V,int nv,int dim,const float *C,int nc,int *idx){
    const int BS=8192;
    float *c2=malloc((size_t)nc*4),*G=malloc((size_t)BS*nc*4),*vtmp=malloc((size_t)nc*4);
    for(int c=0;c<nc;c++){ double s=0; for(int d=0;d<dim;d++) s+=(double)C[c*dim+d]*C[c*dim+d]; c2[c]=(float)s; }
    for(int i0=0;i0<nv;i0+=BS){
        int b=nv-i0<BS?nv-i0:BS;
        dq_matmul(V+(size_t)i0*dim,C,G,b,dim,nc);    /* G[b,nc]=V·Cᵀ */
        /* vDSP argmin(2026-07-27 提速): v=c2-2g 批算+首现最小值索引, 与原标量扫描
         * (严格< 保首现)逐位同判 → 确定性不变。profile: 标量扫描曾 8768 采样登顶。 */
        for(int i=0;i<b;i++){ const float*g=G+(size_t)i*nc;
            float nm2=-2.0f;
            vDSP_vsmul(g,1,&nm2,vtmp,1,(vDSP_Length)nc);
            vDSP_vadd(vtmp,1,c2,1,vtmp,1,(vDSP_Length)nc);
            float mv; vDSP_Length mi;
            vDSP_minvi(vtmp,1,&mv,&mi,(vDSP_Length)nc);
            idx[i0+i]=(int)mi; }
    }
    free(c2);free(G);free(vtmp);
}

/* ---- 编码: kmeans(步长子采样,8轮) + GPTQ 列组反馈 + 行乘子; 输出码本/索引/g_r + dequant ---- */
static void vq_encode_full(const float *W,int rows,int cols,int dim,int nc,
                           const float *X,int n_act,
                           float *Cb_out,int *idx_out,float *gr_out,float *Wq_opt){
    size_t N=(size_t)rows*cols; int nv=(int)(N/dim);
    int nsub=nv<200000?nv:200000, stride=nv/nsub;
    float *sub=malloc((size_t)nsub*dim*4);
    for(int i=0;i<nsub;i++) memcpy(sub+(size_t)i*dim,W+(size_t)i*stride*dim,(size_t)dim*4);
    float *C=Cb_out;
    int cst=nsub/nc; if(cst<1)cst=1;
    for(int c=0;c<nc;c++) memcpy(C+(size_t)c*dim,sub+(size_t)(c*cst)*dim,(size_t)dim*4);
    int *ai=malloc((size_t)nsub*sizeof(int));
    double *acc=malloc((size_t)nc*dim*8); long *cnt=malloc((size_t)nc*8);
    for(int it=0;it<8;it++){
        vq_assign_q(sub,nsub,dim,C,nc,ai);
        memset(acc,0,(size_t)nc*dim*8); memset(cnt,0,(size_t)nc*8);
        for(int i=0;i<nsub;i++){ int c=ai[i]; cnt[c]++;
            for(int d=0;d<dim;d++) acc[(size_t)c*dim+d]+=sub[(size_t)i*dim+d]; }
        for(int c=0;c<nc;c++) if(cnt[c]>0)
            for(int d=0;d<dim;d++) C[(size_t)c*dim+d]=(float)(acc[(size_t)c*dim+d]/cnt[c]);
    }
    free(sub);free(ai);free(acc);free(cnt);
    /* f16 往返码本(与存储一致再分配) */
    for(int i=0;i<nc*dim;i++) C[i]=go1b_fp16_to_fp32(go1b_fp32_to_fp16(C[i]));
    int grp=128;
    float *Wk=malloc((size_t)rows*grp*4),*seg=malloc((size_t)rows*dim*4);
    int *sidx=malloc((size_t)rows*sizeof(int));
    double *H=malloc((size_t)grp*grp*8),*Hi=malloc((size_t)grp*grp*8);
    float *XbT=(X&&n_act>0)?malloc((size_t)grp*n_act*4):NULL;
    float *Wq=Wq_opt?Wq_opt:malloc(N*4); int own=Wq_opt?0:1;
    for(int j0=0;j0<cols;j0+=grp){
        int g=cols-j0<grp?cols-j0:grp; int ok=0;
        if(XbT){
            for(int j=0;j<g;j++) for(int t=0;t<n_act;t++) XbT[(size_t)j*n_act+t]=X[(size_t)t*cols+j0+j];
            for(int i=0;i<g;i++) for(int j=i;j<g;j++){ double s=0;
                const float*xi=XbT+(size_t)i*n_act,*xj=XbT+(size_t)j*n_act;
                for(int t=0;t<n_act;t++) s+=(double)xi[t]*xj[t];
                H[(size_t)i*g+j]=H[(size_t)j*g+i]=s; }
            double dm=0; for(int i=0;i<g;i++) dm+=H[(size_t)i*g+i]; dm/=g;
            for(int i=0;i<g;i++) H[(size_t)i*g+i]+=0.02*(dm+1e-9);
            ok=g2_inv(H,g,Hi)==0;
        }
        for(int r=0;r<rows;r++) memcpy(Wk+(size_t)r*g,W+(size_t)r*cols+j0,(size_t)g*4);
        for(int jj=0;jj+dim<=g;jj+=dim){
            for(int r=0;r<rows;r++) memcpy(seg+(size_t)r*dim,Wk+(size_t)r*g+jj,(size_t)dim*4);
            vq_assign_q(seg,rows,dim,C,nc,sidx);
            for(int r=0;r<rows;r++){
                const float *q=C+(size_t)sidx[r]*dim;
                idx_out[((size_t)r*cols+j0+jj)/dim]=sidx[r];
                memcpy(Wq+(size_t)r*cols+j0+jj,q,(size_t)dim*4);
                if(ok&&jj+dim<g){
                    for(int d=0;d<dim;d++){
                        double hjj=Hi[(size_t)(jj+d)*g+jj+d]; if(fabs(hjj)<1e-30)hjj=1e-30;
                        float er=(float)((seg[(size_t)r*dim+d]-q[d])/hjj);
                        if(er==0.0f) continue;
                        const double *hrow=Hi+(size_t)(jj+d)*g; float *wkr=Wk+(size_t)r*g;
                        for(int j2=jj+dim;j2<g;j2++) wkr[j2]-=er*(float)hrow[j2];
                    } } }
        }
    }
    free(Wk);free(seg);free(sidx);free(H);free(Hi); if(XbT)free(XbT);
    /* 行乘子 g_r(闭式 ridge) + f16 往返 */
    if(X&&n_act>=8){
        float *P=malloc((size_t)n_act*rows*4),*Y=malloc((size_t)n_act*rows*4);
        dq_matmul(X,Wq,P,n_act,cols,rows); dq_matmul(X,W,Y,n_act,cols,rows);
        for(int r=0;r<rows;r++){
            double sp2=0,spy=0;
            for(int t=0;t<n_act;t++){ double p=P[(size_t)t*rows+r]; sp2+=p*p; spy+=p*(double)Y[(size_t)t*rows+r]; }
            double lam=sp2/(n_act+1.0),l0=1e-3*sp2+1e-9; if(lam<l0)lam=l0;
            double gr=(spy+lam)/(sp2+lam); if(gr<0.25)gr=0.25; else if(gr>4.0)gr=4.0;
            float gf=go1b_fp16_to_fp32(go1b_fp32_to_fp16((float)gr));
            gr_out[r]=gf;
            float *o=Wq+(size_t)r*cols; for(int j=0;j<cols;j++) o[j]*=gf;
        }
        free(P);free(Y);
    } else for(int r=0;r<rows;r++) gr_out[r]=1.0f;
    if(own) free(Wq);
}

/* ---- 打包/解包 ---- */
static size_t vq_pack(const float *Cb,const int *idx,const float *gr,
                      int rows,int cols,int dim,int nc,uint8_t *out){
    uint8_t *p=out;
    uint32_t mg=VQ_MAGIC; memcpy(p,&mg,4); p+=4;
    uint16_t d16=(uint16_t)dim,n16=(uint16_t)nc; memcpy(p,&d16,2);p+=2; memcpy(p,&n16,2);p+=2;
    uint32_t r32=(uint32_t)rows,c32=(uint32_t)cols; memcpy(p,&r32,4);p+=4; memcpy(p,&c32,4);p+=4;
    for(int i=0;i<nc*dim;i++){ uint16_t h=go1b_fp32_to_fp16(Cb[i]); memcpy(p,&h,2); p+=2; }
    for(int r=0;r<rows;r++){ uint16_t h=go1b_fp32_to_fp16(gr[r]); memcpy(p,&h,2); p+=2; }
    size_t nidx=(size_t)rows*cols/dim;
    if(dim==8){ for(size_t i=0;i<nidx;i++) *p++=(uint8_t)idx[i]; }
    else {      /* 9bit 打包: 位流 LE */
        size_t nb=(nidx*9+7)/8+1; memset(p,0,nb);
        for(size_t i=0;i<nidx;i++){ uint32_t v=(uint32_t)idx[i]&0x1FF; size_t bit=i*9;
            p[bit>>3]   |=(uint8_t)(v<<(bit&7));
            p[(bit>>3)+1]|=(uint8_t)(v>>(8-(bit&7)));
            if((bit&7)>7-1&&(v>>(16-(bit&7)))) p[(bit>>3)+2]|=(uint8_t)(v>>(16-(bit&7)));
        }
        p+=nb;
    }
    return (size_t)(p-out);
}
static int vq_unpack_dequant(const uint8_t *buf,size_t sz,float *Wout,int *rows_o,int *cols_o){
    const uint8_t *p=buf;
    uint32_t mg; memcpy(&mg,p,4); p+=4; if(mg!=VQ_MAGIC) return -1;
    uint16_t dim,nc; memcpy(&dim,p,2);p+=2; memcpy(&nc,p,2);p+=2;
    uint32_t rows,cols; memcpy(&rows,p,4);p+=4; memcpy(&cols,p,4);p+=4;
    if(rows_o)*rows_o=(int)rows; if(cols_o)*cols_o=(int)cols;
    float *C=malloc((size_t)nc*dim*4);
    for(int i=0;i<nc*dim;i++){ uint16_t h; memcpy(&h,p,2); p+=2; C[i]=go1b_fp16_to_fp32(h); }
    float *gr=malloc((size_t)rows*4);
    for(uint32_t r=0;r<rows;r++){ uint16_t h; memcpy(&h,p,2); p+=2; gr[r]=go1b_fp16_to_fp32(h); }
    size_t nidx=(size_t)rows*cols/dim;
    for(size_t i=0;i<nidx;i++){
        uint32_t v;
        if(dim==8) v=p[i];
        else { size_t bit=i*9; v=(uint32_t)p[bit>>3]|((uint32_t)p[(bit>>3)+1]<<8)|((uint32_t)p[(bit>>3)+2]<<16); v=(v>>(bit&7))&0x1FF; }
        size_t r=(i*dim)/cols;
        float g=gr[r]; const float *cb=C+(size_t)v*dim;
        float *o=Wout+i*dim;
        for(int d=0;d<dim;d++) o[d]=cb[d]*g;
    }
    (void)sz; free(C); free(gr); return 0;
}

/* ---- 层侧车(dql_vq_L%02d.bin): [hdr16][表 256×3 u64 off][载荷...] 偏移确定性 ---- */
#define VQSC_MAGIC 0x4C565144u  /* 'DQVL' */
static void vq_sidecar_path(const char*lf,int L,char*out,size_t n){
    strncpy(out,lf,n-1); out[n-1]=0;
    char*sl=strrchr(out,'/'); size_t dl=sl?(size_t)(sl-out+1):0;
    snprintf(out+dl,n-dl,"dql_vq_L%02d.bin",L);
}
static size_t vq_hdr_bytes(void){ return 16 + 256*3*8; }
/* which: 0=w1 1=w3 2=w2; 返回 0=该矩阵无 VQ 载荷 */
static size_t vq_slot_off(int L,int e,int which,int moei,int dim_model){
    size_t off=vq_hdr_bytes();
    size_t hw13=vq_payload_bytes(moei,dim_model,4,512), hw2=vq_payload_bytes(dim_model,moei,4,512);
    size_t cw13=vq_payload_bytes(moei,dim_model,8,256);
    for(int e2=0;e2<e;e2++){
        if(g2_hot_slot(L,e2)>=0) off+=2*hw13+hw2; else off+=2*cw13;
    }
    int hot=g2_hot_slot(L,e)>=0;
    if(!hot&&which==2) return 0;
    if(which>=1) off+= hot?hw13:cw13;
    if(which>=2) off+= hw13;
    return off;
}
static size_t vq_total_bytes(int L,int moei,int dim_model){
    size_t off=vq_hdr_bytes();
    size_t hw13=vq_payload_bytes(moei,dim_model,4,512), hw2=vq_payload_bytes(dim_model,moei,4,512);
    size_t cw13=vq_payload_bytes(moei,dim_model,8,256);
    for(int e2=0;e2<256;e2++) off+= (g2_hot_slot(L,e2)>=0)?(2*hw13+hw2):(2*cw13);
    return off;
}

/* 编码+打包+落盘一条龙; 返回 dequant 浮点(合并态链), cos_out 可 NULL */
static float *vq_export_matrix(const float*W,int rows,int cols,const float*X,int n_act,
                               int dim,int nc,int vqfd,size_t off,double*cos_out){
    float *Wq=malloc((size_t)rows*cols*4);
    float *Cb=malloc((size_t)nc*dim*4),*gr=malloc((size_t)rows*4);
    int *idx=malloc(((size_t)rows*cols/dim)*sizeof(int));
    vq_encode_full(W,rows,cols,dim,nc,X,n_act,Cb,idx,gr,Wq);
    if(vqfd>0&&off>0){
        size_t pb=vq_payload_bytes(rows,cols,dim,nc);
        uint8_t *buf=malloc(pb);
        size_t got=vq_pack(Cb,idx,gr,rows,cols,dim,nc,buf);
        if(got!=pb) fprintf(stderr,"[vq] pack %zu != 预期 %zu (r%d c%d d%d nc%d)\n",got,pb,rows,cols,dim,nc);
        if(pwrite(vqfd,buf,got,(off_t)off)!=(ssize_t)got) perror("vq-pwrite");
        free(buf);
    }
    if(cos_out){ double num=0,na=0,nb=0; size_t N=(size_t)rows*cols;
        for(size_t i=0;i<N;i++){ num+=(double)W[i]*Wq[i]; na+=(double)W[i]*W[i]; nb+=(double)Wq[i]*Wq[i]; }
        *cos_out=num/(sqrt(na)*sqrt(nb)+1e-30); }
    free(Cb);free(idx);free(gr);
    return Wq;
}

/* signref 同形 API: dequant 浮点(合并态链用) */
static float *dq_quant_expert_vq(const float *W,int nrows,int ncols,const float *X,int n_act,int dim,int nc){
    float *Wq=malloc((size_t)nrows*ncols*4);
    float *Cb=malloc((size_t)nc*dim*4); float *gr=malloc((size_t)nrows*4);
    int *idx=malloc(((size_t)nrows*ncols/dim)*sizeof(int));
    vq_encode_full(W,nrows,ncols,dim,nc,X,n_act,Cb,idx,gr,Wq);
    free(Cb);free(idx);free(gr);
    return Wq;
}
#endif
