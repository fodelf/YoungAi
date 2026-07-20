/* ds4quant_layer.c — 整层前向组装 (复用已验证的 C 模块 + st_read HF 读器)。
 * 验证: 跑 layer0 到 Fin(FFN输入), 与 numpy dsv4_fwd 捕获的 ffn_in_L0 对拍。
 * 待续: moe_all + 逐层 run + 动态量化接入。
 * 编译(M1): cc -O3 -lm ds4quant_layer.c -o ds4quant_layer
 */
#include "st_read.c"
#include "ds4quant_fwd.c"
#include "onebit_quant.c"   /* go1b_blk_quantize_joint (joint-LS 输出最优 scale) */

/* go1b 字节 → fp32 dequant (per block: fp16 scale * sign). 供量化后专家前向. */
static void dq_go1b_bytes_dequant(const uint8_t *b, int nrows, int ncols, float *out) {
    int nblk = ncols / GO1B_BLK_QK;
    for (int r=0;r<nrows;r++) {
        const uint8_t *rb = b + (size_t)r*nblk*GO1B_BLK_BYTES;
        for (int bl=0;bl<nblk;bl++) {
            const uint8_t *bd = rb + (size_t)bl*GO1B_BLK_BYTES;
            uint16_t h; memcpy(&h,bd,2);
            /* fp16→fp32 */
            uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,f;
            if(e==0){ if(m==0)f=s<<31; else {e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; f=(s<<31)|(e<<23)|(m<<13);} }
            else if(e==0x1f) f=(s<<31)|(0xff<<23)|(m<<13);
            else f=(s<<31)|((e-15+127)<<23)|(m<<13);
            float scale; memcpy(&scale,&f,4);
            const uint8_t *sg=bd+2;
            for(int k=0;k<GO1B_BLK_QK;k++){ int j=bl*GO1B_BLK_QK+k; if(j>=ncols)break;
                int bit=(sg[k/8]>>(k%8))&1; out[(size_t)r*ncols+j]=bit?scale:-scale; }
        }
    }
}
/* 量化一个专家矩阵(joint-LS)并返回 dequant fp32. X=激活[n_act,ncols]. */
static float *dq_quant_expert(const float *W, int nrows, int ncols, const float *X, int n_act) {
    int nblk=ncols/GO1B_BLK_QK; size_t nb=(size_t)nrows*nblk*GO1B_BLK_BYTES;
    uint8_t *bytes=malloc(nb); go1b_blk_quantize_joint(W,bytes,nrows,ncols,X,n_act);
    float *wq=malloc((size_t)nrows*ncols*sizeof(float)); dq_go1b_bytes_dequant(bytes,nrows,ncols,wq);
    free(bytes); return wq;
}

/* config (DeepSeek-V4-Flash; L0 CR=0 无compressor) */
#define DIM 4096
#define NH 64
#define HD 512
#define RD 64
#define QLR 1024
#define OLR 1024
#define OG 8
#define HCM 4
#define HCIT 20
#define MOEI 2048
#define WIN 128
#define EPS 1e-6f
#define HCEPS 1e-6f
#define ROPE_THETA 10000.0

int main(int argc, char **argv) {
    const char *hf = getenv("DS4_HF"); if (!hf) hf = "/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base";
    st_ctx c; st_open(&c, hf);
    long ids[4] = {100, 200, 300, 400}; int S = 4;
    (void)argc; (void)argv;

    /* embed[ids] → H[S,HCM,DIM] (repeat HCM) */
    long er, ec; float *emb = st_read_weight(&c, "embed.weight", &er, &ec);
    if (!emb) { printf("FAIL embed\n"); return 1; }
    float *H = malloc((size_t)S*HCM*DIM*sizeof(float));
    for (int s=0;s<S;s++) for (int j=0;j<HCM;j++) memcpy(H+((size_t)s*HCM+j)*DIM, emb+(size_t)ids[s]*DIM, DIM*sizeof(float));
    free(emb);

    /* load layer 0 tensors */
    #define LD(v,name) long v##r,v##c; float *v=st_read_weight(&c,name,&v##r,&v##c); if(!v){printf("FAIL %s\n",name);return 1;}
    LD(hc_afn,"layers.0.hc_attn_fn"); LD(hc_asc,"layers.0.hc_attn_scale"); LD(hc_abase,"layers.0.hc_attn_base");
    LD(hc_ffn,"layers.0.hc_ffn_fn"); LD(hc_fsc,"layers.0.hc_ffn_scale"); LD(hc_fbase,"layers.0.hc_ffn_base");
    LD(anorm,"layers.0.attn_norm.weight"); LD(fnorm,"layers.0.ffn_norm.weight");
    LD(wqa,"layers.0.attn.wq_a.weight"); LD(qn,"layers.0.attn.q_norm.weight");
    LD(wqb,"layers.0.attn.wq_b.weight"); LD(wkv,"layers.0.attn.wkv.weight");
    LD(kvn,"layers.0.attn.kv_norm.weight"); LD(sink,"layers.0.attn.attn_sink");
    LD(woa,"layers.0.attn.wo_a.weight"); LD(wob,"layers.0.attn.wo_b.weight");

    /* freqs L0 (CR=0 → orig=0 无 yarn) */
    float *cosr = malloc((size_t)S*(RD/2)*sizeof(float)), *sinr = malloc((size_t)S*(RD/2)*sizeof(float));
    dq_freqs_cis(RD, S, 0.0, ROPE_THETA, 16.0, 32.0, 1.0, cosr, sinr);

    /* hc_pre(attn) → y, post, comb */
    float *y = malloc((size_t)S*DIM*sizeof(float)), *post = malloc((size_t)S*HCM*sizeof(float)), *comb = malloc((size_t)S*HCM*HCM*sizeof(float));
    dq_hc_pre(H, hc_afn, hc_asc, hc_abase, y, post, comb, S, HCM, DIM, 2*HCM+HCM*HCM, HCIT, EPS, HCEPS);
    /* rms(y, attn_norm) */
    float *xn = malloc((size_t)S*DIM*sizeof(float));
    for (int s=0;s<S;s++) dq_rms(y+(size_t)s*DIM, anorm, xn+(size_t)s*DIM, DIM, EPS);
    /* attention */
    float *a = malloc((size_t)S*DIM*sizeof(float));
    dq_attention(xn, wqa, qn, wqb, wkv, kvn, sink, woa, wob, NULL, cosr, sinr,
                 a, S, DIM, NH, HD, RD, QLR, OLR, OG, WIN, 0, 0, EPS);
    /* hc_post(a, H, post, comb) → H2 */
    float *H2 = malloc((size_t)S*HCM*DIM*sizeof(float));
    dq_hc_post(a, H, post, comb, H2, S, HCM, DIM);
    /* hc_pre(ffn) → y2 ; Fin=rms(y2, ffn_norm) */
    float *y2 = malloc((size_t)S*DIM*sizeof(float)), *post2 = malloc((size_t)S*HCM*sizeof(float)), *comb2 = malloc((size_t)S*HCM*HCM*sizeof(float));
    dq_hc_pre(H2, hc_ffn, hc_fsc, hc_fbase, y2, post2, comb2, S, HCM, DIM, 2*HCM+HCM*HCM, HCIT, EPS, HCEPS);
    float *Fin = malloc((size_t)S*DIM*sizeof(float));
    for (int s=0;s<S;s++) dq_rms(y2+(size_t)s*DIM, fnorm, Fin+(size_t)s*DIM, DIM, EPS);

    printf("C Fin_L0 row0 first6:");
    for (int i=0;i<6;i++) printf(" %.5f", Fin[i]);
    printf("\n");

    /* ---- moe_all: gate route(hash) + shared + routed(loop 256专家, st_read逐专家) ---- */
    #define NEXP 256
    #define NACT 6
    #define ROUTE_SCALE 1.5f
    #define SWLIM 10.0f
    LD(gate,"layers.0.ffn.gate.weight");
    LD(t2e_f,"layers.0.ffn.gate.tid2eid");        /* float (I64→float) [vocab,NACT] */
    LD(s1,"layers.0.ffn.shared_experts.w1.weight"); LD(s3,"layers.0.ffn.shared_experts.w3.weight"); LD(s2,"layers.0.ffn.shared_experts.w2.weight");
    long vocab = t2e_fr;
    int *t2e = malloc((size_t)vocab*NACT*sizeof(int));
    for (size_t i=0;i<(size_t)vocab*NACT;i++) t2e[i]=(int)t2e_f[i];
    int *idx = malloc((size_t)S*NACT*sizeof(int)); float *rw = malloc((size_t)S*NACT*sizeof(float));
    dq_gate_route_hash(Fin, gate, t2e, ids, idx, rw, S, DIM, NEXP, NACT, ROUTE_SCALE);
    float *Fout = calloc((size_t)S*DIM, sizeof(float));       /* fp 参考 */
    float *Foutq = calloc((size_t)S*DIM, sizeof(float));      /* 1-bit 量化 (shared 仍fp) */
    dq_expert_fp(Fin, s1, s3, s2, NULL, Fout, S, DIM, MOEI, SWLIM);
    dq_expert_fp(Fin, s1, s3, s2, NULL, Foutq, S, DIM, MOEI, SWLIM);
    for (int e=0;e<NEXP;e++) {
        int nt=0; int toks[64]; float ww[64];
        for (int s=0;s<S;s++) for (int a=0;a<NACT;a++) if (idx[(size_t)s*NACT+a]==e){ toks[nt]=s; ww[nt]=rw[(size_t)s*NACT+a]; nt++; }
        if (!nt) continue;
        char n1[128],n3[128],n2[128];
        snprintf(n1,sizeof(n1),"layers.0.ffn.experts.%d.w1.weight",e);
        snprintf(n3,sizeof(n3),"layers.0.ffn.experts.%d.w3.weight",e);
        snprintf(n2,sizeof(n2),"layers.0.ffn.experts.%d.w2.weight",e);
        long r1,c1; float *w1=st_read_weight(&c,n1,&r1,&c1); float *w3=st_read_weight(&c,n3,&r1,&c1); float *w2=st_read_weight(&c,n2,&r1,&c1);
        float *xsub=malloc((size_t)nt*DIM*sizeof(float)); float *asub=calloc((size_t)nt*DIM,sizeof(float));
        for (int i=0;i<nt;i++) memcpy(xsub+(size_t)i*DIM, Fin+(size_t)toks[i]*DIM, DIM*sizeof(float));
        dq_expert_fp(xsub, w1, w3, w2, ww, asub, nt, DIM, MOEI, SWLIM);
        for (int i=0;i<nt;i++) for (int d=0;d<DIM;d++) Fout[(size_t)toks[i]*DIM+d]+=asub[(size_t)i*DIM+d];
        /* ★动态 1-bit 量化 (joint-LS 用 Fin 激活): 量化 w1/w3/w2 → dequant → expert_fp */
        float *q1=dq_quant_expert(w1,MOEI,DIM,Fin,S), *q3=dq_quant_expert(w3,MOEI,DIM,Fin,S);
        float *hf=malloc((size_t)nt*MOEI*sizeof(float)); /* w2 输入=hf, 用 q1/q3 算 */
        { float *gg=malloc((size_t)nt*MOEI*sizeof(float)),*uu=malloc((size_t)nt*MOEI*sizeof(float));
          dq_matmul(xsub,q1,gg,nt,DIM,MOEI); dq_matmul(xsub,q3,uu,nt,DIM,MOEI);
          for(size_t i=0;i<(size_t)nt*MOEI;i++){float g2=gg[i],u2=uu[i]; if(SWLIM>0){if(u2>SWLIM)u2=SWLIM;if(u2<-SWLIM)u2=-SWLIM;if(g2>SWLIM)g2=SWLIM;} hf[i]=dq_silu(g2)*u2;} free(gg);free(uu);}
        float *q2=dq_quant_expert(w2,DIM,MOEI,hf,nt);
        float *asubq=calloc((size_t)nt*DIM,sizeof(float));
        dq_expert_fp(xsub, q1, q3, q2, ww, asubq, nt, DIM, MOEI, SWLIM);
        for (int i=0;i<nt;i++) for (int d=0;d<DIM;d++) Foutq[(size_t)toks[i]*DIM+d]+=asubq[(size_t)i*DIM+d];
        free(w1);free(w3);free(w2);free(xsub);free(asub);free(q1);free(q3);free(q2);free(hf);free(asubq);
    }
    /* 每层 1-bit 质量: cos(Fout_fp, Fout_quant) + relL2 (C 引擎实测) */
    double dot=0,nf=0,nq=0,err=0;
    for (size_t i=0;i<(size_t)S*DIM;i++){ dot+=(double)Fout[i]*Foutq[i]; nf+=(double)Fout[i]*Fout[i]; nq+=(double)Foutq[i]*Foutq[i]; double d=(double)Fout[i]-Foutq[i]; err+=d*d; }
    printf("C Fout_L0 fp row0: %.5f %.5f | 1bit row0: %.5f %.5f\n", Fout[0],Fout[1],Foutq[0],Foutq[1]);
    printf("★L0 1-bit 质量: cos=%.4f relL2=%.4f★\n", dot/(sqrt(nf)*sqrt(nq)), sqrt(err/nf));
    return 0;
}
