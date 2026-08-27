/* ds4quant_xcap.inc.c — 量化链 x 捕获(2026-08-27, 只被 ds4quant_run.c include)。
 *
 * 依据: 实测 FP 锚 fin vs 判决尺回放 Fin 方向 cos 仅 0.9425 / 相对差 32.5% —— 反修
 * 在 FP 轨迹的 x 上解最优, 部署却喂量化链的 x, 是"层内 ER 20% / 端到端归零"的唯一
 * 实测偏差(GE 不吃 x 故端到端兑现 8.3%, z 吃 x 故只剩 0.2%, 40 倍转化率差为证)。
 * 捕获须与判决同一条链, 且跑校准语料(判决语料会泄漏)。格式=zlayer capload 的
 * raw_ffn_in_L<L>(fp16 [S][DIM])。 */

static const char *g_xcap_out = NULL;   /* --xcap-out: 量化链 Fin 捕获目录(fp16) */
/* f64→f16(numpy .astype(float16) 同, 与 zlayer capload 的 f16_to_f32 配对) */
static uint16_t f64_to_f16_q(double d){
    uint64_t x; memcpy(&x,&d,8);
    uint32_t sign=(uint32_t)((x>>48)&0x8000u);
    int e64=(int)((x>>52)&0x7FF);
    uint64_t man=x&0xFFFFFFFFFFFFFULL;
    if(e64==0x7FF) return (uint16_t)(sign|0x7C00u|(man?0x200u:0u));
    if(e64==0) return (uint16_t)sign;
    int e=e64-1023+15;
    if(e>=31) return (uint16_t)(sign|0x7C00u);
    if(e<=0){
        if(e<-10) return (uint16_t)sign;
        man|=0x10000000000000ULL;
        int sh=(int)(42-e+1);
        uint64_t m=man>>sh, r=man&((1ULL<<sh)-1), half=1ULL<<(sh-1);
        if(r>half||(r==half&&(m&1))) m++;
        return (uint16_t)(sign|(uint32_t)m);
    }
    uint64_t m=man>>42, r=man&((1ULL<<42)-1), half=1ULL<<41;
    if(r>half||(r==half&&(m&1))){ m++; if(m==1024){ m=0; e++; if(e>=31) return (uint16_t)(sign|0x7C00u);} }
    return (uint16_t)(sign|((uint32_t)e<<10)|(uint32_t)m);
}
/* 量化链 Fin 落盘(fp16, zlayer capload 格式 raw_ffn_in_L<L>) */
static void xcap_dump_fin(int L, const float *Fin, int S){
    char xp[1024]; snprintf(xp,sizeof(xp),"%s/raw_ffn_in_L%d",g_xcap_out,L);
    FILE*xf=fopen(xp,"wb");
    if(!xf){ fprintf(stderr,"[xcap] L%d 打不开 %s\n",L,xp); return; }
    uint16_t*h=malloc((size_t)S*DIM*2);
    for(size_t i=0;i<(size_t)S*DIM;i++) h[i]=f64_to_f16_q((double)Fin[i]);
    if(fwrite(h,2,(size_t)S*DIM,xf)!=(size_t)S*DIM) fprintf(stderr,"[xcap] L%d 写不满\n",L);
    free(h); fclose(xf);
    if(L==0||L==42) fprintf(stderr,"[xcap] L%d Fin → %s (%d×%d fp16)\n",L,xp,S,DIM);
}
