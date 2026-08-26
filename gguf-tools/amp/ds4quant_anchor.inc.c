/* ds4quant_anchor.inc.c — FP 锚定域(anchor_t/alloc/save/load, 原 p2 头段字节搬移;
 * 2026-08-26 建锚 MAP_SHARED 直写改造时 p2 超 500 行, 按语义拆出)。 */
/* ===================== FP 锚定 (anchor) ===================== *
 * FP 遍的完整快照, 只算一次: 每层 MoE 输入(校准=部署口径)/FP 路由/层出口 H(累积对比+前缀恢复)
 * + 最终 logits(判决基线)。header 校验 (S, NLAYERS, ids 哈希) 不符自动重建。 */
typedef struct { uint32_t S; uint64_t idh;
    float *fin;      /* [NLAYERS][S][DIM]      每层 MoE 输入 (rms 后) */
    int32_t *ridx;   /* [NLAYERS][S][NACT]     FP 路由专家 id */
    float *rw;       /* [NLAYERS][S][NACT]     FP 路由权重 */
    float *H;        /* [NLAYERS][S][HCM][DIM] 每层出口 hc 状态 */
    float *logits;   /* [S][VOCAB] */
} anchor_t;
static anchor_t ANC; static int ANC_OK=0, ANC_BUILD=0;
/* ★链态锚直写(2026-08-10 反修v4: 量化链 fin/路由/H 逐层 pwrite, 零 RAM; DS4_CHAIN_ANCHOR)★ */
static int g_chfd=-1; static int g_chS=0;
/* ★统一标准(2026-07-29 用户令): 判决语料=rr_hard(与冠军 v4bf 0.7680 同尺)。ANC2=rr FP 锚,
 * H2=贪心逐层推进的 rr 判决链; v5mini 降级为校准+层门护栏(判决数字一律 rr 口径)。 */
static anchor_t ANC2; static int ANC2_OK=0;
static long *g_ids2=NULL; static int g_S2=0;

static const char *anchor_path(void){ const char*p=getenv("DS4_ANCHOR"); return p?p:"/tmp/ds4quant_anchor.bin"; }
static uint64_t dq_ids_hash(const long*ids,int S){
    uint64_t h=1469598103934665603ULL;
    for(int i=0;i<S;i++){ uint64_t v=(uint64_t)ids[i];
        for(int b=0;b<8;b++){ h^=(v>>(8*b))&0xFF; h*=1099511628211ULL; } }
    return h;
}
static int ANC_MMAP=0;   /* 1=ANC 数组指向 mmap(禁 free); 声明前移(建锚直写路要用) */
static int ANC_SHARED=0; static int g_ancfd=-1; static void*g_ancmap=NULL; static size_t g_ancneed=0;
static void anchor_alloc(int S){
    ANC.S=(uint32_t)S;
    size_t fin_b=(size_t)NLAYERS*S*DIM*4, ridx_b=(size_t)NLAYERS*S*NACT*4,
           H_b=(size_t)NLAYERS*S*HCM*DIM*4, lg_b=(size_t)S*VOCAB*4;
    /* ★建锚大 S 全量驻 RAM 会 OOM(S=32768 时 fin+H=115G, 2026-08-26 watchdog 击杀实锤):
     * 会落盘的锚(NL=43+DS4_ANCHOR)改为目标文件 MAP_SHARED 直写 — 写=脏文件页, 内核
     * writeback 落盘, RSS 只剩真工作集; logits 仍由 fwd_all malloc(单段, save 时 pwrite 进尾)。
     * 截断锚(NL<43, 本来不落盘)保留 malloc 原样。mmap 失败=响亮停车, 不许滑回 OOM 路。 */
    if(NLAYERS==43 && getenv("DS4_ANCHOR")){
        g_ancneed=40+fin_b+2*ridx_b+H_b+lg_b;
        g_ancfd=open(anchor_path(),O_RDWR|O_CREAT,0644);
        if(g_ancfd<0||ftruncate(g_ancfd,(off_t)g_ancneed)!=0){
            fprintf(stderr,"anchor: 建锚文件失败 %s\n",anchor_path()); exit(1); }
        g_ancmap=mmap(NULL,g_ancneed,PROT_READ|PROT_WRITE,MAP_SHARED,g_ancfd,0);
        if(g_ancmap==MAP_FAILED){ fprintf(stderr,"anchor: MAP_SHARED 失败\n"); exit(1); }
        uint8_t*p=(uint8_t*)g_ancmap+40;
        ANC.fin=(float*)p;    p+=fin_b;
        ANC.ridx=(int32_t*)p; p+=ridx_b;
        ANC.rw=(float*)p;     p+=ridx_b;
        ANC.H=(float*)p;
        ANC_SHARED=1; ANC_MMAP=1;
        fprintf(stderr,"anchor: 建锚 MAP_SHARED 直写 %.1f GiB(脏页走 writeback, 零驻留)\n",
                g_ancneed/1073741824.0);
        return;
    }
    ANC.fin =malloc(fin_b);
    ANC.ridx=malloc(ridx_b);
    ANC.rw  =malloc(ridx_b);
    ANC.H   =malloc(H_b);
    if(!ANC.fin||!ANC.ridx||!ANC.rw||!ANC.H){ fprintf(stderr,"anchor: 内存分配失败\n"); exit(1); }
}
/* 大块落盘必须分块+逐块 fsync: 19G 锚一把 fwrite 的脏文件页叠在 ~19G 换页捕获缓冲上,
 * 16G 机上 memorystatus 直接 SIGKILL(08-11 FP_ONLY 与 quant 内捕获两连杀实锤)。 */
static int wr_fs(FILE*f,const void*p,size_t nbytes){
    const uint8_t*b=p; const size_t CH=268435456;
    for(size_t o=0;o<nbytes;o+=CH){ size_t n=nbytes-o<CH?nbytes-o:CH;
        if(fwrite(b+o,1,n,f)!=n) return 0;
        fflush(f); fsync(fileno(f)); }
    return 1;
}
static int anchor_save(void){
    /* ★NL<43 禁覆盖(2026-08-03 事故: DS4_NL=1 探针反修现场重建 1 层锚并覆盖 6.46G 全量锚,
     * 全链被迫重跑 FP 锚定遍): 截断跑的锚只在内存用, 不落盘。 */
    if(NLAYERS<43){ fprintf(stderr,"anchor: NL=%d<43 截断锚不落盘(防覆盖全量锚)\n",NLAYERS); return 0; }
    if(ANC_SHARED){
        /* fin/ridx/rw/H 已在文件页里; logits 分块 pwrite 进尾段; 头(40B)最后写=提交标记
         * (中途被杀的半成品 magic 不成立, anchor_load 自动判非缓存重建)。 */
        size_t lg_b=4ul*(size_t)ANC.S*VOCAB, lg_off=g_ancneed-lg_b;
        const uint8_t*b=(const uint8_t*)ANC.logits; const size_t CH=268435456;
        for(size_t o=0;o<lg_b;o+=CH){ size_t n=lg_b-o<CH?lg_b-o:CH;
            if(pwrite(g_ancfd,b+o,n,(off_t)(lg_off+o))!=(ssize_t)n){
                fprintf(stderr,"anchor: logits 写失败\n"); return 0; } }
        if(msync(g_ancmap,g_ancneed,MS_SYNC)!=0) fprintf(stderr,"anchor: msync 警告(继续)\n");
        uint32_t hd[8]={0x32415144u,ANC.S,HCM,DIM,(uint32_t)NLAYERS,VOCAB,NACT,0};
        uint8_t hb[40]; memcpy(hb,hd,32); memcpy(hb+32,&ANC.idh,8);
        if(pwrite(g_ancfd,hb,40,0)!=40){ fprintf(stderr,"anchor: 头写失败\n"); return 0; }
        fsync(g_ancfd);
        return 1;
    }
    FILE*f=fopen(anchor_path(),"wb"); if(!f){ fprintf(stderr,"anchor: 写 %s 失败\n",anchor_path()); return 0; }
    uint32_t hd[8]={0x32415144u,ANC.S,HCM,DIM,(uint32_t)NLAYERS,VOCAB,NACT,0};  /* "DQA2" LE */
    int S=(int)ANC.S, ok=1;
    ok &= fwrite(hd,4,8,f)==8 && fwrite(&ANC.idh,8,1,f)==1;
    ok &= wr_fs(f,ANC.fin ,4ul*(size_t)NLAYERS*S*DIM);
    ok &= wr_fs(f,ANC.ridx,4ul*(size_t)NLAYERS*S*NACT);
    ok &= wr_fs(f,ANC.rw  ,4ul*(size_t)NLAYERS*S*NACT);
    ok &= wr_fs(f,ANC.H   ,4ul*(size_t)NLAYERS*S*HCM*DIM);
    ok &= wr_fs(f,ANC.logits,4ul*(size_t)S*VOCAB);
    fclose(f); if(!ok) fprintf(stderr,"anchor: 写不完整\n");
    return ok;
}
static int anchor_load(int S,uint64_t idh){
    /* ★反修内存架构(2026-07-31 用户令"反修必须跑, 内存大改代码"): 锚 ~4.5G 由 malloc+fread
     * (脏页不可回收)改为整文件 mmap 零拷贝 — MAP_PRIVATE 写时复制(捕获路偶发写只脏所触页),
     * 读页=干净文件页, 内存压力下可逐出重读 ⇒ 反修基线脏内存 -4.5G, 终局 sweep 进 12G 红线。 */
    int fd=open(anchor_path(),O_RDONLY); if(fd<0) return 0;
    uint32_t hd[8]; uint64_t h;
    if(read(fd,hd,32)!=32||read(fd,&h,8)!=8){close(fd);return 0;}
    if(hd[0]!=0x32415144u||hd[1]!=(uint32_t)S||hd[2]!=HCM||hd[3]!=DIM||
       hd[4]!=(uint32_t)NLAYERS||hd[5]!=VOCAB||hd[6]!=NACT||h!=idh){close(fd);return 0;}
    size_t fin_b=(size_t)NLAYERS*S*DIM*4, ridx_b=(size_t)NLAYERS*S*NACT*4, rw_b=ridx_b,
           H_b=(size_t)NLAYERS*S*HCM*DIM*4, lg_b=(size_t)S*VOCAB*4;
    size_t need=40+fin_b+ridx_b+rw_b+H_b+lg_b;
    struct stat stt;
    if(fstat(fd,&stt)!=0||(size_t)stt.st_size<need){ close(fd); return 0; }
    void*mp=mmap(NULL,need,PROT_READ|PROT_WRITE,MAP_PRIVATE,fd,0);
    close(fd);
    if(mp==MAP_FAILED) return 0;
    uint8_t*p=(uint8_t*)mp+40;
    ANC.S=(uint32_t)S; ANC.idh=idh;
    ANC.fin=(float*)p;        p+=fin_b;
    ANC.ridx=(int32_t*)p;     p+=ridx_b;
    ANC.rw=(float*)p;         p+=rw_b;
    ANC.H=(float*)p;          p+=H_b;
    ANC.logits=(float*)p;
    ANC_MMAP=1;
    fprintf(stderr,"anchor: mmap 零拷贝装载 %.2f GiB(干净页可逐出)\n",need/1073741824.0);
    return 1;
}

