/* dsq_lfile.inc.c — 层文件(dql_LXX.bin)读取与修正链: op 记录解析(旧混装+zrec 外挂并
 * 挂)、lfile_load、DQZ2 op 链发射(zc_emit_layer/zc_opt_emit/zchain_write)、bytes_moe
 * 回放前奏。2026-08-31 自 ds4quant_run_p3.inc.c 尾段原字节切出(linecount), 逻辑零改动。 */
/* op 记录解析(主文件旧混装 与 zrec 外挂 共用): curfoff=载荷在其宿主文件内的偏移 */
static int g_parse_ext=0;   /* 1=正在解析 zrec 外挂(记录打 ext 标, 禁原地改写) */
static void parse_op_rec(lfile_t*lf,const char*nm,const uint8_t*pay,uint64_t psz,size_t curfoff){
    if(lf->nops>=LOPS_MAX){   /* zrec 并链后单层=dql 内嵌+外挂之和, 更易逼近容量 */
        fprintf(stderr,"[lfile]★op 链超容量 %d(记录 %.16s 装不下): 静默丢修正=判决模型≠部署模型, "
                       "停车; 提高 LOPS_MAX 重编★\n",LOPS_MAX,nm);
        exit(1);
    }
    lop_t*o=&lf->ops[lf->nops];
    memset(o,0,sizeof(*o)); o->ext=g_parse_ext;
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
        if(zk>0&&zk<=DS4_AMP_ZK_MAX&&(din==(uint32_t)DIM||din==3u*(uint32_t)DIM)&&dout==(uint32_t)DIM
           &&psz>=DS4_AMP_OP_HDR+(uint64_t)2*DS4_AMP_ZL_ELEMS(zk,din,dout)){
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
    /* 走查守卫用整头 116(旧 +112 会在截断文件上把 vd 读越界 4 字节) */
    for(uint32_t i=0;i<nrec&&p+DS4_AMP_REC_HDR<=end;i++){
        char nm[17]; memcpy(nm,p,16); nm[16]=0;
        uint64_t vol,psz; memcpy(&vol,p+DS4_AMP_REC_OFF_VOL,8); memcpy(&psz,p+DS4_AMP_REC_OFF_PSZ,8);
        float m4; memcpy(&m4,p+DS4_AMP_REC_OFF_MEAN,4);
        int vd; memcpy(&vd,p+DS4_AMP_REC_OFF_VD,4);
        const uint8_t*pay=p+DS4_AMP_REC_HDR;
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
      /* ★zrec 并链(2026-08-31 回放盲区正修)★ B路 ZLGATE 与 zlayer INJ=2 落地的 z/GE 住独立
       * zrec_L%02d.bin(dql 不动), 而回放只执行 dql 内嵌 op ⇒ sweep 基线/终验/判决尺全都看不见
       * 已落地修正, 引擎(zrec→zchain type5/6)却会执行 —— 判决的模型≠部署的模型, 跨调用的
       * 序贯前提也断裂。这里把 zrec 记录并进 op 链, 回放=部署。现役写者(zlayer INJ=2 /
       * ZLGATE zrec 直写)都不同时注入 dql ⇒ 无双重应用; 空文件=INJ=2 的"闸拒"标记, 跳过。 */
      { char zp[512], zdir[512]; snprintf(zdir,sizeof(zdir),"%s",path);
        char*zsl=strrchr(zdir,'/'); if(zsl)*zsl=0; else snprintf(zdir,sizeof(zdir),".");
        snprintf(zp,sizeof(zp),"%s/zrec_L%02d.bin",zdir,(int)Lh);
        int zfd=open(zp,O_RDONLY);
        if(zfd>=0){ struct stat zst;
            if(!fstat(zfd,&zst)&&zst.st_size>=DS4_AMP_REC_HDR){
                lf->opsmsz=(size_t)zst.st_size;
                lf->opsmap=mmap(NULL,lf->opsmsz,PROT_READ,MAP_PRIVATE,zfd,0);
                if(lf->opsmap==MAP_FAILED){ lf->opsmap=NULL; lf->opsmsz=0; }
                else { const uint8_t*q=lf->opsmap,*qe=lf->opsmap+lf->opsmsz; int nz0=lf->nops;
                    g_parse_ext=1;
                    while(q+DS4_AMP_REC_HDR<=qe){ char nm2[17]; memcpy(nm2,q,16); nm2[16]=0;
                        uint64_t psz2; memcpy(&psz2,q+DS4_AMP_REC_OFF_PSZ,8);
                        int vd2; memcpy(&vd2,q+DS4_AMP_REC_OFF_VD,4);
                        const uint8_t*pay2=q+DS4_AMP_REC_HDR; q=pay2+psz2; if(q>qe) break;
                        if(vd2==1) parse_op_rec(lf,nm2,pay2,psz2,(size_t)(pay2-(const uint8_t*)lf->opsmap)); }
                    g_parse_ext=0;
                    if(lf->nops>nz0) fprintf(stderr,"[zrec并链] L%02d +%d op ← %s(部署态回放)\n",
                                             (int)Lh,lf->nops-nz0,zp);
                } }
            close(zfd); }
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
            /* din 按记录真值走(旧写死 DIM: ftA op(zdin=3D)导出时 V 静默截 2/3 且头自述错) */
            uint32_t zk=(uint32_t)o->zlk, din=(uint32_t)(o->zdin?o->zdin:DIM), dout=(uint32_t)DIM;
            size_t nh=DS4_AMP_ZL_ELEMS(zk,din,dout);
            psz=DS4_AMP_OP_HDR+(uint32_t)(2*nh);
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
    const char*ld=g_cli.layer_dir; if(!ld) return;
    char op2[512]; snprintf(op2,sizeof(op2),"%s/opt_L%02d.bin",ld,L);
    FILE*f=fopen(op2,"wb"); if(!f) return;
    uint32_t magic=0x325A5144, one=1;
    fwrite(&magic,4,1,f); fwrite(&one,4,1,f);
    zc_emit_layer(f,(uint32_t)L,lf);
    fclose(f);
}
static void zchain_write(void){
    const char*p=g_cli.zchain; if(!p) return;
    if(g_cli.minvol_maxl>0){   /* ★探针/部分层跑禁写(2026-08-03 事故: 7层探针把 43 层终值 zchain 覆盖成空链) */
        fprintf(stderr,"[zchain] 探针模式(--minvol-maxl)跳过落盘, 防覆盖全量终值\n"); return; }
    const char*ld=g_cli.layer_dir; if(!ld) return;
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
static int replay_layer_skipped(void){ return 0; }   /* 层消融脚手架已删(2026-08-31 用完即删律) */
