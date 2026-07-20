/* st_read.c — 最小 DeepSeek V4 HF 读器 (C): FP8 E4M3 + 128×128 块 F32 scale safetensors。
 * 对应 ds4reader.py。忠实移植: e4m3 LUT + 块 scale dequant + safetensors JSON 头(最小解析)。
 * 验证: --selftest 读一个真实 HF 权重前几值, 与 numpy R.read_weight 对拍。
 * 编译: cc -O3 -DST_READ_SELFTEST -lm st_read.c -o st_selftest
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static float ST_LUT[256];
static void st_lut_init(void) {
    for (int b = 0; b < 256; b++) {
        int s=(b>>7)&1, e=(b>>3)&0xF, m=b&0x7; float sign=s?-1.f:1.f;
        if (e==0) ST_LUT[b]=sign*(m/8.0f)*powf(2.0f,-6);
        else if (e==0xF && m==0x7) ST_LUT[b]=NAN;
        else ST_LUT[b]=sign*(1.0f+m/8.0f)*powf(2.0f,(float)(e-7));
    }
}

/* 在 JSON 文本里找 "name":{...} 段, 取 dtype/shape[2]/data_offsets[2]. 返回段起始或 NULL. */
static const char *st_find(const char *hdr, const char *name, char *dtype, long *shape, long *off) {
    char key[512]; snprintf(key,sizeof(key),"\"%s\"",name);
    const char *p=strstr(hdr,key); if(!p) return NULL;
    const char *d=strstr(p,"\"dtype\":\""); if(!d) return NULL; d+=9;
    int i=0; while(*d && *d!='"' && i<15) dtype[i++]=*d++; dtype[i]=0;
    const char *sh=strstr(p,"\"shape\":["); if(!sh) return NULL; sh+=9;
    shape[0]=strtol(sh,(char**)&sh,10); shape[1]=1;
    if(*sh==',') { sh++; shape[1]=strtol(sh,(char**)&sh,10); }
    const char *of=strstr(p,"\"data_offsets\":["); if(!of) return NULL; of+=16;
    off[0]=strtol(of,(char**)&of,10); if(*of==',') of++; off[1]=strtol(of,(char**)&of,10);
    return p;
}

typedef struct { char hf[1024]; char *idx_json; } st_ctx;

static char *st_slurp(const char *path, long *len) {
    FILE *f=fopen(path,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){free(b);fclose(f);return NULL;} b[n]=0; fclose(f); if(len)*len=n; return b;
}

static void st_open(st_ctx *c, const char *hf) {
    st_lut_init(); snprintf(c->hf,sizeof(c->hf),"%s",hf);
    char p[1200]; snprintf(p,sizeof(p),"%s/model.safetensors.index.json",hf);
    c->idx_json=st_slurp(p,NULL);
    if(!c->idx_json){ fprintf(stderr,"st_read: no index %s\n",p); exit(1); }
}

/* 找 name 所在 shard 文件名 (index weight_map 里 "name": "shard", 冒号后可能有空格). */
static int st_shard(st_ctx *c, const char *name, char *shard) {
    char key[512]; snprintf(key,sizeof(key),"\"%s\"",name);
    const char *p=strstr(c->idx_json,key); if(!p) return 0; p+=strlen(key);
    while(*p && *p!=':') p++; if(*p==':') p++;      /* 跳到冒号后 */
    while(*p==' '||*p=='\t'||*p=='\n') p++;          /* 跳空白 */
    if(*p!='"') return 0; p++;                       /* 开引号 */
    int i=0; while(*p&&*p!='"'&&i<255) shard[i++]=*p++; shard[i]=0; return 1;
}

/* 读 shard 头: 返回 header json (malloc) + data_start. */
static char *st_shard_hdr(st_ctx *c, const char *shard, long *data_start) {
    char p[1300]; snprintf(p,sizeof(p),"%s/%s",c->hf,shard);
    FILE *f=fopen(p,"rb"); if(!f) return NULL;
    uint64_t n; if(fread(&n,8,1,f)!=1){fclose(f);return NULL;}
    char *hdr=malloc(n+1); if(fread(hdr,1,n,f)!=n){free(hdr);fclose(f);return NULL;} hdr[n]=0;
    *data_start=8+(long)n; fclose(f); return hdr;
}

/* read_weight(name) → dequant f32 [R*C] (malloc). fp8 E4M3 * 128×128 块scale. R/C out. */
float *st_read_weight(st_ctx *c, const char *name, long *R_out, long *C_out) {
    char shard[256]; if(!st_shard(c,name,shard)){fprintf(stderr,"st: no %s\n",name);return NULL;}
    long ds; char *hdr=st_shard_hdr(c,shard,&ds); if(!hdr) return NULL;
    char dt[16]; long shape[2],off[2];
    if(!st_find(hdr,name,dt,shape,off)){free(hdr);return NULL;}
    long Rr=shape[0],Cc=shape[1]; char p[1300]; snprintf(p,sizeof(p),"%s/%s",c->hf,shard);
    FILE *f=fopen(p,"rb"); if(!f){free(hdr);return NULL;}
    float *w=malloc((size_t)Rr*Cc*sizeof(float));
    if(strcmp(dt,"F8_E4M3")==0){
        uint8_t *buf=malloc((size_t)Rr*Cc); fseek(f,ds+off[0],SEEK_SET);
        if(fread(buf,1,(size_t)Rr*Cc,f)!=(size_t)Rr*Cc){free(buf);free(w);free(hdr);fclose(f);return NULL;}
        for(size_t i=0;i<(size_t)Rr*Cc;i++) w[i]=ST_LUT[buf[i]]; free(buf);
        /* 块scale name→.scale, F32 [R/128,C/128] */
        char sname[512]; snprintf(sname,sizeof(sname),"%s",name);
        char *ww=strstr(sname,".weight"); if(ww) strcpy(ww,".scale");
        char shard2[256]; st_shard(c,sname,shard2); long ds2; char *hdr2=st_shard_hdr(c,shard2,&ds2);
        char dt2[16]; long ssh[2],soff[2];
        if(hdr2 && st_find(hdr2,sname,dt2,ssh,soff)){
            long sbr=ssh[0],sbc=ssh[1]; char p2[1300]; snprintf(p2,sizeof(p2),"%s/%s",c->hf,shard2);
            FILE *f2=fopen(p2,"rb"); float *sc=malloc((size_t)sbr*sbc*sizeof(float));
            fseek(f2,ds2+soff[0],SEEK_SET); fread(sc,4,(size_t)sbr*sbc,f2); fclose(f2);
            for(long r=0;r<Rr;r++) for(long cc=0;cc<Cc;cc++)
                w[(size_t)r*Cc+cc]*=sc[(size_t)(r/128)*sbc+(cc/128)];
            free(sc);
        }
        if(hdr2) free(hdr2);
    } else if(strcmp(dt,"BF16")==0){
        uint16_t *buf=malloc((size_t)Rr*Cc*2); fseek(f,ds+off[0],SEEK_SET);
        fread(buf,2,(size_t)Rr*Cc,f);
        for(size_t i=0;i<(size_t)Rr*Cc;i++){ uint32_t u=(uint32_t)buf[i]<<16; memcpy(&w[i],&u,4);} free(buf);
    } else if(strcmp(dt,"I64")==0){ /* int64 → float (eid≤255精确) */
        int64_t *buf=malloc((size_t)Rr*Cc*8); fseek(f,ds+off[0],SEEK_SET);
        fread(buf,8,(size_t)Rr*Cc,f);
        for(size_t i=0;i<(size_t)Rr*Cc;i++) w[i]=(float)buf[i]; free(buf);
    } else { /* F32 */
        fseek(f,ds+off[0],SEEK_SET); fread(w,4,(size_t)Rr*Cc,f);
    }
    fclose(f); free(hdr); if(R_out)*R_out=Rr; if(C_out)*C_out=Cc; return w;
}

#ifdef ST_READ_SELFTEST
int main(int argc,char**argv){
    const char *hf=getenv("DS4_HF"); if(!hf) hf="/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base";
    st_ctx c; st_open(&c,hf);
    const char *name=argc>1?argv[1]:"layers.0.attn_norm.weight";
    long R,C; float *w=st_read_weight(&c,name,&R,&C);
    if(!w){printf("FAIL read %s\n",name);return 1;}
    printf("%s R=%ld C=%ld first6:",name,R,C);
    for(int i=0;i<6 && i<R*C;i++) printf(" %.6f",w[i]);
    printf("\n"); free(w); return 0;
}
#endif
