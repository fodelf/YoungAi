/* corr_stats.c — dump per-tensor magnitude stats (max|.|, mean|.|, L2, nonzero%)
 * for the F32 tensors in a go1b corr sidecar GGUF.  Used to verify whether the
 * four-loss z solve produced a MEANINGFUL correction or a near-zero/degenerate
 * one (the latter => corr applies but changes nothing => "z doesn't work").
 *
 *   cc -O2 -o corr_stats corr_stats.c && ./corr_stats ../../gguf/ds4-go1b-corr.gguf blk.0.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static void die(const char*m){ fprintf(stderr,"corr_stats: %s\n",m); exit(1); }
static void* xmalloc(size_t n){ void*p=malloc(n); if(!p)die("oom"); return p; }
static uint64_t r_u64(FILE*f){ uint64_t v; if(fread(&v,8,1,f)!=1)die("r u64"); return v; }
static uint32_t r_u32(FILE*f){ uint32_t v; if(fread(&v,4,1,f)!=1)die("r u32"); return v; }
static char* r_str(FILE*f){ uint64_t n=r_u64(f); char*s=xmalloc(n+1); if(n&&fread(s,1,n,f)!=n)die("r str"); s[n]=0; return s; }

enum { GT_U8=0,GT_I8,GT_U16,GT_I16,GT_U32,GT_I32,GT_F32,GT_BOOL,GT_STR,GT_ARR,GT_U64,GT_I64,GT_F64 };
static size_t gt_size(uint32_t t){
    switch(t){ case GT_U8:case GT_I8:case GT_BOOL:return 1; case GT_U16:case GT_I16:return 2;
        case GT_U32:case GT_I32:case GT_F32:return 4; case GT_U64:case GT_I64:case GT_F64:return 8; default:return 0; }
}
static void skip_kv(FILE*f,uint32_t t){
    if(t==GT_STR){ uint64_t n=r_u64(f); if(fseek(f,(long)n,SEEK_CUR))die("seek"); return; }
    if(t==GT_ARR){ uint32_t et=r_u32(f); uint64_t n=r_u64(f);
        if(et==GT_STR){ for(uint64_t i=0;i<n;i++){ uint64_t l=r_u64(f); if(fseek(f,(long)l,SEEK_CUR))die("seek"); } }
        else { size_t s=gt_size(et); if(!s)die("arr elem"); if(fseek(f,(long)(s*n),SEEK_CUR))die("seek"); }
        return; }
    size_t s=gt_size(t); if(!s)die("kv type"); if(fseek(f,(long)s,SEEK_CUR))die("seek");
}

#define GGML_TYPE_F32 0
#define ALIGN 32ull
static uint64_t pad_up(uint64_t x,uint64_t a){ return (x+a-1)/a*a; }

typedef struct { char name[96]; uint64_t ne[4]; uint32_t type; uint64_t rel; } tinfo;

int main(int argc,char**argv){
    if(argc<2) die("usage: corr_stats FILE.gguf [name-substr]");
    const char*path=argv[1]; const char*filt=argc>2?argv[2]:NULL;
    FILE*f=fopen(path,"rb"); if(!f)die("open");
    char magic[4]; if(fread(magic,1,4,f)!=4||memcmp(magic,"GGUF",4))die("bad magic");
    (void)r_u32(f); uint64_t n_tensors=r_u64(f), n_kv=r_u64(f);
    for(uint64_t k=0;k<n_kv;k++){ char*key=r_str(f); uint32_t vt=r_u32(f); skip_kv(f,vt); free(key); }
    tinfo*T=xmalloc(sizeof(tinfo)*n_tensors);
    for(uint64_t t=0;t<n_tensors;t++){
        char*nm=r_str(f); snprintf(T[t].name,sizeof T[t].name,"%s",nm); free(nm);
        uint32_t nd=r_u32(f); T[t].ne[0]=T[t].ne[1]=T[t].ne[2]=T[t].ne[3]=1;
        for(uint32_t d=0;d<nd&&d<4;d++) T[t].ne[d]=r_u64(f);
        for(uint32_t d=4;d<nd;d++) (void)r_u64(f);
        T[t].type=r_u32(f); T[t].rel=r_u64(f);
    }
    long pos=ftell(f); if(pos<0)die("ftell");
    uint64_t data_start=pad_up((uint64_t)pos,ALIGN);
    for(uint64_t t=0;t<n_tensors;t++){
        if(filt && !strstr(T[t].name,filt)) continue;
        if(T[t].type!=GGML_TYPE_F32){ printf("%-30s type=%u (non-F32, skip)\n",T[t].name,T[t].type); continue; }
        uint64_t n=T[t].ne[0]*T[t].ne[1]*T[t].ne[2]*T[t].ne[3];
        float*buf=xmalloc(n*4);
        if(fseek(f,(long)(data_start+T[t].rel),SEEK_SET))die("seek data");
        if(fread(buf,4,n,f)!=n)die("read data");
        double mx=0,sum2=0,suma=0; uint64_t nz=0;
        for(uint64_t i=0;i<n;i++){ double v=buf[i],a=fabs(v); if(a>mx)mx=a; sum2+=v*v; suma+=a; if(a>1e-9)nz++; }
        printf("%-30s ne=[%llu,%llu] n=%-9llu max|.|=%-10.4g mean|.|=%-10.4g L2=%-10.4g nz=%.1f%%\n",
            T[t].name,(unsigned long long)T[t].ne[0],(unsigned long long)T[t].ne[1],
            (unsigned long long)n, mx, suma/(double)n, sqrt(sum2), 100.0*(double)nz/(double)n);
        free(buf);
    }
    fclose(f); return 0;
}
