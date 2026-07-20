/* lut_cov.c — print, per layer, which experts a SPARSE go1b residual actually
 * covers (blk.{L}.ffn_res_lut.weight[e] >= 0 => covered, -1 => raw 1-bit).
 * Cross-reference with a routing dump (DS4_DUMP_ACTIVE "PICKS L{n} ...") to see
 * whether a test prompt's routed experts fall inside the residual's coverage.
 *
 *   cc -O2 -o lut_cov lut_cov.c && ./lut_cov ../../gguf/ds4-go1b-res.gguf > cov.txt
 * emits:  COV L{n} ncov=K : e0 e1 ...
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static void die(const char*m){ fprintf(stderr,"lut_cov: %s\n",m); exit(1); }
static void* xmalloc(size_t n){ void*p=malloc(n); if(!p)die("oom"); return p; }
static uint64_t r_u64(FILE*f){ uint64_t v; if(fread(&v,8,1,f)!=1)die("r u64"); return v; }
static uint32_t r_u32(FILE*f){ uint32_t v; if(fread(&v,4,1,f)!=1)die("r u32"); return v; }
static char* r_str(FILE*f){ uint64_t n=r_u64(f); char*s=xmalloc(n+1); if(n&&fread(s,1,n,f)!=n)die("r str"); s[n]=0; return s; }
enum { GT_U8=0,GT_I8,GT_U16,GT_I16,GT_U32,GT_I32,GT_F32,GT_BOOL,GT_STR,GT_ARR,GT_U64,GT_I64,GT_F64 };
static size_t gt_size(uint32_t t){ switch(t){ case GT_U8:case GT_I8:case GT_BOOL:return 1; case GT_U16:case GT_I16:return 2;
    case GT_U32:case GT_I32:case GT_F32:return 4; case GT_U64:case GT_I64:case GT_F64:return 8; default:return 0; } }
static void skip_kv(FILE*f,uint32_t t){
    if(t==GT_STR){ uint64_t n=r_u64(f); if(fseek(f,(long)n,SEEK_CUR))die("seek"); return; }
    if(t==GT_ARR){ uint32_t et=r_u32(f); uint64_t n=r_u64(f);
        if(et==GT_STR){ for(uint64_t i=0;i<n;i++){ uint64_t l=r_u64(f); if(fseek(f,(long)l,SEEK_CUR))die("seek"); } }
        else { size_t s=gt_size(et); if(!s)die("arr"); if(fseek(f,(long)(s*n),SEEK_CUR))die("seek"); } return; }
    size_t s=gt_size(t); if(!s)die("kv"); if(fseek(f,(long)s,SEEK_CUR))die("seek");
}
#define ALIGN 32ull
static uint64_t pad_up(uint64_t x,uint64_t a){ return (x+a-1)/a*a; }
typedef struct { char name[96]; uint64_t ne[4]; uint32_t type; uint64_t rel; } tinfo;

int main(int argc,char**argv){
    if(argc<2) die("usage: lut_cov FILE.gguf");
    FILE*f=fopen(argv[1],"rb"); if(!f)die("open");
    char magic[4]; if(fread(magic,1,4,f)!=4||memcmp(magic,"GGUF",4))die("bad magic");
    (void)r_u32(f); uint64_t nt=r_u64(f), nkv=r_u64(f);
    for(uint64_t k=0;k<nkv;k++){ char*key=r_str(f); uint32_t vt=r_u32(f); skip_kv(f,vt); free(key); }
    tinfo*T=xmalloc(sizeof(tinfo)*nt);
    for(uint64_t t=0;t<nt;t++){
        char*nm=r_str(f); snprintf(T[t].name,sizeof T[t].name,"%s",nm); free(nm);
        uint32_t nd=r_u32(f); T[t].ne[0]=T[t].ne[1]=T[t].ne[2]=T[t].ne[3]=1;
        for(uint32_t d=0;d<nd&&d<4;d++) T[t].ne[d]=r_u64(f);
        for(uint32_t d=4;d<nd;d++) (void)r_u64(f);
        T[t].type=r_u32(f); T[t].rel=r_u64(f);
    }
    long pos=ftell(f); uint64_t data_start=pad_up((uint64_t)pos,ALIGN);
    for(uint64_t t=0;t<nt;t++){
        if(!strstr(T[t].name,"ffn_res_lut")) continue;
        int L=-1; sscanf(T[t].name,"blk.%d.",&L);
        uint64_t n=T[t].ne[0]*T[t].ne[1];
        float*buf=xmalloc(n*4);
        if(fseek(f,(long)(data_start+T[t].rel),SEEK_SET))die("seek data");
        if(fread(buf,4,n,f)!=n)die("read");
        int ncov=0; for(uint64_t e=0;e<n;e++) if(buf[e]>=0.0f) ncov++;
        printf("COV L%d ncov=%d :", L, ncov);
        for(uint64_t e=0;e<n;e++) if(buf[e]>=0.0f) printf(" %llu",(unsigned long long)e);
        printf("\n");
        free(buf);
    }
    fclose(f); return 0;
}
