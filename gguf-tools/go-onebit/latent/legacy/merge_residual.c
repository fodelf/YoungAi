/* merge_residual.c — merge N partial go1b residual sidecar GGUFs (each carrying a
 * DISJOINT set of layers, produced by a per-machine cluster run) into one sidecar.
 *
 * Cluster flow: each host runs emit_residual over the layer range whose HF shards
 * it holds locally (fast local reads, no NFS), producing a partial residual GGUF;
 * this tool concatenates them on the coordinator. Tensors are copied verbatim
 * (byte-identical), so the merge is lossless. KVs are rebuilt (present/sparse/
 * n_present/layer.{i}); ds4 residual_load keys off the blk.{L}.* tensor names.
 *
 * Usage: ./merge_residual --out merged.gguf part_a.gguf part_b.gguf [...]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>

#define GGUF_VERSION 3
#define DEFAULT_ALIGN 32
enum { GT_UINT32=4, GT_INT32=5, GT_FLOAT32=6, GT_BOOL=7, GT_STRING=8, GT_ARRAY=9,
       GT_UINT64=10, GT_INT64=11, GT_FLOAT64=12 };
#define GGML_TYPE_F32 0
#define GGML_TYPE_GO1B 40
#define MAXT 4096

static void die(const char *m){ fprintf(stderr,"merge_residual: %s\n",m); exit(1); }
static void *xmalloc(size_t n){ void*p=malloc(n?n:1); if(!p)die("OOM"); return p; }

/* ---- writers (mirror emit_residual) ---- */
static void w_u32(FILE*f,uint32_t v){ if(fwrite(&v,4,1,f)!=1)die("w u32"); }
static void w_u64(FILE*f,uint64_t v){ if(fwrite(&v,8,1,f)!=1)die("w u64"); }
static void w_str(FILE*f,const char*s){ uint64_t n=strlen(s); w_u64(f,n); if(n&&fwrite(s,1,n,f)!=n)die("w str"); }
static void w_kv_bool(FILE*f,const char*k,bool v){ w_str(f,k); w_u32(f,GT_BOOL); uint8_t b=v?1:0; if(fwrite(&b,1,1,f)!=1)die("w bool"); }
static void w_kv_i32(FILE*f,const char*k,int32_t v){ w_str(f,k); w_u32(f,GT_INT32); w_u32(f,(uint32_t)v); }
static void w_kv_str(FILE*f,const char*k,const char*v){ w_str(f,k); w_u32(f,GT_STRING); w_str(f,v); }
static size_t pad_up(size_t n,size_t a){ return (n+a-1)/a*a; }
static void wpad(FILE*f,size_t n){ static const uint8_t z[64]={0}; while(n){ size_t c=n<sizeof z?n:sizeof z; if(fwrite(z,1,c,f)!=c)die("w pad"); n-=c; } }

/* ---- reader ---- */
static uint64_t r_u64(FILE*f){ uint64_t v; if(fread(&v,8,1,f)!=1)die("r u64"); return v; }
static uint32_t r_u32(FILE*f){ uint32_t v; if(fread(&v,4,1,f)!=1)die("r u32"); return v; }
static char* r_str(FILE*f){ uint64_t n=r_u64(f); char*s=xmalloc(n+1); if(n&&fread(s,1,n,f)!=n)die("r str"); s[n]=0; return s; }

static void skip_kv_value(FILE*f, uint32_t type){
    switch(type){
        case GT_STRING:{ uint64_t n=r_u64(f); if(fseek(f,(long)n,SEEK_CUR))die("seek"); break; }
        case GT_BOOL: if(fseek(f,1,SEEK_CUR))die("seek"); break;
        case GT_UINT32: case GT_INT32: case GT_FLOAT32: if(fseek(f,4,SEEK_CUR))die("seek"); break;
        case GT_UINT64: case GT_INT64: case GT_FLOAT64: if(fseek(f,8,SEEK_CUR))die("seek"); break;
        default: die("unsupported KV type (arrays not expected in residual sidecar)");
    }
}

static uint64_t tensor_bytes(uint32_t type, const uint64_t ne[3]){
    if(type==GGML_TYPE_GO1B) return ne[2]*ne[1]*((ne[0]+255)/256)*34ull;
    if(type==GGML_TYPE_F32)  return ne[0]*ne[1]*ne[2]*4ull;
    die("unexpected tensor type"); return 0;
}

typedef struct { char name[80]; uint64_t ne[3]; uint32_t type; uint64_t size;
                 char src[512]; uint64_t src_abs_off; uint64_t out_off; } mtensor;

int main(int argc,char**argv){
    const char *out_path=NULL; const char *inputs[64]; int nin=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--out")&&i+1<argc) out_path=argv[++i];
        else if(argv[i][0]!='-'&&nin<64) inputs[nin++]=argv[i];
        else die("bad arg");
    }
    if(!out_path||nin<1) die("usage: --out FILE part1.gguf part2.gguf ...");

    mtensor *T=xmalloc(sizeof(mtensor)*MAXT); int nt=0;
    int sparse=0; int layers_present[512]; int nlayers=0;

    for(int fi=0; fi<nin; fi++){
        FILE*f=fopen(inputs[fi],"rb"); if(!f)die("open input");
        char magic[4]; if(fread(magic,1,4,f)!=4||memcmp(magic,"GGUF",4))die("bad magic");
        (void)r_u32(f);                 /* version */
        uint64_t n_tensors=r_u64(f), n_kv=r_u64(f);
        for(uint64_t k=0;k<n_kv;k++){
            char*key=r_str(f); uint32_t vt=r_u32(f);
            if(!strcmp(key,"ds4.residual.sparse")){ uint8_t b; if(fread(&b,1,1,f)!=1)die("r"); sparse|=b; }
            else skip_kv_value(f,vt);
            free(key);
        }
        /* tensor metadata */
        int base=nt;
        for(uint64_t t=0;t<n_tensors;t++){
            if(nt>=MAXT)die("too many tensors");
            mtensor*mt=&T[nt];
            char*nm=r_str(f); snprintf(mt->name,sizeof mt->name,"%s",nm); free(nm);
            uint32_t nd=r_u32(f); mt->ne[0]=mt->ne[1]=mt->ne[2]=1;
            for(uint32_t d=0;d<nd&&d<3;d++) mt->ne[d]=r_u64(f);
            for(uint32_t d=3;d<nd;d++) (void)r_u64(f);
            mt->type=r_u32(f); mt->size=tensor_bytes(mt->type,mt->ne);
            uint64_t rel=r_u64(f);
            snprintf(mt->src,sizeof mt->src,"%s",inputs[fi]);
            mt->src_abs_off=rel;   /* fixed up to absolute below */
            /* collect layer number from blk.{L}. */
            int L; if(sscanf(mt->name,"blk.%d.",&L)==1){
                int seen=0; for(int j=0;j<nlayers;j++) if(layers_present[j]==L){seen=1;break;}
                if(!seen&&nlayers<512) layers_present[nlayers++]=L;
            }
            nt++;
        }
        long pos=ftell(f); if(pos<0)die("ftell");
        uint64_t data_start=pad_up((size_t)pos,DEFAULT_ALIGN);
        for(int t=base;t<nt;t++) T[t].src_abs_off += data_start;   /* rel -> absolute */
        fclose(f);
    }

    /* assign output offsets */
    uint64_t roff=0; for(int i=0;i<nt;i++){ T[i].out_off=roff; roff+=pad_up(T[i].size,DEFAULT_ALIGN); }

    FILE*f=fopen(out_path,"wb"); if(!f)die("open out");
    if(fwrite("GGUF",1,4,f)!=4)die("magic");
    w_u32(f,GGUF_VERSION);
    w_u64(f,(uint64_t)nt);
    w_u64(f,(uint64_t)(4+nlayers));
    w_kv_str(f,"general.architecture","ds4-residual");
    w_kv_bool(f,"ds4.residual.present",true);
    w_kv_bool(f,"ds4.residual.sparse",sparse?true:false);
    w_kv_i32(f,"ds4.residual.n_present",nlayers);
    for(int i=0;i<nlayers;i++){ char key[48]; snprintf(key,sizeof key,"ds4.residual.layer.%d",i); w_kv_i32(f,key,layers_present[i]); }
    for(int i=0;i<nt;i++){
        w_str(f,T[i].name); w_u32(f,3);
        for(int j=0;j<3;j++) w_u64(f,T[i].ne[j]);
        w_u32(f,T[i].type); w_u64(f,T[i].out_off);
    }
    long pos=ftell(f); if(pos<0)die("ftell");
    uint64_t data_off=pad_up((size_t)pos,DEFAULT_ALIGN);
    wpad(f,data_off-(size_t)pos);
    unsigned char *cp=xmalloc(1<<20);
    for(int i=0;i<nt;i++){
        FILE*sf=fopen(T[i].src,"rb"); if(!sf)die("open src");
        if(fseek(sf,(long)T[i].src_abs_off,SEEK_SET))die("seek src");
        uint64_t left=T[i].size;
        while(left){ size_t c=left<(1<<20)?left:(1<<20); if(fread(cp,1,c,sf)!=c)die("read src"); if(fwrite(cp,1,c,f)!=c)die("write"); left-=c; }
        fclose(sf);
        wpad(f,pad_up(T[i].size,DEFAULT_ALIGN)-T[i].size);
    }
    free(cp);
    if(fclose(f))die("close");
    double g=0; for(int i=0;i<nt;i++) g+=T[i].size; g/=1073741824.0;
    fprintf(stderr,"merge_residual: wrote %s  inputs=%d  tensors=%d  layers=%d  sparse=%d  %.2f GiB\n",
            out_path,nin,nt,nlayers,sparse,g);
    free(T);
    return 0;
}
