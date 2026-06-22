#!/usr/bin/env python3
"""Report per-layer tensor sizes of a DS4 GGUF (to plan resident-fit trimming)."""
import sys, struct
MAGIC = 0x46554747
T_U8,T_I8,T_U16,T_I16,T_U32,T_I32,T_F32,T_BOOL,T_STR,T_ARR,T_U64,T_I64,T_F64 = range(13)
SCALAR = {T_U8:1,T_I8:1,T_U16:2,T_I16:2,T_U32:4,T_I32:4,T_F32:4,T_BOOL:1,T_U64:8,T_I64:8,T_F64:8}
def ru32(fp): return struct.unpack("<I", fp.read(4))[0]
def ru64(fp): return struct.unpack("<Q", fp.read(8))[0]
def skip(fp,t):
    if t==T_STR: fp.read(ru64(fp))
    elif t==T_ARR:
        et=ru32(fp); n=ru64(fp)
        for _ in range(n): skip(fp,et)
    else: fp.read(SCALAR[t])
fp=open(sys.argv[1],"rb")
assert ru32(fp)==MAGIC
ver=ru32(fp); nt=ru64(fp); nkv=ru64(fp)
align=32
for _ in range(nkv):
    k=fp.read(ru64(fp)); vt=ru32(fp)
    if k==b"general.alignment" and vt==T_U32: align=ru32(fp)
    else: skip(fp,vt)
infos=[]
for _ in range(nt):
    nm=fp.read(ru64(fp)); nd=ru32(fp); dims=[ru64(fp) for _ in range(nd)]; typ=ru32(fp); off=ru64(fp)
    infos.append([nm.decode(),dims,typ,off])
info_end=fp.tell(); data_start=(info_end+align-1)//align*align
fp.seek(0,2); fsz=fp.tell()
order=sorted(range(len(infos)),key=lambda i:infos[i][3])
size={}
for k,i in enumerate(order):
    nxt=data_start+infos[order[k+1]][3] if k+1<len(order) else fsz
    size[i]=nxt-(data_start+infos[i][3])
# group by layer
import collections
L=collections.defaultdict(lambda:{"exps":0,"dense_ffn":0,"attn":0,"norm":0,"other":0})
glob=0
for i,(nm,dims,typ,off) in enumerate(infos):
    s=size[i]
    if nm.startswith("blk."):
        l=int(nm.split(".")[1]); rest=nm.split(".",2)[2]
        if "_exps" in rest: L[l]["exps"]+=s
        elif rest.startswith("ffn_"): L[l]["dense_ffn"]+=s
        elif rest.startswith("attn"): L[l]["attn"]+=s
        elif "norm" in rest: L[l]["norm"]+=s
        else: L[l]["other"]+=s
    else: glob+=s
GIB=1073741824
print(f"total file {fsz/GIB:.2f} GiB | globals(embed/head/etc) {glob/GIB:.2f} GiB")
print(f"{'layer':>5} {'experts':>9} {'denseFFN':>9} {'attn':>7} {'norm':>6}  type")
tot_exp=0
for l in sorted(L):
    d=L[l]; e=d['exps']; tot_exp+=e
    typ = "MoE(experts)" if e>0 else ("DENSE-ffn" if d['dense_ffn']>0 else "?")
    print(f"{l:>5} {e/GIB:>8.3f} {d['dense_ffn']/GIB:>8.3f} {d['attn']/GIB:>6.3f} {d['norm']/GIB:>5.3f}  {typ}")
print(f"\n总专家字节 {tot_exp/GIB:.2f} GiB over {len(L)} layers")
# 按当前切分汇总
def slice_bytes(lo,hi,head):
    b=glob*(1 if head else 0)
    for l in range(lo,hi+1): b+=sum(L[l].values())
    return b/GIB
print(f"当前切分: M4层0-9 含globals={slice_bytes(0,9,True):.2f}G | M1层10-42={slice_bytes(10,42,False):.2f}G")
