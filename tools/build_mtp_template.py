import struct
SRC="gguf/reactgo-prog.gguf"; OUT="gguf/mtp_template.gguf"; ALIGN=32
F32,F16,Q8_0,Q2_K,IQ2=0,1,8,10,16
T_STR=8;T_ARR=9;SZ={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
f=open(SRC,"rb"); assert f.read(4)==b"GGUF"
ver=struct.unpack("<I",f.read(4))[0]; nt=struct.unpack("<Q",f.read(8))[0]; nkv=struct.unpack("<Q",f.read(8))[0]
ru32=lambda:struct.unpack("<I",f.read(4))[0]; ru64=lambda:struct.unpack("<Q",f.read(8))[0]
def skipv(t):
    if t==T_STR: f.seek(ru64(),1)
    elif t==T_ARR:
        et=ru32(); n=ru64()
        if et==T_STR:
            for _ in range(n): f.seek(ru64(),1)
        else: f.seek(n*SZ[et],1)
    else: f.seek(SZ[t],1)
raw=open(SRC,"rb").read()
kv_keep=bytearray(); nkv_keep=0
for _ in range(nkv):
    kstart=f.tell()
    klen=ru64(); key=f.read(klen).decode("utf-8","replace"); skipv(ru32())
    # 剥掉主模型的 expert_keep_map(43层, 层0=hash=256)。MTP 用量化器为它发的自己那份,
    # 否则 runtime 读到主模型的 keep-map → model_expert_kept_count(0)=256 → MTP validate 拒裁过的32。
    if "expert_keep_map" not in key:
        kv_keep+=raw[kstart:f.tell()]; nkv_keep+=1
kv_bytes=bytes(kv_keep)   # 复制 KV(已剔除 keep-map)
# 解析张量 infos + 算每个 size(由 offset 差)
infos=[]; 
for _ in range(nt):
    nm=f.read(ru64()).decode(); nd=ru32(); dims=[ru64() for _ in range(nd)]; ty=ru32(); off=ru64()
    infos.append([nm,dims,ty,off])
data_start=(f.tell()+ALIGN-1)//ALIGN*ALIGN
f.seek(0,2); fsz=f.tell()
order=sorted(range(nt),key=lambda i:infos[i][3])
size={}
for k,i in enumerate(order):
    nxt = data_start+infos[order[k+1]][3] if k+1<len(order) else fsz
    size[i]=nxt-(data_start+infos[i][3])
b0={nm:(dims,ty,size[i]) for i,(nm,dims,ty,off) in enumerate(infos) if nm.startswith("blk.0.") and "tid2eid" not in nm}
# 组装 MTP 32 张量: (name, dims, type, size)
def sz_calc(dims,ty):
    n=1
    for d in dims: n*=d
    return {F32:n*4,F16:n*2,Q8_0:n//32*34}[ty]
mtp=[]
for nm,(dims,ty,s) in b0.items():
    mtp.append((nm.replace("blk.0.","mtp.0.",1),dims,ty,s))
spec=[("mtp.0.exp_probs_b.bias",[256],F32),("mtp.0.e_proj.weight",[4096,4096],Q8_0),
      ("mtp.0.h_proj.weight",[4096,4096],Q8_0),("mtp.0.enorm.weight",[4096],F32),
      ("mtp.0.hnorm.weight",[4096],F32),("mtp.0.norm.weight",[4096],F32),
      ("mtp.0.hc_head_base.weight",[4],F32),("mtp.0.hc_head_fn.weight",[16384,4],F16),
      ("mtp.0.hc_head_scale.weight",[1],F32)]
for nm,dims,ty in spec: mtp.append((nm,dims,ty,sz_calc(dims,ty)))
# 写模板: header + KV + tensor infos(重算 offset) + 稀疏零数据
out=bytearray(); out+=b"GGUF"+struct.pack("<I",ver)+struct.pack("<Q",len(mtp))+struct.pack("<Q",nkv_keep)
out+=kv_bytes
off=0; offs=[]
for nm,dims,ty,s in mtp:
    offs.append(off); off=(off+s+ALIGN-1)//ALIGN*ALIGN
for (nm,dims,ty,s),o in zip(mtp,offs):
    nb=nm.encode(); out+=struct.pack("<Q",len(nb))+nb+struct.pack("<I",len(dims))
    for d in dims: out+=struct.pack("<Q",d)
    out+=struct.pack("<I",ty)+struct.pack("<Q",o)
pad=(ALIGN-(len(out)%ALIGN))%ALIGN; out+=b"\0"*pad
total_data=off
w=open(OUT,"wb"); w.write(out); w.truncate(len(out)+total_data); w.close()
print(f"✓ 模板 {OUT}: {len(mtp)} 张量, header {len(out)}B, 稀疏数据 {total_data/1e9:.2f}GB")
print(f"  专家张量: {[n for n,_,_,_ in mtp if 'exps' in n]}")
print(f"  含 exp_probs_b: {'是' if any('exp_probs_b' in n for n,_,_,_ in mtp) else '否'}, MTP专属: {sum(1 for n,_,_,_ in mtp if any(x in n for x in ['e_proj','h_proj','enorm','hnorm','hc_head',' norm']))}")
