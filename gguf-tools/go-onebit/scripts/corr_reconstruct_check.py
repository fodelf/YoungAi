#!/usr/bin/env python3
# corr_reconstruct_check.py — corr 侧车单层数学门(runbook §3): 提 U/V/C + 真实X, 重建 vs R。
# 用法: corr_reconstruct_check.py corr.gguf L20 /tmp/algo_solve/x_L20.npy /tmp/algo_solve/r_L20.npy
import numpy as np, struct, sys
def parse_gguf(path):
    f=open(path,'rb'); assert f.read(4)==b'GGUF'; struct.unpack('<I',f.read(4))
    nt=struct.unpack('<Q',f.read(8))[0]; nkv=struct.unpack('<Q',f.read(8))[0]
    for _ in range(nkv):
        n=struct.unpack('<Q',f.read(8))[0]; f.read(n); t=struct.unpack('<I',f.read(4))[0]
        assert t==7; f.read(1)
    tens={}
    for _ in range(nt):
        n=struct.unpack('<Q',f.read(8))[0]; name=f.read(n).decode()
        nd=struct.unpack('<I',f.read(4))[0]; ne=[struct.unpack('<Q',f.read(8))[0] for _ in range(nd)]
        struct.unpack('<I',f.read(4)); off=struct.unpack('<Q',f.read(8))[0]; tens[name]=(ne,off)
    return f,tens,(f.tell()+31)//32*32
DM=4096
gguf,L,xp,rp=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
f,tens,doff=parse_gguf(gguf)
def load(nm):
    ne,off=tens[nm]; cnt=1
    for d in ne: cnt*=d
    f.seek(doff+off); return np.frombuffer(f.read(cnt*4),dtype='<f4').copy(),ne
U,neU=load(f'blk.{L[1:]}.corr_U'); V,_=load(f'blk.{L[1:]}.corr_V'); C,_=load(f'blk.{L[1:]}.corr_C')
d_l=neU[0]; U=U.reshape(DM,d_l); V=V.reshape(d_l,DM); C=C.reshape(256,d_l)
X=np.load(xp); R=np.load(rp)
cos_sum=0; mag_sum=0; N=min(64,len(X))
for t in range(N):
    vx=V@X[t]; z=C[0]*6; corr=U@(z*vx)
    cos_sum+=(corr@R[t])/(np.linalg.norm(corr)*np.linalg.norm(R[t])+1e-9)
    mag_sum+=np.abs(corr).mean()/(np.abs(R[t]).mean()+1e-9)
cos=cos_sum/N; mag=mag_sum/N
print(f"{L}: mean cos={cos:.4f} 幅度比={mag:.2f} 门(cos>=0.85)={'PASS' if cos>=0.85 else 'FAIL'}")
