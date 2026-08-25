"""zlever/wproj.py — 量化语义的全还原算法测试(零训练/零回归)。
算法: 感知基 Q = 激活协方差特征方向(共享/层); 每专家 z_e = 真实权重残差
ΔW=Wf−Wq 在 Q 上的投影系数(闭式, 存的是真权重分量 → 无过拟合概念);
运行时 = dequant 后打权重补丁 W_q + Δ_m, 专家前向原样(SwiGLU 精确)。
判据: held-out token 上 patched 前向 vs fp 前向的误差能量挽回(对 q2 基线)。
用法: wproj.py <hf> <layers_dir> <anchor> <层> <专家数> [m列表 "64 128 256 512"]
"""
import os, sys, numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','scripts'))
from probe_behavior_spectrum import st_index, st_mxfp4, vq_slot, vq_dequant
from probe_layer_behavior import anchor_layer, swiglu
hf,ld,ap,L,NE = sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5])
MS=[int(v) for v in (sys.argv[6] if len(sys.argv)>6 else "64 128 256 512 1024").split()]
D=4096; NTOK=1716
wmap=st_index(hf)
X0,ridx,rw,NACT=anchor_layer(ap,L,NTOK)
blob=open(os.path.join(ld,f"dql_vq_L{L:02d}.bin"),"rb").read()
perm=np.random.RandomState(0).permutation(NTOK); ntr=int(NTOK*0.75)
tr,ho=perm[:ntr],perm[ntr:]
Xtr=X0[tr].astype(np.float64); Xho=X0[ho].astype(np.float32)
# 感知基(输入侧, 共享/层): 激活协方差特征方向, 只用训练 token
S1=Xtr.T@Xtr
ew,Q=np.linalg.eigh(S1)            # 升序
Q=Q[:,::-1].astype(np.float32)     # 降序主方向 [D,D]
sel=list(range(0,256,max(1,256//NE)))[:NE]
# 隐藏侧基: 用探针专家的 swiglu 隐层激活池(训练 token)
print(f"L{L}: 感知基就绪, 建隐藏侧基(池化 {len(sel)} 专家)…", flush=True)
Hpool=[]
cacheP={}
for e in sel:
    P={}
    for nm,wh in (("w1",0),("w3",1),("w2",2)):
        off=vq_slot(blob,e,wh)
        Wf=st_mxfp4(hf,wmap,f"layers.{L}.ffn.experts.{e}.{nm}.weight")
        Wq=vq_dequant(blob,off)
        if Wf.shape!=Wq.shape: Wf=Wf.T
        P[nm]=(Wf.astype(np.float32),Wq.astype(np.float32))
    cacheP[e]=P
    xs=X0[tr[:400]].astype(np.float32)
    Hpool.append(swiglu(xs@P["w1"][0].T,xs@P["w3"][0].T))
H=np.vstack(Hpool).astype(np.float64)
S2=H.T@H
_,Q2=np.linalg.eigh(S2)
Q2=Q2[:,::-1].astype(np.float32)   # [2048,2048]
res={m:[] for m in MS}
for e in sel:
    P=cacheP[e]
    D1=P["w1"][0]-P["w1"][1]; D3=P["w3"][0]-P["w3"][1]; D2=P["w2"][0]-P["w2"][1]
    Yf=swiglu(Xho@P["w1"][0].T,Xho@P["w3"][0].T)@P["w2"][0].T
    Yq=swiglu(Xho@P["w1"][1].T,Xho@P["w3"][1].T)@P["w2"][1].T
    e0=float(((Yf-Yq).astype(np.float64)**2).sum())
    for m in MS:
        Qm=Q[:,:m]; Q2m=Q2[:,:min(m,2048)]
        W1p=P["w1"][1]+ (D1@Qm)@Qm.T
        W3p=P["w3"][1]+ (D3@Qm)@Qm.T
        W2p=P["w2"][1]+ (D2@Q2m)@Q2m.T
        Yp=swiglu(Xho@W1p.T,Xho@W3p.T)@W2p.T
        r=1-float(((Yf-Yp).astype(np.float64)**2).sum())/e0
        res[m].append(r)
    print(f"  e{e:>3}: "+"  ".join(f"m={m}:{res[m][-1]*100:5.1f}%" for m in MS), flush=True)
print(f"\n★ L{L} 权重残差·感知基投影(零训练, held-out 前向, {len(sel)} 专家):")
for m in MS:
    a=np.array(res[m]); m2=min(m,2048)
    volE=(2*m*2048+m2*4096)*2/2**20
    print(f"  m={m:>4}: 挽回均值 {a.mean()*100:5.1f}%  最差 {a.min()*100:5.1f}%"
          f"  | 系数体积 {volE:.2f}MB/专家 fp16 → 层 {volE*256/1024:.1f}GB", flush=True)
