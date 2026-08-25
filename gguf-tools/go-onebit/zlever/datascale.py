"""zlever/datascale.py — z^L 是否数据受限的诊断(用 zcache, 分钟级)。
固定 held-out, 训练 token 数 ∈ {1/4,1/2,全}, 看 rec@rank 曲线随数据量的走势:
仍在涨=数据墙(加大锚是免体积杠杆); 已平=结构极限(线性肩部真实)。
用法: datascale.py <anchor> <layers_dir> <层>
"""
import os, sys, numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','scripts'))
from probe_layer_behavior import anchor_layer
ap,ld,L = sys.argv[1],sys.argv[2],int(sys.argv[3])
D=4096; NTOK=1716; KS=[256,512,1024]
X0,ridx,rw,NACT=anchor_layer(ap,L,NTOK)
zc=np.load(os.path.join(ld,f"zcache_L{L:02d}.npz")); dH=zc["dH"]
X=X0.astype(np.float64)
perm=np.random.RandomState(0).permutation(NTOK); ntr=int(NTOK*0.75)
tr_full,ho=perm[:ntr],perm[ntr:]
eho=float((dH[ho].astype(np.float64)**2).sum())
for frac in (0.25,0.5,1.0):
    tr=tr_full[:int(ntr*frac)]
    colw=np.sqrt(dH[tr].var(0)+1e-12)
    rms=np.sqrt((X[tr]**2).mean(1,keepdims=True))
    r2=np.random.RandomState(1)
    Xa=np.vstack([X[tr],(X[tr]+r2.randn(len(tr),D)*0.04*rms)*np.sqrt(0.25)])
    Ra=np.vstack([dH[tr]*colw,(dH[tr]*colw)*np.sqrt(0.25)])
    G=Xa@Xa.T; G[np.diag_indices_from(G)]+=3.0*np.trace(G)/Xa.shape[1]+1e-10
    al=np.linalg.solve(G,Ra)
    U,S,Vt=np.linalg.svd(Xa.T@al,full_matrices=False)
    out=[]
    for k in KS:
        kk=min(k,len(S))
        pred=(X[ho]@U[:,:kk]*S[:kk])@Vt[:kk]/np.maximum(colw,1e-12)
        out.append((1-float(((dH[ho]-pred.astype(np.float32)).astype(np.float64)**2).sum())/eho)*100)
    print(f"训练token={len(tr):>4}: rec@{'/'.join(map(str,KS))} = "+"/".join(f"{v:.1f}" for v in out)+"%", flush=True)
