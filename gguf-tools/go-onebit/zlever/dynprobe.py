"""zlever/dynprobe.py — 动态 z 判决探针(闭式, 位置切分协议)。
问题: 冻结 z^L 吃剩的残差里有多少是 x/位置的函数(动态可抓)?
模型: 修正 = Σ_g (1+δ_g(x))·pred_g(x), 方向冻结, 只解门 δ_g(x)=θ_gᵀ[1,φ(x)] — 线性闭式。
特征 φ: A=x 在感知基前8方向的投影; B=+log1p(pos)+rms(x)(上下文条件)。
另测: 长上下文分桶 z^L(解只用 pos 640..1287)vs 全段解。
评测: 一律 held=位置1287..1716(解算未见, 上下文真实)。
用法: dynprobe.py <anchor> <layers_dir> <层> [G=16]
"""
import os, sys, numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','scripts'))
from probe_layer_behavior import anchor_layer
ap,ld,L = sys.argv[1],sys.argv[2],int(sys.argv[3])
G=int(sys.argv[4]) if len(sys.argv)>4 else 16
D=4096; NTOK=1716; K=1024
X0,ridx,rw,NACT=anchor_layer(ap,L,NTOK)
dH=np.load(os.path.join(ld,f"zcache_L{L:02d}.npz"))["dH"]
X=X0.astype(np.float64)
tr=np.arange(0,1287); ev=np.arange(1287,NTOK)
btr,bval=np.arange(0,1029),np.arange(1029,1287)

def solve_frozen(rows,lam=3.0):
    colw=np.sqrt(dH[rows].var(0)+1e-12)
    rms=np.sqrt((X[rows]**2).mean(1,keepdims=True))
    r2=np.random.RandomState(1)
    Xa=np.vstack([X[rows],(X[rows]+r2.randn(len(rows),D)*0.04*rms)*np.sqrt(0.25)])
    Ra=np.vstack([dH[rows]*colw,(dH[rows]*colw)*np.sqrt(0.25)])
    Gm=Xa@Xa.T; Gm[np.diag_indices_from(Gm)]+=lam*np.trace(Gm)/Xa.shape[1]+1e-10
    al=np.linalg.solve(Gm,Ra)
    A,S,Bt=np.linalg.svd(Xa.T@al,full_matrices=False)
    return A[:,:K],S[:K],Bt[:K],colw

eev=float((dH[ev].astype(np.float64)**2).sum())
def rec(pred_ev): return (1-float(((dH[ev]-pred_ev).astype(np.float64)**2).sum())/eev)*100

A,S,Bt,colw=solve_frozen(tr)
F=(X@(A*S)).astype(np.float32)                       # [n,K] 方向系数
predF=(F@Bt.astype(np.float32))/np.maximum(colw,1e-12)
base=rec(predF[ev])
print(f"L{L}: 冻结 z^L 基线(位置held) = {base:.1f}%", flush=True)
# 分组部分预测 P3[t,g,:](原始空间)
gs=K//G
P3=np.zeros((NTOK,G,D),dtype=np.float32)
for g in range(G):
    sl=slice(g*gs,(g+1)*gs)
    P3[:,g,:]=(F[:,sl]@Bt[sl].astype(np.float32))/np.maximum(colw,1e-12)
R2=dH-predF.astype(np.float32)
# 特征
Pdir=A[:,:8].astype(np.float32)                      # 感知基前8方向(输入侧)
phiA=(X.astype(np.float32)@Pdir)                     # [n,8]
pos=np.log1p(np.arange(NTOK,dtype=np.float32))[:,None]
xrms=np.sqrt((X0**2).mean(1,keepdims=True)).astype(np.float32)
def gate_fit(feats,rows_fit,rows_val):
    # 标准化(fit 行), δ_g(x)=θ_gᵀ[1,φ]; 目标 min‖R2 − Σ_g δ_g·P3_g‖²(colw加权空间)
    mu=feats[rows_fit].mean(0); sd=feats[rows_fit].std(0)+1e-6
    Phi=np.hstack([np.ones((NTOK,1),dtype=np.float32),(feats-mu)/sd])   # [n,F]
    NF=Phi.shape[1]; NPAR=G*NF
    cw=colw.astype(np.float32)
    Gm=np.zeros((NPAR,NPAR)); bv=np.zeros(NPAR)
    for t in rows_fit:
        Pg=P3[t]*cw[None,:]                           # [G,D] scaled
        Sg=(Pg@Pg.T).astype(np.float64)               # [G,G]
        f=Phi[t].astype(np.float64)                   # [F]
        Gm+=np.kron(Sg,np.outer(f,f))
        r=(R2[t]*cw).astype(np.float64)
        bv+=np.kron(Pg@r,f)
    best=None
    for rr in (1e-3,1e-2,1e-1):
        Gm2=Gm.copy(); Gm2[np.diag_indices_from(Gm2)]+=rr*np.trace(Gm2)/NPAR+1e-10
        th=np.linalg.solve(Gm2,bv).reshape(G,NF)
        dlt=Phi@th.T.astype(np.float32)               # [n,G]
        predD=np.einsum('ng,ngd->nd',dlt,P3)
        rv=1-float(((R2[rows_val]-predD[rows_val]).astype(np.float64)**2).sum())/float((R2[rows_val].astype(np.float64)**2).sum())
        if best is None or rv>best[0]: best=(rv,rr,th)
    _,rr,th=best
    dlt=Phi@th.T.astype(np.float32)
    predD=np.einsum('ng,ngd->nd',dlt,P3)
    return predD,rr
predA,rrA=gate_fit(phiA,btr,bval)
print(f"L{L}: +动态门(x投影8维, 岭={rrA:g}) = {rec((predF+predA)[ev]):.1f}%", flush=True)
featsB=np.hstack([phiA,pos,xrms])
predB,rrB=gate_fit(featsB,btr,bval)
print(f"L{L}: +动态门+位置/尺度特征(岭={rrB:g}) = {rec((predF+predB)[ev]):.1f}%", flush=True)
# 长上下文分桶: 解只用 pos 640..1287
A2,S2,Bt2,colw2=solve_frozen(np.arange(640,1287))
predL=((X@(A2*S2)).astype(np.float32)@Bt2.astype(np.float32))/np.maximum(colw2,1e-12)
print(f"L{L}: 长上下文分桶 z^L(解640..1287) = {rec(predL[ev]):.1f}%  [对照全段解 {base:.1f}%]", flush=True)
