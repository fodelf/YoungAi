"""zlever/layerz.py — 正典层级 z^L 测试(引擎 ds4_z 语义, 三段式产物③)。
口径 = 引擎挂载点原样: routed += U·diag(z)·Vᵀ·x, 每层一份, k_L 可调。
目标 = 整层 routed MoE 输出残差 ΔH(教师路由: fp/q 同 ridx/rw, 不含路由漂移),
求解 = zlever 四损失纪律(dither 增广 + classify 列权 + 对偶 ridge, λ 内部验证自选),
判决 = held-out 25% token 的层误差能量挽回, rank 曲线到 4096 全秩。
体积 = k×8192×2B/层(fp16 U/V), 43 层全模型; 全秩也只 2.9GB — 无 256 专家倍乘。
用法: layerz.py <hf> <layers_dir> <anchor> <层> [ntok=1716]
"""
import os, sys, numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','scripts'))
from probe_behavior_spectrum import st_index
from probe_layer_z_validate import build_dH
hf,ld,ap,L = sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
NTOK=int(sys.argv[5]) if len(sys.argv)>5 else 1716
KGRID=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096]; LGRID=[3.0,10.0,30.0]
wmap=st_index(hf)
print(f"L{L}: 建 ΔH(教师路由, {NTOK} token)…", flush=True)
X0,dH,Hfp=build_dH(hf,wmap,ld,ap,L,NTOK)
eh=float((dH**2).sum()); ef=float((Hfp**2).sum())
print(f"  ΔH 能量/层输出能量 = {eh/ef*100:.1f}%(层量化误差占比)", flush=True)
X=X0.astype(np.float64); n=NTOK; ntr=int(n*0.75)
perm=np.random.RandomState(0).permutation(n); tr,ho=perm[:ntr],perm[ntr:]
sperm=np.random.RandomState(2).permutation(ntr)
cut=int(ntr*0.8); subtr,subval=tr[sperm[:cut]],tr[sperm[cut:]]

def aug(Xt,Rt):
    rms=np.sqrt((Xt**2).mean(1,keepdims=True))
    r2=np.random.RandomState(1)
    Xa=np.vstack([Xt,(Xt+r2.randn(*Xt.shape)*0.04*rms)*np.sqrt(0.25)])
    Ra=np.vstack([Rt,Rt*np.sqrt(0.25)])
    return Xa,Ra

def solve(Xt,Rt,lam):
    Xa,Ra=aug(Xt,Rt)
    G=Xa@Xa.T; G[np.diag_indices_from(G)]+=lam*np.trace(G)/Xa.shape[1]+1e-10
    al=np.linalg.solve(G,Ra)
    return np.linalg.svd(Xa.T@al,full_matrices=False)

def curve(Xe,Rt,U,S,Vt,colw):
    A=Xe@U; B=Vt/colw[None,:]
    P=np.zeros((Xe.shape[0],B.shape[1])); out=[]; j=0
    for k in KGRID:
        while j<k and j<len(S):
            P+=np.outer(A[:,j]*S[j],B[j]); j+=1
        out.append(float(((Rt-P)**2).sum()))
    return np.array(out)

colw=np.sqrt(dH[tr].var(0)+1e-12)
best=None
for lam in LGRID:
    U,S,Vt=solve(X[subtr],dH[subtr]*colw,lam)
    res=curve(X[subval],dH[subval],U,S,Vt,colw)
    ev=float((dH[subval]**2).sum())
    sc=(1-res[KGRID.index(64)]/ev)+(1-res[KGRID.index(256)]/ev)
    print(f"  λ={lam:g}: subval 挽回@64/256 = {(1-res[KGRID.index(64)]/ev)*100:.1f}/{(1-res[KGRID.index(256)]/ev)*100:.1f}%", flush=True)
    if best is None or sc>best[0]: best=(sc,lam)
lam=best[1]
U,S,Vt=solve(X[tr],dH[tr]*colw,lam)
hres=curve(X[ho],dH[ho],U,S,Vt,colw)
eho=float((dH[ho]**2).sum())
print(f"\n★ L{L} 层级 z^L(教师路由, λ*={lam:g}) held-out 挽回曲线:")
for i,k in enumerate(KGRID):
    print(f"  rank={k:>4}: {(1-hres[i]/eho)*100:5.1f}%   43层体积 {k*8192*2*43/1e9:6.2f} GB")
