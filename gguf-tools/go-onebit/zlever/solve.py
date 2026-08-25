"""zlever/solve.py — 单层活rank/活λ z 求解+held-out 评估(基建正式件, README 记口径)。
设计(用户定案): rank 不钉死 — 每专家出完整 rank-收益曲线, 在层预算(总rank)内按
边际能量收益贪心分配(误差大的专家多吃 rank); λ 也不钉死 — 每专家在小网格内用
内部验证集自选。held-out 25% token 只作终判, 不参与任何选择。
用法: solve.py <hf> <layers_dir> <anchor> <层> <专家数> [预算avg_rank列表 如 "32 48"]
"""
import os, sys, heapq, numpy as np
from collections import Counter
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','scripts'))
from probe_behavior_spectrum import st_index, st_mxfp4, vq_slot, vq_dequant
from probe_layer_behavior import anchor_layer, swiglu
hf,ld,ap,L,NEXP_TEST = sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5])
BUDGETS=[int(b) for b in (sys.argv[6] if len(sys.argv)>6 else "32 48").split()]
KGRID=[8,16,24,32,48,64,96]; LGRID=[3.0,10.0,30.0]; KMAX=KGRID[-1]
wmap=st_index(hf); X0,ridx,rw,NACT=anchor_layer(ap,L,1716)
blob=open(os.path.join(ld,f"dql_vq_L{L:02d}.bin"),"rb").read()
X=X0.astype(np.float64); n=1716; ntr=1287
rng=np.random.RandomState(0); perm=rng.permutation(n)
tr,ho=perm[:ntr],perm[ntr:]
sperm=np.random.RandomState(2).permutation(ntr)
subtr,subval=tr[sperm[:1029]],tr[sperm[1029:]]

def aug(Xt,Rt):
    # dither 增广(smooth+fixed 损失进解), 与量化器同口径 0.04·rms / 权 0.25
    rms=np.sqrt((Xt**2).mean(1,keepdims=True))
    r2=np.random.RandomState(1)
    Xa=np.vstack([Xt,(Xt+r2.randn(*Xt.shape)*0.04*rms)*np.sqrt(0.25)])
    Ra=np.vstack([Rt,Rt*np.sqrt(0.25)])
    return Xa,Ra

def curve(Xe,Rt,U,S,Vt,colw):
    # KGRID 逐点评估(原始未缩放空间), 返回每 k 的剩余能量
    A=Xe@U; B=Vt/colw[None,:]
    P=np.zeros((Xe.shape[0],B.shape[1])); out=[]; j=0
    for k in KGRID:
        while j<k and j<len(S):
            P+=np.outer(A[:,j]*S[j],B[j]); j+=1
        out.append(float(((Rt-P)**2).sum()))
    return np.array(out)

def rsvd(Xa,al,k,q=2,os_=32,seed=3):
    # 隐式 W=Xaᵀal 的随机截断 SVD(不成形 4096×4096 的 W, λ 选择用)
    r=np.random.RandomState(seed)
    Y=Xa.T@(al@r.randn(al.shape[1],k+os_))
    for _ in range(q):
        Y=Xa.T@(al@(al.T@(Xa@Y))); Y,_=np.linalg.qr(Y)
    Q,_=np.linalg.qr(Y)
    Ub,S,Vt=np.linalg.svd((Q.T@Xa.T)@al,full_matrices=False)
    return Q@Ub[:,:k],S[:k],Vt[:k]

sel=list(range(0,256,max(1,256//NEXP_TEST)))[:NEXP_TEST]
xs=X0.astype(np.float32)
EX=[]   # (e, λ*, e_subval, val剩余[k], e_ho, ho剩余[k])
for e in sel:
    P={}; ok=True
    for nm,wh in (("w1",0),("w3",1),("w2",2)):
        off=vq_slot(blob,e,wh)
        if not off: ok=False; break
        Wf=st_mxfp4(hf,wmap,f"layers.{L}.ffn.experts.{e}.{nm}.weight")
        Wq=vq_dequant(blob,off)
        if Wf.shape!=Wq.shape: Wf=Wf.T
        P[nm]=(Wf,Wq)
    if not ok:
        print(f"  e{e:>3}: 缺 vq 槽 跳过", flush=True); continue
    Yf=swiglu(xs@P["w1"][0].T,xs@P["w3"][0].T)@P["w2"][0].T
    Yq=swiglu(xs@P["w1"][1].T,xs@P["w3"][1].T)@P["w2"][1].T
    R=(Yf-Yq)
    colw=np.sqrt(R[tr].var(0)+1e-12)          # classify 列权(方差大维度优先)
    # ── λ 自选: subtr 拟合 / subval 评分(rec@32+rec@48), held-out 不碰
    Xa_s,Ra_s=aug(X[subtr],R[subtr]*colw)
    G0=Xa_s@Xa_s.T; trG=np.trace(G0)/Xa_s.shape[1]
    best=None
    for lam in LGRID:
        G=G0.copy(); G[np.diag_indices_from(G)]+=lam*trG+1e-10
        al=np.linalg.solve(G,Ra_s)
        U,S,Vt=rsvd(Xa_s,al,KMAX)
        res=curve(X[subval],R[subval],U,S,Vt,colw)
        ev=float((R[subval]**2).sum())
        score=(1-res[KGRID.index(32)]/ev)+(1-res[KGRID.index(48)]/ev)
        if best is None or score>best[0]: best=(score,lam,res,ev)
    _,lam,vres,ev=best
    # ── 终评: 全 tr 重拟合(λ*) + 精确 SVD + held-out 曲线
    Xa,Ra=aug(X[tr],R[tr]*colw)
    G=Xa@Xa.T; G[np.diag_indices_from(G)]+=lam*np.trace(G)/Xa.shape[1]+1e-10
    al=np.linalg.solve(G,Ra)
    U,S,Vt=np.linalg.svd(Xa.T@al,full_matrices=False)
    hres=curve(X[ho],R[ho],U[:,:KMAX],S[:KMAX],Vt[:KMAX],colw)
    eho=float((R[ho]**2).sum())
    EX.append((e,lam,ev,vres,eho,hres))
    rc="/".join(f"{(1-hres[i]/eho)*100:.1f}" for i in range(len(KGRID)))
    print(f"  e{e:>3}: λ*={lam:g} ho挽回@{'/'.join(map(str,KGRID))} = {rc}%", flush=True)

def alloc(avg):
    # 层预算内贪心: 每步=专家曲线相邻网格段, 收益=subval 能量增益/rank 成本
    budget=avg*len(EX); nxt=[0]*len(EX); kass=[0]*len(EX); used=0; h=[]
    def push(ei):
        ki=nxt[ei]
        if ki>=len(KGRID): return
        _,_,ev,vres,_,_=EX[ei]
        prev=ev if ki==0 else vres[ki-1]; pk=0 if ki==0 else KGRID[ki-1]
        heapq.heappush(h,(-(prev-vres[ki])/(KGRID[ki]-pk),prev-vres[ki],KGRID[ki]-pk,ei,ki))
    for ei in range(len(EX)): push(ei)
    while h:
        _,gain,cost,ei,ki=heapq.heappop(h)
        if gain<=0 or used+cost>budget: continue
        kass[ei]=KGRID[ki]; used+=cost; nxt[ei]=ki+1; push(ei)
    return kass,used

lc=Counter(ex[1] for ex in EX)
print(f"\nλ* 分布: "+"  ".join(f"λ={k:g}:{v}" for k,v in sorted(lc.items())))
for avg in BUDGETS:
    kass,used=alloc(avg)
    etot=sum(ex[4] for ex in EX)
    rec1=[(ex[4]-(ex[4] if k==0 else ex[5][KGRID.index(k)]))/ex[4] for ex,k in zip(EX,kass)]
    egain=sum(ex[4]-(ex[4] if k==0 else ex[5][KGRID.index(k)]) for ex,k in zip(EX,kass))
    ks=np.array(kass); r1=np.array(rec1)
    print(f"★ L{L} {len(EX)}专家 活rank 预算avg={avg}(总rank {used}/{avg*len(EX)}): "
          f"层能量挽回 {egain/etot*100:.1f}%  专家均值 {r1.mean()*100:.1f}%  最差 {r1.min()*100:.1f}%")
    print(f"  rank分布 min/中位/max = {ks.min()}/{int(np.median(ks))}/{ks.max()}"
          f"  | 43层体积 {avg*8192*2*256*43/1e9:.1f} GB")
    if avg in KGRID:
        fi=KGRID.index(avg)
        fr=np.array([(ex[4]-ex[5][fi])/ex[4] for ex in EX])
        fg=sum(ex[4]-ex[5][fi] for ex in EX)
        print(f"  vs 钉死rank={avg}(λ活): 层能量挽回 {fg/etot*100:.1f}%  专家均值 {fr.mean()*100:.1f}%")
