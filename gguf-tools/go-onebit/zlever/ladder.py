"""zlever/ladder.py — L00 冲 100% 的恢复阶梯(全部用户架构组件, 闭式解)。
阶梯: A=z^L(产物③, rank1024) → B=+corr(产物①: out+=Σ_sel U(C[e]⊙Vx), 共享UV+每专家C)
      → C=+GE/β(zchain type5: 每专家门控标量 g_e·w·Yq + β_e)。
每级 held-out 计边际挽回, 判据=层误差能量挽回率, 教师路由。
用法: ladder.py <hf> <layers_dir> <anchor> <层> [ntok=1716] [d_l=512]
"""
import os, sys, numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','scripts'))
from probe_behavior_spectrum import st_index, st_mxfp4, vq_slot, vq_dequant
from probe_layer_behavior import anchor_layer, swiglu
hf,ld,ap,L = sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
NTOK=int(sys.argv[5]) if len(sys.argv)>5 else 1716
DL=int(sys.argv[6]) if len(sys.argv)>6 else 512
D=4096; KL=1024
wmap=st_index(hf)
X0,ridx,rw,NACT=anchor_layer(ap,L,NTOK)
blob=open(os.path.join(ld,f"dql_vq_L{L:02d}.bin"),"rb").read()
cache=os.path.join(ld,f"zcache_L{L:02d}.npz")
if os.path.exists(cache):
    print(f"L{L}: 载对级缓存 {cache}", flush=True)
    zc=np.load(cache)
    dH=zc["dH"]; prow=zc["prow"]; pe=zc["pe"]; pw=zc["pw"]; pYQ=zc["pYQ"]; pDY=zc["pDY"]
else:
    print(f"L{L}: 建 ΔH+对级缓存(教师路由, {NTOK} token)…", flush=True)
    need=sorted(set(int(e) for e in ridx.reshape(-1)))
    dH=np.zeros((NTOK,D),dtype=np.float32)
    prow=[]; pe=[]; pw=[]; pYQ=[]; pDY=[]
    for i,e in enumerate(need):
        rows,slots=np.where(ridx==e)
        if len(rows)==0: continue
        xs=X0[rows].astype(np.float32); w=rw[rows,slots].astype(np.float32)
        P={}
        for nm,wh in (("w1",0),("w3",1),("w2",2)):
            Wf=st_mxfp4(hf,wmap,f"layers.{L}.ffn.experts.{e}.{nm}.weight")
            off=vq_slot(blob,e,wh); Wq=vq_dequant(blob,off) if off else None
            if Wq is not None and Wf.shape!=Wq.shape: Wf=Wf.T
            P[nm]=(Wf,Wq if Wq is not None else Wf)
        Yf=swiglu(xs@P["w1"][0].T,xs@P["w3"][0].T)@P["w2"][0].T
        Yq=swiglu(xs@P["w1"][1].T,xs@P["w3"][1].T)@P["w2"][1].T
        dH[rows]+=w[:,None]*(Yf-Yq)
        prow.append(rows); pe.append(np.full(len(rows),e,dtype=np.int32))
        pw.append(w); pYQ.append(Yq.astype(np.float32)); pDY.append((Yf-Yq).astype(np.float32))
        if (i+1)%64==0: print(f"  …{i+1}/{len(need)} 专家", flush=True)
    prow=np.concatenate(prow); pe=np.concatenate(pe); pw=np.concatenate(pw)
    pYQ=np.vstack(pYQ); pDY=np.vstack(pDY)
    np.savez(cache,dH=dH,prow=prow,pe=pe,pw=pw,pYQ=pYQ,pDY=pDY)
X=X0.astype(np.float64)
perm=np.random.RandomState(0).permutation(NTOK); ntr=int(NTOK*0.75)
tr,ho=perm[:ntr],perm[ntr:]
eho=float((dH[ho].astype(np.float64)**2).sum())
print(f"  对级 {len(prow)} 条  held-out 层误差能量={eho:.3e}", flush=True)

def rec(Rho): return (1-float((Rho.astype(np.float64)**2).sum())/eho)*100

# ── A: z^L rank1024(λ=3, 四损失纪律)
lam=3.0
colw=np.sqrt(dH[tr].var(0)+1e-12)
rms=np.sqrt((X[tr]**2).mean(1,keepdims=True))
r2=np.random.RandomState(1)
Xa=np.vstack([X[tr],(X[tr]+r2.randn(ntr,D)*0.04*rms)*np.sqrt(0.25)])
Ra=np.vstack([dH[tr]*colw,(dH[tr]*colw)*np.sqrt(0.25)])
G=Xa@Xa.T; G[np.diag_indices_from(G)]+=lam*np.trace(G)/Xa.shape[1]+1e-10
al=np.linalg.solve(G,Ra)
Uw,Sw,Vtw=np.linalg.svd(Xa.T@al,full_matrices=False)
predL=(X@Uw[:,:KL]*Sw[:KL])@Vtw[:KL]/np.maximum(colw,1e-12)
R2=dH-predL.astype(np.float32)
print(f"\n★ 阶梯A z^L(rank{KL}): held-out 挽回 {rec(R2[ho]):.1f}%  [~16.8MB/层]", flush=True)

# ── B: +corr(共享 U_out/V + 每专家 C[e], 闭式两步; 岭强度内部验证自选)
f=(X@Uw[:,:DL]).astype(np.float32)                      # V·x 特征(层解的输入方向)
colw2=np.sqrt(R2[tr].var(0)+1e-12)
R2s=R2*colw2
_,_,Vt2=np.linalg.svd(R2s[tr],full_matrices=False)       # 输出方向(残差PCA)
Uo=Vt2[:DL]                                              # [DL,D]
P2=R2s@Uo.T                                              # [n,DL] 投影目标
M=np.zeros((NTOK,256),dtype=np.float32)
for s in range(NACT): M[np.arange(NTOK),ridx[:,s]]=1.0
sperm=np.random.RandomState(2).permutation(ntr)
cut=int(ntr*0.8); btr,bval=tr[sperm[:cut]],tr[sperm[cut:]]
def solve_C(rows,ridge_rel):
    Mr=M[rows]; Cm=np.zeros((256,DL),dtype=np.float64)
    for i in range(DL):
        A=Mr*f[rows,i][:,None]
        Gc=A.T@A; r_=ridge_rel*np.trace(Gc)/256+1e-10
        Gc[np.diag_indices_from(Gc)]+=r_
        Cm[:,i]=np.linalg.solve(Gc,A.T@P2[rows,i])
    return Cm
ebv=float((R2[bval].astype(np.float64)**2).sum())
bestB=None
for rr in (0.1,0.3,1.0,3.0,10.0):
    Cm=solve_C(btr,rr)
    pv=((f[bval]*(M[bval]@Cm))@Uo)/np.maximum(colw2,1e-12)
    g=(1-float(((R2[bval]-pv.astype(np.float32)).astype(np.float64)**2).sum())/ebv)*100
    print(f"  corr岭={rr:g}: subval 边际 {g:+.1f}%", flush=True)
    if bestB is None or g>bestB[0]: bestB=(g,rr)
C=solve_C(tr,bestB[1])
S=M@C                                                    # [n,DL] Σ_sel C[e]
predC=((f*S)@Uo)/np.maximum(colw2,1e-12)
R3=R2-predC.astype(np.float32)
if bestB[0]<=0.0:
    R3=R2.copy()
    print(f"  corr 边际≤0(岭扫尽), 本级跳过", flush=True)
print(f"★ 阶梯B +corr(d_l={DL}, 岭={bestB[1]:g}): held-out 挽回 {rec(R3[ho]):.1f}%  [corr ~{(2*D*DL+256*DL)*2/2**20:.1f}MB/层]", flush=True)

# ── C: +GE/β(每专家 g_e·(w·Yq) + β_e·1, 联合 ridge; 岭强度内部验证自选)
NP=512
tok_pairs=[[] for _ in range(NTOK)]
for p in range(len(prow)): tok_pairs[prow[p]].append(p)
def build_ge(rows):
    Gg=np.zeros((NP,NP)); bg=np.zeros(NP)
    for t in rows:
        ps=tok_pairs[t]
        if not ps: continue
        vecs=[(int(pe[p]),pw[p]*pYQ[p]) for p in ps]
        r3t=R3[t].astype(np.float64)
        for a_,(ea,va) in enumerate(vecs):
            bg[ea]+=float(va@r3t); bg[256+ea]+=float(r3t.sum())
            for eb,vb in [vecs[b_] for b_ in range(a_,len(vecs))]:
                d=float(va@vb); Gg[ea,eb]+=d
                if ea!=eb: Gg[eb,ea]+=d
            sa=float(va.sum())
            for eb,vb in vecs:
                Gg[ea,256+eb]+=sa; Gg[256+eb,ea]+=sa
            for eb,_ in vecs:
                Gg[256+ea,256+eb]+=float(D)
    return Gg,bg
def ge_pred_resid(rows,g,beta):
    tot=0.0
    for t in rows:
        r=R3[t].astype(np.float64).copy()
        for p in tok_pairs[t]:
            r-=g[pe[p]]*(pw[p]*pYQ[p]).astype(np.float64)+beta[pe[p]]
        tot+=float((r**2).sum())
    return tot
Gv,bv=build_ge(btr)
ebv3=float((R3[bval].astype(np.float64)**2).sum())
bestC=None
for rr in (1e-4,1e-3,1e-2,1e-1):
    Gr=Gv.copy(); r_=rr*max(np.trace(Gr)/NP,1.0)
    for d_ in range(NP): Gr[d_,d_]+=r_
    gb=np.linalg.solve(Gr,bv); gg=gb[:256]; bb=gb[256:]
    gain=(1-ge_pred_resid(bval,gg,bb)/ebv3)*100
    print(f"  GE岭={rr:g}: subval 边际 {gain:+.1f}%", flush=True)
    if bestC is None or gain>bestC[0]: bestC=(gain,rr)
if bestC[0]>0.0:
    Gg,bg=build_ge(tr)
    r_=bestC[1]*max(np.trace(Gg)/NP,1.0)
    for d_ in range(NP): Gg[d_,d_]+=r_
    gb=np.linalg.solve(Gg,bg); g=gb[:256]; beta=gb[256:]
    R4=R3.copy()
    for t in range(NTOK):
        for p in tok_pairs[t]:
            R4[t]-=np.float32(g[pe[p]])*pw[p]*pYQ[p]+np.float32(beta[pe[p]])
else:
    R4=R3; print("  GE 边际≤0(岭扫尽), 本级跳过", flush=True)
print(f"★ 阶梯C +GE/β(岭={bestC[1]:g}): held-out 挽回 {rec(R4[ho]):.1f}%  [~2KB/层]", flush=True)
volA=KL*2*D*2/2**20; volB=(2*D*DL+256*DL)*2/2**20
print(f"\n总体积 ≈ {(volA+volB):.0f}MB/层 → 43层 {(volA+volB)*43/1024:.2f} GB", flush=True)
