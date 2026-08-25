"""zlever/full10.py — 10G 预算内组合定型测试(L00 全 256 专家)。
组合 = 权重残差·感知基投影侧车(q4 系数 + 每专家 λ·能量自适应选向, ~208MB/层)
     + 行为栈 z^L+corr+GE(~24MB/层), 共 ≈232MB/层 → 43 层 ≈10GB。
全闭式零训练; 判据 = held-out 层误差能量挽回(vs 原始 q2 基线 ΔH)。
用法: full10.py <hf> <layers_dir> <anchor> <层> [每专家系数预算KB=812]
"""
import os, sys, numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','scripts'))
from probe_behavior_spectrum import st_index, st_mxfp4, vq_slot, vq_dequant
from probe_layer_behavior import anchor_layer, swiglu
hf,ld,ap,L = sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
KB=float(sys.argv[5]) if len(sys.argv)>5 else 812.0
D=4096; NTOK=1716; MCAND=1024; DL=512; KL=1024
wmap=st_index(hf)
X0,ridx,rw,NACT=anchor_layer(ap,L,NTOK)
blob=open(os.path.join(ld,f"dql_vq_L{L:02d}.bin"),"rb").read()
perm=np.random.RandomState(0).permutation(NTOK); ntr=int(NTOK*0.75)
tr,ho=perm[:ntr],perm[ntr:]
Xtr=X0[tr].astype(np.float64)
S1=Xtr.T@Xtr; ew1,Q=np.linalg.eigh(S1)
Q=Q[:,::-1].astype(np.float32); lam1=ew1[::-1].copy()
# 隐藏侧基: 8 探针专家池化
Hp=[]
for e in range(0,256,32):
    P={}
    for nm,wh in (("w1",0),("w3",1)):
        off=vq_slot(blob,e,wh)
        Wf=st_mxfp4(hf,wmap,f"layers.{L}.ffn.experts.{e}.{nm}.weight")
        Wq=vq_dequant(blob,off)
        if Wf.shape!=Wq.shape: Wf=Wf.T
        P[nm]=Wf.astype(np.float32)
    xs=X0[tr[:300]].astype(np.float32)
    Hp.append(swiglu(xs@P["w1"].T,xs@P["w3"].T))
H=np.vstack(Hp).astype(np.float64)
ew2,Q2=np.linalg.eigh(H.T@H)
Q2=Q2[:,::-1].astype(np.float32); lam2=ew2[::-1].copy()
NCOEF=int(KB*1024/0.5)     # q4 = 0.5B/系数
print(f"L{L}: 基就绪. 每专家系数预算 {NCOEF} 个(q4 {KB:.0f}KB). 建 patched 对级…", flush=True)

def q4(v):
    s=np.abs(v).max()/7.0+1e-20
    return (np.clip(np.round(v/s),-7,7)*s).astype(np.float32)

need=sorted(set(int(e) for e in ridx.reshape(-1)))
dH0=np.zeros((NTOK,D),dtype=np.float32)   # 原始 q2 残差(基线)
dH=np.zeros((NTOK,D),dtype=np.float32)    # patched 残差
prow=[];pe=[];pw=[];pYQ=[]
for i,e in enumerate(need):
    rows,slots=np.where(ridx==e)
    if len(rows)==0: continue
    xs=X0[rows].astype(np.float32); w=rw[rows,slots].astype(np.float32)
    P={}
    for nm,wh in (("w1",0),("w3",1),("w2",2)):
        off=vq_slot(blob,e,wh)
        Wf=st_mxfp4(hf,wmap,f"layers.{L}.ffn.experts.{e}.{nm}.weight")
        Wq=vq_dequant(blob,off)
        if Wf.shape!=Wq.shape: Wf=Wf.T
        P[nm]=(Wf.astype(np.float32),Wq.astype(np.float32))
    D1=P["w1"][0]-P["w1"][1]; D3=P["w3"][0]-P["w3"][1]; D2=P["w2"][0]-P["w2"][1]
    C1=D1@Q[:,:MCAND]; C3=D3@Q[:,:MCAND]; C2=D2@Q2[:,:MCAND]
    # 候选(矩阵,方向): 得分=λ·||系数||², 成本=系数条长; 贪心塞预算
    cand=[]
    for j in range(MCAND):
        cand.append((lam1[j]*float((C1[:,j]**2).sum()),2048,0,j))
        cand.append((lam1[j]*float((C3[:,j]**2).sum()),2048,1,j))
        cand.append((lam2[j]*float((C2[:,j]**2).sum()),4096,2,j))
    cand.sort(key=lambda c:-c[0]/c[1])
    left=NCOEF; sel=[[],[],[]]
    for sc,cost,mi,j in cand:
        if cost<=left: sel[mi].append(j); left-=cost
    W1p=P["w1"][1].copy(); W3p=P["w3"][1].copy(); W2p=P["w2"][1].copy()
    if sel[0]:
        js=np.array(sel[0]); W1p+=q4(C1[:,js])@Q[:,js].T
    if sel[1]:
        js=np.array(sel[1]); W3p+=q4(C3[:,js])@Q[:,js].T
    if sel[2]:
        js=np.array(sel[2]); W2p+=q4(C2[:,js])@Q2[:,js].T
    Yf=swiglu(xs@P["w1"][0].T,xs@P["w3"][0].T)@P["w2"][0].T
    Yq0=swiglu(xs@P["w1"][1].T,xs@P["w3"][1].T)@P["w2"][1].T
    Yqp=swiglu(xs@W1p.T,xs@W3p.T)@W2p.T
    dH0[rows]+=w[:,None]*(Yf-Yq0)
    dH[rows]+=w[:,None]*(Yf-Yqp)
    prow.append(rows); pe.append(np.full(len(rows),e,dtype=np.int32))
    pw.append(w); pYQ.append(Yqp.astype(np.float32))
    if (i+1)%32==0: print(f"  …{i+1}/{len(need)} 专家", flush=True)
prow=np.concatenate(prow); pe=np.concatenate(pe); pw=np.concatenate(pw); pYQ=np.vstack(pYQ)
X=X0.astype(np.float64)
e0ho=float((dH0[ho].astype(np.float64)**2).sum())
def recA(Rho): return (1-float((Rho.astype(np.float64)**2).sum())/e0ho)*100
print(f"\n★ 权重侧车(q4+自适应选向): held-out 层挽回 {recA(dH[ho]):.1f}%", flush=True)

# ── 行为栈 z^L + corr + GE(与 ladder v2 同纪律, 目标=patched 残差)
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
print(f"★ +z^L(rank{KL}): held-out 层挽回 {recA(R2[ho]):.1f}%", flush=True)
f=(X@Uw[:,:DL]).astype(np.float32)
colw2=np.sqrt(R2[tr].var(0)+1e-12)
R2s=R2*colw2
_,_,Vt2=np.linalg.svd(R2s[tr],full_matrices=False)
Uo=Vt2[:DL]; P2=R2s@Uo.T
M=np.zeros((NTOK,256),dtype=np.float32)
for s in range(NACT): M[np.arange(NTOK),ridx[:,s]]=1.0
sperm=np.random.RandomState(2).permutation(ntr)
cut=int(ntr*0.8); btr,bval=tr[sperm[:cut]],tr[sperm[cut:]]
def solve_C(rows,rr):
    Mr=M[rows]; Cm=np.zeros((256,DL))
    for i in range(DL):
        A=Mr*f[rows,i][:,None]
        Gc=A.T@A; Gc[np.diag_indices_from(Gc)]+=rr*np.trace(Gc)/256+1e-10
        Cm[:,i]=np.linalg.solve(Gc,A.T@P2[rows,i])
    return Cm
ebv=float((R2[bval].astype(np.float64)**2).sum())
bestB=None
for rr in (1.0,3.0,10.0):
    Cm=solve_C(btr,rr)
    pv=((f[bval]*(M[bval]@Cm))@Uo)/np.maximum(colw2,1e-12)
    g=(1-float(((R2[bval]-pv.astype(np.float32)).astype(np.float64)**2).sum())/ebv)*100
    if bestB is None or g>bestB[0]: bestB=(g,rr)
if bestB[0]>0:
    C=solve_C(tr,bestB[1])
    predC=((f*(M@C))@Uo)/np.maximum(colw2,1e-12)
    R3=R2-predC.astype(np.float32)
else: R3=R2
print(f"★ +corr(岭={bestB[1]:g}): held-out 层挽回 {recA(R3[ho]):.1f}%", flush=True)
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
            for eb,_ in vecs: Gg[256+ea,256+eb]+=float(D)
    return Gg,bg
Gg,bg=build_ge(tr)
r_=1e-4*max(np.trace(Gg)/NP,1.0)
for d_ in range(NP): Gg[d_,d_]+=r_
gb=np.linalg.solve(Gg,bg); g=gb[:256]; beta=gb[256:]
R4=R3.copy()
for t in range(NTOK):
    for p in tok_pairs[t]:
        R4[t]-=np.float32(g[pe[p]])*pw[p]*pYQ[p]+np.float32(beta[pe[p]])
print(f"★ +GE/β: held-out 层挽回 {recA(R4[ho]):.1f}%", flush=True)
vol=KB/1024*256+24
print(f"\n总体积 ≈ {vol:.0f}MB/层 → 43层 {vol*43/1024:.1f} GB(权重侧车 q4 {KB:.0f}KB/专家 + 行为栈 24MB/层)", flush=True)
