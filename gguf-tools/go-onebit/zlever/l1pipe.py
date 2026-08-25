"""zlever/l1pipe.py — L00 单层全链路(量化语义, 产物落盘, 文件重载复现)。
管线: ①感知基(激活协方差, 层共享) ②两遍权重残差: 遍1=系数均值(共享补丁Δ̄, 近免费),
遍2=去均值系数按 λ·能量选向+qN 量化(每专家 z_e) ③行为栈 z^L+corr+GE(闭式)
④产物落盘 zlever_L%02d/ ⑤从文件重载全链复现 held-out 数字(路跑通判据)。
用法: l1pipe.py <hf> <layers_dir> <anchor> <层> [每专家KB=700] [qbits=4]
"""
import os, sys, numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','scripts'))
from probe_behavior_spectrum import st_index, st_mxfp4, vq_slot, vq_dequant
from probe_layer_behavior import anchor_layer, swiglu
hf,ld,ap,L = sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
KB=float(sys.argv[5]) if len(sys.argv)>5 else 700.0
QB=int(sys.argv[6]) if len(sys.argv)>6 else 4
D=4096; NTOK=1716; MC=1024; DL=512; KL=1024
OUT=os.path.join(ld,f"zlever_L{L:02d}")
os.makedirs(OUT,exist_ok=True)
wmap=st_index(hf)
X0,ridx,rw,NACT=anchor_layer(ap,L,NTOK)
blob=open(os.path.join(ld,f"dql_vq_L{L:02d}.bin"),"rb").read()
perm=np.random.RandomState(0).permutation(NTOK); ntr=int(NTOK*0.75)
tr,ho=perm[:ntr],perm[ntr:]
Xtr=X0[tr].astype(np.float64)
# smooth 进感知基: 协方差用 dither 增广口径(与行为栈同配方 0.04·rms/权0.25)
rmsX=np.sqrt((Xtr**2).mean(1,keepdims=True))
Xd=Xtr+np.random.RandomState(1).randn(*Xtr.shape)*0.04*rmsX
ew1,Q=np.linalg.eigh(Xtr.T@Xtr+0.25*(Xd.T@Xd))
Q=Q[:,::-1][:,:MC].astype(np.float32).copy(); lam1=ew1[::-1][:MC].copy()
Hp=[]
def loadW(e):
    P={}
    for nm,wh in (("w1",0),("w3",1),("w2",2)):
        off=vq_slot(blob,e,wh)
        Wf=st_mxfp4(hf,wmap,f"layers.{L}.ffn.experts.{e}.{nm}.weight")
        Wq=vq_dequant(blob,off)
        if Wf.shape!=Wq.shape: Wf=Wf.T
        P[nm]=(Wf.astype(np.float32),Wq.astype(np.float32))
    return P
for e in range(0,256,32):
    P=loadW(e); xs=X0[tr[:300]].astype(np.float32)
    Hp.append(swiglu(xs@P["w1"][0].T,xs@P["w3"][0].T))
H=np.vstack(Hp).astype(np.float64)
ew2,Q2=np.linalg.eigh(H.T@H)
Q2=Q2[:,::-1][:,:MC].astype(np.float32).copy(); lam2=ew2[::-1][:MC].copy()
hrms=np.sqrt((H**2).mean(0)).astype(np.float32)      # 隐藏维激活尺度(感知)
NCO=int(KB*1024/(QB/8.0))
print(f"L{L}: 基就绪. 遍1=系数均值+基线残差(classify列权用)…", flush=True)
Sm=[np.zeros((2048,MC)),np.zeros((2048,MC)),np.zeros((4096,MC))]
dH0=np.zeros((NTOK,D),dtype=np.float32)
need=sorted(set(int(e) for e in ridx.reshape(-1)))
for i,e in enumerate(need):
    P=loadW(e)
    Sm[0]+=P["w1"][0].astype(np.float64)@Q-(P["w1"][1].astype(np.float64)@Q)
    Sm[1]+=P["w3"][0].astype(np.float64)@Q-(P["w3"][1].astype(np.float64)@Q)
    Sm[2]+=P["w2"][0].astype(np.float64)@Q2-(P["w2"][1].astype(np.float64)@Q2)
    rows,slots=np.where(ridx==e)
    if len(rows):
        xs=X0[rows].astype(np.float32); w=rw[rows,slots].astype(np.float32)
        Yf=swiglu(xs@P["w1"][0].T,xs@P["w3"][0].T)@P["w2"][0].T
        Yq0=swiglu(xs@P["w1"][1].T,xs@P["w3"][1].T)@P["w2"][1].T
        dH0[rows]+=w[:,None]*(Yf-Yq0)
    if (i+1)%64==0: print(f"  …{i+1}/{len(need)}", flush=True)
CM=[(s/len(need)).astype(np.float32) for s in Sm]
colw0=np.sqrt(dH0[tr].var(0)+1e-12).astype(np.float32)   # classify 输出列权(w2 选向)
def qn(v,b):
    lv=2**(b-1)-1
    s=np.abs(v).max(axis=0,keepdims=True)/lv+1e-20
    return np.clip(np.round(v/s),-lv,lv).astype(np.int8),s.astype(np.float32)
print(f"遍2=选向(四损失打分)+q{QB}+patched对级+行为栈…", flush=True)
dH=np.zeros((NTOK,D),dtype=np.float32)
prow=[];pe=[];pw=[];pYQ=[]
esave={}
for i,e in enumerate(need):
    rows,slots=np.where(ridx==e)
    if len(rows)==0: continue
    xs=X0[rows].astype(np.float32); w=rw[rows,slots].astype(np.float32)
    P=loadW(e)
    C=[P["w1"][0]@Q-P["w1"][1]@Q, P["w3"][0]@Q-P["w3"][1]@Q, P["w2"][0]@Q2-P["w2"][1]@Q2]
    Ct=[C[k]-CM[k] for k in range(3)]
    # classify 感知打分: 隐藏维影响权 hw=‖W2fp列‖·hrms(w1/w3), 输出列权 colw0(w2)
    hw=(np.linalg.norm(P["w2"][0],axis=0)*hrms)
    s1=((hw[:,None]*Ct[0])**2).sum(0)*lam1
    s3=((hw[:,None]*Ct[1])**2).sum(0)*lam1
    s2=((colw0[:,None]*Ct[2])**2).sum(0)*lam2
    cand=[]
    for j in range(MC):
        cand.append((float(s1[j]),2048,0,j))
        cand.append((float(s3[j]),2048,1,j))
        cand.append((float(s2[j]),4096,2,j))
    cand.sort(key=lambda c:-c[0]/c[1])
    left=NCO; sel=[[],[],[]]
    for sc,cost,mi,j in cand:
        if cost<=left: sel[mi].append(j); left-=cost
    Wp=[P["w1"][1].copy(),P["w3"][1].copy(),P["w2"][1].copy()]
    QQ=[Q,Q,Q2]; sv={}
    for k in range(3):
        Wp[k]+=CM[k]@QQ[k].T
        if sel[k]:
            js=np.array(sorted(sel[k]))
            code,sc_=qn(Ct[k][:,js],QB)
            Wp[k]+=(code.astype(np.float32)*sc_)@QQ[k][:,js].T
            sv[f"j{k}"]=js.astype(np.int16); sv[f"c{k}"]=code; sv[f"s{k}"]=sc_
        else:
            sv[f"j{k}"]=np.zeros(0,dtype=np.int16); sv[f"c{k}"]=np.zeros((Wp[k].shape[0],0),dtype=np.int8); sv[f"s{k}"]=np.zeros((1,0),dtype=np.float32)
    esave[e]=sv
    Yf=swiglu(xs@P["w1"][0].T,xs@P["w3"][0].T)@P["w2"][0].T
    Yqp=swiglu(xs@Wp[0].T,xs@Wp[1].T)@Wp[2].T
    dH[rows]+=w[:,None]*(Yf-Yqp)
    prow.append(rows); pe.append(np.full(len(rows),e,dtype=np.int32)); pw.append(w); pYQ.append(Yqp.astype(np.float32))
    if (i+1)%32==0: print(f"  …{i+1}/{len(need)}", flush=True)
prow=np.concatenate(prow);pe=np.concatenate(pe);pw=np.concatenate(pw);pYQ=np.vstack(pYQ)
X=X0.astype(np.float64)
e0=float((dH0[ho].astype(np.float64)**2).sum())
def rec(R): return (1-float((R.astype(np.float64)**2).sum())/e0)*100
print(f"★ 权重侧车(均值+去均值q{QB}): held-out {rec(dH[ho]):.1f}%", flush=True)
# 行为栈
lam=3.0
colw=np.sqrt(dH[tr].var(0)+1e-12)
rms=np.sqrt((X[tr]**2).mean(1,keepdims=True))
rr_=np.random.RandomState(1)
Xa=np.vstack([X[tr],(X[tr]+rr_.randn(ntr,D)*0.04*rms)*np.sqrt(0.25)])
Ra=np.vstack([dH[tr]*colw,(dH[tr]*colw)*np.sqrt(0.25)])
G=Xa@Xa.T; G[np.diag_indices_from(G)]+=lam*np.trace(G)/Xa.shape[1]+1e-10
al=np.linalg.solve(G,Ra)
Uw,Sw,Vtw=np.linalg.svd(Xa.T@al,full_matrices=False)
ZU=(Uw[:,:KL]*Sw[:KL]).astype(np.float32); ZV=Vtw[:KL].astype(np.float32)
predL=(X.astype(np.float32)@ZU)@ZV/np.maximum(colw,1e-12)
R2=dH-predL
print(f"★ +z^L: held-out {rec(R2[ho]):.1f}%", flush=True)
f=(X.astype(np.float32)@Uw[:,:DL].astype(np.float32))
colw2=np.sqrt(R2[tr].var(0)+1e-12)
R2s=R2*colw2
_,_,Vt2=np.linalg.svd(R2s[tr].astype(np.float64),full_matrices=False)
Uo=Vt2[:DL].astype(np.float32); P2=R2s@Uo.T
M=np.zeros((NTOK,256),dtype=np.float32)
for s in range(NACT): M[np.arange(NTOK),ridx[:,s]]=1.0
Cc=np.zeros((256,DL))
for i in range(DL):
    A=M[tr]*f[tr,i][:,None]
    Gc=(A.T@A).astype(np.float64); Gc[np.diag_indices_from(Gc)]+=3.0*np.trace(Gc)/256+1e-10
    Cc[:,i]=np.linalg.solve(Gc,A.T@P2[tr,i].astype(np.float64))
Cc32=Cc.astype(np.float32)
predC=((f*(M@Cc32))@Uo)/np.maximum(colw2,1e-12)
R3=R2-predC
print(f"★ +corr: held-out {rec(R3[ho]):.1f}%", flush=True)
NP=512
tok_pairs=[[] for _ in range(NTOK)]
for p in range(len(prow)): tok_pairs[prow[p]].append(p)
Gg=np.zeros((NP,NP)); bg=np.zeros(NP)
for t in tr:
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
        for eb,vb in vecs: Gg[ea,256+eb]+=sa; Gg[256+eb,ea]+=sa
        for eb,_ in vecs: Gg[256+ea,256+eb]+=float(D)
rg=1e-4*max(np.trace(Gg)/NP,1.0)
for d_ in range(NP): Gg[d_,d_]+=rg
gb=np.linalg.solve(Gg,bg); gvec=gb[:256].astype(np.float32); bvec=gb[256:].astype(np.float32)
R4=R3.copy()
for t in range(NTOK):
    for p in tok_pairs[t]:
        R4[t]-=gvec[pe[p]]*pw[p]*pYQ[p]+bvec[pe[p]]
print(f"★ +GE/β: held-out {rec(R4[ho]):.1f}%", flush=True)
# ── 产物落盘
np.savez(os.path.join(OUT,"basis.npz"),Q=Q.astype(np.float16),Q2=Q2.astype(np.float16))
np.savez(os.path.join(OUT,"shared_mean.npz"),cm0=CM[0].astype(np.float16),cm1=CM[1].astype(np.float16),cm2=CM[2].astype(np.float16))
np.savez(os.path.join(OUT,"stack.npz"),ZU=ZU.astype(np.float16),ZV=ZV.astype(np.float16),colw=colw.astype(np.float32),
         fU=Uw[:,:DL].astype(np.float16),Cc=Cc32.astype(np.float16),Uo=Uo.astype(np.float16),colw2=colw2.astype(np.float32),
         g=gvec,beta=bvec)
ez={}
for e,sv in esave.items():
    for k,v in sv.items(): ez[f"e{e}_{k}"]=v
np.savez(os.path.join(OUT,"expert_z.npz"),**ez)
tot=sum(os.path.getsize(os.path.join(OUT,fn)) for fn in os.listdir(OUT))
print(f"产物落盘 {OUT}: {tot/2**20:.0f}MB(npz未压q4按int8存, 真q4口径={KB*256/1024+49:.0f}MB/层)", flush=True)
# ── 从文件重载复现(路跑通判据): 抽 3 专家逐 token 校验 patched 前向一致
zf=np.load(os.path.join(OUT,"expert_z.npz")); bs=np.load(os.path.join(OUT,"basis.npz")); sm=np.load(os.path.join(OUT,"shared_mean.npz"))
Qr=bs["Q"].astype(np.float32); Q2r=bs["Q2"].astype(np.float32)
ok=True
for e in (need[0],need[len(need)//2],need[-1]):
    P=loadW(e)
    Wp=[P["w1"][1].copy(),P["w3"][1].copy(),P["w2"][1].copy()]
    QQ=[Qr,Qr,Q2r]; CMr=[sm["cm0"].astype(np.float32),sm["cm1"].astype(np.float32),sm["cm2"].astype(np.float32)]
    for k in range(3):
        Wp[k]+=CMr[k]@QQ[k].T
        js=zf[f"e{e}_j{k}"]
        if len(js):
            Wp[k]+=(zf[f"e{e}_c{k}"].astype(np.float32)*zf[f"e{e}_s{k}"])@QQ[k][:,js.astype(np.int64)].T
    rows,slots=np.where(ridx==e)
    xs=X0[rows].astype(np.float32)
    Yr=swiglu(xs@Wp[0].T,xs@Wp[1].T)@Wp[2].T
    d=float(np.abs(Yr-pYQ[pe==e]).max())
    print(f"  重载校验 e{e}: max|Δ|={d:.2e} {'✓' if d<3e-2 else '✗'}", flush=True)
    ok=ok and d<3e-2
print(f"★ 路{'跑通' if ok else '未通'}: 产物文件→重载→前向一致", flush=True)
