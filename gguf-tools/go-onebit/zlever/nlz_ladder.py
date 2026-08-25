"""zlever/nlz_ladder.py — 非线性侧车容量阶梯(2026-08-09 用户令: 86G 底座找还原点)。
在单层量化误差 dH(x) 上, 同字节预算对打多种侧车形式, held 段挽回率为唯一判据:
  lin-k   线性 z^L rank-k(zlayer 同解, 参考基线)
  ftA-k   特征提升 [x, x⊙x, relu(x)] ridge 后 SVD 截 rank-k(部署=+1矩阵乘+逐元)
  pwC-R×k 路由/输入分片: x kmeans R 片, 每片独立 rank-k ridge(分段线性=非线性)
  mlpD-r  微型 MLP x→r→D(numpy adam, 部署=2 小矩阵乘+silu)
  knn     kNN oracle(无穷容量代理, 非参数: held 的 dH 用 fit 近邻加权重构)
         → knn 挽回率 ≈ "误差从 x 可学出的上限", 是还原点问题的概念天花板。
用法: nlz_ladder.py <hf> <layers_dir> <anchor> <层> [env: DS4_ZL_NTOK/DS4_ZL_NFIT]
前置: 先跑 zlayer.py 同层建 zcache_LXX.npz(本工具复用其 dH 缓存)。
产物: <layers_dir>/nlz_report_LXX.json; 日志 stderr 逐配置一行。
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','scripts'))
from probe_layer_behavior import anchor_layer, swiglu
from probe_behavior_spectrum import st_index, st_mxfp4, vq_slot, vq_dequant

def rsvd(M,r,it=2,seed=5):
    rs=np.random.RandomState(seed)
    Q=np.linalg.qr(M@rs.randn(M.shape[1],r+8).astype(M.dtype))[0]
    for _ in range(it): Q=np.linalg.qr(M@ (M.T@Q))[0]
    B=Q.T@M
    Ub,Sb,Vb=np.linalg.svd(B,full_matrices=False)
    return (Q@Ub)[:,:r],Sb[:r],Vb[:r]

hf,ld,ap,L = sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
NTOK=int(os.getenv("DS4_ZL_NTOK","1716")); NF=int(os.getenv("DS4_ZL_NFIT","1287"))
D=4096
t0=time.time()
X0,ridx,rw,NACT=anchor_layer(ap,L,NTOK)
zc=np.load(os.path.join(ld,f"zcache_L{L:02d}.npz"))
dH=zc["dH"].astype(np.float32)          # (NTOK,D) 量化误差(路由加权层输出差)
X=X0.astype(np.float32)
_fr=os.getenv("DS4_ZL_FIT_RANGES"); _er=os.getenv("DS4_ZL_EV_RANGE")
if _fr:
    tr=np.concatenate([np.arange(int(a),int(b)) for a,b in (seg.split(":") for seg in _fr.split(","))])
    _a,_b=_er.split(":"); ev=np.arange(int(_a),int(_b))
else:
    tr=np.arange(0,NF); ev=np.arange(NF,NTOK)
e0=float((dH[ev].astype(np.float64)**2).sum())
rep={"L":L,"NTOK":NTOK,"NFIT":NF,"e0":e0,"runs":[]}
# ★A类满额探针 v2: 全专家×三矩阵×秩曲线, 与 z 侧车同口径(残差dH全层held挽回)★
if os.getenv("NLZ_BROT"):
    # ★B类旋转单层探针(2026-08-11 用户令"找L37试一试"): 死层 VQ 误差在 Hadamard 旋转系下
    # 是否显著缩小(QuaRot/SpinQuant 方向判决)。mini-VQ(kmeans dim4 nc512, 激活感知列加权)
    # 只测旋转增量, 非产线码本; w2 跳过(w1w3=2/3字节, 定向足)。
    prow=zc["prow"]; pe=zc["pe"]
    XQP=os.getenv("DS4_ZL_XANCHOR"); XQ,_,_,_=anchor_layer(XQP,L,NTOK)
    wmap=st_index(hf)
    rng=np.random.default_rng(0)
    def fht(A):   # A[...,D]·H/sqrt(D), D=2^k
        D=A.shape[-1]; lead=A.shape[:-1]
        A=A.astype(np.float64).reshape(-1,D).copy(); h=1
        while h<D:
            A=A.reshape(-1,D//(2*h),2,h)
            a=A[:,:,0,:].copy(); b=A[:,:,1,:].copy()
            A[:,:,0,:]=a+b; A[:,:,1,:]=a-b
            A=A.reshape(-1,D); h*=2
        return A.reshape(*lead,D)/np.sqrt(D)
    def km_vq(W,sx):   # 列加权 mini-VQ: 返回反加权后的 Wq
        Ws=(W*sx[None,:]).astype(np.float32); blk=Ws.reshape(-1,4)
        sub=blk[rng.choice(len(blk),min(60000,len(blk)),replace=False)]
        C=sub[rng.choice(len(sub),512,replace=False)].astype(np.float64)
        for _ in range(12):
            d=((sub[:,None,:]-C[None,:,:])**2).sum(-1); a=d.argmin(1)
            for k in np.unique(a): C[k]=sub[a==k].mean(0)
        q=np.empty(len(blk),np.int32)
        for o in range(0,len(blk),262144):
            d=((blk[o:o+262144,None,:]-C[None,:,:])**2).sum(-1); q[o:o+262144]=d.argmin(1)
        return C[q].reshape(Ws.shape)/sx[None,:]
    def oerr(W,Wq,x):
        E=W-Wq; y=x@W.T; dy=x@E.T
        return float((dy*dy).sum())/max(float((y*y).sum()),1e-12)
    order=sorted(np.unique(pe).tolist(),key=lambda e:-int((pe==e).sum()))[:8]
    accp=[]; accr=[]
    for e in order:
        idx=np.where(pe==e)[0]; rows=np.unique(prow[idx])
        if len(rows)<24: continue
        x=XQ[rows].astype(np.float64); xr=fht(x)
        sx=np.sqrt((x**2).mean(0))+1e-8; sxr=np.sqrt((xr**2).mean(0))+1e-8
        for nm in ("w1","w3"):
            Wf=st_mxfp4(hf,wmap,f"layers.{L}.ffn.experts.{e}.{nm}.weight").astype(np.float64)
            if Wf.shape[1]!=x.shape[1]: Wf=Wf.T
            ep=oerr(Wf,km_vq(Wf,sx),x)
            Wr=fht(Wf)                       # y=Wx=(WH)(H^T x)
            er=oerr(Wr,km_vq(Wr,sxr),xr)
            accp.append(ep); accr.append(er)
        print(f"  L{L} brot e{e} plain={np.mean(accp):.4f} rot={np.mean(accr):.4f}",flush=True)
    p=float(np.mean(accp)); r=float(np.mean(accr))
    print(f"★L{L} B旋转探针: 输出空间相对误差 plain={p:.4f} rot={r:.4f} Δ={(1-r/p)*100:+.1f}% (对={len(accp)})",flush=True)
    sys.exit(0)
if os.getenv("NLZ_ERF"):
    # ★EoRA残差重加权探针(2026-08-11 用户令"试一试"): 方向=ΔW_w2 加权SVD top-r(确定性不学),
    # 幅度α对层输出残差 dH 重解(每专家r个标量+ridge ⇒ 无记忆化空间); 药方向α→0 自动避开
    # "撤药"陷阱(L27 r4=-0.1 签名)。w2-only: 线性路特征精确且 COADAPT 药即在 w2。
    XQP=os.getenv("DS4_ZL_XANCHOR"); XQ,_,_,_=anchor_layer(XQP,L,NTOK)
    wmap=st_index(hf)
    blob=open(os.path.join(ld,f"dql_vq_L{L:02d}.bin"),"rb").read()
    prow=zc["prow"]; pe=zc["pe"]; pw=zc["pw"]
    tr_mask=np.zeros(NTOK,bool); tr_mask[tr]=True
    ev_mask=np.zeros(NTOK,bool); ev_mask[ev]=True
    RR=int(os.getenv("NLZ_ERF_R","8")); lam=float(os.getenv("NLZ_ERF_LAM","1.0"))
    e0=float((dH[ev].astype(np.float64)**2).sum())
    Rw=dH.astype(np.float64).copy()
    pred=np.zeros((len(ev),4096),np.float64)
    ev_pos={int(t):i for i,t in enumerate(ev)}
    order=sorted(np.unique(pe).tolist(),key=lambda e:-int((pe==e).sum()))[:128]
    ndone=0; nrej=0; amags=[]
    for e in order:
        idx=np.where(pe==e)[0]
        f_tr=idx[tr_mask[prow[idx]]]; f_ev=idx[ev_mask[prow[idx]]]
        if len(f_tr)<24 or len(f_ev)<2: continue
        o1=vq_slot(blob,e,0); o3=vq_slot(blob,e,1); o2=vq_slot(blob,e,2)
        if not (o1 and o3 and o2): continue
        Wf=st_mxfp4(hf,wmap,f"layers.{L}.ffn.experts.{e}.w2.weight").astype(np.float64)
        Wq=vq_dequant(blob,o2).astype(np.float64)
        if Wf.shape!=Wq.shape: Wf=Wf.T
        W1q=vq_dequant(blob,o1); W3q=vq_dequant(blob,o3)
        x_tr=XQ[prow[f_tr]].astype(np.float64); x_ev=XQ[prow[f_ev]].astype(np.float64)
        if W1q.shape[1]!=x_tr.shape[1]: W1q=W1q.T
        if W3q.shape[1]!=x_tr.shape[1]: W3q=W3q.T
        h_tr=swiglu(x_tr@W1q.T,x_tr@W3q.T); h_ev=swiglu(x_ev@W1q.T,x_ev@W3q.T)
        sh=np.sqrt((h_tr**2).mean(0))+1e-8
        dW=Wf-Wq
        U,S,V=rsvd((dW*sh[None,:]).astype(np.float32),RR)
        U=U.astype(np.float64); S=S.astype(np.float64); V=V.astype(np.float64)/sh[None,:]
        w_tr=pw[f_tr].astype(np.float64); w_ev=pw[f_ev].astype(np.float64)
        G_tr=(h_tr@V.T)*w_tr[:,None]; G_ev=(h_ev@V.T)*w_ev[:,None]
        Rr=Rw[prow[f_tr]]
        A=(G_tr.T@G_tr)*(U.T@U)*np.outer(S,S)
        b=S*np.einsum("ni,nd,di->i",G_tr,Rr,U)
        a=np.linalg.solve(A+lam*(np.trace(A)/RR+1e-12)*np.eye(RR),b)
        ctr=(G_tr*(a*S)[None,:])@U.T
        cev=(G_ev*(a*S)[None,:])@U.T
        # ★token门(2026-08-12 SPEAR机制/用户支柱③): 推理可得特征=修正能量|c(x)|²;
        # fit行按能量降序取净增益前缀最大处为阈值τ(单自由度, 无记忆化空间), ev同τ应用。
        if int(os.getenv("NLZ_ERF_GATE","0")):
            en=(ctr**2).sum(1); ben=2*(Rr*ctr).sum(1)-en
            oi=np.argsort(-en); cum=np.cumsum(ben[oi])
            m=int(cum.argmax())+1
            if cum[m-1]<=0: nrej+=1; continue
            tau=en[oi[m-1]]
            ctr=ctr*(en>=tau)[:,None]
            cev=cev*(((cev**2).sum(1))>=tau)[:,None]
        fit_gain=float(2*(Rr*ctr).sum()-(ctr**2).sum())
        if fit_gain<=0: nrej+=1; continue
        Rw[prow[f_tr]]-=ctr
        for j,i_ in enumerate(f_ev): pred[ev_pos[int(prow[i_])]]+=cev[j]
        ndone+=1; amags.append(float(np.abs(a).mean()))
        if ndone%32==0: print(f"  L{L} erf …{ndone}",flush=True)
    dev=dH[ev].astype(np.float64)
    rec=(1-float(((dev-pred)**2).sum())/e0)*100
    print(f"★L{L} ERF残差重加权(w2-only r{RR} lam={lam}): held挽回 {rec:.1f}%  专家={ndone} 拒={nrej} α均绝={np.mean(amags) if amags else 0:.2f}",flush=True)
    sys.exit(0)
if os.getenv("NLZ_EORA"):
    XQP=os.getenv("DS4_ZL_XANCHOR")
    XQ,_,_,_=anchor_layer(XQP,L,NTOK)
    wmap=st_index(hf)
    blob=open(os.path.join(ld,f"dql_vq_L{L:02d}.bin"),"rb").read()
    prow=zc["prow"]; pe=zc["pe"]; pw=zc["pw"]   # ADDON缓存: pw已含GE, dH=贪心修后残差
    tr_mask=np.zeros(NTOK,bool); tr_mask[tr]=True
    ev_mask=np.zeros(NTOK,bool); ev_mask[ev]=True
    RS=(4,8,16,32)
    # ★r8购入(2026-08-11 用户令"如果r8有收益我愿意买"): 死层唯一正机制的产物落盘。
    # U·S折叠+V已除权重白化(sd) ⇒ 部署即 W_q + U@V, fp16 存(修正项精度足)。
    save_r=int(os.getenv("NLZ_EORA_SAVE_R","0")); sav={}
    spec=int(os.getenv("NLZ_EORA_SPEC","0"))      # 谱诊断: 打印每专家ΔW加权奇异谱头部占比
    nexp_cap=int(os.getenv("NLZ_EORA_NEXP","128"))
    pred={r:np.zeros((len(ev),4096),dtype=np.float64) for r in RS}
    ev_pos={int(t):i for i,t in enumerate(ev)}
    order=sorted(np.unique(pe).tolist(),key=lambda e:-int((pe==e).sum()))[:nexp_cap]
    ndone=0; cov=0
    for e in order:
        idx=np.where(pe==e)[0]
        f_tr=idx[tr_mask[prow[idx]]]; f_ev=idx[ev_mask[prow[idx]]]
        if len(f_tr)<24 or len(f_ev)<2: continue
        P={}
        for nm,wh in (("w1",0),("w3",1),("w2",2)):
            Wf=st_mxfp4(hf,wmap,f"layers.{L}.ffn.experts.{e}.{nm}.weight")
            off=vq_slot(blob,e,wh); Wq=vq_dequant(blob,off) if off else None
            if Wq is not None and Wf.shape!=Wq.shape: Wf=Wf.T
            P[nm]=(Wf.astype(np.float64),(Wq if Wq is not None else Wf).astype(np.float64))
        x_tr=XQ[prow[f_tr]].astype(np.float64); x_ev=XQ[prow[f_ev]].astype(np.float64)
        w_ev=pw[f_ev].astype(np.float64)
        sx=np.sqrt((x_tr**2).mean(0))+1e-8
        hq_tr=swiglu(x_tr@P["w1"][1].T,x_tr@P["w3"][1].T)
        sh=np.sqrt((hq_tr**2).mean(0))+1e-8
        F={}
        for nm,sd in (("w1",sx),("w3",sx),("w2",sh)):
            dW=P[nm][0]-P[nm][1]
            U,S,V=rsvd((dW*sd[None,:]).astype(np.float32),36)
            F[nm]=(U.astype(np.float64),S.astype(np.float64),V.astype(np.float64)/sd[None,:])
            if save_r:
                sav[f"e{e}_{nm}_U"]=(F[nm][0][:,:save_r]*F[nm][1][:save_r]).astype(np.float16)
                sav[f"e{e}_{nm}_V"]=F[nm][2][:save_r].astype(np.float16)
            if spec:
                S2=F[nm][1]**2; tot=float(S2.sum())
                fr=lambda k:float(S2[:k].sum())/max(tot,1e-18)*100
                # 谱头占比是 rank-36 窗内口径(全谱在dim4 VQ下≈满秩), 只作层间对比
                print(f"  L{L} spec e{e} {nm}: top4={fr(4):.1f}% top8={fr(8):.1f}% top16={fr(16):.1f}% top32={fr(32):.1f}% s1/s36={float(F[nm][1][0]/max(F[nm][1][-1],1e-18)):.2f}",flush=True)
        Yf_ev=swiglu(x_ev@P["w1"][0].T,x_ev@P["w3"][0].T)@P["w2"][0].T
        Yq_ev=swiglu(x_ev@P["w1"][1].T,x_ev@P["w3"][1].T)@P["w2"][1].T
        rows_i=[ev_pos[int(prow[i])] for i in f_ev]
        for r in RS:
            C={nm:(F[nm][0][:,:r]*F[nm][1][:r])@F[nm][2][:r] for nm in ("w1","w3","w2")}
            Yc=swiglu(x_ev@(P["w1"][1]+C["w1"]).T,x_ev@(P["w3"][1]+C["w3"]).T)@(P["w2"][1]+C["w2"]).T
            contrib=w_ev[:,None]*(Yc-Yq_ev)
            for j,ri in enumerate(rows_i): pred[r][ri]+=contrib[j]
        ndone+=1; cov+=len(f_ev)
        if ndone%32==0: print(f"  L{L} eora2 …{ndone}/{len(order)}",flush=True)
    dev=dH[ev].astype(np.float64)
    line=" ".join(f"r{r}:{(1-float(((dev-pred[r])**2).sum())/e0)*100:.1f}%" for r in RS)
    print(f"  eora2-full     全层held挽回(z同口径) {line}  专家={ndone} 覆盖ev对={cov}",flush=True)
    json.dump({"L":L,"eora2":{str(r):1-float(((dev-pred[r])**2).sum())/e0 for r in RS}},
              open(os.path.join(ld,f"nlz_eora2_L{L:02d}.json"),"w"))
    if save_r and sav:
        od=os.getenv("NLZ_EORA_OUT",ld); os.makedirs(od,exist_ok=True)
        sp=os.path.join(od,f"eora_r{save_r}_L{L:02d}.npz")
        np.savez_compressed(sp,**sav)
        print(f"  eora产物已存 {sp} experts={ndone} rank={save_r}",flush=True)
    print(f"★L{L} eora2探针完 {time.time()-t0:.0f}s",flush=True); sys.exit(0)
# ★每专家 rank-1 方向修正探针 v2(stagewise 残差消减, 修双bug: sc维度错/多专家叠加超冲)★
if os.getenv("NLZ_PEXP"):
    prow=zc["prow"]; pe=zc["pe"]; pw=zc["pw"]; pYQ=zc["pYQ"]
    tr_mask=np.zeros(NTOK,bool); tr_mask[tr]=True
    ev_mask=np.zeros(NTOK,bool); ev_mask[ev]=True
    R=dH.astype(np.float64).copy()          # 工作残差(stagewise 消减)
    pred=np.zeros_like(dH)
    order=sorted(np.unique(pe).tolist(), key=lambda e:-int((pe==e).sum()))
    nfit=0
    for e in order:
        idx=np.where(pe==e)[0]
        rt=idx[tr_mask[prow[idx]]]; re_=idx[ev_mask[prow[idx]]]
        if len(rt)<24: continue
        Y=(pYQ[rt]*pw[rt,None]).astype(np.float64)
        Rr=R[prow[rt]]
        lam=1e-3*float((Y*Y).sum())/max(len(rt),1)
        v=Y.mean(0); nv=np.linalg.norm(v)
        if nv<1e-12: continue
        v/=nv; u=None
        for _ in range(8):
            a=Y@v
            u=(Rr*a[:,None]).sum(0)/((a*a).sum()+lam)
            b=Rr@u
            vn=(Y*b[:,None]).sum(0)
            nv=np.linalg.norm(vn)
            if nv<1e-18: break
            v=vn/nv
        if u is None: continue
        a=Y@v
        fit_gain=float(2*(Rr*(np.outer(a,u))).sum()-((a*a).sum())*float(u@u))
        if fit_gain<=0: continue                 # 单专家验收: fit 残差必须真降
        for grp,upd in ((rt,True),(re_,False)):
            if len(grp)==0: continue
            Yg=(pYQ[grp]*pw[grp,None]).astype(np.float64)
            contrib=np.outer(Yg@v,u)
            np.add.at(pred,prow[grp],contrib.astype(np.float32))
            np.subtract.at(R,prow[grp],contrib)   # ev 行同步消减(下一专家看见的残差一致)
        nfit+=1
    r_ev=1-float(((dH[ev]-pred[ev]).astype(np.float64)**2).sum())/e0
    e0tr=float((dH[tr].astype(np.float64)**2).sum())
    r_tr=1-float(((dH[tr]-pred[tr]).astype(np.float64)**2).sum())/e0tr
    nb=nfit*(4096+4096)*2
    print(f"  pexp-r1v2      挽回held={r_ev*100:6.2f}% (fit段 {r_tr*100:.2f}%)  专家数={nfit}  侧车={nb/1e6:.1f}MB",flush=True)
    rep["runs"].append({"form":"pexp-r1v2","recov":r_ev,"MB":nb/1e6})
    json.dump(rep,open(os.path.join(ld,f"nlz_pexp_L{L:02d}.json"),"w"),indent=1)
    print(f"★L{L} pexp探针完 {time.time()-t0:.0f}s",flush=True); sys.exit(0)
def rec(tag,pred_ev,nbytes,note=""):
    r=1-float(((dH[ev]-pred_ev.astype(np.float32)).astype(np.float64)**2).sum())/e0
    rep["runs"].append({"form":tag,"recov":r,"MB":nbytes/1e6,"note":note})
    print(f"  {tag:14s} 挽回={r*100:6.2f}%  侧车={nbytes/1e6:7.1f}MB {note}",flush=True)
    return r

def ridge_solve(F,Y,lam_scale=3.0):
    """对偶 ridge: F(n,f) → W(f,D); n<f 时用 G=F@F.T"""
    G=(F@F.T).astype(np.float64)
    G[np.diag_indices_from(G)]+=lam_scale*np.trace(G)/F.shape[1]+1e-9
    al=np.linalg.solve(G,Y[tr].astype(np.float64) if Y.shape[0]==NTOK else Y.astype(np.float64))
    return (F.T@al)  # (f,D)

# ---- 线性参考(zlayer 同族, 无 dither 简版) ----
W=ridge_solve(X[tr],dH[tr])
A,S,Bt=np.linalg.svd(W,full_matrices=False)
for k in (512,1024,2048,4096):
    if k>min(W.shape): break
    pred=(X[ev]@(A[:,:k]*S[:k]))@Bt[:k]
    rec(f"lin-{k}",pred,k*(D+D)*2)

# ---- A: 特征提升 ----
def feat(M):
    n=np.sqrt((M**2).mean(1,keepdims=True))+1e-6
    return np.concatenate([M, (M*M)/n, np.maximum(M,0)],1).astype(np.float32)
Fa_tr=feat(X[tr]); Fa_ev=feat(X[ev])
Wf=ridge_solve(Fa_tr,dH[tr])
Af,Sf,Bf=np.linalg.svd(Wf,full_matrices=False)
for k in (512,1024,2048):
    pred=(Fa_ev@(Af[:,:k]*Sf[:k]))@Bf[:k]
    rec(f"ftA-{k}",pred,k*(3*D+D)*2)

# ---- ftA+GE 合体(NLZ_GE_FOLD=1: zlayer 同款 GE 解在 ftA 残差上, 四件套真值) ----
if os.getenv("NLZ_GE_FOLD"):
    kF=2048
    predF_tr=((Fa_tr@(Af[:,:kF]*Sf[:kF]))@Bf[:kF]).astype(np.float32)
    predF_ev=((Fa_ev@(Af[:,:kF]*Sf[:kF]))@Bf[:kF]).astype(np.float32)
    prow=zc["prow"]; pe=zc["pe"]; pw=zc["pw"]; pYQ=zc["pYQ"]
    tok_pairs=[[] for _ in range(NTOK)]
    for pp in range(len(prow)): tok_pairs[prow[pp]].append(pp)
    Rfull=np.zeros_like(dH); Rfull[tr]=dH[tr]-predF_tr; Rfull[ev]=dH[ev]-predF_ev
    Gg=np.zeros((256,256)); bg=np.zeros(256)
    for t in tr:
        ps=tok_pairs[t]
        vecs=[(int(pe[pp]),(pw[pp]*pYQ[pp]).astype(np.float64)) for pp in ps]
        rt=Rfull[t].astype(np.float64)
        for a_,(ea,va) in enumerate(vecs):
            bg[ea]+=va@rt
            for b_ in range(a_,len(vecs)):
                eb,vb=vecs[b_]; d=va@vb; Gg[ea,eb]+=d
                if ea!=eb: Gg[eb,ea]+=d
    Gg[np.diag_indices_from(Gg)]+=1e-3*max(np.trace(Gg)/256,1.0)
    gg=np.linalg.solve(Gg,bg)
    gf=(1.0+gg).astype(np.float32)
    eng=0.0
    for t in ev:
        r=Rfull[t].astype(np.float64).copy()
        for pp in tok_pairs[t]:
            r-=(gf[pe[pp]]-1.0)*(pw[pp]*pYQ[pp]).astype(np.float64)
        eng+=float((r**2).sum())
    rec("ftA2048+GE合体",None if False else dH[ev]-0,0)  # 占位防误用
    rep["runs"].pop()  # 撤占位
    comb=1-eng/e0
    rep["runs"].append({"form":"ftA2048+GE","recov":comb,"MB":67.1,"note":"四件套合体"})
    print(f"  ftA2048+GE合体 挽回={comb*100:6.2f}%  侧车=   67.1MB (非线性+四损失+感知+GE)",flush=True)
    if os.getenv("NLZ_ONLY_GEFOLD"):
        json.dump(rep,open(os.path.join(ld,f"nlz_report_L{L:02d}_gefold.json"),"w"),indent=1)
        print(f"★L{L} GE合体完 {time.time()-t0:.0f}s",flush=True); sys.exit(0)

# ---- C: 输入分片 piecewise ----
def kmeans(M,R,it=12):
    C=M[np.random.RandomState(0).choice(len(M),R,replace=False)].copy()
    for _ in range(it):
        d=((M[:,None,:]-C[None,:,:])**2).sum(2) if len(M)*R*M.shape[1]<2e9 else None
        if d is None:
            a=np.zeros(len(M),dtype=int)
            for i in range(0,len(M),1024):
                a[i:i+1024]=(((M[i:i+1024,None,:]-C[None,:,:])**2).sum(2)).argmin(1)
        else: a=d.argmin(1)
        for r_ in range(R):
            m=a==r_
            if m.any(): C[r_]=M[m].mean(0)
    return C
for R,k in ((4,768),):
    Cc=kmeans(X[tr],R)
    def assign(M):
        return (((M[:,None,:]-Cc[None,:,:])**2).sum(2)).argmin(1)
    atr=assign(X[tr]); aev=assign(X[ev])
    pred=np.zeros((len(ev),D),dtype=np.float32); nb=0
    for r_ in range(R):
        mt=atr==r_; me=aev==r_
        if mt.sum()<64:
            continue
        Wp=ridge_solve(X[tr][mt],dH[tr][mt])
        Ap,Sp,Bp=np.linalg.svd(Wp,full_matrices=False)
        kk=min(k,int(mt.sum()))
        if me.any(): pred[me]=((X[ev][me]@(Ap[:,:kk]*Sp[:kk]))@Bp[:kk]).astype(np.float32)
        nb+=kk*(D+D)*2
    rec(f"pwC-{R}x{k}",pred,nb)

# ---- D: 微型 MLP(adam, silu) ----
def mlp(r,epochs=int(os.getenv("NLZ_EPOCHS","150")),lr=1e-3,bs=512):
    rs=np.random.RandomState(2)
    W1=(rs.randn(D,r)*np.sqrt(2.0/D)).astype(np.float32); b1=np.zeros(r,np.float32)
    W2=(rs.randn(r,D)*np.sqrt(1.0/r)).astype(np.float32)
    mW1=np.zeros_like(W1);vW1=np.zeros_like(W1);mW2=np.zeros_like(W2);vW2=np.zeros_like(W2)
    mb1=np.zeros_like(b1);vb1=np.zeros_like(b1)
    Xt=X[tr]; Yt=dH[tr]; n=len(tr); step=0
    for ep in range(epochs):
        pm=np.random.RandomState(ep).permutation(n)
        for i in range(0,n,bs):
            idx=pm[i:i+bs]; xb=Xt[idx]; yb=Yt[idx]
            z1=xb@W1+b1; s=1/(1+np.exp(-z1)); h=z1*s
            pr=h@W2; g=(pr-yb)*(2.0/len(idx))
            gW2=h.T@g; gh=g@W2.T
            gz=gh*(s*(1+z1*(1-s)))
            gW1=xb.T@gz; gb1=gz.sum(0)
            step+=1; b1c=1-0.9**step; b2c=1-0.999**step
            for P,Gd,M,V in ((W1,gW1,mW1,vW1),(W2,gW2,mW2,vW2),(b1,gb1,mb1,vb1)):
                M*=0.9; M+=0.1*Gd; V*=0.999; V+=0.001*Gd*Gd
                P-=lr*((M/b1c)/(np.sqrt(V/b2c)+1e-8)+1e-4*P)
    z1=X[ev]@W1+b1; s=1/(1+np.exp(-z1)); pred=(z1*s)@W2
    return pred,(D*r+r+r*D)*2
for r in (512,1024,2048):
    pred,nb=mlp(r)
    rec(f"mlpD-{r}",pred,nb)

# ---- 残差增强: 线性满秩打底, 非线性只追打线性残差 ----
predL_ev=(X[ev]@W).astype(np.float32); predL_tr=(X[tr]@W).astype(np.float32)
Rtr=dH[tr]-predL_tr; Rev=dH[ev]-predL_ev
base=1-float(((dH[ev]-predL_ev).astype(np.float64)**2).sum())/e0
print(f"  [残差增强] 线性打底={base*100:.2f}%, 以下为线性+X 的组合挽回", flush=True)
WfR=ridge_solve(Fa_tr,Rtr)
AfR,SfR,BfR=np.linalg.svd(WfR,full_matrices=False)
for k in (512,1024):
    predR=(Fa_ev@(AfR[:,:k]*SfR[:k]))@BfR[:k]
    rec(f"lin+ftA-{k}",predL_ev+predR.astype(np.float32),k*(3*D+D)*2)
_Yt=dH; dH_save=dH
dH=np.where(True,dH,dH)  # 占位
def mlp_on(Yt_tr,r,epochs=int(os.getenv("NLZ_EPOCHS","150")),lr=1e-3,bs=512):
    rs=np.random.RandomState(2)
    W1=(rs.randn(D,r)*np.sqrt(2.0/D)).astype(np.float32); b1=np.zeros(r,np.float32)
    W2=(rs.randn(r,D)*np.sqrt(1.0/r)).astype(np.float32)
    mW1=np.zeros_like(W1);vW1=np.zeros_like(W1);mW2=np.zeros_like(W2);vW2=np.zeros_like(W2)
    mb1=np.zeros_like(b1);vb1=np.zeros_like(b1)
    Xt=X[tr]; n=len(tr); step=0
    for ep in range(epochs):
        pm=np.random.RandomState(ep).permutation(n)
        for i in range(0,n,bs):
            idx=pm[i:i+bs]; xb=Xt[idx]; yb=Yt_tr[idx]
            z1=xb@W1+b1; sg=1/(1+np.exp(-z1)); h=z1*sg
            pr=h@W2; g=(pr-yb)*(2.0/len(idx))
            gW2=h.T@g; gh=g@W2.T
            gz=gh*(sg*(1+z1*(1-sg)))
            gW1=xb.T@gz; gb1=gz.sum(0)
            step+=1; b1c=1-0.9**step; b2c=1-0.999**step
            for P,Gd,M,V in ((W1,gW1,mW1,vW1),(W2,gW2,mW2,vW2),(b1,gb1,mb1,vb1)):
                M*=0.9; M+=0.1*Gd; V*=0.999; V+=0.001*Gd*Gd
                P-=lr*((M/b1c)/(np.sqrt(V/b2c)+1e-8)+1e-4*P)
    z1=X[ev]@W1+b1; sg=1/(1+np.exp(-z1))
    return ((z1*sg)@W2),(D*r+r+r*D)*2
for r in (512,1024):
    predR,nb=mlp_on(Rtr,r)
    rec(f"lin+mlp-{r}",predL_ev+predR.astype(np.float32),nb)
# 残差kNN(线性之上还有无近邻可学结构)
nrmR=np.sqrt((X**2).sum(1,keepdims=True))+1e-9
XnR=X/nrmR
simR=XnR[ev]@XnR[tr].T
for K in (4,16):
    topR=np.argpartition(-simR,K,axis=1)[:,:K]
    wvR=np.take_along_axis(simR,topR,1); wvR=np.maximum(wvR,0); wvR/=wvR.sum(1,keepdims=True)+1e-9
    predR=np.einsum('ek,ekd->ed',wvR,Rtr[topR])
    rec(f"lin+knn-{K}",predL_ev+predR.astype(np.float32),0,"(oracle)")

# ---- kNN oracle(概念天花板) ----
nrm=np.sqrt((X**2).sum(1,keepdims=True))+1e-9
Xn=X/nrm
sim=Xn[ev]@Xn[tr].T
for K in (1,4,16):
    top=np.argpartition(-sim,K,axis=1)[:,:K]
    wv=np.take_along_axis(sim,top,1); wv=np.maximum(wv,0); wv/=wv.sum(1,keepdims=True)+1e-9
    pred=np.einsum('ek,ekd->ed',wv,dH[tr][top])
    rec(f"knn-{K}",pred,0,"(oracle 非参数, 不占字节)")

json.dump(rep,open(os.path.join(ld,f"nlz_report_L{L:02d}.json"),"w"),indent=1)
print(f"★L{L} 非线性阶梯完 {time.time()-t0:.0f}s → nlz_report_L{L:02d}.json",flush=True)
