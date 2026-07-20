"""Prefill-only numpy forward for DeepSeek-V4-Flash, streaming fp8 weights from NFS.
Ported faithfully from official inference/model.py + kernel.py (prefill / start_pos=0 path).
Indexer skipped (exact for chunk_len<=2048: top-512 selects all <=512 compressed entries).
act_quant QAT-noise sims skipped (full precision). Layer-major to amortize expert IO.
Captures per-layer FFN input + routing for the Go-domain manifold analysis.
"""
import os, sys, json, time, numpy as np
import ds4reader as R

C = json.load(open(os.path.join(R.HF, "config.json")))
DIM=C["hidden_size"]; NH=C["num_attention_heads"]; HD=C["head_dim"]; RD=C["qk_rope_head_dim"]
QLR=C["q_lora_rank"]; OLR=C["o_lora_rank"]; OG=C["o_groups"]; NL=C["num_hidden_layers"]
NEXP=C["n_routed_experts"]; NACT=C["num_experts_per_tok"]; MOEI=C["moe_intermediate_size"]
WIN=C["sliding_window"]; EPS=C["rms_norm_eps"]; NHASH=C["num_hash_layers"]
ROUTE_SCALE=C["routed_scaling_factor"]; SWLIM=C["swiglu_limit"]
CR=C["compress_ratios"]
RS=C["rope_scaling"]; ROPE_THETA=C["rope_theta"]; CROPE_THETA=C["compress_rope_theta"]
ORIG=RS["original_max_position_embeddings"]; FACTOR=RS["factor"]; BF=RS["beta_fast"]; BS=RS["beta_slow"]
HCM=C["hc_mult"]; HCIT=C["hc_sinkhorn_iters"]; HCEPS=C["hc_eps"]
SKIP_ROUTED = os.environ.get("SKIP_ROUTED","0")=="1"
print(f"cfg dim{DIM} heads{NH} hd{HD} rd{RD} layers{NL} exp{NEXP} act{NACT} moei{MOEI} win{WIN} skip_routed{SKIP_ROUTED}",flush=True)

def rms(x,w):
    x=x.astype(np.float32); v=np.mean(x*x,-1,keepdims=True)
    return (x*np.reciprocal(np.sqrt(v+EPS))*w).astype(np.float32)
def silu(z): return z/(1.0+np.exp(-z))
def sigmoid(z): return 1.0/(1.0+np.exp(-z))
def softmax(z,axis):
    z=z-np.max(z,axis,keepdims=True); e=np.exp(z); return e/np.sum(e,axis,keepdims=True)

def freqs_cis(dim, seqlen, orig, base, factor, bfast, bslow):
    def cdim(nr): return dim*np.log(orig/(nr*2*np.pi))/(2*np.log(base))
    def crange(lo,hi):
        l=int(np.floor(cdim(lo))); h=int(np.ceil(cdim(hi))); return max(l,0),min(h,dim-1)
    fr=1.0/(base**(np.arange(0,dim,2,dtype=np.float64)/dim))
    if orig>0:
        lo,hi=crange(bfast,bslow)
        if lo==hi: hi+=0.001
        ramp=np.clip((np.arange(dim//2,dtype=np.float64)-lo)/(hi-lo),0,1)
        smooth=1-ramp
        fr=fr/factor*(1-smooth)+fr*smooth
    t=np.arange(seqlen,dtype=np.float64)
    ang=np.outer(t,fr)
    return np.cos(ang)+1j*np.sin(ang)   # [seqlen, dim/2]

# per-layer rope tables (depend on compress vs SWA), computed lazily up to max S
_FC={}
def layer_fc(L,S):
    key=(L,S)
    if key not in _FC:
        if CR[L]:
            fc=freqs_cis(RD,S,ORIG,CROPE_THETA,FACTOR,BF,BS)
        else:
            fc=freqs_cis(RD,S,0,ROPE_THETA,FACTOR,BF,BS)
        _FC[key]=fc
    return _FC[key]

def apply_rope(xp, fc, inverse=False):
    # xp[...,RD], fc[S,RD/2] complex, S==xp.shape[0]
    rd=xp.shape[-1]; S=xp.shape[0]
    xc=xp.reshape(xp.shape[:-1]+(rd//2,2)).astype(np.float32)
    z=xc[...,0]+1j*xc[...,1]
    f=np.conj(fc) if inverse else fc
    fb=f.reshape((S,)+(1,)*(z.ndim-2)+(rd//2,))
    z=z*fb
    out=np.empty(xc.shape,np.float32); out[...,0]=z.real; out[...,1]=z.imag
    return out.reshape(xp.shape)

def overlap_transform(t, val):  # t[s',ratio,2d] -> [s',2ratio,d]
    sp,r,dd=t.shape; d=dd//2
    new=np.full((sp,2*r,d),val,dtype=np.float32)
    new[:,r:,:]=t[:,:,d:]
    new[1:,:r,:]=t[:-1,:,:d]
    return new

def load_layer(L):
    g=R.get; p=f"layers.{L}."
    W={}
    W['an']=g(p+"attn_norm.weight"); W['fn']=g(p+"ffn_norm.weight")
    for t in ("hc_attn_fn","hc_attn_base","hc_attn_scale","hc_ffn_fn","hc_ffn_base","hc_ffn_scale"):
        W[t]=g(p+t)
    W['wq_a']=g(p+"attn.wq_a.weight"); W['q_norm']=g(p+"attn.q_norm.weight")
    W['wq_b']=g(p+"attn.wq_b.weight"); W['wkv']=g(p+"attn.wkv.weight")
    W['kv_norm']=g(p+"attn.kv_norm.weight"); W['sink']=g(p+"attn.attn_sink")
    W['wo_a']=g(p+"attn.wo_a.weight"); W['wo_b']=g(p+"attn.wo_b.weight")
    if CR[L]:
        W['cwkv']=g(p+"attn.compressor.wkv.weight"); W['cwgate']=g(p+"attn.compressor.wgate.weight")
        W['cnorm']=g(p+"attn.compressor.norm.weight"); W['cape']=g(p+"attn.compressor.ape")
    W['gate']=g(p+"ffn.gate.weight")
    W['gbias']=g(p+"ffn.gate.bias") if (p+"ffn.gate.bias") in R._wm() else None
    W['tid2eid']=g(p+"ffn.gate.tid2eid").reshape(C["vocab_size"],NACT).astype(np.int64) if (p+"ffn.gate.tid2eid") in R._wm() else None
    # shared expert
    W['s1']=g(p+"ffn.shared_experts.w1.weight"); W['s3']=g(p+"ffn.shared_experts.w3.weight"); W['s2']=g(p+"ffn.shared_experts.w2.weight")
    return W

def compressor(x, W, L):  # x[S,4096] -> kvc[s',512] or None
    S=x.shape[0]; ratio=CR[L]; d=HD; overlap=(ratio==4); coff=1+overlap
    if S<ratio: return None
    kv=(x@W['cwkv'].T).astype(np.float32); score=(x@W['cwgate'].T).astype(np.float32)
    remainder=S%ratio; cutoff=S-remainder
    if remainder>0: kv=kv[:cutoff]; score=score[:cutoff]
    sp=cutoff//ratio
    kv=kv.reshape(sp,ratio,coff*d); score=score.reshape(sp,ratio,coff*d)+W['cape'][None]
    if overlap:
        kv=overlap_transform(kv,0.0); score=overlap_transform(score,-np.inf)
        sm=softmax(score,1); kv=(kv*sm).sum(1)
    else:
        sm=softmax(score,1); kv=(kv*sm).sum(1)
    kv=rms(kv,W['cnorm'])
    fc=layer_fc(L,S)[:cutoff:ratio]
    kv[...,-RD:]=apply_rope(kv[...,-RD:],fc)
    return kv

def attention(x, W, L):
    S=x.shape[0]; fc=layer_fc(L,S)[:S]
    qr=rms(x@W['wq_a'].T, W['q_norm'])           # [S,1024]
    q=(qr@W['wq_b'].T).reshape(S,NH,HD).astype(np.float32)
    q*=np.reciprocal(np.sqrt(np.mean(q*q,-1,keepdims=True)+EPS))
    q[...,-RD:]=apply_rope(q[...,-RD:],fc)
    kv=rms(x@W['wkv'].T, W['kv_norm'])           # [S,512]
    kv[...,-RD:]=apply_rope(kv[...,-RD:],fc)
    kvc=compressor(x,W,L) if CR[L] else None
    Sc=0 if kvc is None else kvc.shape[0]
    kv_all=kv if kvc is None else np.concatenate([kv,kvc],0)  # [N,512]
    N=kv_all.shape[0]; scale=HD**-0.5
    scores=np.einsum('shd,nd->shn',q,kv_all).astype(np.float32)*scale  # [S,NH,N]
    # causal masks
    ar=np.arange(S)
    win_ok=(np.arange(S)[None,:]<=ar[:,None]) & (np.arange(S)[None,:]>ar[:,None]-WIN)  # [S,S]
    mask=np.zeros((S,N),bool); mask[:,:S]=win_ok
    if Sc>0:
        ratio=CR[L]; comp_ok=(np.arange(Sc)[None,:] < ((ar[:,None]+1)//ratio))         # [S,Sc]
        mask[:,S:]=comp_ok
    scores=np.where(mask[:,None,:],scores,-np.inf)
    m=np.max(scores,-1,keepdims=True)                         # [S,NH,1]
    e=np.exp(scores-m)                                        # masked -> 0
    denom=e.sum(-1,keepdims=True)+np.exp(W['sink'].reshape(1,NH,1)-m)
    w=e/denom
    o=np.einsum('shn,nd->shd',w,kv_all).astype(np.float32)    # [S,NH,512]
    o[...,-RD:]=apply_rope(o[...,-RD:],fc,inverse=True)
    o=o.reshape(S,OG,(NH*HD)//OG)                             # [S,8,4096]
    woa=W['wo_a'].reshape(OG,OLR,(NH*HD)//OG)                 # [8,1024,4096]
    o=np.einsum('sgd,grd->sgr',o,woa).reshape(S,OG*OLR)       # [S,8192]
    return (o@W['wo_b'].T).astype(np.float32)                 # [S,4096]

def hc_sinkhorn(mixes, scale, base):
    S=mixes.shape[0]
    pre=sigmoid(mixes[:,:HCM]*scale[0]+base[:HCM])+HCEPS
    post=2*sigmoid(mixes[:,HCM:2*HCM]*scale[1]+base[HCM:2*HCM])
    comb=mixes[:,2*HCM:].reshape(S,HCM,HCM)*scale[2]+base[2*HCM:].reshape(HCM,HCM)
    comb=softmax(comb,2)+HCEPS
    comb=comb/(comb.sum(1,keepdims=True)+HCEPS)
    for _ in range(HCIT-1):
        comb=comb/(comb.sum(2,keepdims=True)+HCEPS)
        comb=comb/(comb.sum(1,keepdims=True)+HCEPS)
    return pre,post,comb

def hc_pre(h, fn, scale, base):     # h[S,HCM,DIM]
    S=h.shape[0]; x=h.reshape(S,-1).astype(np.float32)
    rsq=np.reciprocal(np.sqrt(np.mean(x*x,-1,keepdims=True)+EPS))
    mixes=(x@fn.T)*rsq
    pre,post,comb=hc_sinkhorn(mixes,scale,base)
    y=(pre[:,:,None]*h).sum(1)
    return y.astype(np.float32),post,comb
def hc_post(a, resid, post, comb): # a[S,DIM], resid[S,HCM,DIM]
    return (post[:,:,None]*a[:,None,:]+np.einsum('sjk,skd->sjd',comb,resid)).astype(np.float32)

def expert_fp(x, w1,w3,w2, weight=None):
    gate=(x@w1.T).astype(np.float32); up=(x@w3.T).astype(np.float32)
    if SWLIM>0: up=np.clip(up,-SWLIM,SWLIM); gate=np.minimum(gate,SWLIM)
    h=silu(gate)*up
    if weight is not None: h=weight*h
    return (h@w2.T).astype(np.float32)

# Per-layer capture extras stashed here by moe_all (raw router logits, gate
# weights, routed-only aggregate). Module-level so external moe_all callers
# keep the (y, idx) 2-tuple contract untouched. run(capture=True) persists them.
CAP={}

def gate_route(x, W, ids):
    raw=(x.astype(np.float32)@W['gate'].T)           # [Ntot,256] RAW logits —
    scores=np.sqrt(np.log1p(np.exp(raw)))            # runtime delta is added HERE
    orig=scores                                      # (pre-sqrtsoftplus domain)
    if W['tid2eid'] is not None:
        idx=W['tid2eid'][ids]                        # [Ntot,NACT]
    else:
        sc=scores+W['gbias'][None]
        idx=np.argpartition(-sc,NACT-1,axis=1)[:,:NACT]
    w=np.take_along_axis(orig,idx,1)
    w=w/(w.sum(1,keepdims=True)); w=w*ROUTE_SCALE
    return idx, w.astype(np.float32), raw

def moe_all(Fin, Ids, W, L):
    Nt=Fin.shape[0]
    idx,wt,raw=gate_route(Fin,W,Ids)                 # [Nt,NACT]
    y=expert_fp(Fin,W['s1'],W['s3'],W['s2'])         # shared
    shared=y.copy()
    if not SKIP_ROUTED:
        flat_e=idx.reshape(-1); flat_w=wt.reshape(-1)
        tok=np.repeat(np.arange(Nt),NACT)
        for e in np.unique(flat_e):
            sel=flat_e==e
            t=tok[sel]; ww=flat_w[sel][:,None]
            w1=R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight")
            w3=R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight")
            w2=R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight")
            # 顺序量化 hook: 若设了 QUANT_HOOK, 用该层真实(传播后)输入 Fin[t] 校准+量化专家。
            # 默认 None → 纯 fp8 (现有用途不变)。
            qh=globals().get("QUANT_HOOK")
            if qh is not None:
                w1,w3,w2=qh(L,e,w1,w3,w2,Fin[t])
            contrib=expert_fp(Fin[t],w1,w3,w2,ww)
            np.add.at(y,t,contrib)
    CAP['route_logits']=raw            # teacher raw router logits [Nt,256]
    CAP['route_w']=wt                  # gate weights actually applied [Nt,NACT]
    CAP['routed']=(y-shared)           # routed-only aggregate (corr target space)
    return y.astype(np.float32), idx

def run(chunks, ids_list, outdir, capture=True):
    os.makedirs(outdir,exist_ok=True)
    emb=R.get("embed.weight")
    H=[np.repeat(emb[ids][:,None,:],HCM,1).astype(np.float32) for ids in ids_list]  # [S,HCM,DIM]
    t0=time.time()
    for L in range(NL):
        W=load_layer(L); tl=time.time()
        resblk=[]
        Fins=[]; bounds=[0]
        for c,h in enumerate(H):
            resid=h
            y,post,comb=hc_pre(h,W['hc_attn_fn'],W['hc_attn_scale'],W['hc_attn_base'])
            a=attention(rms(y,W['an']),W,L)
            h=hc_post(a,resid,post,comb)
            resid2=h
            y2,post2,comb2=hc_pre(h,W['hc_ffn_fn'],W['hc_ffn_scale'],W['hc_ffn_base'])
            fin=rms(y2,W['fn'])
            Fins.append(fin); bounds.append(bounds[-1]+fin.shape[0])
            resblk.append((resid2,post2,comb2))
        Fin=np.concatenate(Fins,0); Ids=np.concatenate(ids_list,0)
        Fout,Idx=moe_all(Fin,Ids,W,L)
        for c in range(len(H)):
            s,e=bounds[c],bounds[c+1]
            resid2,post2,comb2=resblk[c]
            H[c]=hc_post(Fout[s:e],resid2,post2,comb2)
        if capture:
            np.save(f"{outdir}/ffn_in_L{L}.npy",Fin.astype(np.float16))
            np.save(f"{outdir}/ffn_out_L{L}.npy",Fout.astype(np.float16))
            np.save(f"{outdir}/route_L{L}.npy",Idx.astype(np.int16))
            np.save(f"{outdir}/route_logits_L{L}.npy",CAP['route_logits'].astype(np.float16))
            np.save(f"{outdir}/route_w_L{L}.npy",CAP['route_w'].astype(np.float16))
            np.save(f"{outdir}/routed_L{L}.npy",CAP['routed'].astype(np.float16))
        nu=len(np.unique(Idx)) if not SKIP_ROUTED else 0
        print(f"L{L:2d} ratio{CR[L]:3d} uniq_exp{nu:3d} {time.time()-tl:5.1f}s tot{time.time()-t0:6.0f}s",flush=True)
        del W
    # head + next-token accuracy per chunk
    hcfn=R.get("hc_head_fn"); hcb=R.get("hc_head_base"); hcs=R.get("hc_head_scale")
    fnorm=R.get("norm.weight"); hw=R.get("head.weight")
    accs=[]; tki=[]; tkv=[]; TOPKL=64   # teacher final-logit top-K (agreement/KL vs student)
    for c,h in enumerate(H):
        S=h.shape[0]; x=h.reshape(S,-1)
        rsq=np.reciprocal(np.sqrt(np.mean(x*x,-1,keepdims=True)+EPS))
        mixes=(x@hcfn.T)*rsq
        pre=sigmoid(mixes*hcs+hcb)+HCEPS
        y=(pre[:,:,None]*h).sum(1)
        y=rms(y,fnorm)
        logits=y@hw.T
        if capture:
            part=np.argpartition(-logits,TOPKL-1,axis=1)[:,:TOPKL]
            vals=np.take_along_axis(logits,part,1)
            ordr=np.argsort(-vals,1)
            tki.append(np.take_along_axis(part,ordr,1).astype(np.int32))
            tkv.append(np.take_along_axis(vals,ordr,1).astype(np.float16))
        pred=np.argmax(logits[:-1],1); tgt=ids_list[c][1:]
        accs.append(float(np.mean(pred==tgt)))
        globals().setdefault("PRED",[]).append(pred)          # 供 fp8 vs 量化 top-1 一致率
        globals().setdefault("LOGITS",[]).append(logits[:-1]) # 供 KL
        _tk=globals().get('TK')
        if _tk is not None:
            print(f"  chunk{c} next-token ([ok] true -> predicted):",flush=True)
            for i in range(min(30,len(pred))):
                ok="OK" if pred[i]==tgt[i] else ".."
                print(f"    [{ok}] {_tk.decode([int(tgt[i])])!r:>14} -> {_tk.decode([int(pred[i])])!r}",flush=True)
    if capture and tki:
        np.save(f"{outdir}/final_topk_idx.npy",np.concatenate(tki,0))
        np.save(f"{outdir}/final_topk_val.npy",np.concatenate(tkv,0))
    print("next-token top1 acc per chunk:",["%.3f"%a for a in accs],"mean %.3f"%np.mean(accs),flush=True)
    return accs

if __name__=="__main__":
    # tiny smoke: one short Go snippet
    from tokenizers import Tokenizer
    tk=Tokenizer.from_file(os.path.join(R.HF,"tokenizer.json"))
    code=open(sys.argv[1]).read() if len(sys.argv)>1 else (
        'package main\n\nimport (\n\t"fmt"\n\t"sort"\n)\n\n'
        'func main() {\n\tnums := []int{5, 2, 8, 1, 9}\n\tsort.Ints(nums)\n'
        '\tfor i, n := range nums {\n\t\tfmt.Printf("%d: %d\\n", i, n)\n\t}\n}\n')
    ids=np.array(tk.encode(code).ids,dtype=np.int64)
    S=int(os.environ.get("S","128")); ids=ids[:S]
    print("smoke S=",len(ids),flush=True)
    capdir=os.environ.get("CAP_DIR","")   # set CAP_DIR to persist a full capture
    run([ids],[ids],capdir or "/tmp/cap_smoke",capture=bool(capdir))
