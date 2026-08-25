"""zlever/mkzchain.py — L00 z^L(k≤16)+GE → 原生 DQZ2 zchain(真值链 wave-2)。
干净口径: 解算只用 anchor 位置 128..1715, truth 段(0..127)解算未见 → 无 in-sample 污染。
z^L: zcache 残差上对偶 ridge(λ=3, dither 增广, colw classify)→ 折 colw → SVD 截 k。
GE: k截断 z^L 之后的残差对每专家门控缩放 ridge(type5=1+g)。
用法: mkzchain.py <anchor> <layers_dir> <out.bin> [K=16] [层=0]
"""
import os, sys, struct, numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','scripts'))
from probe_layer_behavior import anchor_layer
ap,ld,outp = sys.argv[1],sys.argv[2],sys.argv[3]
K=int(sys.argv[4]) if len(sys.argv)>4 else 16
L=int(sys.argv[5]) if len(sys.argv)>5 else 0
D=4096; NTOK=1716
X0,ridx,rw,NACT=anchor_layer(ap,L,NTOK)
zc=np.load(os.path.join(ld,f"zcache_L{L:02d}.npz"))
dH=zc["dH"]; prow=zc["prow"]; pe=zc["pe"]; pw=zc["pw"]; pYQ=zc["pYQ"]
X=X0.astype(np.float64)
SPLIT=sys.argv[7] if len(sys.argv)>7 else 'head128'
if SPLIT=='pos1287': tr=np.arange(0,1287); ev=np.arange(1287,NTOK)   # 位置切分: 评测段=序列尾部(上下文真实)
else: tr=np.arange(128,NTOK); ev=np.arange(0,128)
colw=np.sqrt(dH[tr].var(0)+1e-12)
rms=np.sqrt((X[tr]**2).mean(1,keepdims=True))
r2=np.random.RandomState(1)
Xa=np.vstack([X[tr],(X[tr]+r2.randn(len(tr),D)*0.04*rms)*np.sqrt(0.25)])
Ra=np.vstack([dH[tr]*colw,(dH[tr]*colw)*np.sqrt(0.25)])
G=Xa@Xa.T; G[np.diag_indices_from(G)]+=3.0*np.trace(G)/Xa.shape[1]+1e-10
al=np.linalg.solve(G,Ra)
Wz=(Xa.T@al)/colw[None,:]
A,S,Bt=np.linalg.svd(Wz,full_matrices=False)
pred=(X[ev]@(A[:,:K]*S[:K]))@Bt[:K]
e0=float((dH[ev].astype(np.float64)**2).sum())
r=1-float(((dH[ev]-pred.astype(np.float32)).astype(np.float64)**2).sum())/e0
print(f"z^L k={K}: 评测段({SPLIT}, 解算未见)层挽回 {r*100:.1f}%", flush=True)
R=dH-((X@(A[:,:K]*S[:K]))@Bt[:K]).astype(np.float32)
tok_pairs=[[] for _ in range(NTOK)]
for p in range(len(prow)): tok_pairs[prow[p]].append(p)
Gg=np.zeros((256,256)); bg=np.zeros(256)
for t in tr:
    ps=tok_pairs[t]
    vecs=[(int(pe[p]),(pw[p]*pYQ[p]).astype(np.float64)) for p in ps]
    rt=R[t].astype(np.float64)
    for a_,(ea,va) in enumerate(vecs):
        bg[ea]+=va@rt
        for b_ in range(a_,len(vecs)):
            eb,vb=vecs[b_]; d=va@vb; Gg[ea,eb]+=d
            if ea!=eb: Gg[eb,ea]+=d
Gg[np.diag_indices_from(Gg)]+=1e-3*max(np.trace(Gg)/256,1.0)
g=np.linalg.solve(Gg,bg)
ge=(1.0+g).astype(np.float16)
print(f"GE: 1+g ∈ [{float(ge.min()):.4f}, {float(ge.max()):.4f}] 均值 {float(ge.mean()):.4f}", flush=True)
V16=A[:,:K].astype(np.float16); z16=S[:K].astype(np.float16); U16=np.ascontiguousarray(Bt[:K].T).astype(np.float16)
pay6=struct.pack('<IfII',K,1.0e6,D,D)+z16.tobytes()+U16.tobytes()+V16.tobytes()
pay5=ge.tobytes()
out=bytearray(); out+=struct.pack('<II',0x325A5144,43)
for Li in range(43):
    ops=[(5,pay5),(6,pay6)] if Li==L else []
    out+=struct.pack('<II',Li,len(ops))
    for ty,pay in ops: out+=struct.pack('<II',ty,len(pay))+pay
open(outp,'wb').write(out)
print(f"→ {outp} ({len(out)/1e6:.2f} MB)  [type5 GE + type6 z^L k={K}]", flush=True)

# ── DQO2 侧车(量化器回放权威来源): 同载荷写 116B 头记录
if len(sys.argv)>6:
    dqo=sys.argv[6]
    def rec(nm,pay):
        h=bytearray(116)
        h[0:len(nm)]=nm.encode()
        struct.pack_into('<Q',h,88,len(pay))
        struct.pack_into('<i',h,112,1)          # vd=1 落地链
        return bytes(h)+pay
    body=rec('bf.GE',pay5)+rec('zl.RRR',pay6)
    open(dqo,'wb').write(b'DQO2'+struct.pack('<II',0,2)+body)
    print(f"→ {dqo} (DQO2 2 records)", flush=True)
