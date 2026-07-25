import os, sys
sys.path.insert(0,"/Users/fodelf/ds4-main/gguf-tools/go-onebit/quant")
sys.path.insert(0,"/Users/fodelf/ds4-main/gguf-tools/go-onebit/calib/pyfwd")
os.environ.setdefault("DS4_HF","/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base")
os.environ["DS4_GO2B_ACT_SCALE"]="1"
import numpy as np, ds4reader as R
from go2b_encode import encode_go2b
def Q1(W):
    m=np.abs(W).mean(1,keepdims=True); return np.sign(W)*m
def ocos(Wt,Wq,X):
    Ot=X@Wt.T; Oq=X@Wq.T
    return np.linalg.norm(Ot-Oq)/np.linalg.norm(Ot)
active={}
for line in open("/tmp/prog_active.txt"):
    if line.startswith("L"): L=int(line.split(":")[0][1:]); active[L]=[int(x) for x in line.split(":")[1].split()]
print("L | stacked输出relL2 | joint输出relL2 | 改善%")
tot_s=tot_j=0.0; nl=0
for L in range(20,43):
    xp=f"/tmp/algo_solve/x_L{L}.npy"
    if not os.path.isfile(xp): continue
    Xh=np.load(xp)[:256].astype(np.float32)
    exps=active.get(L,list(range(3)))[:3]
    ss=js=0.0;n=0
    for e in exps:
        try: W=R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight").astype(np.float32)
        except: continue
        b=Q1(W); stacked=b+Q1(W-b)
        blk,joint=encode_go2b(W,Xh=Xh)
        ss+=ocos(W,stacked,Xh); js+=ocos(W,joint,Xh); n+=1
    if n==0: continue
    ss/=n; js/=n; imp=100*(ss-js)/ss
    tot_s+=ss; tot_j+=js; nl+=1
    print(f"L{L} | {ss:.4f} | {js:.4f} | {imp:+.1f}%")
if nl: print(f"★均值 {nl}层: stacked={tot_s/nl:.4f} joint={tot_j/nl:.4f} 改善={100*(tot_s-tot_j)/tot_s:+.1f}%")
