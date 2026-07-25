import os, sys
sys.path.insert(0, "/Users/fodelf/ds4-main/gguf-tools/go-onebit/quant")
sys.path.insert(0, "/Users/fodelf/ds4-main/gguf-tools/go-onebit/calib/pyfwd")
os.environ.setdefault("DS4_HF", "/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base")
os.environ["DS4_GO2B_ACT_SCALE"]="1"
import numpy as np
import ds4reader as R
from go2b_encode import encode_go2b
def Q1(W):
    m=np.abs(W).mean(1,keepdims=True); return np.sign(W)*m
def ocos(Wt,Wq,X):   # 输出级cos: (Wt@x) vs (Wq@x)
    Ot=X@Wt.T; Oq=X@Wq.T
    return np.linalg.norm(Ot-Oq)/np.linalg.norm(Ot), (Ot.flatten()@Oq.flatten())/(np.linalg.norm(Ot)*np.linalg.norm(Oq)+1e-9)
active={}
for line in open("/tmp/prog_active.txt"):
    if line.startswith("L"): L=int(line.split(":")[0][1:]); active[L]=[int(x) for x in line.split(":")[1].split()]
print("★输出级(专家 W@x vs 重建@x, FP教师为真)★")
print("      | stacked(1+1bit) 输出relL2/cos | joint go2b(2bit+act) 输出relL2/cos | 判决")
for L in [20,21,22]:
    Xh=np.load(f"/tmp/algo_solve/x_L{L}.npy")[:256].astype(np.float32)
    experts=active.get(L,list(range(4)))[:4]
    ss=sc=js=jc=0.0;n=0
    for e in experts:
        W=R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight").astype(np.float32)
        b=Q1(W); stacked=b+Q1(W-b)
        blk,joint=encode_go2b(W,Xh=Xh)
        es,cs=ocos(W,stacked,Xh); ej,cj=ocos(W,joint,Xh)
        ss+=es;sc+=cs;js+=ej;jc+=cj;n+=1
    print(f"L{L} Xh={Xh.shape}| relL2={ss/n:.4f} cos={sc/n:.4f} | relL2={js/n:.4f} cos={jc/n:.4f} | {'★joint赢' if js/n<ss/n else 'stacked赢'}")
