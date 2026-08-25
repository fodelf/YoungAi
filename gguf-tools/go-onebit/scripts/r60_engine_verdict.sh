#!/bin/bash
# r60_engine_verdict.sh — 引擎 32 针终审(部署态真口径, 全流程自包含可复制)。
# 判什么: 合并回传后的模型在引擎上的真实 KL(vs 教师锚 logits 前 32)。
# 含: 判决数据自重建(重启/清 tmp 后可直接跑) + proven 安全 flags(NO_RESIDENCY 必带,
#     2026-08-06 23:00 panic 教训: 漏它=watchdogd 饿死内核 panic) + KL 计算。
# 用法: r60_engine_verdict.sh [model=gguf/go-onebit/ds4-r30.gguf] [tag=chain]
#   tag=chain: 用模型内嵌 op 链; tag=bare: 挂空链(裸判)。
set -euo pipefail
cd /Users/fodelf/git/ds4-main
MDL="${1:-gguf/go-onebit/ds4-r30.gguf}"
TAG="${2:-chain}"
M1=192.168.1.2
ANCHOR_M1=/Users/fodelf/ds4-main/gguf/go-onebit/r30/anchor_r30_s1716.bin
IDS_M1=/Users/fodelf/ds4-main/gguf/go-onebit/g7/rr_calib_prog_v5mini.ids
LOG(){ echo "[verdict $(date +%H:%M:%S)] $*" >&2; }

# ── ①判决数据(缺则从 M1 锚重建: ref32=教师 logits 前32, ids32=判决语料前32, 空链) ──
if [ ! -s /tmp/ref32.bin ] || [ ! -s /tmp/ids32.txt ]; then
    LOG "判决数据缺, 从 M1 锚重建"
    ssh $M1 "python3 - <<'PYEOF'
import struct, numpy as np
A='$ANCHOR_M1'
with open(A,'rb') as f:
    hd=struct.unpack('<8I',f.read(32)); S,HCM,DIM,NL,V,NACT=hd[1],hd[2],hd[3],hd[4],hd[5],hd[6]
off=40+NL*S*DIM*4+NL*S*NACT*8+NL*S*HCM*DIM*4
with open(A,'rb') as f:
    f.seek(off); np.fromfile(f,dtype=np.float32,count=32*V).tofile('/tmp/ref32.bin')
open('/tmp/ids32.txt','w').write('\n'.join(open('$IDS_M1').read().split()[:32])+'\n')
print('ref32+ids32 ok')
PYEOF"
    scp -q $M1:/tmp/ref32.bin $M1:/tmp/ids32.txt /tmp/ 2>/dev/null \
        || { scp -q $M1:/tmp/ref32.bin /tmp/ && scp -q $M1:/tmp/ids32.txt /tmp/; }
fi
if [ ! -s /tmp/zl_empty.bin ]; then
    python3 -c "import struct;open('/tmp/zl_empty.bin','wb').write(struct.pack('<4sI',b'DQZ2',43)+b'\x00'*(43*8))"
fi

# ── ②引擎 32 位前向(proven 安全 flags; bare=挂空链 override 内嵌链) ──
D=/tmp/verdict_$TAG; rm -rf $D; mkdir -p $D
ZARG=""; [ "$TAG" = bare ] && ZARG="--zchain /tmp/zl_empty.bin"   # bash3.2: 空数组+set -u 会炸, 用字符串
LOG "引擎 32 针($TAG)起跑"
DS4_EVAL_IDS=/tmp/ids32.txt DS4_EVAL_NO_BOS=1 DS4_EVAL_CHUNK=8 \
DS4_EVAL_LOGITS=$D/ev.bin \
DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_NO_RESIDENCY=1 DS4_METAL_PREFILL_CHUNK=8 \
./ds4 -m "$MDL" $ZARG 2>&1 | /usr/bin/grep -E "zchain|EVAL_IDS" | tail -3

# ── ③KL 判决 ──
python3 - "$TAG" $D/ev.bin <<'PYEOF'
import numpy as np, sys
S,V=32,129280
t=np.fromfile("/tmp/ref32.bin",dtype=np.float32,count=S*V).reshape(S,V).astype(np.float64)
s=np.fromfile(sys.argv[2],dtype=np.float32,count=S*V).reshape(S,V).astype(np.float64)
def sm(x): x=x-x.max(1,keepdims=True); e=np.exp(x); return e/e.sum(1,keepdims=True)
P,Q=sm(t),sm(s)
kl=(P*(np.log(P+1e-12)-np.log(Q+1e-12))).sum(1).mean()
print(f"★ 引擎32针({sys.argv[1]}): KL={kl:.4f} top1={(t.argmax(1)==s.argmax(1)).mean()*100:.1f}%")
print("参照: 教师op链=1.1015 / 裸=1.139 / 部署态量化器裸=1.070")
PYEOF
