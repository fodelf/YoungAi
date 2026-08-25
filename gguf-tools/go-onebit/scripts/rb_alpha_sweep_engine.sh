#!/bin/bash
# RB 烘焙 α 扫描 — 引擎部署态口径 (每点 ~9 分钟)。
# 与 rb_alpha_sweep.sh(量化器只读回放口径)互补; 2026-08-06 定案: 量化器口径
# teacher-forced 虚好(0.68 vs 引擎 1.14), α 终审必须走引擎口径。
# 在 ds4-r30 的 COW 克隆上原位换 α, 32 位置 KL vs 教师锚 logits。
# 前提: /tmp/ref_new.bin /tmp/ids32.txt /tmp/zl_empty.bin /tmp/route_bias_r30.bin
# 用法: rb_alpha_sweep_engine.sh <clone.gguf> <cur_alpha> <alpha1> [alpha2 ...]
set -euo pipefail
cd /Users/fodelf/git/ds4-main
MDL=$1; CUR=$2; shift 2
RB=/tmp/route_bias_r30.bin
SC=gguf-tools/go-onebit/scripts
for A in "$@"; do
    echo "== α: $CUR → $A (原位 rebake) =="
    python3 $SC/route_bias_rebake.py "$MDL" $RB "$CUR" $RB "$A" | tail -1
    CUR=$A
    D=/tmp/rid32_a$A; rm -rf $D; mkdir -p $D
    DS4_EVAL_IDS=/tmp/ids32.txt DS4_EVAL_NO_BOS=1 DS4_EVAL_CHUNK=8 \
    DS4_EVAL_LOGITS=$D/ev.bin DS4_METAL_EXPERT_OFFLOAD=1 \
    ./ds4 -m "$MDL" --zchain /tmp/zl_empty.bin 2>&1 | tail -2
    python3 - "$A" $D/ev.bin <<'PYEOF'
import numpy as np, sys
S,V=32,129280
t=np.fromfile("/tmp/ref_new.bin",dtype=np.float32,count=S*V).reshape(S,V).astype(np.float64)
s=np.fromfile(sys.argv[2],dtype=np.float32,count=S*V).reshape(S,V).astype(np.float64)
def sm(x): x=x-x.max(1,keepdims=True); e=np.exp(x); return e/e.sum(1,keepdims=True)
P,Q=sm(t),sm(s)
kl=(P*(np.log(P+1e-12)-np.log(Q+1e-12))).sum(1).mean()
print(f"★ alpha={sys.argv[1]}: KL={kl:.4f} top1={(t.argmax(1)==s.argmax(1)).mean()*100:.1f}%")
PYEOF
done
echo "== 扫描完; 模型当前 α=$CUR (参照: α0=1.3068 α2.5=1.1396, 量化器TF底=0.682) =="
